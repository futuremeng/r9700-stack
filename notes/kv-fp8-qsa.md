# fp8 (e4m3) main KV cache for QSA: what was already there, what we wrote, and the 2.06x -> ~3.8x pool it buys (2026-11-03)

Brian, after the CPU/KV audit: "第一条这个发现值得试一下" — the first row of the KV list was **fp8 / 4-bit KV**, which
`PROGRESS.md:896` had parked as "fp8 KV is refused by stock QSA ('requires a BF16 main KV cache')". Reading the code
says that gate is three `raise`s and one missing read path, not a design limit: **the write side already does fp8.**
This note is the S0 investigation plus the code that opens it. Nothing here is measured on GPUs yet — the box was
serving the whole session, so S1/S2 are offline and S3 (restart + gates) waits for a window.

## What is already fp8-capable, and where the walls are

| | |
|---|---|
| writer | **already fp8.** The fused QSA prepare Triton kernel stores the main K/V through `_to_dst_dtype()`, which converts to `tl.float8e4nv` when the destination cache is fp8 (`vllm models/qwen4_exp/amd/ops/qsa_prepare.py:123-140, 268`), with descales `main_k_scale=attn._k_scale_float`, `main_v_scale=attn._v_scale_float` (`amd/indexer_qsa.py:323-326`) |
| reader (ROCm) | **missing.** Three guards: `amd/qsa.py:190` (layer, `cache_dtype not in ("auto","bfloat16")`), `:273` (layer, `kv_cache_torch_dtype != bfloat16`), `:113` (impl). Ours: `attn/qsa.py:138` and a bf16-only `kernels/r9k_qsa.hip` |
| reader (NVIDIA) | upstream already has it — `models/qwen4_exp/nvidia/qsa.py:306` reads fp8 with the same per-layer scales. We are filling the AMD half of an upstream feature |
| precedent in our stack | GLM-5.3-Flash runs an fp8 KV pool on these cards via `r9k_sparse_mla.hip` + `models/glm5next.py:140-168` ("stock's fp8-pool path is CDNA assembly and its Triton path is bf16-only"), with a fixed host descale and no per-call sync. Same shape of problem |
| scales | `set_default_quant_scales` registers them at 1.0. The checkpoint ships `model-kvscales.safetensors` + `kv_calibration_report_mtp.json` for the **MTP layer only** (K `observed_max` 90.5 → scale 0.4978, V 11.06 → 0.1217) and `kv_cache_scheme: null`: nothing near e4m3's 448 ceiling, so at 1.0 nothing clips and a scale is a precision choice, not correctness |

## The capacity arithmetic (checked against the running instance's own log)

`log setup-120827.log`: attention block size **448** tokens ("attention page size >= mamba page size"), mamba page
padded **1.98%**, `Available KV cache memory: 7.22 GiB` a card, `GPU KV cache size: 540,202 tokens`,
`Maximum concurrency for 262,144 tokens per request: 2.06x`.

* 48 layers, 12 of them `full_attention` (`full_attention_interval = 4`); 2 KV heads × 256 head_dim × bf16 =
  **1,024 B/token/layer** → **12,288 B/token/card**, and page math checks: 448 × 1,024 = 458,752 B ≥ the 449,828 B
  one-page GDN state.
* 7.751e9 B / 540,202 = **14,348 B/token/card**: 85.6% main KV, the rest the indexer's side caches plus the
  per-request GDN/CircularBuffer pages (36 layers × 458,752 B = 16.5 MB a request).
* fp8 on the main cache only: 12,288 → 6,144 B/token ⇒ **pool ≈ 0.96–1.02M tokens (+78–89%)**, and
  **262k-context concurrency 2.06x → ~3.7–3.9x**.
* Side effect: an fp8 attention page is 512 B/token, so vLLM must pick block **896** to keep page ≥ mamba page (the
  padding stays ~2%); token granularity doubles, which wastes ≤896 tokens per request (~1.4% of the bigger pool).
* Speed is not the prize: QSA decode reads `indexer_budget = 2048` tokens × 1,024 B × 12 layers = 25 MB a token
  against ~512 GB/s of GDDR7 — halving it is ~1% of the card. Long-context prefill may take a little. **Capacity is
  the whole reason to do this.**

## What was written (offline, running instance untouched)

* `kernels/r9k_qsa.hip`: the attention kernel is now `template <int GQA, typename KV>`. For fp8 only the two stage
  loads change — K reads 16 B of **16** fp8 (`KCHUNK` 512 → 256) and V^T uses `global_load_tr_b64_v2i32`; both go
  through `fp8x8_to_bf16` (`v_cvt_pk_f32_fp8`, the same helper `r9k_attn.hip` already uses) into an **unchanged bf16
  LDS layout**, so LDS size, occupancy and the WMMA pipeline are the bf16 kernel's. K's descale rides in on
  `scale_log2`, V's on the output normalizer (also correct through the split+merge path: every split of a row scales
  its partial by the same constant). New entry `r9k_qsa_attn_f8(..., double kdesc, double vdesc)`.
  Compiles for gfx1201: 14 kernel instantiations (7 GQA × 2 dtypes), both symbols exported, no `hidden_hostcall_buffer`.
* `r9700_vllm/attn/qsa.py`: dtype-aware dispatch (`FP8` cache → `r9k_qsa_attn_f8`), and `kv_descale(layer)` reads
  `layer._k_scale_float / _v_scale_float` **once per layer** — the same two floats the writer quantized with, so the
  descale cannot drift, and no per-call host sync (the `glm5next.py` rule).
* `r9700_vllm/compat/qsa_kvfp8.py` (+ `gate.py` entry `qsa_kvfp8`, `__init__.py` registration): with
  `R9K_QSA_KVF8=1` the QSA layer is constructed with `cache_config.cache_dtype` temporarily reported as
  `bfloat16` — which also keeps the **side** caches bf16, as intended — and the layer's own `kv_cache_dtype` /
  `kv_cache_torch_dtype` are then set to fp8, which is what `get_kv_cache_spec()` reads. Self-disabling when
  `R9K_QSA_KVF8` is unset (the default) or the target moved.
* `tests/test_qsa_fp8.py`: per case it prints three numbers — our bf16 kernel vs a torch reference, our **fp8 kernel
  vs a reference over the same fp8-quantized K/V** (that is the correctness assertion: it must be as tight as bf16),
  and the quantization error itself (the accuracy question, isolated). Cases cover GQA 6/12, rows 1–2048,
  nsplit 1 and split+merge, page 448/896, descale 1.0 and 0.5.

## The window protocol (S3 — needs the box)

1. `cd kernels && OUT=libr9k.so HIPCFLAGS=-I$HOME/opt/rocm-dev/opt/rocm/core-10.1/include ./build.sh` (build to a
   temp `OUT` first; the running instance holds the old file mapped).
2. `/home/jiulu/venv-r9700/bin/python3 tests/test_qsa_fp8.py` — the fp8 column must match the bf16 column's order of
   magnitude before any serving run is trusted.
3. `R9K_QSA_KVF8=1 KVDTYPE=fp8_e4m3` in the launcher (`serve/native-flashnext.sh:78` already has `KVDTYPE`, and
   `harness.py` already records `kvdtype`, so the A/B bookkeeping exists). Read the launch log for: block size now
   **896**, `GPU KV cache size` ≈ 1.0M tokens, `Maximum concurrency for 262,144` ≈ 3.8x.
4. Gates, in repo order: `bench/quality.py` (GSM8K 300 + HumanEval + needle at 250k), `bench/gsm8k_prod.py`,
   `bench/probe32.sh`, then a paired `bench/ab.sh` run. Watch **tokens per step** (MTP acceptance, 3.1 today) as
   well as decode: fp8 KV can move acceptance without moving kernel time.
5. Rollback is one env var (`R9K_QSA_KVF8=0`, no `KVDTYPE`) — nothing in the bf16 path changed except that it now
   goes through the same templated kernel with `kdesc = vdesc = 1`.

## Measured on this box, 2026-11-03 (idle window 13:23-13:52; two restarts, box left on bf16)

**Kernel test (`tests/test_qsa_fp8.py`, one card, running instance untouched).** rel = max abs error / max |ref|,
kernel vs its own reference:

| rows | GQA | page | nsplit | descale | bf16 rel | fp8 rel | fp8/bf16 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 448 | auto | 1.0, 1.0 | 4.8e-3 | 2.1e-3 | 0.875 |
| 1 | 6 | 896 | auto | 1.0, 1.0 | 2.4e-3 | 2.2e-3 | 0.966 |
| 3 | 6 | 896 | auto | 0.5, 0.25 | 3.6e-3 | 1.7e-3 | 0.943 |
| 8 | 6 | 896 | auto | 1.0, 1.0 | 3.2e-3 | 3.1e-3 | 0.934 |
| 16 | 6 | 896 | 1 | 1.0, 1.0 | 4.0e-3 | 3.9e-3 | 0.906 |
| 64 | 6 | 448 | 1 | 0.5, 1.0 | 3.6e-3 | 3.5e-3 | 0.923 |
| 256 | 6 | 896 | 1 | 1.0, 1.0 | 3.6e-3 | 3.5e-3 | 0.893 |
| 2048 | 6 | 896 | 1 | 1.0, 1.0 | 3.3e-3 | 3.5e-3 | 0.986 |

The fp8 read path is as tight as the bf16 one (and the non-unity descale rows confirm K and V both descale), and the
kernel itself comes out **7-11% faster** on these shapes (half the bytes in the global loads; LDS/WMMA untouched).

**Serving A/B, same box, same launcher, two fresh instances.**

| | bf16 (production) | fp8_e4m3 main KV |
|---|---:|---:|
| attention block size / mamba pad | 448 / 1.98% | **880 / 0.16%** |
| `GPU KV cache size` | 540,202 tokens | **1,021,931 tokens (+89.2%)** |
| `Maximum concurrency for 262,144` | **2.06x** | **3.90x** |
| `Available KV cache memory` | 7.22 GiB | 7.31 GiB |
| needle 30k/67k/150k/250k (cold+warm) | 8/8 recorded | **8/8**, 250,078-token prompts hit at depth 105,747 |
| GSM8K no-think 300 @8 (`quality.py`) | **96.0%** (288/300) | 95.3% (286/300) |
| `probe32` @32x1200 (2 samples) | 1,807.8 / 1,811.0 | 1,785.0 / 1,801.5 (−0.7%) |
| `probe32` @32x300 (3 samples) | (1 s. off) / 1,593.8 / 1,595.6 | 1,527.8 / 1,575.8 / 1,534.8 (−4%) |
| `probe32` @16x300 medians | 1,131.3 | 1,122.0 (−0.8%) |
| accepted/draft (MTP-2, from `/metrics`) | 1.604 | 1.549 (−3.4%) |
| power at busy==100 | median 300 W/card | median 300 W/card |
| init engine (warm compile cache) | 21.0 s (0.5 s compile) | 95.2 s (**32.1 s compile** — the fp8 cache dtype invalidates the AOT graph; one-time) |

Reading: **the pool doubles exactly as predicted** (2.06x -> 3.90x at 262k), retrieval to 250k is unchanged, and
timing is flat-to-slightly-down: −0.7% at the 1,200-token waves (inside this probe's own spread), −4% at the 300-token
ones (partly the −3.4% acceptance), and the prefill side showed nothing. The GSM8K gap is two questions out of 300,
which the repo's own bar calls "no detectable difference at this n" -- the 800-question pair below settles it.

So this is a **capacity change**, as designed: it is worth turning on for long-context concurrency (this box hit
82% KV usage under three streams today), and the honest cost is one 32 s recompile at launch, a few percent of
acceptance at short waves, and a quality question that needs the 800-question gate before it becomes a default.

## A build.sh landmine found on the way (fixed)

`for sym in ...; do nm -D "$OUT" | grep -q " T $sym$"; done` under `set -o pipefail` is a flake: `grep -q` closes the
pipe, `nm` dies of SIGPIPE (141), the pipeline reads as "missing", and build.sh fails a clean build -- it killed two
builds in a row on *different* symbols (`r9k_gemm_fp8_block_qa`, then `r9k_fp8_prefill_block`) and the identical loop
passed against the same file seconds later. Fixed: one `nm -D` pass into a temp file, grep that.

## Artifacts and state left behind

* `kernels/libr9k.so` and `r9700_vllm/kernels/libr9k.so` now contain `r9k_qsa_attn_f8`; the bf16 production instance
  relaunched on that .so and measured the same pool and throughput as before, so the templated kernel costs the bf16
  path nothing. Previous builds kept as `libr9k.so.pre-f8` in both directories.
* Both instances replaced with `mv`/rename (new inode): the running server's mapping was never written through.
* Box left running its **bf16 production config** (block 448, 540,202 tokens, 2.06x, WEBUI=same, MTP=2, NSEQ=32,
  MAXLEN=262144). To run fp8: `R9K_QSA_KVF8=1 KVDTYPE=fp8_e4m3 bash serve/native-flashnext.sh`.
* Logs: fp8 instance `~/logs/kvf8-132642.log`, bf16 baseline `~/logs/bf16base-133829.log`, probes `~/logs/probe32.txt`.

## Paired arms, same afternoon (13:51-14:00 bf16, 14:06-14:15 fp8; same script, same order, `~/logs/gatepair-*.log`)

One fresh instance per arm, every gate at the repo's own parameters:

| gate | bf16 | fp8_e4m3 main KV |
|---|---:|---:|
| `quality.py 800 8` GSM8K no-think | 95.6% (765/800) | **96.0% (768/800)** |
| needle 3.8k/15.2k/30.4k | 3/3 | 3/3 |
| `needle_long` 30k/67k/150k/250k cold+warm | 8/8 | 8/8 |
| prefill ladder 483/1,964/7,777/31,147/77,811 tok | 2,586 / 7,938 / 9,705 / 8,928 / 8,252 | 2,614 / 7,886 / 9,702 / 8,965 / 8,282 |
| `probe32` @32x300 (3 samples) | 1,611.8 / 1,640.4 / 1,615.0 | 1,511.0 / 1,550.8 / 1,574.0 |
| `probe32` @32x1200 (2 samples) | 1,817.5 / 1,831.3 | 1,788.0 / 1,772.6 |
| `probe32` @16x300 (3 samples) | 1,140.9 / 1,133.6 / 1,149.8 | 1,070.5 / 1,090.6 / 1,100.6 |
| power at busy==100 | median 299 W | median 300 W |
| `GPU KV cache size` | 540,202 tok / 2.06x | 1,009,899 tok / 3.85x |

HumanEval (`bench/humaneval.py 164`, greedy, thinking on, 164 problems both arms): **bf16 159/164 = 96.95%**
(failed 116,145,32,38,50) against **fp8 161/164 = 98.17%** (failed 145,32,38). McNemar: b=2 (bf16-only failures),
c=0 -> exact p=0.50. Nothing to see, and if anything the direction favours fp8.

**Prefill is flat to +-0.5% at every size up to 77.8k tokens**, which answers the bandwidth question: half the bytes
in the main KV does not buy prefill on this box, because prefill at NBT=8192 is compute-bound, not KV-bound.

**Decode is down 1.5-4%, and it is acceptance, not the kernel.** Pooled over two instances per arm: @32x1200
1,814 (bf16) against 1,787 (fp8) = -1.5%; @32x300 1,612 against 1,543 = -4.3%; @16x300 1,141 against 1,091 = -4.4%.
Meanwhile the kernel alone measured 7-11% *faster* (table above). The `/metrics` counters say why: accepted tokens per
draft is 1.604 and 1.620 on the two bf16 instances, 1.549 on the fp8 one, i.e. tokens per decode step 2.61 -> 2.55,
**-2.3%**, which is the whole gap at 1,200-token waves and most of it at 300-token ones. Quantized V (and the
logits built from quantized K) shifts which continuations the target accepts. Caveat: those counters are cumulative
over the mixed workload each instance served before the read, not over the probe alone.

`bench/ab.sh` -- the repo's formal randomized-interleaved A/B -- **cannot run on this box**: it drives
`serve/serve.sh` and checks `docker ps` for the `vllmstock` container, and this box has no docker and no root
(the reason `serve/native-flashnext.sh` exists). The paired arms above are the closest available substitute: two
fresh instances per config, 6 `probe32` samples per config, but not order-randomized, and both arms of a pair were
adjacent in time.

## Recommendation this data supports (not yet applied)

Turn it on for this box's traffic: the ledger's p99 prompt is 245,792 tokens, KV usage hit 82% with three streams
under it, and fp8 takes 2.06x to 3.85x at 262k with no measurable quality cost (800-question GSM8K +3, HumanEval +2,
needle perfect to 250k). The price is a one-time 32 s recompile at first launch, 1.5-4% of aggregate decode, and a
block size of 880 that makes each freed block 2x coarser. Flip: `R9K_QSA_KVF8=1 KVDTYPE=fp8_e4m3`. Box left on bf16.

## Still open (honest list)

* A CPU smoke of the test harness (`tests/test_qsa_fp8.py`'s `make_case`/`reference`/`quant` with `DEV=cpu`) says the
  plumbing is right and gives the envelope on **synthetic** keys: e4m3 costs ~5% per K/V element (scale 1.0, K max
  ~150) and 0.10-0.21 of max attention output, because a random-K softmax is peaked and a 5% K error flips which key
  wins. Real attention is not random, but that is the reason the quality gate is part of the protocol and not a
  formality.

* **Not measured**: an order-randomized pair (impossible here, see `bench/ab.sh` above), the acceptance effect with
  *calibrated* per-layer scales -- today the 12 main layers run at scale 1.0, and a calibration pass could take the
  -2.3% tokens/step with it -- and fp8 under real traffic at 80%+ KV usage, where the +87% pool is the whole point
  and no probe in this file reproduces it.
* **Scales for the 12 main layers** are 1.0 defaults. If the quality gate is unhappy, the fix is a calibration pass in
  the same style as the shipped MTP report (`pad 1.1`, `divisor`, `sequences 949`), not a different format.
* e4m3 has 3 mantissa bits: ~3-6% relative error per element is the floor. 4-bit KV (radiance's `k6_v4`/`fp8_v4`) is
  a strictly larger ask — QSA's indexer cache is already separately quantized (`attn/indexer_logits.py:72`), which is
  why the top-k *selection* is untouched by any of this.
* If the pool gain shows up but the quality gate is a wash, the next lever is `--kv-cache-dtype` for the **side**
  caches too (they are ~14% of the per-token bytes here), which would be a second note.
