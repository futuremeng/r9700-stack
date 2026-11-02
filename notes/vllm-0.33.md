# vLLM 0.33.0 on this stack: what moved since the pin, and what following it costs (2026-11-02)

"vllm 有更新吗？我们要不要跟一下？" — checked 2026-11-02 against three things at once: the release line
(v0.33.0), the ROCm 10 nightly line our image pins, and `main`. Short answer: **yes, follow v0.33.0 — it is
107 commits past our pin, it needs no plugin change, and the wheel is a drop-in for the native venv. Do not
follow `main` yet**: one upstream rename breaks two of our imports there.

## What 0.33.0 is

- Tag `v0.33.0` → commit **`4ba787ec9`** (`[CI][Bugfix] Pin the ColUltraVBERT test checkpoint…`, commit dated
  2026-10-31 07:15 UTC, tagged 2026-11-01 09:21 UTC; rc0-rc3 tags precede it). As of 2026-11-02 there is **no
  GitHub release page and no PyPI 0.33.0** (PyPI still stops at 0.32.0) — but the wheels are built and served
  (last-modified 2026-11-01 09:45 UTC).
- **Our image pin is an ancestor of the release.** `e4a5b5537` (the 2026-10-25 `nightly-rocm100` pin) → `4ba787ec9`
  is **107 commits, status `ahead`**; `4ba787ec9` → `main` (`5689c2790`, 2026-11-02) is another **487 commits,
  `diverged`**. So the release line is the cheap step and `main` is the expensive one, not a superset ordering.
- The 107 commits are mostly KV-offload, Rust frontend, CI and new-model work. Only five touch ground we stand
  on (below).

## Wheels: the native box can follow without docker

`wheels.vllm.ai/rocm/<pin>/` carries **two flavors per pin**: `rocm72/` (the ROCm 7.2-era build our venv uses)
and `rocm100/` (the pip-SDK ROCm 10 build). For `4ba787ec9`:

- `vllm-0.33.0+rocm72-cp312…manylinux_2_39_x86_64.whl`, 325.7 MB, HTTP 200 verified.
- **torch and triton are the same versions we already run**: `torch-2.13.0+git733fca1`,
  `triton-3.8.0+git669b31ac`. So a pin bump is a vLLM-only change on this box — no torch, no SDK, no driver.
- Two dependency deltas in the wheel: it **needs `orjson`** (new hard dep; installed) and it pins
  **`amd-aiter==0.1.26`** while our venv has `0.1.24.post1`. AITER is off in our launch (`AITER=0`), so the test
  venv keeps `0.1.24.post1` and measures the same thing we serve; `0.1.26` is in the same index and goes in when
  we adopt.

**Two stale things this uncovered in our own docs** (fixed in the same commit as this note):

- The flavor is renamed upstream — `+rocm723` → **`+rocm72`** — and **per-commit indexes expire**: the URL
  `README.md` and `notes/native-4xr9700.md` documented (`…/rocm/5df63a72…/rocm723/`) now returns 404, so a fresh
  box could not be built from the recipe as written. `setup.py`'s `--check` hint was worse: it printed
  `download.pytorch.org/whl/rocm6.4`.
- Per-commit indexes exist for `main` commits too (`24324ef0f`, `45fd60d4d` → 200) but not for every one
  (`5689c2790`, `d492ee786` → 404): builds appear with CI and disappear. Track the release line, not a moving sha.

## The test venv, and how to re-run it

`~/venv-r9700-033` is a `cp -a` of `~/venv-r9700` (11 GB, shebangs in `bin/` re-pointed) with one swap — the
torch/triton/aiter/openmpi and both editable installs (`-e .`, `-e serve/plugin`) are the ones the box already
runs, and `libr9k.so` is untouched (host hipcc, ROCm 10.1):

```
~/venv-r9700-033/bin/python -m pip install --no-deps vllm==0.33.0+rocm72 \
    --extra-index-url https://wheels.vllm.ai/rocm/4ba787ec9959e566d3b587a796dff07673324da5/rocm72/
~/venv-r9700-033/bin/python -m pip install orjson
```

For serving, keep the caches off the production ones: `VENV=~/venv-r9700-033 VLLM_CACHE_ROOT=~/vllm-native-cache-033
TRITON_CACHE_DIR=~/vllm-native-triton-033 bash serve/native-flashnext.sh` (1.5 GB + 240 MB of cache get rebuilt,
and the 110.6 GiB checkpoint reads again — that is what a pin bump actually costs in wall time).

## Import-surface audit (static, before running anything)

Every `from vllm…` in `r9700_vllm/` — **63 modules, 91 symbols** — checked against the `v0.33.0` tree and against
`main` (blobless clone, `git show` per module):

| ref | result |
|---|---|
| `v0.33.0` | **all present.** Two apparent misses (`vllm.distributed.tensor_model_parallel_all_reduce`, `vllm.models.glm5next`) are lazy re-exports, not removals |
| `main` | **two modules gone**: `vllm.models.qwen4_exp.amd.indexer_qsa` and `vllm.models.qwen4_exp.common.qsa_cache`, renamed by [`#64110`](https://github.com/vllm-project/vllm/pull/64110) `d492ee786` (2026-10-30, "Remove FlashAttention dependency and standardize QSA naming") to `amd/qsa_indexer.py` and `common/qsa.py`. Our imports of `apply_qsa_rope` (`attn/indexer_glue.py`, `models/qwen4_exp.py`) and `QSAForwardMetadata` break there. `#64110` is **not** in 0.33.0 |

`vllm/plugins/__init__.py` and `vllm/v1/metrics/stats.py` have **no diff** between our pin and 0.33.0: the
Monitor/About plugin (endpoint + stat-logger entry points, `FinishedRequestStats`) is untouched by this bump.

## What 0.33.0 changes in our surface (5 items)

1. **`a55937f25` `#60084` — 32-bit buffer addressing restored for `TRITON_ATTN` KV views.** New
   `set_triton_ptr_range()` in `v1/attention/backends/utils.py`, called from `TritonAttentionImpl` on ROCm for the
   fp8 KV views. Triton's AMD backend only uses 32-bit addressing for pointers under 2 GiB, and a view into a big
   storage loses it; the fix sets `ptr_range` to the distance from the view to the **end of the region**. We run
   fp8 KV through our `R9kTriton3DImpl` subclass at 262k — this is the one item in 0.33.0 that could move our
   prefill/decode numbers, and the one that could bite: it asserts kernels never read past the end of the region.
   Our subclass overrides `forward`, so the measurement has to be ours, not stock's.
2. **`00b6c4c56` `#59774` — Octave KV (`ROCM_OCTAVE`), a native 3-bit KV cache backend.** +1,180 lines of
   `csrc/attention`, a new backend in the ROCm priority list, and it is what grew `models/qwen4_exp/amd/qsa.py`
   by 212 lines in this range. **Gated to RDNA3 and CDNA3** (`get_rdna_version() == 3 or get_cdna_version() == 3`,
   "decoder attention only") — so it is *not* available on gfx1201, and it will not show up as a KV-density option
   for the 262k context until RDNA4 is validated. It is now also in the backend priority list, so a mis-gated
   selection would be our problem, not stock's.
3. **`rocm.py` +45**: `get_rdna_version()` (4 / 3 / 0) is new — worth using in `r9700_vllm` instead of our own arch
   probes; `_maybe_default_dsv4_mxfp4_indexer` is CDNA4-only and DeepSeek-only, no effect on Qwen4Exp/GLM5next.
4. **`platforms/__init__.py`**: a platform plugin that raises during detection is now logged at **warning** level
   when it is out-of-tree (was debug). Our `r9700_vllm.platform:detect` must not throw — on a bad card that changes
   a silent line into the first thing a reader sees. It did not fire in the 0.33.0 gate below (detection succeeded).
5. **`0f6b26191` `#61879` added `FusedMoEExperts.post_gemm2_hook`** — "called with the unweighted GEMM2 rows and
   the output to weight and reduce into, **replacing the experts' own top-k reduction**". That is upstream building
   the extension point our `moe/fold.py` monkeypatches around (output alias + shared-expert fold, both OFF by
   default since 2026-10-07 at ~0.1 ms of a 16.7 ms step). Retiring those two patches onto the hook is now
   possible on a supported API; not done here, and it should be a separate change with its own A/B.

## Gates on this box (jiulu-9700, 2026-11-02, both venvs, paired wherever the box let us)

The small gates were run against the standing 0.32 launch (one card visible, ~1.5 GiB free); the serving gates ran
on a 0.33 launch with the cards to itself. Three boots went between them (see the hardware note), so where a number
has a pair, the pair is named.

| gate | 0.32.0 (production venv) | 0.33.0 (`~/venv-r9700-033`) |
|---|---|---|
| `tests/test_integration.py` | ALL OK, 34 checks | **ALL OK, 34 checks** — no plugin change needed |
| `tests/test_ple_int6.py` | ALL OK | **ALL OK** (head_dim 160/256, device and UVA rows, rel 1.67e-03) |
| `tests/test_cache_shapes.py` | ALL OK | **ALL OK** — eager 900 shapes + graph capture + 12 replays equal |
| `tests/test_gemm_fp8.py` | **ALL OK**: lm_head 124160x2560 M=1 fp8 522 µs (608 GB/s) vs bf16 torch 1,021 µs; fp8-block 8192x2560 M=4 15-17 µs | **ALL OK**: 523 µs / 1,021 µs and 17 µs. The first 0.33 sample read 30 µs (708 GB/s) on that line; re-runs interleaved 0.32/0.33/0.32/0.33 read 15, 17, 17, 17 µs — noise, recorded because it looked like a regression for about a minute |
| `tests/test_moe_mxfp4.py` | **ALL OK**: gate_up 202.3 µs (598 GB/s) at 139 touched experts, down 103.7 µs (584 GB/s) | **ALL OK**: 202.2 µs (598 GB/s), 103.9 µs (582 GB/s) |

The first OOM (`test_gemm_fp8` at 08:22) was capacity, not the version: the standing launch held 32.6 of 34.2 GB on
every card and that test's `x.float() @ w.float().T` reference asks for 1.19 GiB on its own. With the cards free both
venvs run it through.

### Serving, same recipe both sides: `MAXLEN=262144 NSEQ=32 NBT=8192 PREFIX_CACHE=0 WEBUI=same`, uncapped

The 0.32 column is the record [context-262k.md](context-262k.md) measured **the same morning** (05:27-05:45 UTC) on
the same box and the same recipe; the 0.33 column is a fresh launch at 09:17 (log
`~/logs/ctx262k-033-091658.log.out`, KV line) and a second at 09:46 (`ctx262k-033b-094629.log.out`).

| probe | 0.32.0 | 0.33.0 |
|---|---|---|
| KV pool, tokens | 540,202 / 544,919 / 546,391 | **535,781** and **545,949** (two launches) |
| `probe32.sh` 32 in flight, 300-token waves | 1,568.3 / 1,573.7 / 1,587.5 → 1,576 | 1,353.7 (first request the launch ever served) / **1,577.8 / 1,562.5** |
| `probe32.sh` 32 in flight, 1,200-token waves | 1,799.5 / 1,832.7 → 1,816 | **1,812.8 / 1,817.0** |
| `probe32.sh` 16 in flight, 300-token waves | 1,133.1 / 1,135.8 / 1,142.7 → 1,137 | **1,107.3 / 1,145.0 / 1,122.0** |
| power with `gpu_busy_percent == 100` | median 299 W, max 300.0 W | **median 300 W, max 315 W** (n=139) |
| ladder 60,061 tok (cold TTFT / prefill) | 6.93 s / 8,672 tok/s | 8.42 s / 7,130 first-ever request, then **6.96 s / 8,628** |
| ladder 120,066 | 14.89 s / 8,062 | **15.03 s / 7,988** (repeat 15.10 / 7,951) |
| ladder 180,070 | 24.22 s / 7,436 | **24.36 s / 7,391** (repeat 24.39 / 7,383) |
| needle at each tier | found, 7-token answer | **found at all three** |
| `bench/quality.py 100 8` | GSM8K 99/100, 33,909 prompt tokens in 40 s; needle 3/3 | **GSM8K 98/100**, 34,248 tokens in 41 s; **needle 3/3** (3,828 / 15,196 / 30,354) |
| `bench/itl.py 1 2 4` | not recorded for this recipe (the 14.3 ms row is the 32k / MTP-3 launch) | **13.0 / 14.8 / 18.5 ms** median, p90 13.3 / 15.1 / 19.0 |
| single-stream decode | 186.7 tok/s at the 32k/MTP-3 recipe | **187.8 / 189.5 tok/s** at 262k/MTP-2 — not a pair, both listed so nobody pairs them later |
| `bench/sanity_stress.py --pre ""` 60 rounds at conc 33 / 48 / 64 (= NSEQ+1 / 1.5x / 2x) | release practice, run per launch | **0 bad of 1,980 / 0 of 2,880 / 0 of 3,840** |
| `--pre docs,bb --depth 24000 --conc 33` 10 rounds (long prefills first) | | **0 bad of 330** |
| `bench/soak.py --conc 16 --seconds 1500 --lo 200 --hi 30000` | not recorded on this box (the 664 / 792 / 690-request soaks in the release notes are VM100 at the cap) | **882 ok, 0 errors, 0 context-length rejections** in 1,505 s; prompts 192-28,271 tok (median 13,750), **8,135 prompt tok/s**, 150.0 output tok/s, median latency 27.3 s. VRAM **flat**: 16 samples, min == max on every card (30.53 / 30.55 / 30.55 / 30.48 GiB) |

**What `#60084` (32-bit buffer addressing on the `TRITON_ATTN` KV views) is worth on our path: nothing measurable,
and it broke nothing.** 1,562-1,578 against 1,576 aggregate at 32x300 (-0.4%), 7,988 / 7,391 against 8,062 / 7,436
prefill at 120k / 180k (-0.9% / -0.6%), which is inside the spread [context-262k.md](context-262k.md) already
records between instances of the same recipe. The check that would have caught a kernel reading past the end of the
region it now bounds is the ladder (needles found at 60k / 120k / 180k, 30.5 GiB/card high-water) and the
long-prefill sanity run (0 bad of 330) — both clean. The KV-pool figure needs the same treatment the 0.32 note
already gives it: 535,781 and 545,949 across two 0.33 launches against 540,202 / 544,919 / 546,391 across three
0.32 launches. Read it as +-1% launch spread, not as a version effect.

Nothing on the production venv, the production caches or the running launch was touched on 2026-11-02: `0.33.0` was
never installed into `~/venv-r9700`, and `compat/gate.py`'s TESTED lists were deliberately left without it — that
line means "serving gate passed on it", and BetterBench is still outstanding. Two patches (`mtp_allowlist`,
`glm5next_rdna4`) warn on both 0.32.0 and 0.33.0 exactly as they did before.

**One more probe, quoted so it is not mistaken for a delta later.** `bench/bench.py`'s concurrency sweep on the 0.33
launch: aggregate 176 / 459 / 826 / 1,231 / 1,775 / 1,583 / 1,796 tok/s and per-stream median 176 / 119 / 106 / 81 /
58 / 55 / 41 at 1 / 4 / 8 / 16 / 32 / 48 / 64 in flight (300-token outputs). Against the same sweep recorded in
[native-4xr9700.md](native-4xr9700.md) -- 193 / 544 / 900 / 1,203 / 1,537 / 1,454 / 1,532 -- it reads *both* below
at low concurrency and 15% above at 32 and up. That is not a version effect: that table is the **32k recipe on
2026-10-31**, and the paired measurements for today are the `probe32` rows and the ladder, which sit within 1%. The
NSEQ=32 knee the old note describes still shows up the same way: aggregate peaks, and the per-stream spread blows up
past 32 (55.5-63.4 at 32, 33.0-62.5 at 48).

## Hardware note, because it decides the schedule

[incident-2026-11-02/README.md](incident-2026-11-02/README.md) records seven self-resets in one night on this box,
three of them idle, cause still open (BMC/PSU/watchdog material) — and four more the same day. The gate ran anyway,
because the box was already down when the window opened, and it took two launches with it:

- the first 0.33.0 launch came up at 09:17, served `probe32` + the ladder + `quality.py` clean, and died **mid
  request** at 09:39:21 in a platform reset (`pcieport c0:03.1 [12] Timeout`); its log has **zero tracebacks, zero
  HIP errors and zero aborts in 8,584 lines**;
- the 0.32.0 launch before it died the same way at 09:09:32 (`pcieport c4:00.0 [ 7] BadDLLP`), also healthy to the
  last line.

So: **0.33.0 is not what is unstable here**, and a soak that a hard reset cuts in half proves nothing either way.
The soak and BetterBench rows stay open until the platform stops resetting, not because the version failed them.

## Upstream to watch (not in 0.33.0)

| what | state 2026-11-02 | why it matters here |
|---|---|---|
| `#62799` Triton skinny GEMM for 6-128 rows on RDNA4 | open, updated 2026-10-30 — **measured here, not worth carrying** (next section) | the only upstream item in this whole sweep with a two-digit R9700 win in it: the author measures **+28-29% output throughput with DSpark at conc 1-16 on one R9700** (gfx1201), 0.75-0.81x GEMM time over 36 shapes, because BF16 GEMMs with 6-128 rows fall from `wvSplitK` (600 GB/s) to hipBLASLt (420-470 GB/s) |
| `#57861` avoid RowWise FP8 scaled MM on RDNA4 | open, updated 2026-11-01 | vLLM 0.28+ picks RowWise on gfx1200/1201 where ChannelWise decodes faster |
| `#62718` split `skinny_gemms.cu` per kernel family | open, updated 2026-11-02 | ROCm build time (~58 s serialized); matters if we start source-building |
| `#64874` `Workspace is locked` assert in `rocm_fp8_paged_mqa_logits` (GLM-5.3-Flash, high decode concurrency) | open, filed 2026-10-31 | our `models/glm5next.py:270` reserves the decode logits through `current_workspace_manager()` — read this one before moving any GLM launch onto a new pin |
| `#63735` RDNA launch config for `kernel_paged_attention_2d`, `#63054` / `#64614` 64-bit row offsets in the Triton and RDNA-hybrid W4A16 GEMMs | merged, **main only** | RDNA4 fixes that will not reach a wheel until 0.34 |

## Measured, so we do not chase it: `#62799` is not worth carrying here

`tuning/skinny_probe.py` (new, 2026-11-02) graph-times the three per-rank BF16 decode GEMMs of Flash-Next at
TP4 across the row counts the PR targets — the router gate `[M,2560] x [512,2560]^T` (2.62 MB, **51 calls a
step**), the hyper-connection down `[336,10240]` (6.88 MB) and up `[10240,320]` (6.55 MB, 48 calls a step) — for
`torch.matmul`, `torch.mm(out_dtype=fp32)` and our own `r9k_router` kernel forced past its default `MAX_M`.

| shape | graph-replay floor on this box | hipBLASLt | our kernel | net of the 4.9 µs graph-launch |
|---|---|---|---|---|
| empty captured graph | **4.88 µs** | — | — | — |
| 64-element elementwise | 10.64 µs | — | — | ≈ 5.8 µs |
| 128x128 bf16 matmul | 12.12 µs | — | — | ≈ 7.2 µs |
| router `[512,2560]` at M=16 | — | 15.04 µs (bf16 out) / 17.60 µs (fp32 out **+ cast**, which is what the served op does) | 29.2 µs (2-3x worse — the note in PROGRESS: it re-reads activations per expert) | hipBLASLt ≈ 10.1, served path ≈ 12.7 |
| router at M=1..8 | — | 14.8-15.6 µs (168-177 GB/s) | **12.4-14.3 µs (183-212 GB/s)** | the default `MAX_M=8` is the measured crossover |
| HC down `[336,10240]` at M=1..8 | — | 24.2 µs (284 GB/s) | — | — |

Two things kill the idea:

1. **Our decode path is not on hipBLASLt where the PR says it should be.** The launcher serves with
   `R9K_HC_FP8_DECODE=1 R9K_HC_FP8_DECODE_MAXM=255` (`serve/native-flashnext.sh:21`): the HC down/up GEMMs run
   per-token fp8 on our WMMA kernels for every decode M up to 255, and `R9K_ROUTER_MAX_M` (default 8) puts the
   router gate on `r9k_router` for the small-M end. The band the PR fills is mostly already filled here.
2. **What is left is latency-bound, not bandwidth-bound.** hipBLASLt reads the 2.6 MB router weight in 15.0 µs —
   a 128x128 matmul (12.1 µs) is 2.9 µs of it. The PR's 420-470 GB/s figure is for much larger weights; on our
   per-rank decode shapes the floor is the launch, so a 600 GB/s streaming kernel buys almost nothing: the whole
   recoverable envelope at M=16, taking the band floor as the best case, is `(17.6 - 12.1) us x 51 calls = 280 us`,
   i.e. **2.2% of a 13 ms step**, and that assumes a kernel as fast as a trivial matmul.

Two smaller findings worth keeping, both from the same probe:

- **`R9K_ROUTER_MAX_M` above 8 is a loss, and the kernel's own limit is 32.** At M=16/24/32 the r9k router costs
  29.2 / 41.1 / 43.7 µs against hipBLASLt's 14.6-14.9 — so raising the knob to the kernel's `r9k_router_max_m()`
  (32) would add ~1.5 ms a step. The default 8 is not timid, it is the measured crossover.
- **The one candidate the probe exposes is the cast, not the GEMM.** Above `MAX_M` the served op is
  `torch.mm(out_dtype=fp32)` followed by a bf16 cast (`r9700_vllm/router.py:72`), and the cast measures
  **2.6 µs a call** (15.04 -> 17.60) = **130 us a step, ~1.0%**, plus 51 fewer launches. Handing the gate fp32
  instead (its softmax/top-k works in fp32 anyway) is a one-knob A/B, not a new kernel. It changes router rounding
  on near-ties, so it owes `tests/test_router_r9k.py`'s top-10 agreement and the GSM8K/needle gate — and by the
  bar this stack already uses for `moe/fold.py` (~0.1 ms of a 16.7 ms step "does not buy two patches"), ~0.13 ms
  buys one flag and no patch at all. **Not run yet.**

## Verdict and plan

**Follow v0.33.0; do not follow `main`.** As of 2026-11-02 10:19 UTC the box is serving the standing 262k recipe on
`~/venv-r9700-033` (`vllm 0.33.0+rocm72`), and everything on the standing gate list that this platform lets us run
has come back level: integration, four kernel gates, `probe32` at three protocols, the retrieval ladder with its
needles, GSM8K + needle, `sanity_stress` at NSEQ+1 / 1.5x / 2x and with 24k prefills first, and a 25-minute
16-client soak with VRAM flat. **What is still open before the pin moves for good**: BetterBench (the one quality
number this gate has not produced), and a soak that survives to the end on a box that is not resetting.

Steps, in order:

1. Fix the wheel URLs/flavor (done with this note).
2. Kernel gates that need VRAM, plus the 262k launch on `~/venv-r9700-033` with separate caches, measuring the two
   things 0.33.0 could change — `set_triton_ptr_range` on our fp8 KV path and the KV-cache token count. **Done**;
   both came out inside the launch-to-launch spread.
3. `sanity_stress` at NSEQ+1 / 1.5x / 2x, `soak` 1500 s, GSM8K, `bench/probe32.sh` bands. **Done**, table above.
   Not done: BetterBench, `bench/humaneval.py`, the 27B TP2 and GLM configs (that launch lives on the eight-card
   VM, and it owes `#64874` a read first).
4. When BetterBench is in: `0.33.0` into `compat/gate.py`'s TESTED lists, `amd-aiter 0.1.26` in the venv, the image
   `BASE` (the nightly line has no 0.33-era ROCm 10 image pinned yet — a separate question from the venv), the
   README/notes pin text and the release note.
5. Separately: `post_gemm2_hook` vs `moe/fold.py`, and `get_rdna_version()` replacing our arch probes. Both are
   plugin changes with their own A/B, not part of a pin bump.
