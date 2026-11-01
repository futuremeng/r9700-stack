# Native bring-up on the 4x R9700 box (no docker, no root) — 2026-10-31

The box: 4x Radeon AI PRO R9700 (gfx1201, 31.9 GiB each, PCIe **Gen5 x16** under four embedded PLX
switches), EPYC Genoa, 93 GiB RAM, Ubuntu 24.04, **ROCm 10.1** installed system-wide
(`/opt/rocm` -> `core-10.1`, HIP runtime 7.16.26385, hipcc present, **no `-dev` packages**), amdgpu
driver 7.1.9.31600000. No docker, and `sudo` needs a password: `serve/serve.sh` (`sudo docker run`)
cannot run here, and the pinned `vllm/vllm-openai-rocm` nightly cannot be pulled (registry-1.docker.io
is unreachable from this network; huggingface.co too — weights came from `hf-mirror.com`).

So this is the same product (stock vLLM + `r9700_vllm` + libr9k) installed **into a venv** instead of
the image: `~/venv-r9700`, `vllm==0.32.0+rocm723` / `torch==2.13.0+git733fca1` / `triton 3.8.0` from
`https://wheels.vllm.ai/rocm/5df63a72.../rocm723/`, plugin `pip install -e .`, `libr9k.so` built by
`kernels/build.sh` with the host hipcc for gfx1201. `serve/native-flashnext.sh` is the container-free
launcher (same knobs as `serve/serve.sh`).

## What the host needed (three unpacked .debs, no root)

| problem | fix |
|---|---|
| `hip/hip_runtime.h` missing (no `hip-dev` on the host) — `build.sh` fails | `amdrocm-runtime-dev10.1_10.1.0-3_amd64.deb` from `stable.repo.amd.com/rocm/core/packages/ubuntu2404/`, `dpkg-deb -x` to `~/opt/rocm-dev`, then `HIPCFLAGS=-I$HOME/opt/rocm-dev/opt/rocm/core-10.1/include ./build.sh`. The build then passes every export check and the `hidden_hostcall_buffer` check (P2P-safe). |
| the ROCm 7.2-era wheels link OpenMPI 4, which the host does not have (`libmpi_cxx.so.40: cannot open shared object file`) | `apt-get download libopenmpi3t64 openmpi-common libpmix2t64 libhwloc15 libevent-2.1-7t64 libevent-pthreads-2.1-7t64 libevent-extra-2.1-7t64 libevent-openssl-2.1-7t64 libmunge2`, `dpkg-deb -x` to `~/opt/openmpi`, `LD_LIBRARY_PATH=$HOME/opt/openmpi/usr/lib/x86_64-linux-gnu` (the launcher sets it). |
| the pip `amdsmi` wrapper loads `/opt/rocm/lib/libamd_smi.so` (ROCm 10.1) and dies on `amdsmi_set_gpu_clk_range` | the wheel's own `libamd_smi.so` goes where its loader looks first: `ln -s .../site-packages/amdsmi/libamd_smi.so ~/venv-r9700/lib/lib/libamd_smi.so.26`. |
| torch/RCCL use the **system** `libamdhip64.so.7` / `libhsa-runtime64` / `librccl` (checked with `ldd`) | so libr9k compiled by the host ROCm 10.1 hipcc is the right binary — no ABI mismatch. |

## One plugin change (this tree)

`models/qwen4_exp.py`: the bf16 -> MXFP4 requant of the MTP MLP (rev 6fe3e746 ships fused bf16 experts)
chunked its rows at a fixed `1 << 18` = **2.68 GiB of fp32 per chunk on the device, several copies at
once**. On this checkpoint that peak does not fit after the target weights (27.85 GiB allocated, the
draft's first `bucketize` asked for 5 GiB): the launch died in `quantize_mxfp4_search`. The chunk row
count is now `R9K_MTP_QUANT_ROWS` (default unchanged at `1 << 18`); the launcher runs it at 32768.
Load-time only, no numerics change: weights 19.16 GiB a GPU, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Gates on this box

* `kernels/build.sh`: all 68 export checks, no hostcall kernels.
* kernel unit gates (`test_moe_mxfp4`, `test_ple_int6`, `test_gemm_fp8`): ALL OK
  (MoE gate_up 598 GB/s at 139 touched experts; fp8-block GEMM 1,126-1,230 GB/s; lm_head fp8 523 us vs bf16 1,021).
* `tests/test_integration.py` against **vLLM 0.32.0** (the release, not the pinned nightly): **ALL OK**
  (29 checks; the plugin logs two "patch untested on vLLM 0.32.0" lines for `mtp_allowlist` /
  `glm5next_rdna4` and installs both).
* `bench/quality.py 100 8`: GSM8K no-think **97/100**, needles found at 3.8k / 15.2k / 30.4k tokens.
* `bench/sanity_stress.py --rounds 6 --conc 8`: see below.

## Speed (Qwen3.8-Flash-Next, TP4, MTP-3, 32k context, NSEQ=16, UTIL=0.97, cards uncapped)

checkpoint `tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ` revision 2026-10-01 (= rev 6fe3e746), 110.6 GiB
on disk: 64.5 GiB routed experts (512 experts), 38.7 GiB int6 PLE (pinned host, 9.7 GiB/rank UVA), 7.2
GiB everything else. **19.16 GiB of weights a GPU**, KV cache **554,982 tokens**, draft acceptance
4,088/7,290 = 56.1% (2.68 tokens a step).

| | this box (Gen5 host, uncapped) | README v0.3.8 gate (210 W cap) |
|---|---:|---:|
| single-stream decode | **186.7 tok/s** | 225 tok/s |
| step (median ITL, conc 1) | **14.3 ms** | 14.3 ms |
| ITL conc 4 / 8 | 18.1 / 22.6 ms | - |
| aggregate @4 / @8 / @16 (`bench/bench.py`) | 372 / **841 / 1,164** tok/s | 530 (@8) / 944 (@16, probe) |
| prefill 1,951 / 7,789 / 31,166 tok | **7,702 / 9,633 / 8,963** tok/s | 5,175 / 7,141 / 7,233 (2k/8k/32k) |
| KV cache | 554,982 tokens | 518,481 |

Decode is the one number below the gate: same 14.3 ms step, but 2.68 tokens a step here against ~3.1 in
the reference run — acceptance, not kernels (the reference reports 62.2% acceptance; `bench/bench.py`'s
essay prompts at temperature 0 give 56.1%).

Against the **Strata** server that ran on this same box the same afternoon (Qwen3.8-Flash-Next IQ3_S
GGUF, 4-card layer split, 8 slots): single-stream decode 90-97 tok/s, prompt-4k 2,369 tok/s, prompt-16k
4,185 tok/s, four requests at once 166-194 tok/s. The vLLM stack with the MXFP4 experts + fp8 dense path
is ~2x single-stream decode, 2.3-4x prompt processing and ~6x at 16 concurrent requests — for ~4.5
effective bits a weight against IQ3_S's 3, and with a 555k-token KV pool against Strata's ~22k slots.

## Knob A/B on this box (each change measured alone, one commit each)

**`--max-num-batched-tokens` 4096 -> 8192** (native TP4 default now; two launches, `bench/prefill.py` repeated 3x
at NBT=8192 so the spread is shown, `bench/bench.py`, `bench/itl.py`):

| | NBT 4096 | NBT 8192 |
|---|---:|---:|
| prefill 7.8k tok | 9,331 | **9,686-9,867** (+4-5%) |
| prefill 31k tok | 8,909 | **9,190-9,221** (+3.3-3.5%) |
| single-stream decode | 187.0 | 187.0 |
| aggregate @8 / @16 | 842 / 1,163 | 844 / **1,214** (+4.4%) |
| KV pool | 563,692 tok | 499,815 tok (-64k, still 15.2x at 32k) |

vLLM itself warns about the 4096 chunk under speculative decoding (`max_num_scheduled_tokens is set to 4096
... may lead to suboptimal`). `serve/flashnext.sh` stays at 4096: at TP2 with offloaded experts that chunk
price has no KV room.

**Negative control: the N-rank P2P all-reduce is worth +20.6% decode on this host** (`R9K_ARN=0` = RCCL only,
same launch otherwise, NBT=8192). That is a different answer from `PROGRESS.md` 2026-09-18 ("P2P gives ~nothing
at TP2") -- that box is PCIe **Gen3** behind PLX switches and runs TP2 with the experts offloaded, where host->GPU
expert traffic, not collectives, owns the step. This box is **Gen5 x16** (measured 49-50 GB/s per card, NVMe 5.8 GB/s)
and everything is resident, so the 96 collectives a Flash-Next step dominate:

| | plugin N-rank AR (default) | `R9K_ARN=0` (RCCL) | the path is worth |
|---|---:|---:|---:|
| single-stream decode | 187.0 tok/s | 155.1 | **+20.6%** |
| median ITL, conc 1 | 14.1 ms | 16.3 ms | 2.2 ms a step |
| aggregate @8 / @16 | 843.8 / 1,214 | 749.8 / 1,170 | +12.5% / +3.8% |
| prefill 7.8k / 31k | 9,686-9,867 / 9,190-9,221 | 8,983 / 8,479 | +8-10% |
| KV pool | 499,815 tok | 491,934 tok | RCCL keeps ~8k more tokens of scratch

So the step's remaining collective time (48 layers x 2 collectives, one-shot 6.8-10 us measured on a 4-rank
stand-in in `notes/glm53.md`) bounds any further comm work at roughly 1 ms a step, ~7%.

**Cold-start breakdown** (same launch, warm compile/Triton cache): engine start -> weight load starts 32 s,
safetensors + conversion **144 s**, compile + graph capture + KV profile ~60 s, total ~236 s. The disk is not the
limit (O_DIRECT single stream: 5.8 GB/s); the load is per-tensor work and page-cache thrash -- 4 ranks stream the
same 110.6 GiB while 38.7 GiB of RAM is pinned for the PLE table, and two of the expert shards took 24 s and 43 s
while their neighbours took 3 s. A pack/pre-converted + parallel-read loader is where those ~2 minutes live.

**`--max-num-seqs` 16 -> 32** (native TP4 default now; same launch otherwise, `bench/bench.py aggregate`, a
~20k-token concurrent-prefill probe, `bench/quality.py`, `bench/sanity_stress.py`):

| | NSEQ 16 | NSEQ 32 |
|---|---:|---:|
| aggregate @16 | 1,155-1,214 | 1,158-1,197 |
| aggregate @32 | 1,207 / 1,226 | **1,429 / 1,437 / 1,440** |
| aggregate @48 | - | 1,366 (the knee is 32, not 48) |
| single-stream decode | 185.6-187.0 | 186.7-186.9 |
| long-context prefill @16 / @24 (~20k tok each) | 9,376 / 9,311 | 9,323 / 9,298 |
| KV pool | 499,815 tok | 491,520-498,571 |

At NSEQ=16, 32 clients only buy +5% over 16 (they queue in two waves); at 32 they buy +18%. Prefill is
compute-bound and does not move with slots, and vLLM still admits only 15.2 requests at the full 32k context,
so nothing breaks for long-context traffic. Gate at the new defaults: GSM8K 97/100, needles 3/3,
`sanity_stress --rounds 4 --conc 16` 0 bad of 64.

**`R9K_AR4_SDMA=1` does not pay here either** (same launch otherwise, NSEQ=32 / NBT=8192): single-stream decode
187.0 and ITL 14.0 ms unchanged, aggregate @8 844 -> 822, @32 1,429-1,440 -> 1,402, prefill 7.8k 9,686-9,867 ->
9,736. Matches `PROGRESS.md` 2026-10-07 (the DMA-push variant of the ar4 stage loses): the kernel-driven pushes
already saturate the Gen5 uplink and the SDMA path just adds a queue round trip. Not adopted.

**Load-time decomposition** (`--load-format dummy` for the floor, `R9K_FOLD=0` for the folded-exponent repack,
`bench`-free launches, the `Model loading took N seconds` line, NSEQ=32 config):

| | seconds |
|---|---:|
| dummy (construct + PLE pinned table + graph-independent init, no checkpoint) | **8.8** |
| real load, `R9K_FOLD=0` | 116.5 |
| real load, defaults | 143.8 |
| the same 110.6 GiB read once, O_DIRECT single stream | ~19 |

So the fold repack is ~27 s and the remaining ~108 s is safetensors iteration plus the per-tensor repacks
(MXFP4 expert repack, fp8 hyper-connection quantization, int6 PLE shard writes) -- four ranks doing it at once
while 38.7 GiB of RAM is already pinned, which is also why the per-shard progress bars disagree wildly in the
same launch (1:13 to 2:52 across ranks, and identical expert shards taking 3 s, 24 s and 43 s). py-spy cannot
attach without ptrace on this box (`Permission Denied`), so a real flame graph needs `kernel.yama.ptrace_scope=0`
or an in-tree load profiler. `R9K_MTP_MLP=off` is not a usable probe: the draft does not load without it.

## Utilization on this box (steady state, sampled at conc 1 / 8 / 16 / 32, NSEQ=32 / NBT=8192)

| conc | tokens/s (400-token streams) | W a card | sclk | box CPU | tok per joule |
|---:|---:|---:|---:|---:|---:|
| 1 | 187-200 | 212-234 | 3.2-3.4 GHz (full) | 5.4% | 0.21 |
| 8 | 911 | 264-316 | 2.3-3.4 GHz | 5.4% | 0.77 |
| 16 | 1,239 | 282-309 | 2.0-3.3 GHz | 5.4% | 1.05 |
| 32 | 1,533 | 288-334 | **1.9-3.3 GHz** | 5.4% | 1.30 |

* **VRAM is the binding resource**: 31.3-31.4 GiB of the 31.86 GiB usable a card -- 19.2 weights + 8.2 KV
  (491,520 tokens) + 1.9 graphs + ~1.5-2 context/activation. `rocm-smi --showpids` per worker 31.3-31.4 GiB.
* **CPU is nowhere near a limit and does not move with concurrency**: ~11 of 192 cores at every conc -- each
  `VLLM::Worker_TP` 2.06 cores (rank 0 3.03), `VLLM::EngineCore` exactly 1.00 (the scheduler spin loop), the
  API server 0.04-0.05. 247 core-seconds per 24 s window at conc 1 and at conc 32. GPU use reads 100% throughout,
  and the KFD `CU OCCUPANCY` column reads 0-6%, so the cards are latency/bandwidth/power-bound, never CU-bound.
  The idle cores cannot shorten decode: vLLM's expert offload keeps expert *weights* in host RAM but computes
  them on the GPU -- there is no host expert pool like Strata's `32 expert-pool workers + the host thread`, and
  on a Gen5 host with everything resident a host-side gather+compute would lose (EPYC ~200 GB/s aggregate
  against 4 x ~512 GB/s of GDDR7). The headroom is real but it is only useful for what the CPU already does:
  the 144 s checkpoint pass, torch.compile, and tokenization (4% at conc 32, no pressure).
* **What caps concurrency here is the power envelope, not memory or CPU**: ~300 W a card with the DPM pulling
  sclk from 3.3 GHz (single stream) down through 2.0-2.3 GHz at 16-32 streams. Efficiency therefore improves
  ~6x from one stream to 32 (0.21 -> 1.30 tok/J) and then flattens -- the same wall the 210 W reference caps
  sit on the far side of.
* **Host RAM**: 56 GiB used + 39 GiB cache, `MemAvailable` 37 GiB; Pss 12.0 GiB a worker, of which the int6 PLE
  table is 9.7 GiB pinned (`hipHostRegister`, so it does not show in `Unevictable`, which reads 30 MB).
  Swap holds 4 GiB and the lifetime `pswpout` is 96.4M pages (~360 GiB): the load streams 110.6 GiB through a
  93 GiB box while pinning 38.7 GiB. During serving the swap traffic is negligible (pswpin +101 pages in 20 s
  at conc 32).
* Operational trap found while measuring: killing the API server alone leaves the four `VLLM::Worker_TP` and the
  `VLLM::EngineCore` alive holding 31 GiB a card (`rocm-smi --showpids` is the way to see them), and the next
  launch dies with `Free memory on device cuda:0 (0.0/31.86 GiB)`. Kill those PIDs too; that is also where the
  swap residue came from.

**Concurrency: always read the per-stream rate next to the aggregate** (`bench/bench.py` and
`bench/harness.py` now report both: `aggN_tok_s` is what the box earns, `streamN_tok_s` is what a client feels --
each request's own completion tokens over its own latency, queueing and TTFT included). Same server, 300-token
outputs:

| in flight | aggregate | per stream (median) | spread | median wall for 300 tok |
|---:|---:|---:|---:|---:|
| 1 | 193 | 193.0 | - | 1.6 s |
| 4 | 544 | 146.8 | 136-157 | 2.0 s |
| 8 | 900 | 119.6 | 113-123 | 2.5 s |
| 16 | 1,203 | 79.8 | 75-90 | 3.8 s |
| **32** | **1,537** | **50.5** | 48-54 | 5.9 s |
| 48 | 1,454 | 48.2 | **30-53** | 6.2 s |
| 64 | 1,532 | 34.8 | 24-56 | 9.2 s |

The aggregate saturates at ~1,530 from 32 streams on (the power wall above), while the per-stream rate keeps
falling and its spread blows up past 32: at 48 in flight the aggregate drops and the slowest stream is at 30.4
against the median 48.2. So the `NSEQ=32` knee shows up twice, once in throughput and once in fairness, and the
aggregate alone would have hidden the second one. Even eight streams in flight give every client 119.6 tok/s --
faster than the fastest single stream the GGUF/Strata stack on this box managed (90-97).

## Two-card probe (2026-10-31): what host RAM costs, in GiB per rank

Flash-Next on two cards is the repo's offloaded-expert recipe (`serve/flashnext.sh` defaults to TP2 with
`OFFLOAD_GB=34` a rank, "needs the 256 GB VM"). This box has **93.8 GiB**, so the probe was run with the biggest
offload that fits: `GPUS=0,1 TP=2 OFFLOAD_GB=12 UTIL=0.96 MAXLEN=32768` (NSEQ=8 / NBT=4096 from the launcher's
new TP-aware defaults). Cards 0 and 1 hang off separate PLX switches (0000:00:01.1 and 0000:00:03.1), which is
what README.md:131 asks for; the box is single-NUMA so there is no local/remote choice to make.

| | TP4 (running) | TP2 probe | TP2 at the repo default |
|---|---|---|---|
| weights resident | 19.16 GiB a card, none offloaded | 24.56 GiB a card, 12.22 offloaded | ~14 GiB a card, 34 offloaded |
| `Available KV cache memory` | 8.2 GiB -> **491,520 tok**, 15.0x at 32k | **2.18 GiB -> 134,795 tok**, 4.11x | ~167k tokens (their box) |
| load time | 143.8 s | **327.5 s**, and it swapped 4.1 -> 6.1 GiB on the way | -- |
| single-stream decode | 187 tok/s | 62 tok/s warm, then 18.9, then **9.5** minutes later | -- |
| aggregate / per stream | 900 / 119.6 @8, 1,203 / 79.8 @16, 1,537 / 50.5 @32 | 147 / 18.8 @8, 135 / 12.5 @16 | 502 @16 on their box |
| prefill | 9,633 / 9,437 tok/s (7.8k / 31k) | 4,190 / 4,582 (8.9k / 28.9k) | -- |

The decode collapse is the interesting part: 62 -> 18.9 -> 9.5 tok/s over a few minutes of serving, while the
swap traffic during a 100-token decode was 60 pages. It is not disk thrash -- with 2.18 GiB of KV and most of the
expert set living in host RAM, the streamed-expert path is what degrades once the resident expert set churns.
Host RAM does not just size the cache here, it decides whether two cards are usable at all.

**The measured host-memory formula** (components from the log, they add up to the measured RSS exactly):

```
host GiB  ~=  38.7 (PLE int6 table, TP-independent: 160,000,768 rows x 130 B = 19.4 GiB a rank at TP2,
                   9.7 GiB a rank at TP4 -- the same 38.7 GiB either way)
            + TP x OFFLOAD_GB                       (log: "Total CPU offloaded parameters: 12.22" a rank)
            + TP x 4.3                              (torch / python / CUDA host side per rank)
            + page cache for the 110.6 GiB checkpoint read
```

TP2 probe steady state: 2 x 35.9 GiB + 0.8 = 72.6 GiB RSS, peak 86.4 GiB + 6.1 GiB swap = **92.5 GiB touched**
against 93.8 installed, low-water `MemAvailable` 14.6 GiB. Per rank 19.4 + 12.2 + 4.3 = 35.9 = the measured RSS.

So, for the RAM question:

* **4 cards fits, but the load leans on swap.** Steady state is comfortable -- 56 GiB used, `MemAvailable` 37 GiB,
  nothing faulting during serving (60 pages in during a 100-token decode) -- and there is 30+ GiB of headroom.
  But on a completely clean start (87 GiB available, 102 MiB swapped) the TP4 load still ends with **4.1 GiB in
  swap, ~0.53 GiB of it inside each rank**: the pinned 38.7 GiB plus four ranks' load-time temporaries plus the
  page cache the kernel wants to keep for a 110.6 GiB read is more than 93.8 GiB wants to hold. It costs little
  (decode measured at baseline right after), but 128 GiB would take the last of that risk out.
* **Overlapping two servers on this box measurably costs speed.** The launch that ran while the TP2 ranks were
  still alive (78 GiB used) came up with ~0.5 GiB swapped a rank and read **171.3 tok/s single-stream**;
  the same config started clean reads **186.4**. Aggregate across those two states stayed inside the usual
  restart spread (1,148-1,190 @16). That is the practical reason for the launcher's new pre-flight guard.
* **2 cards with the repo's default needs ~135 GiB**: 38.7 + 2x34 + 8.6 = 115 GiB resident, plus ~20 GiB of page
  cache for the checkpoint read. 128 GiB would be tight (that is only 13 GiB of headroom over the resident part),
  **192 GiB is the size that makes TP2 a comfortable, supported config** -- the reference box has 256 GiB.
* Every extra 1 GiB of `OFFLOAD_GB` costs **2 GiB** of host RAM at TP2 (4 GiB at TP4), and at 93.8 GiB we could
  only afford 12 a rank, which left 2.18 GiB of KV -- i.e. the two-card path is boxed in from both sides here:
  offload more and the host runs out, offload less and the KV pool disappears.
* **1 card is not a memory question at all** (well, it is: 38.7 PLE + 64.45 experts + 4.3 = ~107 GiB resident
  before page cache, so >128 GiB just to start), and no recipe in this repo runs Flash-Next at TP1. Treat the
  27B dense line as the single-card model.
* Practical reading: more RAM buys *redundancy and the two-card fallback*, not speed. Even done properly (192 GiB,
  34 GiB a rank), the repo's own two-card numbers are ~45% of four-card aggregate.

One more operational trap, hit twice while probing: **SIGTERM does not stop the workers.** `pkill -f 'vllm serve'`
(and even SIGTERM to the ranks) leaves `VLLM::Worker_TP*` alive holding ~27-31 GiB a card, and the next launch
fails with `Free memory on device cuda:0 (0.0/31.86 GiB)`. Use `kill -9` on the PIDs from `rocm-smi --showpids`.

## Draft depth: the default on this box is now MTP=2 (was 3), and the reason is the row budget

Two fresh instances, 3 x 300-token samples at 32 in flight each side, plus ITL and the quality gate:

| | MTP=3 | MTP=2 |
|---|---|---|
| rows per decode step at NSEQ=32 | 32 x 4 = **128** | 32 x 3 = **96** |
| accepted tokens a step (1 stream) | 2.52 of a possible 4 | 2.22 of a possible 3 |
| single-stream decode (bench.py, TTFT excluded) | **180.1** | 170.4 |
| single-stream, 300-tok request incl. TTFT | **197.6** | 179.8 |
| median ITL @1 / @8 / @32 | 14.0 / 23.4 / 49.8 ms | 13.0 / 20.8 / **37.1 ms** |
| @16 aggregate / per stream | 1,202 / 80.7 | **1,265 / 83.4** |
| @32 aggregate, 3 samples -> median | 1,385, 1,488, 1,509 -> 1,488 | 1,762, 1,538, 1,750 -> **1,755** |
| @32 per stream | 45.8 - 49.6 | **51.0 - 57.8** |
| GPU KV cache | 491,520 tok | **502,442 tok** |
| prefill 7.8k / 31k | 9,741 / 8,932 | 9,832 / 9,208 |
| GSM8K / needles | 97/100, 3/3 | 98/100, 3/3 |

So MTP=2 wins everywhere except one-stream latency: **+18% aggregate and +18% per-stream at 32 in flight, ITL at
32 down 26%**, and 10,922 more KV tokens -- for **9% of single-stream decode**. The mechanism is the same wall the
utilization section found: above ~16 streams the cards sit at ~300 W, and a decode step costs NSEQ x (MTP+1)
rows. The third draft row buys ~0.30 accepted tokens a step and costs 32 rows a step, so it is worth paying for
only when the box is otherwise idle. `MTP=3 bash serve/native-flashnext.sh` is the single-interactive-stream setting.

Two things this measurement exposed about the method, worth keeping in mind for every future A/B here:

* @32 samples are **bimodal** -- 1,538/1,564 against 1,750/1,762/1,755 for the same config, tracking whether the
  DPM is holding ~2.0 GHz or ~3.1 GHz. Medians of at least three samples, never a single number.
* low-concurrency aggregates are noisy at the 10% level (the first @4 run read 328, its repeat 471); prefer
  @8/@16/@32 for A/B and use ITL as the tie-breaker since it is per-step and much tighter.

Both follow-ups, measured:

* **MTP=1 completes the curve**: @32 1,553 / 1,700 / 1,638 -> median **1,638**, below MTP=2's 1,755; @16 1,171 /
  75.2; single-stream 146.2. It does have the tightest ITL of the three -- 11.9 / 17.6 / **29.8 ms** at 1 / 8 / 32
  streams -- and the largest KV pool (527,085 tokens). So the aggregate peaks at a draft depth of **2** for this
  workload: 1,488 (3) -> 1,755 (2) -> 1,638 (1), with accepted tokens a step at one stream 2.52 / 2.22 / 1.74 out
  of a possible 4 / 3 / 2. If the thing you care about is per-token latency at high concurrency rather than
  tokens/s, MTP=1 is the setting to look at.
* **NSEQ=48 with MTP=2** (144 rows a step, still above MTP=3/NSEQ=32's 128) does not move the knee: @32 median
  1,794 (1,745 / 1,809 / 1,794) against 1,755 at NSEQ=32 -- inside the spread; @16 1,252 against 1,265; ITL
  12.9 / 21.0 / 37.4 ms, indistinguishable. And 48 in flight earns 1,811 aggregate at **39-40 tok/s a stream**
  against 57-59 with 32 in flight, for 4,854 fewer KV tokens. `--max-num-seqs 32` stands.

Running default after all of this: `MTP=2 NSEQ=32 NBT=8192 UTIL=0.97`, measured on the restored instance at
1,779 aggregate / 58.0 a stream with 32 in flight, 189.2 single-stream, KV 496,374 tokens.

## Cold start anatomy (`R9K_LOAD_PROF=1`, `r9700_vllm/utils/prof.py`, `bench/loadpath.py`)

`Model loading took` for four TP4 launches of the same config: **143.8 / 149.3 / 175.7 / 178.8 / 179.9 s** -- a
25% spread that is page-cache state, not code. Inside it, per rank:

| | measured |
|---|---|
| safetensors iterator pull | **0.2 s** (zero-copy mmap views) |
| consumer side ("place") | 77.5 - 156 s |
| ... of which the experts group | **120.8 s** -- 192 tensors, 59.8 GiB of checkpoint bytes a rank |
| ... 1306 small tensors ("other") | 30.2 s |
| ... the one lm_head tensor | 4.6 s |
| ... hc (387 tensors) | 0.5 s |
| **everything the plugin does at load** | **1.3 s**: fold_stats 0.4, `_prep_mxfp4` 0.4, `permute_in_place` and `pack_scales_gpu` ~0, ct linear 0.1 |
| draft bf16 -> MXFP4 requant | 4.9 s |
| `--load-format dummy` (allocate, no content) | 8.8 s |
| reading the 110.6 GiB off disk once | ~19 s (5.8 GB/s) |
| one real 0.78 GiB fused expert tensor, bulk H2D | **0.02 s = 47 GiB/s**; per-expert strided TP slice only 2.5x slower |

So the load is neither repacking nor the plugin: each of the four ranks iterates the **whole** checkpoint
(59.8 GiB of expert bytes plus 38.8 GiB of PLE, so ~390 GiB of host reads across four ranks) against one 5.8 GB/s
disk and 93.8 GiB of RAM, and that is where the minutes go. The TP shard is a *column slice of every expert*, so
the expert-id filter vLLM offers (`local_expert_ids`) cannot skip anything at TP -- every rank needs all 512
experts, a quarter of each.

Two claims this corrects:

* The fold guard is **0.4 s**, not the ~27 s the `R9K_FOLD=0` ablation implied: that A/B compared two different
  page-cache states. `R9K_FOLD=1` stays adopted for its *runtime* win; its load cost is noise.
* `--load-format fastsafetensors` is not available on this stack: it fails at the first tensor with
  `[FAIL] framework hinted GPU runtime 'libamdhip64.so'` (fastsafetensors 0.5.0 is CUDA-only).

~~Queued next: tune vLLM's safetensors prefetch (8 threads / 16 MiB a rank by default).~~ **Wrong on both counts,
and the run that tested it wedged the box** (see `notes/load-oom-svm-wedge.md`):

* There is no default prefetch here. `weight_utils.py:1115` is `should_prefetch = strategy == "prefetch"`, and
  with no `--safetensors-load-strategy` it auto-enables only on NFS/Lustre *and* when the checkpoint fits in 90%
  of available RAM. Every rank of every Flash-Next launch logs: `Filesystem type for checkpoints: EXT4.
  Checkpoint size: 110.60 GiB. Available RAM: 40.74 GiB.` + `Auto-prefetch is disabled ...`. So
  `--safetensors-prefetch-num-threads 1` tunes a path that never runs; the A/B arms were identical.
* Forcing `--safetensors-load-strategy=prefetch` on this stack would stage later shards into the same RAM that is
  already 40 GiB short of the 110.6 GiB the four ranks stream -- vLLM logs `This may cause out-of-memory errors`.
  Not attempted, and it should not be.
* Page-cache cold vs warm is **not** worth 2.02x -- that pair (`208.4 / 103.2 s`) was read off a machine that was
  already degrading under an unmapped previous attempt. Re-measured 2026-11-01 with `bench/gate.sh` verifying
  teardown between launches, the totals are 115.67 s boot-cold vs 120.60 s warm, i.e. the *sum* barely moves --
  but that is an artifact of two components cancelling (see "Cold-start anatomy, five instances"): the target
  experts term is 33-108 s and the draft pass 5.5-76 s depending on cache state. Fourth load-path number in this
  line of work that turned out to be machine state, not the config under test.
* The two-launch protocol still applies, and now a hard gate: no launch unless every `vllm serve|VLLM::` process
  is gone and `MemAvailable >= 55 GiB` -- see `bench/coldstart_ab.sh`, which replaces `bench/prefetch_ab.sh`.

## Throughput anchor with the protocol pinned (2026-11-01, post-reboot)

A 4-5% @16 deficit against the recorded 1,265 turned out to be mostly a protocol question, so the protocol is now
written down (`bench/probe32.sh`, same `bench.aggregate(conc, max_tokens)` calls the numbers came from):

| protocol | agg tok/s | per-stream median |
|---|---|---|
| 32 in flight, 300-token waves | 1,600.1 / 1,586.5 / 1,598.0 -> **1,598** (+-0.5%) | 51.8 |
| 32 in flight, 1,200-token waves | 1,836.3 / 1,808.8 -> **1,823** | 58.1 |
| 16 in flight, 300-token waves | 1,094.4 / 1,131.3 / 1,107.1 -> **1,107** | 72.6 |
| 16 in flight, 200-token waves (`bench.py run`, the recorded protocol) | 1,210.9 / 1,231.2 / 1,168.4 / 1,185.9 -> **1,198** | 79.1 |

What that says: at a fixed protocol the box repeats to +-0.5%, but **`max_tokens` moves the @32 aggregate by 15%**
(1,598 at 300 vs 1,823 at 1,200) and the sign of the length effect flips between 16 and 32 streams. Every historical
aggregate therefore has to be read with its `max_tokens`; the recorded MTP table ("@32 1,762 / 1,538 / 1,750 ->
median 1,755, 57.6 a stream") sits closer to long-wave behaviour than today's 300-token waves read, which is where
most of the apparent -9% lives. On the apples-to-apples one (`bench.py run`, 200-token waves at 16 in flight) today
reads 1,198 against the recorded 1,265, i.e. **-5%**, with single-stream decode unchanged (171.6 median against
170.4 recorded) and the speculative config verified in the launch line: `{'method': 'mtp',
'num_speculative_tokens': 2}`, acceptance 148,182/99,534 = **1.49 accepted a draft** (pos0 82.6%, pos1 66.3%).

The power envelope is *not* the explanation, and the earlier suspicion of it was a measurement error: sampling power
around a 6-second wave gives a 258 W median. Gated on `gpu_busy_percent == 100` (140 samples) the cards sit at
**299 W median, 300.0 W max on all four**, flat, with GPU busy at 100 and 44-47 C -- exactly the ~300 W ceiling
recorded above, and no powercap is installed in this guest (`/usr/local/sbin` empty, no `r9700-*` units; the 225 W
cap belongs to the production guests, see `host/README.md`). Other fingerprints: GPU KV cache **496,374 tokens**
(15.15x at 32k) against 502,442 recorded for MTP=2 (-1.2%), `Available KV cache memory: 7.25 GiB`; GSM8K 98/100 and
needles 3/3; temps and counters clean. Open, low priority: whether that -5% and the 1.2% smaller KV pool are
instance drift or something real -- settle it by rerunning `probe32.sh` on the next launch rather than by trusting
the single historical samples.

## Cold-start anatomy, five instances with the phase timers (2026-11-01)

Same config every time (TP4, NSEQ=32, MAXLEN=32768, OFFLOAD_GB=0), every relaunch behind `bench/gate.sh`,
`R9K_LOAD_PROF=1` with the phase snapshots. Per rank:

| inst | Cached at start | target `place` | `experts` 192t/59.8 GiB | draft wall (requant) | load total | disk reads a rank |
|---|---|---|---|---|---|---|
| 0 boot-cold | 0.35 GiB | 98.0 s | 87.8 s | ~6 s | **115.7 s** | -- |
| 1 | 39 GiB | 100.7 s | 88.2 s | 8.0 (6.2-6.6) | **120.6 s** | -- |
| 2 | 38 | 49.9 s | 32.8-35.7 s | ~76 (72.3-73.8) | **138.6-140.2** | -- |
| 3 | 38 | 132.8 s | 101.2-108.0 s | 7.7 (5.6) | **155.3 s** | majflt 136-148k -> 185-199k |
| 4 | 38 | 74.0 s | 43.2 s | 7.1-64.4 (5.2-47.7) | **149.9 s** | -- |
| 5 | 38 | 124.0 s | 102.2 s | 5.5-7.3 (2.6-5.1) | **143.6 s** | **23.0-28.3 GiB** target, 24.0-32.5 by draft end |

What that buys, in order of confidence:

1. **The total is a bad metric and my earlier "page cache is worth 4%" understates the picture.** Cold start spans
   **115.7-155.3 s (34%)** over the first five instances, and **115.7-186.1 s** counting the E1 runs below, made of
   two large components that move in *opposite* directions: target `experts` (32.8-121.5 s) and the draft pass
   (5.5-76 s). Cache state at launch biases one down and the other up, which is why the sums looked stable across a
   40 s spread. Every number in this section is per rank; ranks agree with each other to ~5% inside an instance
   except the draft pass (inst4: one rank 5.2 s, another 47.7 s -- they race).
2. **The target pass re-reads the checkpoint from the block device: 23-28 GiB a rank** (`/proc/self/io`
   `read_bytes` at target end; 0.0 GiB at target start) even though `Cached` sits at 38-46 GiB. The reason is our
   own footprint: four ranks take ~50 GiB of anon, of which **9.7 GiB a rank is the pinned int6 PLE table**
   (`hipHostRegister`), and the streaming window is only ~1/4 of the 110.6 GiB. The kernel evicts pages the other
   ranks still need, so ~1/4 of what each rank streams comes off disk again. `Cached` also never exceeds 46 GiB
   against a 110.6 GiB checkpoint -- there is no regime here where the file fits.
3. **The checkpoint bytes are not read with `read()`.** `rchar` is 0.5-0.6 GiB total while 110.6 GiB is streamed,
   so safetensors traffic is mmap page-cache work (`minflt` ~1M a rank across the load). Page-cache misses do not
   show as `majflt` (127-153k a rank = ~0.5 GiB, i.e. anon swap-in), which is why the earlier counters could not
   see the problem; only `read_bytes` does. That is the counter to keep on this path.
4. **Reclaim stall is not where the seconds go**: PSI `memory` grows by only ~2.7 s across the whole target pass
   (avg10 12-15% at the peak but total stall small). Yet every load writes ~5 GiB of swap (`SwapFree` 7 -> 2 GiB),
   which is the same precondition that made the amdgpu SVM deadlock reachable (`notes/load-oom-svm-wedge.md`).
5. **The draft pass is a duplicate read of the whole checkpoint index**: `mtp_weights` iterates 2182 tensors /
   110.6 GiB per rank to consume a few GiB of bf16 MTP MLP weights that it then requantizes to MXFP4. When those
   bytes are still resident it costs 5-8 s; when they have been evicted it costs 60-76 s. That is the single most
   avoidable term on this path.

Ranked experiments, cheapest first (E1 run, negative; E2/E3 not run):

* **E1 -- RUN 2026-11-01, NEGATIVE** (see "E1 measured"): pre-reading the draft's 4.86 GiB of bf16 MTP spans
  (34 tensors, 3 contiguous ranges) never fires, because they are still resident when the target pass ends. The
  duplicate read happens during consumption, which readahead cannot prevent.
* **E2 (now top) split the experts `place` time** into plugin repack (`_prep_mxfp4` / fold), stock `weight_loader`
  copy, and H2D. The term is 32.8-121.5 s for the same 59.8 GiB a rank, and disk bandwidth, PCIe and reclaim stall
  are each ruled out above, so the ~0.5 GB/s is most likely the format transformation itself. If so the fix is a
  faster repack path (or doing the fold once per checkpoint rather than once per rank), not an I/O fix.
* **E3 lower the load-time footprint.** ~40 GiB of pinned PLE table lands before the first byte is read, the pass
  then runs at `MemAvailable` ~= `Cached`, and 5.4 GiB of swap is written in an 11 s burst. Cutting that is a
  stability win first (it is the incident's precondition) and should also cut the 22-29 GiB of re-reads. Harder
  than it looks: the PLE table is allocated at model construction and the weight loader writes into it during the
  stream, so deferring it means buffering 38.8 GiB of PLE bytes instead.

### E1 measured: prestage does not fire (2026-11-01 04:32-04:48)

| run | R9K_PRESTAGE_MTP | load | `place` | `experts` | draft wall | disk reads charged to the prestage window |
|---|---|---|---|---|---|---|
| ctrl6 | 0 (off) | 149.8 s | 124.7 | 92.4 | 12.8-13.6 s | -- |
| on7 | 1 (on) | 186.1 s | 155.9 | 112.8 | 9.4-17.5 s | **+0.00 GiB, all four ranks** |
| inst8 | 1 (on) | 161.9 s | 137.6 | 121.5 | 8.7-9.3 s | +0.00 GiB |

The hypothesis is dead on its own instrument: the fadvise queues fine (`prestage mtp: 4.9 GiB in 3 ranges ... 0.01 s
to queue`) but issues **zero** I/O, because at the end of the target pass the MTP span is still resident (`Cached`
46 GiB). The duplicate read nevertheless shows up per rank in the *draft* window (+1.7 to +5.7 GiB), i.e. the pages
are evicted **while** the draft pass consumes them -- and pre-reading cannot prevent eviction. The control's
per-rank pairing also dulls the earlier story: pid 15192 re-read +5.70 GiB (the whole span) and still finished in
12.9 s, the same as the ranks that read +1.10 GiB. So the 60-76 s draft instances (inst2, inst4) are a rarer, still
uncaught regime, and there is no measurable total win here (the on7/inst8 totals are the target pass moving, not
the draft). E1 stays in the tree as a 0.01 s probe that fires only in the case it was written for.

### Footprint time series during a load (`bench/loadsample.sh`, inst8)

* **04:44:40 -> 04:44:45: worker RSS 6.3 -> 45.9 GiB before the checkpoint stream starts** (target stream opens at
  04:44:48). ~40 GiB of it is the pinned int6 PLE table, 9.7 GiB a rank, allocated by `ple/int6.py` at model
  construction -- i.e. the footprint is already there when the first byte is read.
* The whole target pass then runs with `MemAvailable` pinned at **40.8 GiB** and `Cached` at 37-41 GiB: available
  RAM ~= page cache, knife-edge. Worker RSS oscillates 48-68 GiB (`rss_max` up to 19.8 GiB against a 12.5 GiB
  steady state), so ~18 GiB of transient staging buffers ride on top.
* Swap is not a creep but a burst: 0.1 GiB until 04:45:46, then **0.7 -> 4.2 -> 5.4 GiB inside 11 s**
  (04:45:48-04:45:59), in the middle of the experts phase. This is the same regime that made the amdgpu SVM
  deadlock reachable.
* `Mlocked` reports 25.9 MiB: `hipHostRegister` memory is not counted there, so locked RAM is invisible to that
  counter. Use worker RSS plus what the plugin knows it allocated.

What that rules out for the experts term (92-121 s to place 59.8 GiB a rank = 0.5-0.65 GB/s a rank):

* **not disk bandwidth** -- 22-29 GiB of re-reads over ~130 s is 0.2 GB/s a rank, 0.8 GB/s aggregate, while the raw
  checkpoint streams at 5.8 GB/s (110.6 GiB in 19 s, measured earlier in this line of work);
* **not PCIe** -- each rank moves 19.65 GiB to its GPU, ~4-10 s of H2D for all four;
* **not reclaim stall** -- PSI memory grows ~2.7 s across the entire pass.

Which leaves the transformation itself: 59.8 GiB of expert checkpoint bytes a rank come in, the per-rank model is
19.65 GiB, so the loader is rewriting ~3x of data on the way in at ~0.5 GB/s. The measurement that would settle it
(not yet run) is splitting the experts group's `place` time into plugin repack (`_prep_mxfp4` / fold), stock
`weight_loader` copy, and the H2D submit.

## The load's stalls are swap I/O, not the checkpoint (inst11-20, 2026-11-01)

Nine launches with the copy probe, the stack sampler, the watchdog and the extended footprint sampler. Each step
killed a hypothesis:

1. **`Tensor.copy_` accounting.** Per rank the target pass makes **148,998 copy_ calls that write only 27.3 GiB but
   spend 136-163 s of wall**. Split by call size: `n<1ms=148,122 (22-30 s)` and **`>1s = 12-25 calls = 83-121 s`**.
   So the load's cost is ~15 pathological copies, not the 10^5 small ones (those average 0.18 ms).
2. **Which copies.** 0.2 MiB copies of `model.language_model.layers.N.mlp.experts.{gate_up,down}_proj_packed`
   (uint8 = MXFP4 packed) inside stock `_load_w13`/`_load_w2`; slowest 59 s; **all four ranks stall in the same
   window** (+58.2 / +58.5 / +63.6 / +88.3 s), and it is a different layer each time.
3. **Not PCIe.** `bench/h2d_probe.py`: a 0.2 MiB pageable copy is **0.049 ms** (3.9 GiB/s), 16-64 MiB copies run at
   **27-47 GiB/s**, pinned is only 1.3-1.6x better. All 70 GiB of aggregate H2D is a few seconds of PCIe.
4. **Not the pinned tables by themselves.** `bench/svm_probe.py`: 1, then 4 processes each `hipHostRegister`-ing
   **9.69 GiB** and hammering 0.2 MiB copies: p50 0.022 ms, p99 0.032-0.055 ms, max 3.6 ms. No stalls at all.
5. **Not page-ins by the copying thread either.** getrusage around each slow call: the 54.9 s and 50.4 s calls show
   **minflt+4..9, majflt+2..26** -- essentially no faults of their own.
6. **What it really waits on** (watchdog thread reading `/proc/self/task/<pid>/{wchan,syscall,stat}` every 0.5 s):
   `state=D wchan=lock_mm_and_find_vma syscall=-1`, `state=D wchan=folio_wait_bit_common`, plus
   `state=S wchan=futex_do_wait syscall=202`. A page fault waiting on the mm's mmap_lock, and a wait on a page whose
   PG_locked is set -- i.e. a read in flight somewhere else in the process.
7. **Who owns the device then** (2 s sampler with PSI-io, diskstats deltas and per-worker `read_bytes`), in the
   window that contains the 59 s stall:

   | gpu busy | device read | device write | worker `read_bytes` | swap in use | PSI io / mem |
   |---|---|---|---|---|---|
   | **3 %** | 1.2-1.5 GB/s | 1.2-1.5 GB/s | **0 MiB/s** | 5611 -> 5975 MiB | 5-7 % / 2-3 % |

   The device is saturated by the kernel's own swap churn, and the ranks are *not* the ones reading. The pins are
   not the blocked window either: `hipHostRegister` of 9.69 GiB takes 4.66-4.71 s and finishes before the stalls
   begin (hostmem now logs those windows).

Mechanism: the load over-commits RAM -- **4 x 9.69 GiB of registered (hence unswappable) host memory allocated
before the first byte is read**, plus ~15 GiB of anon and ~46 GiB of page cache on a 93.8 GiB box with only
8.39 GiB of swap. The kernel therefore writes ~5.9 GiB of swap at ~1.4 GB/s on the same NVMe that holds the
mmap'd checkpoint; every `copy_` that touches a non-resident page queues behind that, GPU idle. Whether and when
the kernel decides to do this is what makes a load 115.7 s or 186.1 s.

Fix candidates, cheapest first (C1 and C2 were run, see the next section; C3 was not needed once C2 landed):

* **C1 (system, one line, needs root) -- stop the kernel from swapping during the load**: `vm.swappiness=1`, or
  `swapoff -a` for the harsher A/B. Prediction: the `>1s` bucket collapses from 83-121 s to a few seconds and the
  total falls toward 70-100 s, because the cost moves to checkpoint re-reads, which are sequential and measured at
  5.8 GB/s. Falsified if the `>1s` bucket stays large while swap stays unwritten.
* **C2 (plugin, medium) -- shrink the unswappable footprint**: allocate the PLE table as ordinary anon, fill it
  during the stream, and `hipHostRegister` it at stream end (4.7 s a rank). Gives reclaim ~39 GiB of options back.
  Needs the UVA view created after registration, so `ple/int6.py` has to be restructured.
* **C3 (plugin, small) -- cut the transient staging**: RSS peaks at 68 GiB against 50 GiB steady; the 64 x 286 MiB
  `cpu->cpu dststrided` copies and vLLM's own "a strided CPU-to-CUDA copy can allocate a full-shard CUDA
  temporary" path are the candidates.

Two tool bugs found on the way, both worth remembering: `sys._current_frames()` is keyed by the CPython thread id
(which for the main thread is the pthread handle, e.g. 127864336392320), **not** the kernel TID, while
`/proc/self/task/<tid>/` wants the PID -- using one id for both fails silently one way and with ENOENT the other.
And a sampler that ran `pgrep -f 'VLLM::Worker'` put that literal in a child's argv, where `serve/native-flashnext.sh`'s
own guard (`pgrep -cf 'VLLM[::]'`) matched it and refused a launch that had just passed the gate.

## How to run it again

```
# kernels
cd ~/r9700-stack/kernels && HIPCFLAGS=-I$HOME/opt/rocm-dev/opt/rocm/core-10.1/include ./build.sh \
  && cp libr9k.so ../r9700_vllm/kernels/
# serve (TP4, everything in VRAM)
cd ~/r9700-stack && GPUS=0,1,2,3 TP=4 OFFLOAD_GB=0 NSEQ=16 MAXLEN=32768 UTIL=0.97 \
  bash serve/native-flashnext.sh           # log: ~/logs/serve-tp4.out  (NBT defaults to 8192)
# check
~/venv-r9700/bin/python3 bench/warmup.py && ~/venv-r9700/bin/python3 bench/bench.py run
~/venv-r9700/bin/python3 bench/quality.py 100 8
```

Not done here: the 2-card offloaded-experts recipe (needs ~120 GiB of host RAM, this box has 93), the
27B and GLM configs, `--kv-cache-dtype fp8`, the P2P communicator A/B, and a power-capped run for a
like-for-like decode comparison against the 210 W gate numbers.

## C1 was inert, C2 (late pinning) fixed it (inst sw1-21, late22-24)

First a correction to the table above: the sampler's device counter summed `/proc/diskstats` whole-disk lines *and*
their partition lines, which report the same sectors twice, so "1.2-1.5 GB/s read + 1.2-1.5 GB/s write" was about
**2x too high** (real: ~0.6-0.75 GB/s each way). The qualitative finding is unchanged -- the device was doing swap
writes concurrently with reads while the ranks' own `read_bytes` stood still and the GPUs sat at 3 %.

* **C1 `vm.swappiness=1` (applied by the operator) did nothing, and that is itself the result.** With the knob at 1
  the kernel still wrote the same ~5.9 GiB of swap (peak 5982 MiB) and the `>1s` bucket was unchanged (sw1-21: 14-19
  calls, 117-127 s; place 169.9 s, total 189.6 s). A swappiness knob steers the *balance* between anon and page-cache
  reclaim, but 38.8 GiB of the anon is **registered**, so it is not in the anon reclaim set at all: the kernel's only
  levers were the ~15 GiB of ordinary anon and the checkpoint cache. The fix has to change what is *unswappable*.
* **C2 late pinning works.** `R9K_PLE_PIN_LATE=1` allocates the int6 PLE table as ordinary host memory, fills it
  during the stream (`load_int6_shard` writes through `_r9k_host.narrow`, which does not care about pinning) and
  calls `hipHostRegister` + rebuilds the UVA view at the end of the target stream -- still before the first forward,
  the dummy run and graph capture.

  | instance | pin | total load | place | experts | `>1s` copies | hipHostRegister |
  |---|---|---|---|---|---|---|
  | sw1-21 | early | 189.6 s | 169.9 s | 138.3 s | 14-19 calls, 117-127 s | 4.7 s, before the stream |
  | late22 | early | 174.8 s | 151.6 s | 113.1 s | 16-22 calls, 103-110 s | 5.0 s |
  | **late23** | **late** | 118.6 s | **48.6 s** | **37.3 s** | **0 calls, 0.0 s** | 7.8-8.2 s, after the stream |
  | **late24** | **late** | **72.5 s** | 47.1 s | 37.4 s | **0 calls, 0.0 s** | 6.4-6.6 s, after the stream |

  Registration is ~1.4-1.7x more expensive late (it has to pull the table's pages back off swap: 39 GiB in ~8 s =
  ~5 GB/s) and swap peaks at **all 8 GiB** during the load instead of 5.9 GiB -- the tables are now the pressure
  valve -- but `MemAvailable` never fell below 36.2 GiB, kern.log shows no new blocked tasks, nothing was OOM-killed,
  and the four-rank simultaneous 5-60 s `copy_` stalls disappeared completely. Steady state afterwards is unchanged:
  swap back to ~4 GiB free, KV cache 496,374 tokens (which is the recorded value; late23's 487,879 was instance
  drift), GSM8K 98/100, needles 3/3, `@32 x300` 1574 tok/s, `@32 x1200` 1804, `@16 x300` 1134, 300 W median at
  busy==100.
* **With the stalls gone, the load's shape is finally legible.** late24: place 47.1 s = experts 37.4 s + other 8.6 s,
  and the copy histogram is now `n<1ms=147,366 (29.7 s) 1-10ms=1,513 (3.7 s) 10-100ms=82 (2.5 s) 0.1-1s=37 (7.4 s)
  >1s=0`. So essentially *all* remaining place time is the 147k small copies doing real work (27.3 GiB), the draft
  pass is 3.3-6.1 s, and there is no I/O anomaly left in the target pass.
* **The draft pass's "44 s requant" was not the requant.** Splitting it (to-GPU / search / cat) gives search
  **1.5-1.6 s over 5 calls** and to-GPU 0.7 s in late24, while a standalone `quantize_mxfp4_search` microbench
  (`bench/mtpquant_probe.py`, run with the GPUs free) settles at **4.3 GiB-of-bf16/s** plus a fixed **0.39 s** of
  launch overhead. late23's 44.1 s was almost entirely `x.cuda()` paging in the draft's bf16 bytes from a device
  that was busy swapping. So the draft pass's 3.5-48.8 s swing is a page-cache question, not a quantization one --
  and it is the last big swing left in the load.
* **The cold-cache A/B (operator ran `sync; echo 3 > /proc/sys/vm/drop_caches` before each launch, `Cached` 0.06 GiB
  both times) gives the cleanest comparison, and it points the same way:**

  | cold start | total | place | experts | `>1s` copies | swap peak | device read peak | checkpoint read by the 4 workers |
  |---|---|---|---|---|---|---|---|
  | late pin (cold-late25) | **68.7 s** | 47.3 s | 37.7 s | **0** | 8191 MiB | 6.8 GB/s | 118.7 GiB (1.07x the file) |
  | early pin (cold-early26) | 114.5 s | 96.6 s | 55.7 s | 1-5 calls, 50.8-52.5 s each | 6286 MiB | 6.9 GB/s | 115.7 GiB |

  Two things worth keeping: the duplicate reads are **the same in both** (23.6-26.7 GiB per rank at `target end`), so
  late pinning does not reduce I/O -- it removes the *waiting*; and the shared page cache already keeps read
  amplification at ~1.07x of the checkpoint size for four ranks, which is why no one reads this file four times.
  Cold+early (114.5 s) is also much better than warm+early (174.8-189.6 s): with an empty cache the kernel has
  somewhere to go before it starts swapping, so the pathological window needs a full cache to appear.
* **Late pinning is now the default** (`R9K_PLE_PIN_LATE=0` restores the old behaviour). Five instances with it:
  68.7 (cold) / 72.3 / 72.5 / 118.6 s; three without: 114.5 (cold) / 174.8 / 189.6 s. Throughput and quality are
  unchanged in every case (`@32 x300` 1543-1608, `@32 x1200` 1800-1815, `@16 x300` 1107-1206 tok/s, GSM8K 98-99/100,
  needles 3/3, KV 496,374 tokens, ~299 W at busy==100).
* Remaining: (i) **swap headroom** -- late pinning deliberately uses all 8 GiB of swap during the load (peak 8191 MiB
  of 8191) while `MemAvailable` stays at ~36-37 GiB; more swap would take the last sharp edge off (needs root, and the
  earlier incident was a reclaim/GPU-fence deadlock, so a bigger buffer is worth having); (ii) re-test **E1 prestage
  with late pinning** (it failed early-pinned because the MTP span was evicted during consumption); (iii) the last big
  swing is the draft pass (3.4-48.8 s), and it is data movement into the draft, not the requant; (iv) the experts term
  (35.5-37.7 s) is now plain work: 147k copies of 27.3 GiB, ~30 s of copy wall time, with nothing pathological left.

## How to reproduce the load-path analysis

```bash
bash bench/gate.sh                                   # verified stop + memory-return gate
cd ~/r9700-stack && setsid bash bench/loadsample.sh &    # 2 s series: RAM, cache, swap, gpu busy, PSI io,
                                                         #   device rd/wr, per-worker read_bytes
R9K_LOAD_COPIES=1 R9K_LOAD_STACKS=1 setsid bash bench/relaunch.sh instN   # gate, launch, then print fingerprints
                                                         #   + load anatomy (copy classes, slow calls, snaps)
~/venv-r9700/bin/python3 bench/h2d_probe.py 0            # pageable vs pinned H2D, per size
PIN_GIB=9.7 COPIES=4000 DEV=0 ~/venv-r9700/bin/python3 bench/svm_probe.py # does a 9.7 GiB pin stall small copies?
```

Switches: `R9K_LOAD_PROF=1` (group timers + phase snapshots), `R9K_LOAD_STACKS=1` (SIGPROF stack sampler: which
function), `R9K_LOAD_COPIES=1` (`copy_` accounting, fault deltas, slow-call names, watchdog wchan),
`R9K_PRESTAGE_MTP=0|1` (the E1 control), `R9K_PLE_PIN_LATE=0|1` (default 1: register the PLE tables after the stream).
Read the pair: the log's `slow call` offsets (`+N s into the load`, with `probe at HH:MM:SS` in the header line) line up
with the sampler CSV's clock columns.

For a **cold-cache** launch (what a first boot sees), the operator has to drop the cache after the gate and before the
launch -- `sync; echo 3 > /proc/sys/vm/drop_caches`, check that `Cached` in `/proc/meminfo` fell to ~0.06 GiB. It needs
root and it is safe only with no vLLM processes running, i.e. after `bench/gate.sh` has passed.
