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
