# Nemotron-3.5-Super-VL-120B-A12B on four R9700s (2026-11-18)

Checkpoint: `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` (hf-mirror), at `~/models/Nemotron-Super-VL-120B-NVFP4`:
88 safetensors shards, 169,505 tensors, 70 GB on disk, all index tensors present (checked shard by shard against
`model.safetensors.index.json`, and every shard's declared byte length against its file size).

Arch `NemotronH_Omni_Reasoning_V3` — vLLM 0.32 registers it as `nano_nemotron_vl` (`NemotronH_Nano_VL_V2`, which maps
`language_model.backbone -> language_model.model` and reuses `nemotron_h.py` plus `nemotron_h_mtp.py`). Quant is
`modelopt_mixed` (`MIXED_PRECISION`, fp8 KV). Layer mix 40 Mamba2 + 40 MoE + 8 attention, hidden 4096, vocab 131072,
**512 routed experts, top-22, MoE latent 1024**: `up_proj` [2688,1024] and `down_proj` [1024,2688] per expert, i.e. an
*ungated* MLP — one matrix per side, not the gate|up halves every other MoE here has. `hf_quant_config.json` lists
41,156 quantized entries: 41,042 routed-expert NVFP4 (`up_proj`/`down_proj`), 113 FP8, 83 dense NVFP4 (mostly
`mixer.shared_experts.{up,down}_proj`).

## Launch that works

```bash
MODEL=$HOME/models/Nemotron-Super-VL-120B-NVFP4 SERVED_NAME=SuperVL MTP=4 \
NSEQ=16 CGSIZES=1,2,4,8,16,24,32,40,48,64,80,96,128,160,192,256 \
TP=4 MAXLEN=32768 EAGER=0 REASONING=nemotron_v3 TOOLCALL=pythonic WEBUI=same \
EXTRA="--trust-remote-code --mm-processor-kwargs {}" \
bash serve/native-flashnext.sh
```

The `CGSIZES` list has to cover `NSEQ x (MTP+1)` -- 80 rows at NSEQ=16/MTP=4 -- or the decode step runs padded; the
launcher's default list carries 40 now but not 80/160, because those sizes cost graph pool (a few hundred MB of KV)
for launches that never reach them. `NBT` stays at its default 8192: 16384 measured worse (see the concurrency
section).

`R9K_MODELOPT_DENSE=native|mxfp4|stock` picks how the dense NVFP4 linears are served (default `native`, see the
levers section); the MoE side has `R9K_NVFP4_MOE=mxfp4|native` and `R9K_DISABLE=nvfp4` puts the whole model back on
stock emulation.

## What the plugin adds for this model

| piece | where | why |
|---|---|---|
| `modelopt_mixed` registration | `r9700_vllm/__init__.py`, `quant/modelopt.py` | stock registers nothing for it on ROCm; the checkpoint's routed experts are NVFP4, and ROCm has no NVFP4 MoE backend, so stock serves them through the dequantizing emulation |
| NVFP4 -> MXFP4 at load | `quant/modelopt.py R9kModelOptNvfp4MoEMethod` + the existing `quant/nvfp4.py nvfp4_to_mxfp4` | reuses the libr9k grouped MXFP4 x FP8 GEMMs; all 512x40 expert matrices convert in the load path, no offline repack, no extra disk |
| `uses_weight_scale_2_pattern() -> True` | same class | `RoutedExperts` picks the scale *name* from that predicate; the base CT method says False, and the FP4 loader then tries to stuff a 172,032-byte tensor into a scalar: `RuntimeError: shape '[]' is invalid for input of size 172032` |
| scale-sign bridge | `_row_div` in the same class | ModelOpt dequantizes `w = e2m1 * s16 * scale_2`; the CT helper divides (`e2m1 * s16 / row_div`), so the divisor must be `1/scale_2`. Passing `scale_2` instead gives a 10^11 % error — measured in `/tmp/probe_mopt.py` before any of this ran |
| ungated-MLP activation kernel | `kernels/r9k_moe_mxfp4a8.hip r9k_relu2_quant_fp8` | libr9k's epilogue was `silu(gate) * up` over a w13 whose rows are the two halves (`g = row*2I`, `u = g + I`). Nemotron-H is `activation_without_mul` + `mlp_hidden_act=relu2` (vLLM `MoEActivation.RELU2_NO_MUL`): relu^2 over one half, then the same per-row e4m3 quant |
| capability flags | `moe/experts.py` | `_supports_no_act_and_mul() -> True`; `_supports_activation` admits `RELU2_NO_MUL` **only if the loaded libr9k.so carries the symbol** (`K.relu2_quant_supported()`), so an older .so falls back to stock instead of failing at the first decode; `moe_problem_size` reports `N1`, not `N1//2`, when the MLP is ungated |

Build: `cd kernels && HIPCFLAGS=-I$HOME/opt/rocm-dev/opt/rocm/core-10.1/include OUT=libr9k-relu2.so ./build.sh`, then
`cp libr9k-relu2.so ../r9700_vllm/kernels/libr9k.so` — `serve/native-flashnext.sh` hard-exports
`R9K_LIB=$REPO/r9700_vllm/kernels/libr9k.so`, so a .so left in `kernels/` is never loaded. The previous build is kept
as `r9700_vllm/kernels/libr9k.so.pre-relu2-20261118`; `R9K_DISABLE=nvfp4` puts the model back on stock emulation.

Kernel check (`/tmp/probe_relu2.py`): the fp8 bytes are **identical to vLLM's own reference** (relu in place, then
square) at M = 1..128 and I = 255/1024/672/2688, the per-row scales match exactly, and the dequantized values sit
2.4-2.7% off exact relu^2 — which is e4m3 rounding, not the kernel.

## Numbers (single stream, temp 0, thinking on, measured through the gateway as `usage.completion_tokens` / wall)

| step | 60-token cap | 300-token cap |
|---|---|---|
| first light: eager, stock NVFP4 MoE, MTP=2 | 26.9-27.5 | 24.5-26.0 |
| eager + libr9k MXFP4 experts, MTP=2 | 28.3-29.9 | 27.4-27.8 |
| **graphs** (`EAGER=0`), stock MoE, MTP=2 | 73.0-73.4 | 71.2-72.8 |
| graphs + libr9k, MTP=2 | 86.2-89.6 | 81.4-84.0 |
| graphs + libr9k, MTP=3 | 98.0-104.6 | 83.6-94.7 |
| **graphs + libr9k, MTP=4** (shipped) | 111.8-118.7 | 94.2-102.3 |
| graphs + libr9k, MTP=6 | 108.3-123.7 | 94.7-98.6 |
| graphs + libr9k, MTP=10 | 111.6-129.8 | 85.5-89.3 |

Two things worth their price here, in order:

1. **`--enforce-eager` was costing 3x.** 73 tok/s compiled vs 27 eager, on the *same* weights and the same MTP. This
   model is 88 backbone layers of small GEMMs; at TP4 with one stream, eager spends its life in launch overhead, not
   in the MoE. Compiling costs 24.5 s at boot, 111 s of engine warm-up, and 2 GiB of KV (10.39 -> 8.42 GiB a card).
2. **The libr9k experts are worth +18-22% short / +12-17% long on top of that** (73 -> 86 at MTP=2). Less dramatic
   than the flashnext case because the expert problem is small: 22 of 512 experts, 2688x1024 per rank.

MTP acceptance (vLLM's own counter): 2.62 at MTP=2, 3.02 at MTP=3, 4.73 at MTP=10 — of a maximum of MTP+1. The knee
for a single stream is MTP=4-6; past that the drafts stop paying for themselves. `MTP=4` was picked because it is at
the top of the long-generation column and no worse in concurrency.

Concurrency, 8 streams x 300 tokens: **458 tok/s aggregate (57 each)** at MTP=4, 441 (55 each) at MTP=6.

Memory: 17.72 GiB a card for the converted weights (18.52 GiB stock), KV 8.42 GiB a card under graphs = 1,450,583
tokens of pool at `MAXLEN=32768`. Nothing is host-resident: the conversion happens after the load and stays in VRAM,
so the LRU/host expert cache and its pinned-memory budget are not in play for this checkpoint.

## Against the DGX Spark (GB10) reference numbers

Community Spark numbers for this checkpoint: 38-50 tok/s single user with LiLiCorr spec decode (~7 drafts, higher
with thinking off), **19-20 tok/s without speculative decoding**, 8 users ~100 tok/s aggregate, 16 users ~146.

Four R9700s land at 94-102 tok/s a stream and 458 tok/s at 8 streams — ~2.3x single-user and ~4.5x at 8 users. Not
apples to apples in three ways worth naming: Spark runs native NVFP4 on Blackwell while this box requantizes the
experts to MXFP4 (see the caveat below); the Spark recipes often quote thinking-off; and their 128 GB unified memory
holds a 1M-token context that this box's four pools do not.

## Still stock, and the caveats

- **What is still stock now**: the Mamba2 state path (conv1d + the selective-scan chunk/recurrence kernels, 40 layers,
  stock Triton under torch.compile) and the vision tower (bf16, only touched when a request carries pictures). The
  backbone's projections are all on kernels now: FP8 scaled-mm for the 113 FP8 entries, libr9k native NVFP4 for the
  83 dense NVFP4 entries, libr9k grouped MXFP4 x FP8 for the 41,042 expert matrices.
- The **image path is untested.** The processor is enabled (`--mm-processor-kwargs {}`; anything Qwen-shaped is
  rejected: `NanoNemotronVLProcessor.__init__() got an unexpected keyword argument 'size'`), patch 16,
  1024-13312 patches.
- The NVFP4 -> MXFP4 conversion is a **requantization**, not a format change: block-16 e4m3 scales become block-32
  e8m0. Against an exact NVFP4 dequantization the weights are ~11% off in relative L2 (`up_proj` 11.2%,
  `down_proj` 11.0%). What that costs at the answer level is in the A/B below: nothing measurable at n = 200, which is
  a bound and not a proof. The flashnext MXFP4 checkpoint — MXFP4 at rest — does not pay it even in principle, and
  the dense NVFP4 linears do not pay it either (the native kernel keeps their bits).
- **Images are untested.** The processor is enabled (`--mm-processor-kwargs {}`; anything Qwen-shaped is rejected:
  `NanoNemotronVLProcessor.__init__() got an unexpected keyword argument 'size'`), patch 16, 1024-13312 patches.
- `MAXLEN=32768` is deliberate caution, not a limit: the pool holds 1.45M tokens.
- The watcher is **not** armed (its pid was killed before this work so an experimental load could not be treated as a
  production crash fault), and the box's stored launch config (`262k.json`) still describes the Qwen stack.

## A/B: what the conversion buys, and what it costs in answers

Two arms, identical launches (graphs, `MTP=4`, `MAXLEN=32768`, `NSEQ=8`), the only difference `R9K_DISABLE=nvfp4`.
Speed waves use `enable_thinking: false` so they measure the decode path rather than the reasoning length; GSM8K runs
with thinking on, as the script ships. The wave wrapper retries a transient gateway failure instead of losing the wave
— the first pass of the arm lost exactly one wave to one 502, and an empty wave reads as a regression.

| measure | stock NVFP4 emulation | libr9k MXFP4 conversion | delta |
|---|---|---|---|
| single stream, 300 tokens | 78.9 | 102.3 | +30% |
| 8 streams x 300 tokens, aggregate | 262-272 (35-38 a stream) | 442-462 (57-62) | **+68-72%** |
| 8 streams x 1200 tokens, aggregate | 289-299 (37-38) | 495-505 (64-65) | **+66-71%** |
| prefill, 30k prompt, cold | 3,762 tok/s | 6,390 tok/s | +70% |
| decode inside a 30k context | 16.2 | 17.8-18.2 | +10% |
| needle at 30,073 tokens, 3 depths | hit | hit | — |
| GSM8K 200 questions, conc 8 | 190/200 = **95.0%** (9 truncated), 7.1 min | 193/200 = **96.5%** (6 truncated), 3.6 min | 2x faster, accuracy indistinguishable |

The concurrency column is where the win lives: at one token a step the expert weights are a latency problem, at eight
they are a bandwidth problem, and the emulation reads the same bytes twice over. Single-stream decode gains 30%, a
full wave gains 70%.

On quality: 1.5 points on 200 questions is three questions, well inside the +-3 point sampling width, so the honest
sentence is **no measurable cost at this sample size**, not "equal". The full 1,319-question set (n=1319, +-1.4
points) is what would bound it properly; it costs ~17 min a arm at this speed rather than ~4.

## Next levers, measured rather than assumed (2026-11-18, later the same day)

| idea | what the measurement said | verdict |
|---|---|---|
| retune the MoE tile configs for this checkpoint's shapes | `tuning/decode_moe_sweep.py --w13 672,1024 --w2 1024,672 --E 512 --topk 22`: at the 8-stream width (40 tokens -> 880 rows) gate_up 174.5 us against the best of every legal config at 170.1, down 173.1 against 171.8 | **1-2.5%, not worth a config change.** The Flash-Next defaults carry this checkpoint nearly as well as it can be carried |
| the Mamba2 projections are bf16 and dominate the weight stream | **wrong assumption, and the checkpoint says so**: the `hf_quant_config.json` FP8 list *is* the backbone -- 80 entries are `mixer.in_proj` / `mixer.out_proj` (40 Mamba2 layers x 2) and ~32 are the attention `q/k/v/o`. They come up on `ROCmFP8ScaledMMLinearKernel` | nothing to do; written down so the assumption is not re-bought |
| TP4 all-reduce is plain RCCL | the launch's `Using ['PYNCCL'] all-reduce backends` line is vLLM listing the sub-backends *inside* `CudaCommunicator`; the plugin's communicator comes from `R9700Platform.get_device_communicator_cls` and installs the r9k N-rank one-shot for `ws > 2` by default (`comm/r4d_ar.py`) | already on libr9k ar4 for decode-sized messages |
| the 83 dense NVFP4 linears (`mixer.shared_experts.{up,down}_proj`) run on the dequantizing emulation | served them from libr9k's **native NVFP4** kernel instead (`quant/modelopt.py:_dense_native`, the ModelOpt twin of `nvfp4.py:make_native_dense_scheme_cls`): exact e2m1 codes and e4m3 per-16 scales, the per-tensor global scale becomes a per-row fp32 multiplier, activations fp8 per row | **kept** (`R9K_MODELOPT_DENSE=native`, `stock` and `mxfp4` also available): wave +5-12%, single stream unchanged, GSM8K unchanged |
| `cudagraph_capture_sizes` misses the decode shape | the decode batch is `NSEQ x (MTP+1)` rows = **40** at NSEQ=8 / MTP=4, and the shipped list had 32 then 48: every GEMM in the step ran 20% padded. 40 is now in the launcher's default list | **kept**: the same all-stock arm measured 262-272 -> 306-328 tok/s on an 8-stream wave (+17%) |

Speed after both, single stream / 8 x 300 / 8 x 1200: **93.9 / 487-559 / 516-580 tok/s** (the wave columns move +-10% run to run on this box, which is the same caveat PROGRESS 2026-10-24 records for the Qwen stack: *"need a repeated-run harness before trusting <15% differences"* -- the native-dense gain is inside that band, the CGSIZES fix is outside it). GSM8K-200 across the three arms: 190/200 stock, 193/200 plugin MoE, 191/200 plugin MoE + native dense.

Two things that cost more time than they should have:

1. **A plugin module that fails to import is a WARNING, not a failure.** A `NameError` from a missing `import os` in `quant/modelopt.py` came out as `r9700_vllm: modelopt_mixed not registered: NameError(...)`; `/health` answered 200, the engine served, and an entire measurement arm ran on stock kernels while I thought it was the plugin arm. The check that takes ten seconds: `python -c "import r9700_vllm; r9700_vllm.register(); from vllm.model_executor.layers.quantization import get_quantization_config as g; print(g('modelopt_mixed').__module__)"` -> must print `r9700_vllm.quant.modelopt` (a bare `import r9700_vllm` does not register anything: registration is `register()`, which vLLM calls as the plugin entry point).
2. **ModelOpt's FP4 block scales: measured, not assumed.** The dense hook reads `layer.weight_scale` either row major or as the CUTLASS 128x4 interleave (`nvfp4_utils.swizzle_blockscale`; `_unswizzle_scale` is its inverse). Rather than reading the loader, the probe runs the stock emulation (authoritative) and the libr9k path on the same rows for each candidate layout and keeps the one that agrees -- 0.1014 for row major at `1344x4096`, and the unswizzled candidate is not even constructible there (1344 is not a multiple of 128, so the swizzle buffer would be larger than the tensor). A wrong layout costs >1.0, so the reject threshold at 0.25 is not a coin flip. Note the direction of the precision change: native runs fp8 activations, the emulation it replaces runs bf16 activations under a W4A16 reading, and the checkpoint's own recipe is W4A4 -- so this is more precise than the recipe and slightly less than the emulation.

## Concurrency width, and a knob that went the wrong way (2026-11-18, same day)

The engine shipped at `NSEQ=8` because that is what the Qwen recipe uses; Super VL's wave is a different shape, so
`NSEQ=16` was tried, with the graph list extended to cover it (`NSEQ x (MTP+1)` = 80 rows, then 160 for 32 streams):

| launch | @1 | @8 x300 | @16 x300 | @16 x1200 | cold prefill 30k | TTFT 30k | KV pool |
|---|---|---|---|---|---|---|---|
| NSEQ=8, NBT=8192, list to 256 (shipped earlier) | 93.9 | 487-559 | — | — | 6,390/s | 4.71 s | 1,450,583 tok |
| NSEQ=16, **NBT=16384**, list to 256 +80 +160 | 112.3 | 477-510 | 742-752 (49 each) | 813 (53 each) | **5,310/s** | **5.66 s** | 825,753 tok |
| **NSEQ=16, NBT=8192**, list +80 +160 (this arm) | 97.0 | 505 | **712-752 (47-50 each)** | **813** | **6,787/s** | **4.43 s** | 1,102,970 tok |

* **`NBT=16384` is a regression, not a win**: cold prefill 6,787 -> 5,310 tok/s, TTFT 4.43 -> 5.66 s, and it ate
  276k tokens of KV pool (1.10M -> 0.83M) by raising the activation peak the sizing run sees. The recipe's 8192 stays.
  (The intuition that a bigger prefill chunk must make GEMMs more efficient is wrong here; it is already MoE-bound
  at 8192.)
* **`NSEQ=16` is a real capacity win and costs nothing at one stream**: 8 streams stay where they were, 16 streams
  move the aggregate to 712-752 tok/s at ~48 tok/s a stream (against Spark's 16-user ~146 tok/s), and a 30k-token
  context still decodes at 18 tok/s with 16 in flight. The cost is KV pool (1.45M -> 1.10M tokens) from the larger
  graph pool, which is still 33 requests deep at 32,768 tokens.
* **The single-stream column is now seen spanning 93.9-112.3 tok/s across launches that differ by nothing that
  should touch it.** That is a +-10% measurement floor, and it is the reason the repeated-run harness is on the owed
  list rather than in the footnote: without interleaved A/A runs, half of what this section says about the small
  knobs is luck.

## What is left in the step (sizing, 2026-11-18) -- the Mamba2 hypothesis died

The plan was to profile the decode step, find the Mamba2 conv/scan share, and decide whether a hand kernel was worth
days. Engine-level tracing is not available for this model on this box: `--profiler-config
{"profiler":"torch","torch_profiler_activities":["CUDA"]}` starts (`Profiler started.`) and then the workers stop
making progress -- five minutes later EngineCore dies with `RPC call to sample_tokens timed out` and the trace
directory stays empty. Same family as the old rocprofv3/`PROF=1 PROFACT=CUDA` complaints in `notes/decode-step.md`,
just harsher here (4 ranks + graphs).

So sizing goes the way the rest of `tuning/` does it: graph-timed standalone probes at the checkpoint's real shapes,
plus subtraction. New tool `tuning/decode_mamba_probe.py` (shapes read from a safetensors header rather than guessed:
`in_proj [18560,4096]` FP8 *with an `input_scale`, i.e. static W8A8*, `conv1d [10240,1,4]` = 8192 inner + 1024 B +
1024 C from 8 groups x 128 state, `out_proj [4096,8192]` FP8; per card at TP4 that is heads 32 x 64, groups 2, conv
2560). Microseconds a layer, 40 mamba layers in the model:

| decode batch | in_proj fp8 | conv1d update | selective_state_update | out_proj fp8 | a layer | x40 = ms/step |
|---|---:|---:|---:|---:|---:|---:|
| S=8, MTP=4 | 17.3 | 4.6 | 52.4 | 15.3 | 89.6 | 3.58 |
| **S=16, MTP=4 (shipped)** | 26.5 | 4.9 | **98.6** | 22.1 | 152.0 | **6.08** |
| S=32, MTP=4 | 39.6 | 5.8 | 183.3 | 27.8 | 256.5 | 10.26 |
| S=16, no MTP | 16.8 | 4.0 | 30.2 | 15.0 | 66.0 | 2.64 |

At 16 streams the step is ~70 ms (48 tok/s a stream divided by ~3.4 accepted tokens per step). So:

* all 40 mamba layers together: **~9% of the step**;
* the part a hand kernel could replace -- conv1d update + selective state update -- **4.1 ms, ~6%**;
* the two projections (1.9 ms) are already on library scaled-mm kernels.

A 2x scan kernel is therefore worth ~2-3% of the step, which is *inside* the +-10% launch-to-launch floor this box
measures. The conclusion holds even if the engine's real path costs twice what the probe sees (the probe takes the
plain Triton SSU path with an fp32 state cache, no ReplaySSM, so it is a plausible overestimate of bandwidth rather
than an underestimate). **Mamba2 scan is dropped as a lever.** That is the value of sizing before building: the last
remaining "big unknown" of the backbone turned out to be small, for the cost of one probe.

What the same subtraction says *is* big at 16 streams: the MoE GEMMs are 16.5 ms (40 layers x 412 us at 1760 rows,
from `tuning/decode_moe_sweep.py`) = 26% of the step, mamba 9%, and **~45 ms is still unaccounted** -- all-reduce at
~640 KB messages 2-3 times a layer, MoE glue (topk / align / quant), the drafter's 4 draft passes, 8 attention
layers, lm_head, sampler. `tuning/ar_push_bench.py` and `tuning/ar_profile.py` already exist, so all-reduce is the
next cheap thing to size.

One operator gotcha found on the way: a standalone script against this stack cannot import torch without the OpenMPI
libraries (`OSError: libmpi_cxx.so.40`); `serve/native-flashnext.sh` puts `$OPENMPI` on `LD_LIBRARY_PATH` and a bare
shell does not.

## How much of the step is communication (measured, 2026-11-18)

The engine already measures this at every launch (`comm/calibrate.py`). This box, bf16, us a call, `RCCL / one-shot /
two-shot`:

| message | RCCL | one-shot | two-shot |
|---|---:|---:|---:|
| 5 K | 23 | 7 | 10 |
| 20 K | 37 | 8 | 9 |
| 40 K | 37 | 13 | 10 |
| 80 K | 37 | 17 | 12 |
| 160 K | 41 | -- | 18 |
| 320 K | 47 | -- | 28 |
| 480 K | 52 | -- | 32 |

Chosen policy at boot: `one-shot <= 20 KiB, two-shot <= 480 KiB, RCCL above, compressed (ar4, 4-bit) from 512 KiB`.

Super VL's message size is `NSEQ x (MTP+1) x hidden 4096 x 2 B`: **40 KB at one stream, 320 KB at eight, 640 KB at
sixteen** (so 16 streams is the only arm that reaches the compressed path). The count a step is **88**: mamba
`out_proj` 40 + MoE 40 + attention `o_proj` 8 (`shared_experts` are built with `reduce_results=False`, so they fold
into the MoE's single reduce rather than adding one), plus up to ~8 from the drafter. Therefore:

| arm | us a call | x88 | step | AR share |
|---|---:|---:|---:|---:|
| 1 stream | 10 (two-shot) | 0.9 ms | ~32 ms | **2.7%** |
| 8 streams | 28 (two-shot) | 2.5 ms | ~68 ms | **3.6%** |
| 16 streams | 640 KB compressed -- unmeasured point; exact at that size extrapolates to ~55-60 us | <= 5.3 ms | ~70 ms | **<= 7.5%, upper bound** |

RCCL is 1.7-3x worse than what runs here at every size we use (47 vs 28 us at 320 KB, 23 vs 7 at 5 KB), so the big
communication win was already taken; what is left (quant threshold, pair topology, block counts) is worth 2-3% a knob.

**Two holes in the tooling found while doing this**, both cheap to fix and neither worth a restart budget today:
`R9K_AR_HIST=1` instruments only the TP=2 `R9kAllReduce` -- the TP4 N-rank/ar4 path we actually run has no
histogram, so it logs nothing (the sizes above are derived from shapes, not observed); and `R9K_ARN_CAL_SIZES_KB`
silently drops probes above the scratch pool (~480 KiB), so the compressed point at our exact 640 KB message is
unmeasured on this box.

**And the profiler is unusable here, not merely noisy.** Second attempt, deliberately tiny: CUDA activity only,
`record_shapes:false`, one 40-token request inside the window. The request returned 500 after 300 s, EngineCore died,
the trace directory stayed empty. So there will be no per-kernel ledger for this model; sizing has to come from
standalone graph-timed probes and subtraction (`tuning/decode_mamba_probe.py` is the pattern).

## The rule used to kill levers (so the next one is killed the same way)

For each candidate: compute the **ceiling** -- what a *perfect* fix removes from the step -- and compare it with this
box's measurement floor, +-10% of launch-to-launch throughput (~7 ms on a 70 ms step). A ceiling under ~5 ms does not
get engineering time, because it can never be shown. Applied today: all-reduce ceiling <= 5.3 ms and most of it already
captured (no); Mamba2 conv+scan 4.1 ms measured (no); MoE GEMMs are 16.5 ms = 26% of the step but near HBM bandwidth
for the experts touched and the legal tile configs are 1-2.5% off, so the realistic ceiling is 2-4 ms (no).
What is left with a possible >10% ceiling is the *unaccounted mass* -- MoE glue (topk / align / sort at top-22 of 512
experts), the drafter's 4 draft passes, attention, lm_head, sampler, and the launch-gap time between ~1,700 kernels a
step. The next measurement, not the next line of kernel code, is a standalone probe of the MoE glue at this
checkpoint's shapes.

## Gotchas that cost time

1. **`hf_quant_config.json` must be in the checkpoint dir.** The MTP draft path (`nemotron_h_mtp.py` ->
   `get_draft_quant_config`) reads that file, not `config.json`'s `quantization_config`, and without it the launch
   dies at `ValueError: Cannot find the config file for modelopt_mixed`. The HF repo carries it (5.4 MB); the
   download script's config-file list has to include it.
2. **Reasoning parser is `nemotron_v3`.** `REASONING=none` is not a parser name in 0.32, and the Qwen parser is the
   wrong grammar.
3. **`--mm-processor-kwargs {}`** — the recipe's Qwen `{"size": ...}` default crashes the processor constructor.
4. The GGUF route is a dead end on this venv: `QuantizationMethods` in 0.32 has no `gguf`, and the local llama.cpp
   build is CPU-only.
