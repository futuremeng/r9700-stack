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
TP=4 MAXLEN=32768 NSEQ=8 EAGER=0 REASONING=nemotron_v3 TOOLCALL=pythonic WEBUI=same \
EXTRA="--trust-remote-code --mm-processor-kwargs {}" \
bash serve/native-flashnext.sh
```

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

- The **83 dense NVFP4 linears** (`shared_experts.{up,down}_proj` and three others) are still on stock
  `EmulationNvFp4LinearKernel`. They are 0.2% of the quantized entries but they run every token; routing them through
  the dense MXFP4 path is the next lever, and the 262k Qwen config's `R9K_TARGET_LMHEAD=mxfp4` shows that path works.
- The NVFP4 -> MXFP4 conversion is a **requantization**, not a format change: block-16 e4m3 scales become block-32
  e8m0. Against an exact NVFP4 dequantization the weights are ~11% off in relative L2 (`up_proj` 11.2%,
  `down_proj` 11.0%). What that costs at the answer level is in the A/B below: nothing measurable at n = 200, which is
  a bound and not a proof. The flashnext MXFP4 checkpoint — MXFP4 at rest — does not pay it even in principle.
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
