<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/r9700-stack-lockup-dark.svg">
    <img src="docs/r9700-stack-lockup-light.svg" alt="r9700-stack" width="320">
  </picture>
</p>

Tuned GPU kernels and a vLLM plugin that make **Qwen3.8**, **GLM-5.3-Flash** and **DeepSeek-V4.1-Flash** run fast on **AMD Radeon AI PRO R9700**
cards (gfx1201 / RDNA4) on **stock vLLM and stock ROCm**. No fork, no patched source, no vendored binaries, no
re-quantised weights: everything loads at runtime through vLLM's own extension points, so vLLM and ROCm update
underneath it.

Current release: **v0.4.0** (2026-11-03, the vLLM 0.32 line), [release notes](notes/release-v0.4.0.md),
[CHANGELOG.md](CHANGELOG.md).

<p align="center">
  <a href="docs/r9700-stack-page.png"><img src="docs/r9700-stack-page.svg" alt="r9700-stack v0.4.0: what runs on it, the weight formats, the speed"></a>
  <br><sub>The release on one page: what the stack is, the four model configurations it runs, and how fast they run --
drawn by <code>docs/summary-page.py</code> from the same numbers the tables below carry, so a figure nobody can
re-draw cannot drift from them. Its canvas is 1000 px against github.com's ~980 px column, so the smallest type on
it lands at 14 px; the tag's own hand-drawn <code>docs/r9700-stack-v0.4.0.svg</code> carries the same claims at
11 px on a 1200-unit canvas, which measures out at 9.0 px in that column -- clicked-to-read, so the generated page
is what this README carries. The picture links to <code>docs/r9700-stack-page.png</code>, the same drawing at
2000 × 2532, for zooming. Rows tagged "210 W gate" are the release configuration, "the native box" is one uncapped
4x R9700 box, and "-500 MHz" is upstream's eight-card gate; the three are not comparable across.</sub>
</p>

<p align="center">
  <a href="#what-v040-gives-you">What it gives you</a> · <a href="#what-the-kernels-do">Kernels</a> ·
  <a href="#speed"><b>Speed</b></a> ·
  <a href="#watching-it-the-monitor-and-about-pages">Watching it</a> ·
  <a href="#deploying-it"><b>Deploying it</b></a> · <a href="#licence-and-credits">Licence</a>
</p>

<p align="center"><b>Bring a box up, or check one that already runs this repo:</b><br>
<code>./setup.sh --yes --launch native-flashnext.sh --webui same</code><br>
<sub>It reports the cards with their 32 GB BAR0 actually placed, <code>/dev/kfd</code>, the kernel cmdline,
memlock, docker or the native path, every checkpoint, then builds and starts. Nothing here fetches a wheel or a
checkpoint, and a launch that is already running is left alone. Both paths, image included:
<a href="#deploying-it"><b>Deploying it</b></a>.</sub></p>

## Four cards, the numbers worth knowing

Qwen3.8-Flash-Next (MoE, MXFP4 experts plus fp8 dense) on four Radeon AI PRO R9700, TP4, MTP-3, 31.9 GiB a card:

| | measured | where it came from |
|---|---:|---|
| **decoding, one stream** | **225 tok/s** (best sample 228) | 14.3-14.4 ms a step; release configuration, 210 W cap, 32k |
| **decoding, 32 streams** | **1,823 tok/s** aggregate (best sample 1,836.3) | 1,200-token waves, `bench/probe32.sh`; the native box, uncapped. The fp8 main-KV launch this box now defaults to reads **1,792** (1,789 / 1,796) at the same protocol |
| **reading a prompt** | **9,633 tok/s** | 7,789-token prompt, the native box uncapped (7,141 at the 210 W gate); **9,702** on the fp8 launch, 7,792 tokens |
| **context** | **262,144 tokens**, needle found at 250,078 | the checkpoint's own ceiling, running on this box today; KV pool 1,009,899 tokens with the fp8 main KV the launcher now defaults to, 540,202 at bf16 |
| **decoding, in real work** (not a benchmark) | **158 tok/s** with the box to yourself, **61 tok/s** as a request actually finishes | 1,400 real requests off the request journal, prompts p50 74k; the benchmark rows above describe a machine nobody sits on |

<sub>Two different machines, so the rows are not comparable with each other; each one names its own. The last row is
not a benchmark at all — it is what the running serve reported about other people's requests, and it is the row to
believe when asking what a request will feel like. Full protocols,
sample lists and the measurements that were tried and rejected: <a href="#speed">Speed</a>,
<a href="notes/native-4xr9700.md">notes/native-4xr9700.md</a>,
<a href="notes/context-262k.md">notes/context-262k.md</a>.</sub>

## What v0.4.0 gives you

- **Stock vLLM 0.32 on stock ROCm 10.** Pinned to the 2026-10-25 ROCm 10 nightly (vLLM 0.32.1rc1, torch 2.13); the
  plugin registers through vLLM's platform, quantization, model and attention-backend entry points and carries its
  own kernels (`libr9k`, HIP and Triton). Five compatibility fixes for the 0.32 line are in this release.
- **Four models, unmodified checkpoints:**
  - **Qwen3.8-27B-NVFP4** (dense) with the **DFlash2-FP8** block drafter: two cards, TP2.
  - **Qwen3.8-Flash-Next** (MoE, 288 routed experts, MXFP4 experts + fp8 dense): four cards TP4, or two cards with the
    experts streamed from pinned host RAM through a device LRU cache. On four cards it runs at the checkpoint's own
    **262,144-token** context, and what limits long context there is prefill scheduling, not the KV cache
    ([notes/context-262k.md](notes/context-262k.md)).
  - **GLM-5.3-Flash** (sparse MLA with the DeepSeek-V3.2-style indexer, RedHat NVFP4 checkpoint with its MTP layer):
    eight cards TP8, 32k production recipe; 262k launches retrieve correctly to ~75k tokens on this nightly (216k on
    the previous pin, see the release notes).
  - **DeepSeek-V4.1-Flash** (552B MoE: 384 routed experts, Engram n-gram tables, the DSpark drafter) from the published
    **EXL3 2.9 bpw pack** (196 GiB): eight cards, tensor parallel 8 with expert parallel, the Engram tables in pinned host
    memory, 32k context in the gated recipe and a 128k recipe that retrieves at 126.7k-token prompts. New in v0.4.0;
    notes/exl3-plan.md is the full story.
- **Weight formats, handled at load:** NVFP4 checkpoints converted to MXFP4 with MSE-searched block exponents (folded
  exponent, bit-exact on the 27B); fp8 dense, drafter and lm-head GEMMs with per-token fp8 activations; fp8 KV cache;
  the Flash-Next per-layer-embedding table as int6 in pinned host memory; MXFP4 x fp8 grouped MoE GEMMs with routing,
  activation and top-k sum fused around them; **EXL3 trellis weights** (QTIP-style, 2-6 bits) through exllamav3's gfx1201
  extension for the dense linears and our own batched expert decode kernel, with the 6-bit head and the shared experts
  dequantized to bf16 where the format's 128-wide blocks cannot be sharded.
- **Speed at a 210 W cap per card (single stream / aggregate / prompt processing):**
  - Qwen3.8-27B TP2: **216 tok/s** (4.5 tokens a step with the drafter), 527 / 640 tok/s at 8 / 16 streams, 3.9k tok/s
    on an 8k prompt, 367k-token KV cache.
  - Qwen3.8-Flash-Next TP4: **210-225 tok/s** (MTP-3), 913-944 tok/s at 16 streams, 6.8-7.5k tok/s on an 8k prompt,
    70-80 ms to first token, ~480k-token KV cache.
  - GLM-5.3-Flash TP8: **100 tok/s** (MTP-3; 75-80 before v0.3.8's host-memory collective), 161 tok/s at four streams,
    3.1-3.4k tok/s prompt processing at 2k-32k, needles found at 12.7k / 30k / 67k tokens on this nightly (216k on the previous pin).
  - DeepSeek-V4.1-Flash TP8 (EXL3 2.9 bpw, DSpark k = 5 with adaptive verification): **70-75 tok/s on prose, 158 on code**
    single stream, 187 / 336 tok/s aggregate at four streams, 3.4k tok/s prompt processing, decode flat to 28k of context,
    GSM8K 97-98 %. Two DGX Sparks on TensorFold serve the same pack at 81 / 118 single stream and 154 / 166 at four streams.
- **Speculative decoding on every family:** MTP-3 for Flash-Next and GLM, the DFlash2 drafter for the 27B, the checkpoint's
  DSpark drafter for DeepSeek, all on the same kernels, with the draft metadata, indexer logits and all-reduces inside the HIP
  graphs.
- **Multi-card communication over plain PCIe:** P2P one-shot / two-shot all-reduce and all-gather for 2 and 4 cards,
  a hierarchical 8-rank all-reduce across four PCIe switches (pair stages over P2P, the cross-switch stage through pinned
  host memory, one fused kernel for 32 KB - 2 MB messages), a compressed 4/6-bit all-reduce for prefill-sized messages, RCCL
  above that.
- **Gated like a release, every time:** kernel unit gates, the integration gate against the installed vLLM, strict
  sanity above `max_num_seqs`, mixed-length soaks and long-context needles on every configuration before a tag;
  paired GSM8K (1,319 questions, McNemar) and HumanEval on the numerics (27B: GSM8K 95.75%, HumanEval 162/164;
  GLM: GSM8K 97/100).

## What the kernels do

- **MoE experts**: grouped MXFP4 x fp8 GEMMs with the routing, activation, per-token quant and top-k sum fused
  around them; a device LRU expert cache with the experts streamed from pinned host memory for the two-card
  Flash-Next configuration.
- **Dense layers**: fp8 and NVFP4/MXFP4 weight-only GEMMs on WMMA, the activation quant fused into the GEMM at
  decode widths; NVFP4 checkpoints converted to MXFP4 at load.
- **Hyper-connections** (Flash-Next): the down + hc_silu and up + gated-mean GEMMs on fp8 copies of the weights at
  every row count, the bf16 copies freed after quantisation; combine-norm and gate-mix kernels for prefill.
- **Attention and recurrence**: paged attention for prefill, the query-sparse attention indexer and attention
  kernels, one fused Gated DeltaNet decode core per layer (conv update, gating, recurrence, norm), one GDN state
  page per request under speculative decoding.
- **Communication**: P2P all-reduce and all-gather over PCIe for 2 and 4 cards (one-shot and two-shot, fine-grained
  memory, HIP-graph safe), a compressed 4/6-bit all-reduce for prefill-sized messages, RCCL above that.
- **Speculative decoding**: MTP (Flash-Next) and the DFlash2 block drafter (27B) on the same kernels, with an int6
  per-layer-embedding table in pinned host memory.
- **Memory**: the KV cache sized from what the model actually uses, not torch.compile's transients; cudagraphs
  only where they pay.

Every default ships gated: the full 1,319-question GSM8K set paired per question against the previous numerics
(McNemar), HumanEval, strict sanity under overload, a 25-minute mixed-length soak on every configuration, the
multi-rank communication tests and the kernel unit gates. v0.3.2 kernels (unchanged in v0.3.3): GSM8K 95.75% (paired
p = 1.00 against v0.3.1), HumanEval 162 / 164; v0.3.3 validation: 0 bad answers in 5,060 strict-sanity checks,
0 errors in 1,224 soak requests.

## Speed (210 W with a -42 mV offset, 32k context; the v0.4.0 gate numbers are in the release notes)

Three kinds of number below, and they are not comparable across. The first four tables are the release
configuration, a 210 W cap per card with a -42 mV offset and 32k context (the v0.4.0 gate numbers are in the release
notes); the next two are one native box with the cards uncapped, the same recipe at 32k and then at the checkpoint's
full 262,144-token context. Those are all **benchmarks**: fixed prompt lengths, equal `max_tokens` waves, no
arrivals while you measure. The last subsection is **not** a benchmark — it is 1,400 real requests that arrived at
the running serve, and its decode numbers are roughly a third of the benchmark ones for reasons that are understood
rather than surprising. Read the benchmark tables for what the kernels can do; read the real-traffic subsection for
what a user gets.

### The release configuration: four, two and eight cards, 210 W cap, 32k context

**Qwen3.8-Flash-Next (MXFP4 experts, fp8 dense), four cards, TP4, MTP-3**

| | |
|---|---:|
| single-stream decode | **225 tok/s**, 14.3 ms a step (best sample 228) |
| aggregate decode at 1 / 2 / 4 / 8 streams (BetterBench) | 175 / 277 / 410 / 530 tok/s |
| aggregate decode at 8 / 16 streams (probe) | 620 / 944 tok/s |
| time to first token | 70-80 ms |
| prompt processing at 2k / 8k / 16k / 32k | 5,175 / 7,141 / 7,489 / 7,233 tok/s |
| KV cache | 518,481 tokens |

<details>
<summary><b>The same model on two cards, and the 27B on two cards</b></summary>

**Qwen3.8-Flash-Next, two cards, TP2, experts in host RAM**

| | PCIe Gen3 host (test box) | PCIe Gen5 host (production) |
|---|---:|---:|
| single-stream decode | 122 tok/s | 143 tok/s |
| aggregate at 8 / 16 streams | 177 / 163 | 257 / 245 |
| prompt processing at 8k | 2,594 | 3,724 |
| KV cache | 221,184 tokens | |

**Qwen3.8-27B-NVFP4, two cards, TP2, DFlash2 speculative decoding**

| | |
|---|---:|
| single-stream decode | 222 tok/s (4.5 tokens a step), 22 ms a step |
| aggregate at 8 / 16 streams | 527 / 640-658 tok/s (probe; equal-prompt bursts match the previous release) |
| prompt processing at 8k | 3,901-4,143 tok/s |
| KV cache | 367,494 tokens |

</details>

**DeepSeek-V4.1-Flash EXL3 2.9 bpw, eight cards, TP8 + expert parallel, DSpark k = 5 adaptive (eight-card runs at a -500 MHz
clock offset on this box's 120 V feed)**

| | |
|---|---:|
| single-stream decode, prose / code / GSM8K-style | 70-75 / 158 / 114-121 tok/s |
| per-stream decode at four streams, prose / code | 47 / 84 tok/s (187 / 336 aggregate) |
| decode after a 2k / 16k / 28k-token prompt | 82 / 80 / 75 tok/s |
| prompt processing | 2,500 tok/s on one 12k prompt, 3,000 aggregate on four at once, 3,400 on the warm short-prompt probe (the first prompt of a process JIT-compiles four Triton kernels) |
| KV cache | 329k tokens at MAXLEN 32768 (ten 32k requests), 314k at MAXLEN 131072 (`KVMEM=0.8`; one pool, replicated on every card) |
| accuracy | GSM8K 195 / 200 on the release launch (196 and 194 on the Phase 1 / 2 trees; greedy, thinking on); needles 9 / 9 at 8k, 16k, 28k; strict sanity 0 bad of 105 above max_num_seqs |

### The same four cards uncapped, at 32k

The native box below, on the recipe it ran before the context was raised (`MAXLEN=32768 NSEQ=16`, cards uncapped):
19.16 GiB of weights a card, draft acceptance 56.1% (2.68 tokens a step). This is where the prefill and aggregate
numbers in the block at the top of this README come from; [notes/native-4xr9700.md](notes/native-4xr9700.md) has
every sample list and the bands they repeat to.

| | |
|---|---:|
| single-stream decode | 186.7 tok/s, 14.3 ms a step (18.1 / 22.6 ms at 4 / 8 streams) |
| aggregate at 4 / 8 / 16 streams (`bench/bench.py`, 400-token streams) | 372 / 841 / 1,164 tok/s |
| aggregate, 32 in flight, 300-token waves (`bench/probe32.sh`) | 1,600.1 / 1,586.5 / 1,598.0 -> 1,598 tok/s |
| aggregate, 32 in flight, 1,200-token waves | **1,836.3 / 1,808.8 -> 1,823 tok/s** |
| aggregate, 16 in flight, 200-token waves (`bench.py run`, the recorded protocol) | 1,210.9 / 1,231.2 / 1,168.4 / 1,185.9 -> 1,198 tok/s |
| prompt processing at 1,951 / 7,789 / 31,166 tokens | **7,702 / 9,633 / 8,963 tok/s** |
| KV cache | 554,982 tokens |

Note the decode column and the release table above disagree: uncapped, a single stream runs at 186.7 tok/s, while the
210 W cap with a -42 mV offset reads 225 there. That is recorded, not reconciled -- the two boxes differ in host,
pin and clocks, and the 210 W figure has never been re-measured on this one.

### The same four cards uncapped, at the checkpoint's full 262,144 tokens

The native 4x R9700 box: vLLM 0.32.0+rocm723 in a venv, cards uncapped, so compare it against the uncapped numbers
in [notes/native-4xr9700.md](notes/native-4xr9700.md), not against the 210 W tables above. Same model, four cards,
TP4, MTP-3. Except where a row names another launch, this is the launch that runs on this box today
(`MAXLEN=262144 NSEQ=32 NBT=8192 PREFIX_CACHE=0`), re-measured after a restart.

| | |
|---|---:|
| KV cache at MAXLEN=262,144 | **540,202 tokens**; vLLM: `Maximum concurrency for 262,144 tokens per request: 2.06x` |
| the same launch with an **fp8 e4m3 main KV** (the default since 2026-11-03: 12 full-attention layers quantized, the QSA side caches left bf16) | **1,009,899-1,021,931 tokens**, `3.85-3.90x`; attention block size 880 instead of 448; prefill **2,614 / 7,886 / 9,702 / 8,965 / 8,282 tok/s at 485 / 1,976 / 7,792 / 31,131 / 77,733 tokens** (bf16: 2,586 / 7,938 / 9,705 / 8,928 / 8,252 -- flat); 32 in flight, 1,200-token waves **1,789 / 1,796** (bf16 1,817 / 1,831) and 300-token waves 1,511-1,563 (bf16 1,594-1,640), and that 1.5-4% is MTP acceptance -- 2.55 against 2.61 tokens a step -- not the kernel, which measures 7-11% faster; GSM8K 800 @8 **96.0 against 95.6**, HumanEval 164 **98.17 against 96.95** (McNemar b=2, c=0, p=0.50), needle intact cold and warm to 250,078 tokens: [notes/kv-fp8-qsa.md](notes/kv-fp8-qsa.md) |
| KV cache at MAXLEN=32,768 (same launch shape) | 497,588 tokens: raising context 8x made the pool **bigger**, because the sizing run is bounded by `NBT`, not by MAXLEN |
| prompt processing at 60k / 120k / 180k | 8,672-8,742 / 8,062-8,064 / 7,416-7,436 tok/s, each tier sent twice back to back with no prefix cache to help it |
| time to first token at 60k / 120k / 180k | 6.87-6.93 / 14.89 / 24.22-24.28 s, needle found at 60,061 / 120,066 / 180,070 tokens |
| the same 60k prefill as the engine measures it | **8,908-8,917 tok/s**: that request's own 60,061 prompt tokens over its own 6.74 s of prefill (the Monitor's per-request row, `FinishedRequestStats`) |
| retrieval ladder, a `PREFIX_CACHE=1` launch: 30,070 / 67,063 / 150,057 / 250,078 tokens | found at every tier (30-70% depth): TTFT 4.25 / 7.73 / 19.32 / **36.52 s**, prefill 7,068 / 8,671 / 7,768 / 6,847 tok/s, and **0.28-0.70 s** for a context the prefix cache has already seen |
| prompt processing at 30k / 250k, prefix cache off | 9,097 (the 8 x 30k wave) / 6,832 (a needle run) tok/s -- **-25% for 8x the context**, which is what 12 of 48 layers attending plus the QSA indexer buys |
| decode at depth (400 tokens inside a 250k context) | 141 tok/s single stream, on a repetitive haystack, so high draft acceptance |
| 32 in flight, 300-token waves (`bench/probe32.sh`) | **1,568.3 / 1,573.7 / 1,587.5 tok/s**; the 32k recipe's band at this protocol is 1,574-1,598 across instances, so the full context costs nothing measurable |
| 32 in flight, 1,200-token waves | 1,799.5 / 1,832.7 (32k band: 1,823) -- `max_tokens` alone moves this protocol by 15%, so a wave length belongs next to every aggregate number |
| 16 in flight, 300-token waves | 1,133.1 / 1,135.8 / 1,142.7 (32k band: 1,107) |
| 32 concurrent 300-token requests, wall clock from one process | 3.7-4.0 s: all 32 running at once, KV pool peak 20.7%, nothing waiting, because 32 x 300 tokens is two 8,192-token steps |
| 8 concurrent 30k prompts | 26.2 s end to end: TTFTs 3.6 -> 26.2 s in 3.3 s steps (one prefill each) with the pool only **11.7%** full |
| 3 concurrent 250k prompts | queue to 36.9 / 73.7 / 109.2 s; no OOM, all three retrieve |
| card power at `gpu_busy_percent` 100 | median **299 W** a card (132 samples, 179-303) |
| standing gate on the launch that runs today | GSM8K **99/100** (33,909 prompt tokens in 40 s), `bench/quality.py` needle 3/3 at 3,828 / 15,196 / 30,354 tokens |
| VRAM per card | 30.3-30.5 of 31.86 GiB at every tier up to 180k, ~1.35 GiB spare |

The limit is not the KV cache. Only 12 of 48 layers do full attention (24 KiB a token, 6.00 GiB per full-length
sequence), so a 262k request needs 586 of 1,230 blocks and the pool holds **2.06** of them -- a residency
ceiling, not the concurrency the box serves: long prefills serialize, so above roughly 30k tokens a prompt
"more than 2 in flight" already means queueing while the pool sits a tenth full. `--long-prefill-token-threshold`
(+ `_adaptive`) measured as a fairness knob, not a speed one: six 30k requests behind a 250k one go 40.3-55.8 s
-> **7.8-27.7 s** at an unchanged wall clock (55.9 vs 56.8 s), while three 250k prefills in parallel push the pool
to 99.1% and cost 22% (109.2 -> 133.6 s), so it stays off. Method, the queue measurements and the metric-reading
traps: [notes/context-262k.md](notes/context-262k.md).

### The same launch with real work on it, not a benchmark

Everything above this line was measured by sending requests on purpose: same prompt lengths, same `max_tokens`, and
nothing else arriving while the meter runs. That is the right way to compare builds, and the wrong way to answer
"what will my request get". The launch running today also kept a **request journal** (`~/logs/requests-8080.jsonl`,
one fsynced row per finished request, added to survive hard resets — timings and token counts only, no request
bodies), and by late afternoon it held **1,400 completed requests**: 13.8 h of wall clock, 301 minutes with traffic
in them, prompts p50 **74,381** tokens (p95 173,099, max 240,067), generated p50 348 tokens (p95 1,904). The count
keeps growing and the statistics do not: recomputed an hour later on 1,417 rows, prompt p50 is 74,066 and generated
p50 346. Nothing was sent to make this table; `tools/watch-live.py` only samples.

Each request's own decode rate is `completion_tokens / timings.decode_s` — the engine's own decode span, with
queueing and prefill excluded. Concurrency is rebuilt from the rows' own start and end stamps at 1 Hz, taking the
**maximum** over the request's decode window, not the value at its midpoint:

| how many requests shared the engine during its decode | share of active seconds | own decode, median |
|---|---:|---:|
| one — the stream had the box to itself | 32.7% | **157.7 tok/s** (p10 124, p90 181) |
| two | 18.6% | 119.1 tok/s |
| three to four | 31.8% | **22.8 tok/s** |
| five to six | 12.4% | 15.6 tok/s |
| seven to nine | 4.6% | 17.3 tok/s |
| **all 1,400 requests** | — | **60.8 tok/s** (mean 75.7, p10 8.7, p90 163.0) |

So the honest answer to "how fast does one stream decode" is **two numbers: 158 tok/s alone, 61 tok/s as a request
actually finishes**, and the difference is not decode speed — it is what sharing a prefill-bound engine costs.
Concurrency during traffic ran p50 2, p90 6, max 9.

- **The cliff is between two and three streams, and it is scheduling, not kernels.** Two to three in flight drops a
  stream from 119 to 23 tok/s. With `NBT=8192` and prompts of 50-170k, a third long prompt turns engine steps into
  ~1 s prefill chunks (measured: step p50 951 ms), and every stream is credited its accepted tokens once per step.
  The knob that addresses it is `--long-prefill-token-threshold`, not anything in the decode path — see
  [notes/context-262k.md](notes/context-262k.md).
- **Single-stream decode barely cares about context on this model.** Alone, generating ≥200 tokens: **161.9 tok/s**
  under 20k prompt, **158.2** at 20-80k, **149.7** at 80-160k, **140.3** above 160k — a 262k context costs one stream
  ~13%, because decode here streams MoE weights and attention's KV read is not the dominant term.
- **Prefill from real traffic matches the benchmark**, which is the useful cross-check: the same rows carry
  `timings.prefill_s`, and the rate is **median 7,911 tok/s** — 8,593 under 20k, 8,641 at 20-80k, 7,867 at 80-160k,
  **7,238 above 160k** — inside the 6.8-9.1k band the ladder measures, with a size falloff fixed-length benches
  cannot show.
- **The denominator decides any aggregate**, so all four are recorded: **16.6 tok/s** over the whole wall clock
  (63% of it an idle box), **45.4 tok/s** over active minutes only, **24.8 tok/s** token-weighted
  (`Σgenerated / Σdecode_s`, dragged down because long decodes are the ones that ran while busy), **60.8 tok/s** as
  the median of per-request rates — the last is the shape that answers "what will my request feel like".
- **158 is not a regression against the 225 above.** That figure is MTP-3 with ~3.1 accepted tokens a step; this
  launch runs **MTP=2** at 2.0-2.4 accepted a step, on contexts of 50-170k rather than 32k. If single-stream decode
  is ever wanted back, the knob is drafter depth, not the attention path.

Method, the preemption rows this window produced, and the trap that made the first version of this table wrong
(midpoint concurrency sampled 361 requests as "alone" and reported 137 tok/s): [notes/context-262k.md](notes/context-262k.md).

How each number above was produced, and what was tried and rejected: [PROGRESS.md](PROGRESS.md) and `notes/`.

## Watching it: the Monitor and About pages

<p align="center">
  <a href="docs/r9700-stack-monitor.png"><img src="docs/r9700-stack-monitor.png" alt="The Monitor page: engine state and the KV pool, request counts at the engine, generation and speculation, every card's busy% and VRAM, the engine's timings over the poll window, and the request list with each request's own wall clock, token counts and prefill"></a>
  <br><sub>The Monitor page on the launch that runs today: engine state and the KV pool, generation and speculation,
  every card's busy% and VRAM, the engine's timings over the poll window, and the request list carrying each
  request's own wall clock, token counts and prefill. The frame is 1688 x 1953, shown as wide as the page column;
  click it for the pixels.</sub>
</p>

<p align="center">
  <img src="docs/r9700-stack-monitor-cards.svg" alt="The Monitor's five headline cards, drawn from one kept GET /api/status: model Qwen3.8 at a 262,144-token context, TP4 MAXLEN 262,144 NSEQ 32 NBT 8,192 prefix cache off; KV pool 10.1% of 540,644 tokens; 2 running, 6 waiting, 6 for scheduling capacity, 0 deferred, 0 finished in the poll window; generation 2.0 tok/s, 14,915 prompt tok/s, 0.0 req/min, 14,999 tokens per iteration record prompt whole decode per step over the last 2 s; speculation 100.0% accepted, 20 requests finished since the engine started, 2.00 accepted tokens per draft over the last 2 s">
  <br><sub>The same five cards drawn rather than screenshotted: `docs/monitor-cards.py`, stdlib, drawing one
  recorded `GET /api/status` (the poll itself is committed next to it as `docs/r9700-stack-monitor-cards.json`, so
  the figure can be re-drawn and checked without a serve). `bench/capture-cards.py` caught it 4 seconds into eight
  concurrent ~30,000-token prompts on the standing 262k launch: two prefills run, six wait **for scheduling
  capacity**, none deferred, and the pool is only 10.1% full -- a queue, not a memory wall, and the reason the
  Requests card prints the split. Every rate on the figure says what it was averaged over: the Generation card's
  2.0 tok/s and 14,999-token iteration record are both means of the last 2 s, and that pair is the engine crediting
  whole prefills while its two running streams crawl. (That number is `vllm:iteration_tokens_total`, which calls
  itself "tokens per engine step" and is not one on this build -- a chunked prefill is credited in a single record,
  measured at 29,969 tokens in one record against NBT 8,192. The card says "iteration record" for that reason.)
  Drawn because the screenshot above lands at 0.58 scale inside this
  column and its text is unreadable, and a cropped wide strip would be smaller per panel, not larger: the column
  width is the limit for the whole strip.</sub>
</p>

A stdlib sidecar (`serve/webui.py`: no framework, nothing extra to install) that reads what a running launch already
publishes, `/health`, `/v1/models`, `/metrics`, plus the amdgpu sysfs of every card. Two shapes, and each page says
which one it is:

- **`WEBUI=1`, or `bash serve/webui.sh` against a launch that is already up** (sidecar, port 8081): out of process,
  so it keeps answering after the engine dies, and in `WEBUI_MODE=proxy` it sits between client and serve and
  records every request body. Recording costs a measured 2-3 ms a request, flat in prompt length (+5.3% of a
  76-token request, +0.5% of a 4.4k-token one); the default `observe` mode changes nothing in the inference path.
- **`WEBUI=same`** (merged into the launch, on its own port, `serve/plugin`): hooks vLLM's endpoint and stat-logger
  plugin groups, so the engine hands over each request's own queue / prefill / decode seconds and token counts,
  which no proxy can infer; the price is a page that dies with the launch and never sees a request body. It needs
  `pip install -e serve/plugin` and a `VLLM_PLUGINS` allowlist generated by `serve/webui-plugins.py`, not written
  by hand.

The About page hands out client config that is right where you paste it: the base URL is the address that answered
the page (`/v1` for the OpenAI SDK and LiteLLM, the root for curl), `openai/<served name>` rather than the
checkpoint path, the API key never printed, and the reasoning parser accounted for (`--reasoning-parser qwen3`
makes reasoning tokens count against `max_tokens`, so a small budget returns `content = None`).

Both pages' headers name the build that is answering: the plugin's version, and the branch, commit and dirtiness of
the tree that plugin was loaded from - today `vLLM 0.32.0 · plugin 0.4.0 · 953a3901`, with a `*` on the commit when
the tree has uncommitted edits, and the same facts as a row in About. "Is the live serve the new one?" used to cost
an ssh and a look at the launch log; now it is a glance, and the string survives into the screenshot someone sends.

The headline rates default to a **15 minute mean**, not the poll, and the window is selectable (this poll / 1 min /
15 min / 1 h). That is not decoration - on a launch with six streams and 90 k-token prompts a single poll is a coin
toss. Measured: one poll read 9.9 tok/s of generation and no prefill at all, another caught in a gap read 0, while the
quarter hour around them was 51.7 tok/s of generation and 7,893 tok/s of prefill. Each card still carries this poll in
its second line, because "is it stuck right now" is a question a 15 min mean cannot answer. The spans are deltas of the
engine's cumulative counters, not means of the 1 s samples - averaging those would weight an idle second the same as a
second with six streams running - and a span the page has not had time for is labelled with what it actually averaged
(`90s`, `2m`) rather than borrowing the name it asked for. The waveforms get the same treatment: 90 points at one per
poll, or an hour at one point per 15 s, each of those points the rate over its interval rather than a sample of one
poll. Every one of those lines is a **slot**: one line high whatever the engine reports, so a number growing from
`2.0` to `8,481` changes the value and never moves the page. The copy is written to fit the tightest card the layout
can produce (24 characters at 12 px) and the full sentences ride in the element's title.

The Requests card splits the waiting queue the way the engine labels it: `N waiting · M for scheduling capacity ·
K deferred`, and the split is the whole point on this model. Eight ~30,000-token prompts measured holding
`running 2 · waiting 6 · capacity 6 · deferred 0` with the KV pool at 8.8-10.2%: the queue is long prefills taking
the step one chunk at a time, not memory, and "waiting" alone would have read like the opposite.

What the pages refuse to invent: vLLM publishes nothing about a request while it is in flight, so there is no
prefill progress bar -- `prompt_tokens_total` is credited when a request ends, and the one live signal during a
long prefill is the KV pool filling (measured: 5.1 -> 32.1% in 3.0% steps for a single 180,066-token prompt).
Cards whose sysfs answers `EBUSY` are shown as unread rather than as 0%. Where every number comes from, and what
cannot be answered per request from outside: [notes/webui.md](notes/webui.md).

The PCIe card follows activity, not history: a chain's error cell prints `45 corr · 0 uncorr` while counters are
moving (this poll, or above one per minute over the trailing span), a drawn i-in-a-circle when the counters exist
but the span rate is at or under 1/min, and a bare dash when nothing has moved since boot; uncorrectable counters
and a link below its rating stay red whatever the rate. **The error cell is a button wherever there are numbers**:
clicking pins a block of text under the grid - the chain, the negotiated link and each port's rating, where the
slot is registered, totals since boot, the counter names, this window's delta, the rate over its span, how long
the calm has run, and the last event - which you can select, `Copy`, and close with `Esc`. A tooltip cannot be
copied, and this is text people paste into vendor reports.

What the page's own JavaScript renders can be checked against a live serve without a browser: `node
tests/render-monitor.mjs` runs `serve/web/monitor.js` unmodified in a stub DOM and prints the cards and the request
rows. That covers the JS and the payloads it is fed (a per-request card borrowing an engine-wide number, a real 0
printed as a dash, a token count scaled wrong); it does not cover CSS, fonts or layout, which need a real browser.
The figure above is covered too: `tests/test_readme.py` checks that every number in its caption is a number the
drawn figure actually contains, so the caption cannot drift from the poll it claims to describe.

## Deploying it

Requirements: 2x, 4x or 8x Radeon AI PRO R9700, ROCm 10 on the host, the model weights under `~/models`
(`Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ`, `Qwen3.8-27B-NVFP4` with `Qwen3.8-27B-DFlash2-FP8`,
`GLM-5.3-Flash-NVFP4-RedHat` for eight cards; for DeepSeek-V4.1-Flash a directory composed of the EXL3 pack's shards and
config plus the base checkpoint's two Engram shards and tokenizer -- `serve/ds41-exl3.sh`'s header has the recipe -- and
about 200 GB of host RAM for the Engram tables). Two ways to run it: in the image (docker), or with **no docker at
all** (a venv plus a `libr9k.so` built on the host, `serve/native-flashnext.sh`), which is what the box behind
these numbers runs.

0. **Set the box up, or check one that already runs this repo**: `./setup.sh` (bootstrap: it finds a Python 3.10+
   that can make a venv with pip, then runs `setup.py`, which holds the checks). It walks the box and prints the
   file each requirement came from: cards seen with their 32 GB BAR0 actually placed (`/sys/bus/pci/devices/*/resource`,
   not a ReBAR claim), `/dev/kfd`, kernel cmdline, memlock, **docker or the native path**, the pinned image and
   whether its nightly pin moved, the second checkout `serve.sh` mounts, every checkpoint the serve scripts name,
   and the tests that need no GPU.
   ```bash
   ./setup.sh --check                          # report only, changes nothing
   ```
   ```bash
   ./setup.sh --yes --launch native-flashnext.sh --webui same   # check, then start the native launch
   ```
   ```bash
   ./setup.sh --docker --launch 27b            # the container path: asked for, default no
   ```
   ```bash
   ./update.sh                                 # git pull here (refuses on tracked edits), re-check
   ```
   **The native path is the default, and the container path is the one thing asked for, with a default of no**
   (`--docker` takes it, `--no-docker` declines without the prompt, and `--yes` does not answer it). Every later
   step follows that answer: with the container path off the image is not inspected or built, the second checkout
   `serve.sh` mounts is not cloned, the container recipes are labelled in the launch list rather than offered,
   and a checkpoint only they want is a note instead of a warning. The native checks it runs instead: the venv
   importing vLLM/torch/hip with the same `LD_LIBRARY_PATH` `serve/native-flashnext.sh` sets (the ROCm wheels
   need a locally unpacked OpenMPI runtime, no root), and the host-built `libr9k.so` **as freshness, not
   existence** -- its hash, whether the copy `kernels/build.sh` wrote and the copy the launcher loads through
   `R9K_LIB` agree, and whether either is older than the newest `.hip` it is built from (step 1a's build line
   ends in a `cp`, and a build that skipped it is a build that never reached the serve). What it does with a
   launch: a native launch starts inside a `tmux` session -- it has to outlive the ssh session that started it by
   four minutes of loading, there is a pane to attach to while it loads, and the session is then the liveness
   probe (`--no-tmux` keeps the bare detached start and gives all three up), it refuses
   up front when the GPUs are still held by a stopped launch or when `OFFLOAD_GB` wants more pinned memory than
   this box's `RLIMIT_MEMLOCK` allows, it notices a launch that died instead of waiting out the 7-minute
   `/health` window, and it prints that log's last lines plus the `STOP=1` that clears the ranks. Each run
   records the native venv, the `libr9k.so` hash and the launch (recipe, log, session) under
   `.r9700-state/`, so the next run reports drift and answers "did my launch die, and which log do I read" (the
   whole launch method -- what supervises it, what refuses it, the `SIGTERM` fact behind the `STOP=1` -- is
   [notes/launch-methodology.md](notes/launch-methodology.md)). It names the venv/pip/`hipcc` commands rather than
   running them: the ROCm wheels are ~20 GB and are never fetched by a script, and nothing here touches `~/models`,
   the launch caches, or a launch that is already up.
   The host-side BAR0/ACS/PCIe-switch setup is in neither script: that is [host/README.md](host/README.md).

1a. **Native (no docker)**. One-time on the host; `./setup.sh --check` reports each missing piece with the command
   to run (the exact versions and URLs live in [notes/native-4xr9700.md](notes/native-4xr9700.md)):
   ```bash
   python3 -m venv ~/venv-r9700
   ```
   ```bash
   ~/venv-r9700/bin/pip install vllm==0.32.0+rocm723 --extra-index-url https://wheels.vllm.ai/rocm/<pin>/rocm723/
   ```
   ```bash
   pip install -e .                            # the r9700 plugin
   ```
   ```bash
   ~/venv-r9700/bin/pip install -e serve/plugin    # Monitor and About inside the serve (WEBUI=same)
   ```
   ```bash
   cd kernels && HIPCFLAGS=-I$HOME/opt/rocm-dev/opt/rocm/core-10.1/include ./build.sh && cp libr9k.so ../r9700_vllm/kernels/
   ```
   torch 2.13.0+git733fca1 and triton 3.8.0 come with that wheel. The index is per-commit and old pins expire:
   the one this release fetched 404s as of 2026-11-02, and vLLM 0.33 renamed the flavor to `rocm72`. No 0.32-era
   pin is live any more (six checked), so the venv IS the artifact of record -- keep it, don't plan to rebuild it.
   notes/vllm-0.33.md: the 0.33.0 gate result and why we did not move. The build line above is host hipcc, gfx1201,
   all 68 export checks, and it ends in the `cp` because the launcher loads the copy under `r9700_vllm/kernels/`.
   Those ROCm wheels link OpenMPI 4 and the host does not have it: unpack the runtime with `dpkg-deb -x` under
   `~/opt/openmpi` (no root; the deb list is in the note), and the launcher's `OPENMPI=` line puts it on
   `LD_LIBRARY_PATH`. Every serve knob then works exactly as in the container: step 2's block with
   `serve/native-flashnext.sh` in place of `serve/serve.sh`. What this path does not need: dockerd/containerd, an
   image, or the second build checkout `serve.sh` mounts at `/opt/r9700`. Measured on this box with the 262k
   launch up: four `VLLM::Worker_TP*` at 12.5 GiB RSS each, `VLLM::EngineCore` 1.22, API server 1.51, and
   `Mlocked` **0**, because on four cards the experts live in VRAM and nothing is pinned; `MemAvailable` 37.2 of
   93.8 GiB, 39.5 GiB of it reclaimable page cache from reading the checkpoint. The pinned host RAM the two-card
   offload configuration wants (about 120 GB, 38.7 GiB of it the PLE table) is what this path leaves out; it was
   never a docker allowance, and what the container layer itself costs is unmeasured here, since this box has no
   docker binary on it.

1b. **In the image**. Build it once (stock vLLM ROCm 10 nightly plus the plugin; the kernels compile inside):
   ```bash
   docker build -t r9700/vllm:dev docker/
   ```
   The pinned nightly (2026-10-25, vLLM 0.32.1rc1, torch 2.13) is in `docker/Dockerfile` (`--build-arg BASE=` for another);
   the image also builds exllamav3 v1.6.6's extension for gfx1201 (the EXL3 kernels, MIT).
   `serve.sh` runs the image against the second checkout of this repo (default `~/r9700-build/repo`, mounted at
   `/opt/r9700`) so the running plugin is code you can change without rebuilding; `./setup.sh` names it and offers
   to clone it. Container-only recipes are labelled in `setup.py`'s launch list, and asking for one on a run that
   took the native path is refused with the `--docker --launch <name>` command that would take it.

2. **Serve** (the same knobs either way):
   In the image:
   ```bash
   bash serve/27b.sh                                # 27B-NVFP4, two cards
   ```
   ```bash
   GPUS=0,1,2,3 TP=4 OFFLOAD_GB=0 NSEQ=16 bash serve/flashnext.sh    # Flash-Next, four cards
   ```
   ```bash
   bash serve/flashnext.sh                          # Flash-Next, two cards, experts in host RAM
   ```
   ```bash
   MODEL=/models/GLM-5.3-Flash-NVFP4-RedHat GPUS=0,1,2,3,4,5,6,7 TP=8 OFFLOAD_GB=0 EAGER=0 MTP=3 KVDTYPE=fp8 MAXLEN=32768 NSEQ=4 NBT=8192 PREFIX_CACHE=0 VLLM_ROCM_USE_AITER=1 R9K_AG_MAX_KB=2048 'SPEC_EXTRA="use_local_argmax_reduction": true' 'R9K_FP8_LINEARS=self_attn\.(in_proj_qkvbfg_a|o_proj|f_b_proj|g_b_proj)$' bash serve/serve.sh
   ```
   ```bash
   bash serve/ds41-exl3.sh                          # DeepSeek-V4.1-Flash EXL3, eight cards (notes/exl3-plan.md)
   ```
   Native, no container -- what this box runs today: four cards, the checkpoint's full 262k context, and the
   fp8 main KV the launcher now defaults, with Monitor and About on the serve's own port.
   ```bash
   MAXLEN=262144 NSEQ=32 NBT=8192 PREFIX_CACHE=0 WEBUI=same bash serve/native-flashnext.sh
   ```
   `serve/native-flashnext.sh` is the container-free launcher for a box with no docker (plugin and vLLM in a venv,
   `libr9k` built with the host `hipcc`, [notes/native-4xr9700.md](notes/native-4xr9700.md)); it takes the same knobs
   as `serve/serve.sh` and, since 2026-11-03, defaults `MAXLEN` to the value in the command above. It used to default
   to 32,768, and that default was a trap: a launch that answered `/health` in 150 s looked healthy while serving
   a fraction of the context the checkpoint declares (`text_config.max_position_embeddings = 262144`), on a box
   whose kept request ledger runs to 262,059-token prompts with a p99 of 245,792. `setup.py` now compares the
   running serve's `max_model_len` against the checkpoint's declared context and says so. That native command is what
   the box is running: at the full
   context the KV cache gets bigger rather than smaller (540,202 against 497,588 tokens) and the 32-stream
   throughput wave stays in the band quoted above.
   OpenAI-compatible API on port 8080 (`PORT=`), model name `Qwen3.8` (`DeepSeek-V4.1` for the DeepSeek launcher). Every serve script carries the measured
   defaults with the reason next to each knob; `R9K_*=stock` or `=0` switches any kernel back to vLLM's.
   `WEBUI=1` on any of them, or `WEBUI=same` to put the pages on the launch's own port: "Watching it" above.
3. **Check it** (what each one proves is next to it; run them in this order):
   ```bash
   ~/venv-r9700/bin/python3 bench/warmup.py         # the launch answers; the lazily-JIT'd kernels are compiled in
   ```
   ```bash
   ~/venv-r9700/bin/python3 bench/quality.py 100 8  # the standing gate: GSM8K + the needle ladder
   ```
   ```bash
   ~/venv-r9700/bin/python3 bench/sanity_stress.py  # strict 8-way sanity above max_num_seqs, under load
   ```
   ```bash
   ~/venv-r9700/bin/python3 bench/soak.py           # mixed-length soak, prompts of a random length
   ```
   ```bash
   ~/venv-r9700/bin/python3 bench/eval.py selftest  # reproducibility first; then `run NAME` / `compare A B`
   ```
   ```bash
   ~/venv-r9700/bin/python3 bench/humaneval.py      # HumanEval
   ```
   ```bash
   ~/venv-r9700/bin/python3 bench/needle_long.py    # retrieval to the checkpoint's own 262k
   ```
   ```bash
   ~/venv-r9700/bin/python3 bench/webui_ab.py       # what the Monitor's relay costs a request (2-3 ms, measured)
   ```
   `needle_long.py` measures its token counts against the checkpoint's tokenizer and plants a nonce every ~40
   sentences, so a prefix cache cannot carry the haystack; it reports per-card VRAM high-water per tier.
   Inside the image, drop the `~/venv-r9700/bin/`. Kernel unit tests are in `tests/`, and
   `bench/bringup-tests.sh` runs the single-GPU set of them.

Notes for the host: cards on the same PCIe switch for tensor parallelism, one card per switch for offloaded
experts; a KVM guest with an emulated PCIe switch needs the overlay in `overlay/emulated-switch`
(`OVERLAYS=emulated-switch`, the serve scripts' default) and the guest kernel flags in
[notes/mes-timeouts.md](notes/mes-timeouts.md); the two-card Flash-Next configuration needs about 120 GB of host RAM,
the four-card one needs no pinned host RAM at all. The release checklist and the upgrade recipe for a newer vLLM
nightly are in [notes/picking-up.md](notes/picking-up.md).

## Licence and credits

Apache-2.0 ([LICENSE](LICENSE)); provenance of the kernel structures and the libraries this grew out of in
[CREDITS.md](CREDITS.md) and [notes/independence.md](notes/independence.md).
