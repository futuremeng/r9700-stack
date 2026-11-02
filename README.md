<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/r9700-stack-lockup-dark.svg">
    <img src="docs/r9700-stack-lockup-light.svg" alt="r9700-stack" width="320">
  </picture>
</p>

<h3 align="center">Qwen3.8 and GLM-5.3-Flash, fast on Radeon AI PRO R9700 cards.</h3>
<p align="center">
  Tuned HIP and Triton kernels plus a vLLM plugin, on <b>stock vLLM and stock ROCm</b>: no fork, no patched source,
  no vendored binaries, no re-quantised weights. Everything loads at runtime through vLLM's own extension points, so
  vLLM and ROCm update underneath it.
</p>

<p align="center">
  <a href="#what-v038-gives-you">What it gives you</a> · <a href="#what-the-kernels-do">Kernels</a> ·
  <a href="#speed"><b>Speed</b></a> ·
  <a href="#watching-it-the-monitor-and-about-pages">Watching it</a> ·
  <a href="#deploying-it"><b>Deploying it</b></a> · <a href="#licence-and-credits">Licence</a>
</p>

<p align="center">
  <a href="https://github.com/futuremeng/r9700-stack/releases/tag/v0.3.8"><img src="https://img.shields.io/github/v/tag/futuremeng/r9700-stack?label=release&color=2f81f7" alt="release v0.3.8"></a>
  <a href="notes/release-v0.3.8.md"><img src="https://img.shields.io/badge/release%20notes-v0.3.8-2f81f7" alt="release notes v0.3.8"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/futuremeng/r9700-stack?label=license&color=2f81f7" alt="Apache-2.0"></a>
  <img src="https://img.shields.io/badge/vLLM-0.32%20(stock)-2f81f7" alt="stock vLLM 0.32">
  <img src="https://img.shields.io/badge/ROCm-10%20(stock)-2f81f7" alt="stock ROCm 10">
  <img src="https://img.shields.io/badge/Radeon%20AI%20PRO%20R9700-2%20to%208%20cards-6c757d" alt="two to eight Radeon AI PRO R9700 cards">
</p>

<p align="center"><sub>Current release <b>v0.3.8</b> (2026-10-31, the vLLM 0.32 line) ·
<a href="notes/release-v0.3.8.md">release notes</a> · <a href="CHANGELOG.md">CHANGELOG.md</a></sub></p>

<p align="center"><b>Bring a box up, or check one that already runs this repo:</b><br>
<code>./setup.sh --yes --launch native-flashnext.sh --webui same</code><br>
<sub>It reports the cards with their 32 GB BAR0 actually placed, <code>/dev/kfd</code>, the kernel cmdline,
memlock, docker or the native path, every checkpoint, then builds and starts. Nothing here fetches a wheel or a
checkpoint, and a launch that is already running is left alone. Both paths, image included:
<a href="#deploying-it"><b>Deploying it</b></a>.</sub></p>

## Four cards, the numbers worth knowing

Qwen3.8-Flash-Next (MoE, MXFP4 experts plus fp8 dense) on four Radeon AI PRO R9700, TP4, MTP-3, 31.9 GiB a card:

| | best measured | where that came from |
|---|---:|---|
| **decoding, one stream** | **225 tok/s** (best sample 228) | 14.3-14.4 ms a step; release configuration, 210 W cap, 32k |
| **decoding, 32 streams** | **1,823 tok/s** aggregate (best sample 1,836.3) | 1,200-token waves, `bench/probe32.sh`; the native box, uncapped |
| **reading a prompt** | **9,633 tok/s** | 7,789-token prompt, the native box uncapped (7,141 at the 210 W gate) |
| **context** | **262,144 tokens**, needle found at 250,078 | the checkpoint's own ceiling, running on this box today; KV pool 540,202 tokens |

<sub>Two different machines, so the rows are not comparable with each other; each one names its own. Full protocols,
sample lists and the measurements that were tried and rejected: <a href="#speed">Speed</a>,
<a href="notes/native-4xr9700.md">notes/native-4xr9700.md</a>,
<a href="notes/context-262k.md">notes/context-262k.md</a>.</sub>

## What v0.3.8 gives you

<p align="center">
  <a href="docs/r9700-stack-v0.3.8.png"><img src="docs/r9700-stack-v0.3.8.svg" alt="r9700-stack v0.3.8: what runs on it, the weight formats, the speed"></a>
</p>

<sub>The release on one page: the configurations it runs, the formats it loads, the four-card numbers -- as measured
for **v0.3.8** at the 210 W gate, which is why its Flash-Next row says 210 tok/s where the table above says
**225**: the table carries the newer measurements (14.3-14.4 ms a step, same 210 W gate, 32k), and every row there
names the machine and protocol it came from. Drawn by `docs/summary-image.py` from the release notes.
Its smallest type is 9.8 px at github.com's column width, which is small: the picture links to
`docs/r9700-stack-v0.3.8.png`, the same drawing at 2400 × 1220.</sub>

- **Stock vLLM 0.32 on stock ROCm 10.** Pinned to the 2026-10-25 ROCm 10 nightly (vLLM 0.32.1rc1, torch 2.13); the
  plugin registers through vLLM's platform, quantization, model and attention-backend entry points and carries its
  own kernels (`libr9k`, HIP and Triton). Five compatibility fixes for the 0.32 line are in this release.
- **Three models, unmodified checkpoints:**
  - **Qwen3.8-27B-NVFP4** (dense) with the **DFlash2-FP8** block drafter: two cards, TP2.
  - **Qwen3.8-Flash-Next** (MoE, 288 routed experts, MXFP4 experts + fp8 dense): four cards TP4, or two cards with the
    experts streamed from pinned host RAM through a device LRU cache. On four cards it runs at the checkpoint's own
    **262,144-token** context, and what limits long context there is prefill scheduling, not the KV cache
    ([notes/context-262k.md](notes/context-262k.md)).
  - **GLM-5.3-Flash** (sparse MLA with the DeepSeek-V3.2-style indexer, RedHat NVFP4 checkpoint with its MTP layer):
    eight cards TP8, 32k production recipe; 262k launches retrieve correctly to ~75k tokens on this nightly (216k on
    the previous pin, see the release notes).
- **Weight formats, handled at load:** NVFP4 checkpoints converted to MXFP4 with MSE-searched block exponents (folded
  exponent, bit-exact on the 27B); fp8 dense, drafter and lm-head GEMMs with per-token fp8 activations; fp8 KV cache;
  the Flash-Next per-layer-embedding table as int6 in pinned host memory; MXFP4 x fp8 grouped MoE GEMMs with routing,
  activation and top-k sum fused around them.
- **Speed at a 210 W cap per card (single stream / aggregate / prompt processing):**
  - Qwen3.8-27B TP2: **216 tok/s** (4.5 tokens a step with the drafter), 527 / 640 tok/s at 8 / 16 streams, 3.9k tok/s
    on an 8k prompt, 367k-token KV cache.
  - Qwen3.8-Flash-Next TP4: **210-225 tok/s** (MTP-3), 913-944 tok/s at 16 streams, 6.8-7.5k tok/s on an 8k prompt,
    70-80 ms to first token, ~480k-token KV cache.
  - GLM-5.3-Flash TP8: **100 tok/s** (MTP-3; 75-80 before v0.3.8's host-memory collective), 161 tok/s at four streams,
    3.1-3.4k tok/s prompt processing at 2k-32k, needles found at 12.7k / 30k / 67k tokens on this nightly (216k on the previous pin).
- **Speculative decoding on both families:** MTP-3 for Flash-Next and GLM, the DFlash2 drafter for the 27B, all on the
  same kernels, with the draft metadata, indexer logits and all-reduces inside the HIP graphs.
- **Multi-card communication over plain PCIe:** P2P one-shot / two-shot all-reduce and all-gather for 2 and 4 cards,
  a hierarchical 8-rank all-reduce across four PCIe switches, a compressed 4/6-bit all-reduce for prefill-sized
  messages, RCCL above that.
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

## Speed

Two different machines below, and they are not comparable across: the first three tables are the release
configuration, a 210 W cap per card with a -42 mV offset and 32k context (v0.3.8's gate numbers are in the release
notes); the last two are one native box with the cards uncapped, the same recipe at 32k and then at the checkpoint's
full 262,144-token context.

### The release configuration: four and two cards, 210 W cap, 32k context

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

The Requests card splits the waiting queue the way the engine labels it: `N waiting · M for scheduling capacity ·
K deferred`, and the split is the whole point on this model. Eight ~30,000-token prompts measured holding
`running 2 · waiting 6 · capacity 6 · deferred 0` with the KV pool at 8.8-10.2%: the queue is long prefills taking
the step one chunk at a time, not memory, and "waiting" alone would have read like the opposite.

What the pages refuse to invent: vLLM publishes nothing about a request while it is in flight, so there is no
prefill progress bar -- `prompt_tokens_total` is credited when a request ends, and the one live signal during a
long prefill is the KV pool filling (measured: 5.1 -> 32.1% in 3.0% steps for a single 180,066-token prompt).
Cards whose sysfs answers `EBUSY` are shown as unread rather than as 0%. Where every number comes from, and what
cannot be answered per request from outside: [notes/webui.md](notes/webui.md).

What the page's own JavaScript renders can be checked against a live serve without a browser: `node
tests/render-monitor.mjs` runs `serve/web/monitor.js` unmodified in a stub DOM and prints the cards and the request
rows. That covers the JS and the payloads it is fed (a per-request card borrowing an engine-wide number, a real 0
printed as a dash, a token count scaled wrong); it does not cover CSS, fonts or layout, which need a real browser.
The figure above is covered too: `tests/test_readme.py` checks that every number in its caption is a number the
drawn figure actually contains, so the caption cannot drift from the poll it claims to describe.

## Deploying it

Requirements: 2x, 4x or 8x Radeon AI PRO R9700, ROCm 10 on the host, the model weights under `~/models`
(`Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ`, `Qwen3.8-27B-NVFP4` with `Qwen3.8-27B-DFlash2-FP8`,
`GLM-5.3-Flash-NVFP4-RedHat` for eight cards). Two ways to run it: in the image (docker), or with **no docker at
all** (a venv plus a `libr9k.so` built on the host, `serve/native-flashnext.sh`), which is what the box behind
these numbers runs.

0. **Set the box up, or check one that already runs this repo**: `./setup.sh` (bootstrap: it finds a Python 3.10+
   that can make a venv with pip, then runs `setup.py`, which holds the checks). It walks the box and prints the
   file each requirement came from: cards seen with their 32 GB BAR0 actually placed (`/sys/bus/pci/devices/*/resource`,
   not a ReBAR claim), `/dev/kfd`, kernel cmdline, memlock, **docker or the native path**, the pinned image and
   whether its nightly pin moved, the second checkout `serve.sh` mounts, every checkpoint the serve scripts name,
   and the tests that need no GPU.
   ```
   ./setup.sh --check                                    # report only, changes nothing
   ./setup.sh --yes --launch native-flashnext.sh --webui same   # check, then start the native launch
   ./update.sh                                           # git pull here (refuses on tracked edits), re-check
   ```
   On a box with no docker it says so and checks the native path instead: the venv importing vLLM/torch/hip with
   the same `LD_LIBRARY_PATH` `serve/native-flashnext.sh` sets (the ROCm wheels need a locally unpacked OpenMPI
   runtime, no root), the host-built `libr9k.so`, and `pip install -e serve/plugin` for `WEBUI=same`. It names the
   venv/pip/`hipcc` commands rather than running them: the ROCm wheels are ~20 GB and are never fetched by a
   script, and nothing here touches `~/models`, the launch caches, or a launch that is already up. The host-side
   BAR0/ACS/PCIe-switch setup is in neither script: that is [host/README.md](host/README.md).

1a. **Native (no docker)**. One-time on the host; `./setup.sh --check` reports each missing piece with the command
   to run (the exact versions and URLs live in [notes/native-4xr9700.md](notes/native-4xr9700.md)):
   ```
   python3 -m venv ~/venv-r9700
   ~/venv-r9700/bin/pip install vllm==0.32.0+rocm723 --extra-index-url \
     https://wheels.vllm.ai/rocm/<pin>/rocm723/          # torch 2.13.0+git733fca1 and triton 3.8.0 come with it
   pip install -e .                                      # the r9700 plugin
   ~/venv-r9700/bin/pip install -e serve/plugin          # Monitor and About inside the serve (WEBUI=same)
   cd kernels && HIPCFLAGS=-I$HOME/opt/rocm-dev/opt/rocm/core-10.1/include ./build.sh \
     && cp libr9k.so ../r9700_vllm/kernels/              # host hipcc, gfx1201, all 68 export checks
   ```
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
   ```
   docker build -t r9700/vllm:dev docker/
   ```
   The pinned nightly (2026-10-25, vLLM 0.32.1rc1, torch 2.13) is in `docker/Dockerfile` (`--build-arg BASE=` for another).
   `serve.sh` runs the image against the second checkout of this repo (default `~/r9700-build/repo`, mounted at
   `/opt/r9700`) so the running plugin is code you can change without rebuilding; `./setup.sh` names it and offers
   to clone it. Container-only recipes are labelled in `setup.py`'s launch list.

2. **Serve** (the same knobs either way):
   ```
   # in the image
   bash serve/27b.sh                                                  # 27B-NVFP4, two cards
   GPUS=0,1,2,3 TP=4 OFFLOAD_GB=0 NSEQ=16 bash serve/flashnext.sh     # Flash-Next, four cards
   bash serve/flashnext.sh                                            # Flash-Next, two cards, experts in host RAM
   MODEL=/models/GLM-5.3-Flash-NVFP4-RedHat GPUS=0,1,2,3,4,5,6,7 TP=8 OFFLOAD_GB=0 EAGER=0 MTP=3 KVDTYPE=fp8 \
     MAXLEN=32768 NSEQ=4 NBT=8192 PREFIX_CACHE=0 VLLM_ROCM_USE_AITER=1 R9K_AG_MAX_KB=2048 \
     'SPEC_EXTRA="use_local_argmax_reduction": true' 'R9K_FP8_LINEARS=self_attn\.(in_proj_qkvbfg_a|o_proj|f_b_proj|g_b_proj)$' \
     bash serve/serve.sh                                               # GLM-5.3-Flash, eight cards (notes/glm53.md)
   # native, no container: what this box runs today
   MAXLEN=262144 NSEQ=32 NBT=8192 PREFIX_CACHE=0 WEBUI=same \
     bash serve/native-flashnext.sh      # Flash-Next, four cards, the full 262k context,
                                         # Monitor and About on the serve's own port
   ```
   `serve/native-flashnext.sh` is the container-free launcher for a box with no docker (plugin and vLLM in a venv,
   `libr9k` built with the host `hipcc`, [notes/native-4xr9700.md](notes/native-4xr9700.md)); it takes the same knobs
   as `serve/serve.sh` and defaults to `MAXLEN=32768`. The line above is what the box is running: at the full
   context the KV cache gets bigger rather than smaller (540,202 against 497,588 tokens) and the 32-stream
   throughput wave stays in the band quoted above.
   OpenAI-compatible API on port 8080 (`PORT=`), model name `Qwen3.8`. Every serve script carries the measured
   defaults with the reason next to each knob; `R9K_*=stock` or `=0` switches any kernel back to vLLM's.
   `WEBUI=1` on any of them, or `WEBUI=same` to put the pages on the launch's own port: "Watching it" above.
3. **Check it** (what each one proves is next to it; run them in this order):
   ```
   ~/venv-r9700/bin/python3 bench/warmup.py        # the launch answers; the lazily-JIT'd kernels are compiled in
   ~/venv-r9700/bin/python3 bench/quality.py 100 8 # the standing gate: GSM8K + the needle ladder
   ~/venv-r9700/bin/python3 bench/sanity_stress.py # strict 8-way sanity above max_num_seqs, under load
   ~/venv-r9700/bin/python3 bench/soak.py          # mixed-length soak, prompts of a random length
   ~/venv-r9700/bin/python3 bench/eval.py selftest # reproducibility first; then `run NAME` / `compare A B`
   ~/venv-r9700/bin/python3 bench/humaneval.py     # HumanEval
   ~/venv-r9700/bin/python3 bench/needle_long.py   # retrieval to the checkpoint's own 262k: token counts measured
                                                   # against the checkpoint's tokenizer, a nonce every ~40 sentences
                                                   # so a prefix cache cannot carry the haystack, per-card VRAM
                                                   # high-water per tier
   ~/venv-r9700/bin/python3 bench/webui_ab.py      # what the Monitor's relay costs a request (2-3 ms, measured)
   ```
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
