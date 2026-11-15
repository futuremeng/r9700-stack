# v0.4.4 — one port answers three serves, and the pages were rebuilt around that (2026-11-15)

262 commits since [v0.4.3](release-v0.4.3.md). The model path is not untouched in this range: **45 of those commits
touch `r9700_vllm/` or `kernels/`**. So read the gate paragraph before anything else — this tag says what was
measured, where, and by which launch, and it does not claim a gate that was not run.

## What was gated, and what was not

* **Measured on this box**: 4× Radeon AI PRO R9700, native launch path (`setup.py --no-docker --yes --config 262k`),
  **vLLM 0.32.0**, `Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ`, 262,144 ctx, TP4, MTP spec decode (2 tokens), fp8_e4m3 KV.
  Every number below comes from the launch that carries this version: `stack.plugin 0.4.4 · master · f33fe493`,
  clean tree, `GPU KV cache size: 902,153 tokens / 6.65 GiB` — byte-for-byte the same KV figure as the previous
  launch (`logs/setup-024353.log`), so this restart is comparable and the `~0.45 GiB` "restart once, then measure"
  caveat in `setup.py` did not apply.
* **Not re-run for this tag**: the eight-card release gate of [v0.4.3](release-v0.4.3.md). This box has not run an
  eight-card gate since; nothing here should be read as one.
* **Still owed by this box**: the paired **vLLM 0.33** gate. The pin moved upstream in `c14c7e5` ("this box owes its
  paired 0.33 gate") and the 0.32-vs-0.33 comparison landed in `0e34785` — a tie at speed, +2.8% KV, three findings.
  The launch running today is still 0.32.0.
* **A correction, stated where it can be found**: the tag `snapshot/pages-front-door-20261115` says "no model-path
  change". That is wrong — see the 45 commits above. What is true: the *serving* changes in this range are the ones
  listed under "Plugin and serving path" below, they were exercised on this box, and no eight-card gate covers them.

## Changed

### The front door — one port, loopback behind it (`serve/gateway.py`)

`0.0.0.0:8080` is the only outward port; the engine moved to `127.0.0.1:8180` and the CPU serves to
`127.0.0.1:8090/8092` with their watchers on `8091/8093`. The gateway reads the request line and pipes bytes — it
never parses a body, which is why it costs what it costs: **~0.8 ms/request** on small requests, with
`tests/test_gateway.py` failing the build if the hop exceeds **2 ms** (16 checks: exact-vs-namespace routing,
`Expect: 100-continue`, SSE not buffered, keep-alive, HEAD, a 60 KB body's digest). Routing is exact API paths
(`/v1/embeddings`, `/v1/rerank`, `/rerank`, …) plus namespace prefixes (`/web/`, `/api/`), so `/v1/embeddings-x`
does not reach the embedder. Two bugs fixed here are worth remembering: `/api/sidecars` parked the event loop
(blocking `urlopen` inside an async handler — now `asyncio.to_thread` + `gather`, cached 1.5 s behind a single
flight), and it closed the keep-alive connection under Chrome, which is what the browser reported as
`net::ERR_… canceled` in the Monitor.

### Two CPU sidecars, and the arithmetic that keeps them honest

`Qwen3-Embedding-0.6B-Q8_0` on cores 16-31 (`:8090`, `-t 15` — 15 because cpu 17 is offline, and `-t 16` measured
*slower*) and `Qwen3-Reranker-0.6B-Q8_0` on cores 32-47 (`:8092`, `-t 16`), each with its own watcher
(`serve/embed-watch.py`), each advertising a stable alias (`qwen3-embed`, `qwen3-rerank`) instead of a checkpoint
path. Measured: embedding **1,112 tok/s**, rerank **1,014 tok/s ≈ 3.4 query-document pairs/s**; 10 documents of
~100 tokens rerank in **1.03 s**, ~250 tokens in **2.9 s**. With all three busy the engine's ITL did not move
(13.0 → 12.9 ms). The red line from [incident-2026-11-01](incident-2026-11-01/) stays: mmap only, never `--mlock`,
and `Committed_AS` is read alongside every sidecar card. With all three up it sits at **87.9 GiB of 93.8 GiB** of
RAM — and during weight loading the *reservation* reads **250 GiB** for ~30 s before settling; that transient is
normal here, what killed the 11-01 launch was the sustained state.

### The pages

* **Monitor**: sidecar cards (rate, `-t`/`-c`, busy/deferred, `rss … · host …` with `Committed_AS`), declared
  statically in `monitor.html` so `tests/render-monitor.mjs` can measure slot widths; the section hides when
  `/api/sidecars` is not there, because the page is also reachable on the engine's own loopback port.
* **About**: the Connect card is a grid — one column per model family, one **row** per config field (model name,
  base URL, LiteLLM provider, LiteLLM `api_base`, `model_info.mode`, which process answers). The card before it
  showed one model and one curl on a port that answers three families, and that cost real time today: a rerank
  model declared `openai/` (LiteLLM's rerank API has no openai provider) and an `api_base` carrying the route, which
  dialed `POST /v1/rerank/rerank` and got a 404 naming no path. Verified end to end with **litellm 1.109.1** against
  this gateway: `/health` 3/3 healthy, and chat / embeddings / rerank all answered through the proxy (rerank scored
  0.9905 vs 0.0003). Copy buttons are one shared style now, and one that has nothing to copy says "Nothing yet"
  instead of writing an empty string and printing "Copied".

### Plugin and serving path (the 45 commits, the ones that change what is served)

* `6811dfc` the MoE gate cap baked into the recipe: `R9K_ROUTER_MAX_M=4`, not 8 and not 0.
* `b13d63d` fp8 e4m3 main KV for Qwen4Exp QSA — the writer was already fp8, so three guards and one read path.
* `a23c782` images default **on** for both launch paths, and only the last submitted turn's pictures are processed
  (`r9700_vllm/compat/mm_last_turn.py`).
* `d4aea4f` the stock GDN fallback gets the range check our own paths already had (`R9K_GDN_DEBUG=1`).
* The rest are instruments, not behaviour: the window probe and its A/B family (`pwal`: heartbeat, populate-vs-
  control, per-card clock/busy pairing, `prof.splitcall`), which is where the "is the device computing or is the
  queue asleep" question finally got numbers.

### Tests

`test_setup`, `test_readme`, `test_gateway`, `test_webui` pass; `render-monitor` and `render-about` run against a
live serve, the engine's own loopback port, and fixtures. Two harness lies were found and fixed while doing this:
both DOM stubs implemented `classList.toggle(name, force)` without `force` (so a page hiding columns was reported
as hiding columns it was showing), and the Monitor's "printing this poll under a 15m label" check fired at idle
magnitudes where 0.0088 tok/s and a 0.0 live poll are the same printed number.

## Measured on the launch that carries this version

| | this launch (0.4.4) | previous launch (0.4.3) |
|---|---|---|
| ITL through `:8080`, conc 1 / 4 / 16 | **13.1 / 18.6 / 27.4 ms** | 13.1 / 18.5 / 27.0 ms |
| GPU KV cache | 902,153 tokens / 6.65 GiB | 902,153 tokens / 6.65 GiB |
| `Committed_AS`, engine + both sidecars | **87.9 GiB** of 93.8 (`MemAvailable` 31.6 GiB) | 85.3 GiB ([embed-sidecar-box-test.md](embed-sidecar-box-test.md)) |
| `Committed_AS` during weight loading | reads **250 GiB** of reservation for ~30 s, settles to 87 | same shape |

The version bump is a string; the ITL column moving by 0.1-0.4 ms is the run-to-run spread this box shows at
concurrency 4 and above, not an effect of the release.

## Not in this tag, and owed

* The eight-card gate, and this box's paired vLLM 0.33 gate (above).
* `docs/summary-page.py` still carries `VERSION = "v0.4.2"` and that release's numbers, so the generated release
  figure the README points at predates two releases. It is generated, and re-drawing it means measuring its
  numbers first — not editing a badge.
