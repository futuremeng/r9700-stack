# The Monitor and About pages (serve/webui.py, serve/web)

A web front end for a running `serve/` launch: what the engine is, what the cards are doing, and — if
you put it in the path — every request that went through. The layout is ported from
[Strata](https://github.com/Niko1221/Strata)'s web app (MIT; see [CREDITS.md](../CREDITS.md)), whose
Monitor and About views are the shape we wanted. The parts that Strata's server could answer from
inside and ours cannot are said out loud on the page instead of being guessed at.

```
bash serve/webui.sh                 # watch the serve on 8080 from http://127.0.0.1:8081/monitor
WEBUI=1 bash serve/serve.sh         # the same, started for you, with this launch's recipe on About
WEBUI_MODE=proxy bash serve/webui.sh   # also relay /v1/*, which is what fills the request list
```

## Why a sidecar, and what it costs

Stock vLLM has **no per-request API**. It has `/health`, `/v1/models` and `/metrics` — the last a set of
cumulative counters, not a history. Strata's Monitor can list every request because Strata's server
keeps those records in its own process; here, nothing does. So:

| mode | what it does | what it costs |
|---|---|---|
| `observe` (default) | `GET /health`, `/v1/models`, `/metrics` every `POLL_S` (2 s), plus amdgpu sysfs reads | nothing in the inference path at all; the request list stays empty and the page says why |
| `--proxy` | the same, plus it relays API traffic and records it | one extra TCP hop and one thread per in-flight request: **2-3 ms per request, flat in prompt length** (+5.3% of a 76-token request, +0.45% of a 4.4k-token one, [measured live](#on-the-live-launch)) |
| `merged` (`WEBUI=same`) | the pages served by the serve itself, on its own port, from `serve/plugin` | ~11 ms of registry render every 2 s on a thread inside the serve; and two capabilities no in-process route can have (below) |

Request records (input, output, reasoning, wall-clock, first token, token counts) exist **only** in
`--proxy`, and only for the traffic that came through this port. `--key` guards the API and the relay
(not the HTML, which holds nothing): without it an open relay to a keyed upstream is worse than no
monitor.

## Merged into the serve's own port (`WEBUI=same`, `serve/plugin`)

One package, two entry points, installed with
`~/venv-r9700/bin/pip install --no-build-isolation -e serve/plugin` (`setup.sh` checks it and says so):

| group | class | what it does |
|---|---|---|
| `vllm.endpoint_plugins` | `WebMonitor` | the routes: `/`, `/monitor`, `/about`, `/web/*`, `/api/*`. `attach_endpoint_plugins` puts a plugin's `GET /` **ahead of** the server's own root, by design |
| `vllm.stat_logger_plugins` | `RequestLog` | sees every finished request as the engine reports it: `IterationStats.finished_requests` → `FinishedRequestStats` (`vllm/v1/metrics/stats.py:288-306`) |

The second one is why merging is not a downgrade. vLLM's per-request stats carry `queued_time`,
`prefill_time`, `inference_time`, `decode_time`, `mean_time_per_output_token`, `num_cached_tokens`,
`max_tokens_param` and the finish reason. **The queue/prefill/decode split per request is measured here
and cannot be measured from outside at all** — the sidecar can only average histograms over a window, and
`monitor.html` said so. Live, one request through the merged serve:

```
queue 0.000 s   prefill 0.040 s   decode 0.650 s   66 -> 120 tokens (0 cached)   184.5 tok/s per accepted token
```

`/metrics` also stops being a socket round-trip: the poller renders the process's own Prometheus registry
(`generate_latest(get_prometheus_registry())`) — the same bytes the serve's `/metrics` returns.
Measured cost of that render: **~11 ms per 2 s poll** for 65 KB (0.5% of a core, on a thread, never on
the event loop: every route handler is a plain `def`, so FastAPI runs it on its threadpool, and the sysfs
reads — which can block on EBUSY — and the rocm-smi subprocess live on their own threads).

What merged mode cannot have, and no route can invent:

- **the request text.** The engine reports tokens and timings, never bodies. Captured bodies need the
  sidecar in `--proxy`. The detail pane says so in one line instead of showing three empty panes.
- **in-flight rows.** A stat logger sees a request when it *finishes*. The running/waiting counts are
  still live, from the gauges.
- **a page that outlives the engine.** Same port, same process: when the launch dies the monitor dies
  with it, and the cards panel goes with it — which is the reason the sidecar stays, and why on a box with
  [watchdog and PSU incidents](incident-2026-11-02/) the sidecar on 8081 is the one to leave running.

**`VLLM_PLUGINS` is a strict allowlist, not an "also load these" list — writing it by hand cost a launch.**
Endpoint plugins load only when named, so `WEBUI=same` has to set `VLLM_PLUGINS`. But for the other groups
vLLM loads *everything discoverable* when that variable is unset, and only the named ones when it is set.
So `VLLM_PLUGINS=r9700_webui` silently disabled **this repo's own `r9700` plugin** — the one that teaches
compressed-tensors the MXFP4-pack-quantized scheme — and the engine died four minutes into the load with
`NotImplementedError: No compressed-tensors compatible scheme was found for layer_name=...in_proj_qkvz`.
The value is therefore built from discovery (`serve/webui-plugins.py`), never typed:

```
lora_filesystem_resolver,lora_hf_hub_resolver,quark_online_quant,r9700,r9700_webui
```

and a test asserts the repo's own plugin survives it.

## Where each number comes from

| page number | source |
|---|---|
| model, served name | `GET /v1/models` |
| KV pool in tokens, block size, TP/PP, max context, KV dtype, scheduler | the **labels** of vLLM's own `/metrics` series (`cache_config_info` and friends) — vLLM publishes its configuration nowhere else; which labels exist moves between releases, so a dash means this build does not publish it |
| running / waiting / swapped / KV pool used % | gauges `vllm:num_requests_*`, `vllm:gpu_cache_usage_perc` — **max over ranks** for the gauge (`sum` would say 112% when two ranks are at 56%; the sentence that matters is "the worst rank is at X") |
| preemptions (window and since launch) | `vllm:num_preemptions_total`, read as a cumulative count next to the window delta. It is the counter that says a long-context load has gone past what the pool can hold, and the window alone hides it: the two preemptions this box has ever counted happened seconds apart at 14:24 and 14:29, each on a ~160k-token prompt with the pool at 100%, and each cost its request a whole re-prefill (first token 20.9 s / 30.2 s, that request's own decode falling to 14.8 / 3.4 tok/s). A build that does not publish the counter leaves it a dash, never a 0 |
| tok/s, req/min, prompt tok/s | counter deltas over the poll window (`generation_tokens_total`, `prompt_tokens_total`, `e2e_request_latency_seconds_count`); `prompt_tokens_total` is credited when a request FINISHES, so the line is labelled "credited at request end" |
| prefill speed of one request | inside the serve: that request's own `prompt - cached` tokens over its own `prefill_time` (whole prompt when nothing was cached), computed when the record is made. Through a relay: prompt tokens over first-token latency, printed with that basis because it includes engine queue |
| first token, queue, prefill, decode means | `_sum`/`_count` deltas of `time_to_first_token_seconds`, `request_queue_time_seconds`, `request_prefill_time_seconds`, `request_decode_time_seconds` (`bench/ttft_breakdown.py` reads the same family) |
| speculation acceptance %, drafts/request, accepted tokens/draft | `spec_decode_num_accepted_tokens_total` ÷ `spec_decode_num_draft_tokens_total` (the older builds only have `..._num_drafts_total`, which gives drafts per request but not the token ratio); `accepted ÷ num_drafts_total` is accepted tokens per draft, which cannot exceed `num_speculative_tokens` — 2.0 at MTP=2, and that ceiling is on the card because it is the check that catches a wrong counter pair |
| tokens per iteration record, records per request | `iteration_tokens_total_{sum,count}` — **not a step size despite the metric's own HELP text**; see below |
| prefix cache hits % | `prefix_cache_hits_total` ÷ `prefix_cache_queries_total` |
| card busy % / VRAM / temperature / sclk / mclk | `/sys/class/drm/card*/device` (`gpu_busy_percent`, `mem_info_*`, `hwmon*/{temp1_input,power1_average}`, `hwmon*/freq*_label` matched by `sclk`/`mclk`, not by index) |
| the launch recipe on About | the meta file `serve/serve.sh` and `serve/native-flashnext.sh` write when `WEBUI` is set (`--meta`), not something inferred. It is named after the port the **pages** are served on — `PORT+1` for the sidecar, `PORT` for `WEBUI=same`, since merged mode has no second port. Write it to one name and read it from the other and the card silently disappears: that is how the merged About page lost it |

**`iteration_tokens_total` is labelled "tokens per engine_step" and is not one.** Measured against single
requests on 0.32.0 (delta of `_sum` and `_count` across one request, plus which histogram buckets moved): a
4,893-token prompt produced **2 records** (4,893 and 4) and a 29,969-token prompt produced **3** (29,969, 4, 1),
against `--max-num-batched-tokens 8192`. So `loggers.py:1273`, which observes
`prompt_token_stats.computed + num_generation_tokens`, credits a whole chunked prefill in **one** record while a
decode step gets a record of its own. Two consequences: the sum/count ratio tracks prompt size rather than the
step budget and can sit far above NBT (9,621 since launch, 14,999 in a 2 s window on the 262k launch), and
`iteration_tokens_total_count / requests` is records per request, not steps. The card says "tokens per iteration
record (prompt whole, decode per step)"; the tooltip carries this measurement. Anyone who wants a real step rate
has to count steps themselves — nothing in `/metrics` gives one.

**`prompt_tokens_total` is credited when a request finishes, not as prefill proceeds.** Measured against a
single 180,066-token prompt: the counter sat at 432,262 through the whole prefill and jumped to 612,332 in one
go at the end, so the "prompt tok/s" card is a step function at completion (89,580 tok/s for one poll) and not
a live prefill rate; the line now reads "prompt tok/s (credited at request end)". The per-request mean lives in
the record instead: 60,061 tokens over a 6.74 s prefill prints as 8,908 tok/s on that row, with the numerator
dropped to computed tokens whenever the engine reported cached ones. What DOES move live during prefill: `vllm:kv_cache_usage_perc`, sampled every 2 s while
`running=1` and `generation_tok_s=0`, climbed 5.1 / 8.1 / 11.1 / 14.1 / 17.1 / 20.1 / 23.1 / 24.6 / 27.6 /
30.5 / 32.1% in 3.0% steps (3.0% of the 540,202-token pool = 16,206 tokens = two 8,192-token chunks per ~2 s
sample, about 7,400-8,100 tok/s at that spacing, in line with the 8,639-8,759 tok/s the same script measures
directly at 60k). So the KV pool bar on the Monitor already IS a live prefill
indicator for the engine as a whole, and it is not made into a per-request progress bar: the denominator would
be that request's prompt length, which the engine only reports at the end.

A per-request prefill bar was asked for and refused for that reason. vLLM's `SchedulerStats` has nothing per
request in flight, and `SchedulerIterationDetails` (`num_ctx_requests`, `num_ctx_tokens`, `elapsed_ms`, per
engine step, no request ids) is only built when `--enable-logging-iteration-details` is set
(`config/observability.py:99` default False, gated at `v1/engine/core.py:582`), whose visible effect is one
INFO line per engine step on the path everything else is measured on. Strata can draw the bar
(`prompt_read / prompt_total`, `serve/server.py:2102`) because it owns the engine's terminal protocol and the
tokenizer; stock vLLM gives neither.

**What cannot be answered per request from outside:** queue wait and the prefill/decode split. vLLM counts
those in aggregate, so in `observe`/`proxy` the Monitor shows them as window means under "Engine timings"
and the per-request rows carry a dash. Inside the serve (`serve/plugin`) the engine hands over each request's
own split and the rows carry it; for the breakdown from outside use `bench/ttft_breakdown.py`, which
differences `/metrics` around a controlled set of requests.

**A row has to carry what its label claims.** Both row builders list from one key list (`ROW_KEYS` in
`serve/webui.py`) because the cards read the *list*, not the detail. While `usage` and `timings` lived only in
the detail, the "Last decode speed ... One stream" card fell back to the engine-wide `1/inter_token_latency`
and still called itself one stream: measured on the live merged page, **27.4 tok/s** next to a request whose
own figure was **87.8 tok/s** (27.4 is `1000 / 36.4 ms`, the engine step), and a request's own prefill appeared
nowhere, only a 2 s mean, so a 250,029-token prompt and a 22-token one looked identical. Rows now print
`22 -> 296 tok`, `prefill 0.12 s`, `queue ...`, `preempted ...`, the wall-clock card splits that one request,
and the speed card prints a dash plus the reason rather than borrowing a number. The three speed bases are
different measurements and are labelled apart: that request's own tokens over its own decode span (in merged
mode, per accepted token, the engine's mean), `1/inter_token_latency` (one engine step, shared by every stream
running), and `generation_tokens_total` over the window (all streams summed: 871.7 tok/s with 32 concurrent).

**The model name is the alias, not the checkpoint.** `GET /v1/models` is the authority: this launch serves
`Qwen3.8` (`--served-model-name ${SERVED_NAME:-Qwen3.8}`), whose `root` is
`~/models/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ`. Asking by the root, by the directory name, or with different
case gets vLLM's own 404, which quotes what it was asked and lists what it accepts:

```
The model `qwen3.8` does not exist. Valid aliases: Qwen3.8.
```

That line is the whole diagnosis of a "model does not exist": it names the aliases of the serve that
answered, which also tells you whether you reached the serve you meant (the `serve/baseline-tcclaviger/*`
recipes alias `Qwen3.8-Flash-Next` and `Qwen3.8`; a launch given `SERVED_NAME=` answers to neither of these
names). The About page prints the alias, the base URL to use, and the path with the words "not a name a client
can ask for" next to it — see "Copy-paste client config" below.

## Cards: what sysfs refuses on this box

Measured 2026-11-02, on the four-R9700 host with nothing running on the cards:

```
cat /sys/class/drm/card1/device/gpu_busy_percent   -> Device or resource busy
cat /sys/class/drm/card1/device/hwmon2/temp1_input -> Device or resource busy   (all temp/power/freq too)
cat /sys/class/drm/card1/device/mem_info_vram_used -> 59916288                  (fine)
```

so the reader never reports a refused attribute as zero: the card block names what it could not read.
`SMI=1` additionally calls `rocm-smi --showuse --showmeminfo vram --showtemp --showpower --showclocks
--showproductname --json` and fills only the gaps, joining by **PCI bus id** (`PCI_SLOT_NAME` in
`uevent` vs `PCI Bus` in rocm-smi) because rocm-smi's `card0` is not the sysfs's `card0`. That is
opt-in for a reason: rocm-smi opens `/dev/kfd` on every call, which on a box you are measuring is a
perturbation, not a read. Keep it off during gates and soaks; keep it on when you want the clocks.
`power_w`/`temp_c` come back `N/A` from rocm-smi in the state this box was in after the
watchdog/PSU events ([notes/incident-2026-11-02/](incident-2026-11-02)) — dropped, not zeroed.

**What the refusals correlate with is not what I guessed.** Follow-up reads, same host, same
card set, with the native launch up and then stopped:

| state of the box | `gpu_busy_percent` |
|---|---|
| nothing running (first observation, 2026-11-02 morning) | EBUSY on all four; VRAM answers (57 MiB) |
| native launch up, engine idle | answers 24-29% |
| native launch up, three 600-token streams | answers 46-100% |
| native launch stopped, nothing running | answers 0%, `unread` empty |

So "a compute context holds the card" is not the rule — it refused with nothing running and answered
while the engine was up. What it does guarantee: any of these attributes can fail on any poll, and the
page says which one failed instead of printing 0. Both directions have been exercised against the live
sidecar without restarting it (four cards refusing, then four answering).

## Copy-paste client config (About)

About exists so a client can be configured off the page, so its blocks are generated, not typed: a curl call, a
LiteLLM proxy config, and a direct `litellm.completion`. Three rules they encode, all measured:

- **The base URL is the address the browser reached the serve on** (`location.origin`), never the sidecar's
  internal `--upstream` (`127.0.0.1` on the serve's box — pasted into a laptop, that is your own laptop; this
  box is reached over the network). LiteLLM and the OpenAI SDK get it **with** `/v1`, curl gets it without.
- **The model string is `openai/<served name>`.** The checkpoint path (`/v1/models`' `root`) is on the page
  with the words "not a name a client can ask for", because asking by it returns a 404 that lists the aliases.
- **`max_tokens` counts reasoning.** These launches pass `--reasoning-parser qwen3`, and litellm 1.106.6 at
  `max_tokens=64` against this serve returned `content = None` with `reasoning_tokens = 64`: the whole budget
  went into the chain of thought. The snippet asks for 128 and prints both `message.content` and
  `message.reasoning_content`. The parser name reaches the page through the recipe file — the launchers pass
  `${REASONING:-qwen3}` to vLLM **and** record it in the meta, and a test asserts the two lines name the same
  variable so the page cannot describe a serve that is not running.

Verified end to end (litellm installed into `/tmp`, not into either venv): the direct snippet as rendered by
the page printed content and reasoning; the YAML block, loaded and driven through `litellm.Router`, answered too
— `model used -> Qwen3.8`.

## Endpoints

```
GET /                     Monitor          GET /web/<file>      this page's own assets
GET /monitor  /about      the two pages    GET /healthz         the sidecar's own state (sidecar only)
GET /api/status           engine config + live + per-window rates + history + cards + host + meta
GET /api/requests         the retained list (last --keep, newest first)
GET /api/requests?id=...  one request: bodies from the relay, timings from the engine
GET /api/host             CPU, threads, RAM, kernel, uptime
```

Everything else is relayed to the upstream. Nothing is written to disk; the list is memory and dies
with the process. `--keep 100` default, `--poll 2` default. Merged mode (`serve/plugin`) serves the same
paths on the serve's port — no `/healthz` (the serve has `/health`), and everything the serve itself
serves (`/v1/*`, `/metrics`) is untouched apart from `GET /`, which the plugin replaces on purpose.

## Tests

`tests/test_webui.py` (135 checks, no GPU and no network needed) and `tests/test_webui_plugin.py`
(32 checks; the record store and the stat logger run anywhere, the routes need FastAPI, and vLLM's own
route-replacement tier runs only where vLLM imports — `LD_LIBRARY_PATH=~/opt/openmpi/usr/lib/x86_64-linux-gnu
~/venv-r9700/bin/python tests/test_webui_plugin.py`). Three more cover the parts that are not the sidecar:
`tests/test_readme.py` (36 checks, stdlib alone: the README's card figure re-draws from the committed `/api/status`
poll string for string, no line runs off the canvas and none runs past the bottom of the card that drew it, no
wrapped line ends on a connector, the caption cannot state a number the figure does not show, the title lockup and
the release page sit where they belong and link to a picture big enough to read), `tests/test_lockup.py` (14 checks
on the same lockup rendered, which needs an SVG renderer — `rsvg-convert` or cairosvg — and skips saying so without
one), and `tests/render-monitor.mjs`, which runs the page's own JavaScript unmodified in a stub DOM against a live
serve and prints what its cards and rows actually render — the JS and its payloads, not CSS or layout.

- `/metrics` collapsing: buckets skipped, counters summed over ranks, gauges taken as max, KV capacity
  and configuration read out of labels, and a counter that did not move reporting *no data* rather
  than zero — the first window after a start has no window, and dividing the whole run's counters by
  0.001 s is a number this box generates enough of already.
- the request ring: eviction, the fields the list carries (never the bodies), the 256 KiB truncation
  flag, decode speed from `usage` versus an estimated token count (and the flag saying which).
- the sysfs reader against a fake tree, including a card whose `gpu_busy_percent` refuses to answer.
- rocm-smi parsing out through its banner, joined by PCI bus id, filling only gaps, and an unknown
  card becoming its own row.
- end to end against a fake vLLM: SSE relayed byte for byte, first-token stamp and wall-clock from
  the stream that passed, `usage` from the final chunk, a 500 relayed as it was and recorded as an
  error, `observe` relaying while recording nothing, the `--key` gate on both the API and the relay,
  `/web/../` traversal refused (a plain `startswith` check would have served this file: `serve/webui.py`
  starts with `serve/web`), and the pages' ids/assets/CSS variables cross-checked against the HTML so a
  renamed id fails a test instead of blanking a card.
- merged mode: the engine record's queue/prefill/decode split and what a `0` means there (a request that
  never queued must read 0.000 s, not a dash — found live), `/web/` subpaths (the fonts) on the merged
  port, that a `merged` hub knows it is merged before its first request finishes (`bool(EngineRecords)` is
  False while empty), that it sees the cards (`cfg["gpus"]` defaulted to None and the page came up
  cardless), and that building `VLLM_PLUGINS` keeps `r9700`.

## What the relay costs, and what recording costs

`bench/webui_ab.py` runs the same requests through three arms in one process: `direct` (the serve),
`relay` (the sidecar in observe mode, which only forwards) and `record` (proxy mode, which also parses
the SSE deltas as they pass), at three prompt tiers (`--tiers 45,1500,3000`) and two shapes
(`--shape first|stream|all`). The tier is the axis that matters: the hop is a fixed cost, so its
**share** of a request should fall as the prompt grows. Against a **loopback stand-in that answers in
~0 ms** -- so the milliseconds are ours, and its prefill cost is a fiction, ignore its % column --
the hop stays flat while the prompt grows 45 -> 1200 tokens:

| shape | arm | median | first token | tok/s | vs direct |
|---|---|---|---|---|---|
| `max_tokens=1` (the whole request is ~1 ms) | direct | 0.92 ms | 0.89 ms | — | — |
| | relay | 1.86 ms | 1.86 ms | — | **+0.95 ms** |
| | record | 1.85 ms | 1.84 ms | — | +0.93 ms |
| streamed 64 tokens (~272 ms) | direct | 271.5 ms | 203.5 ms | 235.7 | — |
| | relay | 273.3 ms | 205.0 ms | 234.3 | +1.74 ms (+0.6%) |
| | record | 273.0 ms | 204.9 ms | 234.5 | +1.46 ms (+0.5%) |

Across tiers (streamed 64 tokens, stand-in, `--n 8`): **+1.3-1.7 ms** at 45 tokens, **+1.8-2.1 ms** at
600, **+1.3-1.6 ms** at 1200 -- flat, which is the whole argument: the hop does not scale with the
prompt, so its share falls as the request gets longer.

So: about **1 ms per request** for the hop (one extra loopback connection and a thread), flat in
prompt length, and **recording is free inside that** -- relay and record are within noise of each
other, because the delta parsing happens on bytes that were already being copied.

### On the live launch

Native vLLM 0.32.0, four cards, MTP=2, warmed, box otherwise idle, `--n 8` per cell (prompt tokens are
what the engine reports, so the chat template's overhead is included: asking for 45 gets 76):

| prompt tok | shape | direct | relay | record | hop | recording |
|---|---|---|---|---|---|---|
| 76 | `max_tokens=1` | 47.5 ms | 50.0 | 50.3 | **+2.54 ms (+5.3%)** | +0.23 ms |
| 76 | streamed 64 | 660.7 ms | 659.3 | 658.1 | -1.37 ms (-0.2%) | -1.16 ms |
| 2,099 | `max_tokens=1` | 241.6 ms | 243.5 | 243.8 | **+1.96 ms (+0.8%)** | +0.23 ms |
| 2,099 | streamed 64 | 476.0 ms | 478.8 | 479.0 | **+2.78 ms (+0.6%)** | +0.18 ms |
| 4,399 | `max_tokens=1` | 454.6 ms | 457.0 | 458.0 | **+2.43 ms (+0.5%)** | +0.99 ms |
| 4,399 | streamed 64 | 710.0 ms | 713.2 | 713.5 | **+3.18 ms (+0.45%)** | +0.35 ms |

The hop is **2-3 ms and flat in prompt length**, so its share falls from 5.3% of a 76-token request to
0.45% of a 4.4k-token one, and recording costs **0.2-1 ms on top of relaying**. It is larger than the
stand-in's ~1 ms (the engine's own first-token path is longer, and the relay adds a connection to a
process that is not the one holding the model). The one negative row is arm order, not a win: the arms
always run direct, relay, record, and at `n=8` the engine warming between arms is the same order of
magnitude as the hop. Read the single-request cells at a 76-token prompt as roughly the resolution floor
of the measurement.

```bash
python3 bench/webui_ab.py http://127.0.0.1:8080 --n 8 --tiers 45,1500,3000 --shape all
```

## What the live launch taught the parser

The fake engine in `tests/test_webui.py` was written from the container image's `/metrics`. The native
vLLM 0.32.0+rocm723 differs, and each difference showed up as a panel reading `—`:

| 0.32.0 native | older / container | what the sidecar does now |
|---|---|---|
| `vllm:kv_cache_usage_perc` | `vllm:gpu_cache_usage_perc` | tries both, first one wins |
| `vllm:inter_token_latency_seconds` | `vllm:time_per_output_token_seconds` | the engine **step** family -> `itl_ms` |
| `vllm:request_time_per_output_token_seconds` | — | per **accepted token** -> `tpot_ms`, a different quantity |
| (absent) | `vllm:num_requests_swapped` | `swapped: null`, not 0 — vLLM retired that counter |
| `vllm:*_created` | — | skipped: a `_created` Unix stamp is not a sample |

**Two inter-token families, and the gap between them is MTP.** `inter_token_latency_seconds` gets one
observation per engine **step** — what a streaming client feels as one delta.
`request_time_per_output_token_seconds` gets one per **request**, holding that request's mean per
*accepted* token, so MTP's accepted drafts divide it. From `/api/status` and by hand from `/metrics`, same
numbers:

```
cumulative step = 15.62 ms    per accepted token = 3.19 ms    accept = 59.7%    requests = 156
by hand: 36.055 / 2309 = 15.62 ms     0.4979 / 156 = 3.19 ms
```

That pair's ratio, 4.9 tokens a step, is impossible on a launch running `num_speculative_tokens: 2` (the ceiling
is 1 + 2 = 3), which is how we found out the sentence this section used to carry — *"a stream's tok/s is
1 / step, never 1 / per-token"* — was backwards. Measured in one window on one stream, same box, same day
(`/api/status` while an 800-token generation ran): step **12.84 ms**, per accepted token **6.27 ms**, and
`generation_tokens_total` moving **155 tokens a second**. 1/6.27 ms = 159.5 tok/s matches the counter; 1/12.84 ms
= 77.9 is the **step** rate and understates a stream by exactly the tokens a step produces (12.84/6.27 = 2.05,
i.e. 1 + ~1.05 accepted of the 2 drafted). A stream's token rate is 1/per-token. Both are still on the page, and
`tokens_per_step` now reconciles them; the cumulative ratio above is kept as an example of a pair that should have
been rejected by the ceiling check instead of believed.

The other half of that measurement matters for freshness, and is why the Monitor's headline row changed:
| what | live while a request runs? | measured |
|---|---|---|
| `generation_tokens_total` | **yes**, per engine step | a 900-token stream moved it ~120 tokens per 0.5 s sample |
| `kv_cache_usage_perc` | **yes**, per step | 0.0360 -> 0.1007 across one 51,859-token prefill |
| `num_requests_running` / `waiting*` | yes | 0 -> 1 within one 0.5 s sample of the request arriving |
| `prompt_tokens_total` | **no** — credited when the request ends | flat at 129 for 10 s, then one jump to 51,988 |
| `iteration_tokens_total_{sum,count}` | **no** — same landing | 510 -> 515 and 53,296 -> 105,163 both at the jump |
| `request_time_per_output_token_seconds` | **no** — one observation per request | `tpot: null` mid-request, 6.3 ms the window it finishes in |
| every latency histogram (TTFT, queue, prefill, decode, e2e) | **no** | observed at request end |

So: decode rates are live (counter-derived, `stream_tok_s_live` = aggregate ÷ running), prefill is not — the
number shown during a long prefill is the **KV gauge's growth in tokens a second** (`kv_tok_s`, measured within
~6% of the truth on that 51,859-token prompt: 5,787 estimated against 5,459 real, and negative when finishing
requests free blocks), with the exact counter rate appearing beside it when the request lands. Polls went from
2 s to 1 s on both beats (engine poll and browser refetch), and the page prints its own age (`1.0 s poll ·
snapshot 0.4 s ago`) instead of letting the reader guess.

**The order on the page, now that there are two kinds of number.** Row 0 is values only — one stream, all
streams, prefill, engine step + first token — at `--st-fs-display`. Row 1 is the four waveforms the sidecar
keeps a ring for, in one row over the same span, each captioned `last · peak · avg · n/90 pts over 90 s`: the
ring is fixed in **points** (HIST=90), so what it looks back at is `90 x poll_s`, and a rate caption averages
only the windows that actually carried a rate (a flat window mid-prefill is a zero, not a measurement — counting
those made a 200 tok/s stream average 31.5); the KV pool series opts out, because an idle pool at 0% *is* the
value. Per-request cards come next, speed and wall-clock before the model name; the aggregate waveform used to be
drawn twice (once large, once small in the Generation card) and is now drawn once.

**The KV pool's progress bar became a waveform too.** The bar said one thing the number above it already said
(“how full right now”); the shape says the thing a long-context question is actually about — how fast the pool
filled, whether it ever stopped, how fast it drains. Its colour keeps the bar’s warning job (warn past 85%,
danger past 95%). It lives on its own card rather than in the trends row because its axis is different — and that
axis is the part that decides whether a waveform means anything:

| series | axis | why |
|---|---|---|
| decode one stream / all streams / streams running | **0 .. peak** | autoscaled, a rate that is zero for 80 of 90 windows draws as a full-height oscillation |
| prefill (KV growth estimate) | autoscaled | it goes negative when finished requests free blocks; a 0 baseline would clip the half that says “the pool is draining” |
| KV pool % | **0 .. 100** | measured: a pool sitting at 0.1-0.3% renders at y=33.9 of 36 — a sliver, which is what it is; autoscaled it filled the box |

`spark()` now takes the axis from the caller (`baseLo` / `baseHi`), and treats a null as a **gap**: the line
breaks into a new subpath and the filled area under it is dropped for that series. Filling across a gap draws
“the series was at baseline there”, which is the opposite of what a null means — for the prefill estimate, whose
first ~30 windows have no value at all, that would have invented a flat prefill. The x axis spans the whole ring
either way, so one point is one poll of time whatever the series was doing in between.

### The shapes a waveform draws when nothing is happening

An anchored series with no samples drew **nothing at all** — the box went empty and the line under it said
`no series yet`, which is the reader left guessing between "nothing is running", "the counter is dead" and "this
page is broken". Two changes:

- **The axis's zero is now drawn** — faint, dashed, under the data. A series with no samples (no streams running,
  so a per-stream rate has no denominator: the payload carries null, not 0) still shows where nothing-happening
  is, and a series genuinely at zero shows a line lying on the axis, which is the finding rather than a rendering
  failure. For the one autoscaled series (the KV-growth estimate) the line goes where zero is, because crossing
  it — the pool draining instead of growing — is the event worth seeing there. A fully autoscaled series with no
  samples still draws an empty box, honestly: without samples there is no axis to state.
- **Each waveform says why in its own terms**: `no stream running, so a per-stream rate has no denominator`,
  `nothing was being generated`, `the KV pool did not grow`, `0 streams running`, each followed by the span and the
  axis it is flat against (`· 90 s back · axis 0..152 tok/s`) when the axis is known. `no series yet` is gone.

`tools/spark-check.js` runs the primitive over the shapes that matter — nothing running, everything zero, a gap in
the middle, a series that goes negative, a single sample — and prints the paths out, since summoning those states in
a browser on demand is the part that never worked:

```
all null, axis 0..peak             spark-base y=34..34
all zero, axis 0..100              spark-base y=34..34 | fill y=34..36 | line y=34..34
half null then 40s                 spark-base y=34..34 | fill y=22..36 | line y=22..22
gap in the middle                  spark-base y=34..34 | line y=25..25
prefill estimate spans zero        spark-base y=23.6..23.6 | fill y=4..36 | line y=4..34
no samples, autoscaled             (no svg — empty box)
```

One trap this closed on the way: a `hi: "peak"` axis over an empty series is `Math.max()` of nothing, which is
`-Infinity` — the axis is clamped to 0 and a series whose bounds are not finite draws nothing rather than a box
full of NaN.

### PCIe: the link under the cards

Asked whether PCIe traffic could be waved, the answer came out **partly**, and the split is the interesting part.

- **Bytes: not collectable here.** Every counter that would answer "how much crossed the link" replies `N/A` on
  these RDNA4 cards — `amd-smi metric --pcie` gives `BANDWIDTH`, `CURRENT_BANDWIDTH_SENT`,
  `CURRENT_BANDWIDTH_RECEIVED`, `MAX_PACKET_SIZE`, `REPLAY_COUNT`, `NAK_SENT/RECEIVED_COUNT` as `N/A` for all four
  GPUs; `--profiling` is not even a valid flag in the `amd-smi` this box ships; there is no `amdgpu` PMU under
  `/sys/bus/event_source/devices/` (only `amd_iommu_*`); and the kernel exposes no byte counter in the device
  directory. So no waveform gets built out of a hope.
- **The link layer: free, and it is the evidence the reset storm was asking for.** `/sys/bus/pci/devices/<bdf>/`
  answers as an ordinary user with no `/dev/kfd`: `current_link_speed` / `current_link_width` against
  `max_link_speed` / `max_link_width`, `power_state`, and the AER counters `aer_dev_correctable` /
  `aer_dev_nonfatal` / `aer_dev_fatal` — each a list of named counters ending in a total.

`read_pcie` walks the **sysfs topology** rather than parsing `lspci`: a PCI device's resolved path sits under its
upstream bridge, so following parents turns an endpoint into a chain — endpoint → switch downstream port →
upstream port → root port — and every level carries its own counters.

**It is drawn as rows, not waveforms — and the reason is a measurement, not taste.** The first version drew four
waves per chain (GT/s, lanes, correctable delta, uncorrectable delta). A day of watching says those two signals
are *static-or-eventual*, which is the worst thing for a curve:

- **Link state is static.** Over 90 consecutive samples: one distinct value, `32.0 GT/s ×16`, never degraded. It
  changes when a link retrains — once, discretely — so a 90-point wave of it is a flat line and the interesting
  case (a link that came up `×8` on `×16` lanes) is a number plus a flag.
- **AER counters are eventual.** 0 of 90 idle samples had a nonzero delta; an eight-stream × 400-token burst
  produced **0** events; yet the counters had moved ~12 times that morning, all clustered around serve restarts.
  A signal that is silent for hours and then jumps twelve times is an *event log*, and a waveform of it is flat
  99.9% of the time while claiming the space.

  That reasoning was corrected the same afternoon, by real traffic rather than by argument: 4–5 concurrent
  60k–170k-token streams produce **~1 event per 5 minutes**, mostly while a big prompt is being prefilled (the
  incident note has the table, with each event matched to the requests in flight at its timestamp). "Eventual"
  survived — an event every few minutes is still an event log, not a curve — but "only at launch" did not.

So a chain is now **one row**: card, the device chain (`c6:00.0 → c5:00.0 → c4:00.0 → c0:03.1`), the negotiated
link (`32.0 GT/s ×16`, and `(max ×16)` stated when it fell short), totals since boot (`36 corr · 0 uncorr`), the
non-zero counter names, and this window's delta (`+0`). One box-wide spark stays — AER events per second across
all chains, zero-anchored so a flat line reads as *zero*, not as *no data* — because the shape worth keeping is
"an event happened at this instant". Alongside it the note lists the last three events **with wall-clock times**
(`12:55:12 c4:00.0 BadTLP +3`), which is the whole point of watching: an event you can line up against a stall.
`aer_deltas` also keeps a baseline-on-first-sighting rule, so counters that were already non-zero when the page
started are history, not events we witnessed.

One naming trap to keep in mind when reading those counter names: **`NonFatalErr` lives inside
`aer_dev_correctable`** (the spec's Advisory Non-Fatal Error bit — the device reported an error and the hardware
recovered). `TOTAL_ERR_NONFATAL` is the one that means trouble, and it reads 0 on this box. The row therefore
labels the group (`corr BadTLP 30 · BadDLLP 1 · NonFatalErr 3`) instead of trusting a bare name that reads like
an uncorrectable fault. Counters can also be **zeroed by the kernel** (link recovery, driver reload), which is why
a decrease is treated as unknown rather than negative.

What it showed in its first running hour (this box, the 11:07 boot, read 12:07–12:14 UTC):

| chain | correctable since boot | what they are | non-fatal / fatal |
|---|---|---|---|
| card2 `c6:00.0 → c5:00.0 → c4:00.0 → c0:03.1` | 28 → **31** | `c4:00.0` upstream port BadTLP 24 → 27; `c0:03.1` Timeout 2; endpoint and downstream port 1 each | 0 / 0 |
| card4 `06:00.0 → 05:00.0 → 04:00.0 → 00:03.1` | 5 | `04:00.0` upstream port BadTLP 3; endpoint and downstream port 1 each | 0 / 0 |
| card1 `c3:00.0 → c2:00.0 → c1:00.0 → c0:01.1` | 0 | — | 0 / 0 |
| card3 `03:00.0 → 02:00.0 → 01:00.0 → 00:01.1` | 0 | — | 0 / 0 |

Three correctable errors landed on card2's upstream port during the seven minutes of watching, while every link
stayed at its maximum **32.0 GT/s x16** — nothing is falling back.

### The request journal: the list outlives the process that held it

The request list lived only in the serve's memory, and this box dies without writing a shutdown record — 15 boots
on 2026-11-02, several of them mid-request. After every reset the Monitor came back with an empty table, and "what
was it doing when it died" had to be reconstructed from side logs written by hand. So finished rows are now appended
to a JSONL journal and replayed at startup:

- **one line per finished request**, holding exactly `ROW_KEYS` — **no bodies**. The engine records never carried
  text, the relay's are capped and large, and the incident question is about timing and tokens.
- **a `boot` id per process** (start epoch + pid), so a list spanning a reset reads as two launches rather than one
  afternoon with strange gaps. The page says which: `request journal ~/logs/requests-8080.jsonl · 3 rows carried
  over from earlier launches · this launch 1793622173-12882`, and "Requests retained" is qualified
  `N carried over` or `this launch only`.
- **fsync per row, measured at 1.98 ms** per journalled request here (0.018 ms without it — the version a hard
  reset eats). Requests finish seconds apart, this is nowhere near the decode path, rows are ~380 bytes, replay
  over a 2000-line file costs 3.4 ms, and the file rotates at 8 MiB down to the last `--keep` rows.
- **a torn tail line is dropped, not rendered.** A reset can cut a write in half, and a half-row that draws as a
  request is worse than no row. A journal that cannot be opened degrades to memory and the page says why, in danger
  tone — a broken journal must never take the serve with it.

Configuration: `--journal PATH` for the sidecar, `R9700_WEBUI_JOURNAL` for the plugin, and the launch sets
`~/logs/requests-<port>.jsonl` by default when the pages are on. **One file per port**, because two monitors sharing
a journal replay each other's rows — the test suite hit exactly that confusion, showing a request count about a
launch that had happened an hour earlier.

**Per-card waveforms: busy and VRAM replace the busy bar.** `gpu_poll` keeps its own ring
(`gpus_history`, same 90 points, one slot per card, aligned by PCI bus id / card name so a card vanishing from
sysfs cannot plot its neighbour's numbers), and each card draws busy on 0..100 and VRAM on 0..that card's own
total, captioned `busy peak 100 · avg 95% · 65/65 pts over 65 s`. Two honest readings, learned while wiring it:

- **busy % is coarse on this box, not a load meter you can read finely.** Same card set, measured across a day:
  refused (EBUSY) with nothing running, 0% after a launch stopped, 24-29% with the engine idle, 46-100% under
  three 600-token streams, and 95-99% for long stretches of MTP-heavy decode. The panel line carries that spread,
  because otherwise a flat 99 reads as "fully loaded" and a fall to 0 as "healthy idle" when either could also
  mean "the launch let go of the card".
- **VRAM on 0..total is the leak question with no script.** The soak runs used to log per-card VRAM to a side
  file to show flat is flat (`min == max` on all four cards); the wave is that evidence, always on, and the
  launch preallocates the pool so the expected shape is a step at boot and a line — anything sloping up is the
  finding.

`tools/cardwave-check.js` runs the card renderer under node against a live `/api/status` with a stub DOM, and
prints what it produced (five waves per card, no bar, the caption text; one text row per PCIe chain) — the browser
is still the real test, but the silent failure this catches is a panel drawing nothing at all.

`tools/watch-live.py` is the same idea for the whole page: it samples a running serve (engine gauges, live rates,
per-card sensors, PCIe chains, `/api/requests`) and prints percentiles, per-card sensor ranges and every AER event
with its timestamp next to what the serve was doing at that instant. It **never sends a request** — measuring
someone else's concurrency by adding your own would contaminate the numbers — which is why it reads the sidecar
when the in-serve build predates the event list.

**Power, temperature and the SM clock joined them** — labelled waves per card: `busy`, `VRAM`, `sclk`, `W`, and
**two temperatures** (`junction`, `mem`; see below). They are **stacked, not overlaid**: watts, degrees and megahertz share no axis, and three autoscaled series
in one box is the exact shape of lie a sparkline tells (a card pinned at its power cap and one idling at 205 of
500 W look identical). Ceilings are fixed per wave: busy 0..100, VRAM 0..the card's own size, sclk 0..observed
peak (this kernel exposes no max-frequency file), power 0..`max(power1_cap, observed)`, and **temperature
0..the card's own published `temp*_crit` (110 °C here)** — the distance to the wall is read off the card rather
than invented. One caption line carries the readings that are read by number rather than shape — live:
`junction peak 95 · avg 90 °C (crit 110) · mem peak 86 · avg 85 °C · power peak 442 · avg 293 W · sclk peak 3,286
· avg 2,876 MHz · busy avg 100% · 90 pts over 92 s`.

Colour means *near the wall*, and the wall is per sensor. **Junction** is coloured against its own `crit` (warn at
95%, danger at 98%) — measured, these cards sit at **90-106 °C junction under real load with crit 110**, so an
absolute "95 °C is a fault" rule would have screamed at normal operation. **Power is never coloured** (this box is
uncapped: passing `power1_cap` is the configuration). **Mem is never coloured** — sysfs publishes no memory
threshold on these cards, so any cut would be made up; mem is read against its neighbours instead, which is
exactly how its story was found.

**There are three temperature sensors per card, and reading the first one is nearly worthless.** Measured while
real traffic ran (180 s at 1 Hz, four cards each drawing 291-293 W): `edge` 68.0 / 67.8 / 66.2 / **77.3** °C,
`junction` **89.7** / **102.4** / 99.4 / 99.1 °C, `mem` 84.8 / 86.0 / 82.7 / **95.7** °C across cards 1-4. The card
the panel had been calling "10 °C hotter" was not hotter at the die at all — its junction equalled card3's — while
its **GDDR7 was 10-13 °C above the pack**, and `temp1` (edge) is the only one of the three that could not see it.
So the sensors are read **by label** now (`temp*_label` → `edge`/`junction`/`mem`), the headline is `junction`
(limit-bearing, and the sensor the fan map follows), `mem` gets its own wave, `edge` stays in the text line, and
the card's `crit` and fan (`fan1_input`, `pwm1`) ride along — fan duty matters because the loop follows the
junction, and card4 runs a *lower* duty (187/255) than cards 2-3 (192/193) while holding the hottest memory on the
box. A page newer than the serve it reads labels the headline from what the payload actually carries (`junction`,
`edge`, or a plain `temp`) rather than inventing a sensor the running build never read. The full thermal write-up
is in [observe-2026-11-02.md](observe-2026-11-02.md).

Three things that came out of wiring the thermal waves:

- **the sensors were being read from the wrong path.** The Monitor had shown a blank temperature, power and clock
  for weeks on a box whose files read fine: the hwmon directory here is `device/hwmon/hwmonN`, and the reader
  globbed `device/hwmon*` — the older layout — so it found the container directory and no files. Both forms are
  globbed now, and the test fixture carries one card of each so this cannot silently regress.
- **`power1_cap` is not the limit here.** Cap reads 300 W while the cards draw 232–494 W (measured over a 90 s
  ring) — this is the uncapped native box, so the axis takes the larger of cap and observed, and the cap stays in
  the text line as a contrast instead of clipping the wave.

The sensor set that answers **depends on what holds the card**: with nothing running, all four answered EBUSY and
only the VRAM files replied (57 MiB); with the serve holding a context, busy 99%, 61–72 °C, 232–494 W and 2.7–3.3
GHz all come back. Refused and zero stay different states in the code — a card that answers nothing is
unreadable, never "0% busy", and its wave breaks instead of dipping to idle.

One caveat on the KV-growth estimate, because it is a difference of a gauge sampled at the poll rate: it is good
on large prefills (5,787 against 5,459 measured) and can spike on small ones and at request boundaries — a peak
of **19,217 tok/s** in the ring right after a 62-token prompt is the pool being reclaimed and re-allocated
between two polls, not 19k tokens of work. That is why the caption reports peak/avg only over windows where the
pool grew, why the series is labelled "from KV growth", and why the exact counter figure is printed beside the
number whenever the request has landed.

**A token family is not a seconds family.** Cumulative means are `_sum / _count`, and for time families that
ratio is seconds, so it is multiplied by 1000 to print ms. Applying that scaling to `request_prompt_tokens`
made the first real `/api/status` report a **mean prompt of 2,028,442 tokens** — the answer is 2,028, and the
engine's context is 32,768. Scaling now applies only to time families; a test asserts a token mean of 500
stays 500.

**A window mean is undefined while a request is in flight.** vLLM observes a latency histogram when a request
**finishes**. Polling every 2 s, a 600-token generation shows `ttft_ms`, `queue_ms`, `prefill_ms` and
`decode_ms` as em-dash for its whole duration, because no observation lands in any of those windows — which
reads as a broken monitor rather than an honest one. The payload carries `since`, the same ratios over the
engine's whole run, and the page falls back to it marked "(since launch)":

```
ttft 237.6 ms · queue 0.01 ms · prefill 224.1 ms · decode 231.1 ms · prompt 2028 tok · accept 59.7%
```

A cumulative mean is not a window mean, so it is labelled as one wherever it shows.

**The KV pool's block label overstates capacity on a hybrid model.** The live launch reports
`num_gpu_blocks=1230, block_size=448, kv_cache_size_tokens="497588"` — and 1230 x 448 = **551,040**, 11% too
high. A hybrid KV has an attention part sized `blocks x block_size` and a mamba state sized per request, so
multiplying them lands on no axis tokens are counted on. The sidecar prefers `kv_cache_size_tokens` whenever
it is labelled, and falls back to the product with the older build's labels, where it is exact
(`block_size=1088, num_gpu_blocks=456` → 496,128 tokens).

Writing that script found a bug worth recording: the pump used `res.read(8192)`, and on a body with no
Content-Length `read(n)` **blocks until n bytes or EOF**. A 300-byte-per-second stream therefore reached
the client entirely at the end -- first token +70 ms on a 272 ms request, which would have made every
first-token number the page shows a lie. It reads `res.read1()` now, and a check in
`tests/test_webui.py` asserts the client's first bytes arrive while the upstream is still generating
(it fails with the old call: `first bytes at 0.48s of 0.48s`).


## Limits, on purpose

- The relay is a monitor, not a load balancer: SSE responses close the connection (`Connection: close`)
  rather than being re-framed, and one upstream connection is opened per request.
- No per-request queue/prefill/decode split (see above), no cancellation UI, no chat tab.
- `--proxy` measures token counts as *deltas seen* when the client did not ask vLLM for a `usage`
  block, and the detail pane says `≈ estimated` when it had to.
