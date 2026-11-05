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

**Two decode denominators, one identity, and no card gets to mix them (2026-11-04).** vLLM's two decode-seconds
families — `inter_token_latency_seconds` (one observation per stream-step, `v1/metrics/stats.py:551`) and
`request_decode_time_seconds` (one per request, at its end, `stats.py:601`) — are the same sum sampled twice:
24,236.14 s over 430,954 observations against 24,216.27 s over 1,812, on the same launch. Both are
**stream-seconds**, so tokens ÷ them is a **per-stream** rate, and a step two streams share is counted twice in
the denominator. It follows that the step interval contains the chunked prefill computed in that step (~1.0 s for
a chunk step, ~20 ms without one), so an ITL-shaped denominator excludes queue, prefill-before-first-token and
no-token steps — **not** prompt chewing, whatever a card's tooltip used to claim.

The card leads with **`generation_tok_s`: summed decode output ÷ wall seconds** -- the throughput reading, what a
second of this traffic produces -- and keeps the other reading labelled on the note line: `152 tok/stream-s ×0.15`,
i.e. per-stream-second × fan-out, which multiplies back to the headline exactly (`decode_streams` = stream-seconds ÷
wall seconds, a COUNT with no 100% ceiling: 0.15 serialized, 2.60 loaded). The per-stream figure is the same quantity
a request row prints (152 tok/s on a live burst whose rows ran 160-200), which is why it stays on the card instead of
disappearing into the payload. Its waveform is on the same summed-wall basis as the headline -- the card is labelled
`Decode · summed, trend` -- because the per-stream denominator does not exist in a window that was all prefill or
idle, and a line of gaps reads as an idle box. The identities the sidecar publishes and `tests/test_webui.py` pins:

    note-line tok/stream-s × decode_streams  ==  the summed headline
    152 × 0.15 = 22 (live burst, serialized)      17.3 × 2.60 = 45.1 (live launch, 15 min, traffic on)
    gen_per_step      ==  tokens_per_step × streams_per_record      42.5 = 2.47 × 17.2 records/s

A field named `…_tok_s` is per wall second; `…_stream_tok_s` is per stream-second; `decode_streams` /
`prefill_streams` / `streams_per_record` are counts that may exceed 1. The old pair of names — `decode_tok_s`
(a per-stream rate labelled "summed") and `decode_time_pct` (the same fan-out printed as 260%) — are gone from the
source and their names are grepped for by the tests, because a reader cannot tell a stream-second denominator
from a span denominator by looking at a number.

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

`tests/test_webui.py` (343 checks, no GPU and no network needed) and `tests/test_webui_plugin.py`
(32 checks; the record store and the stat logger run anywhere, the routes need FastAPI, and vLLM's own
route-replacement tier runs only where vLLM imports — `LD_LIBRARY_PATH=~/opt/openmpi/usr/lib/x86_64-linux-gnu
~/venv-r9700/bin/python tests/test_webui_plugin.py`). Three more cover the parts that are not the sidecar:
`tests/test_readme.py` (42 checks, stdlib alone: the README's card figure re-draws from the committed `/api/status`
poll string for string, no line runs off the canvas and none runs past the bottom of the card that drew it, no
wrapped line ends on a connector, the caption cannot state a number the figure does not show, the title lockup and
the release page sit where they belong and link to a picture big enough to read, and every block under Deploying it
holds one command so the copy button copies one command), `tests/test_lockup.py` (14 checks
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

So: decode rates are live (counter-derived from `generation_tokens_total`, summed over streams per second of span;
the per-stream figure divides by the ITL/request-decode sums, which are **stream-seconds** — see the block below),
prefill is not — the
number shown during a long prefill is the **KV gauge's growth in tokens a second** (`kv_tok_s`, measured within
~6% of the truth on that 51,859-token prompt: 5,787 estimated against 5,459 real, and negative when finishing
requests free blocks), with the exact counter rate appearing beside it when the request lands. Polls went from
2 s to 1 s on both beats (engine poll and browser refetch), and the page prints its own age (`1.0 s poll ·
snapshot 0.4 s ago`) instead of letting the reader guess.

**One caveat on the 1 s poll column, measured 2026-11-04.** The engine advances its counters on its own logging
beat, not on the poll's schedule, so a poll's counter delta can cover more engine time than the spacing between two
polls. Live during a six-request wave the poll line read `×12.34 decoding` with **8** requests running and 496.5
tok/s summed: the deltas covered ~1.5 s of engine time while `dt` said 1.0 s. What that does and does not break:
the **per-stream** rate (40.2 tok/s that poll, 71.5 the next) is a ratio of two deltas over the same interval, so
it is immune; the **summed** rate and the **fan-out** carry the same factor, so their product stays consistent and
the card's arithmetic still closes; the **spans** divide by the ring's own timestamps, so a 1% phase error on one
poll is under 2% over a minute and invisible over 15. Recorded because the fan-out briefly exceeding `running` is
not a bug in the fan-out — averaging stream-seconds over the interval the deltas cover is not the same as sampling
the gauge at one instant.

**A page that stops listening has to look different from a box that went quiet** (2026-11-03, asked as
"why have the Monitor's numbers stopped moving?"). Three ways that used to be indistinguishable, all fixed:

- **One stalled request killed the page.** The loop was `await refresh(); setTimeout(poll, 1000)`, and `api()`
  had no timeout — `fetch` waits forever by default, so the tick that stalled was the last tick there would ever
  be: the tab kept showing its last render indefinitely and looked like an idle machine. Every `api()` call now
  carries a 4 s deadline (`AbortController`, `FETCH_MS`), which turns a stall into one visible error and the
  next tick. Every call these pages make is a GET of a sidecar that answers in well under 0.1 s, so the
  deadline is 40x slack on a request measured in tenths of a millisecond.
- **The loop can no longer end.** The reschedule moved into a `finally`, so nothing — not a bug in the rendering
  of a new payload — can be the last tick.
- **A background tab is throttled**, not dead: Chrome clamps timers to about one tick a minute after five
  minutes hidden and can freeze a tab outright. That is the single most common reason the numbers "stop" while
  the serve is provably fine, so the page now says it: if no answer has arrived for 5 s the age line becomes
  `snapshot is 41 s old -- this tab is not getting answers (…the error…)`, and `visibilitychange` refetches the
  moment the tab comes back instead of waiting for the throttled timer.

While **Pause** is held deliberately, none of this speaks: a paused page is not stale, it is paused.
The serve side is unchanged — `/api/status` was already advancing (samples 1 s apart, `pcie_history` 90 deep),
which is how the question got answered before the fix: frozen counters plus a moving `at` means the box is idle,
not the page.

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

**A count that stopped moving must stop wearing the colour of a count that is moving** (2026-11-03, asked as
"the PCIe errors are not growing and the link is not degrading — can it be shown quieter?"). The row tone used
to be `degraded-while-busy || uncorrectable ? danger : any-correctable-since-boot ? warn : neutral`, so one
load window painted a chain amber for the rest of the boot. Measured on this box right then: the c6 chain sat at
**exactly 42 correctables for 42 minutes**, link at 32.0 GT/s ×16, after a load window whose last event was
09:00:03 — the colour had stopped carrying information.

- The sidecar now carries `quiet_s` per chain: seconds since *this watcher* last saw a counter move. It is a
  watch clock, not the fault's clock — a chain found with counters already raised is seeded at the first
  sighting, the same rule that keeps the first sighting from emitting events ("those errors happened before we
  were watching, so they are not ours to date"). A kernel zeroing of a counter counts as activity, not as a long
  calm spell, because it makes the window's delta unknowable.
- Tone follows activity: a nonzero delta **this window** (a one-poll flash that something just happened), a
  reset, a rate above 1/min, or an unknown rate keeps the colour; otherwise the row goes neutral and the count
  is muted. The window cell then reads `+0 · 0.33/min · quiet 4 min`. The number stays on screen — de-emphasis,
  not concealment, and "no errors" would be a lie about a chain that has 44.

**The rate replaced time-since-last-event as the criterion the same day, and the numbers say by how much.** The
first version dimmed a chain only after `quiet_s > 300`, with 300 s chosen from the weight-load windows
(11.3–45.7 counters/min). Then the box went back to real work, measured over the 24.6 min this sidecar has been
up: **47 requests at a mean prompt of 104,160 tokens (max 169,294), 1.91 requests/min**, and **2 AER events on
the slot-6 chain = 0.081/min**, with the two events 17.9 min apart (10:39:31, 10:57:23). What that implies for
the two rules:

| rule | coloured for | digits for |
|---|--:|--:|
| `quiet_s < 300` (first version) | 5 of those 17.9 min = **28% of the time**, and ~100% at the ~1-event-per-3-min cadence measured with 4–5 concurrent streams | always |
| trailing rate ≤ 1/min (now) | the one poll the counter moved (~1 s) | only while the rate is above 1/min, i.e. during a weight-load window |

So the time gate was not "always coloured" — with ~18 min gaps it dimmed for most of the time — but it re-lit
the row for five minutes after every single event, which at 0.081/min is the opposite of what the event's
rarity means. The rate says what is actually true: this chain adds a twelfth of a counter per minute.

| state | rule | what the row shows |
|---|---|---|
| `danger` | uncorrectable counters, or a link below maximum while the card works | unchanged, whatever the rate |
| digits + colour | movement this poll, or `rate > 1/min` (weight load measured **11.3/min**), or rate unknown | `44 corr · 0 uncorr` |
| `—` | no AER counter raised on this chain since boot (measured: slots 1 and 3, all day) | `—` |
| **info glyph** | correctables exist, nothing this poll, `rate ≤ 1/min` (measured **0.081/min** under 47 requests at mean 104,160 tokens; **~0.2/min** was the 4–5-concurrent-stream measurement) | `a drawn i-in-a-circle`; window cell `+0 · 0.33/min · quiet 4 min` |

**One limit the merged page cannot escape:** it starts with the engine, i.e. *after* the weights are up, so the
launch window -- historically where the bursts are (11.3-45.7/min) -- is outside the page's own clock. Measured at
the 11:05 restart: the page reported `events [], rate 0/min` while an external `reseat-check.py` diff over the
same window counted +4 correctables on the slot-8 chain. So the page's glyph is honest about what it watched, and
the launch window belongs to the external watchers (`tools/reseat-check.py --save/--diff`, `tools/brief.py`).

A chain with **nothing raised at all** is a bare `—` (the sentence "no errors since boot" spent four words on a
fact nobody needs reminded of; the tooltip still says what the dash means).

**The icon is drawn, not typed.** It started as `i`-in-a-circle (U+24D8) and arrived in the browser as a
missing-glyph box -- the body font ships a **Latin subset** (`fonts/outfit-latin-wght.woff2`) and nothing in the
fallback chain has that code point, so the thing that was supposed to whisper "there is detail here" instead
looked like a large question mark. It is now an ASCII `i` inside a CSS circle (`.p-glyph`: 1.35em, 1px
`--st-line` border, 50% radius, muted ink, `cursor:help`), which needs no font support. `tests/test_webui.py`
now enumerates every code point the pages emit (escapes and literals, across js/css/html) and requires it to be
Latin-1 or one of the punctuation marks these pages have rendered for weeks (em dash, curly quotes, ellipsis,
arrow, almost-equal) -- a font gap is otherwise silent.

Three guards: the glyph never applies while **uncorrectable** counters are nonzero; the numbers are still
there to read (see below) so nothing is lost, only demoted; and an **unknown** rate (older sidecar, first
sample) counts as loud.

## The account is a pinned block of text, not a tooltip

The demotion put the numbers in a `title`. That worked for *seeing* and failed for *using*: a tooltip cannot be
selected, cannot be copied, and disappears the moment the pointer leaves -- and this is the text people want to
**paste into a report**. So the count cell is now a `<button>` in every state that has numbers (the sparse glyph,
and the `45 corr · 0 uncorr` tally alike), and clicking it pins a block of real DOM text under the grid:

```
slot 6 · card2 · #3 in slot order · gpu c6:00.0            Copy  Close   Esc closes · the text is selectable
chain         c6:00.0 → c5:00.0 → c4:00.0 → c0:03.1  (gpu → root port)
link          32.0 GT/s ×16   (each port here is rated 32.0 GT/s ×16)  · at its rating
slot          registered at 0000:c4:00.0 (upstream port)  · 4 port(s) on this chain, gpu function at …
since boot    45 correctable · 0 uncorrectable · across 4 device(s) of 4 on this chain
correctable   BadTLP 30 · Timeout 13 · NonFatalErr 3
this window   +0 event(s)
rate          0.00 counters/min, measured over the 5 min this serve has been up
quiet         nothing has moved for 25 min
last event    no counter moved while this page has been open · anything before the engine started belongs
              to aer-watch and journal-pcie.py, not to this page
watched       90 sidecar samples · this page's clock starts after the weights are loaded
```

That block is what `node tests/render-monitor.mjs` prints after driving the page's own click handler -- not a
transcription of the code. Four decisions in it are load-bearing:

* **The panel lives outside `#pcie`.** The rows are replaced wholesale every tick, so a pinned element inside
  them would vanish a second after it opened; pinned state is one variable, `detailBus`.
* **The click is delegated to `#pcie`** for the same reason -- a handler attached to a row is garbage with the
  row. Clicking the same chain again unpins; `Esc` and a `Close` chip also unpin.
* **A repaint writes only the lines whose text actually changed** (`if (el.textContent !== t)`), because
  re-creating the block each second would destroy whatever text the reader has halfway selected. The harness
  counts this: it marks the pinned lines, ticks the page again with nothing changed, and requires **0 writes**
  (measured: `0 line write(s) across 10 pinned lines`).
* **Labels are padded and the block is `white-space:pre-wrap`**, so what is aligned on screen is aligned in what
  `Copy` puts on the clipboard -- no HTML between the reader and the numbers.

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
uncapped: passing `power1_cap` is the configuration). **Mem is graded against its own `temp3_crit` (108 °C,
emergency 113)** — lower than the die's 110/115, so grading memory by the die limit would be wrong in both
directions; each sensor's limits ride the payload (`temp_crit_c`, `temp_mem_crit_c`, `temp_mem_emerg_c`).
Memory is read against its neighbours as well as its limit — that is how its story was found.
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


### What a request row leads with, and what was dropped (2026-11-03)

The row used to lead with what the monitor knew about itself: a 29-character request id, a wall-clock, and
`engine record`. The two numbers a request is actually judged by — how fast its prompt was read and how fast its
tokens arrived — were buried at the end of a comma run, next to `POST …no endpoint in its stats`, a sentence every
engine row said and nobody acted on. A list of 100 rows is read by scanning, so the order is the design. It now
leads, at 21 px in tabular mono:

```
2:47:39 AM            shared ×2.1
prefill 6,875 tok/s   decode 29.5 tok/s
242,882 → 257 tok
first token 35.33 s · prefill 35.33 s · decode 8.70 s · all in 44.03 s · chatcmpl-ba6da5475870446b
```

Gone: the endpoint apology (the endpoint line now only exists when a row HAS one, which is the proxy path),
`engine record` / `stream` / `JSON` (how the monitor obtained the row is a fact about the monitor), and the state
tag on an ordinary row — **1,429 of 1,429 engine rows on this box read `completed`**, so the tag carried no
information until a row is NOT ordinary; the marks that remain are `preempted ×N` (with the re-prefill cost in the
tooltip) and `hit max_tokens` (ended on the client's limit, not on the model stopping). The id moved to last,
where it is still greppable against the server log.

**`shared ×N` is the field that makes a decode rate readable, and it is derived, not measured.** The journal
analysis in [context-262k.md](context-262k.md) says a stream runs **158 tok/s alone, 119 at two in flight and 23
at three or four**, so a decode number printed without its concurrency is not a speed at all — it is a speed and a
guess about the box welded together. `shared_decode(rows)` in `serve/webui.py` computes, for each row, the **mean
number of requests in their decode phase across that row's own decode span, itself counted**, and the tooltip carries both the
window it was counted over ("the N rows this page holds") and the ladder to read it against. Three rules it has to
get right, each with a test:

- **Mean across the span, not the value at its midpoint.** Midpoint sampling calls a request "alone" whenever
  nothing happened to be in flight at that instant — it reported 361 alone requests at 137 tok/s where
  never-shared gives 158.
- **The window is anchored at the END minus `decode_s`.** The two monitor paths do not share a clock origin: a
  merged (engine) row's `wallclock_s` is vLLM's `inference_time`, which **excludes** `queued_time`, while its
  `first_token_s` is `queued_time + prefill_time`. A real row: first token 21.84 s against a wall-clock of
  10.78 s, because it queued 12.32 s before the engine started. `start + first_token` therefore lands *after*
  `start + wallclock`, and a window built that way is inverted.
- **It is computed at read time and never journalled.** `ROW_KEYS` decides what reaches disk; a derived number in
  the journal would be a second copy of the truth, free to disagree with the first.

**The same mixed basis had the page lying about wall-clock.** The "Last wall-clock" card read the raw
`wallclock_s` and its note claimed *"From the engine: queue, prefill and decode together"* — for an engine row that
sum excludes the queue, so a queued request printed a first token later than its own whole span. Both the card and
the detail fact now add `queue_s` on the engine path (the proxy path already timed the whole connection), the note
says which sum it is, the detail label reads `Wall-clock (queue included)`, and the old sentence is asserted absent
in the suite.

Checking it needed the rows, and the rows leave the live serve's memory within a minute — so
`tools/monitor-fixture.py` builds `{"status": …, "requests": …}` from today's journal plus a GET of the running
status, and `tests/render-monitor.mjs` takes that JSON instead of a URL. The fixture runs the wide journal window
through the same `shared_decode` rather than stubbing the field, and it works with **the box down** (status falls
back to an empty payload labelled as such), which is precisely when those rows matter most.
`tests/test_webui.py` is at 240 checks.

## Limits, on purpose

- The relay is a monitor, not a load balancer: SSE responses close the connection (`Connection: close`)
  rather than being re-framed, and one upstream connection is opened per request.
- No per-request queue/prefill/decode split (see above), no cancellation UI, no chat tab.
- `--proxy` measures token counts as *deltas seen* when the client did not ask vLLM for a `usage`
  block, and the detail pane says `≈ estimated` when it had to.

## The page says what ended the previous boot

A hard reset takes the process, the journals and the page's memory of the launch with it, and the Monitor would
come back looking healthy. It no longer has to forget: the firmware keeps one record of the crash in ACPI BERT and
the kernel replays it at the start of the next boot, and `journalctl -k` is readable here without root. So the
Monitor now prints, under the request-journal line, what the last death was — severity, whether the machine check
was uncorrectable, which block the record names, which CPU and physical core, and when — with a second line giving
how many such records the journal still holds and who each of them blamed.

The three pieces worth naming, because each one is where a naive version goes wrong:

- **`smca_bank()`** resolves the record's `IPID` to the block it names, using the same mask arithmetic as the
  kernel's `smca_hwid_mcatypes` table. Without it the page shows `Bank 5`, which means nothing; with it, "the
  execution unit" — and the same lookup is what keeps a memory-controller record (`0x96`) from being described as a
  core fault.
- **`core_of()`** turns a logical CPU number into its physical core and SMT siblings. On this box that is the whole
  argument: `CPU 17` and `CPU 113` are siblings of `core_id 65`, so 18 records name one core, not eighteen victims.
- **`death_history()` reads one boot at a time.** `-b all` on this box scans 517 MB of journal and measured **55 s**;
  one boot measures 0.4 s. The count runs in a thread so the page comes up immediately and gains the tally a second
  or two later.

## Which card is which, and what the page does when the serve is not there (2026-11-03)

### The slot map, from the machine rather than from a name

`card1..card4` are `amdgpu`'s enumeration names for this boot. They are not chassis facts, and after a card is
pulled out and put back they can name a different card. `/sys/bus/pci/slots/` (firmware's slot table, readable
without root here) gives the chassis fact, and the join is subtle: **the slot is named at the card's switch
upstream port, not at the GPU endpoint**, so looking up the endpoint's own BDF finds nothing.

| Monitor name | GPU BDF | physical slot | chain (endpoint → root port) | this boot |
|---|---|---|---|---|
| card3 | `0000:03:00.0` | **1** | `03:00.0 ← 02:00.0 ← 01:00.0 ← 00:01.1` | 32.0 GT/s ×16, 0 corr |
| card1 | `0000:c3:00.0` | **3** | `c3:00.0 ← c2:00.0 ← c1:00.0 ← c0:01.1` | 32.0 GT/s ×16, 0 corr |
| card2 | `0000:c6:00.0` | **6** | `c6:00.0 ← c5:00.0 ← c4:00.0 ← c0:03.1` | **uplink 16.0 GT/s**, 144 corr |
| card4 | `0000:06:00.0` | **8** | `06:00.0 ← 05:00.0 ← 04:00.0 ← 00:03.1` | 32.0 GT/s ×16, 56→59 corr |

So the card worth reseating is **slot 6**. Both PCIe rows and card rows now print it: `card2 · PCI c6:00.0 ·
slot 6`, and the chain row shows `slot 6` as a chip before the BDF path.

`serve/webui.py` gained `read_slots()` (the table) and `slot_of()` (first node of a chain the firmware names),
and `gpu_poll` copies the slot onto each card row by joining chain → card, since neither side alone has both
the GPU's BDF and the slot.

### `tools/reseat-check.py` — snapshot before pulling, diff after booting

```bash
~/venv-r9700/bin/python tools/reseat-check.py --save /tmp/reseat-before.json   # before
# power down, reseat, boot
~/venv-r9700/bin/python tools/reseat-check.py --diff /tmp/reseat-before.json   # after
```

It prints per card: GPU BDF, slot, every chain node with negotiated vs max speed/width and power state, and the
AER totals. `--diff` reports a card that is gone, a card that appeared at a new BDF, a **slot change**, a chain
change, and per-node speed/width changes. Two caveats printed with it: AER counters are per boot (so totals
across a reseat-boot are not comparable — growth inside one boot is the signal), and negotiated speed is all
sysfs gives an ordinary user; the BIOS's *target* speed needs root (`lspci -vvv` answers `access denied` here).
It reads sysfs only — no request to the serve — because a reseat happens when the serve is down anyway.

### The standalone Monitor is the page that survives, so it now carries what the serve would have said

Observed 2026-11-03 04:36: the serve stopped answering on `:8080` for minutes while its pid lived on
(requests were still completing a minute later), and the in-serve Monitor — the page people actually have
bookmarked — went away with it. Three changes:

- **Serve-down banner.** When `reachable` is false the page states the error, prints the command that would
  bring *this* launch back (`launch_hint()` assembles it from the launch's own meta file:
  `MAXLEN=… NSEQ=… NBT=… PREFIX_CACHE=… WEBUI=same bash serve/native-flashnext.sh`), and says why the rest of
  the page still has content.
- **`--journal-attach`** reads the merged monitor's journal **read-only**: replay for display, no append handle,
  no fsync, no rotation, so two Monitors can never replay each other's rows. `tools/restart-sidecar.sh` now
  passes it plus `--meta` by default, which is what let the sidecar show 100 of the dead launch's rows,
  including the 242,882-token request, while `:8080` was dark.
- Two traps found while doing it, both caught by the render harness rather than by a human:
  `launch_hint` was being called **before** the meta file was read (page showed `launch: None`), and the new
  `const where` sat **below** its use in `gpuCard` (TDZ `ReferenceError`, swallowed by the page's `try/catch`,
  blanking the whole Cards panel). The harness prints `PAGE NOTICE` and fails the suite for exactly this class.

## The panel is ordered by the chassis, not by enumeration (2026-11-03)

Decision: **the Cards grid, the PCIe rows and the mapping line are all in physical-slot order**, numbered
`#1..#N` by `pos`. Reason is operational, not cosmetic — the instruction that came out of the incident was
"reseat slot 6", and a panel that lists cards in `amdgpu` enumeration order makes someone translate that
translation by hand while standing at an open machine.

Three rules the implementation keeps:

- **`cardN` is never dropped.** It is the key the sensor history ring and the chain rows are aligned by
  (`at = (s) => …c.bus === g.bus || c.card === g.card`), and renaming it mid-session would mis-plot every
  waveform. So the headline is the slot and the driver name sits underneath:
  `slot 6` / `card2 · #3 in slot order · PCI c6:00.0`.
- **Slot sorting is numeric.** The firmware's names on this box are `1, 1-1, 2, 3, 3-1, 4, 6, 8, 10`; plain
  string sort puts slot 10 second. `slot_key()` parses the number and any sub-slot, and a card with no slot
  name sorts **last** rather than into the middle of chassis order.
- **One order, computed on the server.** `order_by_slot()` runs in `gpu_poll` over both lists, so the API, the
  Cards panel, the PCIe panel and the About page cannot disagree about what "#3" means.

The mapping line spells both numbering systems out in one row:

```text
chassis order  firmware slot numbers, not driver names
  #1 slot 1  card3  GPU 03:00.0  32.0 GT/s ×16
  #2 slot 3  card1  GPU c3:00.0  32.0 GT/s ×16
  #3 slot 6  card2  GPU c6:00.0  16.0 GT/s ×16 of 32   144 corr
  #4 slot 8  card4  GPU 06:00.0  32.0 GT/s ×16          63 corr
  the slot is registered at the card's switch upstream port (01:00.0 for slot 1), not at the GPU endpoint
```

Two bugs found on the way, both the kind that hide: a test asserted `s["gpus"][0]["busy_pct"] == 87`, which was
only true while the order *was* enumeration order (now it identifies cards by name, as the history ring already
does), and `max_speed` was built with webui's `_num()` helper, which answers "is this a number" — so the field
the page prints as a speed carried `True`.

## The PCIe row grid: seven columns, and the widths are now arithmetic (2026-11-03)

**What was actually broken.** Adding the `slot` label made every row emit **seven** children into a
**six**-track grid. CSS grid does not complain: the extra child starts a second grid line, so the
`+0 this window` delta wrapped under the row and the remaining columns looked ragged. The symptom was
"columns are the wrong width"; the cause was a child with no track for it — which is why the harness now
asserts each row fills all seven areas instead of trusting that it looks right.

**Layout.** Named areas, shared by the header row and the data rows, so a legend and its columns cannot drift:

```css
grid-template-areas:"slot drv chain link cnt who win"
```

**Track widths, sized from the longest live strings at the theme's own sizes** (`em` tracks resolve against the
row's 16px; `chain`/`who` are 11px mono, `link`/`cnt`/`win` 13px mono, ~0.6em per glyph):

| column | longest string the page now renders | px needed | track |
|---|---|---|---|
| `chain` | `c6:00.0 → c5:00.0 → c4:00.0 → c0:03.1` (37) | 244 | `minmax(16em,1.7fr)` = 256 |
| `link` | `16.0 GT/s ×16 (idle, max ×16)` (29) | 226 | `minmax(14.5em,auto)` = 232 |
| `cnt` | `144 corr · 0 uncorr` (19) | 148 | `minmax(9.5em,auto)` = 152 |
| `win` | `+12 · 45.00/min · quiet 4.2 h` (29) | 226 | `14.5em` = 232 |
| `who` | `corr BadTLP 30 · Timeout 13` (27) | 178 | `minmax(11.5em,1.5fr)` — **wraps on purpose** |
| `drv` | `unnamed · #12` (13) | 86 | `7.4em` = 118 |
| `slot` | `slot 6` (16px regular) | 58 | `5.2em` = 83 |

Those strings are what `node tests/render-monitor.mjs` printed against the live serve, and the arithmetic is now
asserted rather than written down: `tests/test_webui.py` parses the tracks out of `monitor.css`, computes
`glyphs x 0.6 x font-size` with the sizes that file itself assigns (chain/who 11px mono, link/cnt/win 13px, slot
at 16px), and fails if any track is narrower than its string or if the seven minimums plus the gaps exceed the
breakpoint. **The widths moved because the content moved:** `cnt` shrank from 15em to 9.5em (a quiet chain is now a
`a drawn i-in-a-circle` or a `—`, and the redundant per-row "since boot" — the header says it — left the cell), and `win` grew from
8.4em to 14.5em because it now carries the rate and the calm span (`+0 · 0.00/min · quiet 7 min`).

Minimum single-line row ≈ 1258px + 60px of gaps = **1318px**, so the single-line layout is claimed above 1360px:
two-line (`slot drv chain link` / `cnt cnt who win`) below that, and a three-line form below 900px. Only
`chain` and `who` flex, because a counter list may wrap and a BDF path or a link speed may not.

**Not verified as pixels.** There is no browser on this box (no chromium/firefox/playwright), so these numbers
are arithmetic against the theme's font sizes, not a screenshot. What *is* checked automatically: 7 tracks =
7 areas, both narrow layouts keep all 7 areas, the header row exists on the same grid, and the live render fills
all 7 areas in every row (`tests/render-monitor.mjs`, exits non-zero otherwise).


## The header says which build is answering (2026-11-03)

The question came from the box: "is what is live right now the version I just merged?" Answering it meant an ssh, the
tmux session's launch log, and a hash compared against `git log`. `stack_info()` in `serve/webui.py` reads it once at
Hub start — the plugin's `__version__`, and from the tree that plugin was imported from: branch, short commit, and
whether `git status --porcelain` says anything — and `/api/status` carries it as `stack`:

```
{"plugin": "0.4.0", "branch": "master", "commit": "953a3901", "dirty": false}
```

`brandStack()` in `serve/web/common.js` puts it in the header of both pages — `4× Radeon AI PRO R9700 · vLLM 0.32.0 ·
plugin 0.4.0 · 953a3901`, with `*` on a dirty tree and the branch/sha/clean-or-dirty in the `title` — and About
repeats it as a row of its own. The parts that are not obvious:

- The Monitor repaints the header every poll, so the helper remembers the element's base text and rebuilds from that.
  The first draft appended, which would have read `vLLM 0.32.0 · vLLM 0.32.0 · …` a minute after anyone looked; the
  harness now waits past the 2 s poll and fails if the line grew at all.
- The engine version arrives as a sentence: `vllm 0.32.0 (this process)` from the merged page's own registry, or
  `vllm 0.32.0+rocm723` when the build carries a local tag. The header takes the version token and labels it once;
  About keeps the sentence, because there "(this process)" is the point.
- Fields are absent rather than guessed. No importable plugin (a sidecar running outside the serve's venv) or no
  `.git` next to the code (an installed wheel, not a checkout) drops that part, and the pages print what is there.
- It rides the engine-unreachable payload as well. "Which build is this?" is most useful when the engine is not there
  to answer it, and the merged page is the one that stays up through a stopped launch.
- What it does not track: the static files are read per request, so an edited CSS/JS reaches the browser before the
  header moves. `stack` names the tree the process IMPORTED at start; a watcher of the tree as of now would be a
  livelier and less useful fact — it would describe files, not the running build.

Checked: `tests/test_webui.py` asserts `stack_info()` names the plugin version this tree declares and the checkout's
commit, that both headers carry the id the line is written into, that both pages call the one helper, and that About
has the row; `node tests/render-monitor.mjs` reads the status the live page is fed and fails if the rendered header
does not carry `vLLM <version>` and `plugin <version>`, then waits a poll and fails if it grew. About's row is checked
at text level only — there is no render harness for `about.js`, and no browser on this box.

## The trailing averages: 1 min / 15 min / 1 h beside the 1 s window (2026-11-04)

Asked from the box, with six streams in flight: "decode and prefill are how fast? the Monitor is all realtime, it
does not tell me the actual average level." The measurement the page was giving, and the measurement the question
wanted, are not the same number:

| what was read | generation | prefill | one stream |
|---|---|---|---|
| one 1 s poll, six streams running | 9.9 tok/s | 0 (counter credits at request end) | 890 ms a step |
| another 1 s poll a minute later, drained | 0 tok/s, running 1, KV 3% | 0 | - |
| the ten minutes around them (journal) | 41-43 tok/s across the span | 6,967 tok/s across the span; 7,306 per request | 20.2 tok/s median |
| the whole launch (523 min, 1,189 requests) | 21.6 tok/s | 2,232 tok/s | - |

So the window answers "right now" and the since-launch block answers "since the engine started", and the question
lives in between. `/api/status` now carries `avg` with spans `1m`, `15m`, `1h`, and each rate card prints that line
under its live number: `avg 1m 54 - 8m 56 tok/s all streams`.

How they are computed (`trailing()` in `serve/webui.py`): the Hub keeps a second ring of the cumulative counters,
sampled every 5 s (`AVG_STEP`) for `AVG_KEEP` points, so a span is a **delta of cumulative counters over the delta
of time**, and a span latency is `delta(sum)/delta(count)` of the engine's own histogram families. Averaging the 1 s
samples the sparkline keeps would have been easier and wrong: it weights a second with nothing running the same as a
second with six streams running, and on a launch that spends a quarter of its wall time prefilling, plenty of seconds
are gaps.

Per span the payload carries: `generation_tok_s`, `prompt_tok_s`, `prompt_read_tok_s`, `requests_per_min`,
`stream_tok_s` (the summed rate over the **mean** running gauge of the span — the approximate one, kept for the
tools; it counts a request sitting in a chunked prefill as a producer), `decode_stream_tok_s` (÷ stream-seconds),
`decode_streams` and `prefill_streams` (stream-seconds per wall second: counts, may exceed 1),
`streams_per_record` (ITL observations ÷ records), `prefill_stream_tok_s` (read prompt tokens ÷ request-prefill
seconds; `prefill_stream_tok_s_credited` keeps the cached-in version beside it), `decode_time_s` /
`decode_time_req_s` (the two families' stream-seconds, kept so the agreement can be seen), `itl_ms`/`tpot_ms` /
`ttft_ms`/`queue_ms`/`e2e_ms`/`prefill_ms`/`decode_ms`, `prompt_tokens_mean`/`generation_tokens_mean`,
`tokens_per_step`/`gen_per_step`/`records`, `accept_pct`/`accepted_per_draft`, `prefix_hit_pct`, and
`running_mean`/`waiting_mean`/`kv_used_pct_mean`/`kv_series`.

What it refuses to do:

- **A young span keeps its own name.** `span_s` says what was averaged, and the page relabels a span that has not
  filled (`90s`, `2m`) rather than printing `1h` over 40 s of data. A ring younger than 15 s (`AVG_MIN`) reports
  None: a dash is cheaper than a number wearing a label it has not earned.
- **A counter that went down is a restart, not a rate.** Any span whose generation/prompt/draft counters decreased
  comes back all-None with `reset: true`; the alternative is a spectacular negative number, and the page prints
  "1m engine restarted".
- **A family the base snapshot did not have is a dash**, not a division by the whole cumulative counter (an older
  vLLM gaining a family mid-run would otherwise produce an enormous mean). `tests/test_webui.py` feeds the ring a
  `cur` with every family and fails if any documented field comes back None, which is how the missing
  `request_prefill_time_seconds_count` got caught while writing this.
- **The histogram-backed fields are credited at request end**, so a span shorter than the typical request's wall time
  is a mean over the requests that happened to finish inside it. Measured: a 47 s span reported `prefill_tok_s`
  4,840 while ten-minute spans of the same launch report 7,000-8,200, and the journal agrees with the longer spans.
  The counter-derived rates (`generation_tok_s`, `prompt_tok_s`, `requests_per_min`) have no such bias, because
  those counters move every step.
- The sparklines still span 90 s (`HIST` x poll). Longer trends would be a second chart, not a longer sparkline.

Seen while writing it: this launch is prefill-dominated in the extreme - 70,015,813 prompt tokens against 678,007
generated over 523 minutes, 2.3 requests a minute, mean prompt 57,275 tokens. "Decode speed" on this box is therefore
mostly a question about concurrency: the same engine does ~200 tok/s to a lone stream and ~20 tok/s per stream when
six are running and long prefills are interleaving between them. A single number on a card cannot hold both, which is
why the averages sit next to the live number instead of replacing it.

Tested: `tests/test_webui.py` (12 checks on a synthetic ring with exactly known counters - span rates, the two
token-weighted speeds, mean-concurrency division, histogram means, the pool mean, restart, young ring, partial ring,
missing family, payload shape) and `tests/render-monitor.mjs`, which now prints the five `avg` lines and fails if a
card claims an average the payload has, or prints `1h` over a span of 40 s. Verified against a live serve by running
the standalone sidecar (`serve/webui.py --port 8099`) next to the running one - the merged page picks the same code up
at the next restart of the serve, which was not restarted for this.

### The headline defaults to the quarter hour, and the waveforms gain an hour (2026-11-04, same day)

The averages landed next to the live number, and the live number stayed the headline. That was half the fix: a
reader looks at the big number, and on this box the big number was still a 1 s coin toss. So the four rate cards now
answer to a **window picker** (this poll / 1 min / 15 min / 1 h, default **15 min**, kept in `localStorage`), and the
four waveforms pick between two rings (90 points at one per poll, or 240 points at one per 15 s for an hour).

Why 15 min rather than 1 min: a request here runs 17-50 s of wall time, so a 1 min window is still largely a sample of
one or two requests. Measured in the same minute, `1m` said 30 tok/s and `15m` said 52, and the poll at that instant
said 13. The poll is never removed: every card's second line carries it (`poll 13`), because "is it stuck right now"
is a question a quarter-hour mean cannot answer -- and a step of 905 ms in the poll against 103 ms over the span is
exactly the kind of thing that only the poll shows.

The rules the picker follows:

- **A span has to have filled.** A window counts as usable at 80% of its width. Ask for 1 h on a sidecar that has been
  up for nine minutes and you get the widest span that HAS filled, and the note says so: "1 h has not filled yet -
  showing 1 min (65 s of counters)". The failure mode it prevents is an hour's label over nine minutes of ring.
- **Every span stays labelled with what it averaged.** Young spans print `90s`, `2m`, never the name they asked for,
  and the same rule runs in the second line.
- **The caption span comes from the samples' timestamps, not `points x poll`.** That formula was true of one ring and a
  lie about the other: the two are spaced 15 s differently.

The hour ring (`slow_point`, `SLOW_STEP = 15 s`, `SLOW_KEEP = 240`) takes its rate **over the interval** - generated
and prompt tokens differenced against the previous point, KV growth the same way - because a 1 s window sampled once
every 15 s would draw a heartbeat and miss the work between the beats. Its per-stream rate divides by the mean running
across the interval (taken from the fast ring, which has a sample every poll), not by whatever this poll saw. A
counter that went backwards inside the interval gives None, so the waveform draws a gap rather than a spike of
-100,000,000 tok/s; the KV-growth series is deliberately NOT guarded that way, because a negative there is the pool
draining as requests finish, which is what that series is for.

Checked by `node tests/render-monitor.mjs`, which now works out which window the page should have settled on, pulls
the first number out of each headline and fails if it is not that window's value, fails if a headline prints this
poll under a 15 min label when the two differ by more than 25%, fails if a card whose headline is an average does not
carry `poll <value>`, and presses both trend buttons and fails if the source ring does not change. The handlers are
attached by id (`$("win-15m")`) rather than by a `querySelectorAll` sweep, because the stub DOM has no
`querySelectorAll` - a control the harness cannot press is a control it cannot check.

One test bug fixed on the way: the pinned-PCIe-panel check asserted "repainted with nothing changed" without checking
whether anything had changed. The panel's own content moves between repaints (`quiet 7 min` becomes `quiet 8 min`, an
AER counter lands), so rewriting those lines is correct, and the check failed twice against a live box for the right
behaviour. It now counts writes that wrote **the value already there** and fails only on those, which is what
"rewrites lines it did not have to" was trying to say.

## A number changing width must not move the page: every note line becomes a slot (2026-11-04)

The user's report: **数字长度变化时可能造成换行变化** - when a number's length changes, the wrapping changes, and
the page moves. It is real, and these pages have three places where it happens on every poll.

1. **The note line under a headline.** It carries an explanation plus one or two numbers, and the numbers change
   width every poll: the stream card's number was `2.0` for four polls and `95.7` on the fifth. A note line that
   wraps turns one line into two and the card's height changes.
2. **A long prose note under a card** (`exact, from prompt_tokens_total; the other number is estimated from KV
   growth`). These were built to be long, so they were guaranteed two lines at narrow widths and their exact
   break point depends on the digits in the sentence.
3. **A whole-card reflow.** `grid--rates` uses `repeat(4, minmax(0, 1fr))` and `grid-auto-flow: dense`. A card
   that grows makes its row taller, and every card in that row resizes; `dense` then has room to move cards into
   different columns. The row does not just grow, it *rearranges*: `tests/test_readme.py` pins
   `grid-auto-flow:dense` in monitor.css precisely because reordering these cards under a reader is a bug.

The theme already had one of the two tools. `tokens.css` defines `--st-font-mono`, the JetBrains Mono fallback
chain the release page's own font stack uses, and `web.css` puts it on `.value` with `tabular-nums` - so the
**headline** digits already advance at a fixed width. What it lacked was any rule about the note lines and, worse,
a font mismatch: `.small` and `.winbar` carry no font-family, so the 12 px text under a headline renders in
Inter while the headline above it renders in mono.

### The fix: fixed slots

Every number-bearing note line in the rate row, the trend row and the KV pool card is now a **slot**:

```css
.slot{display:inline-block;min-width:2.5ch;max-width:100%;white-space:nowrap;overflow:hidden;
     text-overflow:ellipsis;font-variant-numeric:tabular-nums}
```

**No slot may wrap, whatever it holds.** A slot's height is one line always, so the card's height never moves, so
the row never moves. Overflow is ellipsized and the full sentence lives in the element's `title`.

The shape that came out of it:

- Rate card = headline + **three slots**: why/which, the other spans, this poll.
- Trend card = wave + **two slots**: the level (`now`/`avg`), the spread (`peak`/`pts`/`span`).
- KV pool = headline + **two slots**: size · used · span, and the spans' means.

The old classes (`small--avg`, the long inline notes) were deleted outright. A line that is allowed to wrap
anywhere is the bug; a slot that ellipsizes is a decision.

### The budget is arithmetic, and the harness knows it

The same grid arithmetic `docs/monitor-cards.py` and `tests/test_readme.py` use: main is capped at 1560 with
32 px of padding, 14 px column gaps, and each card gives up 42 px to padding and border. Text is laid out at
**0.60 em per glyph** - the factor the release page's label-fit check has used all along, and the right one for
`--st-font-mono`.

| viewport | 4-up card, inner | 2-up card, inner | chars at 12 px (4-up / 2-up) |
|---|---|---|---|
| 981 px (the narrowest a 4-up row exists) | 177 px | 416 px | **24** / 57 |
| 1100 px | 219 px | 483 px | 30 / 67 |
| 1560 px (capped) | 354 px | 706 px | **44** / 98 |

Two things followed:

- `dnum()` prints a number as a number: digits grouped by thousands, decimals only below 100, and an integer
  printed without a `.0` (`0` is `0`, a gauge reading of two streams is `2`).
- `tests/render-monitor.mjs` reads the slot list out of `monitor.html` (so it cannot drift from the markup),
  prints what the live page renders into each one, and fails on any slot over the **widest** budget (44). It
  flags any slot over the **tightest** (24) as wide. Writing the copy to fit 24 characters is what makes the
  ellipsis something the reader should never see.

**Digits, not abbreviations (2026-11-04).** This function used to be `knum()` and capped a changed number's width
instead: `999.9`, `16.2k`, `1.01M` - never more than five characters, so a digit count could not change the length
of the sentence around it. Asked to put the numbers back - "prefill 那里" was showing `7.4k` for a rate of 7,428,
which hides the very figure the tooltip next to it explains - the abbreviation went away everywhere on the page:
`1,009,899 tok · 16%`, `vs 4,074,190 in`, `wall 4,316`, `poll 108 · 1.8 req/m`.

The width problem was real, so it did not just disappear: it moved out of the formatter and into the **copy**, where
the harness can measure it. Three lines gave up their tail to pay for it -

| slot | was | is |
|---|---|---|
| `kv-note1` | `1,009,899 tok · 37% · 90s` | `1,009,899 tok · 37%` (the ring's span is in the trend caption already, and in the title) |
| `*-note2` (the four waveforms) | `peak 22,886 · 74/90 · 90s` | `peak 22,886 · 74/90` (25 characters was over the 24 budget) |
| `note-gen` | `145 tok/stream-s ×0.89` | the per-stream reading, with the fan-out that multiplies it back to the summed headline; the pair drops out together when the window has no decode stream-seconds, and the slot says `no decode step` instead of printing a rate with no denominator |

**When the digits win the argument, a span is dropped -- never a digit (2026-11-04).** `spans-gentot` grew into the
24-character budget on its own, by arithmetic nobody argued about: two spans of a counter delta in five-digit numbers
(`15m +79,766 · 14m +79,766`) is already 25 characters once the digits are written out rather than abbreviated. Two
fixes, in order, both in `spanSigned()`: the `+` went, because a cumulative counter's delta cannot fall inside a
window without the engine having restarted and this same slot reporting it as `reset` -- the sign was decoration, and
the harness already forbids a bare signed number in a slot; and when the line still overflows, a **span** is dropped
and named in the title. What goes first is a span repeating another (a launch 16 minutes old has a 900 s "15m" and a
960 s "1h" holding the same count -- two labels on one window), and after that the widest, because the card's big
number IS the since-launch total. `tests/render-monitor.mjs` allows those two omissions and only those: a span whose
window is its own and whose line would have fit, vanishing, is still BAD.

**A fan-out has to be printed to be checked (2026-11-04).** The render harness multiplies the two figures on the
decode card's note line and compares the product with the headline, and it failed on real data: a fan-out of 0.0168
printed as `x0.02`, so the card showed `189 tok/stream-s x0.02` under a headline of `2.8` -- a reader doing the
multiplication gets 3.8 and concludes the card is wrong, which is the same class of failure as a wrong denominator,
just harder to see. `fanS()` now prints three decimals below 0.1 (two up to 10, grouped above that); the extra digit
costs one character and the harness's width check confirms the slot still fits the 24-character column.

Two invariants are checked by `tests/render-monitor.mjs` against the **rendered text** rather than against which
helper the page calls, so they hold whatever produces the string: no slot may print a `k`/`M`
abbreviation (`/\d(?:[.,]\d+)?[kM]\b/`), and no slot may show a bare signed number. The second one is not
hypothetical - the prefill waveform is drawn on the KV pool's growth, which goes negative when finishing requests
free blocks, and its caption used to read `now -172,106 · avg 7,939` under a card labelled prefill. It now reads
`now draining 172,106`, with the average moved to the title (both together are 32 characters against a budget of
24); the sign is still in the drawn waveform, which autoscales through zero for exactly this reason.

**The page could not answer "are the caches in effect?" (2026-11-04).**

**What the payload now has to say about the cache (2026-11-04, the cache-on launch).** Two more of vLLM's
`cache_config_info` labels are read: `hash_block_size` (that is `prefix_match_unit`; `None` means a hit can only land
on a whole physical block, 832 tokens on this launch) and `enable_mamba_shared_prefix_checkpoint` (the fork's knob for
resuming an MTP sibling at a shared-prefix junction; off). About renders the pair as one line - `on · a hit lands on a
832-token boundary` - rather than a bare yes/no, because yes/no was the question that could not distinguish "on and
missing" from "on and structurally unable to hit". One value in the payload is a trap and is labelled as one in the
source: `mamba_block_size` reads **16** while the pool's block is 832, because vLLM's hybrid hook copies
`cache_config.block_size` into it before the platform raises the attention block size (boot log: "Setting attention
block size to 832 tokens"), so 832 / 16 is not a checkpoint count. `mamba_cache_mode` is not in those labels at all;
the boot log's `Mamba cache mode is set to 'align'` is the only place the engine states it, and the page does not
claim it. The prefill note line printed a cache
percent only when the engine had served hits, which is right (`0% cache` on a `PREFIX_CACHE=0` launch is a
permanent nothing), but a *missing* clause reads identically whether the feature is off, on-and-missed, or
unmeasured. The engine publishes the state itself - `config.enable_prefix_caching`, the same string its boot line
prints - so the line now says `wall 6,691 · cache off`, and says nothing when a build publishes neither the config
nor the counters, which is the only case where silence is honest. It shares the slot with the fan-out clause rather
than both printing: `wall 4,316 cache off` fits the 24-character budget, `wall 4,316 cache off ×2.0 prefilling`
does not. The tooltip carries the whole sentence and the counter that proves it (`prompt_tokens_by_source` with
`local_cache_hit` at 0). `tests/render-monitor.mjs` asserts the agreement against a live payload: config says off →
the line must say so and must not print a percent; config says on → it must not claim off.

And for whoever reads the crash logs next: the compile cache is keyed by config, so a test config leaves a second
key behind. `~/vllm-native-cache/torch_compile_cache` holds `a2db778b4c` (the production recipe, 11-03) and
`3ed99c1a7e` (the `PREFIX_CACHE=1` boot, 07:30 the next morning) - 4.2 MB, 16 files each. The relaunch after the
revert wrote **nothing** into that tree and logged `reconstructed serializable fn from standalone compile
artifacts. num_artifacts=50 num_submods=50` on all four ranks: warm reuse, 117 s to `init engine` against 181 s on
the cold boot the day before. Reverting a config is cheap here, not free, and residue like this belongs in the same
note that records the revert.

A wording bug the digit change exposed: the prefill card's this-poll slot fell back to `poll 1s` when nothing had
been credited, which reads like a duration wearing a unit. It says `poll · nothing in 1s` now - the same fact, with
the fact that it is an absence stated in words. The `%` on the pool's spans line had the same shape of bug: it was
appended to the whole line, so a span that has reset rendered as `15m reset %`. Units belong to each number
(`15m 14.7% · 1h reset`), because a unit on the word `reset` is a unit on nothing - and the step card's this-poll
slot, which printed `poll — ms` when the window held no step at all, now prints `poll · no step` for the same
reason: a unit after a dash claims a measurement whose value the reader cannot see.

`kv-note2` was 31 characters - `1m 25.9 · 15m 30.6 · 13m 30.6 %` - and is now `15m 30.6 · 13m 30.6 %`: the pool's
headline *is* its current value, so a one-minute mean of it is the same number rounded.

**What this check is not.** There is no browser on this box, so nothing here lays the page out and measures the
pixels. The guarantee rests on two things instead: the arithmetic above, and the fact that a slot *cannot* wrap
by construction - `white-space:nowrap` makes overflow a clipping event, not a reflow. The worst case the budget
can get wrong is a clipped word, never a moved card. That is the difference between this and the release page's
figures, where the same char metric is used to fit text exactly and a mistake shows as overlapping glyphs.

### What this made visible, and one clause nearly lost

Shortening to fit is not free: the stream card's note used to read `× 1.30 tokens a step`, which is the bridge
between "83 tok/s across all streams" and "118 ms a step", and it left when the line became a slot. It came back
as `poll 890 ms · ×2.33` - five characters - with the explanation ("one engine step turns into that many tokens
with speculation", and, when the window holds no per-token observation, "since launch") moved into the title.
Same for `pts`, the axis range and the sampling interval on the trend row.

A related loss to keep honest: `· 90s back` reads as "90 seconds ago" when the label means "this window spans 90
seconds", so the span now goes where the window label lives - `15 min (150 s)` in the window bar - and the wave's
axis range went to the title with it.

### Kept by tests

- `tests/test_webui.py`: every slot the JS writes is declared in the markup and every slot declared is written
  (an undeclared one silently stops rendering; a stale one shows a dash forever); the `.slot` rule carries
  nowrap + ellipsis + tabular-nums; the rate row is three slots per card and the trend row two, so the two rows
  stay the same shape; `small--avg` is gone.
- `tests/render-monitor.mjs`: the per-slot character budget above, against a live payload.

## The decode headline row becomes counts: the per-stream average leaves the headlines (2026-11-04)

Asked in Chinese, and the question was the one the page kept inviting: **why does the Monitor's aggregate decode come
out below one stream?** 24 tok/s summed against 162 tok/s for a stream alone. The answer is arithmetic, and it is in
notes/context-262k.md: the summed number is per **wall** second (47.6% of this launch's wall seconds hold no request
at all), the engine runs 1.62 streams per record at 2.33 tokens a step, and 5.9% of its records carry a ~1 s prefill
chunk against 20 ms for the rest. 0.476 x 0.191 x 1.615 = 0.147, and 0.147 x 162 = 23.8 -- the number on the screen.

**But the page was making the question harder than it needed to be, and that part is fixable.** The first card in the
headline row was the per-stream mean: `generation_tok_s / running_mean`. Two things were wrong with giving it the
biggest type on the page.

- **Its divisor is not the number of decoding streams.** `num_requests_running` counts a request in the middle of a
  chunked prefill as running -- true in vLLM v1, and the reason the prefill card has to estimate from the KV gauge in
  the first place. So the divisor contains streams that cannot produce a token yet, and the quotient is neither a
  per-stream rate nor an aggregate one. The honest fan-out is `spec_decode_num_drafts_total / iteration_tokens_total`
  = 1.62 streams a step, and nothing on the page said so.
- **Its position invited the comparison that reads as a contradiction.** "Decode · one stream" next to "Decode · all
  streams", with the second smaller than the first, looks like the engine getting slower when more streams run. It is
  not: it is a denominator question, and the honest per-request figure for this launch is a median of 115 tok/s.

**What the row shows now.** A count and a per-record total, both summed over streams, with no division that can be
misread:

| card | before | after |
|---|---|---|
| headline 1 | `Decode · one stream` -- aggregate / running | `Decode · total this launch` -- `generation_tokens_total` (1.00M tok), note line `vs 129.6M in` |
| its slots | spans + poll of the same quotient | the counts credited inside each span (`100s +2,374`), and `poll +N tok` / `poll · none credited` |
| waveform 1 | `Decode · one stream` (a rate per stream) | `Decode · per step` -- generated tokens per iteration record, 2.6-2.8 live |
| waveform 2-4 | unchanged | unchanged: all-streams rate, prefill from KV growth, streams running |

`webui.py` gained the fields for that: `since.generation_tokens_total` / `prompt_tokens_total` / `iterations`, per-span
`generation_tokens` / `prompt_tokens` / `records` / `gen_per_step` / `drafts_per_step` (`iteration_tokens_total_count`
and `_sum` joined the ring's counter list), and `gen_per_step` on both rings' points. The per-stream fields stay in the
payload -- the request rows and `stream_tok_s_hist` still use them, and the long comment at the top of the file about
which histogram family is the honest one is still true -- they simply do not get headlines.

**Two details worth the ink.**

- *A record is not a step.* `iteration_tokens_total_count` counts one record per decode step and one record per
  chunked **prefill**, however many chunks that prompt ran over (measured on 0.32.0: a 29,969-token prompt produced
  records 29,969, 4 and 1). So `gen_per_step` is really "per record", the waveforms' title says so, and the census in
  notes/context-262k.md now carries the resulting error bar (~5% on the record count, nothing on the 75% of seconds).
- *The slot budget cut a number, and that is a decision.* Three spans of counts is `1m +4.1k · 15m +31k · 1h +89k` =
  29 characters, and the tightest 4-up slot holds 24 at 12 px, so the visible line shows the two widest spans and the
  title carries all three. The alternative -- letting the line wrap -- is exactly the bug the previous section was
  about: a wrapping note line changes the card's height, which changes the grid row, which moves every card below it.

**Harness.** `tests/test_webui.py` asserts the new fields are fed by names the ring stores, that no per-stream rate
reaches a headline or a waveform (`A.stream_tok_s`, `h.stream_tok_s`, `rate.stream_tok_s_live` are all absent from
`monitor.js`), and that the trend/rate rows still have the same slot shape with the new names.
`tests/render-monitor.mjs` checks the count card against the engine's own counter (within `dnum`'s rounding to the
nearest integer) and
that it carries a token unit, not a per-second one -- and its live comparison now allows an absolute 1 token/s as
well as a ratio, because on a box that idles between requests 2.0 and 3.0 tok/s are one second of traffic apart.
Both pass against the sidecar (`:8099`, new payload) and against the production serve (`:8080`, whose payload predates
these fields: the card shows a dash rather than inventing a total).

### ...and then the denominator: prefill time comes out of the decode rate (same day) — **SUPERSEDED the same day; kept because it records a wrong turn**

Read this one with the block above ("Two decode denominators, one identity") open. Two of its claims are false: the
two decode-seconds families agreeing to 0.08% is one sum sampled twice, not corroboration, and an ITL-shaped
denominator does **not** exclude prompt chewing (the step interval contains the chunked prefill computed in that
step). `decode_tok_s` was a per-stream rate wearing a "summed" label, and `decode_time_pct` was a fan-out printed
as a percent — 260% on a span where 2.6 streams decoded at an average moment. Both names are gone. The card first
led with the wall sum and printed `×2.60 decoding`; the same day it went the other way round (per-stream-second
headline, wall sum on the note line), and on 2026-11-04 it came back to the **summed headline** with the per-stream
figure and its unit spelled out on the note line. What has not changed since the fix is the substance: both readings
are on the card, their denominators are named in the payload field names, and the product is checked by the render
harness (see "Two decode denominators, one identity" above).

The summed card still read ~15-20 tok/s, and the follow-up question was the right one: **what is the time basis, and
shouldn't the seconds spent chewing prompts not count?** It shouldn't, and now they don't -- with the summed figure kept
beside it rather than replaced.

**What the engine offers as "decode seconds".** Two families, and they agree: `inter_token_latency_seconds` carries
430,954 observations summing to 24,236.14 s over this launch; `request_decode_time_seconds` carries 1,812 observations
summing to 24,216.27 s -- **0.08% apart**. The first is one observation per decode STEP (per stream, so two streams
deciding together accrue twice: it is stream-seconds, not wall seconds), the second is one per finished request. The
step family is used because it grows **every decode step**, while the request family only moves when a request finishes
-- over a 60 s span of pure decode with long generations, the request family can report no denominator at all. When a
build publishes neither, `None`, not a division.

**The card now:**
```
Decode · summed, decode time     166 tok/s
                                 wall 24.4 · 14.7% dec
                                 100s 164
                                 poll 190 · 6.9 req/m
```
Headline = generated tokens ÷ decode seconds. The note line keeps the summed-over-wall figure (`wall 24.4`) and adds the
**share of the span that was decode time** (`14.7% dec`) -- because the ratio of the two rates *is* that share
(24.4 / 166 = 0.147), and without it the pair reads as a contradiction. The spans line is the same quantity over the
other spans; the poll line is the poll's own decode-time rate plus the span's arrival rate, and it reads
`poll · no decode step` when the window held no decode step (no denominator is not a rate of zero).

**Field names say which seconds were counted**, because the last thing this page needs is a rate whose denominator is
guesswork. The surviving rule (written out in "Two decode denominators, one identity" above): `…_tok_s` means per
wall second, `…_stream_tok_s` means per stream-second, and a stream-count field (`decode_streams`,
`prefill_streams`, `streams_per_record`) is a count that may exceed 1 — never a percentage. `decode_time_s` and
`decode_time_req_s` are the two families' stream-seconds, kept side by side so their 0.08% agreement can be seen for
what it is. `generation_tok_s` is the wall figure and lives on the note line, labelled `summed`; the headline is
now `decode_stream_tok_s`. The `Decode · summed, wall` waveform stays
on the wall basis: a stream-second waveform has no denominator in a window that was all prefill, and a line full of
gaps would be read as an idle box.

**What the two bases are for.** The headline answers "how fast does this engine decode when it is decoding" (166 tok/s
now; 158 tok/s median per request over the launch; 41.8 tok/s summed across the whole launch's decode seconds, against
68.8 for the current window). `wall N` answers "how much decode does this traffic produce per second of the day"
(24.4 now, 23.7 over the launch). The gap between them is entirely the traffic mix: for the 661 s window measured, 83%
of the span held somebody's prefill, 94% held a request, and only 29% held a stream in its decode phase.

**Harness.** `tests/test_webui.py` checks the fixture's ring is self-consistent (20 steps a second at 30 ms with 40
tokens a second = 2 tokens a step, so the decode-time denominator cannot be satisfied by an arbitrary sum), that the
step family wins when present, that the request family stands in when it is not, that flat counters give `None` rather
than 0, and that `decode_time_s` / `decode_time_req_s` / `decode_time_pct` are fed. `tests/render-monitor.mjs` requires
the `wall N` figure to still be on the page against `avg[win].generation_tok_s`, and requires the share suffix whenever
a decode-time rate is shown.

### Borrowed from Strata: the prefill numerator counts READ tokens, and a card never invents a zero (2026-11-04)

Read through `~/Strata` (a different stack: llama.cpp-style engine, its own server/monitor/CLI) to see how it keeps the
same numbers. Its accounting is worth describing because it is the same problem solved from the other side -- its
primitives are per-request *seconds*, not engine-wide counters.

**Primitives.** The engine's DONE line gives each request `prompt_ms` (time reading the prompt), `decode_ms` (time
generating), `generated`, `prompt_tokens`, `reused` (prompt tokens the prefix cache already held) and the draft counters.
`serve/server.py:4556` starts a `totals` dict and `:6332-6335` adds each finished request into it: `requests,
prompt_tokens, reused, output_tokens, prompt_ms, decode_ms`.

**Three rates, and the card says which one it is** (`serve/web/app.js`, pinned by `serve/test_speed_average.py`):

| state | value shown | label |
|---|---|---|
| generating | `_tok_s()` -- a **2 s sliding window** (`RATE_WINDOW_S = 2.0`) of `(time, generated)` samples per request, ring of 32, falling back to the whole-request mean when the window span is under `RATE_MIN_SPAN_S = 0.25` s; **summed across parallel requests**; `0.0` while nothing generates | `Decode now` |
| **reading (prefilling)** | **the run average**, not a diluted live value | `Decode · run average` |
| idle | `output_tokens / (decode_ms/1000)` since server start | `Decode · run average` |
| no totals | the last request's own speed | `Decode last request` |
| nothing at all | `null` | `Decode` |

The API monitor page's `averageDecodeSpeed(rows)` is the same idea over the kept rows -- `Σ predicted_n / Σ predicted_ms`
-- and its test asserts the *reason*: "a plain mean of the two (about 38) would let the short one count as much as the
long one". Requests with no timings or one token are dropped. `strata_cli/overview.py:run_average_speed` documents it as
"never an invented zero".

**What we took.**

- **Prefill's numerator is READ tokens.** Strata: `prompt_tokens - reused`. Ours: `prompt_tokens_total` minus
  `prompt_tokens_cached_total` when the build publishes it (vLLM's exact count of tokens it did not read — local
  prefix cache *and* KV-connector transfers), falling back to `prefix_cache_hits_total` (local cache only, counted
  at admission) — in the span (`prefill_stream_tok_s`), in the poll (`prompt_read_tok_s`) and on the engine-state
  line (`N prompt tok/s read`). vLLM credits cached prefix tokens to `prompt_tokens_total`, so with the cache on a
  rate over credited tokens is high by the hit rate. That was written on a `PREFIX_CACHE=0` recipe, so read ==
  credited then and the live numbers did not move (`7,177 tok/s` read, `wall 6,180` unchanged). The cache went ON on
  this recipe the same day (notes/prefix-cache-decision.md), so `prefill_stream_tok_s_credited` alongside
  `prefill_stream_tok_s` is no longer a pair held in reserve: at the 98.2%
  hit rate a replay of this box's own conversations measures, a rate over credited tokens would read ~55x the same
  rate over read tokens (`1 / (1 - 0.982)`), which is why the card prints the read one. When hits are non-zero the prefill card's note line gains `· N% cache` (22
  characters, inside the tightest slot's 24); with the cache off it says nothing, so it is not printed.
- **The zero discipline.** A span with no decode seconds has no decode rate. The card shows a dash, `poll · no decode
  step`, `poll · nothing credited` -- the same refusal in the same three places.

**What stayed different, on purpose.** Strata has no wall-second aggregate at all: its headline decode rate is always
per decode time, and concurrency shows up as *summing per-request live rates across slots* rather than as a diluted
denominator. That is the right choice for its world (one box, usually one or a few requests, a prefix cache it wants to
use). On this box the wall figure is the story -- 99.2% of the tokens processed were prompt tokens -- so `wall 24.4 ·
14.7% dec` stays next to the decode-time headline, with the share line making the two comparable
(24.4 / 166 = 0.147 = the decode share).

**Also noted, not copied.** Its Prometheus shim publishes the *seconds* as counters (`llamacpp:prompt_seconds_total`,
`llamacpp:tokens_predicted_seconds_total`) so the scraping side chooses the denominator, and resets a bucket per scrape
to match how llama.cpp's dashboards read those gauges. The vLLM-shaped names it synthesizes include
`vllm:inter_token_latency_seconds`, observed as `decode_s / (output_tokens - 1)` **weighted by `output_tokens - 1`** --
"the engine reports the decode time, not each token's". vLLM's own ITL is one observation per *step*, which is why our
decode denominator uses its `_sum` and checks it against `request_decode_time_seconds` (0.08% apart on this launch).
Its `vllm:num_requests_running` help text says "Requests reading their prompt or generating" -- the same gauge
semantics that took the per-stream average off our headline.

## The summed decode figure, re-checked against upstream's own arithmetic and against Strata's (2026-11-04)

The headline moved to the summed figure, and it still read wrong to a human. So it was checked against the two
places a reader compares it to, instead of argued about.

**Upstream's aggregate IS this number.** `vllm/v1/metrics/loggers.py` computes
`generation_throughput = self._get_throughput(self.num_generation_tokens, now)`: tokens accumulated since the last
stat line over the wall seconds since the last stat line (`VLLM_LOG_STATS_INTERVAL`, default 10.0 s). Tokens over
wall seconds -- `generation_tok_s`. Measured on the running launch rather than inferred: every printed line of this
boot, multiplied by its 10 s interval, implies **228,295 generated tokens**; `vllm:generation_tokens_total` says
**228,333**. 0.02%, with the residual at the two edges of the log. Repeated the same day over a 62 s window and a 286 s window:
the payload's `generation_tok_s` and the log-implied aggregate come out identical to the first decimal (3.9 and 3.9,
90.9 and 90.9).

**What the log is not is an average a reader can compare against.** Same file:
`log_fn = logger.debug if self.engine_is_idle else logger.info` -- an idle interval is never printed. Over one
903 s span the log printed 420 s of it; the printed lines average **42.4 tok/s** while the same tokens over all
903 s are **19.7**. A factor of 2.15 that comes entirely from intervals vLLM skipped, not from intervals it
measured. The card is upstream's basis; the tail of the log is the mean of the busy intervals; neither is wrong, and
the card now says which of its seconds were empty rather than leaving the reader to find the gap.

**Strata's headline is the other reading.** `serve/web/app.js`:
`avgDecode = totals.output_tokens / (totals.decode_ms / 1000)`, captioned "Decode · run average" and described as
"token-weighted, the same figure the engine's DONE lines and Prometheus report"; `answer_speed` (`serve/server.py`)
is "the one definition of answer speed: every token the model produced ... over the decode time, first generated
token to last", and it refuses a one-token reply because a single token has no first-to-last span. That denominator
is decode seconds, i.e. `decode_stream_tok_s` here, and the `produced >= 2` rule is the same discipline as dividing
by `output_tokens - 1` instead of `output_tokens`. Strata has no summed-over-wall card at all: the summed number is
this page's, which is exactly why its denominator has to be spelled out on it.

**The prompt side, probed rather than reasoned about.** Three identical 2,881-token calls, counters differenced
across them:

    prompt_tokens_total          +8,643  = 3 x 2,881: the prompt AS CALLED, cached tokens included
    prompt_tokens_cached_total   +3,328  = prefix_cache_hits_total: the reuse the engine granted
    prompt_tokens_by_source      +3,328 local_cache_hit, +5,315 local_compute -- and the two sum to the 8,643
    iteration_tokens_total_sum   +5,318  = 5,315 computed prompt tokens + 3 generated, exactly

and over the whole boot the log's "Avg prompt throughput" accumulates the COMPUTED side: 1,791,798 implied by the
printed lines against 1,797,485 of `local_compute`. So the log's prompt line and the payload's READ tokens are one
quantity, and on a launch whose hit rate is 95%, `prompt_tokens_total` is 22x that quantity. The credited figure
stays on the card as a credited figure and never becomes a speed on its own.

**What came out of it: the fan-out's two factors, and the resolution of each.** Each span now carries `busy_s`,
`busy_frac` and `streams_while_busy`. A gap of the ring's 5 s spacing counts as worked when the engine's RECORD
counter moved across it (`iteration_tokens_total_count` -- a step ran exactly when that moved), or when a request was
in flight at either end of it. That is an **upper bound at 5 s resolution**, not a count of busy seconds: measured on
a 62 s span holding four 0.45 s requests, this says 0.33 of the span while the engine's own log says 0.65 of it, and
neither is the truth -- vLLM marks a whole 10 s interval for a request that took half a second, and the ring marks a
whole 5 s gap for the same work. The gauge alone was worse still: read at instants it calls a worked gap idle when a
request arrives and finishes between two samples, which pushes `streams_while_busy` up (measured 0.46 against the
log's 0.66 on a 286 s span before the RECORD counter was used). What is *not* granular is `decode_streams`: it comes
from counters and is exact, and the product `busy_frac x streams_while_busy` is that exact figure by construction --
so the split is a ratio answer to "idle, or sharing?", never a precision claim, and it stays in the tooltip rather
than beside the big number. Each sample stands for the gap that follows it, and the first is charged back to the
span's far edge -- charged only backwards, a span busy from its first sample to its last reads 0.9375 rather than
1.0, which is the specific error `tests/test_webui.py` pins. The summed waveform's second note line prints
`busy 20/90` where it used to print `20/90`, for the same reason a bare fraction should not have to be remembered.

`tests/test_webui.py` pins the four cases: samples busy in every gap give duty 1.0 and `streams_while_busy ==
decode_streams`; traffic that stops mid-span halves the duty while the streams that were running keep their count
(a half-idle fixture whose counters keep moving would test the arithmetic and miss the point, so this one freezes
them the moment the last request leaves); a span with nothing in flight has no `streams_while_busy` at all (0/0 is
not a concurrency measurement); a restart inside the span gives no duty, because samples straddling two processes
have no share to report.
`tests/render-monitor.mjs` re-checks the product against the payload's own window and fails if the card measures
the factors and never states them.

The engine's own page (8080, in-process plugin) picks the fields up at its next restart; the sidecar has them now.
The page's sentence is guarded by `isnum()`, so a payload without them stays silent rather than inventing a duty.

## About now lists the configuration, as three sources that are allowed to disagree (2026-11-04)

"What is this box configured to?" has three different answers, and a page that shows one of them is a page that
gets asked again. The new card prints all three, each labelled with where it came from:

- **recipe** (`meta`) -- what the launcher recorded when the serve was started: `maxlen`, `nbt`, `nseq`,
  `prefix_cache`, `host`, `kvdtype`, `mtp`, the cudagraph sizes, the fork's own `hc_fp8` / `ar_quant` switches.
- **engine config** (`config._all`) -- every label the running build publishes on its info metrics, not only the
  ones the cards read. On the live cache-on launch that is **46 labels against the 15 curated**, and the ones the
  cards never showed are exactly the ones people ask about: `mamba_cache_mode`, `prefix_match_unit`,
  `prefix_caching_hash_algo`, `prefix_cache_retention_interval`, `kv_cache_max_concurrency`, `kv_cache_layout`,
  `sliding_window`, `use_replayssm`. The WANT filter stays for the cards and the checks -- a page that reads
  whatever label happens to appear is a page whose meaning moves when vLLM renames one -- but `_all` is listed, so
  a fork knob nobody mapped is at least visible.
- **environment** -- the whitelisted part of the process's own environment, plus the machine-local env file
  (`LOCAL_ENV`, default `~/.config/r9700-stack/native.env`) parsed from disk each poll, so the page agrees with the
  file an operator just edited. The process rows also say *which process* they describe: on the merged page the
  environment is the serve's, on a sidecar page it is the sidecar's, and `MAXLEN` in a sidecar's env proves nothing
  about the engine.

Two rules the card keeps. **A whitelist, not `os.environ`**: a running serve's environment is hundreds of names,
and this page is served on every interface of the box -- the card lists what changes the page's numbers or how the
box is reached (`VLLM_*`, `R9K_*`, `R9700_*`, `WEBUI_*`, and the launcher's named knobs), and nothing else. `R9K_`
is a whole namespace rather than a list of names because those variables pick **which kernel runs**: they change
what a measured number means without appearing anywhere in the engine's measured config, which is exactly the blind
spot `tools/recipe-check.py` had on 2026-11-04 (notes/launch-methodology.md section 12). **A
credential is named, never printed**: a name matching `KEY|TOKEN|SECRET|PASS|CRED|AUTH` shows as
`set · value withheld`, so "not printed" cannot be misread as "not set"; the same rule applies to the process's own
command line, which the card also prints (it is the restart line) with `--api-key`/`--key` values replaced by
`<withheld>` while the flag stays visible -- a restart line that drops the flag starts a serve the operator cannot
reach. The live check in `tests/test_webui.py` asserts the withheld value appears nowhere in the block.

The disagreement between the three lists is the point: `tools/recipe-check.py` is the tool that turns one into an
exit code, and the card is the place it can be seen before being typed.

### About has a render harness now, and it found something on the first run

`tests/render-about.mjs` runs `serve/web/about.js` unmodified in a stub DOM against a live payload or a saved one
(`tests/render-about.mjs http://127.0.0.1:8099`, or a `/api/status` JSON file). About had no such thing before: the
suite parsed the file and pinned its strings, which catches a syntax error but not a ReferenceError, and a
ReferenceError in one section leaves that section empty with no error on screen -- the failure mode recorded above
for the Monitor's `${unit}`.

On its first run it flagged the merged page on `:8080`: the card renders, but its environment and label sections are
empty, because **the in-process plugin in that serve was started before this card existed** and its payload carries
no `env`, no `argv`, no `config._all`. That is a real difference between the two pages on this box, and it is the
kind that is invisible from the source tree: the files on disk are current, the running process is not. It also
turned into a wording rule -- the page must not say *"there is no machine-local env file at undefined"* about a
payload that has no env block at all. "No env block" means the build is old; "no file" means this box never had one;
the page says which, and what closes the gap is a restart, not a file.

## The sidecar was describing a launch that had already ended (found while adding the Image input card)

Two pages, one recipe file, different answers:

```
:8080 (merged, in-process plugin)  -> 33 recipe keys, multimodal ones included
:8099 (sidecar, same --meta path)  -> 30 recipe keys, no mm facts at all
```

Not a parsing difference: `Hub.__init__` read the recipe **once at startup**, and the sidecar had been alive since
before the serve it describes. So the About page's "How this box was launched" card — and now the Image input card —
was reporting a recipe from an older launch, with nothing on the page saying so. Worse than an empty card: it was a
*populated* card about a serve that no longer exists.

The fix is three small things, all now covered by tests:

- `Hub.refresh_meta()` runs at the top of every poll: one `stat`, and the file is re-read only when its mtime moved
  (called before `with self.lock`, because `threading.Lock` is not reentrant and `poll()` takes it a few lines
  later).
- The payload carries `meta_mtime` and `meta_error`, and the launch card prints `written <time>` — so "which launch
  is this page describing?" is answerable instead than assumed. A failed read sets `meta_error`; it never degrades
  into an empty recipe, because "no recipe" and "could not read the recipe" are different claims.
- `launch_hint` replays the multimodal knobs (`MM=1 MM_IMAGE=… MM_VIDEO=… MM_MAX_PIX=… MM_MIN_PIX=…`) when the
  recipe says the launch had them. Before this, an image-enabled launch's restart command silently came back
  text-only: the page would have been describing one serve and handing out the command for another.

Proof on the live box (not a unit test only): after `touch ~/r9700-web-meta-8080.json`, `:8099`'s `meta_mtime` moved
from `1793839620.29` to `1793840975.67` within one poll, and its key count matches `:8080` at 33.

One honest gap: `:8080` still prints `meta_mtime: null`. The merged page is code that the **engine** imported at
startup, so it updates when the serve restarts — not before. The card that could be refreshed (the sidecar) was
restarted; the one that needs a serve restart is left to say nothing rather than something stale.

The gap closed itself with the next serve restart, which is what it was waiting for. After the machine reboot of
2026-11-05 01:55 (engine `pid 3554`, sidecar `pid 5702`), both pages report the same recipe and the same stamp:
40 recipe keys and `meta_mtime 1793843723.15` on `:8080` and `:8099`, `meta_error: null`; after
`touch ~/r9700-web-meta-8080.json` both moved to `1793843940.10` inside one poll. Two facts on the same page kept
separate while proving that: `:8080`'s argv contains the multimodal flags, `:8099`'s argv contains none — because
`:8099`'s argv is the **sidecar's own** command line, which proves nothing about what the serve accepts. The card
says so in words.

And the reason the recipe keeps growing keys: a page that quotes a launcher fact must read it from the launcher.
The Image input card's knobs row used to end in a hardcoded `MM=0 is this tree's default`, which was true when
written, was untested, and outlived the default flipping to `MM=1`. The launcher now records `mm_default`, the page
prints `this tree's default is MM=<value>`, and a recipe written before the key existed gets
`this recipe does not record the tree's default, so this page does not state one` — see
[multimodal-input.md](multimodal-input.md).

## The PCIe panel becomes one word on the card (2026-11-05)

The panel had already been demoted once — from four cards of waveforms to one text row per chain, because both
things it could report are static-or-eventual. That was not far enough. Asked "can it be quieter still", the answer
this time was: **remove it, and put one word on the card.**

What made the case was the rate, not the counts. Measured on this box through ordinary weight loads: **11.3–45.7
correctable AER counters a minute**, with every port on every chain holding its full 32.0 GT/s ×16 the whole time.
The panel coloured a chain whenever those counters moved, so on a healthy box at work the colour was on, always,
and the one state that deserves colour — a link that negotiated *below what the ports above the card are rated
for* — was indistinguishable from it. That is the failure mode the sparse-glyph step had already tried to fix, and
the fix did not hold: a rate threshold still fires on weather.

The new rule:

* A card prints its **generation**: `Gen5 ×16`. `pci_gen` in `serve/webui.py` turns 32.0 GT/s into 5 (2.5 is Gen1,
  each generation doubles; an unnamed rate is derived by doubling, a zero or unreadable rate is no generation at
  all). The GT/s string is what nobody could read as "one generation short" without a datasheet open; it stays in
  the tooltip and in the account.
* A card that came up **short** says so and says what it fell short of: `Gen4 ×16 of Gen5`, or `Gen5 ×8 of Gen5`
  for the x8-on-an-x16 case that a speed reading hid entirely (both come from `read_pcie`'s `degraded`, which
  already checked width as well as speed).
* Colour stays tied to work: a **parked** card (D3hot between launches — every chain on this box reads 2.5 GT/s
  then) is named in muted type and never coloured; a short link on a card at ≥25% busy is the danger case.
* **No error number is printed anywhere on the page unless that chain is short.** The counters are still read every
  poll, still ride `/api/status` (`pcie`, `pcie_history`, `pcie_events`), still land in the journal, and are still
  what `tools/journal-pcie.py` and `tools/aer-watch.py` read. What was removed is the alarm, not the evidence.
* The account survives as the block under the Cards, opened from the badge of a short chain (`#gpus` carries the
  delegated click, since the cards are rebuilt every tick). Nothing prints numbers for a chain nobody would act on.

Mechanically: `link_facts(chains)` builds `{gen, max_gen, speed, width, max_width, short, parked, corr, uncorr,
slot, chain}` per GPU endpoint and the poll attaches it as `g.link` — joined by chain, not by bus id, for the same
reason the slot join is: the firmware names the slot at the card's switch upstream port (`c4:00`), not at the GPU
endpoint (`c6:00.0`). A card with no chain in the payload gets no `link` key at all, so the page prints no badge
rather than a dash it cannot justify. `slot_map` gained `gen`/`max_gen`/`short`/`parked` so the chassis line says
`Gen5 ×16` in the same voice as the cards.

The tests moved with it, and one class of them went away entirely: the PCIe grid arithmetic (seven named tracks
sized from the longest string each column carries, `test_webui.py`'s 2026-11-03 section) had nothing left to size.
What replaced it asserts the rule rather than the layout — that a card at its rating renders a word and **no
button**, that no card prints a `corr` count unless its badge is a button, and that `pci_gen` reads 2.5/5/8/16/32
as Gen1…Gen5 with 0 as "no reading" rather than "Gen1".

Proved on the live box, both pages (`node tests/render-monitor.mjs`): four cards print `Gen5 ×16`, tone unset, no
button — and **slot 8 / `06:00.0`, the chain that carries 23 correctables since boot, prints none of them.** Then
the same payload with one chain forced to `gen 4, short`: the card reads `Gen4 ×16 · 44 corr · 1 uncorr of Gen5`,
tone `warn`, and its badge opens the account (chain, each port rated 32.0 GT/s ×16, `BadTLP 51 · NonFatalErr 3 ·
Timeout 1`, `UnsupportedRequest 1 non-fatal`, rate, quiet, last event). Two harness facts learned doing it: the
stub DOM's `querySelectorAll` hands back slots rather than descendants, so the badge is read out of the card's
`innerHTML`; and `let status` in `monitor.js` is a lexical binding, not a global property, so the harness keeps a
`clone()` of the `/api/status` response instead of reaching into the page.

The one thing the change did not fix, named so it does not read as fixed: pointed at the **standalone sidecar**
(`:8099`), the harness prints `BAD the header never named the engine`, because `stack_info()` finds no importable
`r9700_vllm` in a sidecar running outside the serve's venv — the fields are absent by design and the header prints
what is there. The merged page on the serve's own port (`:8080`) reads `vLLM 0.32.0 · plugin 0.4.2 · 4f4d1b2a` and
exits 0. Point the harness at `:8080` when the header is the thing under test.
