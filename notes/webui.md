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
| tok/s, req/min, prompt tok/s | counter deltas over the poll window (`generation_tokens_total`, `prompt_tokens_total`, `e2e_request_latency_seconds_count`) |
| first token, queue, prefill, decode means | `_sum`/`_count` deltas of `time_to_first_token_seconds`, `request_queue_time_seconds`, `request_prefill_time_seconds`, `request_decode_time_seconds` (`bench/ttft_breakdown.py` reads the same family) |
| speculation acceptance %, drafts/request | `spec_decode_num_accepted_tokens_total` ÷ `spec_decode_num_draft_tokens_total` (the older builds only have `..._num_drafts_total`, which gives drafts per request but not the token ratio) |
| tokens per engine step, steps per request | `iteration_tokens_total_{sum,count}` |
| prefix cache hits % | `prefix_cache_hits_total` ÷ `prefix_cache_queries_total` |
| card busy % / VRAM / temperature / sclk / mclk | `/sys/class/drm/card*/device` (`gpu_busy_percent`, `mem_info_*`, `hwmon*/{temp1_input,power1_average}`, `hwmon*/freq*_label` matched by `sclk`/`mclk`, not by index) |
| the launch recipe on About | the meta file `serve/serve.sh` writes when `WEBUI=1` (`--meta`), not something inferred |

**What cannot be answered per request:** queue wait and the prefill/decode split. vLLM counts those in
aggregate, so the Monitor shows them as window means under "Engine timings" and the per-request rows
carry a dash. For the real per-request breakdown use `bench/ttft_breakdown.py`, which differences
`/metrics` around a controlled set of requests.

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

`tests/test_webui.py` (109 checks, no GPU and no network needed) and `tests/test_webui_plugin.py`
(32 checks; the record store and the stat logger run anywhere, the routes need FastAPI, and vLLM's own
route-replacement tier runs only where vLLM imports — `LD_LIBRARY_PATH=~/opt/openmpi/usr/lib/x86_64-linux-gnu
~/venv-r9700/bin/python tests/test_webui_plugin.py`).

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

A stream's tok/s is 1 / step (64 tok/s), never 1 / per-token (313 tok/s). Both are on the page, labelled.

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
