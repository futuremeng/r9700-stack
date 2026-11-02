# Changelog

All notable changes to r9700-stack. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project uses [semantic versioning](https://semver.org/). Every number below was measured on 4× (or 2×)
Radeon AI PRO R9700 -- at a 225 W cap through v0.2.3, at 210 W with a -42 mV voltage offset from v0.2.4 on;
[PROGRESS.md](PROGRESS.md) has the method behind each one.

## [Unreleased]

### Added
- **Monitor and About pages for a running launch** (`serve/webui.py`, `serve/web/`, `serve/webui.sh`;
  [notes/webui.md](notes/webui.md)). A stdlib-only host sidecar next to the container: `observe` mode
  (default) reads `/health`, `/v1/models`, `/metrics` and the amdgpu sysfs and changes nothing in the
  inference path; `--proxy` additionally relays the API, which is the only way to get per-request
  records out of stock vLLM at all -- it has no per-request API, only cumulative counters. The Monitor
  carries the KV pool (from `cache_config_info` labels), per-window tok/s, TTFT/queue/prefill/decode
  means, speculation acceptance, prefix-cache hits and per-card busy%/VRAM/sclk; the About page carries
  the configuration, the machine, the base URL and the recipe `serve/serve.sh` wrote for it (`WEBUI=1`).
  Cards whose sysfs attributes answer EBUSY name what they refused instead of reporting 0%, and `SMI=1`
  fills the gaps from rocm-smi joined by PCI bus id (opt-in: it opens `/dev/kfd`). 106 checks in
  `tests/test_webui.py`: metrics collapsing (buckets skipped, gauges max-ed not summed, no data before
  the first window), the request ring, the sysfs and rocm-smi readers, an end-to-end relay against a
  fake vLLM (SSE passed through, a 500 recorded as an error, `observe` relaying while recording nothing,
  path traversal refused), and a stream that must reach the client **while the upstream is still
  generating** -- which caught a real bug: the pump used `res.read(8192)`, and on a body with no
  Content-Length that blocks until n bytes or EOF, so a slow stream would have arrived all at once and
  every first-token number on the page would have been a lie (+70 ms on a 272 ms request). It reads
  `read1()` now. `bench/webui_ab.py` A/Bs direct / relay / record: about **1 ms per request** for the
  hop against a loopback stand-in, and recording costs nothing beyond that. Measured on the live native
  launch (4 cards, MTP=2, warmed, `--n 8`): the hop is **2-3 ms and flat in prompt length** — +2.54 ms
  (+5.3%) on a 76-token request, +1.96 ms (+0.8%) at 2,099, +2.43 ms (+0.5%) at 4,399 — and **recording
  costs 0.2-1 ms on top of relaying**, so the Monitor in proxy mode does not move a request's cost.
  [notes/webui.md](notes/webui.md) has the full table and the arm-order caveat.
- **`setup.sh` / `setup.py` / `update.sh`**: bring a box up, or re-check one that already runs it. One pass over
  what this repo actually needs, with the file each requirement came from named on the line: the cards as the guest
  sees them **and the size BAR0 was really placed at** (the starved card on a PLX switch reports `current size:
  32GB` in its ReBAR capability while `Region 0` is absent -- host/README.md's failure mode, so it is checked here
  rather than assumed), `/dev/kfd` and the render nodes, which cards hang off the same bridge (TP wants that pair,
  host-RAM experts want the split: 114/132/2464 against 81/54/1379), the kernel cmdline (`amdgpu.ppfeaturemask`,
  with `amdgpu.mes_log_enable=1`'s absence recorded as the reason two incident notes exist), `RLIMIT_MEMLOCK` (this
  box answers 11.7 GiB -- where that number in PROGRESS.md comes from), docker **or the native path** (whose ROCm
  vLLM wheels need the locally unpacked OpenMPI: import-tested with the same `LD_LIBRARY_PATH`
  `serve/native-flashnext.sh` sets), the pinned nightly from `docker/Dockerfile` against a recorded marker so a pin
  change offers a rebuild, the second checkout `serve.sh` mounts (the running plugin comes from there, not from this
  folder), every `/models/...` the serve scripts name with who needs it and where `serve/deploy.sh` fetches it from,
  and the tests that need no GPU. `--check` runs nothing that changes anything; `--launch flashnext` starts one and
  waits for `/health`. `update.sh` pulls with the guard Strata's has (it refuses while tracked files are edited and
  prints what `git status` says) and touches no weights, no launch caches, no running launch.
- **`WEBUI=1` on the native path too** (`serve/native-flashnext.sh`), and the recipe file it writes moved out of
  `serve/serve.sh` into `serve/webmeta.sh` so both launchers write the same JSON the About page reads. The sidecar
  starts **before** the engine: it has to answer `/health` for a serve that takes four minutes to load 120 GB, and
  it shows the cards while it does. `setup.py --launch native-flashnext.sh` now starts one (in its own process
  group, logging to `~/logs/setup-<time>.log` — the native launcher `exec`s vLLM in the foreground, and
  `notes/launch-methodology.md` records that vLLM's ranks can ignore `SIGTERM` in the model-load phase, so a
  child of a TTY-interrupted setup is how you get two launches on four cards).
- **The Monitor and About pages can now run on the serve's OWN port** (`WEBUI=same`, `serve/plugin`,
  `./setup.sh --webui same`; [notes/webui.md](notes/webui.md)). One package, two entry points: a
  `vllm.endpoint_plugins` plugin for the routes (`/`, `/monitor`, `/about`, `/web/*`, `/api/*` — vLLM's
  `attach_endpoint_plugins` puts a plugin's `GET /` ahead of the server's own root, by design) and a
  `vllm.stat_logger_plugins` plugin for the records. That second one is the point: vLLM's per-request stats
  (`FinishedRequestStats`, vllm/v1/metrics/stats.py:288-306) carry **queued / prefill / inference / decode
  seconds, mean time per output token and cached tokens per request**, which is a split the sidecar cannot
  get from outside at all — it can only average histograms over a window. Live: `queue 0.000 s · prefill
  0.040 s · decode 0.650 s · 66 → 120 tokens · 184.5 tok/s per accepted token`. `/metrics` stops being a
  socket round-trip too (the poller renders the process's own Prometheus registry: **~11 ms per 2 s poll**
  for 65 KB, on a thread, with every route handler a plain `def` so FastAPI runs it off the event loop).
  What merged mode cannot have is stated on the page rather than papered over: no request bodies (the engine
  reports tokens and timings, never text), no in-flight rows (a stat logger sees a request when it finishes),
  and no page that outlives the launch — which is why the 8081 sidecar stays, and on a box with watchdog and
  PSU incidents is the one to leave running. 32 checks in `tests/test_webui_plugin.py`.
- **About now hands out pasteable client config**: curl, a **LiteLLM proxy config** (`model_list` /
  `litellm_params` with `model: openai/<served name>`, `api_base: <origin>/v1`, `api_key: none` or an env var
  name — never the key itself), and a direct `litellm.completion` that reads both `message.content` and
  `message.reasoning_content`. Each block has a Copy button, and the base URL in them is the address the
  browser reached the serve on, so it pastes correctly on the machine you are sitting at. Verified end to end
  with litellm 1.106.6 installed into `/tmp` (nothing added to either venv): the rendered snippet answered,
  and the YAML block loaded through `litellm.Router` answered too (`model used -> Qwen3.8`). See
  [notes/webui.md](notes/webui.md).
- **262,144 tokens of context runs on this box, and it is not the KV pool that limits it**
  (`bench/needle_long.py`; [notes/context-262k.md](notes/context-262k.md)). The checkpoint's own
  `max_position_embeddings` is 262,144 (`rope_type: default`, theta 1e7 -- no extrapolation to argue about), and
  raising MAXLEN from 32,768 **does not shrink the KV pool** (the memory-profiling run is bounded by
  `--max-num-batched-tokens`, not by MAXLEN): 497,588 → **540,202 tokens**, with vLLM's own line
  `Maximum concurrency for 262,144 tokens per request: 2.06x`. A needle ladder at 30,070 / 67,063 / 150,057 /
  **250,078** tokens retrieves at every tier (36.5 s cold TTFT and 6,847 tok/s prefill at 250k, 0.70 s for the
  same context with prefix caching on, 141 tok/s decoding 400 tokens inside it); GSM8K 10/10, `quality.py`
  needle 3/3 and the 32×300 throughput wave (1,186-1,257 tok/s) are unchanged, per-card VRAM 30.5 of 31.86 GiB.
  The limit is elsewhere: **8 concurrent 30k prompts leave the pool 11.7% full** while their TTFTs space out by
  exactly one prefill each (3.3 s steps, 26.2 s total), 32 short requests run 32-at-once at 20.7%, and three
  concurrent 250k requests queue to 36.9 / 73.7 / 109.2 s without OOM. Two metric-reading traps named: a third
  250k request is queueing, not failure; `waiting_by_reason{reason="capacity"}` is the *ordinary waiting queue*
  (`num_waiting_reqs`, `v1/metrics/loggers.py:1155-1159`) and read 7 there with the pool at 11.7%, while
  `kv_cache_usage_perc` read 32.3% during 4 × 150k when the hybrid groups were the short resource.
  `--long-prefill-token-threshold 2048 --long-prefill-token-threshold-adaptive` measured as a fairness knob, not
  a speed one: six 30k requests behind a 250k one go 40.3-55.8 s → **7.8-27.7 s** at the same wall clock
  (55.9 vs 56.8 s), but three 250k prefills in parallel drive the pool to **99.1%** and the load **22% slower**
  (109.2 → 133.6 s) -- so the recipe keeps it off, and 4096 (= NBT/2, matching the 2.06× memory can hold) is
  the value to use if mixed lengths arrive together. GLM-5.3-Flash's 262k indexer death
  ([release-v0.3.7.md](release-v0.3.7.md)) does not repeat on this checkpoint because QSA scoring here is
  `r9700_qsa_score`, which never materialises stock's fp32 `[rows, capacity]` logits -- 8.6 GiB a card at
  NBT=8192 against 1.35 GiB of headroom -- and `R9K_QSA_SCORE=stock` would put that allocation back.

- **A prefill speed of its own on every request row.** Inside the serve the record now carries that request's
  own `prompt - cached` tokens over its own `prefill_time` (the whole prompt when the engine cached nothing), and
  through a relay it carries prompt tokens over first-token latency with that basis printed, because a relay's
  number includes engine queue. Live: a 60,061-token needle prints 8,908 tok/s against a 6.74 s prefill. The
  numerator drops to computed tokens whenever cached tokens were reported, so a prefix-cache hit cannot report a
  speed the engine never computed.
- **The "prompt tok/s" line now says when it moves.** vLLM credits `prompt_tokens_total` at request end: a
  180,066-token prefill read 0 tok/s for its whole duration and then 89,580 in a single 2 s poll. The card reads
  "prompt tok/s (credited at request end)"; the live signal during a prefill is the KV pool bar, which climbed
  5.1 to 32.1% in 3.0% steps (16,206 tokens a sample) while that request ran.
- **README's Deploying section leads with `./setup.sh` and says there are two paths.** Step 0 is the setup/update
  pair (`--check`, `--yes --launch native-flashnext.sh --webui same`, `./update.sh`), 1a is the native path with the
  commands the checks themselves name (venv, the pinned ROCm wheels, `pip install -e .`, `pip install -e
  serve/plugin`, `kernels/build.sh` with the host hipcc and the copied `libr9k.so`, the locally unpacked OpenMPI),
  1b is the image. Host memory, honestly: measured on the native box with the 262k launch up, four
  `VLLM::Worker_TP*` at 12.5 GiB RSS and `Mlocked` 0, because four cards keep the experts in VRAM, so the ~120 GB of
  host RAM and the 38.7 GiB PLE pinning the two-card offload configuration wants are simply not asked for;
  `MemAvailable` 37.2 of 93.8 GiB. What the container layer itself costs stays unmeasured, since this box has no
  docker binary. `setup.sh`'s header no longer claims docker is required.
- **README layout: a contents line, split Speed headings, and the Monitor page in it.** `## Speed` no longer carries
  a parenthetical the width of a line: it says once that the first three tables are the release configuration
  (210 W cap, -42 mV, 32k) and the last is a different, uncapped box at 262,144 tokens, under two `###` headings.
  A new `## Watching it: the Monitor and About pages` section carries the screenshot
  (`docs/r9700-stack-monitor.png`, as wide as the page column, click for the frame), the two shapes of the pages and what
  each can honestly say, the relay's measured 2-3 ms, and the refusal to draw a prefill progress bar. The Serve
  block now marks which recipes run in the image and which line is the native one, and the check step is a list of
  what each bench proves rather than a paragraph of names.
- **The README's 262k table now carries the standing recipe, re-measured.** New rows, all from the launch that runs
  today: prompt processing 8,672-8,742 / 8,062-8,064 / 7,416-7,436 tok/s and TTFT 6.87-6.93 / 14.89 / 24.22-24.28 s
  at 60,061 / 120,066 / 180,070 tokens (needle found at each, no prefix cache in any of it), the same 60k prefill as
  the engine itself measures it (8,908-8,917 tok/s over its own 6.74 s, from `FinishedRequestStats`),
  `bench/probe32.sh` bands -- 32 x 300: 1,568.3 / 1,573.7 / 1,587.5, 32 x 1,200: 1,799.5 / 1,832.7, 16 x 300:
  1,133.1 / 1,135.8 / 1,142.7 -- card power at busy==100 (median 299 W, 132 samples), and the gate (GSM8K 99/100
  with 33,909 prompt tokens in 40 s, needle 3/3). Each row names which launch it came from.
- **README laid out the way Strata's is, with the four-card best numbers where they cannot be missed.** A centred
  lockup, one-line value proposition, centred nav with the two links that matter bolded, and five badges (release
  tag, release notes, Apache-2.0, stock vLLM 0.32, stock ROCm 10, two-to-eight R9700s). Then a `## Four cards, the
  numbers worth knowing` table: **225 tok/s decoding one stream** (best sample 228, 14.3-14.4 ms a step),
  **1,823 tok/s aggregate at 32 streams** (best sample 1,836.3, 1,200-token waves), **9,633 tok/s reading a prompt**
  (7,789 tokens), **262,144 tokens of context** with the needle found at 250,078 -- each row naming the machine and
  protocol it came from, because the numbers do not come from the same machine. A one-line `./setup.sh --yes
  --launch native-flashnext.sh --webui same` quickstart under the badges. The two-card and 27B tables collapse into
  a `<details>`, and the Monitor screenshot is a centred figure with a `<sub>` caption.
- **The README's title is a drawn lockup now, and the release page moved to the section that describes it**
  (`docs/lockup.py` → `docs/r9700-stack-lockup-{dark,light}.svg`). `# r9700-stack` was a markdown heading sitting
  above a 16:9 summary image whose own header says the same words; the heading is gone and the title is the picture,
  the way Strata opens: four bars for the stack and the four cards, the word **outlined** from the Outfit subset the
  Monitor already ships (weight 650 -- the README title is the same face as the pages, and paths rather than a font
  the reader has to download), dark and light variants switched on `prefers-color-scheme`, `alt="r9700-stack"` so the
  page still has a name for readers who do not get the picture. The summary image went down to
  `## What v0.3.8 gives you`, where its numbers belong, captioned with the script that draws it. `docs/lockup.py`
  asserts the wordmark and the mark stay inside the canvas before it writes, and `tests/test_readme.py` holds the
  arrangement: the README opens on the lockup, both files parse and stay inside their canvas, dark and light are one
  drawing in two inks, the wordmark is outlined rather than `<text>` (which inside an `<img>` renders in whatever
  font that viewer happens to have), and the release image is used exactly once, under the heading that carries it.
  And `tests/test_lockup.py` draws it and looks at the pixels: the word has to render as eleven separate letters,
  the mark as four separate bands, nothing clipped at a canvas edge, the wordmark title-sized rather than shrunk,
  mark and word on one middle line with air between them, and the two schemes one silhouette. It takes whichever
  renderer the box has -- `rsvg-convert` (the distro's `librsvg2-bin`) or `cairosvg` -- and both paths pass on the
  same files; with neither installed it skips, saying so. The docstring carries the apt line, the pip line, the
  no-root way to get cairo (unpack the distro's `.deb` beside the venv and set `LD_LIBRARY_PATH`; nothing is
  written to `/usr`), and the three `rsvg-convert` commands for looking at the README's figures at github.com's
  column width instead of measuring them.
- **The release page in the README now says what its numbers are, and links to a picture you can read.** Drawn to
  the rendered figure at github.com's ~980 px column, its smallest type lands at 9.8 px and the fine print does not
  read -- said plainly by a vision-capable model looking at the render (a `qwen3.8-flash-next-iq3_s` behind a
  LiteLLM proxy on the LAN: the same checkpoint family this repo runs natively, reached over its OpenAI-compatible
  endpoint with an `image_url` block), which is also how the title lockup and the card figure were checked: the
  lockup transcribes as `r9700-stack` with four bars and nothing cut off, and the card figure transcribes card for
  card exactly as the committed poll has it, including `6 waiting · 6 for scheduling capacity · 0 deferred` and
  `14,999 tokens per iteration record (prompt whole, decode per step)`. The caption now
  states that the figure's numbers are the v0.3.8 snapshot at the 210 W gate -- which is why its Flash-Next row
  reads 210 tok/s where the table above it reads 225, the newer measurement -- and the figure links to
  `docs/r9700-stack-v0.3.8.png`, the same drawing at 2400 x 1220 (2.0x the figure's canvas, so the click buys
  something). Three more checks in `tests/test_readme.py` hold the link, the resolution and the caption.
- **The title's mark is four bars in one colour, and a wrapped card line no longer ends on a connector**
  (`docs/lockup.py`, `docs/monitor-cards.py`). These are the two things looking at the renders caught and measuring
  had not: the top bar of the title's stack was in the muted step, which reads as a stray element rather than the
  fourth card -- and the release page's own mark has always been four bars in one colour, so the repo's two lockups
  were not the same drawing. And the Generation card wrapped as "(prompt whole, decode per step) ·" over
  "last 2 s": a line ending on a connector reads as a broken sentence, so `wrap()` moves the connector down with
  the clause it joins. Two more checks come with it (36 in `tests/test_readme.py`): no line a card draws runs past
  that card's own bottom edge -- a panel sized from its content can still be a line too short -- and no drawn line
  ends on a connector. What the renders also showed, and what was left alone: the Monitor's dark cards are bordered
  in `--st-line: #2a3036`, which is faint at the column's width, but the figure matches `serve/web/tokens.css` and
  strengthening it there would make the figure the odd one out on the page it pictures.
- **The README's two figures are looked at, not only measured** (`docs/summary-page.py`, `docs/lockup.py`,
  `docs/monitor-cards.py`). Rendering them at the ~980 px github.com puts a README in found four things the
  geometry checks had passed: the title's stack had a muted top bar while the release page's stack was one colour
  (the mark is four bars in one accent now); a wrapped Monitor card line ended on a connector, reading as a broken
  sentence (`docs/monitor-cards.py` moves a trailing `\u00b7` down with its clause, so the line breaks as
  `(prompt whole, decode per step)` / `\u00b7 last 2 s`); and the release page was unreadable and out of date.
- **The release page is generated now** (`docs/summary-page.py` \u2192 `docs/r9700-stack-v0.3.8.svg`, and the `.png`
  at 2000 \u00d7 1816). The hand-built one it replaces sat on a 1200 px canvas, so its smallest type rendered at
  **9.8 px** in the README column \u2014 unreadable without clicking \u2014 and its numbers were a release behind the
  README's own tables: it said Flash-Next **210** and 27B **216** under a single "210 W cap per card" banner that
  also covered uncapped numbers (913-933 at 16 streams, prefill 6,771-6,830), where the table says **225**. The new
  one is drawn from data on a **1000 px canvas against the ~980 px column** with a **14 px floor** (13.7 px as
  rendered), every box sized from the strings that have to fit in it, and every string measured first: what does not
  fit prints `OVERFLOW` on stderr and the test fails on it, which is how five real overflows got caught while it was
  being written. The numbers are current and each row names its gate: decode **225 / 222 / 100** tok/s labelled
  "210 W gate", and the aggregate / prefill / context column labelled once, "the native box, uncapped" \u2014
  **1,823 tok/s** at 32 in flight in 1,200-token waves, **9,633 tok/s** reading a 7,789-token prompt, **540,202**
  tokens of KV pool at MAXLEN 262,144 with a needle found at **250,078**. `docs/summary-image.py` stays as the
  provenance of the older v0.3.3 PNG. `tests/test_readme.py` is 36 \u2192 42 checks: the page re-draws text for text
  from its script, the canvas and the type floor hold, every number on the page is a number the README carries, each
  decode row names its gate, and the PNG is at least twice the canvas in both directions so the click is worth taking.
- **A Speed table for the four cards uncapped at 32k.** The prefill and aggregate headline numbers had lived only
  in [notes/native-4xr9700.md](notes/native-4xr9700.md); they are in the README now with their sample lists: single
  stream 186.7 tok/s, `bench.py` at 4 / 8 / 16 streams 372 / 841 / 1,164, `probe32.sh` at 32 in flight 1,598 (300-token
  waves) and 1,823 (1,200-token waves), 1,198 at 16 with 200-token waves, prompt processing 7,702 / 9,633 / 8,963
  tok/s at 1,951 / 7,789 / 31,166 tokens, KV 554,982. The table also says out loud what the two boxes disagree on:
  uncapped single-stream decode reads 186.7 here against 225 at the 210 W release gate, and that is recorded, not
  reconciled -- the 210 W figure has never been re-measured on this box.

### Changed
- **The Requests card says why the queue is waiting** (`serve/webui.py`, `serve/web/monitor.js`,
  `serve/web/monitor.html`). vLLM 0.32 publishes `vllm:num_requests_waiting_by_reason`, and the line is now
  `N waiting · M for scheduling capacity · K deferred`. Its HELP text says "'capacity' = waiting for scheduling
  capacity; 'deferred' = deferred by transient constraints (LoRA budget, KV transfer, blocked status)"; what the
  code assigns (`v1/metrics/loggers.py:1150-1159`, `v1/core/sched/scheduler.py:2947-2948`) is
  `capacity = num_waiting_reqs` and `deferred = num_skipped_waiting_reqs` -- so "capacity" is a name for the queue,
  not a claim about KV blocks. Which is the whole point on this model: eight ~30,000-token prompts on the standing
  262k launch held `running 2 / waiting 6 / capacity 6 / deferred 0` with the pool 8.8-10.2% full, six requests
  waiting for a slot in the schedule while the pool sat nearly empty. A build that publishes no reason labels
  reports the split absent, not zero (two checks in `tests/test_webui.py`).
- **Speculation carries a number with a ceiling the eye can check**: `accepted_per_draft` =
  `spec_decode_num_accepted_tokens_total / spec_decode_num_drafts_total`, which cannot exceed 2.0 at MTP=2
  (`v1/spec_decode/metrics.py`: `observe_draft` adds one draft and at most `num_spec_tokens` accepted a step).
  Measured 1.50 and 2.00 in poll windows on the 262k launch, 0.90 over the whole run.
- **The five headline cards are drawn, not screenshotted** (`docs/monitor-cards.py` →
  `docs/r9700-stack-monitor-cards.svg`, + `tests/render-monitor.mjs`). A README image lands inside github.com's
  ~980 px column, so the 1688 x 1953 Monitor screenshot renders at 0.58 scale and its text is unreadable; cropping
  it into a wide strip makes each panel smaller, not larger, because the column is the limit for the whole strip.
  The figure is drawn to the column instead (1000 px, five cards in two rows), every line measured against its own
  font size and wrapped, with any line that still cannot fit reported as a warning instead of leaving the canvas.
  `tests/render-monitor.mjs` runs the page's own `monitor.js` unmodified in a stub DOM against a live serve and
  prints what the cards actually render -- the JS and its payloads, not CSS, fonts or layout.
- **`bench/capture-cards.py`: the sampler that produces that figure.** It fires N concurrent long prompts, polls
  `/api/status` once a second, keeps the whole payload of the poll with the longest waiting queue, and hands that
  file to the drawer -- so the figure is one measured poll rather than a hand-picked second, and the poll is
  committed beside the figure (`docs/r9700-stack-monitor-cards.json`) so the picture can be re-drawn with no serve
  in the loop. Prompt length is calibrated by measurement: one cheap request decides the word count, because this
  filler costs 24.3 prompt_tokens a repeat and a word count landed 3x off (46,165 repeats came back as 93,544
  tokens). It replaces the ad-hoc `/tmp` script that a reboot deleted mid-run.

### Fixed
- **"tokens per engine step" was not a step size, on either card.** `vllm:iteration_tokens_total`'s own HELP text
  says "Histogram of number of tokens per engine_step", and `loggers.py:1273` observes
  `prompt_token_stats.computed + num_generation_tokens` -- but measured against a single request on 0.32.0, a
  4,893-token prompt produced 2 records (4,893 and 4) and a 29,969-token prompt produced 3 (29,969, 4, 1): a
  chunked prefill is credited in **one** record holding the whole prompt, while decode steps record individually.
  So the ratio tracks prompt size and sits above `--max-num-batched-tokens 8192` (29,969 in one record; window
  means of 9,621-14,999 seen here), and the page now calls it what it is -- "tokens per iteration record (prompt
  whole, decode per step)" with the measurement in the tooltip. The number also moved off the Speculation card,
  where "8,019.73 tokens per engine step" sat under a heading about MTP and read as "MTP accepts 8,020 tokens a
  step".
- **The README's card figure and its caption could drift apart, and did.** `tests/test_readme.py` re-draws the
  committed poll and compares it string for string, fails on the drawer's own overflow warnings, and requires every
  number in the alt text and every card number in the caption to be a number the drawn figure contains.
- **The README's 262k throughput row compared two different protocols.** It said "32 concurrent 300-token streams:
  1,186-1,257 tok/s", a wall-clock figure from the issuing process, and readers would compare it against the 32k
  recipe's `bench/probe32.sh` band of 1,574-1,598 and conclude the full context cost 25%. `probe32.sh` run on the
  262k launch gives 1,568.3 / 1,573.7 / 1,587.5 at the same protocol, so the real cost is nothing measurable; the
  wall-clock row is still there, labelled as wall clock from one process, and `notes/context-262k.md` says why the
  two differ. Same note fixes the launch provenance of the retrieval-ladder rows (measured with `NSEQ=4
  PREFIX_CACHE=1`, while the table implied the standing `NSEQ=32 PREFIX_CACHE=0` recipe).
- **`bench/needle_long.py` called a repeat run a prefix-cache hit.** The warm line ended in "(prefix cache)"
  whatever happened; this launch runs `--no-enable-prefix-caching` and the second prefill costs the same
  (60,061 tokens: 6.74 s cold, 6.73 s warm, 8,639 against 8,759 tok/s). The line now says which it was: a
  reuse line quoting both TTFTs when the second run really is cheaper (with prefix caching on, the same script
  measured 0.28-0.70 s warm against 4.25-36.52 s cold), or `warm: no reuse, prefill repeated
  (9,023 -> 9,492 tok/s)` when it is not.
- **The Monitor's per-request numbers were not per request.** The request list and the top cards read the row
  payload, and `usage`/`timings` were left out of it, so the "Last decode speed ... One stream" card fell back
  to the engine-wide `1/inter_token_latency` and still described itself as one stream: on the live merged page
  it read 27.4 tok/s (`1000 / 36.4 ms` of engine step) beside a request whose own decode was 87.8 tok/s, while
  a third figure, 871.7 tok/s, was `generation_tokens_total` summed over every stream in the window. A
  request's own prefill was not on the page at all except as a 2 s mean, so the 250,029-token needle rows and
  the 22-token rows were indistinguishable until opened. Both row builders now list from one `ROW_KEYS`
  (`serve/webui.py`), rows print `22 -> 296 tok · prefill 0.12 s · queue ... · preempted ...`, the wall-clock
  card splits that one request into queue/prefill/decode, and the speed card prints a dash plus the reason
  instead of borrowing a number from a different measurement.
- **The base URL the About page handed out was the wrong string in two ways.** It printed the sidecar's
  internal `--upstream` — `http://127.0.0.1:8080`, which on the box that runs the serve is right and pasted
  into a laptop is your own laptop (this box is reached over the network) — and it had no `/v1`, which is what
  LiteLLM and the OpenAI SDK need while curl does not. Both bases are generated from `location.origin` now, the
  address that actually answered the page, with the internal one kept under “Upstream this page dials” where it
  belongs.
- **A pasted LiteLLM snippet returned `content = None`.** Measured with litellm 1.106.6 against this serve at
  `max_tokens=64`: `reasoning_tokens = 64` — the whole budget went into the chain of thought, because these
  launches pass `--reasoning-parser qwen3` and reasoning counts against `max_tokens`. The snippet asks for 128
  and reads `message.reasoning_content` as well; the parser name comes from the recipe file, and a test asserts
  the launchers' `--reasoning-parser` line and their meta line name the same `${REASONING:-qwen3}`, so the page
  cannot describe a serve that is not running.
- **`VLLM_PLUGINS` is a strict allowlist, and setting it by hand killed the launch.** An endpoint plugin
  loads only when named, so the merged monitor has to set it — but for every other group vLLM loads
  *everything discoverable* when the variable is unset and only the named ones when it is set. Writing
  `VLLM_PLUGINS=r9700_webui` therefore silently disabled **this repo's own `r9700` plugin**, the one that
  teaches compressed-tensors the MXFP4-pack-quantized scheme, and the engine died four minutes into the load
  with `NotImplementedError: No compressed-tensors compatible scheme was found for
  layer_name='language_model.model.layers.0.linear_attn.in_proj_qkvz'`. The value is now built from
  discovery (`serve/webui-plugins.py`) and a test asserts `r9700` survives it.
- **Merged mode came up with no cards and no "merged" flag.** `cfg["gpus"]` defaulted to `None` (the sidecar's
  default is `/sys/class/drm`), and `payload["merged"] = bool(self.eng)` was False for an empty
  `EngineRecords` because it defines `__len__` — so the first seconds of a launch described itself as an
  observing sidecar. Both found by reading the live `/api/status`, both asserted now.
- **A request that never queued read as a dash.** `q or None` in the engine record conflated "no queue wait"
  (a real 0.000 s) with "not reported"; the engine's own number is kept as-is.
- **A token family was being scaled like a seconds family.** Cumulative means are a histogram's `_sum/_count`,
  and for time families that ratio is seconds so it gets ×1000 to print ms. Applied to `request_prompt_tokens`
  it made the Monitor report a **mean prompt of 2,028,442 tokens** on the first real launch (it is 2,028, and
  the engine's context is 32,768). The scaling is now only on time families; a test asserts a token mean of
  500 stays 500. This one only a live engine could catch — the fake in the tests was written from the
  container image's `/metrics`.
- **The native vLLM 0.32.0+rocm723 publishes different metric names than the pinned image, and reading only
  the old ones showed nothing at all**: `kv_cache_usage_perc` (not `gpu_cache_usage_perc`),
  `inter_token_latency_seconds` alongside `request_time_per_output_token_seconds`, no `num_requests_swapped`
  (now `swapped: null` rather than a false 0 — vLLM retired it), `_created` stamps (skipped: not samples).
- **`inter_token_latency_seconds` and `request_time_per_output_token_seconds` are not the same quantity, and
  the gap is MTP.** One is observed per engine **step** — what a client feels as one delta — the other per
  **request** as that request's mean per *accepted* token, so accepted drafts divide it. Live: **15.62 ms a step
  against 3.19 ms per accepted token** at MTP=2, 59.7% acceptance. A stream's tok/s is 1/step (64 tok/s), never
  1/per-token (313 tok/s); the old code preferred the per-token family, which would have overstated every
  stream's speed fivefold. Both are on the page now, labelled, with a test that keeps them apart.
- **KV capacity on a hybrid model is not `blocks x block_size`.** The live launch labels
  `num_gpu_blocks=1230, block_size=448` — 551,040 — while its own `kv_cache_size_tokens` says **497,588**: a
  hybrid pool's attention part is `blocks x block_size` and its mamba state is per request, so the product is on
  no axis tokens are counted on (11% too high). The label is preferred when present, the product used when not
  (on the older build it is exact: 456 x 1088 = 496,128).
- **A window mean is undefined while a long request is in flight**, because vLLM observes a latency histogram
  when a request *finishes*: polling every 2 s, TTFT/queue/prefill/decode read as no-data for the whole
  duration of a 600-token generation. The payload now carries `since` — the same ratios over the engine's whole
  run — and the page marks those numbers "(since launch)" instead of showing a dash it cannot justify.

## [0.3.8] - 2026-10-31

Release notes with the validation record: [notes/release-v0.3.8.md](notes/release-v0.3.8.md).

### Changed
- **Eight-rank all-reduce: the cross-switch stage exchanges through pinned host memory** (`comm/r9k_ar.py` `host=True`,
  `comm/r9k_ar8.py`). The four even ranks pushed peer writes across four root complexes, where the stores stall past ~100 KB in
  flight: 72 us for the 32 KB message a four-row MTP-3 verify step carries. The same kernels over pools mapped from `/dev/shm` and
  pinned with `hipHostRegister` take 47 us (64 KB: 61 against RCCL's 107). Messages of 12-96 KB take the host stage whole, 1-row
  messages keep the peer path, 128 KB and up stay on RCCL; the switch-local pair stage keeps its one-shot to 64 KB and the logits
  all-gather goes through the host stage too. GLM-5.3-Flash TP8 single stream 86.5 -> 100 tok/s (step 32 -> 28.5 ms), BetterBench
  91.4 -> 103.3, GSM8K equal (95.98, paired p = 1.0), soak and sanity clean (notes/glm53.md). Default on; `R9K_AR8_CROSS_HOST=0`
  restores the peer-write stage. TP2 / TP4 paths are unchanged (world-size-8 code).

### Fixed
- Every size rule of the eight-rank object now reads group-wide state agreed at startup: an even rank taking a message on ar8
  while its odd partner ran RCCL deadlocked the first forward.

## [0.3.7] - 2026-10-26

Release notes with the validation record: [notes/release-v0.3.7.md](notes/release-v0.3.7.md).

### Changed
- **The image pin moves to the 2026-10-25 vLLM ROCm 10 nightly** (`nightly-rocm100-e4a5b553`, vLLM 0.32.1rc1.dev466,
  torch 2.13.0+rocm10.0.0, the same ROCm 10.0 SDK): the 0.32 line. Five plugin-side fixes below made it serve all
  three configurations at the v0.3.6 numbers ([notes/vllm-0.32.md](notes/vllm-0.32.md)).
- **Known on this nightly:** GLM-5.3-Flash's 262k launch runs out of VRAM in stock's prefill indexer above ~75k tokens
  of prompt (about 1.5 GiB a card less headroom than the pin; 216k retrieval was fine on v0.3.5's pin). The 32k
  production recipe is unaffected. Details and the recipes tried: [notes/release-v0.3.7.md](notes/release-v0.3.7.md).

### Fixed
- **GLM-5.3-Flash on the vLLM 0.32-era nightly (e4a5b553) died at the first prefill**: upstream gave sparse MLA
  layers a dense-MHA prefill path and the ROCm selector hands it the CK flash-attention backend, whose wheel has no
  gfx1201 kernels (`HIP Function Failed ... device kernel image is invalid`). `R9700Platform` now answers the MLA
  prefill backend question with "none" on gfx1201 (`R9K_MLA_PREFILL=stock` defers to stock), so the sparse layer
  keeps the top-k MQA prefill path it always used here; `tests/test_integration.py` checks both halves.

- **Qwen3.8-27B on the 0.32-era nightly: GPU page fault on the first request.** Stock's Triton attention impl now
  advertises fp8 *query* input on gfx12 (`supports_quant_query_input`, formerly CUDA-only), so with the 27B's fp8 KV
  cache the attention layer quantized q before `forward` and our paged prefill kernel read the half-size fp8 buffer
  as bf16. `R9kTriton3DImpl` pins the flag to False and falls back to stock for any non-16-bit query or output.
  Found by bisection: stock `TRITON_ATTN` served 0 bad of 440; GDN and the drafter were cleared first.
- **27B on the 0.32-era nightly: every DFlash draft rejected under HIP graphs (1.0 tokens a step, 43 tok/s).** Those
  nightlies let the speculator skip the per-step metadata build for attention backends that update draft metadata in
  place inside the graph. Our backend declares kernel block 16 for the paged kernels, so vLLM remaps the hybrid model's
  1808-token KV blocks into a kernel block table at build time, and nothing refreshed it under graph replay: the drafter
  attended to stale blocks. `R9kTriton3DMetadataBuilder` now declines the in-place update (metadata rebuilt every step,
  as before that feature): 217 tok/s mean, 4.46 tokens a step, 0 bad of 440 -- the pinned numbers. Bisected with
  compiled runs: stock attention (both layouts) healthy, ours healthy without graphs, ours + graphs broken, ours +
  graphs + rebuild healthy ([notes/vllm-0.32.md](notes/vllm-0.32.md)).
- **Our Triton-attention builder no longer reports `max_query_len = 1` for small verify batches on nightlies whose
  `unified_attention` runs multi-query batches on its split-KV path itself (`MAX_3D_QUERY_LEN`)**: the new kernel sizes
  its query grid from that value. The paged prefill path is unchanged.
- **Flash-Next on the 0.32-era nightly: two loader-time breaks.** The int6 PLE loader imported stock's
  `compute_ple_shard_overlap`, which moved into `PLEVocabParallelEmbedding._compute_shard_overlap`; the three-line
  overlap arithmetic now lives in `ple/int6.py`. The hyper-connection mix hook read `GatedResidual.lora_rank`, which
  the nightly renamed to `hc_lowrank`; `hc.py` reads either (or the down projection's width).
- **Flash-Next TP4 at `UTIL=0.98` no longer starts on torch 2.13**: the runtime's per-process baseline on an idle
  card grew to ~0.7 GiB, so 31.22 GiB of 31.86 is not free at startup. Use 0.97 on that nightly.

### Changed
- **The PLE short-conv prefill path is retired** (`ple/short_conv.py`, off unless `R9K_PLE_CONV=r9k`). The October
  nightlies replaced `Qwen4ExpPLELayer._short_conv_dilated_prefill_batched` with a fused Triton kernel
  (`ops/ple.py: ple_conv`): bit-identical to ours and 8x faster on gfx1201 (496 vs 4,120 us per 4,096-token chunk,
  80 vs 420-530 MiB peak). Our hook had silently had nothing to bind to since the fe4df3df pin, and
  `tests/test_ple_conv.py` crashed with an `AttributeError`; the test now compares the two implementations and times
  them. Found while testing the vLLM 0.32-era nightly ([notes/vllm-0.32.md](notes/vllm-0.32.md)).
- `overlay/emulated-switch/patch-hostcall.sh` reports compressed offload bundles (CCOB), which its byte patch cannot
  reach, and lists the remaining hostcall kernels by family: on the 0.32-era nightly `_rocm_C` ships compressed and
  its 782 hostcall kernels are all gfx9-only (`paged_attention_ll4mi`, `wvSplitK`), so the unpatched copy serves.

## [0.3.6] - 2026-10-25

Release notes with the validation record: [notes/release-v0.3.6.md](notes/release-v0.3.6.md).

### Fixed
- **Flash-Next checkpoint revision 6fe3e746 (2026-10-01) loads and drafts again** -- Altay Sansal (@tasansal),
  [#1](https://github.com/bkvargyas/r9700-stack/pull/1). The revision's MTP attention KV scales crashed the loader
  and its bf16 MTP MLP defeated the MXFP4 conversion (shared expert dropped, 0% draft acceptance). The scales are
  dropped like the main model's; the bf16 MLP is quantized to MXFP4 at load, with the MSE-searched block exponents
  (`quant/nvfp4.py`). Older fp8 MTP checkpoints are unaffected; Flash-Next TP4 on the September revision re-measured
  within noise of v0.3.5.

## [0.3.5] - 2026-10-22

Release notes with the validation record: [notes/release-v0.3.5.md](notes/release-v0.3.5.md).

### Fixed
- **GLM-5.3-Flash long-context retrieval.** The indexer's K cache is written in pages of 64 pooled entries in 16x16
  tiles; stock vLLM reads it back correctly only on gfx942/gfx950 and falls through to a linear-page kernel
  elsewhere, so the decode-time scores were scrambled (top-512 overlap with the reference 70% at 760 pools, 3% at
  32k) and needle lookups failed from ~12k tokens on. `r9700_vllm/attn/indexer_logits.py` reads the tiled layout
  (matches the torch reference to 0.1 at every size up to 65k pools, 92-100 us a call against the stock kernel's
  153-158); `_install_paged_logits` swaps it in where stock would fall through (`R9K_INDEXER_LOGITS=stock` restores).
  Needles found at 12.7k / 23.7k / 34.7k and, on a 262k launch, at 67k / 150k / 216k; decode at depth 92-110 tok/s.

### Added
- `tests/test_indexer_logits_r9k.py`: the cache written by the real kpool writer, stock's reader and ours against the
  reference at 760 to 65k pools, one row and MTP verify rows.

## [0.3.4] - 2026-10-22

Release notes with the validation record: [notes/release-v0.3.4.md](notes/release-v0.3.4.md).

### Added
- **GLM-5.3-Flash on eight R9700s** from the same plugin on the same nightly: 79.6 tok/s single stream with three
  MTP drafts, 158 tok/s aggregate at four streams, GSM8K 97/100, 262k-token launches (`notes/glm53.md`). The pieces,
  all scoped to the glm5next module path and world size 8: the indexer AITER gate, the sparse-MLA kernel over an fp8
  or bf16 pool (`kernels/r9k_sparse_mla.hip`), the skinny fp32 indexer matmul, post-load per-row fp8 for the KDA
  projections (`R9K_FP8_LINEARS`), the in-graph draft metadata refresh (all draft steps in one graph), and the
  hierarchical 8-rank all-reduce / all-gather (`comm/r9k_ar8.py`: 8 KB cross-stage slices, RCCL above 40 KB).
- NVFP4 MoE experts served natively (`R9K_NVFP4_MOE=native`, exact checkpoint bits); the default stays the MXFP4
  conversion (same speed and accuracy measured, 1.1 GB a card of KV).
- `serve/serve.sh`: `SERVED_NAME`, `KVDTYPE=fp8`, `SPEC_EXTRA` on the MTP branch; `bench/sanity_stress.py THINK=1`.

### Unchanged
- 27B TP2 and Flash-Next TP4 re-validated on this tree (probe, BetterBench, soaks, strict sanity, comm tests,
  single-GPU gates): within noise of v0.3.3.

## [0.3.3] - 2026-10-13

Release notes with the validation record: [notes/release-v0.3.3.md](notes/release-v0.3.3.md).

### Changed
- **The image pin moves to the 2026-10-12 vLLM ROCm 10 nightly** (`nightly-rocm100-fe4df3df`, vLLM
  0.31.1rc1.dev296, torch 2.12+rocm10.0; `docker/Dockerfile`), from the September one. Same kernels; the plugin's
  October compatibility fixes carry it. Flash-Next TP4 decode step 15.0 -> 14.3 ms, single-stream 219 -> 225 tok/s,
  BetterBench combined 176.8 -> 188.0 and 1 / 2 / 4 / 8 streams 172 / 261 / 403 / 514 -> 175 / 277 / 410 / 530,
  prefill and KV cache unchanged (`notes/decode-step.md` round 4). The emulated-switch overlay
  needs its patched pair regenerated for this image (`overlay/emulated-switch/patch-hostcall.sh`). The routing and
  fold patches (`R9K_MOE_ROUTE`, `R9K_MOE_FOLD`, off since v0.3.0) do not start on this vLLM and are not re-ported.

### Added
- The fused MoE decode path (`R9K_MOE_FUSE=1`, `kernels/r9k_moe_mxfp4a8.hip` FUSE 1 / 2): silu(gate) * up in the
  gate_up GEMM's epilogue, the per-row fp8 quant inside the down GEMM's A-load, bit-identical outputs
  (`tests/test_moe_fused.py`). Off: it measures the same step time as the three-launch path.

## [0.3.2] - 2026-10-10

Release notes with the validation record: [notes/release-v0.3.2.md](notes/release-v0.3.2.md).

### Changed (`notes/decode-step.md` round 3; gate: GSM8K 95.75 vs 95.83 paired p = 1.00, HumanEval 162/164)
- The fp8 hyper-connection kernels take any row count (token tiles of 16, `R9K_HC_F8_MT` tiles per block);
  `R9K_HC_FP8_DECODE_MAXM` sets the widest row count on them (now 255 by default, was 16). `R9K_HC_FREE_BF16=1`
  (with `R9K_HC_FP8_DOWN=1`, both now default) drops the bf16 up / down weights after quantisation: **KV cache
  452k -> 516k tokens at TP4 (+14%), 167k -> 221k at TP2 (+33%)**, 8 streams +5.6% and 16 streams +3.4% at TP4,
  8k prefill -0.8% (the tiled fp8 down GEMM's activation quant), single-stream unchanged; the fp8 decode path is
  on at TP2 as well now (the freed pair outweighs its down copy). `R9K_HC_FREE_BF16=0` restores v0.3.1.

## [0.3.1] - 2026-10-10

Release notes with the validation record: [notes/release-v0.3.1.md](notes/release-v0.3.1.md).

### Performance (decode step, Flash-Next TP4; `notes/decode-step.md`)
- **fp8 hyper-connection decode GEMMs** (`kernels/r9k_hc_f8.hip`, `R9K_HC_FP8_DECODE=1`): the down+hc_silu and
  up+gated-mean kernels at decode widths (M <= 16) read the fp8 fragment-order copies the prefill mix path already
  holds (half the bytes of the bf16 kernels; the hc weights were the step's largest stream at 1.29 GB), bf16 WMMA
  on the activations (W8A16, products exact), per-row scale once. 16.42 -> 15.33 ms/step.
- **Fused activation quant in the dense fp8 block GEMM** (`r9k_gemm_fp8_block_qa`, `R9K_FP8_QA=1`, M <= 16): the
  per-token-group-128 quant runs inside the GEMM's A-load, operands bit-identical to the two-launch path; 96
  launches fewer a step. 16.42 -> 16.06; with the fp8 hc decode **15.01 ms/step (-8.6%, single-stream decode
  199 -> 217 tok/s**, 8/16 concurrent and prefill unchanged). Both are the four-card defaults; at TP2 the fp8 hc
  decode stays off (its down copy costs 0.43 GiB of KV a card there, 27k tokens, for +10% single-stream only). Measured and rejected: a 32 KB one-shot all-reduce cap (+0.35 ms), split-K across blocks for
  the hc down kernel (`R9K_HC_FP8_DECODE_KB`, slower). Tests: `tests/test_hc_f8.py`, `tests/test_fp8_qa.py`.


### Fixed
- **Runs on the 2026-10-09 vLLM ROCm 10 nightly** (`8cbd5d03`, 0.31.1rc1, 360 commits past the September pin)
  with the same kernels (`notes/vllm-upgrade-2026-10.md`): the int6 PLE table on vLLM's new PLE embedding
  classes (`allocate_embedding_weight`), the attention backend's block-size query signature, and a workaround
  for an upstream bug that quantizes the DFlash2 drafter's grouped-conv projections despite the checkpoint's
  exclusions (acceptance 4.3 -> 1.0 tokens a step; `models/dflash.py alias_draft_exclusions`). Probes on the
  new nightly: 27B 216 tok/s (4.48 tok/step), Flash-Next TP4 209 tok/s, prefill unchanged, sanity clean. The
  v0.31.0 *release* image is ROCm 7.2.3 and is not a target. `bench/humaneval.py` takes `EVAL_BASE`.

## [0.3.0] - 2026-10-08

**Flash-Next on four cards: 70% more KV cache, 4% more prefill, same decode, on new measured defaults.** Validated on
every configuration on this code (`notes/release-v0.3.0.md`).

### Changed
- **Flash-Next TP4 defaults** (`serve/flashnext.sh`, 2026-10-07): the fp8 hyper-connection up GEMM with the
  gate mix fused (`R9K_HC_FP8=r9k`, below), memory utilization 0.98 and cudagraphs only for prefill chunks up to
  256 tokens. KV cache 279,564 -> 442-475k tokens (+59-70%), 8k prefill 6,769 -> 6,852-7,069 tok/s, decode step
  and concurrency unchanged (16.4 ms, 199 tok/s, 888-905 tok/s at 16). vLLM's profiling run was counting
  torch.compile transients as activation (3.83 GiB against a measured 0.6 GiB model peak, `compat/memsnap.py`),
  and the graphs for 384-2048-token chunks cost 1.4 GiB a card for no measured throughput. Soaked: 25 minutes,
  684 requests, 0 errors, 31.8 of 32.6 GB at the peak; strict sanity 0 bad of 3,940 at concurrency 17 / 32 and
  on long prompts. `KVMEM=7.0` pins the budget at 475k tokens (0.95 GiB free at the peak) when the
  launch-to-launch estimate matters; `UTIL=0.94 CGSIZES=1,...,2048 R9K_HC_FP8=stock` restores the old defaults.

### Added
- fp8 prefill GEMMs on the LDS-tiled WMMA kernel (`r9k_moe_4bit_prefill` nv=3, `kernels/fp8.py`): e4m3 weights
  in the fragment order, one scale a column, 163-182 TFLOPS at 4096 rows (1.3-1.9x hipBLASLt bf16); the dense
  fp8 linear modes take it from 64 rows. The fused hyper-connection up GEMM + sigmoid-gated mean
  (`r9k_fp8_prefill_mix`, weight rows interleaved 16 columns x 4 streams): 268-298 us against the 665-674 us
  bf16 pair at 4096 rows, exact against the stock formula on the same gate; the hc weights are quantized after
  load (`hc.quantize_fp8`). An exact block-scaled variant (`r9k_fp8_prefill_block`, `R9K_FP8_BLOCK_PREFILL=r9k`)
  is correct but 15-25% slower than vLLM's Triton block GEMM on the served shapes: opt-in.
- Prefill all-reduce pipelining by row parts (`comm/pipe.py`, `R9K_AR_PIPE=r9k`, default stock): the layer tail
  (hc combine + mix, MoE, both all-reduces) in 2048-row parts with the attention parts' all-reduces on a comm
  stream (a second ar4 / 2-rank instance; `R9K_AR_PIPE_BLOCKS`, `R9K_AR_PIPE_SDMA`). Correct and gated, but the
  all-reduce kernels time-share the CUs with the GEMMs on this topology: 8k prefill -5.5% (fused pushes) / -6%
  (DMA pushes). Shelved in the tree; `notes/prefill-fp8-pipe.md`.
- `R9K_MEMSNAP=1` (`compat/memsnap.py`, with `EAGER=1`): per-layer / per-part peak memory of the profiling run
  and an allocator snapshot.
- Decode-step fusions on Flash-Next TP4 (`notes/decode-nodes.md`): one-launch MoE routing (`r9k_moe_route`:
  softmax top-k + align tables, replacing four stock launches a layer), the top-k sum with the shared expert's
  output folded in and the modular kernel's output alias on ROCm (`r9k_moe_sum`, `moe/fold.py`), and a one-shot
  P2P all-gather for decode-sized gathers at TP > 2 (`r9k_ag_oneshot_nrank`, 7.8 us vs RCCL 57 at [4, 640]).
  The all-gather alone: step 16.70 -> 16.40 ms, decode 196 -> 200 tok/s; everything on 16.25 ms. GSM8K and
  HumanEval unchanged (paired p = 1.00). Only the all-gather is on by default (`R9K_AG=0` reverts, `R9K_AG_MAX_KB`
  128); the routing and the fold are off (`R9K_MOE_ROUTE=r9k`, `R9K_MOE_FOLD=r9k`, `R9K_FUSED_SUM`): 0.15 ms a
  step was not worth their surface on the production path.
- `notes/mes-timeouts.md`: the RDNA4 `MES(1) failed to respond to msg=INVALIDATE_TLBS` timeouts under KVM
  passthrough, the host crash they caused once, the experiment matrix (three firmware versions, power, page-table
  mode, RAS: no effect) and the workaround now on both of our boxes, `amdgpu.mes_log_enable=1 amdgpu.gpu_recovery=0`
  (0 timeouts in 60 launches vs ~1 per launch, 36 of them under a sanity-plus-soak cycle). Reported to AMD on drm/amd #5759.
- `notes/radiance-test-run.md`: radiance 1.0.4 (its own C++/HIP engine, not vLLM) on two R9700 through our release
  checklist. 27B FP8: correct and stable, equal or better quality, 549k-token KV, 38 s start, decode 112 vs our 200
  tok/s. Flash-Next: 2-6% wrong or off-format answers in batches of 9-16 and prefill collapsing under concurrent long
  prompts; quality at concurrency 1 equal to ours. Re-tested on 1.0.8 (2026-10-05): no change on two cards.

## [0.2.5] - 2026-10-04

The v0.2.4 code with its validation record, and two housekeeping fixes. No behaviour change.

### Fixed
- The plugin announced itself as 0.2.3 in the startup log of the v0.2.4 tag; the version string is now right.

### Added
- The full release checklist for v0.2.4, run the day it was tagged and all green, is in
  `notes/release-v0.2.4.md` (GSM8K full set and HumanEval on all three configurations, the paired thinking-mode
  GSM8K against the September baselines and the reference stack, full BetterBench, mixed soaks, strict sanity under
  overload: 0 bad of 8,340) and summarized in PROGRESS.md 2026-10-03.
- `R9K_GDN_TRACE=1` (diagnosis, off by default) now also covers the draft-less decode path.
- `notes/bug-mes-invalidate-tlbs.md`: the write-up of the gfx1201 MES `INVALIDATE_TLBS` timeouts and the reset that
  took a card off the bus, as a comment for drm/amd issue 5759, with the result that the 2026-09-11 MES firmware
  (0x93) does not reduce them.

## [0.2.4] - 2026-10-03

One state page per request for the GDN layers under speculative decoding: 64% more KV cache for the 27B on two
cards, 18% for Flash-Next on two cards, 9% on four, and four concurrent 27B requests fit on one card where stock
pages admit two. Tagged on the unit gates and strict sanity under overload on every configuration (0 bad of
5,680), by decision; **the full checklist ran the same day, all green**: GSM8K full set and HumanEval on all three
configurations, the paired thinking-mode GSM8K against the September baselines and the reference stack (no
detectable difference), a full BetterBench on all three (decode level or better than the 225 W references,
concurrency up, prefill 0-11% lower), 25-minute mixed soaks with flat VRAM, and 0 bad of 8,340 under overload
(`notes/release-v0.2.4.md`). The test box also moved to 210 W and a -42 mV voltage offset on every card this day;
the probe numbers in this entry are at that setting, the earlier entries at 225 W.

### Added
- **One GDN state page per request under speculative decoding** (`R9K_GDN_STATE=onepage`, the default when
  prefix caching is off; `stock` restores vLLM's pages). vLLM keeps 1 + spec-tokens mamba pages per request so it
  can verify candidates from any accepted position; the plugin's verify kernel instead replays the accepted rows of
  the previous step from a record carried in the page, so one page serves. 27B on two cards: KV cache 223,329 ->
  367,494 tokens (+64.5%), 8 running requests pin 11.5% of the pool instead of 59.4%, decode and 8-way concurrency
  unchanged or better (206 -> 211, 534 -> 586 tok/s), prefill -4%. 27B on one card: 35,576 -> 62,295. Flash-Next
  on two cards: +18%. Bit-identical to the slot design (unit test with multi-step simulation); strict sanity under
  overload 0 bad of 1,890 across the three configurations. Stage 1 requires `mamba_cache_mode` "none"
  (`PREFIX_CACHE=0`); with prefix caching on, stock pages are used. One defect found and fixed before release: with
  one page per request the KV group's block table has a single column, and the conv update's candidate width was
  taken from it on eagerly built steps (mixed prefill+decode batches), rolling the conv window as if one candidate
  per step; the width now comes from the record (PROGRESS.md 2026-10-03 has the trace that found it).
- `tests/test_gdn_onepage_r9k.py`; kernels `r9k_gdn_spec_verify` / `r9k_gdn_spec_commit`; model classes
  `R9kQwen3_5ForConditionalGeneration` / `R9kQwen3_5ForCausalLM`.

### Changed
- Test and production cards run at 210 W with a -42 mV GPU voltage offset (LACT, overdrive enabled in the guest).
  Against 225 W without the offset the 27B on two cards gains ~5% decode from the offset and loses 0.7% decode,
  1.4% at 8 concurrent and 2.2% on an 8k prefill from the lower cap; strict sanity under overload 0 bad of 2,040 at
  the new setting. The 225 W reference numbers in README.md are unchanged and labelled.

### Fixed
- Guest power-cap service: an inline comment in `/etc/r9700-powercap.conf` was read as the cap value and the service
  failed at boot, leaving every card at 225 W.

## [0.2.3] - 2026-10-02

Two things: a short prompt's first token in about half the time (the 27B now level with the reference stack), and
28% more KV cache for Flash-Next on two cards. **Tagged on a smoke test, by decision, not on the v0.2.2 checklist**:
unit gates, strict sanity under overload on all three configurations (0 bad of 3,850), the two long soaks for the
memory change; no quality eval, no full BetterBench. The outputs for short prompts are numerically closer to the
exact recurrence than before, which is a change, so an eval is owed with the next release.

### Changed
- **Time to first token on short prompts: about half.** A prefill step ran vLLM's chunked GDN core eagerly in every
  GDN layer -- six Triton kernels plus glue, ~30 launches and ~1.2 ms of Python per layer whatever the prompt
  length: 48 layers on the 27B, ~60 of the ~97 ms a short prompt waited. Steps with few prefill tokens now run the
  recurrence token by token in one launch per layer (`r9k_gdn_seq`, the arithmetic decode already uses) after the
  causal conv in one launch (`r9k_gdn_conv`, bit-identical to stock's). Client-side median, one request at a time:

  | prompt tokens | 45 | 83 | 159 | 236 | 320 | 404 | 656 |
  |---|--:|--:|--:|--:|--:|--:|--:|
  | 27B, two cards: v0.2.2 | 97 ms | 96 | 147 | ~147 | 148 | 148 | 172 |
  | 27B, two cards: now | **47** | **60** | **81** | **92** | **128** | 142 | 172 |
  | reference stack, same cards | 46 | 68 | 93 | | 125 | | 158 |
  | Flash-Next, four cards: v0.2.2 | 87 | 87 | 88 | 88 | 92 | 107 | 131 |
  | Flash-Next, four cards: now | **44** | **58** | **73** | 87 | 91 | 107 | 132 |

  The same step is what every running request waits through when a new one joins the batch. Token by token is
  slower on the GPU than the chunked form (~1.8 us per token and layer), so it only pays while the step is bound
  by launches rather than compute: `R9K_GDN_PREFILL_MAX` (prefill tokens per step) defaults to 256, where
  Flash-Next on four cards breaks even, and `serve/27b.sh` sets 400 (the 27B breaks even near 440). `=0` restores
  vLLM's core; `R9K_GDN_PREFILL_CONV=stock` keeps its conv. Decode, long-prompt prefill and memory are unchanged;
  an older `libr9k.so` without the two symbols simply keeps the chunked core.
- Output for prompts under the threshold is no longer bit-identical to v0.2.2: the token-by-token fp32 recurrence
  is ~300x closer to an fp64 reference than the chunked bf16 form it replaces (relative error 2e-5 against 5e-3),
  which changes low-order bits of the hidden state and so, now and then, a token.

- **Flash-Next on two cards (offloaded experts): `--gpu-memory-utilization` 0.94 -> 0.96, 28% more KV cache**
  (121k -> 156k tokens on a restarted launch, 95k -> 129k on the first). 0.2.2 rejected 0.96 because it died with
  69 MiB free; that allocation was the short-conv prefill batch fixed in the same release, and it had not been
  retested since. At 0.96 now: the soak that killed it (8 clients, 18k-30k-token prompts, 16 minutes) served 86
  with none failed, the mixed soak (16 clients, 200-30k tokens, 26 minutes) served 227 with none failed, VRAM
  flat at 30.4 of 32.6 GiB, sanity after long prefills 0 bad of 180. Those ran on a first launch; on a restarted
  one (the larger pool) the first 5.7 minutes of the mixed soak peaked at 31,350 of 32,624 MiB with no error, and
  the run was stopped there -- that case has not had a full-length soak. `UTIL=0.94` restores the old value; four
  cards and the 27B are unchanged.

### Added
- `tests/test_gdn_prefill_r9k.py`: the new core against an fp64 token-by-token reference (and stock's distance to
  the same reference), row-mapped runs bit-identical to compact ones, spec-decode sequences bit-identical to
  `r9k_gdn_decode_mtp`, the conv bit-identical to stock's output and state in every layout.
- `bench/ttft_breakdown.py`: time to first token by prompt length next to the server's own queue / prefill /
  iteration counters -- the measurement that showed one 90 ms engine step and nothing else.

### Checked
- Unit gates: the new test, `test_gdn_decode_r9k.py`, `test_gdn_merge.py`, `test_moe_mxfp4.py`, `test_cache_moe.py`,
  `test_gemm_fp8.py`, `test_ple_conv.py`.
- Strict sanity with more requests than `max_num_seqs` (so prefills keep joining running batches), after short and
  after long prefills: 0 bad answers of 3,850 across the 27B, Flash-Next on four cards and Flash-Next on two.
  (Run with the threshold at 512, which sends more steps down the new path than the shipped 256 / 400.)
- Concurrency probes on Flash-Next four cards, three per launch, with and without: 884-887 against 866-880 tok/s
  at 16 -- no change. The 27B has no like-for-like pair yet (527-534 tok/s at 8 with; one probe without).

## [0.2.2] - 2026-10-02

A fix release. Three bugs, all present since 0.1.0, all invisible to fixed-depth benchmarks and to evals at
concurrency 1; each was found by loading the server the way real traffic does. **Upgrade.**
On two cards (Flash-Next with offloaded experts, and the 27B): wrong output for some requests under concurrency, and
(Flash-Next) a memory leak that ends in out-of-memory. On any Flash-Next configuration, four cards first: out of
memory when one step holds prompts of very different lengths. Speed and (correct) outputs are unchanged.

### Fixed
- **Garbled answers on two cards when a request joins a batch in flight.** A race in our 2-rank P2P all-reduce
  (`kernels/r9k_ar.hip`): the half of its double buffer was chosen by a per-block counter while the number of blocks
  followed the message size, so after a small message the next larger one could overwrite the previous message's
  last rows before the other rank had summed them. The last sequence of the batch then decoded garbage (the right
  beginning, then one token repeated). With more requests than `max_num_seqs`: 34-41% of rounds had a bad answer on
  Flash-Next TP2 (5-8% of answers), 8 of 900 answers on the 27B; after the fix 0 of 10,440. Every call now advances
  every block counter (fixed launch grid), and the compressed path shares the exact path's counters. The 4-rank
  kernels had the same construction and are fixed the same way. `tests/test_ar_race.py` reproduces it with the old
  grid and passes with the new one. On 0.1.0-0.2.1, `R9K_R4D_AR=0` (RCCL all-reduce) avoids it.
- **Out of memory in the prefill of Flash-Next's short convolution when a step holds prompts of very different
  lengths.** The stock method (and our drop-in for it) packs all of a step's prefills into `[prefills, longest,
  4*hidden]` and holds about six such tensors: 3,000 + 15 x 70 tokens peak at 4.8 GiB. On four cards a mixed-length
  soak at 16 running sequences went from 29.4 to 32.2 GiB in 40 seconds and died. Now packed by length (groups of
  at most 1.25x the step's tokens), the same bits, 0.4-0.5 GiB. `tests/test_ple_conv.py`.
- **Memory leak in the expert cache: Flash-Next with offloaded experts ran out of VRAM under mixed-length prompts.**
  The fused LRU path kept one set of align buffers per batch shape per layer for good (28 MiB per rank at a
  4,096-token chunk), and under real traffic nearly every prefill step has a new shape. At the 0.2.1 defaults a
  15-minute soak of 18k-30k-token prompts from 8 clients took a card from 30.5 to 32.6 GiB and killed the engine
  (77 requests served, 369 failed); fixed, the same soak serves 111 with none failed and VRAM flat at 31.0 GiB,
  and a 31-minute soak from 16 clients (270 to 28.5k tokens) serves 291 with none failed. Now one buffer set per
  layer, reused by every step. `tests/test_cache_shapes.py`.
- The 0.2.1 notes give Flash-Next TP2-offload prefill as 2,208-3,839 tok/s, 5-14% above 0.2.0. Both are BetterBench
  figures on its shuffled-paragraph filler, and the gain exists only for such narrow prompts (the cache can now
  follow their experts). Real text prefills at about 2,300-2,600 tok/s on both versions. README and PROGRESS say so.

### Added
- `bench/sanity_stress.py`: the 8-way sanity check, strict (the answer and nothing else), in rounds, at any
  concurrency and optionally after long prefills. `bench/soak.py`: mixed-length long-context soak.
  `bench/prefill_kinds.py`: prefill by kind of prompt.
- Release checklist (notes/picking-up.md): stress at more requests than `max_num_seqs`, soak with mixed lengths,
  on every configuration, before a tag.
- A same-day baseline of the 27B against the reference stack on the same two cards (PROGRESS.md): decode 196.6 vs
  197.5 tok/s, prefill 81-87%, TTFT 116 vs 65 ms, concurrency 94-97%.

### Rejected, with numbers
- `UTIL=0.96` for TP2 with offload: ran at 69 MiB free and died when the queue drained (PROGRESS.md 2026-10-01).

### Validation
- On the release code, every configuration: 26 unit gates, the all-reduce suites, full BetterBench (speed unchanged:
  97.4 / 157.3 / 197.5 tok/s for Flash-Next TP2 + offload / Flash-Next TP4 / 27B TP2), strict stress over
  `max_num_seqs` (0 bad of 9,500), mixed-length soaks (1,077 requests served, none failed; VRAM peaks 30.9 / 30.2 /
  32.5 GiB of 32.6).

## [0.2.1] - 2026-09-30

Flash-Next on **two** cards with the experts in host RAM: new defaults, measured end to end, and the tools that
found them. Full BetterBench against the 0.2.0 defaults: single-stream decode 97.4 tok/s (93.8), TTFT 484 ms (555),
prefill 2,208 / 3,558 / 3,839 / 3,793 (2,099 / 3,163 / 3,434 / 3,329), concurrency 87 / 109 / 116 / 118 / 113
(84 / 104 / 108 / 111 / 95), time to first token at 8 concurrent 0.95 s (7.1 s). Four-card numbers and all
numerics are unchanged. And a correction: 0.2.0's decode fusions cost no accuracy.

### Changed
- `serve/flashnext.sh` defaults `NSEQ=8` (the launcher's default was 4; benchmarks passed 16). Each request holds
  18 KV blocks whatever its length, and at TP2 with offloaded experts the graphs and activations for 16 sequences
  took the room of the requests they were for: four to six ran, the rest queued. Pass `NSEQ=16` for TP4.
- `R9K_LRU_THRESH` defaults to `0.99` (was `0.5`). Eight different requests route to ~150 distinct experts a
  layer; over 0.5 x 270 slots the cache manager inserted nothing, on 72% of steps, and every step read 31 experts
  a layer from host (2 GB, 207 ms per forward pass). BetterBench conc-8 with all eight running: 77 -> 118 tok/s.
- `R9K_LRU_GATHER` defaults to `64,16` (was `8,16`): the expert insert copy is ~15% faster at 8+ inserts and equal
  below. Outputs cannot change.
- Quality evals for a change of default use the full 1,319-question test set, not the first 800.

### Added
- `R9K_EXPERT_CACHE_STATS=1`: opt-in counters in the expert cache (distinct routed experts, inserts, experts read
  through from host, steps over the insert threshold), logged every `R9K_EXPERT_CACHE_STATS_SEC` seconds per rank;
  `tests/test_cache_stats.py`. `bench/mix.py` (conc-8 by workload mix) and `tuning/lru_gather_bench.py` (host -> VRAM
  copy rate of the insert kernel).
- What bounds Flash-Next at TP2 with offloaded experts, measured (PROGRESS.md 2026-09-29): the PCIe link. A miss is
  1.245 MiB per rank and the copy runs at the link rate (11.4-13.5 GB/s on PCIe 3); one prompt type misses 4.5%
  of its routed experts per step, four types 14%, which is what the checkpoint's routing profile predicts for 270
  slots (13.3%). The cards draw 206 W with one request and 170 W with eight.
- The card-placement rule, measured: tensor parallel wants both cards on one PLX switch, offloaded experts want
  one card per switch (same-switch pair: 81 / 54 / 1,379 single / conc-8 / prefill-8k vs 114 / 132 / 2,464).
- Operational note in the launcher and README: the first launch of a configuration gets a smaller KV pool than
  every later one (94k against 121k tokens at `NSEQ=8`); restart once.

### Rejected, with numbers (PROGRESS.md)
- 240 expert slots (more KV; -28% on a mixed batch, -6% single stream), `NBT=2048` as a default (-24% prefill),
  re-splitting the slots across layers by routing mass (13.3% -> 12.8% misses). `UTIL=0.96` ran clean with 28% more
  KV but has not held a 32k context under load: not adopted.

### Fixed
- **The quality statement of 0.2.0 was wrong in the project's disfavour.** It said the round-3 decode fusions
  together sit about half a point below the round-2 numerics. On the full 1,319-question test set they do not: the
  default scores 97.04% against 96.82% (5 / 8 discordant, p = 0.58), each of the five fusions alone is equally
  indistinguishable (6 / 6, 0 / 2, 5 / 8, 5 / 7, 6 / 5), and the reference reproduces itself to 1,318 of 1,319
  outputs. The half point came from evaluating on the first 800 questions, where the round-2 numerics score 98.0%
  (95.0% on the other 519).
- PROGRESS.md reported a concurrency regression for Flash-Next TP2 with offloaded experts (conc-8 132 vs 206 on
  2026-09-20). There is none: the two figures came from different harnesses (eight copies of one prompt against
  four different prompts).

## [0.2.0] - 2026-09-29

Qwen3.8-Flash-Next on four R9700s (TP4) goes ahead of the fastest known alternative stack on every metric:
single-stream decode, time to first token, prompt processing at every depth, and aggregate throughput at every
concurrency level -- measured on a PCIe 3 host; PCIe 5 would be faster still. Full BetterBench, TP4, 225 W:
decode 159.2 tok/s vs 134.1 (+19%), step p50 16.8 vs 20.4 ms, TTFT 94 vs 145 ms, prefill 6,408 / 7,365 /
7,451 / 7,182 vs 5,279 / 5,711 / 5,977 / 6,106 tok/s at 2k / 8k / 16k / 32k, concurrency 151 / 231 / 350 / 478 /
635 vs 126 / 197 / 303 / 427 / 542 at 1 / 2 / 4 / 8 / 16. v0.1.0 had Flash-Next at ~85 tok/s on two cards.

### Added
- **Four-card support.** `GPUS` / `TP` / `PORT` / `NAME` launcher knobs, TP=4 MoE and dense configs, and the
  host work that makes four cards behind two PLX switches usable: 32 GB BARs for both cards on each switch
  (`host/`, DKMS `chainfix` 1.2), switch-local P2P on both PEX 8747s via a guest placement module (2× peer
  bandwidth, worth 9-10% of prefill).
- **N-rank P2P all-reduce** (one-shot and two-shot) for TP>2, and a **compressed hierarchical 4-rank
  all-reduce** for prefill-sized messages (on by default).
- **Sparse attention (QSA) for Qwen4Exp**: our own bitmap / attention / merge kernels, and a **WMMA indexer
  scorer** (`r9k_qsa_score`) that scores only the visible columns instead of all 8,192 (prefill +10-12%).
- **Decode fusions**, each replacing a run of tiny per-step launches with one kernel (every HIP-graph node costs
  ~1.5 µs of dispatch on this ROCm): a bf16 skinny **router GEMM** for the MoE gate; fused **Gemma norm + rope**
  for the sparse-attention indexer (1-D or MRoPE positions, any stride); the **hyper-connection** gate mix,
  combine-norm and decode input mix; the **shared expert in four launches** with its sigmoid gate folded into
  the down GEMM epilogue; and the **Gated DeltaNet speculative-decode core** (`r9k_gdn_decode_mtp`: gating,
  delta-rule recurrence over the draft tokens and the gated RMS norm in one launch per sequence and head,
  replacing ~11 graph nodes per layer).
- **Prefill kernels**: per-row output scales from LDS and LDS-staged 16 B row stores in both MoE prefill
  epilogues (TP=4 down GEMM 1,191 → 690 µs); tiles that accept K % BK for the TP=4 down GEMM; the PLE prefill
  short conv on a tiled kernel (26.5 → 4.1 ms per 4k chunk).
- **Hybrid block-fp8 dispatch** and MXFP4 LM heads for decode; TP=4 shapes in the tuning table.
- **Prefill chunks up to 2,048 tokens captured in HIP graphs** (`CGSIZES`), removing ~285 ms of CPU-bound
  eager forward from short prompts (TTFT 137 → 88 ms).
- `R9K_*=stock` knobs for every fusion, `R9K_GDN_DEBUG` / `R9K_GDN_FUSED` diagnosis knobs, `tools/`
  benchmark harnesses (`bench/tp4vs2.sh`, probes, paired eval).
- This changelog; `notes/independence.md` now lists every runtime hook beyond vLLM's extension points.

### Changed
- **Base image: vLLM 0.30 nightly (rocm100, torch 2.12).** The whole stack runs unchanged on it.
- **Prefix caching off by default for Flash-Next** (`PREFIX_CACHE=1` restores it): the short-prefill root cause.
- MTP-3 stays the default after measuring MTP-4 (+4.4% single-stream, -14% at 16 concurrent; `MTP=4` is a knob).
- Both routed MoE GEMMs default to the 64×128 BK=32 prefill tile at TP=2 and TP=4.
- Compressed all-reduce on by default; `ar4` at 128 blocks.

### Fixed
- The fused indexer rope silently fell back to stock in serving: positions arrive as a stride-3 column view of a
  `[T, 3]` buffer, which the kernel now accepts.
- The Gated DeltaNet core must be a piecewise-graph splitting op like the vLLM op it replaces; without that a
  prefill piece captured it with stale kernels and the first request faulted. Registered at plugin load.
- Flash-Next's output gate is `sigmoid`, not silu; the GDN core supports both and its fit check logs its reason.
- Custom-op signatures carry type annotations (schema inference refused the untyped tensors at engine start).
- A staged-store epilogue overwrote its own row tables at cfg 17 (memory fault); the tables now sit past the stage.

### Quality
- Every change to a shipped default is checked by an 800-question chain-of-thought paired eval at concurrency 1
  (paired McNemar), plus the 8-prompt sanity set. The v0.2.0 default scores 97.50% on that eval (no detectable
  difference from the previous default alone, p = 0.22); across runs the round-3 decode fusions together sit
  about half a point below the round-2 build's 97.9-98.0%. `R9K_ROUTER=stock R9K_QSA_GLUE=stock R9K_HC_MIX=stock`
  restores round-2 numerics at ~10% of decode. Details in PROGRESS.md.

## [0.1.0] - 2026-09-22

First release: tuned gfx1201 kernels and a vLLM plugin running Qwen3.8-27B-NVFP4 and Qwen3.8-Flash-Next on 2×
R9700 with stock vLLM and stock ROCm 10.

### Added
- MXFP4 × FP8 MoE GEMM in three shapes (decode with in-block split-K, LDS-tiled prefill with a 17-entry tile
  table, fragment-tiled-activation prefill).
- Paged attention for head_dim 256 (GQA 1-8, block 16, bf16 and fp8 KV).
- 2-rank P2P all-reduce with a compressed path (Walsh-Hadamard rotation + 4-bit groups) for prefill messages.
- Expert LRU cache streaming from pinned host memory, int6 PLE gather, dense fp8 GEMM for LM heads, load-time
  NVFP4 → MXFP4 conversion.
- Registration entirely through vLLM's extension points; Apache-2.0 with one carve-out (see `NOTICE`).

[Unreleased]: https://github.com/bkvargyas/r9700-stack/compare/v0.3.8...HEAD
[0.3.8]: https://github.com/bkvargyas/r9700-stack/compare/v0.3.7...v0.3.8
[0.3.7]: https://github.com/bkvargyas/r9700-stack/compare/v0.3.6...v0.3.7
[0.3.6]: https://github.com/bkvargyas/r9700-stack/compare/v0.3.5...v0.3.6
[0.3.5]: https://github.com/bkvargyas/r9700-stack/compare/v0.3.4...v0.3.5
[0.3.4]: https://github.com/bkvargyas/r9700-stack/compare/v0.3.3...v0.3.4
[0.2.2]: https://github.com/bkvargyas/r9700-stack/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/bkvargyas/r9700-stack/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/bkvargyas/r9700-stack/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/bkvargyas/r9700-stack/releases/tag/v0.1.0
