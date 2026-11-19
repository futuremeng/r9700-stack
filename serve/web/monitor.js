// serve/web/monitor.js -- the Monitor. Two fetches per tick: /api/status (the engine's counters, read
// off vLLM's /metrics) and /api/requests (what came through this page, when it is in the path).
"use strict";
let records = [], status = null, selected = null, detail = null, pane = "input", stampOf = "";
let detailSig = "", follow = true;                    // follow: show the newest row nobody clicked on yet
let paused = false, actionError = "";
// Which window the four headline rates answer to, and which ring the four waveforms draw. Defaults are the
// averaged ones: measured on this box one 1 s poll read 9.9 tok/s of generation while the quarter hour around it
// read 51.7, and a page that prints only the first number teaches the reader to read it as the level.
let rateWin = localStorage.getItem("r9k.rateWin") || "15m";
let trendWin = localStorage.getItem("r9k.trendWin") || "fast";
let lastOk = 0, lastErr = "";                // when a tick last got an answer, and why the last one did not

const active = (r) => !["completed", "error", "disconnected"].includes(r.state);
const rid = (r) => r.id || `req-${r.seq}`;

// -- one PCIe chain: the card's endpoint and every bridge above it ---------------------------------
// Link state and error counters, because the traffic counters are not there to ask: amd-smi's BANDWIDTH,
// CURRENT_BANDWIDTH_SENT / _RECEIVED, MAX_PACKET_SIZE and REPLAY_COUNT answer N/A for every GPU on these RDNA4
// cards, and the kernel exposes no byte counter either. What IS free (ordinary user, no root, no /dev/kfd) is the
// speed and width each link negotiated, and the kernel's AER counters for each device and bridge in the chain.
// Those counters are cumulative since boot, so the waveform is the per-poll delta: a spike is an event with a time
// that can be lined up against a stall, where the slope of a cumulative line hides it. A counter that went down is
// the kernel zeroing it (link recovery, driver reload) -- that draws a gap and says so, because a bar of -24
// would be a fabrication.
// AER counter names are not self-explanatory: `NonFatalErr` lives in the CORRECTABLE file (the spec calls it
// Advisory Non-Fatal Error -- an error the device reported but recovered from), and serve/webui.py tags the
// genuinely uncorrectable ones with `#nf` / `#ft`. Reading `NonFatalErr 3` next to `0 uncorr` would look like a
// contradiction, so the name gets its kind stated.
function aerKind(k) {
  const nf = k.endsWith("#nf"), ft = k.endsWith("#ft");
  return { name: k.replace(/#(nf|ft)$/, ""), kind: ft ? "fatal" : nf ? "non-fatal" : "correctable" };
}

// The PCIe panel was removed on 2026-11-05: a link and an error count per chain moved onto the card
// (gpuCard, g.link), and the error count stopped being printed at all unless a link negotiated below its
// rating -- this box carries dozens of correctables a minute through a weight load at a full 32.0 GT/s,
// so the panel's colour meant "a counter moved", not "the wire went short". What survives here is the
// account itself (pcieDetailLines / paintPcieDetail), opened from a card whose link IS short, and the
// `ago` helper it prints quiet spans with.
const ago = (s2) => (s2 >= 5400 ? `${(s2 / 3600).toFixed(1)} h` : `${Math.round(s2 / 60)} min`);

// -- the notice bar -------------------------------------------------------------------------------
function notice(message, info) {
  const el = $("error");
  el.hidden = !message;
  el.className = `notice${info ? " notice--info" : ""}`;
  if (message) el.textContent = message;
}

// -- the request list -----------------------------------------------------------------------------
// A request row leads with the two numbers it is judged by -- how fast its prompt was read and how fast its
// tokens arrived -- and both are THIS request's own (vLLM's per-request stats), not the box's averages. The
// rest is supporting: what it waited for, how long the pieces took, then what it was. Dropped on the way,
// because a list of 100 rows is read by scanning and clutter costs throughput: the endpoint line (every
// engine row says "no endpoint in its stats"), "engine record" / "stream" / "JSON" (how the row was obtained,
// which is a fact about the monitor), and the state tag on an ordinary row (1,429 of 1,429 engine rows on this
// box read "completed" -- the tag appears when a row is NOT ordinary).
function rqSpeed(label, value, note) {
  const el = document.createElement("span");
  el.className = "rq-speed";
  if (note) el.title = note;
  const l = document.createElement("i"); l.textContent = label;
  const n = document.createElement("b"); n.textContent = isnum(value) ? fmt(value, value >= 1000 ? 0 : 1) : "\u2014";
  const u = document.createElement("s"); u.textContent = isnum(value) ? " tok/s" : "";
  el.append(l, n, u);
  return el;
}
function mark(text, tone, note) {
  const el = document.createElement("span");
  el.className = "rq-mark";
  if (tone) el.dataset.tone = tone;
  el.textContent = text;
  if (note) el.title = note;
  return el;
}
function visibleRows() {
  const q = $("search").value.toLowerCase(), filter = $("filter").value;
  return records.filter((r) => `${rid(r)} ${r.path} ${r.model}`.toLowerCase().includes(q)
    && (filter === "all" || (filter === "active" ? active(r) : r.state === filter)));
}

function list() {
  const visible = visibleRows();
  const next = JSON.stringify([visible.map((r) => [rid(r), r.state, r.wallclock_s, r.shared_decode]), selected]);
  if (next === stampOf) return;                      // a repaint costs selection and scroll position
  stampOf = next;
  const focusId = document.activeElement?.dataset.request;
  $("requests").replaceChildren();
  if (!visible.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.innerHTML = records.length ? "No matching request."
      : status && status.mode === "observe"
        ? "<strong>Nothing is routed here</strong><span class='small'>This sidecar is only watching the serve "
          + "(GET /health, /v1/models, /metrics). Point your client at this port, or start it with --proxy, and "
          + "requests appear here.</span>"
        : "No request yet.";
    $("requests").append(empty);
  }
  for (const r of visible) {
    const b = document.createElement("button");
    b.className = `request${selected === rid(r) ? " selected" : ""}`;
    b.dataset.request = rid(r);
    b.setAttribute("aria-label", `Request ${rid(r)}`);
    b.setAttribute("aria-pressed", String(selected === rid(r)));
    const row = document.createElement("div"); row.className = "row";
    const time = document.createElement("span"); time.textContent = clock(r.started_at);
    const u = r.usage || {}, t = r.timings || {};
    // The engine's own mean per accepted token; fall back to the tokens over its own decode span.
    const decode = isnum(t.predicted_per_second) ? t.predicted_per_second
      : (isnum(u.completion_tokens) && isnum(t.decode_s) && t.decode_s > 0
        ? u.completion_tokens / t.decode_s : null);
    const marks = document.createElement("span"); marks.className = "rq-marks";
    if (r.state && r.state !== "completed") marks.append(mark(r.state, "danger", `state: ${r.state}`));
    if (t.preemptions) {
      marks.append(mark(`preempted ${t.preemptions}\u00d7`, "warn",
        "vLLM v1 re-prefills a preempted request: this one lost its KV blocks to fit another and paid for the "
        + "same prompt twice"));
    }
    if (t.finish_reason === 1) {
      marks.append(mark("hit max_tokens", "", "ended on the client's max_tokens, not on the model stopping"));
    }
    // Crowding is shown only when it is enough to explain something. Below 1.5 the box is not busy enough for
    // the number to change a reading of the rate, and a chip on every row outvotes the chips that mean
    // something (a preempted request, an error state).
    if (isnum(r.shared_decode) && r.shared_decode >= 1.5) {
      marks.append(mark(`shared ×${fmt(r.shared_decode, 1)}`, "",
        `mean requests in flight across this request's own decode window, itself counted, over the ${records.length}`
        + ` rows this page holds. Decode rate is read against it: on this box a stream runs 158 tok/s alone,`
        + " 119 at two, 23 at three or four -- so a low number here is a busy box, not a slow request"));
    }
    row.append(time, marks);
    if (r.path) {
      const ep = document.createElement("div"); ep.className = "endpoint";
      ep.textContent = `POST ${r.path}`;
      row.append(ep);
    }
    const sp = document.createElement("div"); sp.className = "rq-speeds";
    sp.append(rqSpeed("prefill", t.prefill_tok_s,
      "this request's own prompt tokens over its own prefill span"));
    sp.append(rqSpeed("decode", decode,
      "accepted tokens over this request's own decode span: queueing and prefill excluded, and shared with the "
      + "other streams running at the time"));
    const size = document.createElement("div"); size.className = "rq-size";
    size.textContent = (isnum(u.prompt_tokens) || isnum(u.completion_tokens))
      ? `${fmt(u.prompt_tokens ?? 0, 0)} → ${fmt(u.completion_tokens ?? 0, 0)} tok` : "";
    // Supporting detail, ordered by how much it explains about the two numbers above.
    // The clock basis differs by path and it is worth getting right here: an engine row's `wallclock_s` is
    // vLLM's inference_time -- it does NOT include queued_time -- while its `first_token_s` is queue + prefill,
    // which is why a queued request can print a first token LATER than its whole wallclock (a real row: first
    // token 21.8 s, wallclock 10.8 s, queue 12.3 s). The proxy path times the whole connection instead, queue
    // included. So the total is assembled per path rather than assumed.
    const meta = document.createElement("div"); meta.className = "small";
    const engineRow = r.response_format === "engine";
    const total = engineRow ? (isnum(r.wallclock_s) ? r.wallclock_s + (t.queue_s || 0) : null) : r.wallclock_s;
    // What a row carries, and nothing else: when it ran, what it cost the user (first token, whole request), how
    // big it was, and the id to pin it with. `prefill Xs` and `decode Ys` used to sit on this line too, which
    // printed the same fact twice -- the two tok/s figures one line above are computed FROM those spans -- and
    // pushed the request id off the end of the row. Both spans stay in the detail pane, where there is room to
    // say what they include.
    // A status is printed only when it is not what it should be: a list of requests that all worked is the common
    // case, and `HTTP 200` on every row is noise that competes with the marks that carry meaning.
    const bad = r.http_status && (r.http_status < 200 || r.http_status >= 300);
    meta.textContent = [
      isnum(r.first_token_s) ? `first token ${seconds(r.first_token_s)}`
        + (isnum(t.queue_s) && t.queue_s >= 0.001 ? ` (${seconds(t.queue_s)} of it queue)` : "") : "",
      isnum(total) ? `all in ${seconds(total)}${engineRow && isnum(t.queue_s) && t.queue_s >= 0.001
        ? " (queue included)" : ""}` : "",
      isnum(u.prompt_tokens_details?.cached_tokens) && u.prompt_tokens_details.cached_tokens > 0
        ? `${fmt(u.prompt_tokens_details.cached_tokens, 0)} cached` : "",
      bad ? `HTTP ${r.http_status}` : "",
      rid(r)].filter(Boolean).join(" \u00b7 ");
    b.append(row, sp, size, meta);
    b.onclick = () => { follow = false; $("follow").setAttribute("aria-pressed", "false");
      selected = rid(r); detail = null; list(); showDetail().catch((e) => notice(e.message)); };
    $("requests").append(b);
    if (focusId === rid(r)) b.focus({preventScroll: true});
  }
}

// -- one request, read in full ---------------------------------------------------------------------
function content() {
  if (!detail) return;
  // "No input." under the heading "The request body as it arrived." is the failure this function exists to avoid:
  // it reads as a lookup that found an empty body, when in fact nobody on this box kept one -- and the payload
  // already says which process handled the row and what it keeps. The sentence comes from the payload
  // (input_note), because the sidecar is the process that knows which of the three arms a bodyless row belongs to.
  const why = detail.input_note
    ? detail.input_note
    : (detail.source === "journal"
      ? "This row was restored from the request journal, which keeps timings and token counts but never the text."
      : "No process on this box kept a body for this request.");
  const how = "The only process here that ever holds both a request and its answer is the front door, and it keeps "
    + "a copy only of requests that ask: set GATEWAY_CAPTURE=<bytes> in the machine-local env and send the request "
    + "with the header X-R9K-Capture: 1. (A sidecar run as a relay instead -- WEBUI_MODE=proxy -- keeps every body "
    + "it carries, up to 256 KiB.)";
  let capNote = "";
  if (detail.capture) {
    // Merged in from the door's tap. The provenance goes in the note at the bottom, because a truncated copy of a
    // prompt and a complete API response look identical on screen.
    const c = detail.capture;
    capNote = ` Copied by the front door at ${stamp(c.at)}: ${fmt(c.bytes_in)} bytes in, ${fmt(c.bytes_out)} out`
      + (c.input_truncated || c.response_truncated ? ", truncated -- the API response itself was not shortened" : "")
      + (c.state === "incomplete" ? ", and the connection ended before the answer did" : "") + ".";
  }
  if (!detail.input && !detail.response && !detail.output) {
    text("content", why);
    // ONE recipe on the screen. When the sidecar is pointed at the front door its own sentence already names the
    // header and the environment variable; repeating them underneath it leaves a reader wondering which of the two
    // is current. The fuller fallback stays for boxes where nothing wired the two together.
    text("content-note", detail.says_capture
      ? "The same reason covers the other three panes: Output, Reasoning and API response are empty because no "
        + "process on this box kept anything for this id."
      : how);
    return;
  }
  let value = detail[pane];
  if (pane === "output" && !value && detail.response) value = "";
  if (value === undefined || value === null || value === "") {
    value = pane === "response" ? (detail.response || "The response body was not kept (an empty or non-JSON answer).")
      : pane === "reasoning" ? "No separate reasoning content in this response."
      : pane === "output" ? "The answer's text is not in the copy kept for this row -- the API response pane has "
                          + "what was captured, raw." : "No input.";
  }
  if (pane === "input" || pane === "response") { try { value = JSON.stringify(JSON.parse(value), null, 2); } catch (_) {} }
  text("content", value);
  text("content-note", (pane === "input" ? "The request body as it arrived."
    : pane === "output" ? "What the model said. Streaming text, concatenated."
    : pane === "reasoning" ? "reasoning_content, when the model separates it."
    : "The response body as it left the server.")
    + (detail.output_truncated && pane === "output"
      ? " Our copy is truncated at 256 KiB; the API response was not shortened." : "") + capNote);
}

async function showDetail() {
  if (!selected) return;
  const id = selected;
  const value = await api(`/api/requests?id=${encodeURIComponent(id)}`);
  if (selected !== id) return;                       // you clicked another row while this was in flight
  detail = value;
  $("selection").hidden = false; $("no-selection").hidden = true;
  // An engine-stats row has no path and no model: FinishedRequestStats carries neither, so printing the field
  // raw puts the word "null" in the heading of the panel a user is reading.
  text("detail-title", [value.id, value.path, value.model].filter(Boolean).join(" \u00b7 "));
  text("detail-status", value.state);
  $("detail-status").className = `tag tag--${value.state}`;
  text("detail-time", stamp(value.started_at));
  // The whole span the request occupied, which means different sums on the two paths: an engine row's
  // wallclock_s is vLLM's inference_time and does NOT include queued_time, while a proxy row timed the whole
  // connection. Adding the queue here is what makes "First token" below it readable -- without it a queued
  // request shows a first token later than its own wall-clock.
  const dEngine = value.response_format === "engine";
  text("detail-wall", seconds(isnum(value.wallclock_s)
    ? value.wallclock_s + (dEngine ? ((value.timings || {}).queue_s || 0) : 0)
    : (isnum(value.started_at) ? Date.now() / 1000 - value.started_at : null)));
  text("detail-first", seconds(value.first_token_s));
  const t = value.timings || {};
  for (const [id, k] of [["detail-queue", "queue_s"], ["detail-prefill", "prefill_s"],
                         ["detail-decode", "decode_s"]])
    text(id, seconds(t[k]));
  const u = value.usage || {};
  text("detail-prompt", isnum(u.prompt_tokens) ? fmt(u.prompt_tokens) : "—");
  text("detail-tokens", isnum(u.completion_tokens) ? fmt(u.completion_tokens)
    : value.timings?.tokens_are_estimate ? "≈ estimated" : "—");
  text("detail-speed", (isnum(t.predicted_per_second) ? fmt(t.predicted_per_second, 1) : "—")
    + (t.speed_basis && isnum(t.predicted_per_second) ? ` (${t.speed_basis})` : ""));
  // Computed when the record was made, not read off a counter: prompt tokens over that request's own prefill
  // time (engine), or over its own first-token latency (a relay). The basis is printed because the two differ.
  text("detail-prefill-speed", (isnum(t.prefill_tok_s) ? `${fmt(t.prefill_tok_s, 0)} tok/s` : "—")
    + (isnum(t.prefill_tok_s) && t.prefill_basis ? ` · ${t.prefill_basis}` : ""));
  text("detail-format", value.response_format || "text");
  $("detail-error").hidden = !value.error;
  text("detail-error", value.error?.message || "");
  content();
}

// -- the engine, the cards, and the timings ---------------------------------------------------------
function renderStatus(s) {
  const live = s.live || {}, rate = s.rate || {};
  const inflight = (live.running || 0) + (live.waiting || 0);
  // -- the headline row ---------------------------------------------------------------------------
  // Big numbers, and only from what moves per engine step. What does NOT get a headline here is a per-stream
  // rate: the payload still carries stream_tok_s / stream_tok_s_live (webui.py documents which of the two
  // histogram families is the honest one, and the request rows still use them), but the headline row shows a
  // count and a per-record total instead, because a per-stream average over a span divides by a gauge that
  // counts prefilling requests as running, and invites the comparison "aggregate is smaller than one stream"
  // that the step census in notes/context-262k.md answers.
  // -- fixed slots, so a number that changes width cannot move the page ---------------------------
  // Every line in these cards is a SLOT: fixed position, never wraps, ellipsizes, and carries its full text in
  // the title. The reason is arithmetic, not taste. A 4-up card is 177-322 px of text depending on page width
  // (main is capped at 1560 with 32 px padding, a 14 px gap, 42 px of card padding and border), the theme lays
  // text out at 0.60 em per glyph, and these lines are 12 px - so 24 characters fit the tightest 4-up width and
  // 44 the widest. A line that says "13 tok/s" and later "13,570 tok/s while prefilling" changes how many lines
  // the card needs, which changes the height of the whole grid row, which moves every card under it: the page
  // jiggles once a second and a reader stops trusting a number they see change. So the copy is short by
  // construction, and anything longer that a reader needs lives in the title.
  // needs lives in the title. (Not verified as pixels - there is no browser on this box; the arithmetic above
  // is the same one docs/monitor-cards.py and tests/test_readme.py use, and the harness fails a slot whose
  // text is longer than the widest 4-up card can hold.)
  const TT = {                                   // the sentences that do not fit in a slot
    win: "Which window the four numbers below answer to. A window is used only once it has filled (80% of its "
       + "width); if the chosen one has not, the widest span that has is used and this line says so. Each card's "
       + "second line carries the other spans, and its third line carries this poll.",
    fresh: "How old these numbers are: the sidecar polls the engine every poll window, the page refetches on the "
         + "same beat, and this is how long the snapshot has been sitting. When it grows, this tab has stopped "
         + "getting answers.",
    gentot: "generation_tokens_total, cumulative since this engine started -- summed over streams, no division "
          + "anywhere. Next to it: prompt_tokens_total over the same span, which is the ratio that explains this "
          + "page's decode numbers (99.2% of the tokens this box processed were prompt tokens, and the summed "
          + "rate is 1.62 streams a record x 2.33 tokens a step x 6.3 records a second). Per-span counts in the "
          + "title because only two fit the slot: ",
    gen: "Total decode output over WALL seconds: generated tokens divided by the wall seconds of the window. This "
       + "is the throughput reading of the box -- what a minute of this traffic produces -- so the queue, the "
       + "prefill phase, the wait before the first token and idle seconds are all IN the denominator, because they "
       + "are part of the wall. The note line is the other reading of the same traffic and the two are one "
       + "arithmetic: `N tok/stream-s ×M` = the big number, where N is output per DECODE second and M is streams "
       + "decoding per wall second (152 x 0.15 = 22 on a live burst). N's denominator is the stream-seconds the "
       + "engine's decode families report: vLLM observes one inter_token_latency per REQUEST x STEP "
       + "(v1/metrics/stats.py:551), so a step two streams share accrues twice -- correct here (each stream got a "
       + "token out of it) but why the unit says stream -- and it is the same quantity as a request row's speed "
       + "(1 / the engine's mean time per output token), which is why the card and the rows under it read alike: "
       + "18,118 tokens over 119.1 stream-seconds = 152 tok/s on a burst whose rows ran 160-200. M is a COUNT, not "
       + "a share (0.15 serialized, 2.60 loaded, no 100% ceiling) -- a percent on that line is a category error. "
       + "The two decode families agree to 0.08% (24,236.14 s over 430,954 observations against 24,216.27 s over "
       + "1,812 request observations) because they are one sum sampled per step and per request, not two "
       + "measurements. What the per-stream figure cannot remove: under chunked prefill a prompt chunk is computed "
       + "INSIDE a decode-phase step, so the step interval carries that work too (a bare single-stream step "
       + "measures 13.0 ms; this launch's live itl is 15.5 ms, the +19% no counter separates).",
    prefill: "READ tokens per second of REQUEST-prefill time -- the engine's own scheduled -> first-token span, "
           + "summed over streams, so stream-seconds and a per-stream speed. It is NOT the engine's chew rate: "
           + "under chunked prefill a request's prefill span contains the steps in which the engine was decoding "
           + "other requests, and if two prefills run at once this divides by both -- the note line then prints "
           + "`×N prefilling` (today 1.04: long prefills take the 8,192-token step one at a time, so they "
           + "serialize). READ, deliberately: vLLM credits cached and externally transferred prefix tokens to "
           + "prompt_tokens_total, so credited tokens would count work the engine never did and the rate would be "
           + "high by the hit rate; the numerator subtracts prompt_tokens_cached_total (the exact skipped count, "
           + "local + external) when the build publishes it and prefix_cache_hits_total (local only, counted at "
           + "admission) when it does not, and `N% cache` says how much came out. This recipe has had the cache ON "
           + "since 2026-11-04 (upstream's default; notes/prefix-cache-decision.md), so read and credited are NOT "
           + "the same number here -- the first ten minutes of real traffic measured 83% of prompt tokens arriving "
           + "as local_cache_hit -- and `prefill_stream_tok_s_credited` keeps the difference visible next to the "
           + "read figure. Same subtraction Strata's run average makes (`prompt_tokens - reused`, serve/web/app.js). The wall figure is the duty "
           + "cycle on top: ~7,200 over prefill seconds against ~7,000 across the wall. Over a span the "
           + "histogram-backed figures count only requests that finished inside it, which matters when the span is "
           + "shorter than a request's wall time.",
    step: "Mean engine step (inter-token latency) from delta(sum)/delta(count) of the engine's own histogram "
        + "families, so the requests counted are the ones that ran across the span. The poll's value is the only "
        + "one that can show a single slow step - a 905 ms step inside a long prefill reads 103 ms over a span.",
    spans: "The same quantity over the other spans - the spans this number is NOT. A span labelled shorter than "
         + "its name (90s, 2m) is the whole ring so far: the page says what it averaged rather than waiting for "
         + "the hour to fill.",
    poll: "This poll's own value: the gap between two engine polls (about a second). It is the only number here "
        + "that can catch something stuck right now, and the only one that swings on a single slow step. One "
        + "caveat, measured 2026-11-04: the engine advances its counters on its OWN logging beat, so a poll's "
        + "counter delta can cover more (or less) engine time than the poll spacing -- which is why the live "
        + "fan-out once read x12.3 with 8 requests running. The per-stream rate is a ratio of two deltas over the "
        + "same interval, so it survives that phase; the summed rate and the fan-out inflate together, and their "
        + "product stays consistent with each other. Spans are immune: they divide by the ring's own timestamps.",
  };
  const big = (id, v, unit) =>
    { $(id).innerHTML = `${isnum(v) ? dnum(v) : "\u2014"}<span class="unit">${unit}</span>`; };
  // -- which window the four numbers answer to ---------------------------------------------------
  // The default is the 15 min mean: measured here, one 1 s poll read 9.9 tok/s with six streams running and
  // another, a minute later in a gap, read 0, while the quarter hour around them was 51.7. Printing only the
  // poll number teaches a reader to treat it as the level, which is what the window picker exists to answer.
  const AVG_WIDE = { "1m": 60, "15m": 900, "1h": 3600 };
  const WIN_NAME = { live: "poll", "1m": "1 min", "15m": "15 min", "1h": "1 h" };
  const fullSpans = (avg) => Object.keys(AVG_WIDE).filter((k) => avg[k] && !avg[k].reset
    && isnum(avg[k].span_s) && avg[k].span_s >= AVG_WIDE[k] * 0.8);
  const spans = fullSpans(s.avg || {});
  const widest = spans.sort((a, b) => AVG_WIDE[b] - AVG_WIDE[a])[0];
  const win = rateWin === "live" ? "live" : (spans.includes(rateWin) ? rateWin : (widest || "live"));
  const A = win === "live" ? null : (s.avg || {})[win];
  const noSpans = Object.keys(s.avg || {}).length === 0;
  // The bar this sits in is wider than a card, but the harness budgets every slot against the tightest card in
  // the page, so the sentence has to be short: the explanation is the title's job.
  slot("win-note", noSpans ? "poll - no spans here"
    : win === rateWin ? `${WIN_NAME[win]} (${Math.round(A ? A.span_s : (rate.window_s || 0))} s)`
                      : `${WIN_NAME[win]} - ${WIN_NAME[rateWin]} not full`, TT.win);
  ["live", "1m", "15m", "1h"].forEach((k) =>
    $("win-" + k).setAttribute("aria-pressed", String(k === rateWin)));

  // The other spans, in one short line: "1m 30 - 15m 52 - 1h 24". No unit - the card's label and the number
  // above already carry it, and 22 characters is what the tightest 4-up card can hold.
  const spanName = (label, wide, span) => span < wide * 0.8
    ? (span < 600 ? `${Math.round(span / 10) * 10}s` : `${Math.round(span / 60)}m`) : label;
  const spanList = (key) => {
    const parts = [], said = new Set();
    for (const [label, wide] of Object.entries(AVG_WIDE)) {
      const b = (s.avg || {})[label];
      if (!b || label === win) continue;
      if (b.reset) { parts.push(`${label} reset`); continue; }
      const v = b[key];
      if (!isnum(v)) continue;
      const name = spanName(label, wide, isnum(b.span_s) ? b.span_s : wide);
      if (said.has(name)) continue;      // two spans resolve to the same window: say it once, not twice
      said.add(name);
      parts.push(`${name} ${dnum(v)}`);  // unitless on purpose: the card label and the headline carry it
    }
    return parts.join(" \u00b7 ") || "\u2014";
  };
  // Every span, including the one the headline is showing -- for a card with no headline of this quantity (the
  // pool), the full set is the point. `skip` drops spans whose mean would only repeat what is already on the
  // card: the pool's headline IS its current value, so a 1-minute mean of it is the same number rounded.
  // A filled span prints `15m 11.1%`: `unit` belongs to each NUMBER, not to the line, because the line can also
  // read `15m reset` and a unit hung on the end of that would be a unit on nothing.
  const spanAll = (key, skip = [], unit = "") => {
    const parts = [], said = new Set();
    for (const [label, wide] of Object.entries(AVG_WIDE)) {
      if (skip.includes(label)) continue;
      const b = (s.avg || {})[label];
      if (!b) continue;
      if (b.reset) { parts.push(`${label} reset`); continue; }
      const v = b[key];
      if (!isnum(v)) continue;
      const name = spanName(label, wide, isnum(b.span_s) ? b.span_s : wide);
      if (said.has(name)) continue;
      said.add(name);
      parts.push(`${name} ${dnum(v)}${unit}`);
    }
    return parts.join(" \u00b7 ") || "\u2014";
  };

  // The spans, with counts instead of rates: "15m +1,180 \u00b7 1h +22,600". Two, not three, and the harness
  // measures it: with the digits written out, three signed counts are 30 characters and the tightest 4-up slot
  // holds 24. The rule this page follows is that a slot is cut to the budget and the full sentence goes in the
  // title, never that the line is allowed to wrap.
  const spanSigned = (key, skip = ["1m"]) => {
    const parts = [], said = new Set();
    for (const [label, wide] of Object.entries(AVG_WIDE)) {
      if (skip.includes(label)) continue;
      const b = (s.avg || {})[label];
      if (!b || b.reset) continue;
      const v = b[key];
      if (!isnum(v)) continue;
      const name = spanName(label, wide, isnum(b.span_s) ? b.span_s : wide);
      if (said.has(name)) continue;
      said.add(name);
      parts.push({ name, v, span: isnum(b.span_s) ? b.span_s : wide });
    }
    // No sign: this is a cumulative counter's delta inside a window, which cannot go negative without the engine
    // having restarted -- and a restart is exactly what this same slot reports as `reset`. The `+` therefore carried
    // no information while costing a character.
    // And when the line still does not fit, a SPAN is dropped, never a digit: the tightest 4-up column measured 24
    // characters and these counts run to six and seven digits (`15m 104,648 · 16m 104,712` is 25). What goes first
    // is a span that repeats another -- a launch 16 minutes old has a 900 s "15m" and a 960 s "1h" holding the same
    // count -- and after that the widest, because the card's big number IS the since-launch total, so the widest
    // window is the one the card already says. Anything dropped is named in the title rather than lost quietly.
    const line = (ps) => ps.map((p) => `${p.name} ${dnum(p.v)}`).join(" \u00b7 ");
    const dropped = [];
    while (parts.length > 1 && line(parts).length > 24) {
      const dup = parts.findIndex((p, i) => i + 1 < parts.length
        && Math.abs(parts[i + 1].v - p.v) <= Math.max(1, 0.02 * Math.abs(parts[i + 1].v))
        && parts[i + 1].span - p.span <= 0.15 * parts[i + 1].span);
      dropped.push(parts.splice(dup >= 0 ? dup : parts.length - 1, 1)[0]);
    }
    return [line(parts) || "\u2014",
            dropped.length ? " Not printed for want of width: " + line(dropped) + "." : ""];
  };

  const sn = s.since || {};
  const stepV = A ? A.itl_ms : (isnum(rate.itl_ms) ? rate.itl_ms : sn.itl_ms);
  // How many tokens a step turns into: step / per-output-token. The window usually holds no histogram
  // observation at all (vLLM observes one when a request FINISHES), so the since-launch ratio stands in and
  // says so - in the title, because "x1.30" is the whole of what fits in a slot.
  const tpsW = A ? A.tokens_per_step : rate.tokens_per_step;
  const tpsS = isnum(sn.itl_ms) && isnum(sn.tpot_ms) ? sn.itl_ms / sn.tpot_ms : null;
  const tps = isnum(tpsW) ? tpsW : tpsS;
  // Decode, as a COUNT. What used to sit in this first card was the per-stream mean -- aggregate divided by the
  // running gauge -- and it was the most misread number on the page, for two reasons. The gauge counts a request
  // in the middle of a chunked prefill as running, so the divisor includes streams that cannot produce a token
  // yet; and a reader compares the result with the ~162 tok/s a stream gets with the box to itself and concludes
  // that "aggregate" is somehow smaller than a single stream. It is smaller, and the reason is the denominator:
  // wall seconds. Over this launch 47.6% of wall seconds hold no request at all, 5.9% of engine records carry a
  // prefill chunk at ~1 s each (75% of busy seconds), and 1.62 streams decode per record at 2.33 tokens a step --
  // 1.62 x 2.33 x 6.3 records/s = 24 tok/s summed. So the headline row shows a count, which cannot be misread as
  // an average, and the fan-out that does the explaining is the per-step waveform in the trends row.
  // "tok out", because on this launch the pool's 1.01M tokens and this launch's 1.00M generated tokens sit two
  // cards apart and differ by 1%. The unit has to say which million it is.
  big("rate-gentot", sn.generation_tokens_total, "tok out");
  slot("note-gentot", isnum(sn.prompt_tokens_total) ? `vs ${dnum(sn.prompt_tokens_total)} in` : "\u2014",
       TT.gentot + spanAll("generation_tokens").replace(/(\w+) (\S+)/g, "$1 +$2"));
  const gentotSpans = spanSigned("generation_tokens");
  slot("spans-gentot", gentotSpans[0], TT.spans + gentotSpans[1]);
  // "none credited", not "+0.0": in this window the counter moved nothing, which for a count is a fact about the
  // window (the engine was prefilling), not a measurement of zero tokens a second.
  slot("poll-gentot", isnum(rate.generation_tokens) && rate.generation_tokens > 0
       ? `poll +${dnum(rate.generation_tokens)} tok` : `poll \u00b7 none credited`, TT.poll);

  // Decode, headline = summed output per WALL second: generated tokens over the wall seconds of the window. That is
  // the throughput reading of the box (22 tok/s on the burst this launch was tested with), and it is the same basis
  // the summed waveform next door plots, so the card and the trace agree. The note line carries the other reading and
  // the two are one arithmetic: <output per stream-second> x <streams decoding> = the big number
  // (152 tok/stream-s x 0.15 = 22). The per-stream denominator is the stream-seconds the engine's decode families
  // report -- inter_token_latency and request_decode_time are one sum sampled twice (per stream-step and per
  // request), and a step two streams share accrues twice, which is why the unit says STREAM-second. It is the same
  // quantity as a request row's speed, so card and rows read alike: 18,118 tokens over 119.1 stream-seconds =
  // 152 tok/s, rows for that traffic ran 160-200. The card has been wrong in both directions here -- the summed
  // figure once carried a per-stream number under the words "summed, decode time", and before that a per-stream
  // mean divided by the running gauge -- which is why both figures are printed with the multiplication between them:
  // either one is then checkable against the other.
  const wallV = A ? A.generation_tok_s : rate.generation_tok_s;
  const perS = A ? A.decode_stream_tok_s : rate.decode_stream_tok_s;
  const streamsV = A ? A.decode_streams : rate.decode_streams;
  big("rate-gen", wallV, "tok/s");
  // The fan-out has two causes, and the span publishes the split: gaps in which the engine had nothing to do, and
  // gaps in which several streams shared a step. `busy_frac` is the first -- the share of the span covered by
  // worked 5 s gaps, an upper bound at the ring's own resolution -- and `streams_while_busy` the second; their
  // product is the multiplier on the note line, and the fan-out itself is the exact figure of the three. This belongs in the title rather than on the card because it is the answer to
  // the objection every reader makes when the big number is smaller than the engine's own log line: vLLM writes its
  // stat line at DEBUG level when the engine is idle (v1/metrics/loggers.py, `log_fn = logger.debug if
  // self.engine_is_idle`), so tailing the log shows an unweighted mean over the BUSY 10 s intervals. Measured on
  // live traffic 2026-11-04 over a 903 s span: 420 s of it printed anything, those lines average 42.4 tok/s, and the
  // same tokens over all 903 s -- this number -- is 19.7. The card is upstream's own basis; the log is not.
  // Three decimals below 0.1, not two: the render harness multiplies the two printed figures and compares their
  // product to the headline, and a fan-out of 0.0168 printed as "x0.02" breaks that check in front of the reader
  // (168 x 0.02 = 3.4 against a headline of 2.8). It costs one character and the slot still fits the tightest 4-up
  // column, which the same harness measures.
  const fanS = (v) => v < 0.1 ? v.toFixed(3) : (v < 10 ? v.toFixed(2) : dnum(v, 1));
  const swbS = (v) => v < 0.1 ? v.toFixed(3) : v.toFixed(2);
  const busyNote = A && isnum(A.busy_frac) && isnum(A.streams_while_busy) && isnum(streamsV)
    ? ` What the fan-out is made of: ${Math.round(100 * A.busy_frac)}% of this span is covered by gaps in which `
      + `the engine did something (${dnum(A.busy_s, 0)} s of ${dnum(A.span_s, 0)} at the ring's 5 s spacing) -- an `
      + `upper bound at that resolution, since a gap is judged by whether the engine's own record counter moved `
      + `across it. In those gaps ${swbS(A.streams_while_busy)} streams were in their decode phase: `
      + `${fanS(streamsV)} = ${A.busy_frac.toFixed(2)} busy x ${swbS(A.streams_while_busy)} streams, and `
      + `the exact one of those three figures is the fan-out, which comes from counters. Compare the engine's log `
      + `with care: it writes its stat line at debug level when idle and marks a whole 10 s interval for a request `
      + `that took half a second, so its printed lines average only the busy intervals, and coarsely at that.`
    : "";
  // Wall seconds always exist, so the big number always has a denominator. The note line's pair does not: a window
  // with no decode step has no decode stream-seconds, so both the per-stream figure and the fan-out go, and the slot
  // says which case it is instead of printing a rate with no denominator (a measured 0 above stays printed -- the
  // harness fails a dash where a zero belongs).
  slot("note-gen", isnum(perS) && isnum(streamsV)
       ? `${dnum(perS)} tok/stream-s \u00d7${fanS(streamsV)}`
       : "no decode step",
       TT.gen + busyNote + (isnum(perS) ? "" : " This window contains no decode step, so there are no decode stream-seconds to "
         + "divide by: the per-stream figure and the fan-out have no denominator. The summed figure above does -- "
         + "wall seconds -- and a 0 there is a measurement, so it prints as one."));
  slot("spans-gen", spanList("generation_tok_s"), TT.spans);
  // Same basis as the big number: a poll window always has wall seconds, so 0 is a measurement, not a gap. The
  // req/m clause shares the slot; the pair is what fits the tightest 4-up column (24 characters).
  slot("poll-gen", isnum(rate.generation_tok_s)
       ? `poll ${dnum(rate.generation_tok_s)}`
         + (A && isnum(A.requests_per_min) ? ` \u00b7 ${dnum(A.requests_per_min, 1)} req/m` : "")
       : "poll \u00b7 no window", TT.poll);

  // Prefill, both honest speeds. Read prompt tokens per second of REQUEST-prefill time (the engine's own
  // scheduled -> first-token span, summed over streams, so stream-seconds: `prefill_streams` says how many
  // prefills share them) and per second of WALL time (how much chewing an hour contains). This launch runs
  // ~7,200 and ~7,000, a 3% gap because long prefills take the 8,192-token step one at a time; the two would
  // separate the moment two prefills run at once. While a long prompt is running the window's number is the KV
  // gauge's growth in tokens a second -- an estimate, measured within ~6% on a 51,859-token prefill, negative
  // when finishing requests free blocks.
  const kvr = isnum(rate.kv_tok_s) ? Math.max(0, rate.kv_tok_s) : null;
  // READ tokens, not credited ones: vLLM counts cached and externally transferred prefix tokens in
  // prompt_tokens_total, and a rate built on them is not a read speed. The sidecar subtracts
  // prompt_tokens_cached_total when the build publishes it (the exact skipped count, local + external) and
  // prefix_cache_hits_total when it does not (local cache only). The poll's exact fallback follows the same
  // rule as the span (prompt_read_tok_s), so a switch of basis cannot happen between the headline and the
  // spans line.
  const exact = isnum(rate.prompt_read_tok_s) && rate.prompt_read_tok_s > 0 ? rate.prompt_read_tok_s
              : isnum(rate.prompt_tok_s) && rate.prompt_tok_s > 0 ? rate.prompt_tok_s : null;
  const prefillV = A ? A.prefill_stream_tok_s : (exact ?? kvr);
  big("rate-prefill", prefillV, "tok/s");
  // `N% cache` only appears when the engine actually served from cache: on a cache-off launch it would be a
  // permanent `0% cache` saying nothing, and every character in a slot is budgeted (the clause is rounded to a whole
  // percent for exactly that reason: `wall 14,097 · 89.9% cache` is 25 characters, over the 24 the tightest 4-up
  // column takes; `wall 14,097 · 90% cache` is 23). Asked on 2026-11-04 whether the caches were in effect, and the
  // answer was not on this page: a missing cache clause reads identically whether the feature is OFF, on-and-missed,
  // or not measured at all. The engine states the state (`config.enable_prefix_caching`, the same string its own
  // boot line prints), so OFF is said in words -- and only OFF: a build that publishes neither that config nor the
  // counters keeps the silence, which is the one case where silence is honest. It shares the line with the fan-out
  // clause rather than both being printed: `wall 4,316 cache off` fits the budget, `wall 4,316 cache off x2.0
  // prefilling` does not. The recipe turned the cache ON the same day (notes/prefix-cache-decision.md), so this is
  // now the branch that runs, at 83-90% on the box's own traffic.
  const pcState = String(((s.config || {}).enable_prefix_caching ?? "")).toLowerCase();
  const pcOff = pcState === "false" || pcState === "0";
  const fan = A && isnum(A.prefill_streams) && A.prefill_streams > 1.05;
  slot("note-prefill", A && isnum(A.prefix_hit_pct) && A.prefix_hit_pct > 0
       ? `wall ${dnum(A.prompt_tok_s || 0)} \u00b7 ${Math.round(A.prefix_hit_pct)}% cache`
       : A ? `wall ${dnum(A.prompt_tok_s || 0)}`
             + (fan ? ` \u00b7 \u00d7${dnum(A.prefill_streams, 1)} prefilling`
                    : pcOff ? ` \u00b7 cache off` : "")
         : exact !== null ? "exact \u00b7 credits at end"
           : kvr !== null ? "\u2248 from KV growth" : "no prefill in window",
       TT.prefill + (pcOff ? " This launch runs with the prefix cache OFF (the recipe's `--no-enable-prefix-caching`): "
         + "every prompt token here is computed, and the counters show it -- `local_cache_hit` stays at 0. The About "
         + "page states the same fact as a config line. When it is ON, the clause on this line becomes the share of "
         + "prompt tokens the engine skipped." : ""));
  slot("spans-prefill", spanList("prefill_stream_tok_s"), TT.spans);
  // Three different states, and "poll 0.0" was lying about which one it was: nothing credited in the window is
  // not the same measurement as a prefill running at zero tokens a second. prompt_tokens_total credits when a
  // request ENDS, so a 90 k-token prompt in flight has credited nothing, and the only number that moves while
  // it runs is the KV gauge's growth - labelled as the estimate it is.
  slot("poll-prefill",
       exact !== null ? `poll ${dnum(exact)}` + (kvr !== null && kvr > 0 ? ` \u00b7 KV ${dnum(kvr)}` : "")
     : kvr !== null && kvr > 0 ? `poll \u2248${dnum(kvr)}`
     : (rate.window_s || 0) < 3 ? `poll \u00b7 nothing in ${Math.round((rate.window_s || 0) * 10) / 10}s`
     : "poll \u00b7 nothing credited",
       TT.poll + (exact !== null || (kvr !== null && kvr > 0) ? ""
       : " prompt_tokens_total credits when a request ends, so a long prefill still in flight has credited "
         + "nothing here. The number under the headline is the one that moves while it runs."));

  big("rate-step", stepV, "ms");
  const ttftV = A ? A.ttft_ms : (rate.ttft_ms ?? sn.ttft_ms);
  const qV = A ? A.queue_ms : rate.queue_ms;
  slot("note-step", isnum(ttftV) ? `1st ${ms(ttftV)}` + (isnum(qV) ? ` \u00b7 q ${ms(qV)}` : "")
                                 : "no first token yet", TT.step);
  slot("spans-step", spanList("itl_ms"), TT.spans);
  // "no step", not "— ms": a unit after a dash claims a measurement whose value the reader cannot see, which is
  // the same mistake as `15m reset %` on the pool line. The window had no step to time, and the slot says that.
  slot("poll-step", isnum(rate.itl_ms)
     ? `poll ${dnum(rate.itl_ms)} ms` + (isnum(tps) ? ` \u00b7 \u00d7${tps.toFixed(2)}` : "")
     : `poll \u00b7 no step`,
       TT.poll + (isnum(tps) ? ` \u00d7${tps.toFixed(2)} is tokens a step (step time over per-output-token time): `
                              + "one engine step turns into that many tokens with speculation. "
                              + (isnum(tpsW) ? "Measured over this window." : "Since launch - this window holds no "
                                 + "per-token observation, because vLLM takes one when a request finishes.") : ""));

  // Say how old the numbers are instead of letting the reader guess.
  const age = isnum(s.at) ? Math.max(0, Date.now() / 1000 - s.at) : null;
  slot("fresh", `${isnum(s.poll_s) ? s.poll_s.toFixed(1) : "?"}s poll \u00b7 `
       + (age === null ? "?" : `${age < 10 ? age.toFixed(1) : Math.round(age)}s`) + " ago", TT.fresh);

  text("mode", s.mode === "proxy" ? "in the path (recording)"
    : s.mode === "merged" ? "inside the serve (engine records)" : "observing (read-only)");
  text("loaded", !s.reachable ? "Unreachable" : inflight ? "In use" : "Idle");
  $("model-state").className = `value value--sm${inflight ? " busy" : ""}`;
  $("model-state").firstElementChild.className = `dot${inflight ? " dot--busy" : s.reachable ? " dot--live" : " dot--off"}`;
  text("model", s.model_name || (s.models || [])[0]?.id || "no model reported");
  text("upstream", s.upstream || "—");
  const last = records.find((r) => !active(r));
  // "Requests retained: 97" on a box that hard resets is ambiguous between "this launch did 97" and "the last
  // launch's rows, restored". The journal says which, and where the file is.
  // The serve is not answering, and this page is the thing still standing -- so this is where the operator
  // reads what happened and what to type: the error, the launch's own command (assembled from the meta file the
  // launch wrote, not a recipe from memory), and why everything below still has content. Observed 2026-11-03
  // 04:36: the serve stopped answering on :8080 for minutes while its pid lived, and the in-serve Monitor went
  // away with it -- a blank page at exactly the moment a person needs one.
  const dn = $("down-line");
  if (dn) {
    dn.hidden = !!s.reachable;
    if (!s.reachable) {
      dn.dataset.tone = "danger";
      dn.innerHTML = `<b>the serve behind this page is not answering</b>`
        + `<span class="q">${esc(s.error || "unreachable")}</span>`
        + (s.launch ? `<div class="rq-support">bring it back: <code>${esc(s.launch)}</code>`
          + ` <span class="q">the script stops rather than starting on top of ranks that still hold the GPUs</span></div>`
          : "")
        + `<div class="rq-support">this page is the standalone Monitor${s.port ? ` on :${s.port}` : ""}. The cards,`
        + ` their links and slots, and the record of what ended the last boot come from sysfs and the kernel`
        + ` log — none of them need the serve, so they keep reading while it is down.</div>`;
    }
  }
  const j = s.journal || null;
  qualText("count", records.length, j ? (j.restored ? `${j.restored} carried over` : "this launch only") : "");
  // A recent-set fact rather than a live one: how many of the rows this page is holding did not finish cleanly.
  // Streaming rows are not errors, so the two states are kept apart explicitly -- folding them together would put
  // the six requests currently decoding in the same number as the ones that failed.
  const bad = records.filter((r) => r.state && r.state !== "completed" && !active(r));
  qualText("err-count", bad.length, `${records.length} retained`);
  text("err-note", bad.length
    ? `${bad.length} of the ${records.length} rows here did not complete \u2014 the list below has them, `
      + "filtered by Errors"
    : `none of the ${records.length} rows here failed (active streams are not counted here)`);
  const jl = $("journal-line");
  if (jl) {
    jl.hidden = !j;
    jl.textContent = j
      ? (j.ok
        ? `request journal ${j.path} · ${j.restored} row${j.restored === 1 ? "" : "s"} carried over from `
          + `earlier launches${j.readonly && j.followed
            ? ` · ${j.followed} row${j.followed === 1 ? "" : "s"} read live from the attached file since this `
              + "page started" : ""}${j.readonly
            ? " · attached read-only: these rows belong to the launch that owns the file"
            : ` · this launch ${j.boot}`}`
        : `request journal FAILED (${j.error}) — the list is memory only and dies with the serve`)
      : "";
    jl.dataset.tone = j && !j.ok ? "danger" : "";
  }
  const dl = $("death-line");
  if (dl) {
    const d = s.last_death, h = s.deaths;
    const then = (t) => new Date((t || 0) * 1000).toLocaleString([], {
      month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
    const tally = (o) => Object.entries(o || {}).map(([k, v]) => `${k} ×${v}`).join(", ");
    dl.hidden = !d;
    dl.textContent = d ? [
      `the boot before this one ended with a ${d.severity || "hardware"}`
      + `${d.uncorrectable ? ", uncorrectable" : ""} machine check in the ${d.bank_name || "unnamed block"}`
      + ` (bank ${d.bank})`
      + (d.cpu != null ? ` on CPU ${d.cpu}` + (d.core && d.core.siblings
        ? ` = core ${d.core.core_id} (SMT ${d.core.siblings})` : "") : "")
      + ` at ${then(d.at)} · ${d.bert_records ?? "?"} firmware record kept`,
      h ? `${h.records} such record${h.records === 1 ? "" : "s"} across the last ${h.boots_scanned}`
        + ` boot${h.boots_scanned === 1 ? "" : "s"} the journal holds (${then(h.first_at)} to ${then(h.last_at)})`
        + ` · who they blame: ${tally(h.named)}`
        + ` · where: ${tally(h.cores)}` : "counting the earlier ones…"
    ].join("\n") : "";
    dl.dataset.tone = d && (d.severity === "fatal" || d.uncorrectable) ? "danger" : "warn";
  }
  text("active", inflight ? `${Math.round(inflight)} running / waiting at the engine`
    : s.mode === "proxy" ? "Nothing in flight"
      : s.mode === "merged" ? "Finished requests only: the engine reports a request when it ends"
        : "Nothing routed here");
  const lEngine = last && last.response_format === "engine";
  text("wall", last && isnum(last.wallclock_s)
    ? seconds(last.wallclock_s + (lEngine ? ((last.timings || {}).queue_s || 0) : 0)) : "—");
  text("wall-note", lEngine ? "From the engine: its own prefill + decode, plus the queue wait before them"
    : "Through this page · queue, prefill and decode together");
  const lt = (last && last.timings) || {}, lu = (last && last.usage) || {};
  // The split of that one request, not the window mean: this is the number the long-prompt questions are
  // actually about (a 250k needle is prefill, and prefill was previously only visible as a mean over 2 s).
  text("wall-split", isnum(lt.queue_s) || isnum(lt.prefill_s) || isnum(lt.decode_s)
    ? `queue ${seconds(lt.queue_s)} · prefill ${seconds(lt.prefill_s)} · decode ${seconds(lt.decode_s)}`
    : isnum(lu.prompt_tokens) || isnum(lu.completion_tokens)
      ? `${fmt(lu.prompt_tokens ?? 0, 0)} -> ${fmt(lu.completion_tokens ?? 0, 0)} tokens in that request`
      : "that record carries no queue / prefill / decode split");
  // No fallback here. The engine-wide 1/step and the token total over all streams are already two cards
  // away; a per-request card that quietly shows one of them when the request's own speed is missing reads
  // as the request's speed. So: the request's own number, or a dash that says why.
  const own = lt.predicted_per_second;
  $("speed").innerHTML = `${isnum(own) ? fmt(own, 1) : "—"}<span class="unit">tok/s</span>`;
  text("speed-note", isnum(own) ? `that request only · ${lt.speed_basis || "output tokens over its decode span"}`
    : isnum(rate.itl_ms) ? `no speed for that record · engine step ${fmt(rate.itl_ms, 1)} ms, all streams together `
      + `${fmt(rate.generation_tok_s ?? 0, 1)} tok/s`
      : "no speed for that record");

  text("kv-pct", pct(live.kv_used_pct, 1));
  // The pool card draws a waveform (below, where the history ring is) instead of a progress bar: the bar said
  // only "how full right now", which the number above it says too, while the shape is what a long-context
  // question is really about -- how fast it filled, and whether it ever stopped. The tone (ok / warn /
  // danger past 85 and 95%) carries over to the line's colour, so the bar's warning job survives.
  const kv = Math.max(0, Math.min(100, live.kv_used_pct || 0));
  text("running", String(Math.round(live.running || 0)));
  text("waiting", `${Math.round(live.waiting || 0)} waiting`
    // Why they wait, when this vLLM says why (0.32 does): "capacity" is waiting for scheduling capacity,
    // "deferred" is a transient constraint (LoRA budget, KV transfer, blocked status), and the two sum to the
    // waiting count. The total alone misleads on this model: measured, 8 x 30k prompts queued with the KV pool
    // 11.7% full, because long prefills take the 8,192-token step one at a time. A real 0 stays visible --
    // "nothing waiting" is a fact, not a missing value.
    + (isnum(live.waiting_capacity) ? ` · ${Math.round(live.waiting_capacity)} for scheduling capacity` : "")
    + (isnum(live.waiting_deferred) ? ` · ${Math.round(live.waiting_deferred)} deferred` : "")
    + ` · ${fmt(rate.preemptions ?? 0)} preempted in the window`
    // Since launch is the number that predicts the pain: vLLM v1 re-prefills a preempted request, and the first
    // two seen on this box (160k-token prompts with the pool at 100%) cost their requests a 20.9 s / 30.2 s first
    // token and a decode average of 14.8 / 3.4 tok/s. The window only says whether it happened just now.
    + (isnum(live.preemptions_total) ? ` · ${Math.round(live.preemptions_total)} since launch` : "")
    + (live.swapped ? ` · ${Math.round(live.swapped)} swapped` : ""));
  $("gen-toks").textContent = isnum(rate.generation_tok_s) ? fmt(rate.generation_tok_s, 1) : "—";
  text("prompt-toks", `${isnum(rate.prompt_read_tok_s) ? fmt(rate.prompt_read_tok_s, 0) : "—"} prompt tok/s read`
    // vLLM credits prompt_tokens_total when a request FINISHES: a long prefill reads 0 for its whole duration
    // and then lands in one window (measured: 180,066 tokens showed as 89,580 tok/s in a single 2 s poll).
    // The live signal during a prefill is the KV pool bar, not this line.
    + " (credited at request end)"
    + (isnum(live.prefix_hit_pct) && live.prefix_hit_pct > 0 ? ` · ${fmt(live.prefix_hit_pct, 1)}% served from cache` : "")
    + (isnum(rate.requests_per_min) ? ` · ${fmt(rate.requests_per_min, 1)} req/min` : ""));
  // vLLM observes a latency histogram when a request FINISHES, so a window holds no observation at all
  // while a long one is in flight. The cumulative ratio is not a window mean: the page says which it shows.
  const since = s.since || {};
  const pick = (k) => (isnum(rate[k]) ? rate[k] : (isnum(since[k]) ? since[k] : null));
  qualText("accept", pct(pick("accept_pct"), 1),
           isnum(rate.accept_pct) || !isnum(since.accept_pct) ? "" : "since launch");
  text("drafts", isnum(rate.drafts_per_req) ? `${fmt(rate.drafts_per_req, 2)} drafts per request`
    : isnum(since.accept_pct) ? `${fmt(since.requests, 0)} requests finished since the engine started`
      : "no speculation counters");
  // MTP=2 accepts at most 2 draft tokens a step, so this line has a ceiling the reader can check by eye.
  text("per-draft", isnum(pick("accepted_per_draft"))
    ? `${fmt(pick("accepted_per_draft"), 2)} accepted tokens per draft`
      + (isnum(rate.accepted_per_draft) || !isnum(since.accepted_per_draft) ? "" : " (since launch)")
    : "")
  // This line belongs on the Generation card, not under Speculation: on a launch that has run any long prefill the
  // number is in the thousands (9,621 since launch, measured), and under a speculation heading it reads as "MTP
  // accepts 9,620 tokens a step". What it is NOT is a step size. vllm:iteration_tokens_total's own HELP text says
  // "number of tokens per engine_step", but at 0.32.0 the observation is computed prompt tokens + generation
  // tokens (v1/metrics/loggers.py:1273) and a chunked prefill lands in ONE record holding the whole prompt:
  // measured, one 4,893-token prompt produced 2 records (4,893 and 4) and one 29,969-token prompt produced 3
  // (29,969, then 4, then 1), while decode steps get a record each. The ratio therefore tracks prompt size and can
  // sit above max-num-batched-tokens -- 29,969 in one record with NBT=8,192 -- so the line names the counter's
  // unit instead of claiming a chunk size. Window and since-launch values are both labelled: not the same span.
  const tpi = pick("tokens_per_iter"), ipr = pick("iters_per_req");
  text("step-size", isnum(tpi) ? `${fmt(tpi, 0)} tokens per iteration record (prompt whole, decode per step)`
    + (isnum(ipr) ? ` · ${fmt(ipr, 2)} records per request` : "")
    + (isnum(rate.tokens_per_iter) || !isnum(since.tokens_per_iter) ? "" : " (since launch)") : "");

  // -- the trends row ------------------------------------------------------------------------------
  // One helper for all four waveforms, so each gets the same caption: last, peak and average over the span
  // the ring covers. The ring is fixed in POINTS (90), so the span is points x poll_s -- changing the poll
  // changes what a waveform is looking back at. A series that was null for part of it (the prefill estimate
  // before anything ran) draws through the gap rather than as a flat 0, which is why the point count is on
  // the line: 12 of 90 pts means the shape is only as old as those 12 samples.
  const trend = (sparkId, n1, n2, values, d, { keep = (v) => v !== 0, lo = null, hi = null,
                                               empty = "no samples", keptWord = "", title = "" } = {}) => {
    const xs = values.filter(isnum);
    // The axis, per series. Autoscaling a rate to its own min/max makes a series that spends most of the
    // window at zero look like a full-height oscillation, so rates run on 0..peak and are comparable to each
    // other; hi: "peak" takes the peak of what is actually plotted (never below zero -- an empty series has
    // no peak, and Math.max() of nothing is -Infinity, which draws a box full of nothing).
    const top = hi === "peak" ? Math.max(...xs, 0) : hi;
    spark($(sparkId), values, { baseLo: lo, baseHi: top });
    // Which samples count toward peak and average, per series. A 0 in a rate series is not a measured rate:
    // it is a window where the counter was flat because nothing was being generated (mid-prefill, say), and
    // counting those made a 200 tok/s stream average 31.5. The KV-growth series keeps only the windows where
    // the pool GREW -- the rest are finished requests freeing blocks, and averaging them gave "avg -1,004
    // tok/s". The KV pool series keeps everything: an idle pool at 0% IS the value.
    const used = xs.filter(keep);
    if (used.length < 2) {
      slot(n1, empty, title);
      slot(n2, `${xs.length}/${values.length} \u00b7 ${histSpan}`, title);
      return;
    }
    // Two slots, never one wrapped line: the level on the first, the spread and the sampling on the second.
    // Both are written to fit the tightest 4-up card (tests/render-monitor.mjs measures the longest string every
    // slot can print, and the budget is 24 characters at 12 px), so the height of this row does not depend on how
    // many digits the engine happens to be reporting. With the digits written out rather than abbreviated, the
    // ring's own span -- which the trend caption above the sparkline already states -- came out of this line.
    const last = xs[xs.length - 1];
    const avg = used.reduce((x, y) => x + y, 0) / used.length;
    // A negative newest sample in the KV-growth series is the pool FREEING blocks as requests finish, not prefill
    // running backwards. With the digits written out, a leading minus in front of 172,106 is something the reader
    // has to decode on a line that repaints every second, so the drain is named instead -- and the average moves
    // into the title for that case, because `now draining 172,106 · avg 7,939` is 32 characters and the budget is
    // 24. The sign is still in the drawn waveform, which autoscales through zero for exactly this reason.
    const drain = last < 0 && !keep(last);
    slot(n1, drain ? `now draining ${dnum(-last, d)}`
                   : `now ${dnum(last, d)} \u00b7 avg ${dnum(avg, d)}`,
         title + ` (avg ${drain ? dnum(avg, d) + ", " : ""}peak ${dnum(Math.max(...used), d)} over ${histSpan})`);
    slot(n2, `peak ${dnum(Math.max(...used), d)} \u00b7 ${keptWord ? keptWord + " " : ""}${used.length}/${values.length}`,
         title + ` (${used.length} of ${values.length} points kept${keptWord ? ", a point kept means the counter moved in it" : ""}, over ${histSpan}.`
         + (lo !== null && top > lo ? ` The waveform is drawn on a ${dnum(lo)}..${dnum(top)} axis.` : "") + ")");
  };
  const TREND_T = {
    perstep: "Generated tokens per engine record, summed over streams -- never a per-stream mean. It is the "
           + "fan-out times the acceptance length: live over 15 min, 2.85 streams producing per record x ~2.5 "
           + "tokens per stream-step = 7.05 tokens a record, against the ~2.09 a stream turns out of its own step "
           + "(the ratio between the two is the payload's `streams_per_record`, ITL observations over records). At "
           + "0.32 a whole chunked prefill lands in one record, so the line dips exactly where the engine was "
           + "prefilling; a gap is an interval with no record in it.",
    gen: "All streams summed over WALL seconds -- the summed figure, the throughput reading of the box, plotted over "
       + "time. Same basis as the decode card's big number, so the trace and the number agree; what the trace adds "
       + "is the shape -- where the minute went. Flat stretches are the engine prefilling or idle rather than "
       + "decoding, which is why this line's peak and its average differ by an order of magnitude on a "
       + "prefill-heavy launch. The card's note line is the bridge to the other basis: its per-stream-second "
       + "figure x this series' fan-out = this line. Drawn on the wall basis because the per-stream denominator "
       + "does not exist in a window that was all prefill, and a gap-per-second line reads as an idle box.",
    prefill: "Prefill estimated from how fast the KV pool is growing (the counter credits at request end, so the "
           + "counter series would be a flat line with one spike). It goes NEGATIVE when finishing requests free "
           + "blocks; peak and average count only the windows where the pool grew.",
    running: "How many streams the engine is actually running. The aggregate rate is a function of this line: the "
           + "same engine that gives one stream ~200 tok/s gives ~20 each at six, with prefills interleaved.",
  };
  const hist = trendWin === "slow" ? (s.history_slow || []) : (s.history || []);
  const histSec = hist.length > 1 ? Math.max(0, Math.round(hist[hist.length - 1].at - hist[0].at)) : 0;
  const histSpan = histSec >= 120 ? `${Math.round(histSec / 60)}m` : `${histSec}s`;
  $("trend-fast").setAttribute("aria-pressed", String(trendWin === "fast"));
  $("trend-slow").setAttribute("aria-pressed", String(trendWin === "slow"));
  slot("trend-note", `${hist.length} pts over ${histSpan}`,
       trendWin === "slow"
         ? "One point per 15 s for the last hour, each point the rate over its interval rather than a sample of "
           + "one poll. Answers \"what has the box looked like this hour\"; it cannot show a single stuck step."
         : `One point per engine poll (${isnum(s.poll_s) ? s.poll_s.toFixed(1) : "?"} s), 90 points. This is `
           + "where a stuck step, a restart or one huge prefill shows up as a shape.");

  trend("perstep-spark", "perstep-note1", "perstep-note2", hist.map((h) => h.gen_per_step), 2,
        { lo: 0, hi: "peak", empty: "no records this window", title: TREND_T.perstep });
  trend("gen-spark", "gen-note1", "gen-note2", hist.map((h) => h.gen_tok_s), 0,
        { lo: 0, hi: "peak", empty: "nothing generated", keptWord: "busy", title: TREND_T.gen });
  // The prefill waveform is the live estimate (KV growth), not the counter rate: over a long prefill the
  // counter series is a flat line with one spike at the end, which is a fact about vLLM, not about the work.
  // It stays autoscaled: it goes negative when finishing requests free blocks, and a 0 baseline would clip
  // the half that says "the pool is draining".
  trend("prefill-spark", "prefill-note1", "prefill-note2", hist.map((h) => h.kv_tok_s), 0,
        { keep: (v) => v > 0, empty: "the pool did not grow", title: TREND_T.prefill });
  // The fourth waveform is how many streams the engine is actually running: at NSEQ=32 the aggregate rate is
  // a function of this line, and the knee (PROGRESS: the spread past 32) is visible in its shape.
  trend("running-spark", "running-note1", "running-note2", hist.map((h) => h.running), 0,
        { lo: 0, hi: "peak", empty: "0 streams", title: TREND_T.running });
  // The KV pool's own waveform lives on its card, not here: its axis is 0..100 (a quarter-full pool is a
  // quarter of the box), which would flatten every other series if it shared the trends row's scale.
  spark($("kv-spark"), hist.map((h) => h.kv_used_pct), { baseLo: 0, baseHi: 100 });
  $("kv-spark").dataset.tone = kv > 95 ? "danger" : kv > 85 ? "warn" : "ok";
  slot("kv-note1", isnum(live.kv_tokens)
    ? `${dnum(live.kv_tokens)} tok \u00b7 ${kv.toFixed(0)}%`
    : "pool size not in /metrics",
    `The pool this launch was given, and how full it is now (the waveform reaches back ${histSpan}). ` +
    (kv > 85 ? "Above 85% is tight, above 95% is the limit: a request that needs blocks the pool cannot give is "
              + "preempted and re-prefilled from scratch." : ""));
  slot("kv-note2", spanAll("kv_used_pct_mean", ["1m"], "%"),
    "Mean pool used over each span, from the same gauge the waveform draws. The ring cannot answer whether the "
    + "pool has been tight all afternoon; these can. A span labelled shorter than its name is the whole ring so far.");

  for (const [id, k] of [["t-ttft", "ttft_ms"], ["t-queue", "queue_ms"], ["t-prefill", "prefill_ms"],
                         ["t-decode", "decode_ms"], ["t-step", "itl_ms"], ["t-tpot", "tpot_ms"]])
    // The caveat goes on at note size: these six are value--sm headlines, and "(since launch)" set in the
    // headline's own size reads as a second number and buries the first.
    qualText(id, ms(pick(k)), isnum(rate[k]) || !isnum(since[k]) ? "" : "since launch");
  text("window", isnum(rate.window_s) ? `${rate.window_s.toFixed(1)} s` : "—");

  const gpus = s.gpus || [];
  const quiet = gpus.filter((g) => (g.unread || []).length).length;
  text("gpus-note", gpus.length ? `${gpus.length} card${gpus.length > 1 ? "s" : ""} · amdgpu sysfs`
    // gpu_busy_percent on this box is not a finely readable load meter. Measured over a day, same card set:
    // refused (EBUSY) with nothing running, 0% after a launch stopped, 24-29% with the engine idle, 46-100%
    // under three streams, and 95-99% over long stretches of decode. It says roughly how hard the cards are
    // being worked; the precise answers are the rate waveforms above, and the VRAM wave for a leak.
    + " · busy % is coarse here (refused with nothing running, 0 stopped, ~25 idle, 46-100 under load)"
    // The link word each card carries, and the rule that decides when it is more than one word. Said on the
    // panel bar because what a reader would otherwise suspect is a missing sensor: the AER counters are read
    // every poll and deliberately not printed while a link holds its rating.
    + (s.pcie && s.pcie.length
       ? ` · link shown per card: ` + (s.pcie.filter((c) => c.degraded && !c.parked).length
          ? `${s.pcie.filter((c) => c.degraded && !c.parked).length} chain(s) below their rating`
          : `all ${s.pcie.length} at their rating`)
          + ` · AER counters are read every poll and printed only on a card whose link is short`
       : " · no PCIe sysfs readable here, so cards show no link")
    + (s.smi ? " + rocm-smi" : "") + (quiet ? ` \u00b7 ${quiet} refusing sysfs reads` : "")
    : "no amdgpu sysfs readable here");
  // The mapping line, in the machine's own numbering. Two naming systems are in play on this box and the page
  // has to keep both straight: the firmware's slot numbers (1, 3, 6, 8 here) and amdgpu's enumeration names
  // (card1..card4), which are per-boot and can name a different card after a reseat. Reading them off in slot
  // order is what makes "reseat slot 6" an instruction someone can actually follow.
  const sm = $("slot-map");
  const smap = s.slot_map || [];
  if (sm) {
    sm.hidden = !smap.length;
    sm.innerHTML = smap.length
      ? `chassis order <span class="q">firmware slot numbers, not driver names</span> \u00b7 `
        + smap.map((m) => {
            // One word for the link, and the rating named only when the link sits under it. Four "Gen5 x16"s
            // in a row is the picture this box normally shows, and it used to cost four rows of
            // "32.0 GT/s x16 0 corr" that read as noise.
            const link = isnum(m.gen) ? `Gen${m.gen} \u00d7${m.width}` : "link unreadable";
            const short_ = m.short && isnum(m.max_gen)
              ? ` <span class="q">of Gen${m.max_gen}${m.parked ? ", parked" : ""}</span>` : "";
            const errs = m.short && (m.corr || m.uncorr)
              ? ` <span class="q">${m.corr} corr${m.uncorr ? ` / ${m.uncorr} uncorr` : ""}</span>` : "";
            return `<b>#${m.pos} slot ${esc(m.slot != null ? m.slot : "?")}</b>`
              + ` <span class="q">${esc(m.card || "unnamed")}</span> GPU <code>${esc(m.bus)}</code>`
              + ` ${link}${short_}${errs}`;
          }).join(" \u00b7 ")
        + ` <span class="q">the slot is registered at the card's switch upstream port`
        + ` (${esc((smap[0].slot_at || "n/a").replace(/^0000:/, ""))} for slot ${esc(smap[0].slot != null ? smap[0].slot : "?")}),`
        + ` not at the GPU endpoint</span>`
      : "";
    // A parked chain (D3hot between launches, its links idling at Gen1) is not a fault, so it does not colour
    // the line. Only a short link that is not parked does.
    sm.dataset.tone = smap.some((m) => m.short && !m.parked) ? "warn" : "";
  }
  // Cards, chains and the About page are all ordered by the server (chassis order), not by enumeration order:
  // one order across the cards, their links and the About page, so a card number never means a different
  // position in two places. The link rides on the card itself (gpuCard reads g.link); the panel note above says
  // when it says more than "Gen5".
  $("gpus").replaceChildren(...gpus.map((g) => gpuCard(g, s.gpus_history || [])));
  // There is no PCIe panel any more (2026-11-05). What it printed -- link, AER totals, which counters moved,
  // this window's deltas -- either fits in one word on the card or stays in the account a short link opens
  // below. The payload it drew on is still collected every poll (pcie, pcie_history, pcie_events) and still
  // journaled, so tools/journal-pcie.py and the death review keep working with the panel gone.
  paintPcieDetail(s);
  paintLiveState(s);
  paintWatchdog(s);
  paintLengths(s);
}

// One alarm read stays on the first screen even though the alarm row moved to the fourth: engine state, in the
// same strip as "how old is this snapshot". The reason is a measured failure rather than a preference. On
// 2026-11-15 the four big rate numbers kept printing plausible figures for 26 minutes after /health stopped
// answering, because they are differences of counters that the last good poll had filled and the KV gauge that
// the wedged ranks keep holding; meanwhile three VLLM::Worker_TP* sat on ~29 GiB each. A page whose first screen
// is rates must therefore say, in the same line as the freshness of those rates, whether the engine is
// answering at all. Two words, one dot: `Answering`, `Loading`, `Gone`, `Wedged`.
// Context size, prompt sent versus UNCACHED. The serve imposes no length policy on purpose (one port, one pool,
// --max-model-len is the only hard edge, and the front door reads a header and nothing else), so the size of a
// context is a client decision -- and a client can only make it well if the box says what each size costs.
// The second histogram is the one that costs anything: prompt_tokens is traffic, prompt_tokens minus cached_tokens
// is work. Measured on this box 2026-11-16: one 186,739-token context cost 26.3 s cold and 0.69 s warm, and
// running cold prefills of ~165k serialized -- last first-token at 22 s, 47 s, 70 s for one, two and three of
// them, with the short requests alongside waiting 0.8 s, 23 s, 45 s -- while the pool never passed 18% and the
// preemption counter never moved. So a card that printed only prompt size would rank harmless rows and
// expensive ones in the same list.
function paintLengths(s) {
  const sec = $("lengths");
  if (!sec) return;
  const L = s.lengths;
  if (!L || !L.rows) {
    sec.hidden = true;              // an older sidecar sends no lengths block at all: no card, not a zero
    return;
  }
  sec.hidden = false;
  const p = L.prompt || {}, u = L.uncached || {}, bands = L.bands || [];
  text("len-scope", `${fmt(L.rows)} request${L.rows === 1 ? "" : "s"} in this list`
       + (L.span_s ? ` · ${seconds(L.span_s)} from oldest arrival to newest` : ""));

  // The headline numbers, because "30,073 tokens" and "79% cached" are the two things a client of this box can act
  // on, and in the old layout both were buried in small print under a chip row. The chips stay for SHAPE: a p90 of
  // 12k says nothing about the three requests at 250k that are the actual problem.
  const cachePct = p.sum ? (100 * (1 - (u.sum || 0) / p.sum)) : null;
  text("len-p50", p.median == null ? "\u2014" : fmt(p.median));
  text("len-p50-note", "half the requests in view are shorter");
  text("len-p90", p.p90 == null ? "\u2014" : fmt(p.p90));
  text("len-p90-note", u.p90 == null ? "one row in ten is longer"
       : `one row in ten is longer; ${fmt(u.p90)} of it had to be computed`);
  const win = L.window;
  text("len-max", p.max == null ? "\u2014" : fmt(p.max));
  const maxEl = $("len-max");
  if (maxEl) maxEl.dataset.warn = (win && p.max != null && p.max >= 0.9 * win) ? "1" : "0";
  text("len-max-note", (win && p.max != null)
       ? `${(100 * p.max / win).toFixed(0)}% of this serve's ${fmt(win)}-token window`
       : "this serve gave no window to compare it against");
  text("len-cache", cachePct === null ? "\u2014" : `${cachePct.toFixed(0)}%`);
  text("len-cache-note", `${fmt(u.sum)} of ${fmt(p.sum)} prompt tokens the engine had to compute itself`);

  // One bar, one honest scale: the right edge is the window the ENGINE reports on its own /v1/models card, never the
  // number the launch record claims -- a page can be holding rows from a launch that is gone, and a bar drawn
  // against a dead launch's window is a precise-looking lie. No window means no bar.
  const wsec = $("len-window");
  if (wsec) {
    if (!win || p.max == null) {
      wsec.hidden = true;
      text("len-window-note", "");
    } else {
      wsec.hidden = false;
      const pctw = Math.min(100, 100 * p.max / win);
      const fill = $("len-window-fill");
      if (fill) {
        fill.style.width = pctw.toFixed(1) + "%";
        fill.dataset.over = (p.max >= 0.9 * win) ? "1" : "0";
      }
      const step = parseInt((s.meta || {}).nbt || "", 10);
      lenTick("len-window-step", (Number.isFinite(step) && step > 0 && step <= win) ? 100 * step / win : null);
      lenTick("len-window-p50", p.median == null ? null : 100 * p.median / win);
      lenTick("len-window-p90", p.p90 == null ? null : 100 * p.p90 / win);
      text("len-window-note",
           `right edge = the window this serve reports on its own card (${fmt(win)} tokens); fill = the biggest `
           + `context in view (${fmt(p.max)} = ${pctw.toFixed(0)}% of it). `
           + (L.near_window ? `${fmt(L.near_window)} row(s) sit at 90% of the window or more. `
              : `nothing in view comes within 10% of the window. `)
           + (L.over_window ? `${fmt(L.over_window)} row(s) report a prompt at or past the window the engine `
              + "reports, which means these rows and this window disagree -- the list can outlive the launch it came from. "
              : "")
           + ((Number.isFinite(step) && step > 0) ? `The amber tick is one step of prefill (${fmt(step)} tokens, `
              + "from the launch record): above it a prefill cannot hide inside a single step." : ""));
    }
  }

  // One scale for both histograms. Each card used to scale to its own maximum, which made the two full-width bars
  // mean different things a few pixels apart -- and "48 rows came in at 64k-128k, 4 of them had to be computed
  // there" is the whole reason the two are printed next to each other. So full width = the biggest band in EITHER
  // card, and the comparison is a shape rather than a guess.
  const scale = Math.max(1, ...(p.bands || []), ...(u.bands || []));
  lenBars("len-bars-p", bands, p.bands || [], p.band_tok || [], scale, p.n || L.rows, p.sum);
  lenBars("len-bars-u", bands, u.bands || [], u.band_tok || [], scale, u.n || L.rows, u.sum);
  text("len-scale", `both histograms are drawn to one scale: a full bar is ${fmt(scale)} row${scale === 1 ? "" : "s"}. `
       + `Amber bands are the ones whose prefill cannot hide inside a single step (${fmt(8192)} tokens, this `
       + `launch's --max-num-batched-tokens). The two right-hand pairs are rows and their share of the rows in this `
       + `card, then the TOKENS those rows carry and their share of the total above -- because a band with four rows `
       + `in it can still be most of the work.`);
  qualText("len-prompt", `${fmt(p.p95)} p95 · ${fmt(p.max)} max tok`, `median ${fmt(p.median)}`);
  text("len-prompt-note", `${fmt(p.sum)} prompt tokens sent in all`);
  qualText("len-uncached", `${fmt(u.p95)} p95 · ${fmt(u.max)} max tok`, `median ${fmt(u.median)}`);
  text("len-uncached-note", `${fmt(u.sum)} of ${fmt(p.sum)} tokens had to be computed`
       + `; ${fmt(u.over_step)} cost more than one step (8,192 tok or more)`
       + (u.n !== L.rows ? ` · ${fmt(L.rows - u.n)} row(s) carry no cached count and are excluded, `
                            + "not counted as misses" : ""));
  slot("len-read", `${fmt(u.sum)} tok`,
       "prompt tokens minus cached tokens, summed over the rows in this list. This is the work the prefix cache "
       + "did NOT spare, and it is the quantity that sits in front of every other request's first token.");
  slot("len-rate", L.rate_tok_s ? `${fmt(L.rate_tok_s)} tok/s` : "not measurable here",
       L.rate_tok_s
         ? `measured from these rows: uncached tokens ÷ the engine's own prefill_s, over the ${L.rate_rows} `
           + `row(s) big enough to be throughput-bound (8,192 or more uncached, one step at this recipe's `
           + "--max-num-batched-tokens). Over ALL rows the same division reads 1,600-3,050 tok/s because a "
           + "300-token prompt's prefill_s is scheduling, not computation."
         : "no row here carries timings.prefill_s, so no rate is invented from a remembered constant");
  slot("len-cost", L.read_s !== null && L.read_s !== undefined ? seconds(L.read_s) : "—",
       "What this list's cold context cost in ENGINE TIME, at the rate above. It is not latency anyone felt "
       + "directly — it is the share of the step budget that went into computing prompt tokens instead of "
       + "advancing streams. Measured shape on 2026-11-16: 17-23 s of it per cold ~165k context, landing on "
       + "everyone else's first token.");
  const b = L.biggest;
  slot("len-big", b ? `${fmt(b.uncached)} tok uncached` : "—",
       b ? `${b.id} · ${stamp(b.started_at)} · ${fmt(b.prompt)} sent, ${fmt(b.cached)} already cached`
         : "no row in this list has a cached count, so no cold context can be identified");
  text("len-note", (L.clamp ? `${L.clamp} row(s) reported cached > prompt and were clamped to zero uncached. ` : "")
       + (L.no_usage ? `${L.no_usage} row(s) have no usage yet (still in flight) and are excluded. ` : "")
       + "The sample is the request list in view, not the whole launch — with this few rows p99 is one request, "
       + "so read the bands, not the percentile.");
}

// One chip per band, count inside, width proportional to the count. Chips rather than a stacked bar on purpose:
// a 338/0/23/33/6 traffic shape drawn as one bar puts the interesting band (six requests over 64k) under a
// pixel wide, and the whole reason this card exists is the tail.
// One band per row, columns aligned with the axis labels in the HTML: band, bar, rows, share of rows, tokens,
// share of tokens. The bar is the ROW count (that is what the shape claim is), and `scale` comes from the caller
// across BOTH histograms so the two cards are comparable. The token pair is the cost claim: 48 rows in 64k-128k
// is a shape, the 3.9M tokens those rows carry is why anyone should care.
function lenBars(id, bands, counts, mass, scale, rows, sumTok) {
  const el = $(id);
  if (!el) return;
  const sc = Math.max(1, scale || 1);
  // A sidecar that publishes no per-band tokens is answered with em-dashes, not zeros: "0 tokens in this band" and
  // "this payload does not say" are different claims, and the second one is the one that is true.
  const hasMass = Array.isArray(mass);
  const html = bands.map((b, i) => {
    const c = counts[i] || 0;
    const tok = hasMass ? (((mass || [])[i]) || 0) : null;
    const w = c ? Math.max(1.5, (100 * c) / sc) : 0;
    const tail = (b.lo || 0) >= 8192 ? 1 : 0;
    const share = rows ? Math.round((100 * c) / rows) : null;
    const tshare = (sumTok && tok !== null) ? Math.round((100 * tok) / sumTok) : null;
    return `<div class="len-bar" data-band="${b.label}" data-tail="${tail}" data-count="${c}"`
         + ` data-mass="${tok === null ? "none" : tok}" data-scale="${sc}"`
         + ` title="${c} of the ${fmt(rows)} rows in view fall in ${b.label} tokens; those rows carry `
         + `${tok === null ? "no token count in this payload" : fmt(tok)} of the ${fmt(sumTok)} tokens this card `
         + `totals. Bar = rows, drawn on the same scale as the other card (full width = ${fmt(sc)} rows).">`
         + `<span>${b.label}</span><i class="track"><i style="width:${w.toFixed(2)}%"></i></i>`
         + `<b>${fmt(c)}</b><em>${share === null ? "\u2014" : `${share}%`}</em>`
         + `<span class="tok">${lenTok(tok)}</span><em>${tshare === null ? "\u2014" : `${tshare}%`}</em>`
         + `</div>`;
  }).join("");
  if (el.innerHTML !== html) el.innerHTML = html;
}

// Compact on purpose: seven rows of seven-digit token counts is a wall of digits, and this column's job is
// comparison, not arithmetic -- the exact total is the line above the card, and the card's note keeps the exact
// per-band counts in the tooltip.
function lenTok(n) {
  if (!isnum(n) || !n) return "\u2014";
  if (n >= 1e6) return `${fmt(n / 1e6, 1)}M`;
  if (n >= 1e3) return `${fmt(Math.round(n / 1e3))}k`;
  return fmt(n);
}

// A tick on the window bar. Hidden when there is nothing to place (no median in view, or a step larger than the
// window): a tick drawn at a guessed position is worse than no tick, which is the same rule the card applies to
// the prefill rate.
function lenTick(id, pct) {
  const el = $(id);
  if (!el) return;
  if (pct === null || pct === undefined || !Number.isFinite(pct)) {
    el.style.display = "none";
    return;
  }
  el.style.display = "";
  el.style.left = Math.max(0, Math.min(99.4, pct)).toFixed(1) + "%";
}

function paintLiveState(s) {
  const box = $("live-state");
  if (!box) return;
  const st = s.engine_state;
  // An older sidecar answers none of this. Hiding the strip is the honest rendering; an em-dash would imply a
  // measurement that found nothing, which is a different claim (the same rule the alarm row follows).
  if (!st && !s.alert) { box.hidden = true; return; }
  box.hidden = false;
  const W = { alive: ["Answering", "dot--live", ""],
              loading: ["Loading", "dot--busy", "busy"],
              dead: ["Gone", "dot--off", "danger"],
              wedged: ["Wedged", "dot--off", "danger"] };
  const w = W[st] || ["No state read", "", ""];
  $("live-dot").className = `dot ${w[1]}`;
  text("live-word", w[0]);
  box.dataset.tone = w[2];
  const n = isnum(s.ranks_holding) ? s.ranks_holding : null;
  const f = s.faults || null, now = f && f.now, al = s.alert && s.alert.card ? s.alert : null;
  // Short on the page, whole in the title. `slot()` cannot be used here: this is not a fixed-height card slot, it
  // is a flex line, and the note has to be able to shrink.
  text("live-note", [
    n ? `${n} rank(s) on the GPUs` : "",
    now && now.events ? `${now.events} driver fault(s) this launch` : "",
    al ? `alert on ${al.card}` : "",
  ].filter(Boolean).slice(0, 2).join(" \u00b7 "));
  box.title = `${w[0]}: ${
    { alive: "the engine answered /health on this poll and the ranks holding the GPUs are its own",
      loading: "no /health answer yet, nothing has died: this launch is still coming up, which is not the same "
               + "as dead and is not called dead here",
      dead: "no /health answer and no rank holding the GPUs: the launch is over",
      wedged: "the engine died and its workers did not \u2014 ranks go on holding ~29 GiB each while /health is "
              + "silent. It does not clear itself: STOP=1 bash serve/<recipe>, then start it again" }[st]
    || "this sidecar build does not report engine state"}`
    + (n ? `. ${n} rank(s) own a card (kernel driver records, not launch order)` : "")
    + (f && f.error ? `. Driver journal unreadable: ${f.error}` : "")
    + (now && now.events ? `. ${now.events} page-fault event(s) inside the launch this page describes`
                         : (f && f.total === 0 ? ". No page fault logged by the driver this boot" : ""))
    + (al ? `. Standing hardware claim: ${al.what || ""} \u2014 the Hardware screen has the action` : "")
    + ". The alarm row (engine state, faults, preemptions, big-prefill exposure) is on the Hardware screen.";
}

// The full account for one chain, as text. Labels are padded because the panel is `white-space:pre-wrap`, so
// what the reader sees is what the reader copies -- no HTML structure between them and the numbers.
function pcieDetailLines(c, phist = []) {
  const short = (b) => String(b || "").replace(/^0000:/, "");
  const devs = c.devices || [];
  const sum = (k) => devs.reduce((a, x) => a + (x[k] || 0), 0);
  const named = {};
  devs.forEach((x) => Object.entries(x.notable || {}).forEach(([k, v]) => { named[k] = (named[k] || 0) + v; }));
  const cn = Object.entries(named).filter(([k]) => !/#(nf|ft)$/.test(k))
    .map(([k, v]) => `${k} ${v}`).sort((a, b) => parseInt(b.split(" ")[1]) - parseInt(a.split(" ")[1]));
  const un = Object.entries(named).filter(([k]) => /#(nf|ft)$/.test(k))
    .map(([k, v]) => `${aerKind(k).name} ${v} ${aerKind(k).kind}`);
  const at = (s2) => (s2.chains || []).find((x) => x.bus === c.bus) || null;
  const last = phist.length ? at(phist[phist.length - 1]) : null;
  const cor = sum("cor"), unc = sum("nonfatal") + sum("fatal");
  const quiet = isnum(last?.quiet_s) ? last.quiet_s : null;
  const rate = isnum(last?.rate_per_min) ? last.rate_per_min : null;
  const span = Math.max(1, Math.round((last?.window_s || 300) / 60));
  const win = last && isnum(last.cor_d) ? last.cor_d + (last.nonfatal_d || 0) + (last.fatal_d || 0) : null;
  const ev = (((status && status.pcie_events) || []).filter((e) => e.bus === c.bus) || [])[0] || null;
  const L = [];
  const put = (k, v) => { if (v) L.push(k.padEnd(14, " ") + v); };
  put("chain", devs.map((x) => short(x.bdf)).join(" \u2192 ") + "  (gpu \u2192 root port)");
  const mxs = Math.max(...devs.map((x) => x.max_speed || 0), 0), mxw = Math.max(...devs.map((x) => x.max_width || 0), 0);
  put("link", (isnum(c.speed) ? `${c.speed.toFixed(1)} GT/s \u00d7${c.width}` : "\u2014")
    + (mxs ? `   (each port here is rated ${mxs.toFixed(1)} GT/s \u00d7${mxw})` : "")
    + (c.parked ? "  \u00b7 parked: D3hot, idling its links on purpose"
       : c.degraded ? "  \u00b7 BELOW its rating" : "  \u00b7 at its rating"));
  put("slot", `registered at ${c.slot_at || "?"} (${c.slot_role || "?"})`
    + `  \u00b7 ${devs.length} port(s) on this chain, gpu function at the endpoint ${short(c.bus)}`);
  put("since boot", `${cor} correctable \u00b7 ${unc} uncorrectable`
    + (cor || unc ? ` \u00b7 across ${devs.filter((x) => (x.cor || x.nonfatal || x.fatal)).length} device(s) of ${devs.length} on this chain` : ""));
  put("correctable", cn.join(" \u00b7 "));
  put("uncorrectable", un.join(" \u00b7 "));
  put("this window", win === null ? "no sidecar history for this chain" : `+${win} event(s)`);
  put("rate", rate === null ? "unknown: this sidecar predates rate_per_min"
      : `${rate.toFixed(2)} counters/min, measured over the ${span} min this serve has been up`);
  put("quiet", quiet === null ? "unknown" : `nothing has moved for ${ago(quiet)}`);
  put("last event", ev ? `${ev.bdf} ${ev.kind} +${ev.n} at ${new Date(ev.at * 1000).toLocaleTimeString()}`
      // "Nothing moved" and "this build does not report movements" are different facts, and the second one is
      // not this page's to report as the first: an older sidecar has no event list at all.
      : (status && status.pcie_events === undefined)
        ? "this serve's build does not report events \u00b7 restart it for the timestamps"
        : "no counter moved while this page has been open"
        + " \u00b7 anything before the engine started belongs to aer-watch and journal-pcie.py, not to this page");
  put("watched", `${phist.length} sidecar samples \u00b7 this page's clock starts after the weights are loaded`);
  return L;
}

// A pinned block of real text, OUTSIDE the card grid: #gpus is rebuilt every second, so anything inside it
// (and any handler attached to it) is thrown away each tick. Pinned state lives in `detailBus`, and the click
// is delegated to the container for the same reason.
let detailBus = null, detBuilt = null, detLines = [], detHead = null, detCopy = null;
function paintPcieDetail(st) {
  const box = $("pcie-detail");
  const ch = detailBus ? (((st && st.pcie) || []).find((x) => x.bus === detailBus) || null) : null;
  if (!ch) { detBuilt = null; detLines = []; box.hidden = true; box.replaceChildren(); return; }
  if (detBuilt !== detailBus) {                       // pinned to another chain: build once, then update
    detBuilt = detailBus; detLines = []; detHead = document.createElement("b");
    detCopy = document.createElement("button"); detCopy.type = "button"; detCopy.className = "chip";
    detCopy.textContent = "Copy";
    detCopy.onclick = () => navigator.clipboard.writeText(box.textContent)
      .then(() => { detCopy.textContent = "Copied"; setTimeout(() => { detCopy.textContent = "Copy"; }, 1200); })
      .catch(() => {});
    const x = document.createElement("button"); x.type = "button"; x.className = "chip"; x.textContent = "Close";
    x.onclick = () => { detailBus = null; paintPcieDetail(status); };
    const head = document.createElement("div"); head.className = "pcie-detail-head";
    head.append(detHead, detCopy, x, Object.assign(document.createElement("span"),
                                                  { className: "small", textContent: " Esc closes \u00b7 the text is selectable" }));
    box.replaceChildren(head);
  }
  detHead.textContent = `slot ${ch.slot ?? "-"} \u00b7 ${ch.card || "card?"} \u00b7 #${ch.pos ?? "?"} in slot order`
    + ` \u00b7 gpu ${ch.bus}`;
  box.hidden = false;
  pcieDetailLines(ch, (st && st.pcie_history) || []).forEach((t, i) => {
    let el = detLines[i];
    if (!el) { el = document.createElement("div"); el.className = "pcie-detail-line"; box.append(el); detLines[i] = el; }
    // Only what actually changed is written. Re-creating the block every second would destroy whatever text the
    // reader has selected to copy -- and being able to copy it is the entire reason this block exists.
    if (el.textContent !== t) el.textContent = t;
  });
}

// The cards are rebuilt every tick, so the handler lives on the container; a handler attached to a card would
// vanish with it. Only a chain that negotiated below its rating emits a `data-bus` badge, which is the state
// whose account is worth reading. Clicking the same card again unpins it.
$("gpus").onclick = (e) => {
  const b = e && e.target && e.target.closest ? e.target.closest("[data-bus]") : null;
  if (!b || !b.dataset || !b.dataset.bus) return;
  detailBus = detailBus === b.dataset.bus ? null : b.dataset.bus;
  paintPcieDetail(status);
};
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && detailBus) { detailBus = null; paintPcieDetail(status); }
});

// -- the CPU sidecars behind the gateway ---------------------------------------------------------
// serve/gateway.py answers /api/sidecars with each sidecar's own watcher state (serve/embed-watch.py), which
// is the only place llama.cpp's metrics dialect is understood. Two reasons this section is careful:
// * the page is also served on the engine's own port, where /api/sidecars does not exist -- a 404 hides the
//   section instead of leaving dead cards full of em-dashes in the grid;
// * what these cards exist to show is the memory line. The GGUFs map clean (reclaimable page cache), but each
//   serve's KV pool is anon and lands in Committed_AS, which is the number the 2026-11-01 wedge is about.
// Which cards exist, written out rather than derived from the payload: a card the page builds from a JSON
// field is a card whose slots no static check can see, and "every slot in the markup is written"
// (tests/test_webui.py) is exactly the check that would go blind. A third serve on the box shows up in
// /api/sidecars but not here -- adding it means adding a row here and a card in monitor.html.
const SIDECAR_CARDS = [
  ["embed-card", "embed-dot", "embed-name", "embed-rate",
   "embed-args", "embed-slots", "embed-mem"],
  ["rerank-card", "rerank-dot", "rerank-name", "rerank-rate",
   "rerank-args", "rerank-slots", "rerank-mem"],
];

async function paintSidecars() {
  let list = [];
  try {
    const d = await api("/api/sidecars");
    list = (d && d.sidecars) || [];
  } catch (e) {
    list = [];                                    // no gateway on this port (or it is not answering)
  }
  const sec = $("sidecars");
  if (!list.length) { sec.hidden = true; return; }
  sec.hidden = false;
  SIDECAR_CARDS.forEach(([card, dot, name, rate, args, slots, mem], i) => {
    const s = list[i];
    if (!s) { $(card).hidden = true; return; }
    $(card).hidden = false;
    const r = s.rates || {}, v = s.serve || {}, h = s.host || {};
    text(name, s.name || "sidecar");
    $(dot).className = `dot${s.health === 200 ? " dot--live" : " dot--off"}`;
    // llama.cpp publishes no request counter, so the rate is prompt TOKENS per second: the serve's own gauge
    // when it has one, else the watcher's own delta over its window.
    const tok = isnum(r.rate) ? r.rate : r.ptok;
    $(rate).innerHTML = `${isnum(tok) ? dnum(tok, 0) : "\u2014"}<span class="unit"> tok/s</span>`;
    slot(args, `-t ${v.threads || "?"} \u00b7 -c ${v.ctx || "?"}`,
         `started with -t ${v.threads || "?"} -c ${v.ctx || "?"} -b ${v.batch || "?"}; the thread count is the online`
         + ` cpu count inside its taskset range, and one thread more than that costs real throughput`);
    slot(slots, `busy ${isnum(r.busy) ? dnum(r.busy, 0) : "\u2014"} \u00b7 def `
                + `${isnum(r.deferred) ? dnum(r.deferred, 1) : "\u2014"}`,
         "busy slots and requests deferred per second (llama.cpp runs 4 slots by default; past that, concurrency "
         + "buys nothing -- raise --parallel instead of the client count)");
    slot(mem, `rss ${isnum(h.embed_rss_kb) ? dnum(h.embed_rss_kb / 1048576, 2) : "\u2014"} \u00b7 host `
              + `${isnum(h.committed_as_kb) ? dnum(h.committed_as_kb / 1048576, 1) : "\u2014"}`,
         "this serve's RSS (weights touched + its KV pool) and the host's Committed_AS: the GGUF maps clean and "
         + "is reclaimable page cache, the KV pool is anon and lands in Committed_AS -- the 2026-11-01 red line "
         + "(notes/load-oom-svm-wedge.md)");
  });
}

// -- the gateway's admission gate ----------------------------------------------------------------
// GET /api/gate is answered only by serve/gateway.py. On the engine's own port that route does not exist, and a
// 404 hides the row rather than leaving two cards of em-dashes -- the sidecar rule, for the same reason.
// Why the row exists at all: the engine's cards say what the KV pool is doing, and they cannot say what the
// front door did about it. A request that waited 20 s in the gateway and then ran at full speed is invisible in
// every engine counter (it never reached the engine while it waited); these two cards are its only record.
// The numbers behind the thresholds are in notes/admission-gate.md: step_ms ~= 13.6 + 1.79 x N + 14.9 ns x
// resident tokens, so the pool (902,153 tokens on this box) is the budget a long request spends, not the
// sequence count.
// Engine state, the driver's fault count, big-prefill exposure and preemptions. The first two come from the
// sidecar's own reads (KFD and the kernel journal), NOT from the engine -- which is the point: the state this row
// exists to catch is a state in which the engine is not answering. Wedged is the state the record missed for 26
// minutes on 2026-11-15 (setup.py's watch_dead now names it); the fault count is the only place six of this box's
// faults were ever written down; and the exposure card is the one that keeps the other three interpretable: if
// nothing over the routing bound ever arrives, "no faults" is not evidence about anything.
function paintWatchdog(s) {
  const sec = $("watchdog");
  const st = s.engine_state, f = s.faults, live = s.live || {}, al = s.alert && s.alert.card ? s.alert : null;
  // A page newer than the sidecar it is talking to: none of these reads exist in that build, and an em-dash row
  // would imply the build measured and found nothing. Hide the row.
  if (!st && !f && !s.alert && live.prompt_big_pct === undefined) { sec.hidden = true; return; }
  sec.hidden = false;
  const WORD = { alive: ["Answering", "dot--live"], loading: ["Loading", "dot--busy"],
                 dead: ["Gone", "dot--off"], wedged: ["Wedged", "dot--off"] };
  const w = WORD[st] || ["\u2014", ""];
  $("wd-dot").className = `dot ${w[1]}`;
  text("wd-state", w[0]);
  const n = isnum(s.ranks_holding) ? s.ranks_holding : null;
  // The line is short and the whole sentence goes in the title, because these are 4-up cards and the harness
  // fails a slot that wraps (the card's height moves and the row stops being a row). What is NOT abbreviated is
  // the actionable half: for a wedged launch the note names the ranks and the title names the command.
  const note = {
    alive: [`${n} rank(s) own the GPUs`,
      `${n} vLLM rank(s) hold a context on a card, read out of the kernel driver's per-process records (KFD), `
      + "not out of the launch order \u2014 which is not the same thing once someone runs GPUS=3,0,1,2 on purpose"],
    wedged: [`${n} rank(s) still hold`,
      `No /health answer while ${n} rank(s) go on holding ~30 GiB each: the engine died and its workers did not. `
      + "On 2026-11-15 that state ran 26 minutes with the record and this page both saying healthy. It is not "
      + "something to wait out \u2014 the recipe's own stop clears it: STOP=1 bash serve/<recipe>."],
    dead: ["no ranks remain",
      "No /health answer and no rank holding the GPUs: the launch is over rather than stuck."],
    loading: ["never answered yet",
      "No /health answer, nothing has died, and no crash signature: this launch is still coming up (weights, "
      + "graph capture), which is not the same as dead and must not be called dead."],
  }[st] || ["\u2014", "this build does not report engine state"];
  slot("wd-state-note", note[0], note[1]);
  // A [gfxhub] page fault is not weather: it is a rank dying, and the launch that took it is over. So the headline
  // answers "has the launch this page is describing faulted", and the rest of the boot goes to the note and the
  // title as history. A badge that kept counting since boot would cry wolf at the machine in front of the reader
  // -- and the card badges follow the same rule (common.js gpuCard reads `faults_launch`).
  const now = f && f.now, earlier = f && f.earlier, attributed = Boolean(now);
  const dot = $("wd-fault-dot"), lbl = $("wd-fault-label");
  if (dot) {
    dot.className = al ? "dot dot--alert"
                       : `dot${attributed ? (now.events ? " dot--off" : " dot--live") : ""}`;
  }
  // The label says WHICH population the number is from, and it moves when the population does: a card headed
  // "this launch" must never carry a since-boot number, because that is exactly the reading a tired operator
  // will otherwise take. With a standing hardware alert the question changes again: not "has this launch
  // faulted" but "this function keeps faulting, whatever runs on it".
  if (lbl) text("wd-fault-label", al ? "Hardware: page faults"
             : (attributed ? "Page faults, this launch" : "Page faults, this boot"));
  if (!f || f.total === null || f.total === undefined) {
    text("wd-faults", "unknown");
    slot("wd-fault-note", "journal unreadable",
        (f && f.error) || "this build does not read the kernel journal, so it cannot say either way \u2014 and "
        + "'no faults' and 'cannot see the kernel log' are different claims");
  } else if (attributed && now.events) {
    // This launch has faulted. Loud, per card, and the count is events not driver lines: one crash can write
    // several lines (11 lines across 8 events on this box since boot, three in the same second on 11-06).
    text("wd-faults", `${dnum(now.events, 0)}`);
    slot("wd-fault-note", Object.entries(now.by_card || {}).sort((a, b) => b[1] - a[1])
        .map(([b2, k]) => `${b2} \u00d7${k}`).join(" \u00b7 ").slice(0, 24),
        `${now.events} page-fault event(s), ${now.total} driver line(s), inside the launch this page describes. `
        + "A page fault is a rank dying: the engine usually does not survive it, and the card under each one is "
        + "named on that card (notes/bug-causal-conv1d-oob.md \u00a715: the card was the constant, the rank was not)."
        + (f.last ? ` Last: ${f.last.at}${f.last.card ? ` on ${f.last.card}` : ""}`
                   + `${f.last.pid ? ` pid ${f.last.pid}` : ""}.` : ""));
  } else if (attributed) {
    text("wd-faults", "0");
    slot("wd-fault-note", `none \u00b7 ${isnum(earlier.events) ? earlier.events : "?"} before`,
        `Nothing has faulted in the launch this page is describing. Earlier this boot: ${earlier.events} event(s), `
        + `${earlier.total} line(s)`
        + (Object.keys(earlier.by_card || {}).length
           ? `, on ${Object.entries(earlier.by_card).map(([b2, k]) => `${b2} \u00d7${k}`).join(", ")}` : "")
        + " -- those belong to launches that are over, and the rank that faulted with them is gone, which is why "
        + "the cards do not carry them. Anchor for \u201cthis launch\u201d: "
        + `${f.anchor ? new Date(f.anchor * 1000).toLocaleString() : "unknown"} \u2014 the earliest of the launch's `
        + "own log file and this page's start (webui launch_anchor)."
        + (f.last ? ` Last fault this boot: ${f.last.at}${f.last.card ? ` on ${f.last.card}` : ""}`
                   + `${f.last.pid ? ` pid ${f.last.pid}` : ""}.` : ""));
  } else {
    // No anchor (or an unparseable journal stamp): the split cannot be made, and the fallback is the since-boot
    // number WITH that said -- "cannot attribute" must not be allowed to look like "this launch is clean".
    text("wd-faults", `${dnum(isnum(f.events) ? f.events : f.total, 0)}`);
    slot("wd-fault-note", Object.entries(f.by_card || {}).sort((a, b) => b[1] - a[1])
        .map(([b2, k]) => `${b2} \u00d7${k}`).join(" \u00b7 ").slice(0, 24),
        "These faults could not be split between this launch and earlier ones (no anchor, or the journal stamps "
        + "could not be parsed), so the card is headed \u201cthis boot\u201d and the per-card counts must be read as "
        + "history, not as a statement about the engine running now."
        + (f.last ? ` Last: ${f.last.at}${f.last.card ? ` on ${f.last.card}` : ""}`
                   + `${f.last.pid ? ` pid ${f.last.pid}` : ""}.` : ""));
  }
  // A standing hardware alert takes the card over, because it is the claim that outranks a count. The number it
  // shows is the alert's own (events on that function inside the window) and the note names the function, so the
  // card cannot read as "this launch is fine" while the box is being told to go and look at a card. Nothing is
  // lost: the launch/boot split is still what the per-card badges say, and every assertion ever made is in
  // .r9700-state/alerts.jsonl.
  if (al) {
    text("wd-faults", `${dnum(al.count, 0)}`);
    const win = Math.max(1, Math.round((al.window_s || 0) / 86400));
    // Whether a HUMAN was told is part of the claim, not a detail under it. A red card that nobody could have
    // seen is the failure mode of every monitoring system ever built, and the robot's refusal (wrong keyword,
    // dead token) is otherwise invisible until someone asks "did you get a message?" three days later.
    const nv = al.notify || null;
    // Three answers, not two. A claim raised before any channel existed is neither delivered nor refused, and
    // calling that "no channel" would be false the moment someone configures one -- so the page asks its own
    // /api/notify and says `untested`, which is the one thing an operator can act on.
    const told = nv ? (nv.sent ? "pinged" : "NOT pinged")
                    : (s.notify && s.notify.configured ? "untested" : "no channel");
    if (lbl) text("wd-fault-label", `Hardware: page faults \u00b7 ${told}`);
    slot("wd-fault-note", `${al.card} \u00d7${dnum(al.count, 0)} / ${win} d`.slice(0, 24),
         `${al.what || ""} \u2014 ${al.do || ""} \u2014 raised `
         + `${isnum(al.age_s) ? `${Math.round(al.age_s / 60)} min` : "?"} ago by setup.py's watcher; the shape of `
         + "each fault (read/write, UTCL2 client, ring) is in the alert, and one crash can log several driver "
         + "lines, so the count is events, not lines. "
         + (nv ? `The ping: ${nv.sent ? "accepted by the robot" : "REFUSED by the robot"} \u2014 errcode `
               + `${nv.errcode === null || nv.errcode === undefined ? "?" : nv.errcode}, `
               + `${JSON.stringify(nv.errmsg || nv.error || "")}, attempted ${nv.at || "?"}.`
               : ((s.notify && s.notify.configured)
                 ? "No send attempt recorded for THIS alert: it was raised before the channel existed (or by a "
                   + "build that did not send). A channel is configured \u2014 About \u2192 Alerts, or "
                   + "tools/fault-report.py --alert-test."
                 : "No send attempt recorded and no channel configured: this claim stays on the page and in "
                   + "alerts.jsonl and nobody is pinged. Set one on About \u2192 Alerts.")));
  } else if (s.alert_error) {
    text("wd-faults", "?");
    slot("wd-fault-note", "alert unreadable", String(s.alert_error));
  }
  // Requests, not steps: /metrics has no per-step prefill-token series, and a big prompt is what puts a big
  // prefill into a step. The bound is the nearest bucket vLLM publishes (1 000), printed rather than implied,
  // because the recipe's knob is 1 024 and 1 000 is not that number.
  const pct = live.prompt_big_pct;
  text("wd-big", isnum(pct) ? `${fmt(pct, 1)}%` : "\u2014");
  slot("wd-big-note", isnum(live.prompt_bound)
    ? `${dnum(live.prompt_big ?? 0, 0)}/${dnum(live.prompt_n ?? 0, 0)} over ${dnum(live.prompt_bound, 0)}`.slice(0, 24)
    : "no histogram",
    isnum(live.prompt_bound)
      ? `${dnum(live.prompt_big ?? 0, 0)} of ${dnum(live.prompt_n ?? 0, 0)} finished requests in this window had a `
        + `prompt over ${dnum(live.prompt_bound, 0)} tokens \u2014 the nearest bucket vLLM publishes, which is NOT `
        + `the recipe's bound. Since launch: ${dnum(live.prompt_big_all ?? 0, 0)} of ${dnum(live.prompt_n_all ?? 0, 0)}`
        + `${isnum(live.prompt_big_all_pct) ? ` (${fmt(live.prompt_big_all_pct, 1)}%)` : ""}. A step whose prefill `
        + "tokens clear the recipe's R9K_GDN_PREFILL_MAX still reaches stock's chunked GDN core, i.e. still builds "
        + "the step shape that carried six device faults. Without this number \u201cno faults since the change\u201d "
        + "is uninterpretable: it can mean the trigger is gone, or that the traffic never got near it."
      : "this build publishes no prompt-size histogram");
  text("wd-preempt", isnum(live.preemptions) ? `${dnum(live.preemptions, 0)}` : "\u2014");
  slot("wd-preempt-note", isnum(live.preemptions_total)
    ? `${dnum(live.preemptions_total, 0)} since launch`.slice(0, 24) : "no preemption series",
    isnum(live.preemptions_total)
      ? `The number on the card is this window; ${dnum(live.preemptions_total, 0)} since launch. A preemption is the `
        + "pool taking blocks back from a running request, and the victim pays a whole re-prefill \u2014 which is "
        + "how a slow window starts long before anything fails"
        + `${isnum(live.waiting_capacity) ? `. Waiting on capacity right now: ${live.waiting_capacity}` : ""}.`
      : "this build reports no preemption series");
}

async function paintGate() {
  let g = null;
  try { g = await api("/api/gate"); } catch (e) { g = null; }        // no gateway on this port
  const sec = $("gate");
  if (!g || !g.config) {
    sec.hidden = true;
    // No /api/gate on this port (the engine's own :8180 answers none): the strip hides rather than showing a door
    // that was never measured.
    const box = $("door-state");
    if (box) box.hidden = true;
    return;
  }
  sec.hidden = false;
  const c = g.config, n = g.counters || {}, gg = g.gauges || {};
  // A gateway old enough to report no `policy` field WAS holding: the words follow the payload, never the page.
  // Shipped since 2026-11-16: observe. What changed in these two cards is the verb -- they used to say what the
  // door did to your request, and now they say what the engine's harm signals say and what the door WOULD have
  // done. Same numbers, minus an intervention measured at +5.0 s on 25 of 28 holds while freeing no KV block at
  // all (notes/admission-gate.md §7).
  const OBS = (g.policy || "hold") !== "hold";
  const sh = g.shut || {};
  // open = the door is up (live); holding = traffic is waiting (busy, not danger: a hold is the gate working);
  // harm = the engine's harm signals are on and the door is deliberately not acting on them; off = switched off
  // at the front door (dot--off is the file's "not running" dot).
  $("gate-dot").className = `dot${!c.GATE_ON ? " dot--off" : g.closed ? " dot--busy" : " dot--live"}`;
  text("gate-state", !c.GATE_ON ? "Off" : OBS ? (g.closed ? "Harm" : "Watch") : (g.closed ? "Held" : "Open"));
  // The first-screen strip: one word about the door, in the same line as the engine's state. The row below this
  // one has the 15-minute history and the counters; what has to be visible while the rates are being read is
  // whether a request arriving now would be held, because that is the difference between "the engine is slow" and
  // "the front door has not let mine in yet" -- and after the 2026-11-15 afternoon, when the door stayed shut for
  // 180 polls straight, the second explanation has to be one glance away, not two screens.
  {
    const box = $("door-state");
    if (box) {
      box.hidden = false;
      const word = !c.GATE_ON ? "Door off" : OBS ? "Door watching" : (g.closed ? "Door held" : "Door open");
      text("door-word", word);
      $("door-dot").className = `dot${!c.GATE_ON ? " dot--off" : g.closed ? " dot--busy" : " dot--live"}`;
      box.dataset.tone = !c.GATE_ON ? "" : g.closed ? "busy" : "";
      const n2 = g.counters || {}, held = n2.held_now ?? n2.held ?? 0;
      // Observe mode says the reassuring half OUT LOUD: a reader must not need to know the code to know whether
      // the front door is standing between them and the engine. Nothing is held here; the number kept is how many
      // arrivals the harm signal found.
      text("door-note", !c.GATE_ON ? "the gate is switched off at the front door"
        : OBS ? `${g.closed ? "harm signal on" : "no harm signal"} · it holds nothing`
        : g.closed ? `${held} request(s) waiting at it` : `small ${dnum(n2.small_passed ?? 0)} passed, ${dnum(held)} held`);
      box.title = `${word}: ${(g.why || "").trim() || (c.GATE_ON ? "no harm signal, so arrivals go straight to the engine"
        : "GATE_ON=0 -- every request goes through regardless of engine state")}`
        + (OBS
          ? `. GATE_POLICY=observe: it reads the engine and holds, refuses and delays NOTHING. What it would have
             done is counted: ${dnum(n2.arrived_shut ?? 0)} request(s) arrived while the harm signal was on
             (${dnum(n2.arrived_shut_big ?? 0)} declaring a body over ${c.GATE_SMALL_BYTES} B, ${dnum(n2.arrived_shut_small ?? 0)}
             under it).` + (isnum(sh.frac)
             ? ` The harm signal was on for ${Math.round(sh.frac * 100)}% of the ring; its current unbroken span is
               ${sh.run_s}s and the longest inside the ring is ${sh.max_s}s.` : "")
             + ` The gate's history and every counter are on the gate row (Recent screen).`
          : `. Hold ceiling ${c.GATE_HOLD_S}s, harm window ${c.GATE_PREEMPT_S}s, no more than ${c.GATE_MAX_HELD} held `
            + `at once, and a declared body of <= ${c.GATE_SMALL_BYTES} B passes a shut door. The gate's history, "was `
            + `it shut at 15:05", and every counter are on the gate row (Recent screen). Held means the gateway opened `
            + `no upstream connection at all: the hop stays a byte pipe and no request body was read.`);
    }
  }
  // the headline is the running total since the gateway started, not "right now": a hold is a thing that
  // happened to a request, and the number a reader wants after the fact is how many it happened to. Under observe
  // the equivalent number is how many arrivals FOUND the harm signal on -- the exact set a holding policy would
  // have held, which is the only way to price a policy without running it on a live box to find out.
  text("gate-held", dnum(OBS ? (n.arrived_shut ?? 0) : (n.held ?? 0)));
  if ($("gate-held-unit")) text("gate-held-unit", OBS ? "while harm" : "held");
  if ($("gate-holds-label")) text("gate-holds-label", OBS ? "Arrivals at the door" : "Holds at the door");
  spark($("gate-spark"), g.hist, { baseLo: 0, baseHi: 1 });
  $("gate-spark").dataset.tone = g.closed ? "warn" : "ok";
  // How far back the strip reaches, printed rather than inferred. The strip exists to answer "was the door shut at
  // 15:05?", and that is only checkable if the span is stated: a ring of polls without a span is a picture that
  // cannot be lined up against a request row from an hour ago.
  if (isnum(g.hist_span_s)) {
    $("gate-spark").title = `${(g.hist || []).length} polls over `
      + `${g.hist_span_s >= 120 ? Math.round(g.hist_span_s / 60) + " min" : g.hist_span_s + " s"}`
      + " -- one bar is one poll of the door, shut or open. A poll whose gauges were missing or stale draws OPEN: "
      + "fail-open is shown, not hidden, so a telemetry fault is never mistaken for traffic.";
  }
  // The reason the gateway gave is one line here and the whole string in the title. The line takes the gateway's
  // own first clause ("3 preemption(s) observed, door shut 12 s more") and the title carries its exact full
  // string, token count and all ("kv 0.860 >= 0.92 (775,650 of 902,153 tokens)"): the tokens are that arithmetic,
  // and a second copy of it in the page is the copy that goes out of date. The tightest 4-up card holds 24
  // characters at 12 px -- the budget the render harness measures -- so the long form goes where it is still
  // readable in full.
  const whyClause = (g.why || "").split(";")[0].split(" (")[0].trim();
  // The preemption reason is the one the gateway words in a sentence rather than an inequality, and the tightest
  // 4-up card holds 24 characters -- so the page shortens it HERE rather than widening the card, keeping the two
  // facts that matter (what is closing it, and for how long much longer) and leaving the gateway's exact string
  // in the title. The pool clause needs no shortening: "kv 0.860 >= 0.92" is already inside the budget.
  const shutFor = isnum(g.thrash_left) && g.thrash_left > 0
    ? `preempting, ${Math.round(g.thrash_left)} s more` : whyClause;
  slot("gate-why", !c.GATE_ON ? "off" : g.closed ? (shutFor || "at the cliff")
       : (g.err ? "open: no gauge" : "no harm signal"),
       (g.why || "") + (g.err ? ` ${g.err}` : "")
       + " -- the harm signals, in the engine's own words: num_preemptions_total MOVING (blocks taken back from a "
       + "running request) and a pool at GATE_KV_HIGH. It is not the engine having a queue: a batch of long prompts "
       + "parks the pool above any useful threshold for its whole duration, and closing on that measured as +25 s on "
       + "every new arrival for no protection at all (notes/admission-gate.md \u00a76). Under GATE_POLICY=observe "
       + "nothing acts on this string; it is the report, and what the door would have done is counted on the card "
       + "beside it. It fails OPEN when the gauges are missing or stale: a front door that stops serving because its "
       + "telemetry broke is worse than no gate.");
  // The pool in tokens, which is the axis the threshold is really about (0.85 of 902,153 is a number of tokens,
  // and one 80k request is 53 short ones). The percent is on the engine's own KV card; the waiting count is on
  // the Requests card. Repeating either here would cost the characters this line needs for the tokens.
  slot("gate-level", isnum(gg.kv)
       ? `${dnum(Math.round(gg.kv * (c.pool_tokens || 0)))} of ${dnum(c.pool_tokens || 0)} tok`
       : "no gauge reading",
       `engine gauges: kv ${gg.kv}, running ${gg.run}, waiting ${gg.wait}, preemptions ${gg.prem} (cumulative; `
       + "the gate compares polls, so it closes when that counter moves). A sequence costs 1.8 ms of step time and "
       + "its own tokens of pool, so the pool is the budget a long request spends (notes/admission-gate.md).");
  const hasSmall = isnum(n.small_passed);
  // OBSERVE: the line carries the traffic SPLIT rather than a hold outcome, because that split is the pair of
  // numbers the 2026-11-16 afternoon showed the door had backwards. What it would have held -- a declared body over
  // GATE_SMALL_BYTES, i.e. a long PROMPT -- was 94.8% served from the prefix cache and cost 7-37 s apiece, while
  // the four rows that actually filled the pool declared ~1.2 KB each and passed the size test. "big" is a declared
  // size CLASS, not a pool cost, and the gloss says so out loud.
  slot("gate-counters", OBS
       ? `big ${dnum(n.declared_big ?? 0)} \u00b7 small ${dnum(n.declared_small ?? 0)}`
       : hasSmall ? `expired ${dnum(n.expired || 0)} \u00b7 small ${dnum(n.small_passed)}`
       // an older gateway has no small-pass escape to report; show its released/expired pair rather than a 0 that
       // would read as "nothing came through".
       : `released ${dnum(n.released || 0)} \u00b7 expired ${dnum(n.expired || 0)}`
       // refused is the one that only exists under the other policy (GATE_HOLD_S=0), and held-over is the one that
       // only exists when the hold queue hits its bound; both are worth the line running wide when they are not
       // zero, because either one is the gate choosing traffic over protection and a silent count would hide that.
       + (n.refused ? ` \u00b7 refused ${dnum(n.refused)}` : "") + (n.held_over ? ` \u00b7 over ${dnum(n.held_over)}` : ""),
       OBS
       ? `GATE_POLICY=observe: the gate reads the engine's harm signals and holds, refuses and delays NOTHING, so
          the headline is the count that replaces "held" -- ${dnum(n.arrived_shut ?? 0)} gated request(s) arrived
          while the harm signal was on, and that is exactly the set a holding policy would have held
          (${dnum(n.arrived_shut_big ?? 0)} of them declaring a body over ${c.GATE_SMALL_BYTES} B, ${dnum(n.arrived_shut_small ?? 0)}
          under it). Since the gateway started: ` + [`seen ${dnum(n.gated_seen ?? 0)} requests on the gated paths`,
            `declared big ${dnum(n.declared_big ?? 0)} (over ${c.GATE_SMALL_BYTES} B)`,
            `declared small ${dnum(n.declared_small ?? 0)}`,
            `arrived ${dnum(n.arrived_shut ?? 0)} while the harm signal was on`, `admitted ${dnum(n.admitted ?? 0)}`,
            `held ${dnum(n.held ?? 0)} and refused ${dnum(n.refused ?? 0)}, both of which stay 0 while observing`,
            isnum(sh.frac) ? `door shut on ${Math.round(sh.frac * 100)}% of the ring, ${sh.run_s}s unbroken now and `
              + `${sh.max_s}s the longest run inside it` : ""].filter(Boolean).join(", ")
          + ". A declared body is a PROMPT size, not a pool cost: the rows that filled this box's pool on "
          + "2026-11-16 declared ~1.2 KB each (max_tokens unset, so their commitment was the rest of the "
          + "262,144-token window), and the long-prompt rows were the ones never preempted "
          + "(notes/admission-gate.md \u00a77)."
       : "expired: it went anyway after the ceiling, because a 30-minute TTFT is the failure this exists to prevent, "
       + "not something to reach for. small: requests whose declared body is under GATE_SMALL_BYTES, admitted "
       + "through a shut door -- they are the ones a pool threshold starves, and they are not what fills the pool. "
       + "Since the gateway started: " + [`held ${dnum(n.held ?? 0)}` + (isnum(n.held_now) ? ` (${dnum(n.held_now)}`
       + ` holding now)` : ""), `released ${dnum(n.released ?? 0)}`, `expired ${dnum(n.expired ?? 0)}`].concat(
         hasSmall ? [`small ${dnum(n.small_passed)}`] : []).concat(
         isnum(n.held_over) ? [`over ${dnum(n.held_over)} (admitted because GATE_MAX_HELD was reached)`] : [],
         [`refused ${dnum(n.refused ?? 0)} (a 503, which only GATE_HOLD_S=0 issues)`,
          `admitted ${dnum(n.admitted ?? 0)} never met a shut door`]).join(", ")
       + ". expired against held is the ratio that says whether the ceiling is a bound or just a delay.");
  slot("gate-ceil", OBS
       // The door's own duty cycle, in one line. The strip draws 900 dots and cannot say "it is effectively shut",
       // which on 2026-11-16 WAS the finding: 0.71 of the ring, a 339 s unbroken span and still running, with a
       // preemption rate of 12.9/min against a 20 s window -- arithmetic under which the door has no way to open.
       ? `shut ${isnum(sh.frac) ? `${Math.round(sh.frac * 100)}%` : "?"} \u00b7 ${isnum(sh.run_s) ? `${sh.run_s}s` : "?"}`
       : isnum(c.GATE_PREEMPT_S) ? `hold ${c.GATE_HOLD_S}s \u00b7 win ${c.GATE_PREEMPT_S}s`
       // an older gateway's payload has no preemption window to name; show its poll rate rather than print
       // 'undefined' into the page.
       : `hold ${c.GATE_HOLD_S}s \u00b7 poll ${c.GATE_METRICS_S}s`,
       OBS
       ? `${g.polls} polls so far, every ${c.GATE_METRICS_S}s; gauges older than ${c.GATE_STALE_S}s count as unknown,
          and unknown opens the door. GATE_POLICY=observe: the thresholds below are still computed, drawn and
          counted, and no request is held, refused or delayed. What WOULD close it: ${c.GATE_KV_HIGH} on the pool
          axis (re-open at ${c.GATE_KV_LOW}) or a preemption within ${c.GATE_PREEMPT_S}s; the small class is a
          declared body of ${c.GATE_SMALL_BYTES} B or less, and at most ${c.GATE_MAX_HELD} would ever be held. To
          arm the intervention: GATE_POLICY=hold bash serve/gateway.sh (restarts the front door only, ~2 s -- and
          bounces the streams already passing through it). Neither way this door calls itself shut is guaranteed to
          END: an event every ${c.GATE_PREEMPT_S}s re-arms the window, and a pool whose steady state sits above
          ${c.GATE_KV_LOW} latches the pool axis, which is why the duty cycle is printed here instead of left to be
          counted off the strip (notes/admission-gate.md §7).`
       : `${g.polls} polls so far; gauges older than ${c.GATE_STALE_S}s count as unknown, and unknown opens the `
       + `door. hold: the ceiling on any single hold (shipped 5 s, because 9 of v1's 14 holds ended by hitting its `
       + `25 s -- a bound usually reached is a delay). window: how long after the LAST preemption the door stays `
       + `shut, which is what makes the shutdown end under a long batch. pool axis: shut at ${c.GATE_KV_HIGH}, `
       + `re-open at ${c.GATE_KV_LOW}. small: a declared body <= ${c.GATE_SMALL_BYTES} B passes a shut door. `
       + `held: never more than ${c.GATE_MAX_HELD} at once, past that the gate admits rather than queueing, and `
       + `poll every ${c.GATE_METRICS_S}s.`);
}

// -- one tick -------------------------------------------------------------------------------------
async function refresh() {
  if (paused) return;
  try {
    const [s, h] = await Promise.all([api("/api/status"), api("/api/requests")]);
    status = s; records = h.requests;
    brandStack(s);
    if (selected && !records.some((r) => rid(r) === selected)) {
      selected = detail = null;
      $("selection").hidden = true; $("no-selection").hidden = false;
    }
    // Nobody has clicked anything: follow the newest row there is, preferring one still in flight. That is
    // the request worth looking at, and it is what the panel was empty for -- in merged mode a row only
    // appears once its request has finished, so the newest row IS the last thing the serve did.
    if (follow) {
      const rows = visibleRows();
      const c = rows.find(active) || rows[0];
      if (c && rid(c) !== selected) { selected = rid(c); detail = null; }
    }
    text("kept", String(h.kept));
    lastOk = Date.now() / 1000; lastErr = "";
    $("observe-note").textContent = h.mode === "observe"
      ? "Start this sidecar with --proxy (or point a client at it) to record requests here."
      : h.mode === "merged" ? "Rows come from the engine as its own per-request stats: timings and token "
        + "counts, never bodies, and a request appears when it finishes. Captured bodies: serve/webui.sh "
        + "with WEBUI_MODE=proxy." : "";
    renderStatus(s);
    paintSidecars();                       // its own fetch, its own failure: never gates the engine's numbers
    paintGate();                           // same: a gateway route the engine's port does not answer is not a
                                           // reason to hold up the engine's own cards
    list();
    if (selected) {
      const row = records.find((r) => rid(r) === selected);
      const sig = row ? `${row.state}|${row.wallclock_s ?? ""}` : "gone";
      // fetch it when it changed, and every tick while it is in flight (its timings are still moving);
      // a finished request that nobody touched does not get refetched every 2 s
      if (detailSig !== sig || (row && active(row)) || !detail) { detailSig = sig; await showDetail(); }
    }
    text("updated", `Live · ${clock(s.at)}` + (s.mode === "observe" ? " · observing"
      : s.mode === "merged" ? " · inside the serve" : " · recording"));
    if (!actionError && s.error) notice(`upstream not answering: ${s.error}`);
    else if (!actionError) notice("");
  } catch (e) {
    lastErr = e.message;
    notice(e.message);
    text("updated", "Disconnected · retrying");
  }
}

// -- controls -------------------------------------------------------------------------------------
$("pause").onclick = () => {
  paused = !paused;
  $("pause").setAttribute("aria-pressed", String(paused));
  $("pause").textContent = paused ? "Resume" : "Pause";
  if (!paused) refresh();
};
$("key-btn").onclick = () => { $("auth").hidden = !$("auth").hidden; if (!$("auth").hidden) $("api-key").focus(); };
$("auth").onsubmit = async (ev) => {
  ev.preventDefault();
  setKey($("api-key").value);
  $("auth").hidden = true;
  await refresh();
};
document.addEventListener("need-key", () => { if ($("auth").hidden) $("auth").hidden = false; });
$("search").oninput = list;
$("filter").onchange = list;
// The two window pickers. Handlers by id rather than a querySelectorAll sweep, so the stub-DOM harness can
// click them: it has no querySelectorAll, and a control the harness cannot press is a control it cannot check.
["live", "1m", "15m", "1h"].forEach((k) => {
  $("win-" + k).onclick = () => {
    rateWin = k;
    try { localStorage.setItem("r9k.rateWin", k); } catch (e) { /* private mode: the choice just does not stick */ }
    if (status) renderStatus(status);
  };
});
[["fast", "fast"], ["slow", "slow"]].forEach(([id, w]) => {
  $("trend-" + id).onclick = () => {
    trendWin = w;
    try { localStorage.setItem("r9k.trendWin", w); } catch (e) { /* as above */ }
    if (status) renderStatus(status);
  };
});
$("follow").onclick = () => {                      // clicking a row pins it; this unpins
  follow = !follow;
  $("follow").setAttribute("aria-pressed", String(follow));
  if (follow) { selected = detail = null; detailSig = ""; refresh().catch(() => {}); }
};
document.querySelectorAll("[data-pane]").forEach((b) => {
  b.onclick = () => {
    pane = b.dataset.pane;
    document.querySelectorAll("[data-pane]").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
    content();
  };
});
$("copy").onclick = async () => {
  try {
    await navigator.clipboard.writeText($("content").textContent);
    text("copy", "Copied");
    setTimeout(() => text("copy", "Copy"), 1500);
  } catch (_) { notice("Clipboard unavailable; select the text and copy it yourself."); }
};
bindTheme();

// A background tab is throttled by the browser (Chrome clamps timers to about one tick a minute after five
// minutes hidden, and can freeze the tab outright), and a fetch that stalls no longer has to be fatal now that
// every api() call has a deadline -- but either way the reader has to be able to tell "the box is quiet" from
// "this page stopped listening". So the tick always reschedules, and a stale page says how stale it is.
(async function poll() {
  try {
    await refresh();
  } catch (e) {
    notice(e.message);           // refresh() already catches; this is so the finally always runs
  } finally {
    const since = lastOk ? Date.now() / 1000 - lastOk : null;
    if (!paused && since !== null && since > 5) {
      // A slot, so going stale cannot re-flow the page: short visible text, the full sentence and the last
      // error in the title.
      slot("fresh", `${Math.round(since)}s stale`,
           `This tab has not had an answer for ${since.toFixed(0)} s - the numbers on the page are frozen at the `
           + `last one that arrived` + (lastErr ? ` (last error: ${lastErr})` : "."));
      if (!lastErr) text("updated", "Stale · retrying");   // a failed fetch already said "Disconnected"
    }
    setTimeout(poll, 1000);
  }
})();

// Coming back to the tab should show now, not whatever the throttled timer gets around to next.
document.addEventListener("visibilitychange", () => { if (!document.hidden && !paused) refresh().catch(() => {}); });
