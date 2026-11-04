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

// A PCIe chain, as a ROW rather than a card of waveforms. Both things it can report are static-or-eventual:
// the link sits at its maximum unless it renegotiates (one number, changing once), and the AER counters moved 0
// times in 90 idle samples and 0 times through an eight-stream burst, but ~12 times around two serve restarts.
// Curves of either are flat and read as a broken panel; totals plus an event with a timestamp say all a curve
// would. Byte traffic is not collectable here (the counters answer N/A, and the PMU needs root or /dev/kfd).
// "12 min" / "1.4 h": the span since a chain last moved is read as a duration, never as a raw second count.
const ago = (s2) => (s2 >= 5400 ? `${(s2 / 3600).toFixed(1)} h` : `${Math.round(s2 / 60)} min`);

function pcieRow(c, phist = [], busyNow = false) {
  const short = (b) => String(b || "").replace(/^0000:/, "");
  const devs = c.devices || [];
  const sum = (k) => devs.reduce((a, x) => a + (x[k] || 0), 0);
  const named = {};
  devs.forEach((x) => Object.entries(x.notable || {}).forEach(([k, v]) => { named[k] = (named[k] || 0) + v; }));
  // Named counters are grouped by kind and the group is labelled once, because the kernel's own name is
  // misleading on its own (`NonFatalErr` is a correctable counter) and per-name suffixes would be noise.
  const cn = Object.entries(named).filter(([k]) => !/#(nf|ft)$/.test(k)).map(([k, v]) => `${k} ${v}`);
  const un = Object.entries(named).filter(([k]) => /#(nf|ft)$/.test(k))
    .map(([k, v]) => `${aerKind(k).name} ${v} ${aerKind(k).kind}`);
  const at = (s2) => (s2.chains || []).find((x) => x.bus === c.bus) || null;
  const last = phist.length ? at(phist[phist.length - 1]) : null;
  const win = last && isnum(last.cor_d) ? last.cor_d + (last.nonfatal_d || 0) + (last.fatal_d || 0) : null;
  const cor = sum("cor"), unc = sum("nonfatal") + sum("fatal");
  const maxw = Math.max(...devs.map((x) => x.max_width || 0), 0);
  const d = document.createElement("div");
  d.className = "pcie-row";
  // "quiet" is a measured state, not a guess: seconds since this watcher last saw a counter move on this chain
  // (`quiet_s`, carried on the sidecar's history entries). This box's c6 chain sat at exactly 42 correctables
  // for 42 minutes with the link at 32 GT/s after a load window ended at 09:00:03 -- a number that is not
  // moving must not keep the colour of a number that is, or the colour stops being read. 300 s because during a
  // real DMA window this chain errors several times a MINUTE (measured 11.5-45.7 per minute at weight load):
  // five minutes of nothing is not a slow trickle, it is the absence of one. An unknown quiet clock (an older
  // sidecar) counts as loud -- the dimming is allowed to turn itself off, never to invent calm.
  const quiet = isnum(last?.quiet_s) ? last.quiet_s : null;
  // The RATE is the criterion, not the time since the last event. That was measured into: the first version of
  // this rule dimmed a chain only after 300 s without a counter move, and under the traffic this box actually
  // carries (a ~100k-token prompt every ~20 s, one BadTLP every ~3 min) quiet_s never got past 300 -- the row
  // stayed coloured essentially forever, which is the same failure as colouring a number that never moves. So:
  //   digits + colour  -- this poll saw movement (a one-poll flash that something just happened), the rate over
  //                       the trailing span is above 1/min (a weight-load window measured 11.3/min), or the rate
  //                       is unknown (older sidecar: never invent calm);
  //   info glyph       -- correctables exist but the span rate is at most 1/min. Numbers move to the title.
  // `quiet_s` still belongs to the tooltip: it says how long the calm has been, which a rate cannot.
  const rate = isnum(last?.rate_per_min) ? last.rate_per_min : null;
  const span = Math.max(1, Math.round((last?.window_s || 300) / 60));
  const justNow = (win ?? 0) > 0 || !!last?.reset;
  const liveErrs = justNow || !last || rate === null || rate > 1;
  const sparse = !!cor && !unc && !justNow && rate !== null && rate <= 1;
  // Colour means "look here": a link below its maximum, or an uncorrectable error, is the danger; correctable
  // errors are noted but not alarming on their own (they are correctable, and this box has them by the dozen).
  // Two idle states are measured on this box, and neither should be painted as a fault: with the GPUs
  // runtime-suspended (D3hot, no context open) EVERY chain drops to Gen1 (2026-11-03 03:00, all four at
  // 2.5 GT/s x16); with a serve up and nothing served, three chains stay at 32 GT/s while the c6 chain's
  // switch uplink has been sitting at 16 GT/s since this boot (60 s at 2 s, one distinct state, while
  // card4's identical topology held 32 GT/s at the same moment). So the red is tied to work -- a sagging
  // link with no card working is named in muted type and left for the note to put under load, because the
  // alarm that fires at an idle box is the alarm that stops being read.
  const sag = c.degraded && !c.parked && !busyNow;
  d.dataset.tone = (c.degraded && !c.parked && busyNow) || unc ? "danger" : (cor && liveErrs) ? "warn" : "";
  // Slot first, same as the card grid: the row is now findable by the number written on the chassis. `cardN`
  // (the driver's per-boot name) and the BDF path follow, because the BDF is what the kernel's own AER and
  // GHES records use, and the two have to be readable side by side to act on either.
  // Every row emits the same seven children, in the same order, whether or not this card has a slot or a
  // driver name: the grid has seven tracks, and a missing child shifts every later column into the wrong
  // track -- which reads as "the columns are ragged" and not as "this card is unnamed".
  d.innerHTML = `<b class="p-slot">${c.slot ? `slot ${esc(c.slot)}` : esc(c.card || short(c.bus))}</b>`
    // Empty, never a placeholder word: an older build behind this page has no slot/pos in its payload, and a
    // column that printed the literal word "driver" would read as data. The track stays, the cell stays blank.
    // The driver name goes here only when the headline is a slot; with no slot in the payload the headline IS
    // the driver name, and repeating it in the next column is the duplicate-column look people read as a bug.
    + `<span class="small p-drv">${c.slot && c.card ? `${esc(c.card)}${isnum(c.pos) ? ` \u00b7 #${c.pos}` : ""}` : ""}</span>`
    + `<span class="chain p-chain">${devs.map((x) => short(x.bdf)).join(" → ")}</span>`
    + `<span class="link p-link" title="negotiated speed and width. A link below its maximum while the card
      is running is the thing to notice; a parked card (D3hot) idles its links on purpose">`
      + (isnum(c.speed) ? `${fmt(c.speed, 1)} GT/s ×${c.width}`
                        + (c.degraded && maxw
                           ? ` <span class="q">(${c.parked ? "parked, max ×"
                              : sag ? "idle, max ×" : "max ×"}${maxw})</span>`
                           : "") : "—") + `</span>`
    // The glyph is drawn, not typed: the page's body font ships a Latin subset, so U+24D8 came back from the
    // browser as a missing-glyph box (a "big question mark" at a glance) instead of an i-in-a-circle. An ASCII
    // `i` inside a CSS circle needs no font support at all.
    // The count cell is a BUTTON in every state that has numbers: the detail has to be openable, and a `title`
    // tooltip cannot be selected, cannot be copied, and disappears the moment the pointer leaves. A chain with
    // nothing raised stays a plain dash -- there is nothing there to open.
    + `<span class="cnt p-cnt">${sparse
        ? `<button type="button" class="p-glyph" data-bus="${esc(c.bus)}" title="quiet \u2014 click to keep the numbers open">i</button>`
        : cor || unc
        ? `<button type="button" class="p-info" data-bus="${esc(c.bus)}" title="click to keep this chain's full account open">${cor} corr${unc ? ` \u00b7 ${unc} uncorr` : ""}</button>`
        : `<span class="q" title="no AER counter on this chain has moved since boot">\u2014</span>`}</span>`
    + `<span class="who p-who" title="the non-zero AER counters on this chain: correctable ones are recovered by the
      hardware, the others are not">${sparse ? "" : cn.length ? `<span class="q">corr</span> ${cn.join(" · ")}` : ""}`
      + (sparse ? "" : un.length ? ` <span class="unc">${un.join(" · ")}</span>` : "") + `</span>`
    + `<span class="win p-win" title="this poll's deltas; these counters are quiet unless something happens.
      'quiet' counts how long this page has watched this chain with no counter moving -- it says nothing about
      before this page opened, because counters we found already raised are not ours to date">`
      + (win === null ? "<span class=\"q\">window —</span>"
         : (cor && !liveErrs && quiet !== null)
           ? `<span class="q">+0 · ${rate !== null && rate < 10 ? rate.toFixed(2) : rate}/min${quiet !== null ? ` · quiet ${ago(quiet)}` : ""}</span>`
           : `+${win} this window`) + `</span>`;
  return d;
}

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
    if (isnum(r.shared_decode)) {
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
    meta.textContent = [
      isnum(r.first_token_s) ? `first token ${seconds(r.first_token_s)}`
        + (isnum(t.queue_s) && t.queue_s >= 0.001 ? ` (${seconds(t.queue_s)} of it queue)` : "") : "",
      isnum(t.prefill_s) ? `prefill ${seconds(t.prefill_s)}` : "",
      isnum(t.decode_s) ? `decode ${seconds(t.decode_s)}` : "",
      isnum(total) ? `all in ${seconds(total)}${engineRow && isnum(t.queue_s) && t.queue_s >= 0.001
        ? " (queue included)" : ""}` : "",
      isnum(u.prompt_tokens_details?.cached_tokens) && u.prompt_tokens_details.cached_tokens > 0
        ? `${fmt(u.prompt_tokens_details.cached_tokens, 0)} cached` : "",
      r.http_status ? `HTTP ${r.http_status}` : "",
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
  if (detail.source === "engine") {
    // The merged monitor reads the engine's own per-request stats: tokens and timings, never the text.
    // Saying so beats three panes that each invent a different apology.
    text("content", detail.input_note || "The engine reports timings and token counts, not the text.");
    text("content-note", "From vLLM's FinishedRequestStats. To capture bodies, run the separate sidecar "
      + "(serve/webui.sh with WEBUI_MODE=proxy): it sits between the client and the serve.");
    return;
  }
  let value = detail[pane];
  if (pane === "output" && !value && detail.response) value = "";
  if (value === undefined || value === null || value === "") {
    value = pane === "response" ? (detail.response || "The response body was not kept (an empty or non-JSON answer).")
      : pane === "reasoning" ? "No separate reasoning content in this response."
      : pane === "output" ? "No text kept for this request." : detail.input || "No input.";
  }
  if (pane === "input" || pane === "response") { try { value = JSON.stringify(JSON.parse(value), null, 2); } catch (_) {} }
  text("content", value);
  const note = pane === "input" ? "The request body as it arrived."
    : pane === "output" ? "What the model said. Streaming text, concatenated."
    : pane === "reasoning" ? "reasoning_content, when the model separates it."
    : "The response body as it left the server.";
  text("content-note", note + (detail.output_truncated && pane === "output"
    ? " Our copy is truncated at 256 KiB; the API response was not shortened." : ""));
}

async function showDetail() {
  if (!selected) return;
  const id = selected;
  const value = await api(`/api/requests?id=${encodeURIComponent(id)}`);
  if (selected !== id) return;                       // you clicked another row while this was in flight
  detail = value;
  $("selection").hidden = false; $("no-selection").hidden = true;
  text("detail-title", `${value.id} · ${value.path}`);
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
    gen: "Total decode output over total DECODE time: generated tokens divided by the stream-seconds the engine's "
       + "decode families report. Queue, prefill phases, the wait before the first token, idle seconds and steps "
       + "that produced no token are all OUT of the denominator -- that is the point of the number, and it is the "
       + "same quantity as a request row's speed (1 / the engine's mean time per output token), which is why the "
       + "card and the rows under it read alike: 18,118 tokens over 119.1 stream-seconds = 152 tok/s on a live "
       + "burst whose rows ran 160-200 tok/s. The unit is per STREAM-second: vLLM observes one inter_token_latency "
       + "per REQUEST x STEP (v1/metrics/stats.py:551), so a step two streams share is counted twice -- that is "
       + "correct here (each stream got a token out of it) but it is why the label says stream. The two decode "
       + "families agree to 0.08% (24,236.14 s over 430,954 observations against 24,216.27 s over 1,812 request "
       + "observations) because they are one sum sampled per step and per request, not two measurements. "
       + "`N summed ×M` on the note line is the throughput reading of the same traffic: N is generated tokens per "
       + "WALL second (prefill and idle back in the denominator -- what a minute of this traffic produces), M is "
       + "streams decoding per wall second (a COUNT, not a share: 0.15 on a serialized burst, 2.60 with traffic "
       + "on, no 100% ceiling), and the big number times M is exactly N: 152 x 0.15 = 22. The waveform card next "
       + "door stays on the summed-wall basis on purpose -- this denominator does not exist in a window that was "
       + "all prefill or idle, and a line full of gaps reads as an idle box. What this denominator cannot remove: "
       + "under chunked prefill a chunk is computed INSIDE a decode-phase step, so the step interval carries that "
       + "work too. A bare single-stream step measures 13.0 ms in the bench; this launch's live itl is 15.5 ms, "
       + "the +19% that no counter separates -- which is also why the span's per-stream figure (152) sits below "
       + "the mean of its own request rows (167).",
    prefill: "READ tokens per second of REQUEST-prefill time -- the engine's own scheduled -> first-token span, "
           + "summed over streams, so stream-seconds and a per-stream speed. It is NOT the engine's chew rate: "
           + "under chunked prefill a request's prefill span contains the steps in which the engine was decoding "
           + "other requests, and if two prefills run at once this divides by both -- the note line then prints "
           + "`×N prefilling` (today 1.04: long prefills take the 8,192-token step one at a time, so they "
           + "serialize). READ, deliberately: vLLM credits cached and externally transferred prefix tokens to "
           + "prompt_tokens_total, so credited tokens would count work the engine never did and the rate would be "
           + "high by the hit rate; the numerator subtracts prompt_tokens_cached_total (the exact skipped count, "
           + "local + external) when the build publishes it and prefix_cache_hits_total (local only, counted at "
           + "admission) when it does not, and `N% cache` says how much came out. Today the cache is off on this "
           + "recipe (PREFIX_CACHE=0), so read and credited are the same number, and "
           + "`prefill_stream_tok_s_credited` keeps the difference visible if that changes -- the same subtraction "
           + "Strata's run average makes (`prompt_tokens - reused`, serve/web/app.js). The wall figure is the duty "
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
      parts.push(`${name} +${dnum(v)}`);
    }
    return parts.join(" \u00b7 ") || "\u2014";
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
  slot("spans-gentot", spanSigned("generation_tokens"), TT.spans);
  // "none credited", not "+0.0": in this window the counter moved nothing, which for a count is a fact about the
  // window (the engine was prefilling), not a measurement of zero tokens a second.
  slot("poll-gentot", isnum(rate.generation_tokens) && rate.generation_tokens > 0
       ? `poll +${dnum(rate.generation_tokens)} tok` : `poll \u00b7 none credited`, TT.poll);

  // Decode, on the basis the user of this page actually asks for: total decode output over total DECODE time --
  // generated tokens divided by the stream-seconds the engine's decode families report (inter_token_latency and
  // request_decode_time are one sum sampled twice: per stream-step and per request; a step two streams share
  // accrues twice, which is why the unit is per STREAM-second and the label says so). Live on the burst this
  // launch was tested with: 18,118 tokens over 119.1 stream-seconds = 152 tok/s, against the rows' 160-200 tok/s
  // for the same traffic -- A and the per-request rows are the same quantity, so the card and the rows finally
  // read alike. The wall-second sum (what a minute of this traffic produces, 22 tok/s) is NOT dropped: it is the
  // note line, and the fan-out sits between the two as an explicit multiplication -- 152 x 0.15 = 22 -- so a
  // reader who wants the throughput number sees it, and sees why it is smaller. (The card has been wrong in both
  // directions here: the summed figure once carried a per-stream number under the words "summed, decode time",
  // and before that a per-stream mean divided by the running gauge. The unit and the fan-out are what make the
  // pair checkable.)
  const perS = A ? A.decode_stream_tok_s : rate.decode_stream_tok_s;
  const wallV = A ? A.generation_tok_s : rate.generation_tok_s;
  const streamsV = A ? A.decode_streams : rate.decode_streams;
  big("rate-gen", perS, "tok/s");
  // The summed figure and the fan-out have different denominators, so they can be missing separately. Wall
  // seconds always exist, so a wall-second rate of 0 is a MEASUREMENT and stays on the line; decode stream-seconds
  // do not exist until a decode step has been measured, and then the multiplier is what has no denominator -- it
  // goes out and the sentence says which one it was, instead of the whole slot going to a dash (the harness fails
  // a printed dash where a measured zero belongs).
  slot("note-gen", isnum(wallV)
       ? `${dnum(wallV, 1)} summed` + (isnum(streamsV)
           ? ` \u00d7${streamsV < 10 ? streamsV.toFixed(2) : dnum(streamsV, 1)}` : "")
       : "\u2014",
       TT.gen + (isnum(wallV) && !isnum(streamsV)
         ? " No multiplier: this window contains no decode step, so there are no decode stream-seconds to divide "
         + "by. The summed figure has a different denominator -- wall seconds -- and it is a measurement of 0, so "
         + "it is printed as one." : ""));
  slot("spans-gen", spanList("decode_stream_tok_s"), TT.spans);
  // A poll with no decode step gets the sentence ALONE: `poll · no decode step · 0.0 req/m` measured 33 characters
  // against the tightest 4-up slot's 24 (the harness flags it), and an arrival rate is not the missing denominator.
  slot("poll-gen", isnum(rate.decode_stream_tok_s)
       ? `poll ${dnum(rate.decode_stream_tok_s)}`
         + (A && isnum(A.requests_per_min) ? ` \u00b7 ${dnum(A.requests_per_min, 1)} req/m` : "")
       : `poll \u00b7 no decode step`, TT.poll);

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
  // `N% cache` only appears when the engine actually served from cache: with PREFIX_CACHE=0 (this recipe) it would
  // be a permanent `0% cache` saying nothing, and every character in a slot is budgeted. Asked on 2026-11-04
  // whether the caches were in effect, and the answer was not on this page: a missing cache clause reads the same:
  // feature is OFF, on-and-missed, or not measured at all (asked 2026-11-04: "are the caches in effect?"). The
  // engine states the state (`config
  // .enable_prefix_caching`, which is what its own boot line prints), so OFF is now said in words -- and only OFF:
  // a build that publishes neither that config nor the counters keeps the silence, which is the one case where
  // silence is honest. It shares the line with the fan-out clause rather than both being printed: `wall 4,316
  // cache off` fits the 24-character budget, `wall 4,316 cache off ×2.0 prefilling` does not.
  const pcState = String(((s.config || {}).enable_prefix_caching ?? "")).toLowerCase();
  const pcOff = pcState === "false" || pcState === "0";
  const fan = A && isnum(A.prefill_streams) && A.prefill_streams > 1.05;
  slot("note-prefill", A && isnum(A.prefix_hit_pct) && A.prefix_hit_pct > 0
       ? `wall ${dnum(A.prompt_tok_s || 0)} \u00b7 ${dnum(A.prefix_hit_pct, 0)}% cache`
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
        + ` the PCIe chains and slots, and the record of what ended the last boot come from sysfs and the kernel`
        + ` log — none of them need the serve, so they keep reading while it is down.</div>`;
    }
  }
  const j = s.journal || null;
  qualText("count", records.length, j ? (j.restored ? `${j.restored} carried over` : "this launch only") : "");
  const jl = $("journal-line");
  if (jl) {
    jl.hidden = !j;
    jl.textContent = j
      ? (j.ok
        ? `request journal ${j.path} · ${j.restored} row${j.restored === 1 ? "" : "s"} carried over from `
          + `earlier launches${j.readonly
            ? " · <b>attached read-only</b>: these rows belong to the launch that owns the file"
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
                                               empty = "no samples", title = "" } = {}) => {
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
    slot(n2, `peak ${dnum(Math.max(...used), d)} \u00b7 ${used.length}/${values.length}`,
         title + ` (${used.length} of ${values.length} points kept, over ${histSpan}.`
         + (lo !== null && top > lo ? ` The waveform is drawn on a ${dnum(lo)}..${dnum(top)} axis.` : "") + ")");
  };
  const TREND_T = {
    perstep: "Generated tokens per engine record, summed over streams -- never a per-stream mean. It is the "
           + "fan-out times the acceptance length: live over 15 min, 2.85 streams producing per record x ~2.5 "
           + "tokens per stream-step = 7.05 tokens a record, against the ~2.09 a stream turns out of its own step "
           + "(the ratio between the two is the payload's `streams_per_record`, ITL observations over records). At "
           + "0.32 a whole chunked prefill lands in one record, so the line dips exactly where the engine was "
           + "prefilling; a gap is an interval with no record in it.",
    gen: "All streams summed over WALL seconds -- the summed figure, the throughput reading of the box. The decode "
       + "card's big number is this traffic's other reading (output per stream-second) and its note line carries "
       + "the bridge: that number × this series' fan-out = this line. Flat stretches are the engine prefilling or "
       + "idle rather than decoding, which is why the peak and the average of this line differ by an order of "
       + "magnitude on a prefill-heavy launch. This series is drawn on the wall basis because the per-stream "
       + "denominator does not exist in a window that was all prefill, and a gap-per-second line reads as an idle "
       + "box.",
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
        { lo: 0, hi: "peak", empty: "nothing generated", title: TREND_T.gen });
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
    + (s.smi ? " + rocm-smi" : "") + (quiet ? ` · ${quiet} refusing sysfs reads` : "")
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
            const link = isnum(m.speed) ? `${fmt(m.speed, 1)} GT/s \u00d7${m.width}` : "link unreadable";
            const short_ = isnum(m.max_speed) && isnum(m.speed) && m.speed < m.max_speed
              ? ` <span class="q">of ${fmt(m.max_speed, 0)}</span>` : "";
            const errs = (m.corr || m.uncorr)
              ? ` <span class="q">${m.corr} corr${m.uncorr ? ` / ${m.uncorr} uncorr` : ""}</span>` : "";
            return `<b>#${m.pos} slot ${esc(m.slot != null ? m.slot : "?")}</b>`
              + ` <span class="q">${esc(m.card || "unnamed")}</span> GPU <code>${esc(m.bus)}</code>`
              + ` ${link}${short_}${errs}`;
          }).join(" \u00b7 ")
        + ` <span class="q">the slot is registered at the card's switch upstream port`
        + ` (${esc((smap[0].slot_at || "n/a").replace(/^0000:/, ""))} for slot ${esc(smap[0].slot != null ? smap[0].slot : "?")}),`
        + ` not at the GPU endpoint</span>`
      : "";
    sm.dataset.tone = smap.some((m) => isnum(m.max_speed) && isnum(m.speed) && m.speed < m.max_speed) ? "warn" : "";
  }
  // Cards and chains are ordered by the server (chassis order), not by enumeration order -- one order across
  // Cards, PCIe rows and the About page, so a card number never means a different position in two places.
  $("gpus").replaceChildren(...gpus.map((g) => gpuCard(g, s.gpus_history || [])));
  // The PCIe chains get their own panel because they answer a different question than the cards: the cards say
  // what the GPU did, this says whether the wire under it was clean. The wording has to stay honest about scope:
  // this is not bandwidth, it is negotiated link state and the kernel's error counters.
  const chains = s.pcie || [];
  const phist = s.pcie_history || [];
  const ev = s.pcie_events || [];
  const shortb = (b) => String(b || "").replace(/^0000:/, "");
  const tot = chains.reduce((a, c2) => a + (c2.devices || []).reduce((b, x) =>
    b + (x.cor || 0) + (x.nonfatal || 0) + (x.fatal || 0), 0), 0);
  // One spark for the whole box, because the only interesting shape here is "an event happened". It counts AER
  // events per poll, so a spike is an event with a time that can be lined up against a stall; a flat line at
  // its zero axis is the honest picture of a healthy link, and it sits on a known axis so "flat" reads as
  // "zero", not as "no data".
  const perPoll = phist.map((h) => (h.chains || []).reduce((a, x) => a + (isnum(x.cor_d) ? x.cor_d : 0)
    + (isnum(x.nonfatal_d) ? x.nonfatal_d : 0) + (isnum(x.fatal_d) ? x.fatal_d : 0), 0));
  // Is the box actually moving anything? That is what makes a sagging link worth colouring: measured on this
  // box, 16 GT/s on the c6 chain at 3% busy (power management) and the same port at 32 GT/s under load.
  const busyNow = ((status.live || {}).running || 0) > 0
    || (status.gpus || []).some((g) => isnum(g.busy_pct) && g.busy_pct >= 25);

  const pse = $("pcie-spark");
  pse.replaceChildren();
  pse.setAttribute("aria-label", "AER events per second, whole box");
  spark(pse, perPoll, { baseLo: 0, baseHi: Math.max(...perPoll, 1) });
  pse.dataset.tone = perPoll.some((v) => v > 0) ? "warn" : "";
  const when = (t) => new Date((t || 0) * 1000).toLocaleTimeString([], { hour12: false });
  text("pcie-note", chains.length
    ? `${chains.length} link chain${chains.length > 1 ? "s" : ""} · sysfs, no root and no /dev/kfd · `
      + `byte counters answer N/A on these cards, so this is link state and AER errors`
      + ` · ${tot} error${tot === 1 ? "" : "s"} since boot`
      + (s.pcie_events === undefined
        ? " \u00b7 this serve's build does not report events \u2014 restart it for the timestamps"
        : ev.length
          ? ` \u00b7 last: ${ev.slice(0, 3).map((e) => { const { name, kind } = aerKind(e.kind);
              return `${when(e.at)} ${shortb(e.bdf)} ${name}${kind === "correctable" ? "" : ` ${kind}`} +${e.n}`;
            }).join(" \u00b7 ")}`
          : " \u00b7 nothing moved while this page has been watching")
      + (chains.some((c2) => c2.degraded && !c2.parked && busyNow) ? " · a link is below its maximum while work is running" : "")
      + (chains.length && chains.every((c2) => c2.parked) ? " · cards parked: the links idle at Gen1 while nothing runs"
        : chains.some((c2) => c2.degraded)
          ? " \u00b7 a link is below its maximum with nothing running: parked chains always read that way"
            + " \u00b7 if it is still short while cards work, that is a link fault" : "")
      + (phist.some((h) => (h.chains || []).some((x) => x.reset))
        ? " · counters were zeroed during the window" : "")
    : "no PCIe sysfs readable here");
  // A header row on the same grid: with seven columns of mono text, "which number is which" should not be
  // something the reader has to infer from the panel title.
  const phead = document.createElement("div");
  phead.className = "pcie-row pcie-head";
  phead.innerHTML = `<b class="p-slot">slot</b><span class="small p-drv">driver</span>`
    + `<span class="chain p-chain">endpoint \u2192 root port</span><span class="link p-link">link</span>`
    + `<span class="cnt p-cnt">errors since boot</span><span class="who p-who">which counters</span>`
    + `<span class="win p-win">this window</span>`;
  $("pcie").replaceChildren(phead, ...chains.map((c2) => pcieRow(c2, phist, busyNow)));
  paintPcieDetail(s);
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
      : "no counter moved while this page has been open"
      + " \u00b7 anything before the engine started belongs to aer-watch and journal-pcie.py, not to this page");
  put("watched", `${phist.length} sidecar samples \u00b7 this page's clock starts after the weights are loaded`);
  return L;
}

// A pinned block of real text, OUTSIDE the row container: #pcie is rebuilt every second, so anything inside it
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

// The rows are rebuilt every tick, so the handler lives on the container; a handler attached to a row would
// vanish with it. Clicking the same chain again unpins it.
$("pcie").onclick = (e) => {
  const b = e && e.target && e.target.closest ? e.target.closest("[data-bus]") : null;
  if (!b || !b.dataset || !b.dataset.bus) return;
  detailBus = detailBus === b.dataset.bus ? null : b.dataset.bus;
  paintPcieDetail(status);
};
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && detailBus) { detailBus = null; paintPcieDetail(status); }
});

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
