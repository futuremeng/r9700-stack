// serve/web/monitor.js -- the Monitor. Two fetches per tick: /api/status (the engine's counters, read
// off vLLM's /metrics) and /api/requests (what came through this page, when it is in the path).
"use strict";
let records = [], status = null, selected = null, detail = null, pane = "input", stampOf = "";
let detailSig = "", follow = true;                    // follow: show the newest row nobody clicked on yet
let paused = false, actionError = "";

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
function pcieRow(c, phist = []) {
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
  // Colour means "look here": a link below its maximum or an uncorrectable error is the danger; correctable
  // errors are noted but not alarming on their own (they are correctable, and this box has them by the dozen).
  d.dataset.tone = c.degraded || unc ? "danger" : cor ? "warn" : "";
  d.innerHTML = `<b>${c.card || short(c.bus)}</b>`
    + `<span class="chain">${devs.map((x) => short(x.bdf)).join(" → ")}</span>`
    + `<span class="link" title="negotiated speed and width; a link below its maximum is the thing to notice">`
      + (isnum(c.speed) ? `${fmt(c.speed, 1)} GT/s ×${c.width}`
                        + (c.degraded && maxw ? ` <span class="q">(max ×${maxw})</span>` : "") : "—") + `</span>`
    + `<span class="cnt">${cor || unc ? `${cor} corr · ${unc} uncorr` : "no errors"} <span class="q">since boot</span></span>`
    + `<span class="who" title="the non-zero AER counters on this chain: correctable ones are recovered by the
      hardware, the others are not">${cn.length ? `<span class="q">corr</span> ${cn.join(" · ")}` : ""}`
      + (un.length ? ` <span class="unc">${un.join(" · ")}</span>` : "") + `</span>`
    + `<span class="win" title="this poll's deltas; these counters are quiet unless something happens">`
      + (win === null ? "<span class=\"q\">window —</span>" : `+${win} this window`) + `</span>`;
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
function visibleRows() {
  const q = $("search").value.toLowerCase(), filter = $("filter").value;
  return records.filter((r) => `${rid(r)} ${r.path} ${r.model}`.toLowerCase().includes(q)
    && (filter === "all" || (filter === "active" ? active(r) : r.state === filter)));
}

function list() {
  const visible = visibleRows();
  const next = JSON.stringify([visible.map((r) => [rid(r), r.state, r.wallclock_s]), selected]);
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
    const tag = document.createElement("span"); tag.className = `tag tag--${r.state}`; tag.textContent = r.state;
    row.append(time, tag);
    const ep = document.createElement("div"); ep.className = "endpoint";
    ep.textContent = r.path ? `POST ${r.path}` : "reported by the engine (no endpoint in its stats)";
    const meta = document.createElement("div"); meta.className = "small";
    const u = r.usage || {}, t = r.timings || {};
    meta.textContent = `${rid(r)} · ${seconds(r.wallclock_s)} `
      + `${r.response_format === "engine" ? "engine record" : r.stream ? "stream" : "JSON"}`
      + `${r.http_status ? ` · HTTP ${r.http_status}` : ""}`
      // What the row is worth without opening it: how big the request was and where its time went. In merged
      // mode both come straight from the engine's per-request stats; in observe mode there are no rows at all.
      + (isnum(u.prompt_tokens) || isnum(u.completion_tokens)
        ? ` · ${fmt(u.prompt_tokens ?? 0, 0)} -> ${fmt(u.completion_tokens ?? 0, 0)} tok` : "")
      + (isnum(t.prefill_s) ? ` · prefill ${seconds(t.prefill_s)}`
        + (isnum(t.prefill_tok_s) ? ` (${fmt(t.prefill_tok_s, 0)} tok/s)` : "") : "")
      + (isnum(t.queue_s) && t.queue_s >= 0.001 ? ` · queue ${seconds(t.queue_s)}` : "")
      + (t.preemptions ? ` · preempted ${t.preemptions}x` : "");
    b.append(row, ep, meta);
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
  text("detail-wall", seconds(value.wallclock_s ?? (isnum(value.started_at) ? Date.now() / 1000 - value.started_at : null)));
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
  // Big numbers, and only from what moves per engine step. stream_tok_s_live is aggregate over running:
  // the rate of the stream you are watching, available while it runs -- 1/itl (stream_tok_s) is not, because
  // vLLM observes that histogram when the request FINISHES. The two agree once a request ends.
  const big = (id, v, unit) =>
    { $(id).innerHTML = `${isnum(v) ? fmt(v, 1) : "—"}<span class="unit">${unit}</span>`; };
  big("rate-stream", rate.stream_tok_s_live, "tok/s");
  // Tokens per step reconciles the two clocks: the step clock and the per-accepted-token clock. The window
  // usually holds no histogram observation at all (vLLM observes one when a request FINISHES), so the
  // since-launch ratio fills in and gets labelled, rather than the clause vanishing whenever nothing ended.
  const sn = s.since || {};
  const tpsW = rate.tokens_per_step;
  const tpsS = isnum(sn.itl_ms) && isnum(sn.tpot_ms) ? sn.itl_ms / sn.tpot_ms : null;
  text("rate-stream-note", isnum(rate.stream_tok_s_live)
    ? `aggregate ÷ ${Math.round(live.running || 0)} running · this window`
      + (isnum(rate.itl_ms) ? ` · step ${fmt(rate.itl_ms, 1)} ms` : "")
      + (isnum(tpsW) ? ` × ${fmt(tpsW, 2)} tokens a step`
        : isnum(tpsS) ? ` × ${fmt(tpsS, 2)} tokens a step (since launch)` : "")
    : inflight ? "nothing generated in this window yet" : "nothing running");
  big("rate-gen", rate.generation_tok_s, "tok/s");
  text("rate-gen-note", isnum(rate.generation_tok_s)
    ? `${fmt(rate.generation_tok_s * (rate.window_s || 1), 0)} tokens in ${fmt(rate.window_s || 0, 1)} s`
      + (isnum(rate.requests_per_min) ? ` · ${fmt(rate.requests_per_min, 1)} req/min` : "")
    : "generation_tokens_total, credited per step");
  // Prefill, both numbers, always labelled. prompt_tok_s is the exact window rate but it lands when the
  // request ends (measured: a 51,859-token prefill held the counter flat for 10 s, then one jump), so while
  // a long prompt is running the number shown is the KV gauge's growth in tokens a second -- an estimate,
  // measured to within ~6% of the truth on that prefill, and negative when finishing requests free blocks.
  const kvr = isnum(rate.kv_tok_s) ? Math.max(0, rate.kv_tok_s) : null;
  const exact = isnum(rate.prompt_tok_s) && rate.prompt_tok_s > 0 ? rate.prompt_tok_s : null;
  big("rate-prefill", exact ?? kvr, "tok/s");
  text("rate-prefill-note", exact !== null
    ? `exact, from prompt_tokens_total (credited at request end)`
      + (kvr !== null ? ` · KV growth now ${fmt(kvr, 0)}` : "")
    : kvr !== null ? "≈ estimated from KV growth · the counter credits at request end"
      : "no prefill in this window");
  big("rate-step", isnum(rate.itl_ms) ? rate.itl_ms : (s.since || {}).itl_ms, "ms");
  text("rate-ttft", isnum((rate.ttft_ms ?? (s.since || {}).ttft_ms))
    ? `first token ${ms(rate.ttft_ms ?? (s.since || {}).ttft_ms)}`
      + (isnum(rate.ttft_ms) || !isnum((s.since || {}).ttft_ms) ? "" : " (since launch)")
    : "no first-token observation in this window");
  // Say how old the numbers are instead of letting the reader guess: the engine poll sets the window, the
  // snapshot age says how long this payload has been sitting, and the page refetches on the same beat.
  const age = isnum(s.at) ? Math.max(0, Date.now() / 1000 - s.at) : null;
  text("fresh", `${isnum(s.poll_s) ? s.poll_s.toFixed(1) : "?"} s poll · snapshot `
    + (age === null ? "age unknown" : `${age.toFixed(1)} s ago`));

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
  const j = s.journal || null;
  qualText("count", records.length, j ? (j.restored ? `${j.restored} carried over` : "this launch only") : "");
  const jl = $("journal-line");
  if (jl) {
    jl.hidden = !j;
    jl.textContent = j
      ? (j.ok
        ? `request journal ${j.path} · ${j.restored} row${j.restored === 1 ? "" : "s"} carried over from `
          + `earlier launches · this launch ${j.boot}`
        : `request journal FAILED (${j.error}) — the list is memory only and dies with the serve`)
      : "";
    jl.dataset.tone = j && !j.ok ? "danger" : "";
  }
  text("active", inflight ? `${Math.round(inflight)} running / waiting at the engine`
    : s.mode === "proxy" ? "Nothing in flight"
      : s.mode === "merged" ? "Finished requests only: the engine reports a request when it ends"
        : "Nothing routed here");
  text("wall", last ? seconds(last.wallclock_s) : "—");
  text("wall-note", s.mode === "merged" ? "From the engine: queue, prefill and decode together"
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
  text("kv-tokens", isnum(live.kv_tokens) ? `KV pool ${fmt(live.kv_tokens)} tokens` : "pool size not in /metrics");
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
    + (live.swapped ? ` · ${Math.round(live.swapped)} swapped` : ""));
  $("gen-toks").textContent = isnum(rate.generation_tok_s) ? fmt(rate.generation_tok_s, 1) : "—";
  text("prompt-toks", `${isnum(rate.prompt_tok_s) ? fmt(rate.prompt_tok_s, 0) : "—"} prompt tok/s`
    // vLLM credits prompt_tokens_total when a request FINISHES: a long prefill reads 0 for its whole duration
    // and then lands in one window (measured: 180,066 tokens showed as 89,580 tok/s in a single 2 s poll).
    // The live signal during a prefill is the KV pool bar, not this line.
    + " (credited at request end)"
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
  const trend = (sparkId, noteId, values, unit, d, { keep = (v) => v !== 0, label = " while running",
                                                    lo = null, hi = null, empty = "no samples" } = {}) => {
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
    const spanS = (values.length * (s.poll_s || 0)).toFixed(0);
    // The old whole message was that a series had no points, which left the reader to guess whether the box
    // was empty because
    // nothing is running, because the counter is dead, or because the page is broken. Say which, and name the
    // axis the flat line is drawn against: a per-stream rate with no streams has no denominator, and that is
    // a different statement from a rate that was measured at zero.
    text(noteId, used.length < 2
      ? `${empty} · ${spanS} s back`
        + (lo === null || !(top > lo) ? "" : ` · axis ${fmt(lo, 0)}..${fmt(top, 0)} ${unit}`)
      : `last ${fmt(xs[xs.length - 1], d)} · peak ${fmt(Math.max(...used), d)} · `
        + `avg ${fmt(used.reduce((x, y) => x + y, 0) / used.length, d)} ${unit}${label}`
        + ` · ${used.length}/${values.length} pts over ${spanS} s`);
  };
  const hist = s.history || [];
  trend("stream-spark", "stream-spark-note", hist.map((h) => h.stream_tok_s), "tok/s", 1,
        { lo: 0, hi: "peak",
          empty: "no stream running, so a per-stream rate has no denominator" });
  trend("gen-spark", "gen-spark-note", hist.map((h) => h.gen_tok_s), "tok/s", 1,
        { lo: 0, hi: "peak", empty: "nothing was being generated" });
  // The prefill waveform is the live estimate (KV growth), not the counter rate: over a long prefill the
  // counter series is a flat line with one spike at the end, which is a fact about vLLM, not about the work.
  // It stays autoscaled: it goes negative when finishing requests free blocks, and a 0 baseline would clip
  // the half that says "the pool is draining".
  trend("prefill-spark", "prefill-spark-note", hist.map((h) => h.kv_tok_s), "tok/s", 0,
        { keep: (v) => v > 0, label: " while the pool grew", empty: "the KV pool did not grow" });
  // The fourth waveform is how many streams the engine is actually running: at NSEQ=32 the aggregate rate is
  // a function of this line, and the knee (PROGRESS: the spread past 32) is visible in its shape.
  trend("running-spark", "running-spark-note", hist.map((h) => h.running), "streams", 0,
        { lo: 0, hi: "peak", empty: "0 streams running" });
  // The KV pool's own waveform lives on its card, not here: its axis is 0..100 (a quarter-full pool is a
  // quarter of the box), which would flatten every other series if it shared the trends row's scale.
  spark($("kv-spark"), hist.map((h) => h.kv_used_pct), { baseLo: 0, baseHi: 100 });
  $("kv-spark").dataset.tone = kv > 95 ? "danger" : kv > 85 ? "warn" : "ok";
  text("kv-spark-note", isnum(live.kv_tokens)
    ? `${fmt(live.kv_tokens, 0)} tokens in the pool · ${(hist.length * (s.poll_s || 0)).toFixed(0)} s back`
      + (kv > 85 ? ` · ${kv > 95 ? "at the limit" : "nearly full"}` : "")
    : "pool size not in /metrics");

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
      + (chains.some((c2) => c2.degraded) ? " · a link is below its maximum" : "")
      + (phist.some((h) => (h.chains || []).some((x) => x.reset))
        ? " · counters were zeroed during the window" : "")
    : "no PCIe sysfs readable here");
  $("pcie").replaceChildren(...chains.map((c2) => pcieRow(c2, phist)));
}

// -- one tick -------------------------------------------------------------------------------------
async function refresh() {
  if (paused) return;
  try {
    const [s, h] = await Promise.all([api("/api/status"), api("/api/requests")]);
    status = s; records = h.requests;
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

(async function poll() { await refresh(); setTimeout(poll, 1000); })();
