// serve/web/monitor.js -- the Monitor. Two fetches per tick: /api/status (the engine's counters, read
// off vLLM's /metrics) and /api/requests (what came through this page, when it is in the path).
"use strict";
let records = [], status = null, selected = null, detail = null, pane = "input", stampOf = "";
let detailSig = "", follow = true;                    // follow: show the newest row nobody clicked on yet
let paused = false, actionError = "";

const active = (r) => !["completed", "error", "disconnected"].includes(r.state);
const rid = (r) => r.id || `req-${r.seq}`;

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
      + (isnum(t.prefill_s) ? ` · prefill ${seconds(t.prefill_s)}` : "")
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
  text("detail-format", value.response_format || "text");
  $("detail-error").hidden = !value.error;
  text("detail-error", value.error?.message || "");
  content();
}

// -- the engine, the cards, and the timings ---------------------------------------------------------
function renderStatus(s) {
  const live = s.live || {}, rate = s.rate || {};
  const inflight = (live.running || 0) + (live.waiting || 0);
  text("mode", s.mode === "proxy" ? "in the path (recording)"
    : s.mode === "merged" ? "inside the serve (engine records)" : "observing (read-only)");
  text("loaded", !s.reachable ? "Unreachable" : inflight ? "In use" : "Idle");
  $("model-state").className = `value value--sm${inflight ? " busy" : ""}`;
  $("model-state").firstElementChild.className = `dot${inflight ? " dot--busy" : s.reachable ? " dot--live" : " dot--off"}`;
  text("model", s.model_name || (s.models || [])[0]?.id || "no model reported");
  text("upstream", s.upstream || "—");
  const last = records.find((r) => !active(r));
  text("count", String(records.length));
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
  const kv = Math.max(0, Math.min(100, live.kv_used_pct || 0));
  $("kv-bar").dataset.tone = kv > 95 ? "danger" : kv > 85 ? "warn" : "ok";
  $("kv-bar").firstElementChild.style.width = `${kv}%`;
  text("kv-tokens", isnum(live.kv_tokens) ? `KV pool ${fmt(live.kv_tokens)} tokens` : "pool size not in /metrics");
  text("running", String(Math.round(live.running || 0)));
  text("waiting", `${Math.round(live.waiting || 0)} waiting · ${fmt(rate.preemptions ?? 0)} preempted in the window`
    + (live.swapped ? ` · ${Math.round(live.swapped)} swapped` : ""));
  $("gen-toks").textContent = isnum(rate.generation_tok_s) ? fmt(rate.generation_tok_s, 1) : "—";
  text("prompt-toks", `${isnum(rate.prompt_tok_s) ? fmt(rate.prompt_tok_s, 0) : "—"} prompt tok/s`
    + (isnum(rate.requests_per_min) ? ` · ${fmt(rate.requests_per_min, 1)} req/min` : ""));
  // vLLM observes a latency histogram when a request FINISHES, so a window holds no observation at all
  // while a long one is in flight. The cumulative ratio is not a window mean: the page says which it shows.
  const since = s.since || {};
  const pick = (k) => (isnum(rate[k]) ? rate[k] : (isnum(since[k]) ? since[k] : null));
  text("accept", pct(pick("accept_pct"), 1) + (isnum(rate.accept_pct) || !isnum(since.accept_pct) ? ""
    : " (since launch)"));
  text("drafts", isnum(rate.drafts_per_req) ? `${fmt(rate.drafts_per_req, 2)} drafts per request`
    : isnum(since.accept_pct) ? `${fmt(since.requests, 0)} requests finished since the engine started`
      : "no speculation counters");
  const tpi = pick("tokens_per_iter"), ipr = pick("iters_per_req");
  text("iters", isnum(tpi) ? `${fmt(tpi, 2)} tokens per engine step`
    + (isnum(ipr) ? ` · ${fmt(ipr, 2)} steps per request` : "")
    + (isnum(rate.tokens_per_iter) || !isnum(since.tokens_per_iter) ? "" : " (since launch)") : "");

  const hist = s.history || [];
  spark($("gen-spark"), hist.map((h) => h.gen_tok_s));
  spark($("kv-spark"), hist.map((h) => h.kv_used_pct));

  for (const [id, k] of [["t-ttft", "ttft_ms"], ["t-queue", "queue_ms"], ["t-prefill", "prefill_ms"],
                         ["t-decode", "decode_ms"], ["t-step", "itl_ms"], ["t-tpot", "tpot_ms"]])
    text(id, ms(pick(k)) + (isnum(rate[k]) || !isnum(since[k]) ? "" : " (since launch)"));
  text("window", isnum(rate.window_s) ? `${rate.window_s.toFixed(1)} s` : "—");

  const gpus = s.gpus || [];
  const quiet = gpus.filter((g) => (g.unread || []).length).length;
  text("gpus-note", gpus.length ? `${gpus.length} card${gpus.length > 1 ? "s" : ""} · amdgpu sysfs`
    + (s.smi ? " + rocm-smi" : "") + (quiet ? ` · ${quiet} refusing sysfs reads` : "")
    : "no amdgpu sysfs readable here");
  $("gpus").replaceChildren(...gpus.map(gpuCard));
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

(async function poll() { await refresh(); setTimeout(poll, 2000); })();
