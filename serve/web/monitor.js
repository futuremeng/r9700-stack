// serve/web/monitor.js -- the Monitor. Two fetches per tick: /api/status (the engine's counters, read
// off vLLM's /metrics) and /api/requests (what came through this page, when it is in the path).
"use strict";
let records = [], status = null, selected = null, detail = null, pane = "input", stampOf = "";
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
function list() {
  const q = $("search").value.toLowerCase(), filter = $("filter").value;
  const visible = records.filter((r) => `${rid(r)} ${r.path} ${r.model}`.toLowerCase().includes(q)
    && (filter === "all" || (filter === "active" ? active(r) : r.state === filter)));
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
    const ep = document.createElement("div"); ep.className = "endpoint"; ep.textContent = `POST ${r.path}`;
    const meta = document.createElement("div"); meta.className = "small";
    meta.textContent = `${rid(r)} · ${seconds(r.wallclock_s)} · ${r.stream ? "stream" : "JSON"}`
      + `${r.http_status ? ` · HTTP ${r.http_status}` : ""}`;
    b.append(row, ep, meta);
    b.onclick = () => { selected = rid(r); detail = null; list(); showDetail().catch((e) => notice(e.message)); };
    $("requests").append(b);
    if (focusId === rid(r)) b.focus({preventScroll: true});
  }
}

// -- one request, read in full ---------------------------------------------------------------------
function content() {
  if (!detail) return;
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
  const u = value.usage || {};
  text("detail-prompt", isnum(u.prompt_tokens) ? fmt(u.prompt_tokens) : "—");
  text("detail-tokens", isnum(u.completion_tokens) ? fmt(u.completion_tokens)
    : value.timings?.tokens_are_estimate ? "≈ estimated" : "—");
  text("detail-speed", isnum(value.timings?.predicted_per_second) ? fmt(value.timings.predicted_per_second, 1) : "—");
  text("detail-format", value.response_format || "text");
  $("detail-error").hidden = !value.error;
  text("detail-error", value.error?.message || "");
  content();
}

// -- the engine, the cards, and the timings ---------------------------------------------------------
function renderStatus(s) {
  const live = s.live || {}, rate = s.rate || {};
  const inflight = (live.running || 0) + (live.waiting || 0);
  text("mode", s.mode === "proxy" ? "in the path (recording)" : "observing (read-only)");
  text("loaded", !s.reachable ? "Unreachable" : inflight ? "In use" : "Idle");
  $("model-state").className = `value value--sm${inflight ? " busy" : ""}`;
  $("model-state").firstElementChild.className = `dot${inflight ? " dot--busy" : s.reachable ? " dot--live" : " dot--off"}`;
  text("model", s.model_name || (s.models || [])[0]?.id || "no model reported");
  text("upstream", s.upstream || "—");
  const last = records.find((r) => !active(r));
  text("count", String(records.length));
  text("active", inflight ? `${Math.round(inflight)} running / waiting at the engine`
    : s.mode === "proxy" ? "Nothing in flight" : "Nothing routed here");
  text("wall", last ? seconds(last.wallclock_s) : "—");
  const speed = last?.timings?.predicted_per_second ?? rate.stream_tok_s;
  $("speed").innerHTML = `${isnum(speed) ? fmt(speed, 1) : "—"}<span class="unit">tok/s</span>`;

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
  text("accept", pct(rate.accept_pct, 1));
  text("drafts", isnum(rate.drafts_per_req) ? `${fmt(rate.drafts_per_req, 2)} drafts per request` : "no speculation counters");
  text("iters", isnum(rate.tokens_per_iter) ? `${fmt(rate.tokens_per_iter, 2)} tokens per engine step · `
    + `${fmt(rate.iters_per_req, 2)} steps per request` : "");

  const hist = s.history || [];
  spark($("gen-spark"), hist.map((h) => h.gen_tok_s));
  spark($("kv-spark"), hist.map((h) => h.kv_used_pct));

  for (const [id, v] of [["t-ttft", rate.ttft_ms], ["t-queue", rate.queue_ms], ["t-prefill", rate.prefill_ms],
                         ["t-decode", rate.decode_ms]]) text(id, ms(v));
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
    text("kept", String(h.kept));
    $("observe-note").textContent = h.mode === "observe"
      ? "Start this sidecar with --proxy (or point a client at it) to record requests here." : "";
    renderStatus(s);
    list();
    if (selected) await showDetail();
    text("updated", `Live · ${clock(s.at)}${s.mode === "observe" ? " · observing" : " · recording"}`);
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
