// serve/web/about.js -- the About page. One read of /api/status and one of /api/host; no polling
// loop beyond a slow refresh, because nothing here is a live curve (that is the Monitor's job).
"use strict";
let status = null;

function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"
  })[c]);
}

// facts(el, [[label, value, copy?]]) -- a dash row is dropped, a copy row gets a copy button.
function facts(el, rows) {
  el.innerHTML = rows.filter((r) => r[1] !== null && r[1] !== undefined && r[1] !== "")
    .map(([k, v, copy]) => `<dt>${esc(k)}</dt><dd>${copy ? `<code>${esc(v)}</code>`
      + ` <button class="copy" data-copy="${esc(v)}" aria-label="Copy ${esc(k)}">copy</button>` : esc(v)}</dd>`)
    .join("");
}

const yesno = (v) => (v === undefined || v === null || v === "" ? null : (String(v) === "false" ? "off" : "on"));
const num = (v) => (v === undefined || v === null || v === "" ? null : fmt(Number(v)));

function uptime(s) {
  if (!isnum(s)) return null;
  const d = Math.floor(s / 86400), h = Math.floor(s % 86400 / 3600), m = Math.floor(s % 3600 / 60);
  return (d ? `${d} d ` : "") + (h ? `${h} h ` : "") + `${m} min`;
}

function render() {
  const s = status || {};
  const c = s.config || {}, meta = s.meta || {}, models = s.models || [];
  const model = s.model_name || models[0]?.id || null;
  const host = s.host || {};
  text("mode", s.mode === "proxy" ? "recording requests"
    : s.mode === "merged" ? "inside this serve (serve/plugin)" : "observing (read-only)");
  // "this page is a sidecar" is not true of the merged page, and a page that describes itself wrongly is
  // worse than one that describes nothing.
  if (s.mode === "merged")
    $("api-note").textContent = "this page IS the serve: it was mounted by serve/plugin on this same port, "
      + "it does not relay, and it never sees a request body.";

  facts($("facts-engine"), [
    ["Model", model],
    ["Engine", s.engine?.vllm_version || c.version || null],
    ["Max context", num(c.max_model_len ?? models[0]?.max_model_len) ? `${fmt(c.max_model_len ?? models[0].max_model_len)} tokens` : null],
    ["Parallelism", c.tensor_parallel_size ? `TP ${c.tensor_parallel_size}`
      + (c.pipeline_parallel_size && c.pipeline_parallel_size !== "1" ? ` · PP ${c.pipeline_parallel_size}` : "")
      + (c.data_parallel_size && c.data_parallel_size !== "1" ? ` · DP ${c.data_parallel_size}` : "") : null],
    ["KV pool", isnum(s.live?.kv_tokens) ? `${fmt(s.live.kv_tokens)} tokens` : null],
    ["KV block", c.block_size ? `${c.block_size} tokens/block` : null],
    ["KV dtype", c.kv_cache_dtype || null],
    ["Prefix caching", yesno(c.enable_prefix_caching)],
    ["Scheduler", c.scheduler_policy || null],
    ["Reachable", s.reachable ? `yes · /health ${s.health ?? 200}` : `no: ${s.error || "not answering"}`],
  ]);

  const keys = Object.keys(meta).sort();
  $("launch-card").hidden = !keys.length;
  if (keys.length) {
    text("launch-src", s.meta_src || "--meta");
    facts($("facts-launch"), keys.map((k) => [k, Array.isArray(meta[k]) ? meta[k].join(", ")
      : typeof meta[k] === "object" ? JSON.stringify(meta[k]) : String(meta[k])]));
  }

  facts($("facts-host"), [
    ["Hostname", host.hostname],
    ["CPU", host.cpu ? `${host.cpu}${host.threads ? `, ${host.threads} threads` : ""}` : null],
    ["RAM", isnum(host.ram) ? `${gb(host.ram, 0)} GiB` : null],
    ["Kernel", host.kernel],
    ["Up", uptime(host.uptime_s)],
  ]);

  const gpus = s.gpus || [];
  $("gpus").replaceChildren(...gpus.map(gpuCard));

  const base = s.upstream || "";
  facts($("facts-api"), [
    ["OpenAI base URL", base, true],
    ["Model name", model, true],
    ["Through this page", s.mode === "proxy" ? location.origin : "not routed here — use the base URL above"],
    ["API key", s.key_needed ? "the serve has one (set it below)" : "none set here"],
  ]);
  const curl = `curl ${base}/v1/chat/completions \\\n`
    + `  -H 'Content-Type: application/json'${s.key_needed ? ` \\\n  -H 'Authorization: Bearer $KEY'` : ""} \\\n`
    + `  -d '{"model": "${model || "model"}", "messages": [{"role": "user", "content": "hello"}],\n`
    + `       "max_tokens": 64, "temperature": 0}'`;
  $("curl").textContent = curl;
  document.querySelector("[data-copy-target=curl]").dataset.copy = curl;

  facts($("facts-self"), [
    ["Upstream", base],
    ["Mode", s.mode === "proxy" ? "in the request path (recording)" : "watching /health, /v1/models, /metrics only"],
    ["Poll", isnum(s.poll_s) ? `${s.poll_s} s` : null],
    ["Records kept", String(s.requests_kept ?? 0)],
    ["Series read", isnum(s.metrics_seen) ? `${s.metrics_seen} vllm:* series` : null],
  ]);
  text("updated", `Read ${clock(s.at)}`);
}

async function load() {
  try {
    const [s, host] = await Promise.all([api("/api/status"), api("/api/host")]);
    status = {...s, host};
    $("api-key").value = apiKey;
    render();
    notice("");
  } catch (e) {
    notice(e.message);
    text("updated", "Disconnected");
  }
}

function notice(message, info) {
  const el = $("error");
  el.hidden = !message;
  el.className = `notice${info ? " notice--info" : ""}`;
  if (message) el.textContent = message;
}

$("key-btn").onclick = () => { $("api-key").focus(); $("api-key").select(); };
$("key-save").onclick = async () => { setKey($("api-key").value); await load(); };
$("api-key").onkeydown = (e) => { if (e.key === "Enter") { e.preventDefault(); $("key-save").click(); } };
document.addEventListener("need-key", () => $("api-key").focus());
bindTheme();

(async function loop() { await load(); setTimeout(loop, 5000); })();
