// serve/web/common.js -- the pieces the Monitor and the About page share: theme, API key, fetching,
// number formatting, sparklines, copy buttons. No framework, no network beyond this server.
"use strict";
const $ = (id) => document.getElementById(id);
const KEY = "r9700.web.key";
let apiKey = sessionStorage.getItem(KEY) || "";

// theme: dark is the default here (these pages live next to a rack of cards), light is one click away
document.documentElement.dataset.theme = localStorage.getItem("r9700.theme") || "dark";
function bindTheme(root = document) {
  root.querySelectorAll("[data-theme-toggle]").forEach((b) => {
    b.onclick = () => {
      const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      localStorage.setItem("r9700.theme", next);
      syncTheme(root);
    };
  });
  syncTheme(root);
}
function syncTheme(root = document) {
  const dark = document.documentElement.dataset.theme === "dark";
  root.querySelectorAll("[data-theme-toggle]").forEach((b) => b.setAttribute("aria-checked", String(dark)));
}

// -- fetching ----------------------------------------------------------------------
// The key lives in sessionStorage only: it is the serve key, this page never has one of its own.
function setKey(k) { apiKey = (k || "").trim(); sessionStorage.setItem(KEY, apiKey); }
function headers(json) {
  const h = {};
  if (apiKey) h.Authorization = `Bearer ${apiKey}`;
  if (json) h["Content-Type"] = "application/json";
  return h;
}
async function api(path, body) {
  const r = await fetch(path, {cache: "no-store", headers: headers(body !== undefined),
    ...(body !== undefined ? {method: "POST", body: JSON.stringify(body)} : {})});
  let value = null;
  try { value = await r.json(); } catch (_) { /* an empty body is an answer too */ }
  if (!r.ok) {
    if (r.status === 401) document.dispatchEvent(new CustomEvent("need-key"));
    throw new Error((value && value.error && value.error.message) || `HTTP ${r.status}`);
  }
  return value;
}

// -- numbers -----------------------------------------------------------------------
const isnum = (n) => typeof n === "number" && isFinite(n);
const fmt = (n, d = 0) => (isnum(n) ? n.toLocaleString("en-US", {minimumFractionDigits: d, maximumFractionDigits: d}) : "—");
const seconds = (n) => (isnum(n) ? (n >= 100 ? `${(n / 60).toFixed(1)} min` : `${n.toFixed(2)} s`) : "—");
const ms = (n) => (isnum(n) ? (n >= 1000 ? `${(n / 1000).toFixed(2)} s` : `${n.toFixed(n < 10 ? 2 : 1)} ms`) : "—");
const pct = (n, d = 1) => (isnum(n) ? `${n.toFixed(d)}%` : "—");
const gb = (bytes, d = 1) => (isnum(bytes) ? (bytes / 2 ** 30).toFixed(d) : "—");
const clock = (t) => (isnum(t) ? new Date(t * 1000).toLocaleTimeString() : "—");
const stamp = (t) => (isnum(t) ? new Date(t * 1000).toLocaleString() : "—");

// text() only touches the DOM when the string actually changed: the monitor repaints twice a second
// and a repaint of a list you are trying to select text out of is its own bug.
function text(id, value) {
  const el = $(id);
  if (!el) return;
  const s = value === null || value === undefined ? "—" : String(value);
  if (el.textContent !== s) el.textContent = s;
}

// -- sparkline ---------------------------------------------------------------------
// A polyline of the last N polls, drawn as an SVG the token owns so the theme colours it.
function spark(el, values, {fill = true, baseLo = null, baseHi = null} = {}) {
  const vs = values || [];
  const xs = vs.filter(isnum);
  if (xs.length < 2) { el.replaceChildren(); return; }
  const W = 240, H = 36;
  // Autoscaling to the series' own min and max is what makes a sparkline lie: a KV pool creeping from 0.0% to
  // 2.4% fills the box like a stock chart, and a rate that sits at zero most of the time gets a baseline at
  // zero's height only by accident. A caller that knows the axis anchors it (the pool at 0..100, a rate at
  // 0..peak), and then a sliver reads as a sliver.
  const lo = baseLo !== null ? baseLo : Math.min(...xs);
  const hi = baseHi !== null ? baseHi : Math.max(...xs);
  const span = hi - lo || 1;
  const px = (i) => (i * W) / (vs.length - 1);
  const py = (v) => H - 2 - ((Math.max(lo, Math.min(hi, v)) - lo) / span) * (H - 6);
  // Nulls are gaps, not zeros, and x spans the whole ring either way: one sample stays one slot of time no
  // matter what the series was doing in between (the prefill estimate has no value at all before the first
  // request ran). The filled area under the line only closes over one subpath, so it is dropped when the
  // series has a gap -- shading across a gap reads as "it was at baseline there", the opposite of a null.
  let d = "", pen = false, gaps = 0;
  vs.forEach((v, i) => {
    if (!isnum(v)) { if (pen) gaps++; pen = false; return; }
    d += `${pen ? "L" : "M"}${px(i).toFixed(1)},${py(v).toFixed(1)}`;
    pen = true;
  });
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("preserveAspectRatio", "none");
  svg.setAttribute("class", "spark");
  svg.setAttribute("aria-hidden", "true");
  const mk = (tag, attrs) => {
    const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    return n;
  };
  if (fill && !gaps) svg.append(mk("path", {d: `${d}L${W},${H}L0,${H}Z`, fill: "currentColor", "fill-opacity": ".14", stroke: "none"}));
  svg.append(mk("path", {d, fill: "none", stroke: "currentColor", "stroke-width": "1.6", "stroke-linejoin": "round"}));
  el.replaceChildren(svg);
}

// -- one card's block, shared by both pages ---------------------------------------------------------
// A card that answers nothing is said to answer nothing: on this box several amdgpu attributes return
// EBUSY while a compute context holds the card, and "0% busy" is the wrong answer for that.
function gpuCard(g, ghist = [], i = 0) {
  const d = document.createElement("div");
  d.className = "gpu";
  const line = (a) => a.filter((x) => x).join(" · ");
  const unread = (g.unread || []).length
    ? `<div class="muted small">not readable from sysfs: ${(g.unread || []).join(", ")}</div>` : "";
  const smi = (g.from_smi || []).length
    ? `<div class="muted small">from rocm-smi: ${(g.from_smi || []).join(", ")}</div>` : "";
  const busy = isnum(g.busy_pct) ? `${fmt(g.busy_pct, 0)}<span class="unit">busy</span>` : `<span class="q">—</span>`;
  d.innerHTML = `<b>${g.name || g.card}</b><span class="small">${g.card}</span>`
    + `<div class="pct">${busy}</div>`
    + `<div class="sparkline spark--gpu"></div>`
    + `<div class="muted small">${isnum(g.vram_used) || isnum(g.vram_total) ? `${gb(g.vram_used)} / ${gb(g.vram_total)} GiB VRAM` : ""}</div>`
    + `<div class="sparkline spark--gpu"></div>`
    + `<div class="muted small" data-wave></div>`
    + `<div class="muted small">${line([isnum(g.mem_busy_pct) ? `VRAM ${fmt(g.mem_busy_pct, 0)}% busy` : "",
        isnum(g.sclk_mhz) ? `sclk ${fmt(g.sclk_mhz, 0)} MHz` : "", isnum(g.mclk_mhz) ? `mclk ${fmt(g.mclk_mhz, 0)} MHz` : "",
        isnum(g.temp_c) ? `${g.temp_c.toFixed(0)} °C` : "", isnum(g.power_w) ? `${g.power_w.toFixed(0)} W` : ""])}</div>`
    + unread + smi;
  // Busy and VRAM as waveforms, on the honest axes: busy over 0..100 (a card at 40% is 40% of the box, not the
  // tallest bar in the ring), VRAM over 0..the card's own total -- which is the leak question asked without a
  // script: the soak runs used to log per-card VRAM to a side file to show flat is flat, and this is the same
  // evidence, always on. Align by bus id / card name rather than by index: if a card disappears from sysfs the
  // series keeps meaning instead of quietly plotting the neighbour's numbers.
  const at = (s) => (s.cards || []).find((c) => (g.bus && c.bus === g.bus) || c.card === g.card) || null;
  const series = (k) => ghist.map((s) => { const c = at(s); return c ? c[k] : null; });
  const busyS = series("busy_pct");
  const sparks = d.querySelectorAll(".spark--gpu");
  spark(sparks[0], busyS, { baseLo: 0, baseHi: 100 });
  const vramS = series("vram_used");
  const total = g.vram_total || Math.max(...vramS.filter(isnum), 0);
  spark(sparks[1], vramS.map((v) => (isnum(v) ? v / 2 ** 30 : null)),
        { baseLo: 0, baseHi: total / 2 ** 30 });
  const xs = busyS.filter(isnum);
  const cap = d.querySelector("[data-wave]");
  const spanS = ghist.length > 1 ? ghist[ghist.length - 1].at - ghist[0].at : 0;
  if (cap) cap.textContent = xs.length < 2 ? ""
    : `busy peak ${fmt(Math.max(...xs), 0)} · avg ${fmt(xs.reduce((a, b) => a + b, 0) / xs.length, 0)}% `
      + `· ${xs.length}/${ghist.length} pts over ${spanS.toFixed(0)} s`;
  return d;
}

// -- copy --------------------------------------------------------------------------
document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-copy]");
  if (!b) return;
  const done = () => { const t = b.textContent; b.textContent = "Copied"; setTimeout(() => { b.textContent = t; }, 1200); };
  navigator.clipboard.writeText(b.dataset.copy).then(done).catch(() => {});
});
