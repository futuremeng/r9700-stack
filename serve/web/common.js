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

// A qualifier hanging off a big number -- "since launch" under a value--sm stat -- must not inherit the
// number's size. At --st-fs-xl three words become the loudest thing on the card and outvote the measurement
// they are qualifying, which is backwards: the qualifier is a caveat on the number, not a second headline.
// qual() renders it at note size; qualText() is text() with one attached.
function qual(t) { return t ? ` <span class="qual">${t}</span>` : ""; }
function qualText(id, value, note) {
  const el = $(id);
  if (!el) return;
  const s = value === null || value === undefined ? "—" : String(value);
  const html = `${s}${qual(note)}`;
  if (el.innerHTML !== html) el.innerHTML = html;
}

// -- sparkline ---------------------------------------------------------------------
// A polyline of the last N polls, drawn as an SVG the token owns so the theme colours it.
function spark(el, values, {fill = true, baseLo = null, baseHi = null} = {}) {
  const vs = values || [];
  const xs = vs.filter(isnum);
  const W = 240, H = 36;
  const mk = (tag, attrs) => {
    const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    return n;
  };
  // Autoscaling to the series' own min and max is what makes a sparkline lie: a KV pool creeping from 0.0% to
  // 2.4% fills the box like a stock chart, and a rate that sits at zero most of the time gets a baseline at
  // zero's height only by accident. A caller that knows the axis anchors it (the pool at 0..100, a rate at
  // 0..peak), and then a sliver reads as a sliver.
  const lo = baseLo !== null ? baseLo : (xs.length ? Math.min(...xs) : 0);
  const hi = baseHi !== null ? baseHi : (xs.length ? Math.max(...xs) : 0);
  if (!isFinite(lo) || !isFinite(hi)) { el.replaceChildren(); return; }
  const span = hi - lo || 1;
  const px = (i) => (vs.length > 1 ? (i * W) / (vs.length - 1) : W / 2);
  const py = (v) => H - 2 - ((Math.max(lo, Math.min(hi, v)) - lo) / span) * (H - 6);
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("preserveAspectRatio", "none");
  svg.setAttribute("class", "spark");
  svg.setAttribute("aria-hidden", "true");
  // The zero line, drawn under the data and kept faint: on an anchored axis it says where nothing-happening
  // is, so a series with no samples at all (no streams running, so a per-stream rate has no denominator)
  // still reads as "flat on a known axis" instead of an empty box that looks broken -- and a series genuinely
  // at zero reads as a line sitting ON the axis, which is the finding, not a rendering failure. For an
  // autoscaled series the line goes where zero is when the range spans it, because crossing zero (the KV pool
  // draining rather than growing) is the event worth seeing there.
  const zero = baseLo !== null ? baseLo : (xs.length && lo <= 0 && hi >= 0 ? 0 : null);
  if (zero !== null) {
    svg.append(mk("path", {class: "spark-base", d: `M0,${py(zero).toFixed(1)}L${W},${py(zero).toFixed(1)}`,
                           fill: "none", "stroke-width": "1", "stroke-dasharray": "3 3"}));
  }
  if (xs.length < 2) { el.replaceChildren(svg); return; }
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
  // One labelled row per sensor wave. They are stacked, not overlaid: power in watts, temperature in degrees
  // and the SM clock in MHz have no shared axis, and drawing three autoscaled series in one box is exactly the
  // shape of lie a sparkline can tell -- a card pinned at its power cap and one idling at 205 of 500 W look
  // identical that way. Each gets its own anchored ceiling below.
  const rows = ["busy", "VRAM", "sclk", "W", "°C"];
  d.innerHTML = `<b>${g.name || g.card}</b><span class="small">${g.card}</span>`
    + `<div class="pct">${busy}</div>`
    + rows.map((lab) => `<div class="sparkrow"><span>${lab}</span>`
        + `<div class="sparkline spark--gpu"></div></div>`).join("")
    + `<div class="muted small">${isnum(g.vram_used) || isnum(g.vram_total) ? `${gb(g.vram_used)} / ${gb(g.vram_total)} GiB VRAM` : ""}</div>`
    + `<div class="muted small" data-wave></div>`
    + `<div class="muted small">${line([isnum(g.mem_busy_pct) ? `VRAM ${fmt(g.mem_busy_pct, 0)}% busy` : "",
        isnum(g.sclk_mhz) ? `sclk ${fmt(g.sclk_mhz, 0)} MHz` : "", isnum(g.mclk_mhz) ? `mclk ${fmt(g.mclk_mhz, 0)} MHz` : "",
        isnum(g.temp_c) ? `${g.temp_c.toFixed(0)} °C` : "", isnum(g.power_w) ? `${g.power_w.toFixed(0)} W` : "",
        isnum(g.power_cap_w) ? `cap ${g.power_cap_w.toFixed(0)} W` : ""])}</div>`
    + unread + smi;
  // Align by bus id / card name, not by position: if a card disappears from sysfs the series must keep meaning
  // instead of quietly plotting its neighbour's numbers. A refused read (EBUSY) is a null, so the wave breaks
  // rather than dipping to idle.
  const at = (s) => (s.cards || []).find((c) => (g.bus && c.bus === g.bus) || c.card === g.card) || null;
  const series = (k) => ghist.map((s) => { const c = at(s); return c ? c[k] : null; });
  const vramS = series("vram_used").map((v) => (isnum(v) ? v / 2 ** 30 : null));
  const vramCeil = (g.vram_total || Math.max(...vramS.filter(isnum), 0)) / 2 ** 30;
  // Ceilings: VRAM is the card's own size, power its cap (a wave against the cap is a throttling question),
  // temperature 110 C (the junction neighbourhood -- a fixed ceiling is what makes 65 and 95 comparable), and
  // the clock its own observed peak, since this kernel exposes no max frequency file.
  const sclkPeak = Math.max(...series("sclk_mhz").filter(isnum), 0);
  const powerPeak = Math.max(...series("power_w").filter(isnum), 0);
  // The cap is the sensor's default, not the operating limit: this box runs uncapped and draws 440 W against a
  // power1_cap of 300 W (measured), so the axis is whichever is larger -- clipping a wave at a cap the card is
  // allowed to exceed would hide exactly the thing worth seeing. The cap stays in the text line for contrast.
  const powerCeil = Math.max(g.power_cap_w || 0, powerPeak);
  const vals = [series("busy_pct"), vramS, series("sclk_mhz"), series("power_w"), series("temp_c")];
  const ceils = [100, vramCeil, sclkPeak, powerCeil, 110];
  d.querySelectorAll(".spark--gpu").forEach((el, j) => {
    spark(el, vals[j], { baseLo: 0, baseHi: ceils[j] });
    // Only temperature gets a warning colour: on an uncapped box exceeding power1_cap is the configuration,
    // not a fault, while 95 C is a fault wherever the card is.
    el.dataset.tone = j === 4 && isnum(g.temp_c) && g.temp_c > 95 ? "danger" : "";
  });
  // One summary line for the thermal trio, because those are the waves read by peak and average rather than
  // by shape: did it ever throttle, and how hot did it get.
  const stat = (k, unit, d2 = 0) => {
    const xs = series(k).filter(isnum);
    return xs.length < 2 ? null
      : `${k === "temp_c" ? "temp" : k === "power_w" ? "power" : "sclk"} peak ${fmt(Math.max(...xs), d2)} · `
        + `avg ${fmt(xs.reduce((a, b) => a + b, 0) / xs.length, d2)}${unit}`;
  };
  const cap = d.querySelector("[data-wave]");
  const xs = vals[0].filter(isnum);
  const spanS = ghist.length > 1 ? ghist[ghist.length - 1].at - ghist[0].at : 0;
  const parts = [stat("temp_c", " °C", 0), stat("power_w", " W"), stat("sclk_mhz", " MHz"),
                 xs.length >= 2 ? `busy avg ${fmt(xs.reduce((a, b) => a + b, 0) / xs.length, 0)}%` : null];
  if (cap) cap.textContent = parts.filter((x) => x).length
    ? `${parts.filter((x) => x).join(" · ")} · ${ghist.length} pts over ${spanS.toFixed(0)} s` : "";
  return d;
}

// -- copy --------------------------------------------------------------------------
document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-copy]");
  if (!b) return;
  const done = () => { const t = b.textContent; b.textContent = "Copied"; setTimeout(() => { b.textContent = t; }, 1200); };
  navigator.clipboard.writeText(b.dataset.copy).then(done).catch(() => {});
});
