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

// which build this page belongs to: the header names the engine, the plugin and the commit, so "is the live serve
// the new one?" is answered by the page instead of by ssh. The base text is remembered on first use because the
// monitor repaints this element once a second, and a label that appends a version every second is a bug report
// waiting to happen. A dirty checkout gets `*` and the reason in the title, since the commit alone would claim a
// build the repository does not contain.
function brandStack(s) {
  const b = $("brand-line");
  if (!b) return;
  if (b.dataset.base === undefined) b.dataset.base = b.textContent;
  const stack = (s && s.stack) || {};
  const v = (s && s.engine && s.engine.vllm_version) || "";
  const parts = [];
  // The engine version arrives as whatever this process reports -- "vllm 0.32.0+rocm723", or "vllm 0.32.0
  // (this process)" from the merged page's own registry -- and the header has room for the number, not the
  // sentence. Take the version token and label it once.
  const vm = String(v).match(/\d[\w.]*/);
  if (vm) parts.push("vLLM " + vm[0]);
  if (stack.plugin) parts.push("plugin " + stack.plugin);
  if (stack.commit) parts.push(stack.commit + (stack.dirty ? "*" : ""));
  b.textContent = [b.dataset.base, ...parts].filter(Boolean).join(" · ");
  if (stack.commit) b.title = stack.dirty
    ? `checkout ${stack.branch || "(detached)"} at ${stack.commit}, with uncommitted edits in the tree`
    : `checkout ${stack.branch || "(detached)"} at ${stack.commit}, clean`;
}

// -- fetching ----------------------------------------------------------------------
// The key lives in sessionStorage only: it is the serve key, this page never has one of its own.
const FETCH_MS = 4000;                       // a GET of a local sidecar answers in well under 0.1 s
function setKey(k) { apiKey = (k || "").trim(); sessionStorage.setItem(KEY, apiKey); }
function headers(json) {
  const h = {};
  if (apiKey) h.Authorization = `Bearer ${apiKey}`;
  if (json) h["Content-Type"] = "application/json";
  return h;
}
async function api(path, body, timeoutMs = FETCH_MS) {
  // Every call carries a deadline. The pages re-fetch on a self-rescheduling timer
  // (`await refresh(); setTimeout(poll, 1000)`), so one request that never returns -- a stalled connection, a
  // proxy that swallowed it, a half-closed socket -- would leave the page showing its last render forever,
  // silently, with no way for the reader to tell a quiet box from a dead page. A timeout turns that into one
  // visible error and the next tick.
  const ctl = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; ctl.abort(); }, timeoutMs);
  let r;
  try {
    r = await fetch(path, {cache: "no-store", signal: ctl.signal, headers: headers(body !== undefined),
      ...(body !== undefined ? {method: "POST", body: JSON.stringify(body)} : {})});
  } catch (e) {
    throw new Error(timedOut ? `${path} did not answer within ${(timeoutMs / 1000).toFixed(0)} s`
                             : (e && e.message) || `${path} failed`);
  } finally {
    clearTimeout(timer);
  }
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
// Escape what gets interpolated into innerHTML. Every one of these values comes from sysfs, the engine's
// metrics or the kernel log rather than from a user, but a device label or an error string is a poor place to
// assume that, and about.js has carried the same helper for the same reason.
function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"]/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function gpuCard(g, ghist = [], i = 0) {
  const d = document.createElement("div");
  d.className = "gpu";
  const line = (a) => a.filter((x) => x).join(" · ");
  // Where the card physically is. `card2` is a name the driver assigned this boot and says nothing about the
  // chassis; the BDF is the card's identity and the slot number (from the firmware's own table, via sysfs, no
  // root needed) is where the machine says it is plugged in. On this box: card3 slot 1, card1 slot 3,
  // card2 slot 6, card4 slot 8. A board whose firmware describes no slots simply shows no slot.
  // The headline is the physical slot, because that is the identity that survives a reseat and the one a person
  // standing at the machine can point at. `cardN` is kept underneath: it is the name the driver assigned this
  // boot, and it is also the key the sensor history and the PCIe rows are aligned by, so it is never dropped --
  // and `#N in slot order` says where this card sits in the order the panel is showing them.
  const where = [g.pos && `#${g.pos} in slot order`, g.bus && `PCI ${g.bus}`].filter(Boolean).join(" \u00b7 ");

  const unread = (g.unread || []).length
    ? `<div class="muted small">not readable from sysfs: ${(g.unread || []).join(", ")}</div>` : "";
  const smi = (g.from_smi || []).length
    ? `<div class="muted small">from rocm-smi: ${(g.from_smi || []).join(", ")}</div>` : "";
  const busy = isnum(g.busy_pct) ? `${fmt(g.busy_pct, 0)}<span class="unit">busy</span>` : `<span class="q">—</span>`;
  // One labelled row per sensor wave. They are stacked, not overlaid: power in watts, temperature in degrees
  // and the SM clock in MHz have no shared axis, and drawing three autoscaled series in one box is exactly the
  // shape of lie a sparkline can tell -- a card pinned at its power cap and one idling at 205 of 500 W look
  // identical that way. Each gets its own anchored ceiling below. Two temperature waves rather than one:
  // junction is the sensor with a published limit and the one the fan map follows, mem is the one that actually
  // separates these four cards (measured 2026-11-02: card4 mem 95.7 C against 82.7-86.0 for the others, with its
  // junction equal to card3's). Edge -- what this panel used to draw -- is the least informative of the three and
  // stays in the text line.
  const rows = [["busy", "gpu_busy_percent"], ["VRAM", "mem_info_vram_used"], ["sclk", "freq1 (sclk)"],
                ["W", "power1_average"]];
  // A page newer than the serve it is talking to must not claim a sensor that build never read: an in-serve
  // Monitor started before this change reports temp1 only, so the headline is labelled by what the payload
  // actually carries, and the mem row appears only when there is a mem number to plot.
  const hasMem = isnum(g.temp_mem_c) || ghist.some((s) => (s.cards || []).some((c) => isnum(c.temp_mem_c)));
  const tlab = isnum(g.temp_junction_c) ? "junction"
    : isnum(g.temp_edge_c) ? "edge" : "temp";
  rows.push([tlab, tlab === "junction"
    ? `the die sensor: the one with a published limit (temp*_crit${isnum(g.temp_crit_c) ? ` = ${g.temp_crit_c.toFixed(0)} C` : ""}) and the one the fan map follows`
    : tlab === "edge" ? "the board sensor. This build reports edge separately from the headline; the die "
      + "(junction) and memory numbers came later"
    : "one temperature sensor, which one it is, this build does not say"]);
  if (hasMem) {
    rows.push(["mem", `GDDR7 (crit ${isnum(g.temp_mem_crit_c) ? g.temp_mem_crit_c.toFixed(0) : "?"} C, emergency `
      + `${isnum(g.temp_mem_emerg_c) ? g.temp_mem_emerg_c.toFixed(0) : "?"} C) -- its own limits, not the die's. `
      + "It is the sensor that separates these four cards, so read it both ways: against its limit and against "
      + "its neighbours"]);
  }
  d.innerHTML = `<b>${g.slot ? `slot ${g.slot}` : (g.name || g.card)}</b>`
    + `<span class="small">${[g.card, where].filter(Boolean).join(" \u00b7 ")}</span>`
    + `<div class="pct">${busy}</div>`
    + rows.map(([lab, title]) => `<div class="sparkrow"><span title="${title}">${lab}</span>`
        + `<div class="sparkline spark--gpu"></div></div>`).join("")
    + `<div class="muted small">${isnum(g.vram_used) || isnum(g.vram_total) ? `${gb(g.vram_used)} / ${gb(g.vram_total)} GiB VRAM` : ""}</div>`
    + `<div class="muted small" data-wave></div>`
    + `<div class="muted small">${line([isnum(g.mem_busy_pct) ? `VRAM ${fmt(g.mem_busy_pct, 0)}% busy` : "",
        isnum(g.sclk_mhz) ? `sclk ${fmt(g.sclk_mhz, 0)} MHz` : "", isnum(g.mclk_mhz) ? `mclk ${fmt(g.mclk_mhz, 0)} MHz` : "",
        isnum(g.temp_edge_c) ? `edge ${g.temp_edge_c.toFixed(0)} °C` : "",
        isnum(g.fan_rpm) ? `fan ${fmt(g.fan_rpm, 0)} rpm${isnum(g.pwm) ? ` duty ${g.pwm}/255` : ""}` : "",
        isnum(g.power_w) ? `${g.power_w.toFixed(0)} W` : "",
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
  // The temperature axis comes from the card's own published limit where it publishes one (temp*_crit = 110 C
  // here), so the distance to the wall is visible rather than invented.
  const tempCeil = g.temp_crit_c || 110;
  const vals = [series("busy_pct"), vramS, series("sclk_mhz"), series("power_w")];
  const ceils = [100, vramCeil, sclkPeak, powerCeil];
  const tempIdx = [];
  let memIdx = -1, memCeil = tempCeil;
  vals.push(series("temp_c")); ceils.push(tempCeil); tempIdx.push(vals.length - 1);
  if (hasMem) {
    vals.push(series("temp_mem_c"));
    // the memory wave runs against the memory card's own crit (108 C on these cards, published as temp3_crit),
    // not against the die's: they are different limits and the memory one is lower
    memCeil = g.temp_mem_crit_c || tempCeil;
    ceils.push(Math.max(memCeil, ...vals[vals.length - 1].filter(isnum), 0));
    memIdx = vals.length - 1;
  }
  d.querySelectorAll(".spark--gpu").forEach((el, j) => {
    spark(el, vals[j], { baseLo: 0, baseHi: ceils[j] });
    // Colour means "near the wall", and the wall is per sensor. Junction against its own crit: these cards sit
    // at 90-106 C under load with crit 110, so warn/danger belong in the last few degrees, not at a borrowed
    // 95. Power never gets coloured -- on an uncapped box passing power1_cap is the configuration. Mem never
    // either: no published limit, so any threshold would be invented; it is read against its neighbours.
    // An unlabelled build's single temperature sensor is not coloured either: without knowing which
    // sensor it is, a threshold is a guess.
    // each temperature wave is graded against its own sensor's published crit: die 110 / mem 108 C
    const kt = j === memIdx ? "temp_mem_c" : "temp_c";
    const kc = j === memIdx ? g.temp_mem_crit_c : g.temp_crit_c;
    const near = (tlab === "junction" || j === memIdx) && isnum(g[kt]) && isnum(kc)
      ? (g[kt] > 0.98 * kc ? "danger" : g[kt] > 0.95 * kc ? "warn" : "") : "";
    el.dataset.tone = near;
  });
  // One summary line for the thermal readings, because those waves are read by peak and average rather than by
  // shape: did it ever get near its limit, and how hot did the memory actually get.
  const stat = (k, lab, unit, d2 = 0) => {
    const xs = series(k).filter(isnum);
    return xs.length < 2 ? null
      : `${lab} peak ${fmt(Math.max(...xs), d2)} \u00b7 avg ${fmt(xs.reduce((a, b) => a + b, 0) / xs.length, d2)}${unit}`;
  };
  const cap = d.querySelector("[data-wave]");
  const xs = vals[0].filter(isnum);
  const spanS = ghist.length > 1 ? ghist[ghist.length - 1].at - ghist[0].at : 0;
  const critTxt = isnum(g.temp_crit_c) ? ` (crit ${g.temp_crit_c.toFixed(0)})` : "";
  const parts = [stat("temp_c", tlab, ` \u00b0C${tlab === "junction" ? critTxt : ""}`, 0),
                 hasMem ? stat("temp_mem_c", "mem",
                              ` \u00b0C${isnum(g.temp_mem_crit_c) ? ` (crit ${g.temp_mem_crit_c.toFixed(0)})` : ""}`,
                              0) : null,
                 stat("power_w", "power", " W"), stat("sclk_mhz", "sclk", " MHz"),
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
