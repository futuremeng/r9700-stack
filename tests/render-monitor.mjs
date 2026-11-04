// tests/render-monitor.mjs -- run the Monitor's own JS against a LIVE serve and print what its cards and
// request rows actually render.
//
//   node tests/render-monitor.mjs [BASE|fixture.json]   # BASE defaults to http://127.0.0.1:8080
//
// A .json argument switches to fixture mode: {"status": ..., "requests": ...} is fed to the page instead of a
// live serve, which is how the request rows can be rendered from real journal rows while the box is busy (and
// how the derived per-row fields can be tested against rows that no longer exist anywhere live).
//
// What that does and does not cover. It executes serve/web/common.js and serve/web/monitor.js unmodified in a
// stub DOM, so it checks the JS and the payloads it is fed: which card says what, whether a per-request value
// reaches the row that claims it, whether a missing value prints a dash instead of a borrowed number, whether a
// real 0 survives as 0. It does NOT check CSS layout, fonts, the theme switch's painting, or anything a real
// browser is needed for -- and it is not part of the GPU-free gates, because it needs a serve answering.
// It needs Node (a browser would too); nothing else.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import vm from "node:vm";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.join(HERE, "..", "serve", "web");
const ARG = process.argv[2] || "http://127.0.0.1:8080";
const FIX = ARG.endsWith(".json") ? JSON.parse(readFileSync(ARG, "utf8")) : null;
const BASE = FIX ? "" : ARG.replace(/\/$/, "");

class El {
  constructor(tag = "div", id = "") {
    this.tagName = tag; this.id = id; this._text = ""; this.innerHTML = "";
    this.className = ""; this.dataset = {}; this.attrs = {}; this.children = [];
    this.style = {}; this.hidden = false; this.value = ""; this.disabled = false;
    this.classList = {
      _s: new Set(),
      add(...c) { c.forEach((x) => this._s.add(x)); },
      remove(...c) { c.forEach((x) => this._s.delete(x)); },
      toggle(c) { this._s.has(c) ? this._s.delete(c) : this._s.add(c); },
      contains(c) { return this._s.has(c); },
    };
  }
  get textContent() {
    if (this.children.length && !this._text) return this.children.map((c) => c.textContent).join(" ");
    // A value written as innerHTML (the number-plus-qualifier pattern) still has to read back as text, or the
    // harness reports a card as blank while the browser shows it filled.
    if (!this._text && this.innerHTML) {
      return String(this.innerHTML).replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim();
    }
    return this._text;
  }
  set textContent(v) { this._text = String(v); this.children = []; }
  get firstElementChild() { return this.children[0] || new El(); }
  append(...n) { n.flat().forEach((x) => this.children.push(x)); }
  appendChild(n) { this.append(n); return n; }
  replaceChildren(...n) { this.children = n.flat(); }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return this.attrs[k] ?? null; }
  addEventListener() {}
  querySelectorAll(sel) {
    // Renderers ask for the slots inside a card they are about to fill (the per-sensor wave rows, the caption
    // slot). Hand back real stub elements so what they write is observable instead of silently skipped.
    if (!this._qsa) this._qsa = new Map();
    if (!this._qsa.has(sel)) this._qsa.set(sel, Array.from({ length: 5 }, () => new El("div")));
    return this._qsa.get(sel);
  }
  querySelector(sel) { return this.querySelectorAll(sel)[0]; }
  focus() {}
  remove() {}
}

const byId = new Map();
// The page's own defaults, so a filter that has never been touched behaves like the select in the HTML.
byId.set("filter", new El("select", "filter")); byId.get("filter").value = "all";
byId.set("search", new El("input", "search")); byId.get("search").value = "";

const document = {
  activeElement: null,
  documentElement: new El("html"),
  body: new El("body"),
  createElement: (t) => new El(t),
  createElementNS: (ns, tag) => new El(tag),          // the sparklines are SVG; without this the page throws
  getElementById: (id) => { if (!byId.has(id)) byId.set(id, new El("div", id)); return byId.get(id); },
  querySelectorAll: () => [],
  addEventListener: () => {},
};
const store = () => { const m = new Map(); return { getItem: (k) => (m.has(k) ? m.get(k) : null),
  setItem: (k, v) => m.set(k, v), removeItem: (k) => m.delete(k) }; };

const ctx = {
  // The host's AbortController/AbortSignal, deliberately: api() passes ctl.signal to fetch, and in live mode that
  // fetch is the real Node one -- a signal built from a class defined inside the vm would be rejected as the
  // wrong type. Same objects the page gets in a browser. (This global was missing, which is how the harness found
  // out that api() now puts a deadline on every request: the stub DOM has to grow as the page grows.)
  document, console, Promise, JSON, Math, Date, Number, String, Object, Array, RegExp, Set, Map, Error,
  AbortController, AbortSignal,
  isNaN, parseInt, parseFloat, isFinite, encodeURIComponent, decodeURIComponent, Intl,
  localStorage: store(), sessionStorage: store(),
  navigator: { clipboard: { writeText: async () => {} }, userAgent: "render-monitor" },
  fetch: async (u, o) => {
    if (!FIX) return globalThis.fetch(BASE + u, o);
    const p = String(u).split("?")[0];
    const body = p === "/api/status" ? FIX.status : p === "/api/requests" ? FIX.requests : null;
    if (body === null) return { ok: false, status: 404, json: async () => ({ error: "not in fixture" }) };
    return { ok: true, status: 200, json: async () => body };
  },
  setTimeout: () => 0, setInterval: () => 0,          // never re-arm the page's own poll loop
  clearTimeout: () => {}, clearInterval: () => {}, requestAnimationFrame: () => 0,
};
vm.createContext(ctx);
for (const f of ["common.js", "monitor.js"]) vm.runInContext(readFileSync(path.join(WEB, f), "utf8"), ctx,
                                                              { filename: f });

await new Promise((r) => setTimeout(r, 4000));        // this timer is Node's: it lets one poll land

const t = (id) => (byId.get(id) || new El()).textContent;
const html = (id) => (byId.get(id) || new El()).innerHTML;
const rows = (byId.get("requests") || new El()).children;
const shown = [...byId.keys()].filter((k) => (byId.get(k).textContent || byId.get(k).innerHTML));

for (const [k, v] of Object.entries({
  "mode": t("mode"), "loaded": t("loaded"), "model": t("model"),
  "KV pool": `${t("kv-pct")} | ${t("kv-tokens")}`,
  "Requests": `${t("running")} running | ${t("waiting")}`,
  "Generation": `${t("gen-toks")} tok/s | ${t("prompt-toks")} | ${t("step-size")}`,
  "Speculation": `${t("accept")} | ${t("drafts")} | ${t("per-draft")}`,
  "Wall card": `${t("wall")} | ${t("wall-split")} | ${t("wall-note")}`,
  "Speed card": `${html("speed") || t("speed")} | ${t("speed-note")}`,
  "Counts": `${t("count")} | ${t("active")}`,
  "Engine timings": ["t-ttft", "t-queue", "t-prefill", "t-decode", "t-step", "t-tpot", "window"]
    .map((x) => `${x.replace("t-", "")}=${t(x)}`).join(" · "),
})) console.log(`${k}: ${v}`);

// The card grids are built by gpuCard / pcieCard in common.js and monitor.js: print what the first card of
// each actually says, because a panel that renders an empty div looks identical to a panel with nothing to show.
// The chassis mapping line is the one place the two numbering systems have to agree out loud, so it is printed
// like any other piece of copy: if the join (slot at the switch upstream port, card by BDF) breaks, the line
// goes blank or wrong rather than throwing.
console.log("\nslot map:", t("slot-map").replace(/\s+/g, " ").trim());

for (const [label, id] of [["cards", "gpus"], ["pcie", "pcie"]]) {
  const kids = (byId.get(id) || new El()).children;
  console.log(`\n${label} cards rendered: ${kids.length}`);
  for (const k of kids.slice(0, 2)) {
    const waves = k.querySelectorAll(".spark--gpu").length;
    console.log("  " + k.textContent.replace(/\s+/g, " ").trim().slice(0, 200) + ` [${waves} waves]`);
  }
}

// Grid rows are silent about their own shape: a row that emits six children into a seven-track grid does not
// throw, it just starts a second line, and the panel reads as "the columns are ragged". So the child count is
// asserted here, where it is cheap, rather than discovered by someone squinting at a screenshot.
const PROWS = (byId.get("pcie") || new El()).children;
const AREAS = ["p-slot", "p-drv", "p-chain", "p-link", "p-cnt", "p-who", "p-win"];
const misshapen = PROWS.filter((r) => AREAS.some((a) => !String(r.innerHTML || "").includes(a)))
  .map((r) => { const have = AREAS.filter((a) => String(r.innerHTML || "").includes(a));
    return `${r.textContent.replace(/\s+/g, " ").trim().slice(0, 34)} -> ${have.length}/7 columns,`
      + ` missing ${AREAS.filter((a) => !have.includes(a)).join(" ")}`; });
if (misshapen.length) {
  console.log("\nBAD PCIe GRID ROWS (each must fill all 7 grid areas):");
  for (const m of misshapen) console.log("  " + m);
  process.exitCode = 1;
} else {
  console.log(`\npcie grid: ${PROWS.length} rows, all filling the 7 areas (${AREAS.join(" ")})`);
}
// Pinning is a DELEGATED click on the rows container: the rows are rebuilt every second, so a handler -- or a
// pinned panel -- attached to a row would not survive the tick. Drive it the way a click does, then read what
// the panel says: those lines are the reason the detail is DOM text rather than a `title` tooltip.
const PEL = byId.get("pcie"), DET = byId.get("pcie-detail");
const click = (bus) => PEL.onclick && PEL.onclick({
  target: { closest: (sel) => (sel === "[data-bus]" ? { dataset: { bus } } : null) } });
if (!PEL.onclick) {
  console.log("BAD  #pcie has no delegated click handler, so the count cell cannot be opened"); process.exitCode = 1;
} else {
  click("c6:00.0");
  console.log(`\ndetail pinned on c6:00.0 -- hidden=${DET.hidden}, ${DET.children.length} blocks:`);
  DET.children.forEach((c) => console.log("   |", c.textContent));
  if (DET.hidden || DET.children.length < 5) {
    console.log("BAD  pinning wrote no readable detail"); process.exitCode = 1;
  }
  // Count DOM writes on the pinned lines, then paint again with nothing changed. A panel that rewrote every
  // line each second would destroy the text selection the reader is making -- which is the whole point of it.
  let detWrites = 0, sameWrites = 0;
  const proto = Object.getPrototypeOf(new El("i"));
  const dsc = Object.getOwnPropertyDescriptor(proto, "textContent");
  Object.defineProperty(proto, "textContent", { configurable: true, get: dsc.get,
    // A write is only wasteful if it writes the value that is already there. The panel's own content moves
    // between repaints ("quiet 7 min" becomes "quiet 8 min", an AER counter lands), and rewriting THOSE lines
    // is correct -- an earlier version of this check asserted "nothing changed" without checking, and failed
    // against a link that put a BadTLP on the wire while it ran.
    set(v) { if (this._det) { detWrites++; if (v === dsc.get.call(this)) sameWrites++; } dsc.set.call(this, v); } });
  DET.children.forEach((c) => { if (c.className === "pcie-detail-line") c._det = true; });
  const marked = DET.children.filter((c) => c._det).length;
  await ctx.refresh();
  console.log(`repainted: ${detWrites} line write(s) across ${marked} pinned lines, ${sameWrites} of them the value that was already there`);
  if (sameWrites > 0) { console.log("BAD  a pinned panel rewrites lines to the value they already hold"); process.exitCode = 1; }
  click("c6:00.0");
  console.log(`clicked the same chain again -> hidden=${DET.hidden}, children=${DET.children.length}`);
  if (!DET.hidden) { console.log("BAD  unpinning left the panel open"); process.exitCode = 1; }
  Object.defineProperty(proto, "textContent", dsc);
}
console.log(`\nPCIe note: ${t("pcie-note")}`);
for (const k of (byId.get("pcie") || new El()).children.slice(0, 4))
  console.log("  " + k.textContent.replace(/\s+/g, " ").trim());
console.log(`\nrequest rows rendered: ${rows.length}`);
for (const r of rows.slice(0, 6)) console.log("  " + r.textContent.replace(/\s+/g, " ").trim());
// Which build this page belongs to: the header names the engine, the plugin and the commit, so a screenshot can
// answer "is the live serve the new one?" without an ssh. The stub DOM creates #brand-line on demand, so this
// checks the JS and the payload; the CSS around it still needs a browser.
const brand = t("brand-line");
const STAT = FIX ? FIX.status : await (await fetch(`${BASE}/api/status`)).json();
const want = STAT?.stack?.plugin ? `plugin ${STAT.stack.plugin}` : null;
console.log(`\nbrand line: ${brand}${STAT?.stack ? ` (status stack: ${JSON.stringify(STAT.stack)})` : " (no stack in the payload)"}`);
if (!STAT?.stack) { console.log("BAD  the status payload carries no stack"); process.exitCode = 1; }
if (!/\bvLLM\b/.test(brand || "")) { console.log("BAD  the header never named the engine"); process.exitCode = 1; }
if (want && !(brand || "").includes(want)) { console.log(`BAD  the header does not carry ${want}`); process.exitCode = 1; }
// The monitor repaints once a second; a header that appends its version each time would read "... vLLM 0.32 ·
// vLLM 0.32 · ..." after a minute, so check a second render leaves it standing the same length.
const len0 = (brand || "").length;
if (!FIX) { await new Promise((r) => setTimeout(r, 2500)); }   // longer than the 2 s poll: a repaint really happened
console.log(`brand line after another poll: ${t("brand-line")} (${(t("brand-line") || "").length - len0} chars changed)`);
if (!FIX && (t("brand-line") || "").length !== len0) {
  console.log("BAD  the header grew on repaint -- the version is being appended, not replaced"); process.exitCode = 1;
}

// Which window the four headline numbers answer to, and the spans they are NOT. The default is the 15 min mean,
// falling back to the widest span that has actually filled; the page has to say which it settled on, and every
// card has to carry this poll in its second line -- that demotion of the 1 s sample is the whole point of the
// change, so it is worth a check that the headline really is the span and not the poll.
const isnum = (v) => typeof v === "number" && isFinite(v);
const AVG_W = { "1m": 60, "15m": 900, "1h": 3600 };
const fullSpan = (b, w) => !!b && !b.reset && isnum(b.span_s) && b.span_s >= w * 0.8;
const fulls = Object.entries(AVG_W).filter(([l, w]) => fullSpan(STAT?.avg?.[l], w)).map(([l]) => l);
const winShown = fulls.includes("15m") ? "15m"
  : (fulls.sort((a, b) => AVG_W[b] - AVG_W[a])[0] || "live");
console.log(`\nheadline window: ${winShown}  (${t("win-note")})`);
if (STAT?.avg && winShown === "live" && fulls.length) {
  console.log(`BAD  a span is full (${fulls.join(" ")}) and the headline is still the 1 s poll`); process.exitCode = 1;
}
const firstNum = (h) => { const m = String(h).match(/[\d,]+(?:\.\d+)?/); return m ? Number(m[0].replace(/,/g, "")) : null; };
const near = (a, b) => (a === null || b === null ? a === b : Math.abs(a - b) <= Math.max(0.02, Math.abs(b) * 0.011));
for (const [id, avgKey, rateKey] of [["rate-stream", "stream_tok_s", "stream_tok_s_live"],
                                     ["rate-gen", "generation_tok_s", "generation_tok_s"],
                                     ["rate-prefill", "prefill_tok_s", "prompt_tok_s"],
                                     ["rate-step", "itl_ms", "itl_ms"]]) {
  const want = winShown === "live" ? STAT?.rate?.[rateKey] : STAT?.avg?.[winShown]?.[avgKey];
  const got = firstNum(html(id));
  console.log(`  ${(id + "            ").slice(0, 14)} ${String(html(id)).replace(/<[^>]*>/g, "").trim()}`
              + `   [${winShown} wants ${want === undefined ? "-" : want}]`);
  if (want !== undefined && want !== null && !near(got, want)) {
    console.log(`BAD  ${id} shows ${got} for a ${winShown} window whose value is ${want}`); process.exitCode = 1;
  }
  if (winShown !== "live" && isnum(STAT?.rate?.[rateKey]) && Math.abs(STAT.rate[rateKey] - want) > Math.abs(want) * 0.25)
    if (near(got, STAT.rate[rateKey])) {
      console.log(`BAD  ${id} is printing this poll (${STAT.rate[rateKey]}) under a ${winShown} label`); process.exitCode = 1;
    }
}
// The spans the headline is not, plus this poll.
console.log("\nthe other spans, under each card");
for (const [id, key, rateKey] of [["avg-stream", "stream_tok_s", "stream_tok_s_live"],
                                  ["avg-gen", "generation_tok_s", "generation_tok_s"],
                                  ["avg-prefill", "prefill_tok_s", "prompt_tok_s"],
                                  ["avg-step", "itl_ms", "itl_ms"],
                                  ["avg-kv", "kv_used_pct_mean", "__kv"]]) {
  const line = t(id) || "";
  const shown = Object.entries((STAT?.avg || {})).filter(([l, b]) => isnum(b?.[key]) && l !== winShown);
  console.log(`  ${(id + "          ").slice(0, 12)} ${line}`
              + (shown.length ? `   [${shown.map(([l, b]) => `${l}=${b[key].toFixed(1)}/${Math.round(b.span_s)}s`).join(" ")}]`
                              : "   [no other span has this field]"));
  for (const [lbl, b] of shown) {
    const short = b.span_s < AVG_W[lbl] * 0.8;
    if (!short && !line.includes(lbl + " ")) {
      console.log(`BAD  ${id}: ${lbl} has ${b[key]} over ${Math.round(b.span_s)}s and the line omits it`); process.exitCode = 1;
    }
    if (short && line.includes(lbl + " ")) {
      console.log(`BAD  ${id}: prints ${lbl} for a span of ${Math.round(b.span_s)}s -- a young window wearing an old label`); process.exitCode = 1;
    }
  }
  if (winShown !== "live" && !/poll /.test(line) && !/no trailing averages/.test(line)) {
    console.log(`BAD  ${id}: the headline is an average and the card does not carry this poll`); process.exitCode = 1;
  }
}
// The trend rings: 90 s of per-poll points, or an hour of 15 s interval rates. The caption's span has to come
// from the samples, not from points x poll, which is the thing that would otherwise quietly lie.
console.log("\ntrend source");
console.log(`  90 s ring: ${t("trend-note")}`);
const slowNote = (() => { const el = byId.get("trend-slow"); if (el?.onclick) el.onclick(); return t("trend-note"); })();
console.log(`  1 h ring: ${slowNote}`);
if (/one point per/.test(slowNote)) {
  console.log("BAD  pressing the 1 h trend button left the fast ring in charge"); process.exitCode = 1;
}
if (byId.get("trend-fast")?.onclick) byId.get("trend-fast").onclick();
console.log(`  back to 90 s: ${t("trend-note")}`);
if (!/one point per/.test(t("trend-note"))) {
  console.log("BAD  pressing the 90 s trend button did not restore the fast ring"); process.exitCode = 1;
}
console.log(`  caption now: ${t("gen-spark-note")}`);

// The page swallows render errors into its notice bar, so a broken section can look like an empty one.
const bar = t("error");
if (bar) { console.log("\nPAGE NOTICE:", bar); process.exitCode = 1; }
if (!rows.length) { console.log("\nno request rows rendered"); process.exitCode = 1; }
console.log(`\ncards the page wrote a value into: ${shown.sort().join(" ")}`);
