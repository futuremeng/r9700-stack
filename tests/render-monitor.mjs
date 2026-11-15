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
      toggle(c, on) { if (on === undefined) { this._s.has(c) ? this._s.delete(c) : this._s.add(c); }
                                       else { on ? this._s.add(c) : this._s.delete(c); } },
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

let LAST_STATUS = null;
let LAST_GATE = null;                         // the gate payload the PAGE was fed, same reason as LAST_STATUS
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
  // The payload is kept as well as handed to the page: `let status` inside monitor.js is a lexical binding, not
  // a property of the vm's global object, so the harness cannot read the page's own copy. A clone is the only
  // way to see the body without consuming the response the page is about to read.
  fetch: async (u, o) => {
    if (!FIX) {
      const r = await globalThis.fetch(BASE + u, o);
      if (String(u).split("?")[0] === "/api/status") {
        try { LAST_STATUS = await r.clone().json(); } catch (e) { /* a page poll we did not need */ }
      }
      if (String(u).split("?")[0] === "/api/gate") {
        try { LAST_GATE = await r.clone().json(); } catch (e) { /* ditto */ }
      }
      return r;
    }
    const p = String(u).split("?")[0];
    const body = p === "/api/status" ? FIX.status : p === "/api/requests" ? FIX.requests
               : p === "/api/sidecars" ? (FIX.sidecars ?? null)
               : p === "/api/gate" ? (FIX.gate ?? null) : null;
    if (p === "/api/status") LAST_STATUS = body;                  // the harness reads the same payload
    if (p === "/api/gate") LAST_GATE = body;
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

// The card is the whole PCIe story now (2026-11-05): one badge per card saying the generation it negotiated.
// What the harness has to prove is the rule, not just the presence: a card at its rating renders a word and no
// button (nothing to open), and only a card whose link came up short renders a button that carries its bus.
const CARDS = (byId.get("gpus") || new El()).children;
// The stub DOM has no real tree to query (querySelectorAll hands back slots, not descendants), so the badge is
// read out of the innerHTML the card was built with -- the same string a browser would parse.
const badge = (k) => {
  const m = /<div class="gpu-link"[^>]*>([\s\S]*?)<\/div>/.exec(String(k.innerHTML || ""));
  return m ? { html: m[0], text: m[1].replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim(),
               tone: (/data-tone="([^"]*)"/.exec(m[0]) || [])[1] || "",
               btn: /lk-btn/.test(m[0]) } : null;
};
console.log(`\ncards rendered: ${CARDS.length}`);
for (const k of CARDS) {
  const b = badge(k);
  console.log("  " + k.textContent.replace(/\s+/g, " ").trim().slice(0, 78)
              + ` | link: ${b ? `"${b.text}" tone=${b.tone || "-"}${b.btn ? " [button]" : ""}` : "no badge"}`);
}
const chains = (LAST_STATUS && LAST_STATUS.pcie) || [];   // the payload the page just drew from
const shortc = chains.filter((c) => c.degraded && !c.parked);
const badCards = CARDS.filter((k) => { const b = badge(k); return b && (b.btn !== /data-bus=/.test(b.html)); });
if (badCards.length) {
  console.log(`BAD  ${badCards.length} card(s) whose badge is a button without a bus to open, or the reverse`);
  process.exitCode = 1;
}
// An error count must not reach a card that holds its rating. The counters stay in the payload; the rendered
// cell stays a word. This is the whole point of the change, so it is asserted on what the page draws.
const noisy = CARDS.filter((k) => { const b = badge(k); return b && /\bcorr\b/.test(b.text) && !b.btn; });
if (noisy.length) {
  console.log(`BAD  ${noisy.length} card(s) print an AER count on a link that is not short`); process.exitCode = 1;
}
console.log(`chains: ${chains.length} total, ${shortc.length} short (of which ${
  chains.filter((c) => c.parked).length} parked)`);

// Pinning is a DELEGATED click on the CARDS container: the cards are rebuilt every second, so a handler -- or a
// pinned account -- attached to a card would not survive the tick. Drive it the way a click does, then read what
// the account says: those lines are the reason the detail is DOM text rather than a `title` tooltip.
const GPEL = byId.get("gpus"), DET = byId.get("pcie-detail");
const click = (bus) => GPEL.onclick && GPEL.onclick({
  target: { closest: (sel) => (sel === "[data-bus]" ? { dataset: { bus } } : null) } });
if (!GPEL.onclick) {
  console.log("BAD  #gpus has no delegated click handler, so a short link's badge cannot be opened");
  process.exitCode = 1;
} else if (!shortc.length) {
  console.log("\nno chain is short, so no badge is a button -- nothing to pin (the healthy state, not a fault)");
} else {
  const bus = shortc[0].bus;
  click(bus);
  console.log(`\naccount pinned on ${bus} -- hidden=${DET.hidden}, ${DET.children.length} blocks:`);
  DET.children.forEach((c) => console.log("   |", c.textContent));
  if (DET.hidden || DET.children.length < 5) {
    console.log("BAD  pinning wrote no readable detail"); process.exitCode = 1;
  }
  // Count DOM writes on the pinned lines, then paint again with nothing changed. An account that rewrote every
  // line each second would destroy the text selection the reader is making -- which is the whole point of it.
  let detWrites = 0, sameWrites = 0;
  const proto = Object.getPrototypeOf(new El("i"));
  const dsc = Object.getOwnPropertyDescriptor(proto, "textContent");
  Object.defineProperty(proto, "textContent", { configurable: true, get: dsc.get,
    // A write is only wasteful if it writes the value that is already there. The account's own content moves
    // between repaints ("quiet 7 min" becomes "quiet 8 min", an AER counter lands), and rewriting THOSE lines
    // is correct -- an earlier version of this check asserted "nothing changed" without checking, and failed
    // against a link that put a BadTLP on the wire while it ran.
    set(v) { if (this._det) { detWrites++; if (v === dsc.get.call(this)) sameWrites++; } dsc.set.call(this, v); } });
  DET.children.forEach((c) => { if (c.className === "pcie-detail-line") c._det = true; });
  const marked = DET.children.filter((c) => c._det).length;
  await ctx.refresh();
  console.log(`repainted: ${detWrites} line write(s) across ${marked} pinned lines, ${sameWrites} of them the value that was already there`);
  if (sameWrites > 0) { console.log("BAD  a pinned account rewrites lines to the value they already hold"); process.exitCode = 1; }
  click(bus);
  console.log(`clicked the same card again -> hidden=${DET.hidden}, children=${DET.children.length}`);
  if (!DET.hidden) { console.log("BAD  unpinning left the account open"); process.exitCode = 1; }
  Object.defineProperty(proto, "textContent", dsc);
}
console.log(`\nrequest rows rendered: ${rows.length}`);
for (const r of rows.slice(0, 6)) console.log("  " + r.textContent.replace(/\s+/g, " ").trim());
// Which build this page belongs to: the header names the engine, the plugin and the commit, so a screenshot can
// answer "is the live serve the new one?" without an ssh. The stub DOM creates #brand-line on demand, so this
// checks the JS and the payload; the CSS around it still needs a browser.
const brand = t("brand-line");
// Compare against the payload the PAGE actually drew (the fetch stub clones every /api/status the page makes),
// not against a fresh fetch of our own. On a box that idles between requests two fetches seconds apart disagree
// by more than a 2% ratio allows -- 1.955 against 2.02 tok/s over a 15-minute window -- and that disagreement
// says when we asked, not what the page rendered. A separate fetch also made the check intermittently red on a
// freshly restarted, otherwise quiet box.
const STAT = (FIX && FIX.status) || LAST_STATUS || await (await fetch(`${BASE}/api/status`)).json();
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
const firstNum = (h) => {                       // reads "12.2k" and "1.01M" as the number they stand for
  const m = String(h).match(/[\d,]+(?:\.\d+)?\s*[kKM]?/);
  if (!m) return null;
  const txt = m[0].replace(/,/g, "").trim();
  const mult = txt.endsWith("k") ? 1e3 : txt.endsWith("M") ? 1e6 : 1;
  return Number(txt.replace(/[^\d.]/g, "")) * mult;
};
// vLLM publishes its gauges rounded to three significant figures, so a tight window needs 1.1% rather than 2%;
// a LIVE value also moves between the page's own fetch and ours a second later, so live gets a sanity bound and
// span values, which are stable across that gap, stay tight.
const near = (a, b, tol = 0.011) => (a === null || b === null ? a === b
  : Math.abs(a - b) <= Math.max(0.02, Math.abs(b) * tol));
for (const [id, avgKey, rateKey] of [["rate-gen", "generation_tok_s", "generation_tok_s"],
                                     ["rate-prefill", "prefill_stream_tok_s", "prompt_tok_s"],
                                     ["rate-step", "itl_ms", "itl_ms"]]) {
  // The prefill card's live number is the KV-growth ESTIMATE whenever the counter has not credited yet (it
  // lands at request end), so both are legitimate live values -- they differ by the ~6% the notes measured.
  const want = winShown === "live" ? STAT?.rate?.[rateKey] : STAT?.avg?.[winShown]?.[avgKey];
  // The prefill card's live number is the KV-growth estimate whenever the counter has not credited yet, so both
  // are legitimate live values. A gauge the engine omits entirely (nothing ran in the window) is not a value to
  // compare against, so it drops out of the list rather than turning the comparison into NaN.
  // The prefill card's live number is the KV-growth estimate whenever the counter has not credited yet, and its
  // exact figure is the READ rate (credited minus cache hits), so all three are legitimate live values.
  const wants = (id === "rate-prefill" && winShown === "live"
    ? [want, STAT?.rate?.kv_tok_s, STAT?.rate?.prompt_read_tok_s] : [want]).filter((v) => typeof v === "number" && isFinite(v));
  // A live rate sampled a second apart is not the same number: on a box that idles between requests, 2.0 tok/s
  // and 3.0 tok/s are one second of traffic apart, and a ratio test at three significant figures fails on it.
  // Span values stay tight (0.02) because a span delta is stable across that gap; a live one gets an absolute
  // floor of one token a second as well, which is the resolution of "is it decoding at all".
  const got = firstNum(html(id));
  console.log(`  ${(id + "            ").slice(0, 14)} ${String(html(id)).replace(/<[^>]*>/g, "").trim()}`
              + `   [${winShown} wants ${want === undefined ? "-" : want}]`);
  const tol = winShown === "live" ? 0.30 : 0.02;
  // The tolerance can never be finer than the resolution the page is ALLOWED to print. These cards show at most
  // one decimal at small magnitude, so "2" is the correct rendering of 1.9554 -- and a pure ratio test fails on
  // it (2.28%). It did fail, on a launch restarted into an idle box, where the 15-minute average decode rate was
  // 1.96 tok/s: a magnitude this harness had never been pointed at, because every launch it had watched was busy.
  const printed = String(html(id)).replace(/<[^>]*>/g, "").match(/[\d,]+(?:\.(\d+))?/);
  const tolAbs = Math.max(0.02, 0.5 * 10 ** -(printed ? (printed[1] || "").length : 1));
  const okLive = (w) => Math.abs(got - w) <= Math.max(tolAbs, Math.abs(w) * tol)
    || (winShown === "live" && Math.abs(got - w) <= 1.0);
  if (wants.length && !wants.some(okLive)) {
    console.log(`BAD  ${id} shows ${got} for a ${winShown} window whose values are ${wants.join(" / ")}`);
    process.exitCode = 1;
  }
  // "Is it printing this poll instead of the window?" only means something when the two could print differently.
  // On an idle box this fired on 0.0088 tok/s (178 samples, 8 tokens across 904 s) versus a 0.0 live poll: the
  // 25% ratio calls that a magnitude apart, the card is allowed to round both to the same one decimal, and the
  // check above already accepted the rendering. Same lesson as the tolerance fix above, one clause further in.
  if (winShown !== "live" && isnum(STAT?.rate?.[rateKey]) && Math.abs(STAT.rate[rateKey] - want) > Math.abs(want) * 0.25
      && Math.abs(STAT.rate[rateKey] - want) > tolAbs)
    if (near(got, STAT.rate[rateKey])) {
      console.log(`BAD  ${id} is printing this poll (${STAT.rate[rateKey]}) under a ${winShown} label`); process.exitCode = 1;
    }
}
// The decode card that shows a COUNT: the whole claim of that card is that it is the engine's own cumulative
// counter, summed over streams, with no division anywhere -- so the number on screen has to BE that counter.
// dnum() rounds a rate to the nearest integer (1,012,345 prints as 1,012,345 but 21.49 prints as 21), so the
// comparison against the engine's own counter needs an absolute tolerance as well as a ratio: on a 3 tok/s
// afternoon, rounding alone is a third of the value.
{
  const want = STAT?.since?.generation_tokens_total;
  const got = firstNum(html("rate-gentot"));
  console.log(`  rate-gentot      ${String(html("rate-gentot")).replace(/<[^>]*>/g, "").trim()}`
              + `   [generation_tokens_total is ${want === undefined ? "-" : want}]`);
  if (typeof want === "number" && !near(got, want, 0.005)) {
    console.log(`BAD  rate-gentot shows ${got} but the engine's counter is ${want}`); process.exitCode = 1;
  }
  if (typeof want === "number" && /tok\/s/.test(html("rate-gentot"))) {
    console.log("BAD  the count card carries a per-second unit; it is a count, not a rate"); process.exitCode = 1;
  }
}
// The basis a card CLAIMS, read off monitor.html: the `.label` div that sits above this element in the same card.
// The check below is on the pair (label, number), because this card has been wrong in both directions -- a
// per-stream figure printed under the words "summed, decode time", and before that a per-stream mean divided by the
// running gauge -- and a check on either half alone passes when the two disagree.
const cardLabel = (id) => {
  const h = readFileSync(path.join(WEB, "monitor.html"), "utf8");
  const i = h.indexOf(`id="${id}"`);
  if (i < 0) return "";
  const j = h.lastIndexOf('class="label"', i);
  if (j < 0) return "";
  return h.slice(h.indexOf(">", j) + 1, h.indexOf("</div>", j)).trim();
};
// The card's headline is the SUMMED figure: generated tokens per WALL second -- the throughput reading, and the
// same basis the summed waveform plots. Its note line carries the other reading and the bridge between them:
// `N tok/stream-s ×M`, where N is output per DECODE second and M is streams decoding per wall second, and
// N x M = the big number exactly. A percent on that line is a category error: M is stream-seconds per wall second
// (0.15 serialized, 2.60 loaded), not a share of the span.
{
  const line = t("note-gen") || "";
  const b = STAT?.avg?.[winShown] || {};
  const wantStream = winShown === "live" ? STAT?.rate?.decode_stream_tok_s : b.decode_stream_tok_s;
  const wantStreams = winShown === "live" ? STAT?.rate?.decode_streams : b.decode_streams;
  const fm = line.match(/^([\d.,kKM]+)\s*tok\/stream-s\s*[\u00d7x]([\d.,]+)/);
  const gotStream = fm ? firstNum(fm[1]) : null;
  const gotStreams = fm ? Number(fm[2].replace(/,/g, "")) : null;
  const noStep = /no decode step/.test(line);
  console.log(`  note-gen         ${line}` + (wantStream == null ? "   [no decode stream-seconds in this payload]"
              : `   [${wantStream} tok/stream-s, x${wantStreams}, summed to ${b.generation_tok_s}]`));
  if (wantStream == null && !noStep) {
    console.log("BAD  the payload has no decode stream-seconds, so the note line has to say so rather than print a "
                + "rate with no denominator"); process.exitCode = 1;
  }
  if (typeof wantStream === "number" && !near(gotStream, wantStream, 0.02)) {
    console.log(`BAD  note-gen shows ${gotStream} tok/stream-s for a per-stream-second rate of ${wantStream}`)
    ; process.exitCode = 1;
  }
  if (typeof wantStreams === "number" && Math.abs(gotStreams - wantStreams) > Math.max(0.011, 0.02 * wantStreams)) {
    console.log(`BAD  note-gen shows x${gotStreams} decoding for a fan-out of ${wantStreams}`); process.exitCode = 1;
  }
  // The three numbers on the card have to be one arithmetic: per stream-second x fan-out = the summed headline.
  const shown = firstNum(html("rate-gen"));
  if ([gotStream, gotStreams, shown].every((v) => typeof v === "number")
      && Math.abs(gotStream * gotStreams - shown) > Math.max(0.05, 0.03 * Math.abs(shown))) {
    console.log(`BAD  the card's numbers are not one arithmetic: ${gotStream} x ${gotStreams} != ${shown}`)
    ; process.exitCode = 1;
  }
  if (/% dec/.test(line)) {
    console.log("BAD  note-gen prints a share of the span where the fan-out belongs -- a percent has no meaning "
                + "for stream-seconds per wall second"); process.exitCode = 1;
  }
  // The fan-out has two causes, and the span is supposed to publish the split: seconds with nothing in flight
  // (busy_frac) and seconds with several streams in their decode phase (streams_while_busy). The product is the
  // fan-out on the line; if the payload measures the factors, the card has to state them, and if it states them
  // they have to multiply back. This is the answer to the smallest number on the card being smaller than the
  // engine's own log line -- the log skips idle intervals (debug level when idle), so its lines average only the
  // busy ones: 42.4 tok/s printed against 19.7 over the wall of the same 903 s on 2026-11-04.
  if (typeof wantStreams === "number" && typeof b.busy_frac === "number" && typeof b.streams_while_busy === "number") {
    const prod = b.busy_frac * b.streams_while_busy;
    if (Math.abs(prod - wantStreams) > Math.max(0.011, 0.02 * wantStreams)) {
      console.log(`BAD  this span's busy_frac x streams_while_busy is ${prod.toFixed(3)} but its decode_streams is `
                  + `${wantStreams} -- the split does not account for the fan-out`); process.exitCode = 1;
    }
    const ti = (byId.get("note-gen") || {}).title || "";
    if (!/busy x [\d.]+ streams/.test(ti)) {
      console.log("BAD  the span measures busy_frac and streams_while_busy, and the card never states the split");
      process.exitCode = 1;
    }
  }
  // The label and the number have to be the same claim. The card has been wrong in both directions -- a per-stream
  // figure under "summed", and a per-stream mean divided by the running gauge -- so the check is on the pair: the
  // word in the label picks which payload field the big number has to be. On a serialized burst the two differ by
  // 7x (22 summed against 152 per stream-second), so a mismatch is not a rounding argument.
  const label = cardLabel("rate-gen");
  if (winShown !== "live" && isnum(b.generation_tok_s) && isnum(b.decode_stream_tok_s)
      && Math.abs(b.generation_tok_s - b.decode_stream_tok_s) > b.decode_stream_tok_s * 0.25) {
    const wantsSummed = /summed/.test(label);
    const wrong = wantsSummed ? (!near(shown, b.generation_tok_s, 0.02) && near(shown, b.decode_stream_tok_s, 0.02))
                             : (!near(shown, b.decode_stream_tok_s, 0.02) && near(shown, b.generation_tok_s, 0.02));
    if (wrong) {
      console.log(`BAD  rate-gen shows ${shown} under the label "${label}" -- that label wants `
                  + `${wantsSummed ? b.generation_tok_s : b.decode_stream_tok_s} (the other basis is `
                  + `${wantsSummed ? b.decode_stream_tok_s : b.generation_tok_s})`); process.exitCode = 1;
    }
  }
}
// The spans the headline is not, plus this poll.
console.log("\nthe other spans, and this poll, under each card");
for (const [id, key, rateKey] of [["spans-gentot", "generation_tokens", "generation_tokens"],
                                  ["spans-gen", "generation_tok_s", "generation_tok_s"],
                                  ["spans-prefill", "prefill_stream_tok_s", "prompt_tok_s"],
                                  ["spans-step", "itl_ms", "itl_ms"]]) {
  const line = t(id) || "";
  // spans-gentot shows two spans, not three: with the digits written out three signed counts are 30 characters and
  // the
  // tightest 4-up slot holds 24 (the arithmetic is in notes/webui.md), so 1m lives in the title instead.
  const shown = Object.entries((STAT?.avg || {})).filter(([l, b]) => isnum(b?.[key]) && l !== winShown
              && !(id === "spans-gentot" && l === "1m"));
  console.log(`  ${(id + "            ").slice(0, 15)} ${line}`
              + (shown.length ? `   [${shown.map(([l, b]) => `${l}=${b[key].toFixed(1)}/${Math.round(b.span_s)}s`).join(" ")}]`
                              : "   [no other span has this field]"));
  for (const [lbl, b] of shown) {
    const short = b.span_s < AVG_W[lbl] * 0.8;
    // A span may be left out for either reason the page states in its own comment: the two windows are the same
    // window (a launch 16 minutes old has a 900 s "15m" and a 960 s "1h" holding the same count), or the line with
    // EVERY span in it would not fit the tightest 4-up column -- which is the page's own rule, so the width test
    // has to measure the line the page would have printed, not the subset this loop happens to iterate. Either way
    // the figures that ARE printed have to match the payload, and a span that is dropped for width has to be named
    // in the slot's title rather than vanish; that is what the rest of this loop checks.
    const allSpans = Object.entries((STAT?.avg || {})).filter(([, bb]) => isnum(bb?.[key]));
    const widestLine = allSpans.map(([l2, b2]) => `${l2} ${Math.round(b2[key]).toLocaleString("en-US")}`)
                                .join(" \u00b7 ");
    const tooWide = widestLine.length > 24;
    const redundant = shown.some(([l2, b2]) => l2 !== lbl && isnum(b2[key])
      && Math.abs(b2[key] - b[key]) <= Math.max(1, 0.02 * Math.abs(b2[key]))
      && Math.abs(b2.span_s - b.span_s) <= 0.15 * Math.max(b2.span_s, b.span_s));
    if (!short && !line.includes(lbl + " ") && !redundant && !tooWide) {
      console.log(`BAD  ${id}: ${lbl} has ${b[key]} over ${Math.round(b.span_s)}s and the line omits it`
                  + ` (nothing here is redundant, and ${widestLine.length} characters of spans do fit 24)`)
        ; process.exitCode = 1;
    }
    if (!short && !line.includes(lbl + " ") && !redundant && tooWide) {
      const ti = ((byId.get(id) || new El()).title || "");
      if (!ti.includes(lbl + " ")) {
        console.log(`BAD  ${id}: dropped ${lbl} because ${widestLine.length} characters do not fit 24, but the`
                    + " title never names it -- a span dropped for width is only honest if it is still said");
        process.exitCode = 1;
      }
    }
    if (short && line.includes(lbl + " ")) {
      console.log(`BAD  ${id}: prints ${lbl} for a span of ${Math.round(b.span_s)}s -- a young window wearing an old label`); process.exitCode = 1;
    }
  }
  const poll = t(id.replace("spans-", "poll-")) || "";
  console.log(`  ${(id.replace("spans-", "  poll-") + "          ").slice(0, 15)} ${poll}`);
  if (winShown !== "live" && !/^poll /.test(poll)) {
    console.log(`BAD  the card has no this-poll slot (${id}); the average is all the reader gets`); process.exitCode = 1;
  }
}
// Every slot: one line, and short enough for the tightest card it can be laid out in. The widths are the same
// arithmetic docs/monitor-cards.py and tests/test_readme.py use: main is capped at 1560 with 32 px of padding,
// the row is 4 columns over 14 px gaps, a card gives up 42 px to padding and border, and the theme lays text out
// at 0.60 em per glyph. A 4-up row only exists down to 981 px of viewport, so that is the tight case; a slot
// that does not fit there ellipsizes (its full text is in the title), but the copy is written to fit it.
const CHAR = 0.60, FS = 12;
const inner = (W) => (Math.min(W, 1560) - 64 - 3 * 14) / 4 - 42;
const TIGHT = Math.floor(inner(981) / (FS * CHAR)), WIDEST = Math.floor(inner(1560) / (FS * CHAR));
// The slots are whatever the page declares, so the list cannot drift out of date with the markup.
const SLOTS = [...readFileSync(path.join(WEB, "monitor.html"), "utf8")
  .matchAll(/class="([^"]*)"[^>]*?\bid="([^"]+)"/g)]
  .filter(([, cls]) => cls.split(/\s+/).includes("slot")).map(([, , id]) => id);
console.log(`\nslots (the tightest 4-up card holds ${TIGHT} characters at ${FS} px; the widest holds ${WIDEST})`);
// Two invariants over the whole slot set, checked while printing rather than in one pass afterwards:
//  * a number is printed as a NUMBER. dnum() used to abbreviate ("7.4k" for 7,428) to hold the width, and asked
//    on 2026-11-04 the digits came back: the width is now held by the copy and measured here. A slot that prints
//    "12.3k" or "4.8M" is a regression against that request, whatever function produced it, so the check is on
//    the rendered text rather than on which helper the page calls.
//  * a negative in a rate caption is the one thing a reader most misreads. The KV-growth series goes negative
//    when finished requests free blocks, which is a fact about the pool, not a prefill running backwards; the
//    caption says "now draining N" instead of "now -N" and the sign stays in the drawn waveform.
const ABBREV = /\d(?:[.,]\d+)?[kM]\b/;
for (const id of SLOTS) {
  const line = t(id);
  if (line === null) { console.log(`BAD  no slot called ${id} was written`); process.exitCode = 1; continue; }
  const flag = line.length > TIGHT ? `  ${line.length > WIDEST ? "BAD" : "wide"} (${line.length} > ${TIGHT})` : "";
  console.log(`  ${(id + "              ").slice(0, 16)} ${line}${flag}`);
  const ab = line.match(ABBREV);
  if (ab) {
    console.log(`BAD  ${id} abbreviates a number ("${ab[0]}"); the page prints digits and holds its width by copy`);
    process.exitCode = 1;
  }
  if (/-\d/.test(line) && !/draining|saved|\u2212/.test(line)) {
    console.log(`BAD  ${id} shows a bare signed number ("${line.match(/-\d[^ \u00b7]*/)[0]}") where the caption `
                + `should say what is negative about it`);
    process.exitCode = 1;
  }
  if (line.length > WIDEST) {
    console.log(`       it does not fit the WIDEST card either -- this one wraps, and the card's height moves`);
    process.exitCode = 1;
  }
}
// Whether the prefix cache is ON is a property of the launch, and the page has to answer it: a silent cache
// clause reads the same whether the feature is off, on-and-missed, or unmeasured. So the engine's own config line
// (the same string its boot line prints) drives the copy -- and when it says OFF, the slot must say so and must not
// print a hit percent, because 0% of nothing is not a measurement.
{
  const pc = String(((STAT.config || {}).enable_prefix_caching ?? "")).toLowerCase();
  const line = t("note-prefill");
  console.log(`\nprefix cache (engine config): ${pc || "not published"} -- note line: ${line}`);
  if (pc === "false" || pc === "0") {
    if (!/cache off/.test(line)) {
      console.log(`BAD  the engine says the prefix cache is off and the prefill note line does not say so (${line})`);
      process.exitCode = 1;
    }
    if (/% cache/.test(line)) {
      console.log(`BAD  a cache percent printed while the feature is off (${line})`); process.exitCode = 1;
    }
  } else if (pc === "true" || pc === "1") {
    if (/cache off/.test(line)) {
      console.log(`BAD  the note line claims the cache is off while the engine says it is on (${line})`);
      process.exitCode = 1;
    }
  }
}

// The trend rings: 90 s of per-poll points, or an hour of 15 s interval rates. The caption's span has to come
// from the samples, not from points x poll, which is the thing that would otherwise quietly lie.
console.log("\ntrend source");
const fastNote = t("trend-note");
console.log(`  fast ring: ${fastNote}`);
byId.get("trend-slow")?.onclick();
const slowNote = t("trend-note");
const pressedFast = byId.get("trend-fast")?.attrs["aria-pressed"];
const pressedSlow = byId.get("trend-slow")?.attrs["aria-pressed"];
console.log(`  slow ring: ${slowNote}  (aria-pressed fast=${pressedFast} slow=${pressedSlow})`);
if (slowNote === fastNote) {
  console.log("BAD  pressing the 1 h trend button changed nothing about what the row says"); process.exitCode = 1;
}
if (pressedSlow !== "true" || pressedFast !== "false") {
  console.log("BAD  the buttons' aria-pressed does not follow the ring actually in use"); process.exitCode = 1;
}
byId.get("trend-fast")?.onclick();
console.log(`  back to fast: ${t("trend-note")}`);
if (t("trend-note") !== fastNote) {
  console.log("BAD  pressing the 90 s trend button did not restore the fast ring"); process.exitCode = 1;
}
console.log(`  caption slots now: ${t("gen-note1")} | ${t("gen-note2")}`);

// -- the admission gate row (serve/gateway.py's GET /api/gate) ------------------------------------
// Two things worth a check, both learned from the sidecar row above: the row is there and FILLED when the route
// answers, and it is HIDDEN -- not two cards of em-dashes -- when it does not (the page is also reachable on the
// engine's own port, where no gateway is in front of it). And the words it prints are the GATEWAY's: its reason
// string already carries the token count the gate is protecting, so a second copy of that arithmetic in the page
// is what would go out of date.
{
  // The payload the PAGE was fed (the fetch stub clones every /api/gate the page asks for), not a fetch of our
  // own: on a freshly restarted gateway the page paints with one poll of history while a second fetch seconds
  // later has a dozen, and that disagreement says when we asked, not what the page rendered. Same lesson the
  // brand-line check above learned. It also answers "does this route exist on this port?" with what the page
  // actually saw, which is the question the row's hiding rule is about.
  const GPAY = LAST_GATE;
  const sec = byId.get("gate") || new El();
  const g2 = (id) => t(id).replace(/\s+/g, " ").trim();
  // the sparkline element holds ONE svg child; the paths are inside it. (Counting the sparkline's own children
  // would say 1 for every spark on the page, which is what this check first did.)
  const sparkEl = (byId.get("gate-spark") || new El());
  const strip = (((sparkEl.children || [])[0]) || new El()).children.length;
  console.log(`\ngate row: hidden=${sec.hidden} | ${g2("gate-state")} | ${g2("gate-why")}`
              + ` | ${g2("gate-level")} | held ${g2("gate-held")} | ${g2("gate-counters")} | ${g2("gate-ceil")}`
              + ` | strip ${strip} node(s), ${GPAY?.hist?.length ?? 0} polls`
              + (Number.isFinite(GPAY?.hist_span_s) ? ` over ${Math.round(GPAY.hist_span_s / 60)} min` : "")
              + " of history");
  if (GPAY && sec.hidden) {
    console.log("BAD  /api/gate answers and the row is hidden -- that row is the only record a held request has");
    process.exitCode = 1;
  }
  if (!GPAY && !sec.hidden) {
    console.log("BAD  /api/gate does not answer and the row is showing anyway (em-dashes left in the grid)");
    process.exitCode = 1;
  }
  if (GPAY) {
    for (const id of ["gate-state", "gate-why", "gate-level", "gate-held", "gate-counters", "gate-ceil"]) {
      if (!g2(id) || g2(id) === "\u2014") {
        console.log(`BAD  #${id} still holds the markup placeholder while the row is visible`); process.exitCode = 1;
      }
    }
    const want = GPAY.config?.GATE_ON ? (GPAY.closed ? "Held" : "Open") : "Off";
    if (g2("gate-state") !== want) {
      console.log(`BAD  the gate card says "${g2("gate-state")}" for a payload whose state is "${want}"`);
      process.exitCode = 1;
    }
    // The gateway's reason is not reformatted by the page: the visible line is its FIRST CLAUSE (the tight 4-up
    // card holds 24 characters), and the exact full string -- the one carrying the token count the gate is
    // protecting -- is the element's title. Copying that arithmetic into the page's own words is what goes stale.
    if (GPAY.closed && GPAY.why) {
      const clause = GPAY.why.split(";")[0].split(" (")[0].trim();
      if (Number.isFinite(GPAY.thrash_left) && GPAY.thrash_left > 0) {
        // The gateway's preemption clause is a sentence ("7 preemption(s) observed, door shut 12 s more") and the
        // tightest 4-up card holds 24 characters, so the page shortens it to the two facts that matter -- what is
        // closing the door, and how much longer -- and keeps the gateway's exact wording in the title.
        if (!/^preempting, \d+ s more$/.test(g2("gate-why"))) {
          console.log(`BAD  the page's shut-reason for a preemption window is "${g2("gate-why")}"; it must name `
                      + "the preemption axis and the remaining seconds inside 24 characters");
          process.exitCode = 1;
        }
      } else if (g2("gate-why") !== clause) {
        console.log(`BAD  the page rewrote the gate's reason: "${g2("gate-why")}" wants the clause "${clause}"`);
        process.exitCode = 1;
      }
      const ti = ((byId.get("gate-why") || new El()).title || "").trim();
      if (!ti.startsWith(GPAY.why.trim())) {
        console.log(`BAD  #gate-why's title does not carry the gateway's own string: "${ti.slice(0, 90)}"`);
        process.exitCode = 1;
      }
    }
    const ctr = ((byId.get("gate-counters") || new El()).title || "");
    const hasSmall = Number.isFinite(GPAY.counters?.small_passed), hasOver = Number.isFinite(GPAY.counters?.held_over);
    // expired and small are the two counts that decide whether the door is doing anything at all: expired against
    // held says whether the ceiling is a bound or just a delay, and small says how many cheap requests had to be
    // let through a shut door (which is the sign a pool threshold was starving the wrong traffic). An older
    // gateway reports neither and its page prints released/expired instead, so what is required follows the
    // payload rather than the page.
    for (const k of ["expired", hasSmall ? "small" : "released"]) {
      if (!new RegExp(k + " \\d").test(g2("gate-counters"))) {
        console.log(`BAD  the counters line does not carry ${k}: "${g2("gate-counters")}"`); process.exitCode = 1;
      }
    }
    // the counters the tight line drops have to be stated in the gloss, or they are just hidden numbers
    for (const k of ["released", "refused", "admitted", "held"]
             .concat(hasSmall ? ["small"] : []).concat(hasOver ? ["over"] : [])) {
      if (!new RegExp(k + " \\d").test(ctr)) {
        console.log(`BAD  the counters gloss never states ${k} -- a number hidden from the line has to be in the title`);
        process.exitCode = 1;
      }
    }
    // the ceiling line names the two timers the door runs on. The window is the one that lets the door re-open
    // under a long batch; a pool release line alone could not (2026-11-15: 180 of 180 polls shut). It is only
    // required when the gateway in front of the page reports one -- an older gateway does not, and its page
    // degrades to the poll interval on purpose.
    for (const k of Number.isFinite(GPAY.config?.GATE_PREEMPT_S) ? ["hold", "win"] : ["hold"]) {
      if (!new RegExp(k + " \\d").test(g2("gate-ceil"))) {
        console.log(`BAD  the ceiling line does not name ${k}: "${g2("gate-ceil")}"`); process.exitCode = 1;
      }
    }
    // the strip draws a base line even with no history, so a visible row must never have an empty spark box
    if (!strip) {
      console.log("BAD  the gate's history strip drew nothing"); process.exitCode = 1;
    }
    if ((GPAY.hist || []).length >= 2 && strip < 3) {
      console.log(`BAD  ${GPAY.hist.length} polls of history and the strip drew ${strip} node(s)`); process.exitCode = 1;
    }
    // A strip is the answer to "was the door shut at 15:05?", and the question needs the span to be checkable: a
    // ring of polls with no stated span is a picture that cannot be lined up against anything.
    if (Number.isFinite(GPAY.hist_span_s)) {
      const sti = (sparkEl.title || "");
      const want = GPAY.hist_span_s >= 120 ? `${Math.round(GPAY.hist_span_s / 60)} min` : `${GPAY.hist_span_s} s`;
      if (!sti.includes(want) || !sti.includes(String((GPAY.hist || []).length))) {
        console.log(`BAD  the history strip does not state its span and poll count (wants "${(GPAY.hist || []).length}`
                    + ` polls over ${want}"); it said "${sti.slice(0, 120)}"`); process.exitCode = 1;
      }
    }
  }
}

// -- the watchdog row: engine state, driver faults, big-prefill exposure, preemptions ----------------
// Same two rules the gate row was taught, for the same reason: the row is FILLED when the payload carries the
// reads, HIDDEN when the sidecar is too old to have them (an em-dash row would read as "no faults here", which
// is a claim about the hardware), and the words are the sidecar's -- the fault count comes out of the kernel
// journal and the rank join comes out of KFD, so a page that re-derives either is a page that can disagree with
// the tool an operator runs by hand at the moment of a crash.
{
  const w = LAST_STATUS || {};
  const sec = byId.get("watchdog") || new El();
  const w2 = (id) => t(id).replace(/\s+/g, " ").trim();
  const cards = t("gpus").replace(/\s+/g, " ");
  console.log(`\nwatchdog row: hidden=${sec.hidden} | ${w2("wd-state")} | ${w2("wd-state-note")}`
              + ` | faults ${w2("wd-faults")} (${w2("wd-fault-note")}) [label: ${w2("wd-fault-label")}]`
              + ` | cards badging faults: ${(cards.match(/page fault(s)? here, this launch/g) || []).length}`
              + ` of ${(w.gpus || []).filter((g) => (g.faults_launch || 0) > 0).length}`
              + ` | over bound ${w2("wd-big")} (${w2("wd-big-note")})`
              + ` | preempted ${w2("wd-preempt")} (${w2("wd-preempt-note")})`);
  const hasReads = Boolean(w.engine_state || w.faults || w.live?.prompt_bound !== undefined);
  if (hasReads && sec.hidden) {
    console.log("BAD  the payload carries the watchdog reads and the row is hidden"); process.exitCode = 1;
  }
  if (!hasReads && !sec.hidden) {
    console.log("BAD  an old sidecar sent no watchdog reads and the row is showing em-dashes anyway");
    process.exitCode = 1;
  }
  if (hasReads) {
    // A placeholder left in a visible card is a bug; an em-dash the PAYLOAD earns is not. On a serve that has
    // finished no requests in the window there is no percentage to print, and inventing a 0 there would be the
    // same lie in the other direction -- so the check asks whether the source has a number before it complains.
    const SRC = { "wd-state": () => Boolean(w.engine_state),
                  "wd-state-note": () => Boolean(w.engine_state),
                  "wd-faults": () => isnum(w.faults?.total) || isnum(w.faults?.events),
                  "wd-fault-note": () => Boolean(w.faults),
                  "wd-big": () => isnum(w.live?.prompt_big_pct),
                  "wd-big-note": () => isnum(w.live?.prompt_bound),
                  "wd-preempt": () => isnum(w.live?.preemptions),
                  "wd-preempt-note": () => isnum(w.live?.preemptions_total) };
    for (const id of Object.keys(SRC)) {
      const line = w2(id);
      if (line && line !== "\u2014") continue;
      if (SRC[id]()) {
        console.log(`BAD  #${id} is empty while its payload source carries a number -- the row is visible and the`
                    + " number exists, so this is the page dropping it");
        process.exitCode = 1;
      }
    }
    const WORD = { alive: "Answering", loading: "Loading", dead: "Gone", wedged: "Wedged" };
    if (w.engine_state && w2("wd-state") !== WORD[w.engine_state]) {
      console.log(`BAD  #wd-state says "${w2("wd-state")}" for engine_state "${w.engine_state}"`
                  + " -- the word has to carry the wedged case, that is the whole reason the row exists");
      process.exitCode = 1;
    }
    if (w.engine_state === "wedged" && !/rank\(s\) still hold/.test(w2("wd-state-note"))) {
      console.log(`BAD  a wedged payload whose note does not say the ranks still hold the GPUs: "${w2("wd-state-note")}"`);
      process.exitCode = 1;
    }
    // The wedged state is the one this row exists for, and a reader who has never seen it needs the command in
    // reach -- it lives in the title, since the slot itself is 24 characters wide.
    if (w.engine_state === "wedged" && !/STOP=1/.test(((byId.get("wd-state-note") || new El()).title || ""))) {
      console.log("BAD  the wedged note does not carry the command that clears it"); process.exitCode = 1;
    }
    // The headline is about the LAUNCH being described, not the boot: attributed -> now.events, unattributed ->
    // the since-boot number with the note saying it could not be attributed. A standing hardware alert takes the
    // card over with its OWN number: the alert's claim is about a PCI function over a week, which is a different
    // question from "did the launch on screen fault", and the label says which one is being answered.
    const fa = w.faults || {};
    const al = w.alert && w.alert.card ? w.alert : null;
    const fwant = al ? al.count : (fa.now ? fa.now.events : (isnum(fa.events) ? fa.events : fa.total));
    if ((w.faults || al) && fwant !== null && w2("wd-faults") !== String(fwant).replace(/\B(?=(\d{3})+(?!\d))/g, ",")) {
      console.log(`BAD  #wd-faults says "${w2("wd-faults")}" for a payload with ${fwant} fault event(s)`
                  + ` (boot: ${fa.total} lines${fa.now ? `, this launch: ${fa.now.total} lines` : ", unattributed"})`
                  + (al ? " -- a standing hardware alert owns this card" : " -- the headline is this launch, "
                                                                          + "the history belongs in the note"));
      process.exitCode = 1;
    }
    if (al && !w2("wd-fault-note").includes(al.card)) {
      console.log(`BAD  the hardware alert stands and the note does not name the function it names (${al.card}): `
                  + `"${w2("wd-fault-note")}"`); process.exitCode = 1;
    }
    if (al && !(((byId.get("wd-fault-note") || new El()).title || "").includes("swap"))) {
      console.log("BAD  the hardware alert does not carry the action (swap the card and see whether the faults "
                  + "follow): an alert without a next step is a warning light"); process.exitCode = 1;
    }
    for (const [bus, n] of Object.entries(al ? {} : (w.faults || {}).by_card || {})) {
      const noteAll = w2("wd-fault-note") + " " + (((byId.get("wd-fault-note") || new El()).title) || "");
      if (!noteAll.includes(bus) || !noteAll.includes(`\u00d7${n}`)) {
        console.log(`BAD  the fault note does not name ${bus} ${n}: "${w2("wd-fault-note")}" -- per-card, because`
                    + " the card is what the driver named and the rank is what the engine named (§15)");
        process.exitCode = 1;
      }
      const ti = ((byId.get("wd-fault-note") || new El()).title || "");
      if (w.faults.last && w.faults.last.at && !ti.includes(w.faults.last.at)) {
        console.log("BAD  the fault WHEN is nowhere: the slot is 24 characters by design, so the timestamp and "
                    + "pid have to be in the title");
        process.exitCode = 1;
      }
    }
    // The card itself carries its own count (common.js gpuCard): the row's total is for the eye, the card is for
    // the question "which one of these four".
    for (const [bus, n] of Object.entries((w.faults || {}).by_card || {})) {
      const g = (w.gpus || []).find((x) => String(x.bus || "").endsWith(bus));
      if (!g) {
        console.log(`BAD  the fault note names ${bus} and no card row carries that bus id -- the count has`
                    + " nowhere to land on the card, which is where the §15 question gets asked");
        process.exitCode = 1;
      }
    }
    // A fault line on the card ONLY for the cards that faulted in the launch being described. A [gfxhub] page
    // fault is a rank dying: the faults of a launch that is over describe a machine that no longer exists, so a
    // badge that never clears is a badge the reader learns to ignore. #gpus is one rebuilt blob, so the count of
    // lines is what is checkable from outside; the join itself is checked in tests/test_webui.py.
    { const want = (w.gpus || []).filter((g) => (g.faults_launch || 0) > 0).length;
      const got = (cards.match(/page fault(s)? here, this launch/g) || []).length;
      if (want !== got) {
        console.log(`BAD  ${want} card(s) faulted in this launch per the payload and ${got} card line(s) were drawn`);
        process.exitCode = 1; }
      // (the equality above is what catches the page reading `faults` -- the since-boot number -- where it
      // should read `faults_launch`: more cards would light up than the launch justifies)
    }
    if (!al && (w.faults || {}).last && w.faults.last.card
        && !((w2("wd-fault-note") + " " + (((byId.get("wd-fault-note") || new El()).title) || ""))
             .includes(w.faults.last.card))) {
      console.log(`BAD  the last fault's card is missing from the note: "${w2("wd-fault-note")}"`);
      process.exitCode = 1;
    }
    // The label must name the SAME population the number comes from. A card headed "this launch" carrying a
    // since-boot count is the misreading this whole split exists to prevent; a card with a standing hardware
    // alert must say so, because its number is neither launch nor boot but a function over a window.
    const lbl = w2("wd-fault-label");
    // With an alert the label also carries whether a HUMAN was told: "Hardware: page faults" alone would let a
    // refused webhook (wrong keyword, dead token) look like a working alarm for as long as the card stayed red.
    const wantLbl = al ? /^Hardware: page faults( \u00b7 (pinged|NOT pinged|untested|no channel))?$/
                       : (w.faults?.now ? /this launch/ : /this boot/);
    if (lbl && !wantLbl.test(lbl)) {
      console.log(`BAD  the card is headed "${lbl}" while the payload's numbers are `
                  + `${al ? `a hardware alert on ${al.card} (${al.count} events / ${al.window_s} s)`
                       : (w.faults?.now ? "for this launch" : "since boot, unattributed")}`);
      process.exitCode = 1;
    }
    if (al) {
      const _note = ((byId.get("wd-fault-note") || new El()).title || "");
      if (al.notify) {
        const want2 = al.notify.sent ? /accepted by the robot/ : /REFUSED by the robot/;
        if (!want2.test(_note)) {
          console.log("BAD  the alert records a send (" + (al.notify.sent ? "accepted" : "refused, errcode "
                      + al.notify.errcode) + ") and the card's note does not say so: \""
                      + _note.slice(-160) + "\"");
          process.exitCode = 1; }
        if (al.notify.errcode && !_note.includes(String(al.notify.errcode))) {
          console.log(`BAD  a refused send is described without the robot's errcode ${al.notify.errcode}, which is `
                      + "the only part of the answer that says WHY");
          process.exitCode = 1; }
      } else if (!/No send attempt recorded/.test(_note)) {
        console.log("BAD  an alert with no recorded send attempt must say that, not leave the delivery to be "
                    + "guessed");
        process.exitCode = 1; }
    }
    { const dcls = ((byId.get("wd-fault-dot") || new El()).className || "");
      if (al && !/dot--alert/.test(dcls)) {
        console.log(`BAD  a hardware alert stands and the fault dot is "${dcls}" -- the row has to change `
                    + "colour, not just wording");
        process.exitCode = 1; }
      if (!al && /dot--alert/.test(dcls)) {
        console.log("BAD  the fault dot is alert-coloured with no standing alert in the payload");
        process.exitCode = 1; }
    }
    if (!w.faults && !/unknown/.test(w2("wd-faults"))) {
      console.log(`BAD  no journal read in the payload and #wd-faults says "${w2("wd-faults")}", not "unknown"`);
      process.exitCode = 1;
    }
    // The rank join is drawn on the card, not in a table elsewhere: the question is asked card by card.
    for (const g of w.gpus || []) {
      if (!g.rank) continue;
      const who = `rank ${String(g.rank).replace(/^Worker_/, "")}`;
      if (!cards.includes(who)) {
        console.log(`BAD  card ${g.bus} reports ${g.rank} in the payload and the card does not say "${who}"`);
        process.exitCode = 1;
      }
    }
    if (isnum(w.live?.prompt_bound)
        && !w2("wd-big-note").replace(/,/g, "").includes(String(Math.round(w.live.prompt_bound)))) {
      console.log(`BAD  the exposure card hides the bound its percentage was taken at: "${w2("wd-big-note")}"`);
      process.exitCode = 1;
    }
  }
}

// The page swallows render errors into its notice bar, so a broken section can look like an empty one.
const bar = t("error");
if (bar) { console.log("\nPAGE NOTICE:", bar); process.exitCode = 1; }
if (!rows.length) { console.log("\nno request rows rendered"); process.exitCode = 1; }
console.log(`\ncards the page wrote a value into: ${shown.sort().join(" ")}`);
