#!/usr/bin/env node
// tests/render-about.mjs -- run serve/web/about.js unmodified against a payload, and check the parts of the page
// that are pure text. About has never had this: test_webui.py parses the file and pins its strings, which catches a
// syntax error but not a ReferenceError, and a ReferenceError inside one section of About leaves that section empty
// with no error anywhere (the same failure mode notes/webui.md records for the Monitor, where a `${unit}` in a
// function with no `unit` parameter blanked a card).
//
//   tests/render-about.mjs http://127.0.0.1:8099      # a running Monitor
//   tests/render-about.mjs /tmp/status.json           # a saved /api/status payload
//
// Exit 0 when the page rendered and every claim below held.
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.join(HERE, "..", "serve", "web");
const ARG = process.argv[2] || "http://127.0.0.1:8099";
const FIX = ARG.endsWith(".json") ? JSON.parse(readFileSync(ARG, "utf8")) : null;
// A saved payload may be the bare /api/status JSON or one wrapped in {status: ..., host: ...} (what the page's own
// dump writes). Normalize once, so the PAGE and the CHECKS below are looking at the same object -- reading
// `FIX.status` in one place and `FIX` in another is how a fixture silently renders as an empty status.
const PAY = FIX ? (FIX.status || FIX) : null;
const BASE = FIX ? "" : ARG.replace(/\/$/, "");

class El {
  constructor(tag = "div", id = "") {
    this.tagName = tag; this.id = id; this._text = ""; this.innerHTML = "";
    this.className = ""; this.dataset = {}; this.attrs = {}; this.children = [];
    this.style = {}; this.hidden = false; this.value = ""; this.disabled = false;
    this.classList = { _s: new Set(), add(...c) { c.forEach((x) => this._s.add(x)); },
                       remove(...c) { c.forEach((x) => this._s.delete(x)); },
                       toggle(c) { this._s.has(c) ? this._s.delete(c) : this._s.add(c); },
                       contains(c) { return this._s.has(c); } };
  }
  get textContent() {
    if (this.children.length && !this._text) return this.children.map((c) => c.textContent).join(" ");
    if (!this._text && this.innerHTML) {                       // facts() writes HTML strings
      return String(this.innerHTML).replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim();
    }
    return this._text;
  }
  set textContent(v) { this._text = String(v); this.children = []; this.innerHTML = ""; }
  append(...n) { n.flat().forEach((x) => this.children.push(x)); }
  replaceChildren(...n) { this.children = n.flat(); }
  appendChild(n) { this.append(n); return n; }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return this.attrs[k] ?? null; }
  querySelectorAll() { return []; }
  querySelector() { return new El("div"); }                   // copy targets: About sets dataset on these
  focus() {} remove() {} click() {}
}

const byId = new Map();
const document = {
  activeElement: null, hidden: false, documentElement: new El("html"), body: new El("body"),
  head: { appendChild: () => {} },
  createElement: (t) => new El(t),
  getElementById: (id) => { if (!byId.has(id)) byId.set(id, new El("div", id)); return byId.get(id); },
  querySelectorAll: () => [], querySelector: () => new El("div"), addEventListener: () => {},
};
const store = () => { const m = new Map(); return { getItem: (k) => (m.has(k) ? m.get(k) : null),
  setItem: (k, v) => m.set(k, v), removeItem: (k) => m.delete(k) }; };
const ctx = {
  document, console, Promise, JSON, Math, Date, Number, String, Object, Array, RegExp, Set, Map, Error,
  AbortController, AbortSignal, isNaN, parseInt, parseFloat, isFinite, encodeURIComponent,
  decodeURIComponent, Intl, localStorage: store(), sessionStorage: store(),
  navigator: { clipboard: { writeText: async () => {} }, userAgent: "render-about" },
  location: { origin: FIX ? "http://127.0.0.1:8099" : BASE },
  fetch: async (u, o) => {
    if (!FIX) return globalThis.fetch(BASE + u, o);
    const p = String(u).split("?")[0];
    // /api/host is a separate endpoint: the merged serve answers it, and a bare /api/status payload saved from one
    // does not carry the block. In file mode that means 'no host facts in this fixture', not 'the endpoint is down',
    // so hand back an empty object. URL mode still passes the real 404 through, which is what caught the page's
    // habit of treating one as fatal.
    const body = p === "/api/status" ? PAY
      : p === "/api/host" ? ((FIX && (FIX.host ?? (PAY ? {} : null))) ?? null) : null;
    if (body === null) return { ok: false, status: 404, json: async () => ({ error: "not in fixture" }) };
    return { ok: true, status: 200, json: async () => body };
  },
  setTimeout: () => 0, setInterval: () => 0, clearTimeout: () => {}, clearInterval: () => {},
};
vm.createContext(ctx);
let threw = null;
for (const f of ["common.js", "about.js"]) {
  try { vm.runInContext(readFileSync(path.join(WEB, f), "utf8"), ctx, { filename: f }); }
  catch (e) { threw = `${f}: ${e.message}`; }
}
const $ = (id) => byId.get(id) || new El("div", id);
const t = (id) => ($(id).textContent || "").trim();
await new Promise((r) => setTimeout(r, 3000));                // Node's own timer: let one load land

const S = PAY ?? await (await globalThis.fetch(`${BASE}/api/status`)).json();
const all = Object.keys(((S.config || {})._all) || {});
const efile = ((S.env || {}).file) || {}, evars = Object.keys((S.env || {}).vars || {});
const shown = [...byId.values()].filter((e) => e.textContent).map((e) => e.id).filter(Boolean);
let bad = 0;
const B = (m) => { console.log("BAD ", m); bad++; };

if (threw) B(threw);
// A payload with no env block is a real state, not a rendering bug: an in-process plugin started before this card
// existed serves one. Say which case it is, and keep failing -- the page genuinely cannot answer the question then.
const noEnv = !S.env;
if (noEnv) console.log("NOTE  this payload carries no environment block: the build serving it predates the card");
for (const [id, why] of [["facts-config", "the process/argv/bind rows never rendered"],
                         ["facts-cfgall", "the engine's own label list never rendered"],
                         ["facts-env", "the process environment block never rendered"],
                         ["facts-envfile", "the machine-local env file block never rendered"]]) {
  if (!t(id)) B(`${id} is empty -- ${why}${noEnv ? " (payload has no env/config-extra block: stale build)" : ""}`);
}
console.log("\nconfiguration card:");
console.log("  recipe file (launch card): " + t("launch-src"));
console.log("  " + t("facts-config").replace(/ {2,}/g, " | "));
console.log("  env file note: " + t("env-file-note"));
console.log(`  engine labels listed: ${$("cfg-all-n").textContent} · env names: ${$("env-proc-n").textContent}`
  + ` · file lines: ${$("env-file-n").textContent}`);
if (all.length && !t("facts-cfgall").includes(all[0])) B("the label list does not contain the labels the payload has");
if (Number((t("cfg-all-n").match(/\d+/) || [0])[0]) !== all.length) {
  B(`the label count says ${t("cfg-all-n")} and the payload carries ${all.length}`);
}
if (evars.length && !t("facts-env").includes(evars[0].replace(/^~/, ""))) B("an env name in the payload is not on the page");
if (efile.path && !t("facts-config").includes(efile.path)) B("the page does not name the file it is listing");
if (Object.keys(efile.vars || {}).length && !t("facts-envfile").includes("HOST")) B("HOST is in the file and not on the page");
// A withheld credential must never appear, and its name must say it is withheld rather than looking unset.
const masked = Object.entries((S.env || {}).vars || {}).filter(([k]) => k.startsWith("~"));
for (const [k] of masked) if (!t("facts-env").includes(k.slice(1))) B(`${k} was dropped instead of shown as withheld`);
if (masked.length && !/value withheld/.test(t("facts-env") + t("facts-envfile"))) {
  B("a masked name is shown without saying its value is withheld");
}
if (noEnv) {
  if (!/predates the card/.test(t("env-file-note"))) {
    B("a payload with no env block is explained as a missing file instead of as an old build");
  }
} else if (!S.argv && S.argv !== "") B("the page had no command line to show, and no row saying why");

// Image input card. The engine publishes no multimodal label, so the page has exactly two witnesses -- the live
// process's argv and the recipe file -- and each row must say which one it is. On a sidecar the argv belongs to the
// sidecar, so the page must not claim it answers what the serve accepts.
console.log("\nimage input card:");
console.log("  " + t("facts-mm").replace(/ {2,}/g, " | "));
console.log("  note: " + t("mm-note").slice(0, 160));
const mmtxt = t("facts-mm");
if (!mmtxt && (S.argv || (S.meta || {}).mm_effective)) {
  B("the payload can answer the image-input question and the card is blank");
}
if (mmtxt) {
  if (S.mode === "merged") {
    if (!/The running serve accepts/.test(mmtxt)) B("the merged page never says what the serve accepts");
    if (!/--language-model-only|image\(s\)/.test(mmtxt)) B("the live row answers neither mm state");
  } else if (/The running serve accepts/.test(mmtxt)) {
    B("a sidecar page claims to know what the serve accepts, reading its own command line");
  } else if (!/proves nothing about what the serve accepts/.test(mmtxt)) {
    B("a sidecar page does not say that its own argv says nothing about the serve");
  }
  if (!/px per image token/.test(mmtxt)) B("the px-per-token basis is missing, so a pixel cap is not a cost");
  if (!/label\(s\) in the config block/.test(mmtxt)) B("the card does not say the engine config is silent here");
  if (Number((S.meta || {}).mm_max_pix) > 0 && !/tokens/.test(mmtxt)) {
    B("a size cap is printed without its token cost, which is the number that actually matters");
  }
}
if (S.mode === "merged" && (S.meta || {}).mm_effective === "off" && !/no images, no video/.test(mmtxt)) {
  B("the recipe says mm is off and the live row does not say so");
}
if (!t("mm-note")) B("mm-note is empty -- the measured / not-measured boundary never renders");
// A recipe file and a running process can disagree about multimodal; the card has to name it rather than show two
// rows and hope the reader diffs them.
const mmDisagree = S.argv && (S.meta || {}).mm_effective
  && (S.argv.includes("--language-model-only")) === (String(S.meta.mm_effective) === "on");
if (mmDisagree && !/Recipe and process disagree/.test(mmtxt)) {
  B("recipe and process disagree about images and the card stays silent");
}
if (!mmDisagree && /Recipe and process disagree/.test(mmtxt)) {
  B("the card claims a disagreement the payload does not show");
}
// A sidecar can outlive the launch its recipe describes, so the launch card says when that recipe was written.
if (S.meta_mtime && !/written/.test(t("launch-src"))) {
  B("the launch card does not say which launch its recipe describes");
}

const bar = t("error");
if (bar) { console.log("\nPAGE NOTICE:", bar); bad++; }
console.log(`\nsections the page wrote into: ${shown.sort().join(" ")}`);
console.log(bad ? `\nBADs: ${bad}` : "\nrender: ok");
process.exitCode = bad ? 1 : 0;
