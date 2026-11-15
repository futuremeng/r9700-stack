#!/usr/bin/env node
// tests/render-about.mjs -- run serve/web/about.js unmodified against a payload, and check the parts of the page
// that are pure text. About has never had this: test_webui.py parses the file and pins its strings, which catches a
// syntax error but not a ReferenceError, and a ReferenceError inside one section of About leaves that section empty
// with no error anywhere (the same failure mode notes/webui.md records for the Monitor, where a `${unit}` in a
// function with no `unit` parameter blanked a card).
//
//   tests/render-about.mjs http://127.0.0.1:8080      # a running serve (8080 is the front door here)
//   tests/render-about.mjs /tmp/status.json           # a saved /api/status payload
//
// Exit 0 when the page rendered and every claim below held.
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.join(HERE, "..", "serve", "web");
const ARG = process.argv[2] || "http://127.0.0.1:8080";
const FIX = ARG.endsWith(".json") ? JSON.parse(readFileSync(ARG, "utf8")) : null;
// A saved payload may be the bare /api/status JSON or one wrapped in {status: ..., host: ...} (what the page's own
// dump writes). Normalize once, so the PAGE and the CHECKS below are looking at the same object -- reading
// `FIX.status` in one place and `FIX` in another is how a fixture silently renders as an empty status.
const PAY = FIX ? (FIX.status || FIX) : null;
// Not every fixture is an About fixture. tools/monitor-fixture.py dumps the Monitor's payload, which has no
// argv / bind / env / gateway blocks -- handing it to this harness produced 37 BADs that were all the same fact
// rediscovered 37 times. Refuse once, and say what a usable payload looks like.
if (PAY && !(PAY.argv || PAY.env || PAY.meta)) {
  console.log("BAD  this fixture has no argv/env/meta block, so it is not an About payload."
    + " Make one from a running serve: {" + '"status": <GET /api/status>, "host": <GET /api/host>, '
    + '"sidecars": <GET /api/sidecars from the front door>}');
  process.exit(1);
}
const BASE = FIX ? "" : ARG.replace(/\/$/, "");

class El {
  constructor(tag = "div", id = "") {
    this.tagName = tag; this.id = id; this._text = ""; this.innerHTML = "";
    this.className = ""; this.dataset = {}; this.attrs = {}; this.children = [];
    this.style = {}; this.hidden = false; this.value = ""; this.disabled = false;
    this.classList = { _s: new Set(), add(...c) { c.forEach((x) => this._s.add(x)); },
                       remove(...c) { c.forEach((x) => this._s.delete(x)); },
                       // `force` is not decoration: a stub that ignores it makes classList.toggle(name, false)
                       // turn the class ON, so the page looked like it was hiding columns it was actually showing
                       // (and the harness invented four BADs about a grid that rendered correctly).
                       toggle(c, on) { if (on === undefined) { this._s.has(c) ? this._s.delete(c) : this._s.add(c); }
                                       else { on ? this._s.add(c) : this._s.delete(c); } },
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
  // querySelector is memoised per selector string: About sets dataset.copy on the element it finds, and a stub
  // that hands back a fresh element every call makes those payloads unreadable to the checks below -- which is
  // exactly how a Copy button wired to nothing stayed invisible.
  querySelector(sel) {
    if (!this._qs) this._qs = new Map();
    if (!this._qs.has(sel)) this._qs.set(sel, new El("div"));
    return this._qs.get(sel);
  }
  focus() {} remove() {} click() {}
}

const byId = new Map();
const qsel = new Map();                                     // document.querySelector, memoised per selector
const document = {
  activeElement: null, hidden: false, documentElement: new El("html"), body: new El("body"),
  head: { appendChild: () => {} },
  createElement: (t) => new El(t),
  getElementById: (id) => { if (!byId.has(id)) byId.set(id, new El("div", id)); return byId.get(id); },
  querySelectorAll: () => [], addEventListener: () => {},
  querySelector: (sel) => { if (!qsel.has(sel)) qsel.set(sel, new El("div")); return qsel.get(sel); },
};
const store = () => { const m = new Map(); return { getItem: (k) => (m.has(k) ? m.get(k) : null),
  setItem: (k, v) => m.set(k, v), removeItem: (k) => m.delete(k) }; };
const ctx = {
  document, console, Promise, JSON, Math, Date, Number, String, Object, Array, RegExp, Set, Map, Error,
  AbortController, AbortSignal, isNaN, parseInt, parseFloat, isFinite, encodeURIComponent,
  decodeURIComponent, Intl, localStorage: store(), sessionStorage: store(),
  navigator: { clipboard: { writeText: async () => {} }, userAgent: "render-about" },
  location: { origin: FIX ? "http://127.0.0.1:8080" : BASE },   // the fixture poses as a client of the
                                                             // front door, which is what the page now sees
  fetch: async (u, o) => {
    if (!FIX) return globalThis.fetch(BASE + u, o);
    const p = String(u).split("?")[0];
    // /api/host is a separate endpoint: the merged serve answers it, and a bare /api/status payload saved from one
    // does not carry the block. In file mode that means 'no host facts in this fixture', not 'the endpoint is down',
    // so hand back an empty object. URL mode still passes the real 404 through, which is what caught the page's
    // habit of treating one as fatal.
    const body = p === "/api/status" ? PAY
      : p === "/api/host" ? ((FIX && (FIX.host ?? (PAY ? {} : null))) ?? null)
      // A fixture that carries the gateway's /api/sidecars answer has to hand it to the page as well as to the
      // checks below; withholding it from the page and reading it in the checks calls honest rendering a bug, which
      // is the same mistake render-monitor.mjs made before it started comparing against the payload the page drew.
      : p === "/api/sidecars" ? (FIX && FIX.sidecars
          // a fixture may carry the bare list or the whole {sidecars: [...], at: ...} response; feed the page the
          // response shape, which is what the endpoint answers
          ? (Array.isArray(FIX.sidecars) ? {sidecars: FIX.sidecars} : FIX.sidecars) : null)
      // The alert channel: the fixture carries exactly what the endpoint answers (masked -- the server never
      // hands a token to a page), or nothing at all for the older-build case.
      : p === "/api/notify" ? ((FIX && FIX.notify) || null)
      : null;
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

// The front door. This box answers ONE port to the LAN and keeps the engine on loopback behind
// serve/gateway.py, so the About page has to say which port the browser actually reached, and what else that
// port routes to (/v1/embeddings, /v1/rerank are not this engine answering). Both rows are conditional, so the
// check runs both ways: a serve standing on its own port must NOT claim a front door it does not have.
const ORIGIN = FIX ? "http://127.0.0.1:8080" : BASE;
const OP = String(ORIGIN).match(/^https?:\/\/[^/]*?:(\d+)$/);
const fronted = !!(OP && S.port && String(S.port) !== OP[1]);
const SC = (FIX && FIX.sidecars) || await globalThis.fetch(`${BASE}/api/sidecars`)
  .then((r) => (r.ok ? r.json() : null)).catch(() => null);
const cfg = t("facts-config"), api = t("facts-api");
console.log(`\nfront door: browser on ${ORIGIN}, this process on ${S.bind}:${S.port}`
            + (SC && SC.sidecars ? ` · other routes: ${SC.sidecars.map((x) => x.name).join(" ")}` : " · no gateway"));
if (fronted && !/Reached through/.test(cfg)) B("the config card does not say which port the browser reached");
if (!fronted && /Reached through/.test(cfg)) B("the config card names a front door the payload does not show");
if (SC && SC.sidecars && !/Serves behind this URL/.test(api)) B("the API card does not say the other routes are other processes");
if ((!SC || !SC.sidecars) && /Serves behind this URL/.test(api)) B("the API card names other serves with no gateway answering");

// What this page is FOR, for anyone arriving to configure a client: one section per model family this URL answers,
// each carrying its two strings at a size that outranks the prose, plus blocks that paste without editing. Each
// goes blank silently if the payload or the wiring breaks, and the page still says "render: ok" -- the class of
// rot only a check like this catches.
const cbase = t("ep-chat-base"), ROOT = ORIGIN.replace(/\/$/, ""), cmodel = t("ep-chat-model"), cenv = t("agent-env");
console.log(`\nconnect: ${cbase}  ·  chat model ${cmodel}`);
if (cbase !== `${ROOT}/v1`) B(`the chat base_url cell says "${cbase}", not the address this page was served on plus /v1`);
if (S.model_name && cmodel !== S.model_name) B(`the chat section says "${cmodel}", not ${S.model_name}`);
if (!/OPENAI_BASE_URL=\S+\/v1/.test(cenv)) B("the agent config block has no OPENAI_BASE_URL ending in /v1");
if (S.model_name && !cenv.includes(`OPENAI_MODEL=${S.model_name}`)) B("the agent config block does not carry the model name");
if (SC && (SC.sidecars || []).length && !/the same base URL also answers/.test(cenv)) B("the agent config block hides the other routes on this port");
if (SC && (SC.sidecars || []).some((x) => x.serve && x.serve.alias) && !/model qwen3-|model [a-z0-9.-]+\)/.test(cenv))
  B("the agent config block names the sidecar routes without their model names");

// Three families behind one URL, so three sections with three names -- and each family's curl block has to paste
// ITS name: the card used to show one model name and one curl, and a reader wiring up embeddings pasted a name the
// embeddings serve never heard of (a 404 listing names they had not seen either). Checked per section, per block,
// and against the payload rather than against the page's own strings.
const cp = (id) => { const b = document.querySelector(`[data-copy-target=${id}]`); return (b && b.dataset && b.dataset.copy) || ""; };
const FAM = [["/v1/embeddings", "embed", "curl-embed"], ["/v1/rerank", "rerank", "curl-rerank"]];
const GRID = $("ep-grid");
const off = (c) => GRID.classList.contains(`no-${c}`);
const seen = [];
for (const [route, col, block] of FAM) {
  const e = ((SC && SC.sidecars) || []).find((x) => (x.name || "").trim() === route);
  const alias = e ? (((e.serve || {}).alias || "").trim()) : "";
  if (!e) {
    if (!off(col)) B(`${route} is not answered from this port, but its ${col} column is still in the grid`);
    if (cp(`ep-${col}-model`)) B(`the ${col} column offers a Copy payload for a route this port does not answer`);
    continue;
  }
  if (off(col)) B(`${route} is routed by the front door but its ${col} column is dropped from the grid`);
  if (alias && t(`ep-${col}-model`) !== alias) B(`the ${col} column shows "${t(`ep-${col}-model`)}" where the serve advertises "${alias}"`);
  if (t(`ep-${col}-base`) !== `${ROOT}/v1`) B(`the ${col} base_url is "${t(`ep-${col}-base`)}", not "${ROOT}/v1"`);
  if (alias) {
    seen.push(alias);
    if (!t(block).includes(`"${alias}"`)) B(`${block} does not paste ${alias}; it reads "${t(block).slice(0, 60)}"`);
    if (cmodel && t(block).includes(`"${cmodel}"`)) B(`${block} pastes the chat model name into a ${route} request`);
    if (!cp(`ep-${col}-model`).includes(alias)) B(`the Copy beside the ${col} model name does not carry what is on screen`);
    if (cp(`ep-${col}-base`) !== `${ROOT}/v1`) B(`the Copy beside the ${col} base_url is not what is on screen`);
  }
}
if (cmodel) seen.push(cmodel);
if (new Set(seen).size !== seen.length) B(`two columns offer the same model name to a client: ${seen.join(" / ")}`);
// The one row where the three columns must NOT agree. The reranker's LiteLLM api_base stops at the root, because
// litellm appends /rerank itself; writing the route there produced POST /v1/rerank/rerank and a 404 that named no
// path (2026-11-15), so the depth of that string is a fact worth pinning.
const lbase = t("ep-rerank-lbase");
if (!off("rerank")) {
  if (lbase !== ROOT) B(`the rerank LiteLLM api_base is "${lbase}", not the root "${ROOT}"`);
  if (cp("ep-rerank-lbase") !== lbase) B("the Copy beside the rerank api_base carries something else");
}
// The LiteLLM block has to say what the grid says: one deployment per family this port answers, hosted_vllm for
// the reranker (its rerank API has no openai provider), model_info mode for the CPU serves, and no api_base that
// ends in a route path.
const ly = t("litellm-yaml");
const dep = (ly.match(/- model_name:/g) || []).length;
const wantDep = 1 + (off("embed") ? 0 : 1) + (off("rerank") ? 0 : 1);
if (dep !== wantDep) B(`the LiteLLM block lists ${dep} deployments while this port answers ${wantDep} families`);
if (!off("rerank")) {
  if (!/model: hosted_vllm\//.test(ly)) B("the LiteLLM block gives the reranker a provider its rerank API rejects");
  if (!/mode: rerank/.test(ly)) B("the LiteLLM block omits mode: rerank, so the proxy health-checks it with a chat completion");
  if (!/mode: embedding/.test(ly)) B("the LiteLLM block omits mode: embedding");
  const bad = (ly.match(/api_base: \S+/g) || []).filter((l) => /\/(v1\/)?(embeddings|rerank)$/.test(l));
  if (bad.length) B(`an api_base in the LiteLLM block ends in a route path: ${bad.join(" ")}`);
}
// A Copy button is invisible when it is dead: the click does nothing and no error appears. The static check in
// tests/test_webui.py pins the wiring; this pins that the wiring produced a string, for the buttons on screen.
const needCopy = ["ep-chat-model", "ep-chat-base", "agent-env", "curl", "litellm-yaml", "litellm-py"]
  .concat(off("embed") ? [] : ["ep-embed-model", "ep-embed-base", "curl-embed"])
  .concat(off("rerank") ? [] : ["ep-rerank-model", "ep-rerank-base", "ep-rerank-lbase", "curl-rerank"]);
for (const id of needCopy) if (!cp(id)) B(`the Copy for ${id} has no payload in this render`);
const wired = [...qsel.keys()].filter((k) => k.startsWith("[data-copy-target="));
const empty = wired.filter((k) => !(qsel.get(k).dataset || {}).copy);
const COLS = ["chat", "embed", "rerank"].filter((c) => !off(c));
console.log(`\nconnect grid, columns: ${COLS.join(" / ")}`);
for (const [row, f] of [["model", (c) => t(`ep-${c}-model`)], ["base_url", (c) => t(`ep-${c}-base`)],
                        ["litellm api_base", (c) => c === "rerank" ? t("ep-rerank-lbase") : `${ROOT}/v1`]])
  console.log(`  ${row.padEnd(16)} ${COLS.map(f).join("  |  ")}`);
console.log(`  Copy buttons carrying a payload: ${wired.length - empty.length}/${wired.length}`
            + (empty.length ? ` (empty: ${empty.join(" ")})` : ""));

// -- the alert channel card -------------------------------------------------------
// The page is where an operator pastes a webhook, so two things have to hold. The credential never comes back --
// a page that echoes it is a page any script on it can read. And "saved" must never render as "working": the
// robot answers a wrong keyword or a dead token with HTTP 200 and a JSON errcode, so the card has to show that
// answer, not the file's existence.
const NT = (FIX && FIX.notify)
  || await globalThis.fetch(`${BASE}/api/notify`).then((r) => (r.ok ? r.json() : null)).catch(() => null);
const alertSec = byId.get("alert-card") || new El();
const alertFacts = t("facts-alert");
console.log(`\nalert channel: hidden=${alertSec.hidden}`
            + (NT ? ` | configured=${NT.configured} | ${alertFacts.replace(/ {2,}/g, " | ")}`
                  : " | no /api/notify answering"));
if (!NT) {
  if (!alertSec.hidden) B("the Alerts card is shown with no /api/notify answering -- an older build has nothing to "
                          + "configure and the card would be a form wired to a 404");
} else {
  if (!alertFacts) B("the payload answers the alert channel and the card is blank");
  if (NT.configured) {
    if (!/token withheld/.test(alertFacts)) {
      B("a saved webhook is shown without saying the token is withheld");
    }
    if (NT.channel && NT.channel.host && !alertFacts.includes(NT.channel.host)) {
      B("the card does not name the host the channel points at");
    }
    if (!NT.channel || NT.channel.webhook !== undefined) {
      B("the payload handed the page a webhook field -- the endpoint must answer masked");
    }
  } else if (!/not set/.test(alertFacts)) {
    B("no channel is configured and the card does not say so -- a red Monitor card would imply a ping nobody sent");
  }
  const last = (NT.last || {}).result;
  if (last) {
    if (last.sent && !/accepted by the robot/.test(alertFacts)) B("a delivered send is not reported as delivered");
    if (!last.sent && !/NOT delivered/.test(alertFacts)) B("a refused send is rendered as if it had worked");
    if (!last.sent && last.error && !alertFacts.includes(String(last.error).slice(0, 24))) {
      B(`the refusal reason is missing from the card: ${last.error}`);
    }
  } else if (!/never attempted/.test(alertFacts)) {
    B("no send has ever been attempted and the card claims otherwise");
  }
  const leak = [...byId.values()].filter((e) => /access_token=/.test(e.textContent || "")).map((e) => e.id);
  if (leak.length) B(`a webhook token form is on the page in ${leak.join(", ")} -- the channel is masked for a reason`);
}

const bar = t("error");
if (bar) { console.log("\nPAGE NOTICE:", bar); bad++; }
console.log(`\nsections the page wrote into: ${shown.sort().join(" ")}`);
console.log(bad ? `\nBADs: ${bad}` : "\nrender: ok");
process.exitCode = bad ? 1 : 0;
