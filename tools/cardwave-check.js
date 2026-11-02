// tools/cardwave-check.js -- run the Monitor's GPU-card renderer against a live sidecar payload, under
// node, with a DOM stub just rich enough to see what it produced: two waveforms (busy, VRAM), no busy bar,
// and a caption. A browser is the real test, but this catches the failure mode that matters -- the card
// rendering nothing at all because of a typo, which is silent on the page.
//   node tools/cardwave-check.js serve/web http://localhost:8099/api/status
const fs = require("fs");
const dir = process.argv[2];
const common = fs.readFileSync(dir + "/common.js", "utf8");
const monjs = fs.readFileSync(dir + "/monitor.js", "utf8");
const grab = (s, re) => {
  const m = s.match(re);
  if (!m) throw new Error("no match for " + re);
  return m[0];
};
const code = [grab(common, /^const isnum = .*$/m), grab(common, /^const fmt = .*$/m), grab(common, /^const gb = .*$/m),
              grab(common, /function spark[\s\S]*?\n}\n/), grab(common, /function gpuCard[\s\S]*?\n}\n/),
              grab(monjs, /function pcieCard[\s\S]*?\n}\n/)].join("\n");

const caps = [];                                    // what the caption slot was told to show
const mkEl = (tag) => ({
  tag, attrs: {}, children: [], dataset: {}, _html: "",
  set innerHTML(v) { this._html = v; }, get innerHTML() { return this._html; },
  setAttribute(a, v) { this.attrs[a] = v; },
  replaceChildren(...c) { this.children = c; }, append(...c) { this.children.push(...c); },
  querySelector(sel) { return this.querySelectorAll(sel)[0]; },
  querySelectorAll(sel) {
    if (sel.startsWith("[")) {                        // the caption slot: hand back a recording spy
      const spy = mkEl("span");
      let txt = "";
      Object.defineProperty(spy, "textContent", {get: () => txt, set: (v) => { caps.push(v); txt = v; }});
      return [spy];
    }
    const cls = sel.replace(".", "");
    return (this._html.match(new RegExp('class="[^"]*' + cls + '[^"]*"', "g")) || []).map(() => mkEl("div"));
  },
});
global.document = {
  createElement: (t) => mkEl(t),
  createElementNS: (ns, t) => mkEl(t),
  addEventListener() {},
};
eval(code);

fetch(process.argv[3]).then((r) => r.json()).then((s) => {
  const gh = s.gpus_history || [];
  for (const g of s.gpus) gpuCard(g, gh);
  const d = gpuCard(s.gpus[0], gh);
  console.log("cards:", s.gpus.length, "| ring:", gh.length, "samples over",
              (gh.length > 1 ? gh[gh.length - 1].at - gh[0].at : 0).toFixed(0) + "s");
  console.log("first card", s.gpus[0].card, "bus", s.gpus[0].bus,
              "| wave rows:", (d.innerHTML.match(/spark--gpu/g) || []).length,
              "| busy bars left:", (d.innerHTML.match(/class="bar"/g) || []).length);
  console.log("vram text:", (d.innerHTML.match(/[\d.]+ \/ [\d.]+ GiB VRAM/) || ["(none)"])[0]);
  console.log("captions:", caps.slice(0, 2).map((c) => `"${c}"`).join("\n           ") || "(none)");
  const chains = s.pcie || [], phist = s.pcie_history || [];
  for (const c of chains) {
    const pd = pcieCard(c, phist);
    console.log(`pcie ${String(c.bus).padEnd(9)} card ${String(c.card).padEnd(6)} `
      + `waves ${(pd.innerHTML.match(/spark--gpu/g) || []).length} `
      + `links ${(c.devices || []).length} since-boot-cor ${(c.devices || []).reduce((a, x) => a + (x.cor || 0), 0)}`);
  }
  if (chains.length) console.log("pcie caption:", (() => { const e = pcieCard(chains[0], phist);
    const m = e.innerHTML.match(/since boot[^<]*/g); return m ? m[m.length - 1] : "(in a span, not inline)"; })());
}).catch((e) => console.log("ERR", e.message));
