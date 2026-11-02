// tools/spark-check.js -- run the sparkline primitive against synthetic series under node and print the paths it
// emits. A waveform's whole job is shape, and the shapes that matter most are the awkward ones: nothing running,
// everything at zero, a gap in the middle, a series that goes negative. In a browser those are hard to summon on
// demand; here they are three lines of data.
//   node tools/spark-check.js serve/web
const fs = require("fs");
const src = fs.readFileSync(process.argv[2] + "/common.js", "utf8");
const grab = (re) => {
  const m = src.match(re);
  if (!m) throw new Error("no match for " + re);
  return m[0];
};
const code = [grab(/^const isnum = .*$/m),
              grab(/function spark[\s\S]*?\n}\n/)].join("\n");

const mkEl = (tag) => {
  const el = { tag, attrs: {}, children: [], dataset: {},
               setAttribute(a, v) { this.attrs[a] = v; },
               append(...c) { this.children.push(...c); },
               replaceChildren(...c) { this.children = c; } };
  return el;
};
global.document = { createElement: (t) => mkEl(t), createElementNS: (ns, t) => mkEl(t), addEventListener() {} };
eval(code);

const show = (label, values, opts) => {
  const el = mkEl("div");
  spark(el, values, opts);
  const svg = el.children[0];
  const paths = (svg ? svg.children : []).map((p) => {
    const d = p.attrs.d || "";
    const ys = [...d.matchAll(/,(-?[\d.]+)/g)].map((m) => +m[1]);
    return `${p.attrs.class || (p.attrs.fill === "none" ? "line" : "fill")} y=${ys.length ? Math.min(...ys) : "-"}..${ys.length ? Math.max(...ys) : "-"}`;
  });
  console.log(`${label.padEnd(34)} ${paths.length ? paths.join(" | ") : "(no svg — empty box)"}`);
};

const N = 8;
show("all null, axis 0..peak", new Array(N).fill(null), { baseLo: 0, baseHi: 0 });
show("all zero, axis 0..peak", new Array(N).fill(0), { baseLo: 0, baseHi: 0 });
show("all zero, axis 0..100", new Array(N).fill(0), { baseLo: 0, baseHi: 100 });
show("half null then 40s", [null, null, null, null, 40, 40, 40, 40], { baseLo: 0, baseHi: 100 });
show("gap in the middle", [30, 30, null, null, 30, 30, 30, 30], { baseLo: 0, baseHi: 100 });
show("steady 40", new Array(N).fill(40), { baseLo: 0, baseHi: 100 });
show("KV pool near empty", [0.2, 0.4, 0.3, 0.4, 0.2, 0.3, 0.4, 0.3], { baseLo: 0, baseHi: 100 });
show("prefill estimate spans zero", [-800, 1200, 900, -500, 1500, 0, 700, -200]);
show("one sample only", [42], { baseLo: 0, baseHi: 100 });
show("no samples, autoscaled", new Array(N).fill(null));
