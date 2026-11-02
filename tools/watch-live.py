#!/usr/bin/env python3
"""tools/watch-live.py -- watch a live serve the way the Monitor does, and print what the load actually is.

Sampling only: it reads /api/status (engine gauges, live rates, per-card sensors, PCIe chains) and /api/requests
from the running serve, never sending a request of its own. That matters here: adding load while measuring
someone else's concurrency would contaminate the very numbers being read.

  tools/watch-live.py [seconds] [BASE] [--quiet]

What it is for. The Monitor page answers "what is it doing right now" for a person looking at it; this answers
the same question in a form you can quote: percentiles over the window, per-card sensor ranges, and every PCIe
AER event with its wall-clock time next to what the serve was doing at that instant (the incident question:
does anything on the link happen while a request is in flight?).
"""
import json
import statistics as st
import sys
import time
import urllib.request

BASE = "http://localhost:8080"
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
QUIET = "--quiet" in sys.argv
N = int(ARGS[0]) if ARGS else 120
if len(ARGS) > 1:
    BASE = ARGS[1]
SIDECAR = BASE.replace(":8080", ":8099")          # the sidecar runs the current build: it has the event list


def get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.load(r)


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else None


def line(name, xs, fmt="%.1f"):
    xs = [x for x in xs if isinstance(x, (int, float))]
    if not xs:
        print("  %-26s no samples" % name)
        return
    print(("  %-26s " + fmt + " avg | " + fmt + " p50 | " + fmt + " p95 | " + fmt + " max | n>0 %d/%d")
          % (name, st.mean(xs), pct(xs, .5), pct(xs, .95), max(xs), sum(1 for x in xs if x > 0), len(xs)))


def rng(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return "-" if not xs else ("%.0f-%.0f" % (min(xs), max(xs)))


start = get(BASE + "/api/status")
rows0 = {r["id"] for r in get(BASE + "/api/requests")["requests"]}
try:
    events0 = get(SIDECAR + "/api/status").get("pcie_events") or []
    have_events = True
except Exception:
    events0, have_events = [], False
print("%s watch %ds | serve %s | boot %s | %d streams running | %d rows kept"
      % (time.strftime("%H:%M:%S", time.localtime(start["at"])), N, BASE, start["journal"]["boot"],
         start["live"]["running"], start["requests_kept"]))
seen = len(events0)
S, events = [], []
for i in range(N):
    s = get(BASE + "/api/status")
    S.append(s)
    if have_events:
        ev = get(SIDECAR + "/api/status").get("pcie_events") or []
        for e in ev[:max(0, len(ev) - seen)]:
            L = s["live"]
            events.append((e, L.get("running"), L.get("kv_used_pct")))
            seen += 1
            print("   !! AER %s  %s %s %s +%d  (then: %s streams running, KV %.0f%%)"
                  % (time.strftime("%H:%M:%S", time.localtime(e["at"])), e["card"], e["bdf"], e["kind"], e["n"],
                     L.get("running"), L.get("kv_used_pct") or 0))
    elif i == 0:
        print("   (no event list from this serve build -- restart it, or read the sidecar, for timestamps)")
    if not QUIET and i % 10 == 0:
        L, R = s["live"], s["rate"]
        print("   +%3ds  run %s wait %s  KV %.0f%%  decode %.0f tok/s  stream %.0f tok/s  step %.0f ms"
              % (i, L["running"], L["waiting"], L["kv_used_pct"] or 0, R.get("generation_tok_s") or 0,
                 R.get("stream_tok_s_live") or 0, R.get("itl_ms") or 0))
    time.sleep(1.0)

sp = lambda k: [x["rate"].get(k) for x in S]
lv = lambda k: [x["live"].get(k) for x in S]
print("\n%d samples over %.0f s" % (len(S), S[-1]["at"] - S[0]["at"]))
line("running streams", lv("running"))
line("waiting", lv("waiting"))
line("KV pool %", lv("kv_used_pct"))
line("aggregate decode tok/s", sp("generation_tok_s"))
line("per-stream tok/s", sp("stream_tok_s_live"))
line("engine step ms", sp("itl_ms"))
line("tokens per step", sp("tokens_per_step"), "%.2f")

hist = [x["cards"] if "cards" in x else x for x in S[-1]["gpus_history"][-min(120, len(S)):]
        ] if S[-1]["gpus_history"] else []
print("\nper card, over the ring the serve still holds (%d samples):" % len(hist))
for i, g in enumerate(S[-1]["gpus"]):
    hs = [h[i] for h in hist if i < len(h)]
    print("  card%d busy %s%%  W %s  C %s  MHz %s  VRAM %s / %s GiB"
          % (i + 1, rng([x.get("busy_pct") for x in hs]), rng([x.get("power_w") for x in hs]),
             rng([x.get("temp_c") for x in hs]), rng([x.get("sclk_mhz") for x in hs]),
             rng([(x.get("vram_used") or 0) / 2**30 for x in hs]),
             "%.0f" % ((g.get("vram_total") or 0) / 2**30)))

rows = get(BASE + "/api/requests")["requests"]
new = [r for r in rows if r["id"] not in rows0]
print("\n%d request rows kept | %d finished during the watch" % (len(rows), len(new)))
for r in new[:12]:
    u, t = r.get("usage") or {}, r.get("timings") or {}
    print("   %s  %6s -> %5s tok | first token %5.1fs | decode %6.1fs @ %5.1f tok/s | prefill %6.1fs @ %7.0f tok/s"
          % (time.strftime("%H:%M:%S", time.localtime(r.get("started_at") or 0)), u.get("prompt_tokens"),
             u.get("completion_tokens"), r.get("first_token_s") or 0, t.get("decode_s") or 0,
             t.get("predicted_per_second") or 0, t.get("prefill_s") or 0, t.get("prefill_tok_s") or 0))
big = [r for r in rows if (r.get("usage") or {}).get("prompt_tokens")]
if big:
    pt = [r["usage"]["prompt_tokens"] for r in big]
    ft = [r["first_token_s"] for r in big if isinstance(r.get("first_token_s"), (int, float))]
    sp_ = [r["timings"].get("predicted_per_second") for r in big
           if r.get("timings") and isinstance(r["timings"].get("predicted_per_second"), (int, float))]
    pf = [r["timings"].get("prefill_tok_s") for r in big
          if r.get("timings") and isinstance(r["timings"].get("prefill_tok_s"), (int, float))]
    print("   %d rows: prompt %s tok (max %s) | first token p50 %.1fs p95 %.1fs | decode p50 %.1f tok/s"
          " | prefill p50 %.0f tok/s" % (len(big), rng(pt), max(pt), pct(ft, .5) or 0, pct(ft, .95) or 0,
                                         pct(sp_, .5) or 0, pct(pf, .5) or 0))
print("\nPCIe: %d AER event%s during the watch" % (len(events), "" if len(events) == 1 else "s"))
for ch in S[-1].get("pcie") or []:
    devs = ch["devices"]
    tot = sum(d.get("cor") or 0 for d in devs)
    unc = sum((d.get("nonfatal") or 0) + (d.get("fatal") or 0) for d in devs)
    print("  %-6s %-46s %5.1f GT/s x%-3s %d corr / %d uncorr since boot"
          % (ch.get("card"), " -> ".join(d["bdf"].replace("0000:", "") for d in devs), ch["speed"],
             ch["width"], tot, unc))
