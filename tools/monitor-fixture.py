#!/usr/bin/env python3
"""Build a Monitor fixture: today's live status plus yesterday's finished rows, for the render harness.

    ~/venv-r9700/bin/python tools/monitor-fixture.py [JOURNAL] [BASE] [OUT] [--rows N] [--wedged]

`--wedged` replaces the watchdog fields with the branch a healthy box never shows: a wedged launch (ranks still
holding the GPUs while /health is silent) and several fault lines across fewer events on one card. Keep it in the
pair of runs -- from 2026-11-15 the live serve DOES carry these reads, so a harness that only ever renders a
healthy payload stops covering the copy that exists for the unhealthy one.

Why this exists. `tests/render-monitor.mjs` renders the page against a live serve, so the request rows it
prints are whatever happens to be in the serve's memory at the moment -- and the interesting rows (a 160k
prompt that got preempted, a request that shared the engine with six others) are gone a minute later. The
journal keeps them, so a fixture of `status from the live box + rows from the journal` lets the row layout be
checked offline, against real values, without sending anything to a busy serve.

The rows go through the same `shared_decode` the Monitor applies, so the fixture exercises the derived field
rather than stubbing it.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "serve"))
from webui import Journal, shared_decode                            # noqa: E402


def main(argv):
    journal = Path(argv[0]) if argv else Path.home() / "logs/requests-8080.jsonl"
    base = (argv[1] if len(argv) > 1 else "http://127.0.0.1:8080").rstrip("/")
    out = Path(argv[2] if len(argv) > 2 else "/tmp/monitor-fixture.json")
    rows = 8
    if "--rows" in argv:
        rows = int(argv[argv.index("--rows") + 1])
    # Compute the derived field over a WIDE window and then trim. `shared_decode` counts the requests in
    # flight during each row's decode span, so handing it six rows would report x1 for a request that really
    # shared the engine with five -- the same mistake the page's own tooltip warns about, made by the fixture.
    wide = Journal(journal, keep=max(rows * 12, 240)).replay()
    for r, n in zip(wide, shared_decode(wide)):
        r["shared_decode"] = n
    keep = wide[:rows]
    try:
        with urllib.request.urlopen(f"{base}/api/status", timeout=5) as f:
            status = json.load(f)
        src = base
    except OSError as e:                                   # the box may be down -- which is when these rows matter
        status = {"mode": "fixture", "at": time.time(), "live": {}, "rate": {}, "since": {}, "gpus": [],
                  "gpus_history": [], "pcie": [], "pcie_history": [], "pcie_events": [], "journal": None,
                  "note": "fixture: no live serve was answering, so the status side of this render is empty"}
        src = f"no live serve ({type(e).__name__})"
    # The CPU sidecars the gateway aggregates (GET /api/sidecars). The fixture has to carry them: the render
    # harness reads the slot list out of monitor.html and fails any slot the page never writes, so a card that
    # only exists when a gateway is answering would fail the fixture render -- and a card that quietly never
    # renders is exactly the silent rot these checks exist to catch. Numbers from the live box, 2026-11-15:
    # the embedder at -t 15 (one more thread than that range has online cpus costs ~33% of its throughput),
    # Committed_AS 85.3 of 93.8 GiB with all three serves up.
    try:
        with urllib.request.urlopen(f"{base}/api/sidecars", timeout=5) as f:
            sidecars = json.load(f)
    except OSError:
        sidecars = {"sidecars": [
            {"name": "/v1/embeddings", "health": 200, "rates": {"rate": 1112.4, "busy": 0.0, "deferred": 0.0},
             "serve": {"threads": "15", "ctx": "8192", "batch": "512"},
             "host": {"committed_as_kb": 89397784, "embed_rss_kb": 1618092}},
            {"name": "/v1/rerank", "health": 200, "rates": {"rate": 1014.0, "busy": 1.0, "deferred": 0.0},
             "serve": {"threads": "16", "ctx": "4096", "batch": "512"},
             "host": {"committed_as_kb": 89397784, "embed_rss_kb": 1171012}}], "at": time.time()}
    # The gateway's admission gate (GET /api/gate), for the same reason the sidecars are here: the render
    # harness fails any slot in monitor.html the page never writes, so a card that only exists while a gateway
    # is answering would fail the offline render. The fallback is deliberately the HELD branch -- closed on a
    # pool reading, with the token count inside the reason -- because that is the branch a live idle box will
    # not show you, and it is the one the row exists for. Numbers of the shape this box produces at kv 0.86 of
    # a 902,153-token pool; the strip is 18 open polls followed by 12 shut ones, which is what a load ramp
    # looks like on a 1 s axis.
    try:
        with urllib.request.urlopen(f"{base}/api/gate", timeout=5) as f:
            gate = json.load(f)
    except OSError:
        gate = {"closed": True, "why": "kv 0.860 >= 0.85 (775,851 of 902,153 tokens); engine waiting 3",
                "gauges": {"kv": 0.86, "run": 24.0, "wait": 3.0, "at": time.time()}, "err": None,
                "inflight": 27, "polls": 4210, "hist": [0] * 18 + [1] * 12, "hist_n": 180,
                "gated": ["/v1/chat/completions", "/v1/completions"],
                "counters": {"held": 41, "released": 33, "expired": 6, "refused": 0, "admitted": 812},
                "config": {"GATE_ON": True, "GATE_KV_HIGH": 0.85, "GATE_KV_LOW": 0.78, "GATE_WAIT": 1,
                           "GATE_MAX_INFLIGHT": 0, "GATE_HOLD_S": 25.0, "GATE_METRICS_S": 1.0,
                           "GATE_STALE_S": 5.0, "pool_tokens": 902153}, "at": time.time()}
    # The watchdog row's inputs (webui's KFD rank->card join and its kernel-journal fault count). Same reason the
    # gate and sidecar blocks are here: the render harness fails any slot the page never writes, so a row that
    # only exists against a newer sidecar has to be renderable offline or it is invisible to that check. The
    # fallback is deliberately the branch a healthy box will not show you -- a WEDGED launch (three ranks still
    # holding the GPUs while /health is silent) and two faults on the card that took them, both of the shape the
    # driver wrote on 2026-11-15 -- because those are the states the row exists to catch.
    live = status.setdefault("live", {})
    # --wedged forces the branch even when a healthy live serve answered, because the row's whole reason to exist
    # is a state a healthy box never shows: without this, the wedged copy and the fault-line/event distinction go
    # uncovered the moment the sidecar carries the reads (which is the normal case from 2026-11-15 on).
    forced = "--wedged" in argv
    if forced or "engine_state" not in status:
        status["engine_state"], status["ranks_holding"] = "wedged", 3
    if forced or not status.get("faults"):
        # 3 driver lines across 2 events on one card -- the shape the real journal has on this box (11 lines
        # across 8 events since boot, three of them in the same second on 11-06), because the headline number is
        # how many TIMES it faulted and the per-card number is the line count, and a fixture that shows only one
        # of those two would let the page drop the other without anyone noticing.
        status["faults"] = {"total": 3, "events": 2, "by_card": {"c3:00.0": 3},
                            "last": {"at": "2026-11-15T09:52:00+00:00", "card": "c3:00.0", "pid": 591142},
                            "error": None, "at": time.time()}
    # The split between "this launch" and "earlier this boot" (webui fault_window) is what decides whether a card
    # raises its voice at all. --wedged states it explicitly, and coherently with the wedged engine: the launch on
    # screen is the one that faulted. A live sidecar older than the split sends no `now`, and that arm -- the page
    # showing since-boot and SAYING it cannot attribute -- is left unforced so it stays covered by the plain run.
    fl = status.get("faults") or {}
    if forced and "now" not in fl:
        fl["anchor"] = time.time() - 1800
        fl["now"] = {"total": fl.get("total", 0), "events": fl.get("events", 0), "by_card": fl.get("by_card", {})}
        fl["earlier"] = {"total": 0, "events": 0, "by_card": {}}
        fl["unattributed"] = 0
        status["faults"] = fl
    if forced or "prompt_bound" not in live:
        live.update({"prompt_bound": 1000.0, "prompt_big": 2, "prompt_n": 65, "prompt_big_pct": 3.1,
                     "prompt_big_all": 41, "prompt_n_all": 1180, "prompt_big_all_pct": 3.5})
    live.setdefault("preemptions", 3)
    live.setdefault("preemptions_total", 27)
    for i, g in enumerate(status.get("gpus", [])):
        if forced or "rank" not in g:
            g["rank"] = f"Worker_TP{i}"
            g["rank_pid"] = 659221 + i
            g["rank_gib"] = round((g.get("vram_used") or 30.9e9) / 2**30, 1)
        if forced or "faults" not in g:
            g["faults"] = 3 if (g.get("bus") or "").lower().endswith("c3:00.0") else 0
        if forced or "faults_launch" not in g:
            g["faults_launch"] = ((status.get("faults") or {}).get("now") or {}).get("by_card", {}) \
                .get((g.get("bus") or "").lower(), 0)
    out.write_text(json.dumps({"status": status, "requests": {"requests": keep},
                               "sidecars": sidecars, "gate": gate}, indent=1))
    print(f"{out}: status from {src}, {len(keep)} rows from {journal}")
    for r in keep[-3:]:
        t = r.get("timings") or {}
        print(f"  {r.get('seq')}: prompt {r['usage']['prompt_tokens']} decode {t.get('predicted_per_second')} "
              f"tok/s shared x{r['shared_decode']}")


if __name__ == "__main__":
    main(sys.argv[1:])
