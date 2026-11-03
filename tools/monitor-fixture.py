#!/usr/bin/env python3
"""Build a Monitor fixture: today's live status plus yesterday's finished rows, for the render harness.

    ~/venv-r9700/bin/python tools/monitor-fixture.py [JOURNAL] [BASE] [OUT] [--rows N]

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
    out.write_text(json.dumps({"status": status, "requests": {"requests": keep}}, indent=1))
    print(f"{out}: status from {src}, {len(keep)} rows from {journal}")
    for r in keep[-3:]:
        t = r.get("timings") or {}
        print(f"  {r.get('seq')}: prompt {r['usage']['prompt_tokens']} decode {t.get('predicted_per_second')} "
              f"tok/s shared x{r['shared_decode']}")


if __name__ == "__main__":
    main(sys.argv[1:])
