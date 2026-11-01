#!/usr/bin/env python3
"""Recover the wall-clock duration of every post-load window from log timestamps, and set it beside the number the
profiler printed for it.

Why this exists: until 2026-11-01 the profiler's "target post-load: X s" figure was the **sum of the per-class gaps**
measured inside the wrapped ``named_modules`` loop, not wall time. A nested ``named_modules`` iteration counts the
same interval twice, so the sum can sit anywhere from ~1x to ~2.5x (tail83: 6.6x) of the real duration -- and every
"clean 4.5 s / elevated 11-16 s" comparison in ``notes/native-4xr9700.md`` was made on that sum. The instrument now
prints ``X s of wall (Y s attributed)``, but the older launches can still be read correctly, because both the window
opening line ("r9700 post-load timer on") and its closing line carry the worker pid and a second-resolution
timestamp.

    usage: ~/venv-r9700/bin/python3 bench/window_wall.py ~/logs/relaunch-tail65.log ...

Caveat, stated because the numbers below show it: pairing timestamps is only an estimate. It agrees exactly with the
direct measurement on launches that have the fix (final46: 5 s retro, 5.1-5.3 s reported), and disagrees by 6-8 s on
some older ones in both directions -- which is itself the evidence that the old printed figure was not a duration.
"""
import datetime as dt
import re
import sys

LINE = re.compile(r"\(Worker_TP(\d) pid=(\d+)\).*?(\d\d:\d\d:\d\d) \[[^\]]+\] (.*)")
DUR = re.compile(r"([\d.]+) s(?: of wall)?(?: \(([\d.]+) s attributed\))?")


def report(path: str) -> None:
    opened: dict[str, dt.datetime] = {}
    rows: list[tuple[float, float, float]] = []          # (printed-or-attributed, retro wall, direct wall)
    for line in open(path, errors="replace"):
        m = LINE.search(line)
        if not m:
            continue
        pid, ts, msg = m.group(2), m.group(3), m.group(4)
        t = dt.datetime.strptime(ts, "%H:%M:%S")
        if "post-load timer on" in msg:
            opened[pid] = t
        elif "target post-load:" in msg:
            d = DUR.search(msg)
            if d and pid in opened:
                rows.append((float(d.group(2) or d.group(1)), (t - opened[pid]).total_seconds(),
                             float(d.group(1)) if d.group(2) else float("nan")))
    name = path.rsplit("/", 1)[-1].replace("relaunch-", "").replace(".log", "")
    if not rows:
        print(f"{name:<10} no post-load window lines")
        return
    att = " ".join(f"{a:4.0f}" for a, _, _ in sorted(rows, reverse=True))
    rtl = " ".join(f"{w:3.0f}" for _, w, _ in sorted(rows, reverse=True))
    drc = " ".join(f"{d:3.0f}" for _, _, d in sorted(rows, key=lambda r: -r[1]) if d == d)
    print(f"{name:<10} printed/attribution[{att:<24}] retro wall[{rtl}]" + (f"  reported wall[{drc}]" if drc else ""))


for p in sys.argv[1:]:
    try:
        report(p)
    except OSError as e:
        print(f"{p}: {e}")
