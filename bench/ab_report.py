#!/usr/bin/env python3
"""Compare launches on the numbers that today's tail analysis actually turns on.

    usage: ~/venv-r9700/bin/python3 bench/ab_report.py ~/logs/relaunch-tail73.log ...

Each relaunch log carries, per worker: the post-load window (prof.start_pwal/stop_pwal), the census line that gives
that worker's minor/major faults taken *during* the window (the variable that separated the two shapes of tail 17:1:
~9k when the pass runs in 4.5 s, 150-162k when it runs 11-75 s), the worker's own swap at the start of the window,
the draft pass's wall time (what prestage exists to protect) and the box's disk reads. This just pulls them into one
table so an A/B is one command instead of a pile of greps.
"""
import re
import sys

# "self VmSwap   921 ->   586 MiB ... faults +48290 minor +38158 major   box pswpin +272442"
CENSUS = re.compile(r"self VmSwap\s+(\d+) ->\s+(\d+).*faults \+(\d+) minor \+(\d+) major.*pswpin \+(\d+)")
WINDOW = re.compile(r"target post-load: ([\d.]+) s over")
DRAFT = re.compile(r"load profile \(draft\): bf16->MXFP4 requant [\d.]+ s \([^)]*?([\d.]+) s wall\)")
DISK = re.compile(r"load snap (\S+).*disk reads=\s*([\d.]+) GiB")
LOAD = re.compile(r"Model loading took [\d.]+ GiB memory and ([\d.]+) seconds")
KV = re.compile(r"KV cache size: ([\d,]+) tokens")


def report(path: str) -> None:
    txt = open(path, errors="replace").read()
    wins = [float(x) for x in WINDOW.findall(txt)]
    cen = [(int(a), int(b), int(c), int(d), int(e)) for a, b, c, d, e in CENSUS.findall(txt)]
    draft = [float(x) for x in DRAFT.findall(txt)]
    disk = {k: float(v) for k, v in DISK.findall(txt)}
    name = path.rsplit("/", 1)[-1].replace("relaunch-", "").replace(".log", "")
    maj = sum(c[3] for c in cen)
    swap0 = sum(c[0] for c in cen) // max(1, len(cen))
    dload = max(disk.values()) - min(disk.values()) if len(disk) >= 2 else 0.0
    print(f"{name:<10} windows[{' '.join(f'{w:5.1f}' for w in sorted(wins, reverse=True))}] "
          f"majflt={maj:>7} swap0={swap0:>4} MiB draft_wall={max(draft) if draft else 0:4.1f} "
          f"disk={dload:5.1f} GiB load={float(LOAD.search(txt).group(1)) if LOAD.search(txt) else 0:5.1f} "
          f"KV={KV.search(txt).group(1) if KV.search(txt) else '?'}")


for p in sys.argv[1:]:
    try:
        report(p)
    except OSError as e:
        print(f"{p}: {e}")
