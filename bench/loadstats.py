#!/usr/bin/env python3
"""Per-instance resource stats from the footprint sampler, windowed by each launch's own log timestamps.

The sampler writes one continuous CSV, so a whole-file maximum spans several launches and is meaningless (that mistake
is what this tool exists to avoid). Every relaunch log carries wall-clock stamps for the lines that bracket the load
(`Model loading took ... seconds`, plus the engine's first line), so the window comes from the log itself.

    bench/loadstats.py ~/logs/relaunch-cold-late25.log ~/logs/relaunch-swap32-cold30.log
    bench/loadstats.py --csv ~/logs/loadsample-063808.csv ~/logs/relaunch-*.log
"""
import glob
import os
import re
import sys

STAMP = re.compile(r"INFO (\d\d-\d\d) (\d\d:\d\d:\d\d)")
LOAD = re.compile(r"Model loading took ([0-9.]+) GiB memory and ([0-9.]+) seconds")
HDR = {}


def window(log):
    """(first_seen, load_done) as HH:MM:SS, from the log's own stamps."""
    first = done = None
    with open(log, errors="replace") as f:
        for line in f:
            m = STAMP.search(line)
            if not m:
                continue
            t = m.group(2)
            if first is None:
                first = t
            if LOAD.search(line):
                done = t
                break
    return first, done


def hhmmss(t):
    h, m, s = (int(x) for x in t.split(":"))
    return h * 3600 + m * 60 + s


def stats(csv, a, b, pad=15):
    a, b = hhmmss(a) - 5, hhmmss(b) + pad
    rows = []
    with open(csv) as f:
        head = f.readline().strip().split(",")
        for line in f:
            c = line.strip().split(",")
            if len(c) < len(head):
                continue
            t = hhmmss(c[0])
            if a <= t <= b:
                rows.append(c)
    if not rows:
        return None
    def col(name):
        i = head.index(name)
        out = []
        for c in rows:
            try:
                out.append(float(c[i]))
            except ValueError:
                pass
        return out
    def num(name):
        v = col(name)
        return v
    avail, swap = num("memavail_gib"), num("swapused_mib")
    rd, wr = num("disk_rd_mbs"), num("disk_wr_mbs")
    rank, rss = num("rank_rd_mbs"), num("rss_sum_gib")
    return dict(n=len(rows), avail=min(avail), swap=max(swap), rd=max(rd), wr=max(wr),
                rank_peak=max(rank), rank_tot=sum(rank) * 2 / 1024.0, rss=max(rss))


def main():
    args = sys.argv[1:]
    csv = None
    if args and args[0] == "--csv":
        csv = os.path.expanduser(args[1])
        args = args[2:]
    if not args:
        print(__doc__)
        return 1
    if csv is None:
        csvs = sorted(glob.glob(os.path.expanduser("~/logs/loadsample-*.csv")), key=os.path.getmtime)
        csv = csvs[-1]
    print(f"csv: {csv}")
    print(f"{'instance':<26} {'window':<15} {'rows':>4} {'minAvail':>9} {'maxSwap':>8} {'peakRd':>7} "
          f"{'peakWr':>7} {'peakRankRd':>10} {'rankRdTot':>10} {'peakRSS':>8}")
    for log in args:
        a, b = window(os.path.expanduser(log))
        name = os.path.basename(log).replace("relaunch-", "").replace(".log", "")
        if not a or not b:
            print(f"{name:<26} no load line (or no stamps) in this log")
            continue
        s = stats(csv, a, b)
        if not s:
            print(f"{name:<26} {a+'-'+b:<15} no sampler rows in that window")
            continue
        print(f"{name:<26} {a+'-'+b:<15} {s['n']:>4} {s['avail']:>8.1f}G {s['swap']:>7.0f}M {s['rd']:>6.0f}M "
              f"{s['wr']:>6.0f}M {s['rank_peak']:>9.0f}M {s['rank_tot']:>9.1f}G {s['rss']:>7.1f}G")
    return 0


if __name__ == "__main__":
    sys.exit(main())
