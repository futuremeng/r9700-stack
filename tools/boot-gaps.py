#!/usr/bin/env python3
"""Power-off windows from the journal's boot list: the gap between the last entry of one boot and the first of
the next is how long the box was actually down. On this box that number says whether a physical intervention
(open chassis, pull an expansion board) could have happened inside a given reboot, which is exactly the question
asked on 2026-11-03: today's reboot gaps were under three minutes, and a board reseat is not a 100-second job.

Read-only. Usage: tools/boot-gaps.py [days-back]
"""
import re
import subprocess
import sys
from datetime import datetime

TS = re.compile(r"\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun) (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC")
FMT = "%Y-%m-%d %H:%M:%S"


def boots():
    out = subprocess.run(["journalctl", "--list-boots", "--no-pager"],
                         capture_output=True, text=True).stdout
    rows = []
    for ln in out.splitlines():
        idx = re.match(r"\s*(-?\d+)", ln)
        ts = TS.findall(ln)
        if idx and len(ts) >= 2:
            d = [datetime.strptime(t[1], FMT) for t in ts[-2:]]
            rows.append((int(idx.group(1)), d[0], d[1]))
    rows.sort()
    return rows


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    rows = boots()
    prev = None
    print("boot  最后一次记录            下一次开机              关机时长      说明")
    for n, first, last in rows:
        if prev is not None:
            gap = (first - prev).total_seconds()
            if gap >= 60:
                note = "<-- 足够开箱动手（>=5 min）" if gap >= 300 else "太短：像一次 reboot + POST"
                print(f"{n:>4}  {prev:%m-%d %H:%M:%S}  ->  {first:%m-%d %H:%M:%S}   {gap/60:6.1f} min   {note}")
        prev = last
    print(f"\n（{len(rows)} 个 boot；关机时长 = 上一 boot 最后日志 到 下一 boot 首条日志，"
          f"POST 与掉电无法区分，但足以判断一次开箱操作可不可能发生在这个窗口里）")


if __name__ == "__main__":
    main()
