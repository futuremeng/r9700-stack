#!/usr/bin/env python3
"""How long this box survives a fatal machine check, and how much clean uptime it takes to say anything.

The question this exists to answer honestly (asked on 2026-11-03, after offlining the core whose L2 was throwing
the fatal MCA): "three boots without a fatal record -- is the CPU fine now?" The answer is a number, and it comes
from the box's own history, not from a feeling:

  * each boot's start time comes from `journalctl --list-boots`;
  * a fatal event's TIME is in the record the NEXT boot replays (`mce: ... TIME <epoch>` / `record TIME <epoch>`),
    which is also why looking at the dying boot finds nothing;
  * lifetime-before-death = that TIME minus the dying boot's start;
  * and then: the probability of seeing NO fatal event over T hours of clean uptime, given the observed cadence.

That last number is the one to quote. If it is not small, "nothing happened yet" is not evidence.

Usage: tools/mca-hazard.py [--since "2026-11-03 06:49:24"]   (defaults to the last fatal event)
"""
import math
import re
import subprocess
import sys
from datetime import datetime, timezone

FMT = "%Y-%m-%d %H:%M:%S"
MCETIME = re.compile(r"\bTIME (\d{9,11})\b")


def boots():
    out = subprocess.run(["journalctl", "--list-boots", "--no-pager"], capture_output=True, text=True).stdout
    ts = re.compile(r"\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun) (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC")
    rows = []
    for ln in out.splitlines():
        m = re.match(r"\s*(-?\d+)", ln)
        f = ts.findall(ln)
        if m and len(f) >= 2:
            rows.append((int(m.group(1)),
                         datetime.strptime(f[-2][1], FMT).replace(tzinfo=timezone.utc),
                         datetime.strptime(f[-1][1], FMT).replace(tzinfo=timezone.utc)))
    rows.sort()
    return rows


def replayed(bid, start, window=180):
    """A record TIME printed within `window` seconds of this boot's start belongs to the PREVIOUS boot's death."""
    out = subprocess.run(["journalctl", "-k", "-b", bid, "--no-pager", "-o", "short-iso"],
                             capture_output=True, text=True).stdout
    hits = []
    for ln in out.splitlines():
        m = re.match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", ln)
        t = MCETIME.search(ln)
        if not (m and t):
            continue
        epoch = int(t.group(1))
        try:
            printed = datetime.fromisoformat(m.group(1).replace("T", " ")).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if abs((printed - start).total_seconds()) <= window:
            hits.append(epoch)
    return hits


def main():
    since = None
    if "--since" in sys.argv:
        since = datetime.strptime(sys.argv[sys.argv.index("--since") + 1], FMT).replace(tzinfo=timezone.utc)
    bs = boots()
    lives, clean = [], []
    for (n, a_start, a_end), (n2, b_start, b_end) in zip(bs, bs[1:]):
        h = replayed(str(n2), b_start)
        died = datetime.fromtimestamp(max(h), timezone.utc) if h else None
        if died and a_start < died <= b_start.replace(tzinfo=None).replace(
                tzinfo=timezone.utc) + __import__("datetime").timedelta(seconds=60):
            lives.append(((died - a_start).total_seconds() / 3600.0, a_start, died))
        else:
            clean.append(((a_end - a_start).total_seconds() / 3600.0, a_start))
    print(f"{len(bs)} boots in the journal | {len(lives)} ended in a fatal machine check | "
          f"{len(clean)} without one")
    if lives:
        hrs = sorted(l[0] for l in lives)
        print("\n寿命分布（ fatal 事件前这个 boot 活了多久，小时）：")
        print(f"   min {hrs[0]:7.3f}   p25 {hrs[len(hrs)//4]:7.3f}   中位 {hrs[len(hrs)//2]:7.3f}   "
              f"p75 {hrs[3*len(hrs)//4]:7.3f}   max {hrs[-1]:7.3f}")
        print("   最短的几个（这就是 boot-time 风险，不是运行风险）：" +
              ", ".join(f"{h:.3f}h" for h in hrs[:6]))
        mean = sum(hrs) / len(hrs)
        span = (bs[-1][2] - bs[0][1]).total_seconds() / 3600.0
        print(f"   平均 {mean:.2f} h；观测窗 {span:.1f} h 内 {len(lives)} 次 fatal => 节奏约 1/{span/len(lives):.1f} h")
        # what the observation has to buy
        for p in (0.5, 0.1, 0.05, 0.01):
            print(f"   要让「什么都没发生」变成 p={p:.2f} 的反证，需要连续无 fatal 运行 {-mean*math.log(p):.1f} h")
        last = max(l[2] for l in lives)
        ref = since or last
        up = 0.0
        for n, s, e in bs:
            if s >= ref:
                up += (min(e, datetime.now(timezone.utc)) - s).total_seconds() / 3600.0
        print(f"\n自 {ref:%m-%d %H:%M:%S} 以来累计开机 {up:.2f} h，其中 fatal 事件 0 次")
        print(f"   如果故障完全没有变化，看到这一串干净的概率 = exp(-t/mean) = {math.exp(-up/mean):.3f}")
        print("   " + ("=> 这个读数有信息量" if math.exp(-up/mean) < 0.05
                       else "=> 这个读数还不足以说明故障变了：它完全在原来节奏的沉默里"))
        print("\n（注意：hazard 不是常数 —— 上面「最短的几个」说明有一部分风险在开机头几十秒，")
        print(" offline 那个 unit 在 18 s 才执行，那段窗口没有被覆盖。）")


if __name__ == "__main__":
    main()
