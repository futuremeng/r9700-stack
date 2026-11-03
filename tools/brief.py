#!/usr/bin/env python3
"""One line per call: the state of the box, cheap enough to run every five minutes forever.

This exists because two long-running questions are being watched on this machine (2026-11-03): is the slot-6 PCIe
link still the only erroring chain and at what rate, and is the fatal-machine-check clock accumulating. The
detailed tools are `reseat-check.py` (full snapshot), `journal-pcie.py` (the journal's hardware records) and
`mca-hazard.py` (the statistics of "nothing has happened yet"). This is the line that goes in a log so the
series is readable without opening JSON:

    time        boot  threads  slot6 speed  slot6 cor  rate/min  other chains  serve
    08:53:10    324e  190      32.0         161        0.44      1/0/0         200

Rate is computed against the previous call (state in ~/aer/brief-prev.json), so it is a per-boot, per-window
rate -- the counters themselves reset at enumeration, and this box reboots itself, so an absolute number across
boots would be meaningless. The state therefore carries the boot id: across a boot boundary there is NO previous
sample, and the line prints no rate rather than a negative one.

Read-only: sysfs, one /health request, no writes outside ~/aer.
"""
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "serve"))
import webui  # noqa: E402

STATE = Path(os.path.expanduser("~/aer/brief-prev.json"))


def prev_window(prev, boot):
    """The previous sample, but only if it came from THIS boot.

    AER counters are zeroed at enumeration, so a delta across a boot boundary is not a rate: it is the
    difference of two unrelated counters, and it goes NEGATIVE whenever the machine came up quieter than the
    boot it replaced. This box powers itself off and on, so that is not hypothetical."""
    return prev if prev.get("b") == boot else {}


def main():
    chains = webui.read_pcie("/sys/bus/pci/devices")
    out, prev = {}, json.loads(STATE.read_text()) if STATE.exists() else {}
    now = time.time()
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()[:4]
    prev = prev_window(prev, boot)
    for ch in chains:
        b = ch["bus"]
        dev = ch.get("devices") or ch.get("chain") or []   # read_pcie calls it "devices"; reseat-check renames it to "chain" when snapshotting
        cor = sum(p.get("cor", 0) for p in dev)
        ext = min((p.get("speed", 0) for p in dev), default=0)
        out[b] = {"cor": cor, "ext": ext, "card": ch.get("card"), "slot": ch.get("slot")}
    dt = (now - prev.get("t", now)) / 60.0 if prev else 0.0
    sick = sorted(out.items(), key=lambda kv: -kv[1]["cor"])[:1]
    # /proc/cpuinfo, not sysfs: cpu0 has no `online` file (it cannot be offlined), so counting sysfs `online==1`
    # reads one low and would make the offline-core intervention look different from what `nproc` shows
    threads = sum(1 for ln in Path("/proc/cpuinfo").read_text().splitlines() if ln.startswith("processor"))
    parts = [datetime.now(timezone.utc).strftime("%H:%M:%S"),
             boot,
             f"threads {threads}"]
    for b, v in sorted(out.items(), key=lambda kv: str(kv[1]["slot"])):
        rate = ""
        if prev and dt > 0 and b == sick[0][0]:
            d = v["cor"] - prev["c"].get(b, {}).get("cor", v["cor"])
            rate = f" {d / dt:5.2f}/min"
        parts.append(f"slot{v['slot']}({b}) {v['ext']:.0f}GT cor={v['cor']}{rate}")
    try:
        with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=4) as r:
            parts.append(f"serve {r.status}")
    except Exception as e:
        parts.append(f"serve {type(e).__name__}")
    line = "  ".join(parts)
    print(line)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"t": now, "b": boot, "c": out}))
    log = STATE.parent / "series.log"
    with log.open("a") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    main()
