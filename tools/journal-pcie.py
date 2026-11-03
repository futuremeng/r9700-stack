#!/usr/bin/env python3
"""Per-boot PCIe / MCE timeline from the persistent kernel journal.

Why this exists: the sysfs AER counters the Monitor reads are per-boot totals with no timestamps, so the page
can only say "144 since boot". The kernel journal carries the same events WITH time, per port, per error name
(`pcieport 0000:c4:00.0:    [ 6] BadTLP`), the ACPI/CPER blocks the firmware pushes through GHES
(`{37}[Hardware Error]: aer_cor_status: ...`), and CPU machine-check records. It also replays, at the START of
a boot, the records the firmware kept from the crash that ended the previous one -- carrying the record's own
`TIME` field. That is what turns "the box died with nothing in the log" into "the box died and left a CPU
machine check whose record says when".

Two categories are kept apart on purpose, because conflating them is how you blame today's traffic for
yesterday's fault:
  live      -- the record arrived while this boot was running (log time == record time)
  replayed  -- the record was recovered at boot from the previous crash (its TIME predates this boot)

Read-only: `journalctl --list-boots` and `journalctl -k -b <id>` for the last N boots (default 10).

    tools/journal-pcie.py            # the last 10 boots
    tools/journal-pcie.py 4          # the last 4
    tools/journal-pcie.py --boots 4  # same thing

The count used to be `int(sys.argv[1])`, so `--boots 4` -- the flag this file's own header suggests -- raised a
ValueError traceback. Asking for the last few boots is exactly the moment a tool has to answer instead of
throwing: it is run while something is already wrong.
"""
import argparse
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

# PCIe correctable / uncorrectable AER bit names, so a bare `aer_status: 0x2000` in a CPER block is still
# readable. The kernel prints its own name for most events; this table covers the blocks where it does not.
COR_BITS = {0: "Receiver Error", 6: "Bad TLP", 7: "Bad DLLP", 8: "Replay Timer Timeout",
            12: "Replay Timer Timeout", 13: "Replay Number Rollover"}
TS = re.compile(r"^(\d{4}-\d{2}-\d{2}T[0-9:.]+(?:\+\d\d:\d\d)?)\s+\S+\s+kernel:\s+(.*)$")
PORT = re.compile(r"^(\S+ \d{4}:\d{2}:\d{2}\.\d+):\s+aer_status: 0x([0-9a-f]+)")
NAMED = re.compile(r"^(\S+ \d{4}:\d{2}:\d{2}\.\d+):\s+\[\s*\d+\]\s+(\w+)")
STATUS = re.compile(r"aer_(cor|uncor)_status: 0x([0-9a-f]+)")
CPER = re.compile(r"^\{(\d+)\}\[Hardware Error\]")
MCETIME = re.compile(r"\bTIME (\d+)\b")
NOISE = re.compile(r"ata\d+: SATA link down|SStatus")


def boots(n):
    out = subprocess.run(["journalctl", "--list-boots", "--no-pager"],
                         capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 6 and p[0].lstrip("-").isdigit():
            rows.append((p[0], " ".join(p[2:4]), " ".join(p[4:6])))
    return rows[-n:]


def decode(mask):
    v = int(mask, 16)
    return ", ".join(name for bit, name in sorted(COR_BITS.items()) if v & (1 << bit)) or f"0x{mask}"


def utc(epoch):
    return datetime.fromtimestamp(int(epoch), timezone.utc).strftime("%H:%M:%S")


def scan(bid):
    out = subprocess.run(["journalctl", "-k", "-b", bid, "--no-pager", "-o", "short-iso"],
                         capture_output=True, text=True).stdout
    live = []                 # (log_ts, port, what)
    names = defaultdict(Counter)
    replayed = []             # (log_ts, record_epoch, kind, detail)
    mce = []
    blocks = set()
    first = last = None
    for line in out.splitlines():
        m = TS.match(line)
        if not m:
            continue
        ts, msg = m.group(1), m.group(2).strip()
        first = first or ts
        last = ts
        if NOISE.search(msg):
            continue
        pm = PORT.match(msg)
        if pm:
            live.append((ts, pm.group(1).split()[-1], f"aer_status {decode(pm.group(2))}"))
            continue
        nm = NAMED.match(msg)
        if nm:
            names[nm.group(1).split()[-1]][nm.group(2)] += 1
            continue
        cm = CPER.match(msg)
        if cm:
            blocks.add(cm.group(1))
        st = STATUS.search(msg)
        if st:
            kind, mask = st.groups()
            if int(mask, 16):
                live.append((ts, "CPER", f"{kind} {decode(mask)}"))
        tm = MCETIME.search(msg)
        if tm:
            # The record's own stamp. At boot start it predates the boot: that is the previous death speaking.
            replayed.append((ts, tm.group(1), "record", msg[:110]))
        if msg.startswith("mce:") or "Machine check" in msg:
            mce.append((ts, msg))
    return live, names, replayed, mce, blocks, first, last


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("n", nargs="?", type=int, default=None,
                    help="how many boots to read (default 10); --boots is the same option")
    ap.add_argument("--boots", type=int, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    n = a.n or a.boots or 10
    for bid, start, end in boots(n):
        live, names, replayed, mce, blocks, first, last = scan(bid)
        print(f"\nboot {bid}  {start} -> {end}   (kernel lines to {last})")
        print(f"  AER events {len(live):>3} | CPER blocks {len(blocks):>4} | MCE lines {len(mce):>3}")
        for port in sorted(names, key=lambda p: -sum(names[p].values())):
            print(f"    {port:<12} " + ", ".join(f"{k} {v}" for k, v in names[port].most_common()))
        # Records the firmware kept from the crash that ended THIS boot: printed by the NEXT boot.
        if replayed:
            for ts, epoch, _, _ in replayed[:3]:
                print(f"    record TIME {epoch} = {utc(epoch)} UTC, printed in this boot at {ts[11:19]}")
        for ts, msg in mce[-2:]:
            print(f"    mce {ts[11:19]}: {msg[:130]}")


if __name__ == "__main__":
    main()
