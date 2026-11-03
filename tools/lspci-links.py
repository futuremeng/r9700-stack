#!/usr/bin/env python3
"""Read a saved `lspci -vvv` dump and print the PCIe link negotiation of each GPU chain, side by side.

Why a file instead of asking whoever is at the box to grep: lspci puts a TAB after `LnkSta:` and friends, so a
grep written as 'LnkSta: ' matches nothing and the one decisive line -- `LnkSta: Speed 16GT/s (downgraded)` --
silently disappears from the output. And the other decisive line, `LnkCtl2: Target Link Speed`, is what separates
"the link tried Gen5 and gave it up by itself" from "something told it to run Gen4". Save the dump once
(`sudo lspci -vvv > ~/lspci-vvv-$(date +%m%d-%H%M).txt`) and read everything out of it here.

What to look for, and what it means on this box:
  LnkCap   what the port can do            (all four chains here: 32GT/s x16, Gen5)
  LnkCtl2  Target Link Speed               (still 32GT/s => Gen5 was asked for)
  LnkSta   current speed, and (downgraded) (the flag = an autonomous/hardware change, not a commanded one)
  AER      which error bits are masked     (a masked correctable never reaches the log, so absence of a named
                                           event does not mean absence of errors -- the counters say more)

Usage: tools/lspci-links.py ~/lspci-vvv-1103-0740.txt [--all]
"""
import re
import sys
from pathlib import Path

GPU_CLASS = re.compile(r"\b(VGA compatible controller|Display controller)\b")
FIELDS = ("LnkCap:", "LnkCtl:", "LnkCtl2:", "LnkSta:", "DevCtl2:", "UESta:", "CESta:", "UEMsk:", "CEMsk:")


def blocks(text):
    """lspci -vvv -> {bdf: [lines]}. lspci omits the domain unless -D is given, so accept both spellings and
    normalize to the 0000: form the sysfs paths use."""
    out, cur = {}, None
    for ln in text.splitlines():
        m = re.match(r"^(?:([0-9a-f]{4}):)?([0-9a-f]{2}:[0-9a-f]{2}\.[0-7])\s", ln)
        if m:
            cur = (m.group(1) or "0000") + ":" + m.group(2)
            out[cur] = [ln]
        elif cur:
            out[cur].append(ln)
    return out


def grab(lines, field):
    for i, ln in enumerate(lines):
        if re.search(r"\b" + re.escape(field), ln):
            val = ln.split(":", 1)[1].strip() if ":" in ln else ""
            nxt = lines[i + 1].strip() if i + 1 < len(lines) and lines[i + 1].strip() and \
                not re.match(r"^\s{1,8}\w+\w*:", lines[i + 1]) else ""
            return val + ((" | " + nxt) if nxt else "")
    return "-"


def chain(bdf):
    """BDF -> its chain up to the root port, from sysfs (works whether or not lspci saw the hierarchy)."""
    p = Path(f"/sys/bus/pci/devices/{bdf}")
    if not p.exists():
        return [bdf]
    return [a.name for a in reversed(p.resolve().parents) if (Path("/sys/bus/pci/devices") / a.name).exists()]


def main():
    if len(sys.argv) < 2:
        print(__doc__.splitlines()[-2])
        sys.exit(2)
    b = blocks(Path(sys.argv[1]).read_text(errors="replace"))
    show_all = "--all" in sys.argv
    gpus = [bdf for bdf, ls in b.items() if GPU_CLASS.search(ls[0]) and bdf.split(":")[1] != "00"]
    gpus = [g for g in gpus if (Path("/sys/bus/pci/devices") / g).exists()]
    for g in sorted(gpus):
        print(f"\n=== GPU {g}   chain: {' <- '.join(chain(g))}")
        for d in chain(g):
            ls = b.get(d, [])
            if not ls:
                print(f"  {d}: not in the dump")
                continue
            print(f"  {d}  {ls[0].split(':', 1)[1].split('(')[0].strip()[:38]}")
            for f in FIELDS:
                v = grab(ls, f)
                if v != "-" and (show_all or f in ("LnkCap:", "LnkCtl2:", "LnkSta:", "CEMsk:", "UEMsk:")):
                    print(f"      {f:<8} {v}")


if __name__ == "__main__":
    main()
