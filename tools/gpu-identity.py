#!/usr/bin/env python3
"""Which physical GPU is sitting in which slot -- the identity lock a card swap needs.

Why this exists: a card swap is the decisive test for "is the marginal PCIe link the card or the slot", and it is
worthless unless you can say afterwards which physical card ended up where. The PCI identity (BDF, slot number)
follows the SLOT; the `ASIC_SERIAL` from amd-smi follows the CARD. Printing both before and after the swap is
what turns "I moved the cards" into a result that means something.

Pairing note: rocm-smi's GPU[N] ordering is not the same ordering amd-smi uses, and neither is guaranteed stable
across reboots -- so both fields are read from ONE `amd-smi static --asic --bus` call, per GPU block, and joined
to the slot by BDF. Save the output before and after a physical change:

    tools/gpu-identity.py > ~/aer/identity-$(date +%m%d-%H%M).txt

Read-only: it runs amd-smi and reads sysfs, and sends nothing to the serve.
"""
import json
import re
import subprocess
import sys
from pathlib import Path


def amd_static():
    out = subprocess.run(["amd-smi", "static", "--asic", "--bus"],
                         capture_output=True, text=True).stdout
    gpus = []
    cur = {}
    for ln in out.splitlines():
        m = re.match(r"^GPU: (\d+)\s*$", ln)
        if m:
            if cur:
                gpus.append(cur)
            cur = {"index": int(m.group(1))}
            continue
        m = re.match(r"^\s*(ASIC_SERIAL|BDF|MARKET_NAME|MAX_PCIE_SPEED|MAX_PCIE_WIDTH):\s*(.+?)\s*$", ln)
        if m:
            cur[m.group(1)] = m.group(2)
    if cur:
        gpus.append(cur)
    return gpus


def slots():
    """gpu BDF -> (link speeds of its chain), from the same sysfs the watcher reads.

    No slot label here: sysfs `physical_slot` exists only on hotplug-capable slots, and these GPU ports have no
    pciehp (which is itself a fact -- it is why pulling that expansion board live would have produced no neat
    Surprise-Removal log line). The BDF identifies the slot anyway; tools/reseat-check.py is the file that
    labels them (slot 1 = 03:00.0, slot 3 = c3:00.0, slot 6 = c6:00.0, slot 8 = 06:00.0 on this box).
    """
    out = {}
    for p in Path("/sys/bus/pci/devices").glob("0000:*"):
        bdf = p.name
        cls = (p / "class").read_text().strip() if (p / "class").exists() else ""
        if not cls.startswith("0x03"):         # any display controller (these report 0x030000, not 0x0302xx)
            continue
        try:
            spd = (p / "current_link_speed").read_text().strip()
        except OSError:
            spd = "?"
        # walk up to the root port of this chain and note its speed: that is the external hop
        up = p.resolve()
        ext = []
        for anc in up.parents:
            d = Path("/sys/bus/pci/devices") / anc.name
            if d.exists() and (d / "current_link_speed").exists():
                ext.append((anc.name, d.joinpath("current_link_speed").read_text().strip()))
        slot = "?"
        for anc in up.parents:
            d = Path("/sys/bus/pci/devices") / anc.name / "physical_slot"
            if d.exists():
                v = d.read_text().strip()
                if v:
                    slot = v          # present only on hotplug-capable slots; empty here, see the docstring
                    break
        out[bdf.split(":", 1)[1]] = {"slot": slot, "gpu_link": spd, "chain": ext[:3]}
    return out


def main():
    g = amd_static()
    s = slots()
    print(f"at {subprocess.run(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], capture_output=True, text=True).stdout.strip()}")
    print(f"{'amd idx':>7}  {'ASIC_SERIAL (follows the card)':30} {'BDF (follows the slot)':20}  "
          f"{'gpu hop':14} chain up to the root port")
    for x in sorted(g, key=lambda x: x.get("BDF", "")):
        bdf = x.get("BDF", "?")
        short = bdf.split(":", 1)[1] if ":" in bdf else bdf
        sl = s.get(short, {})
        ch = ", ".join(f"{b}={v}" for b, v in sl.get("chain", []))
        print(f"{x.get('index', '?'):>7}  {x.get('ASIC_SERIAL', '?'):30} {bdf:20}  "
              f"{sl.get('gpu_link', '?'):14} {ch}")
    print("\nBDF/slot identity follows the SLOT; ASIC_SERIAL follows the CARD. Compare a before/after pair by "
          "serial to know which card moved where. amd-smi's GPU[N] ordering is NOT stable -- match on serial.")


if __name__ == "__main__":
    main()
