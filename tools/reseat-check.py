#!/usr/bin/env python3
"""reseat-check — what to run before and after a GPU is pulled out of the machine and put back.

`card2` is a name the driver assigned this boot. It says nothing about the chassis, and after a reseat the two
things that can silently change are exactly the two things a person needs to know about: the BDF (a card in a
different slot enumerates at a different address, so "card2" may be a different card) and the link the BIOS
trained when it came back up. This reads both from sysfs — no root, no /dev/kfd, and no request to the serve,
so it works while the serve is down (which is usually when a card gets reseated).

    python3 tools/reseat-check.py                      # the table: card, GPU BDF, slot, chain, link, AER
    python3 tools/reseat-check.py --save before.json   # the state to compare against
    python3 tools/reseat-check.py --diff before.json   # what changed after

Measured on this box (2026-11-03): four GPUs in firmware slots 1, 3, 6 and 8 —

    card3  0000:03:00.0  slot 1     card1  0000:c3:00.0  slot 3
    card2  0000:c6:00.0  slot 6     card4  0000:06:00.0  slot 8

— with card2 the card whose switch uplink (c4:00.0 <-> c0:03.1) had been sitting at 16.0 GT/s of its 32.0 since
the boot, on the chain carrying 144 correctable AER events.

Two honesty notes printed in the diff too:

- AER counters are **per boot** and zeroed when the kernel enumerates, so after a reseat-boot their totals are
  not comparable — growth inside one boot is what means anything.
- `speed`/`width` are the *negotiated* values; the *target* the BIOS asked for is in the configuration space,
  which ordinary users cannot read here (`lspci -vvv -s c4:00.0` answers `Capabilities: <access denied>`).
  Confirming "the link was asked for Gen5 and trained Gen4" needs root, and is the question a reseat answers.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "serve"))
import webui                                                        # noqa: E402


def snapshot():
    gpus = webui.read_gpus("/sys/class/drm")
    # read_gpus reports full BDFs ("0000:c3:00.0"); the chain summary strips the domain ("c3:00.0").
    by_bus = {(g.get("bus") or "").replace("0000:", ""): g.get("card") for g in gpus}
    chains = webui.read_pcie("/sys/bus/pci/devices")
    slot_by_ep = {ch["bus"]: ch.get("slot") for ch in chains if ch.get("slot")}
    for g in gpus:
        g["slot"] = slot_by_ep.get((g.get("bus") or "").replace("0000:", "").lower())
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        boot = None
    out = {"at": time.time(), "boot_id": boot, "uptime_s": round(float(
        Path("/proc/uptime").read_text().split()[0]), 1) if os.path.exists("/proc/uptime") else None,
        "cards": []}
    self_busy = {(g.get("bus") or "").replace("0000:", ""): g.get("busy_pct") for g in gpus}
    for ch in chains:
        ch["card"] = by_bus.get(ch["bus"])                     # name the chain by the card sysfs calls it
    for ch in sorted(chains, key=lambda c: (c.get("card") or c["bus"])):
        def sum_(k):
            return sum((d.get(k) or 0) for d in ch["devices"])
        out["cards"].append({
            "card": ch.get("card"), "gpu_bus": ch["bus"], "slot": ch.get("slot"),
            "slot_at": ch.get("slot_at"), "parked": ch.get("parked"),
            "chain": [{"bdf": d["bdf"], "role": d["role"], "speed": d.get("speed"),
                       "width": d.get("width"), "max_speed": d.get("max_speed"),
                       "max_width": d.get("max_width"), "power_state": d.get("power_state"),
                       "cor": d.get("cor"), "nonfatal": d.get("nonfatal"), "fatal": d.get("fatal"),
                       # The named split is the diagnosis: on this box both riser chains count their bad packets
                       # at the SWITCH UPSTREAM PORT (BadTLP 77 on c4:00.0, 50 on 04:00.0) while the root ports
                       # count retries (Replay Timer Timeout 62 on c0:03.1, 7 on 00:03.1) -- the same link seen
                       # from both ends. Totals alone would lose that, and totals are per boot anyway.
                       "notable": d.get("notable") or {}}
                      for d in ch["devices"]],
            "aer_cor": sum_("cor"), "aer_nonfatal": sum_("nonfatal"), "aer_fatal": sum_("fatal"),
            # What the card itself says, when sysfs will answer — busy/VRAM/clocks come from the same sysfs the
            # Monitor reads, so a reseat check and a Monitor screenshot agree by construction.
            "busy_pct": self_busy.get(ch["bus"]),
        })
    return out


def table(s):
    print(f"# {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(s['at']))}"
          f"  boot {str(s['boot_id'])[:8]}  up {s['uptime_s']} s")
    for c in s["cards"]:
        print(f"{(c['card'] or '?'):6} GPU {c['gpu_bus']:9} slot {str(c['slot'] or '-'):4}"
              f"  parked={c['parked']}  AER cor={c['aer_cor']} unc={c['aer_nonfatal'] + c['aer_fatal']}")
        for d in c["chain"]:
            flag = ""
            if d["speed"] and d["max_speed"] and d["speed"] < d["max_speed"]:
                flag = f"  <-- below its maximum (max {d['max_speed']})" + (", parked" if c["parked"] else "")
            print(f"    {d['bdf']:16} {d['role']:13} {d['speed']} GT/s x{d['width']:<3}"
                  f" state={d['power_state']}{flag}")


def diff(old, new):
    lines = []
    ob = {c["gpu_bus"]: c for c in old["cards"]}
    nb = {c["gpu_bus"]: c for c in new["cards"]}
    for bus in sorted(set(ob) - set(nb)):
        lines.append(f"MISSING after: {ob[bus]['card']} was GPU {bus} in slot {ob[bus]['slot']}")
    for bus in sorted(set(nb) - set(ob)):
        lines.append(f"NEW after: {nb[bus]['card']} at GPU {bus} in slot {nb[bus]['slot']}")
    for bus in sorted(set(ob) & set(nb)):
        a, b = ob[bus], nb[bus]
        if a["slot"] != b["slot"]:
            lines.append(f"{a['card']}: slot {a['slot']} -> {b['slot']} (the card is in a different physical slot)")
        if [d["bdf"] for d in a["chain"]] != [d["bdf"] for d in b["chain"]]:
            lines.append(f"{a['card']}: chain {[d['bdf'] for d in a['chain']]} -> {[d['bdf'] for d in b['chain']]}")
        # Named counters, per node, before and after. `nb` above is the card map -- shadowing it here broke the
        # loop after the first card (KeyError on the second), which is the kind of bug a diff tool hides until
        # the moment you need it, so the names here are deliberately long.
        for da, db in zip(a["chain"], b["chain"]):
            names_before = {k: v for k, v in (da.get("notable") or {}).items() if v}
            names_after = {k: v for k, v in (db.get("notable") or {}).items() if v}
            if names_before and not names_after and da.get("cor") != db.get("cor"):
                pass                                        # older snapshot without the named split: no claim
            if names_before and names_after and names_before != names_after:
                lines.append(f"{a['card']} {da['bdf']}: counters moved -- before: "
                             + (" ".join(f"{k} {v}" for k, v in names_before.items()) or "quiet")
                             + " | after: " + (" ".join(f"{k} {v}" for k, v in names_after.items()) or "quiet"))
            if names_before and not names_after:
                lines.append(f"{a['card']} {da['bdf']}: counters zeroed at enumeration (new boot); before this "
                             f"boot read: " + " ".join(f"{k} {v}" for k, v in na.items()))
            if (da["speed"], da["width"]) != (db["speed"], db["width"]):
                lines.append(f"{a['card']} {da['bdf']}: {da['speed']} GT/s x{da['width']} -> "
                             f"{db['speed']} GT/s x{db['width']}"
                             + ("  (back to its maximum)" if db["speed"] == db["max_speed"] else ""))
    if old["boot_id"] != new["boot_id"]:
        lines.append("different boot: AER counters are zeroed at enumeration, so compare growth within one boot,"
                     " not these totals")
    else:
        # Same boot, so the only number worth having is a RATE: correctables per minute between the two
        # snapshots. Totals are per boot and this box reboots on its own, so a total can never be compared
        # across the reseat -- a rate can, and a rate is what answers "did reseating change anything".
        mins = (new["at"] - old["at"]) / 60.0
        if mins > 0.5:
            for bus in sorted(set(ob) & set(nb)):
                a, b = ob[bus], nb[bus]
                d = (b["aer_cor"] + b["aer_nonfatal"] + b["aer_fatal"]) \
                    - (a["aer_cor"] + a["aer_nonfatal"] + a["aer_fatal"])
                lines.append(f"{a['card']} slot {b['slot']} {bus}: {a['aer_cor']} -> {b['aer_cor']} correctable"
                             f" ({d:+d} in {mins:.1f} min = {d / mins:.2f}/min)")
    return lines or ["no change in slot assignment, BDF, chain, link speed or width"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", help="write this snapshot to a JSON file")
    ap.add_argument("--diff", help="compare the current state against a snapshot saved earlier")
    a = ap.parse_args(argv)
    s = snapshot()
    table(s)
    if a.save:
        Path(a.save).write_text(json.dumps(s, indent=1))
        print(f"\nsaved -> {a.save}")
    if a.diff:
        print("\n## compared with " + a.diff)
        for ln in diff(json.loads(Path(a.diff).read_text()), s):
            print("  " + ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
