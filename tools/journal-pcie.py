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
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# PCIe correctable / uncorrectable AER bit names, so a bare `aer_status: 0x2000` in a CPER block is still
# readable. The kernel prints its own name for most events; this table covers the blocks where it does not.
COR_BITS = {0: "Receiver Error", 6: "Bad TLP", 7: "Bad DLLP", 8: "Replay Timer Timeout",
            12: "Replay Timer Timeout", 13: "Replay Number Rollover"}
TS = re.compile(r"^(\d{4}-\d{2}-\d{2}T[0-9:.]+(?:\+\d\d:\d\d)?)\s+\S+\s+kernel:\s+(.*)$")
# A PCI BDF, hex like everything else on this box. Both regexes below used \d for the bus, so ports named with a
# hex letter -- 0000:c4:00.0, the busiest AER source on this machine -- matched NOTHING, and the tool quietly
# reported a fraction of the events while sysfs said otherwise. tests/test_journal_mca.py pins it.
BDF = r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]"
PORT = re.compile(r"^(\S+ " + BDF + r"):\s+aer_status: 0x([0-9a-f]+)")
NAMED = re.compile(r"^(\S+ " + BDF + r"):\s+\[\s*\d+\]\s+(\w+)")
STATUS = re.compile(r"aer_(cor|uncor)_status: 0x([0-9a-f]+)")
CPER = re.compile(r"^\{(\d+)\}\[Hardware Error\]")
MCETIME = re.compile(r"\bTIME (\d+)\b")
NOISE = re.compile(r"ata\d+: SATA link down|SStatus")

# The line that actually names the killer, and the fields that turn it into a diagnosis. The kernel prints the
# MCA bank and the raw MCA_STATUS; everything else (which block, correctable or not, context intact or not) is
# in the status word, and the APIC id plus the CPU number say WHERE -- the CPU number is what makes two deaths
# one physical core or ten, because the SMT siblings of a core share its L2.
MCABANK = re.compile(r"CPU (\d+): Machine Check:\s*\d*\s*Bank (\d+): ([0-9a-f]{8,16})")
SEVERITY = re.compile(r"event severity: (\w+)")
APICID = re.compile(r"Local APIC_ID: (0x[0-9a-f]+)")
SYNDIPID = re.compile(r"SYND ([0-9a-f]+) IPID ([0-9a-f]+)")
BERTN = re.compile(r"BERT: Total records found: (\d+)")
FRU = re.compile(r"fru_text: (\S+)")
TOPOLOGY = Path(__file__).resolve().parent.parent / ".r9700-state" / "cpu-topology.json"
OFFLINE_CONF = Path(__file__).resolve().parent.parent / "host" / "offline-core.cpus"


def offline_conf():
    """host/offline-core.cpus -> {cpu: {core, package, siblings}}.

    Needed because the cores worth naming are usually offline by the time anyone reads this: sysfs has dropped
    their topology, and the snapshot only holds cpus that were online when it was taken. The conf is the
    human-verified pairing, written while they were still up."""
    try:
        txt = OFFLINE_CONF.read_text()
    except OSError:
        return {}
    cpus, meta = [], {}
    for ln in txt.splitlines():
        ln = ln.split("#")[0].strip()
        if not ln:
            continue
        if "=" in ln:
            for kv in ln.split():
                k, _, v = kv.partition("=")
                meta[k] = v
        else:
            cpus = [c for c in ln.split() if c.isdigit()]
    sib = ",".join(cpus)
    return {c: {"core": meta.get("core", "?"), "package": meta.get("package", "?"), "siblings": sib}
            for c in cpus}

# Architectural MCACOD families (AMD64 / Intel share this encoding): the low bits say the class, and for the
# cache class bits 3:2 carry the level. 0x0108 is therefore "cache error, level 2" -- an L2 failure.
MCACOD_CLASS = [(0x0000, "micro-architectural"), (0x00C0, "TLB"), (0x0100, "cache"),
                (0x0140, "bus/pipeline"), (0x0180, "master error"), (0x01C0, "virtual instruction"),
                (0x0181, "model-specific")]


def mca_fields(status_hex):
    """Decode an MCA_STATUS word: the flags that say how bad it is, and the codes that say what."""
    v = int(status_hex, 16)
    f = {"VAL": (v >> 63) & 1, "OVER": (v >> 62) & 1, "UC": (v >> 61) & 1, "EN": (v >> 60) & 1,
         "MSCOD": (v >> 16) & 0xFFFF, "MCACOD": v & 0xFFFF}
    f["AddrV"] = (v >> 58) & 1
    f["PCC"] = (v >> 57) & 1                      # processor context corrupt: the fault is not recoverable
    cod = f["MCACOD"]
    cls = next((name for mask, name in MCACOD_CLASS if (cod & 0xFF00) == mask and (cod & 0xF) in (0, 8)), None)
    cls = cls or next((name for mask, name in MCACOD_CLASS if (cod & 0xFFC0) == mask), "architectural 0x%04x" % cod)
    if cls == "cache":
        cls = f"cache, level {(cod >> 2) & 3}"
    f["class"] = cls
    return f


def topology_snapshot(path=None):
    """Write (cpu -> core, package, siblings) for every ONLINE cpu, while they are online.

    This exists because sysfs drops a cpu's topology directory the moment it is offlined -- so the tool that
    proves 'these two deaths are the same physical core' loses the proof exactly after the operator takes the
    surgical step of offlining that core. Snapshot first, then the record keeps its meaning."""
    p = Path(path) if path else TOPOLOGY
    out = {}
    for d in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*")):
        n = d.name[3:]
        if not n.isdigit() or not (d / "online").exists():
            continue
        core, pkg, sib = core_of(int(n))
        if core is not None:
            out[n] = {"core": core, "package": pkg, "siblings": sib}
    if out:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out))
    return out


def core_of(cpu, cache=None):
    """(core_id, package, sibling list) for a CPU number, from sysfs, falling back to the snapshot.

    This is the step that turned 'two CPUs' into 'one core': CPU 17 and CPU 113 looked like two suspects until
    thread_siblings_list said 17,113 -- the SMT siblings of a core, which share its L2."""
    t = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    try:
        sib = (t / "thread_siblings_list").read_text().strip()
        core = (t / "core_id").read_text().strip()
        pkg = (t / "physical_package_id").read_text().strip()
        return core, pkg, sib
    except (OSError, AttributeError):
        c = cache if cache is not None else (json.loads(TOPOLOGY.read_text()) if TOPOLOGY.exists() else {})
        e = c.get(str(cpu)) or offline_conf().get(str(cpu))
        return (e["core"], e["package"], e["siblings"]) if e else (None, None, None)



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
    return parse(out.splitlines())


def parse(lines):
    """One boot's kernel lines -> the records. Split out from scan() so the decoder can be tested against a
    captured block without a journal (and without a crashed box to crash on)."""
    live = []                 # (log_ts, port, what)
    names = defaultdict(Counter)
    replayed = []             # (log_ts, record_epoch, kind, detail)
    mce = []
    mca = []                  # (log_ts, cpu, bank, status_hex, fields)
    sev, apic, synd_ipid, bert, fru = [], [], None, None, []
    blocks = set()
    first = last = None
    for line in lines:
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
        for rx, bag in ((SEVERITY, sev), (APICID, apic), (FRU, fru)):
            g = rx.search(msg)
            if g:
                bag.append(g.group(1))
        b = BERTN.search(msg)
        if b:
            bert = int(b.group(1))
        si = SYNDIPID.search(msg)
        if si:
            synd_ipid = si.groups()
        mb = MCABANK.search(msg)
        if mb:
            mca.append((ts, int(mb.group(1)), int(mb.group(2)), mb.group(3), mca_fields(mb.group(3))))
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
    return live, names, replayed, mce, blocks, first, last, mca, sev, apic, synd_ipid, bert, fru


def mca_line(mca, sev, apic, synd_ipid):
    """One readable line for the thing that killed the boot, with the CPU reduced to a physical core."""
    if not mca:
        return None
    ts, cpu, bank, status, f = mca[0]
    core, pkg, sib = core_of(cpu)
    how = "fatal/uncorrectable" if f["UC"] else "correctable"
    how += ", context corrupt" if f["PCC"] else ""
    where = f"cpu {cpu}"
    if core is not None:
        where += f" = core {core} of package {pkg} (siblings {sib})"
    return (f"    MCA bank {bank} status 0x{status}: {f['class']}, {how}"
            f" | MSCOD 0x{f['MSCOD']:04x} MCACOD 0x{f['MCACOD']:04x}"
            + (f" | SYND 0x{synd_ipid[0]} IPID 0x{synd_ipid[1]}" if synd_ipid else "")
            + (f" | APIC {apic[0]}" if apic else "") + f" | {where}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("n", nargs="?", type=int, default=None,
                    help="how many boots to read (default 10); --boots is the same option")
    ap.add_argument("--boots", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--json", help="append one JSON line per boot to this file (the firmware keeps only ONE "
                                   "BERT record, so the archive has to be ours)")
    a = ap.parse_args()
    n = a.n or a.boots or 10
    topo = topology_snapshot()          # while online: the proof that two cpu numbers are one core
    arc = open(a.json, "a") if a.json else None
    for bid, start, end in boots(n):
        live, names, replayed, mce, blocks, first, last, mca, sev, apic, synd_ipid, bert, fru = scan(bid)
        print(f"\nboot {bid}  {start} -> {end}   (kernel lines to {last})")
        print(f"  AER events {len(live):>3} | CPER blocks {len(blocks):>4} | MCE lines {len(mce):>3}"
              + (f" | BERT records kept {bert}" if bert is not None else ""))
        # The machine check first: on this box it is the thing that ends boots, and the AER counters beside it
        # are a separate, chronic, mostly harmless story. Printing the counters first reads as "link problem".
        ml = mca_line(mca, sev, apic, synd_ipid)
        if ml:
            print(ml)
        if sev:
            print(f"    CPER severity {', '.join(sorted(set(sev)))}"
                  + (f" | fru_text {', '.join(sorted(set(fru)))}" if fru else ""))
        for port in sorted(names, key=lambda p: -sum(names[p].values())):
            print(f"    {port:<12} " + ", ".join(f"{k} {v}" for k, v in names[port].most_common()))
        # Records the firmware kept from the crash that ended THIS boot: printed by the NEXT boot.
        if replayed:
            for ts, epoch, _, _ in replayed[:3]:
                print(f"    record TIME {epoch} = {utc(epoch)} UTC, printed in this boot at {ts[11:19]}")
        for ts, msg in mce[-2:]:
            print(f"    mce {ts[11:19]}: {msg[:130]}")
        if arc:
            ts, cpu, bank, status, f = (mca[0] if mca else (None,) * 5)
            core, pkg, sib = core_of(cpu, topo) if cpu else (None, None, None)
            arc.write(json.dumps({"boot": bid, "start": start, "end": last,
                                  "aer_events": len(live), "cper_blocks": len(blocks),
                                  "bert": bert, "severity": (sev or [None])[0],
                                  "apic": (apic or [None])[0], "synd_ipid": list(synd_ipid or []),
                                  "mca": None if not mca else {"ts": ts, "cpu": cpu, "bank": bank,
                                                               "status": status, "class": f["class"],
                                                               "uc": f["UC"], "pcc": f["PCC"],
                                                               "mcacod": f"0x{f['MCACOD']:04x}",
                                                               "mscod": f"0x{f['MSCOD']:04x}",
                                                               "core": core, "package": pkg,
                                                               "siblings": sib},
                                  "replayed_at": [r[0] for r in replayed[:1]],
                                  "record_time": [r[1] for r in replayed[:1]]}) + "\n")
    if arc:
        arc.close()
        print(f"\nappended {n} boot records -> {a.json}")


if __name__ == "__main__":
    main()
