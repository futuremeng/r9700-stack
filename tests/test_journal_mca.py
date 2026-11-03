#!/usr/bin/env python3
"""tests/test_journal_mca.py - the machine-check decoder, against a block the box really left behind.

Why this file exists: the tool it tests is what finally answered "why does this box die", and it answered it by
decoding fields that were already in the journal and were never printed (MCA bank, MCACOD, UC/PCC, and the CPU
number reduced to a physical core). A decoder like that has to be pinned by a test, because the failure mode is
silent: a regex that stops matching prints "no MCA" about a crash that did happen, which is exactly the wrong
answer to have wrong.

The captured block is boot `ec0d0751`'s opening lines -- the record the firmware kept from the crash that ended
the PREVIOUS boot (that is where a death's record lives: not in the boot that died, in the next one) -- with the
kernel's own lines in it verbatim.

Run: ./.venv/bin/python3 tests/test_journal_mca.py      (stdlib only; reads no journal)
"""
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import importlib.util
spec = importlib.util.spec_from_file_location("journal_pcie", os.path.join(ROOT, "tools", "journal-pcie.py"))
J = importlib.util.module_from_spec(spec)
spec.loader.exec_module(J)

fail = 0


def ck(name, cond, detail=""):
    global fail
    if not cond:
        fail += 1
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond or not detail else f"  {detail}"))


# Verbatim from `journalctl -k -b -1 -o short-iso` on this box (2026-11-03), trimmed to the record.
BOOT = [
    "2026-11-03T05:47:29+00:00 jiulu-9700 kernel: Linux version 7.0.0-42-generic",
    "2026-11-03T05:47:29+00:00 jiulu-9700 kernel: {1}[Hardware Error]: event severity: fatal",
    "2026-11-03T05:47:29+00:00 jiulu-9700 kernel: {1}[Hardware Error]:  fru_text: ProcessorError",
    "2026-11-03T05:47:29+00:00 jiulu-9700 kernel: {1}[Hardware Error]:   section_type: IA32/X64 processor error",
    "2026-11-03T05:47:29+00:00 jiulu-9700 kernel: {1}[Hardware Error]:   Local APIC_ID: 0x82",
    "2026-11-03T05:47:29+00:00 jiulu-9700 kernel: {1}[Hardware Error]:    Error Structure Type: cache error",
    "2026-11-03T05:47:29+00:00 jiulu-9700 kernel: {1}[Hardware Error]:     Processor Context Corrupt: true",
    "2026-11-03T05:47:29+00:00 jiulu-9700 kernel: {1}[Hardware Error]:     Uncorrected: true",
    "2026-11-03T05:47:29+00:00 jiulu-9700 kernel: BERT: Total records found: 1",
    "2026-11-03T05:47:29+00:00 jiulu-9700 kernel: record TIME 1793684832 not really a record line",
    "2026-11-03T05:47:29+00:00 jiulu-9700 kernel: mce: [Hardware Error]: CPU 17: Machine Check: 0 Bank 5: "
    "aea0000000000108",
    "2026-11-03T05:47:29+00:00 jiulu-9700 kernel: mce: [Hardware Error]: TSC 0 ADDR 71bea59a1efa "
    "MISC d0140ff600000000 PPIN 2b61290dcb2002b SYND 4d000000 IPID 500b024029a00",
    "2026-11-03T05:47:29+00:00 jiulu-9700 kernel: pcieport 0000:c4:00.0:    [ 6] BadTLP",
    "2026-11-03T05:57:43+00:00 jiulu-9700 kernel: audit: rate limit exceeded",
]

live, names, replayed, mce, blocks, first, last, mca, sev, apic, synd_ipid, bert, fru = J.parse(BOOT)

ck("the MCA line is found and the bank is read", len(mca) == 1 and mca[0][2] == 5, str(mca)[:160])
cpu, bank, status, f = (mca[0][1], mca[0][2], mca[0][3], mca[0][4]) if mca else (None,) * 4
ck("the cpu number is kept -- it is what turns deaths into one core or ten", cpu == 17, str(cpu))
ck("MCACOD 0x0108 decodes as a level-2 (L2) cache error",
   f["MCACOD"] == 0x0108 and f["class"] == "cache, level 2", f["class"])
ck("UC and PCC are read out of the status word (uncorrectable, context corrupt => fatal)",
   f["UC"] == 1 and f["PCC"] == 1 and f["VAL"] == 1, str(f))
ck("MSCOD is kept separate from MCACOD", f["MSCOD"] == 0x0000, hex(f["MSCOD"]))
f2 = J.mca_fields("aea0000001000108")
ck("the other death's word decodes to the same class with MSCOD 0x0100",
   f2["MCACOD"] == 0x0108 and f2["MSCOD"] == 0x0100 and f2["class"] == "cache, level 2", str(f2))
ck("SYND and IPID are captured (they are what makes the vendor reply)",
   synd_ipid == ("4d000000", "500b024029a00"), str(synd_ipid))
ck("the CPER severity and fru_text are captured (fatal / ProcessorError)",
   sev == ["fatal"] and fru == ["ProcessorError"], f"{sev} {fru}")
ck("BERT record count is captured -- the firmware keeps ONE, so the archive must be ours",
   bert == 1, str(bert))
ck("the APIC id is captured", apic == ["0x82"], str(apic))
ck("a correctable PCIe event in the same boot is counted per port -- and a hex bus id like c4:00.0 counts too",
   names["0000:c4:00.0"]["BadTLP"] == 1, f"{live} {dict(names)}")
ck("first/last kernel timestamps are the boot's own life span",
   first.startswith("2026-11-03T05:47:29") and last.startswith("2026-11-03T05:57:43"), f"{first} {last}")

line = J.mca_line(mca, sev, apic, synd_ipid)
ck("the printed line names the class, the fatality and the codes",
   "MCA bank 5" in line and "cache, level 2" in line and "fatal/uncorrectable" in line
   and "MCACOD 0x0108" in line and "context corrupt" in line, line)

# The pairing question: sysfs loses an offlined cpu's topology, so the fallback chain matters.
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    J.TOPOLOGY = td / "cpu-topology.json"
    ck("an unknown cpu still resolves through the saved conf",
       J.core_of(17) == ("65", "0", "17,113"), str(J.core_of(17)))
    ck("... and so does its SMT sibling -- the two are the same core",
       J.core_of(17)[:2] == J.core_of(113)[:2] and J.core_of(113)[2] == "17,113", str(J.core_of(113)))
    J.TOPOLOGY = td / "nothing.json"
    ck("no sysfs and no snapshot: None, not a crash", J.core_of(9999) == (None, None, None))

conf = Path(ROOT) / "host" / "offline-core.cpus"
ck("host/offline-core.cpus names a pair, not a bare number",
   conf.exists() and len(J.offline_conf()) >= 2, str(J.offline_conf()))
online = {c: (Path(f"/sys/devices/system/cpu/cpu{c}/online").read_text().strip()
              if Path(f"/sys/devices/system/cpu/cpu{c}/online").exists() else "?")
          for c in J.offline_conf()}
if all(v == "1" for v in online.values()):
    sibs = {J.core_of(c)[2] for c in online}
    ck("while the pair is online, sysfs agrees they are one core (the guard in the offline script depends on this)",
       len(sibs) == 1, str(sibs))
else:
    print(f"  skip sibling re-verification: the pair is offline ({online}); sysfs has no topology for them -- "
          f"that is why the conf exists")

ck("a boot with no machine check says so rather than printing a blank",
   J.mca_line([], [], [], None) is None)

print("PASS" if fail == 0 else "FAIL", f"({fail} failing)")
sys.exit(1 if fail else 0)
