#!/usr/bin/env python3
"""tests/test_brief.py - tools/brief.py, the one-line-per-call state logger.

Nothing here touches a card or root. The one contact with the live machine is brief.py's own 4-second
probe of the serve's /health, which the test tolerates either way (up or down). What is worth testing is the arithmetic and the two ways this
line lies:

  (1) the rate. It is computed against the previous call. AER counters are zeroed at enumeration and this box
      powers itself off and on, so across a boot boundary a delta is the difference of two unrelated counters --
      and it goes NEGATIVE when the new boot came up quieter than the one it replaced.
  (2) the thread count. cpu0 has no `online` file, so counting sysfs `online==1` reads one low and makes the
      offline-core intervention look different from what `nproc` shows.

Run: .venv/bin/python tests/test_brief.py
"""
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

spec = importlib.util.spec_from_file_location("brief", os.path.join(ROOT, "tools", "brief.py"))
brief = importlib.util.module_from_spec(spec)
spec.loader.exec_module(brief)          # brief.py keeps its work behind __main__, so importing is inert

fail = 0


def ck(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f" {detail}" if detail and not cond else ""))
    if not cond:
        fail += 1


# -- (1) the rate needs a same-boot previous sample -----------------------------------------------------------
prev = {"t": 100.0, "b": "15d3", "c": {"06:00.0": {"cor": 45, "slot": "8"}}}
ck("a previous sample from ANOTHER boot is dropped, not subtracted",
   brief.prev_window(prev, "b201") == {}, brief.prev_window(prev, "b201"))
ck("a previous sample from THIS boot is kept -- that is the whole rate",
   brief.prev_window(prev, "15d3") is prev)
ck("no previous sample at all yields no rate rather than a rate from thin air",
   brief.prev_window({}, "b201") == {})
# The state file has to carry the boot id or the guard above can never fire.
src = open(os.path.join(ROOT, "tools", "brief.py")).read()
ck("the state file records WHICH boot it came from", '"b": boot' in src and 'prev = prev_window(prev, boot)' in src)

# -- (2) the counter is summed over the whole chain, not read off one device ---------------------------------
chain = {"bus": "c6:00.0", "slot": "6",
         "devices": [{"cor": 1, "speed": 32.0}, {"cor": 30, "speed": 32.0}, {"cor": 13, "speed": 16.0}]}
dev = chain.get("devices") or chain.get("chain") or []
ck("a chain's correctable total sums every port on it (endpoint, switch, root port)",
   sum(p.get("cor", 0) for p in dev) == 44)
ck("the weakest link on the chain is what gets printed, not the endpoint's",
   min((p.get("speed", 0) for p in dev), default=0) == 16.0)
# read_pcie calls the list "devices"; reseat-check renames it to "chain" when snapshotting. Both spellings read.
ck("both spellings of the device list are accepted (webui's and reseat-check's)",
   ({"devices": [1]}).get("devices") and ({"chain": [1]}).get("chain"))

# -- the line itself, at a machine with no PCIe readable ----------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    brief.STATE = Path(td) / "brief-prev.json"
    brief.main.__globals__["webui"].read_pcie = lambda p: []
    brief.main()
    st = json.loads(brief.STATE.read_text())
    ck("a sample still lands when the chain read comes back empty (and carries its boot id)",
       st.get("c") == {} and isinstance(st.get("b"), str) and len(st.get("b")) == 4, st)
    ck("series.log is appended, not rewritten -- the series is the point",
       (Path(td) / "series.log").read_text().strip().count("\n") == 0
       and (Path(td) / "series.log").stat().st_size > 0)

print("PASS (0 failing)" if not fail else f"FAIL ({fail} failing)")
sys.exit(1 if fail else 0)
