#!/usr/bin/env python3
"""tests/test_window_probe.py - the tiny-kernel latency probe must split submit from wait, stay cheap, and never break a launch, stay cheap, and never break a launch.

Run: LD_LIBRARY_PATH=$HOME/opt/openmpi/usr/lib/x86_64-linux-gnu ~/venv-r9700/bin/python3 tests/test_window_probe.py
"""
import io
import logging
import os
import sys
import time

os.environ["R9K_LOAD_PROF"] = "1"
os.environ["R9K_PWAL_PROBE"] = "1"
os.environ["R9K_PWAL_PROBE_MS"] = "1"          # sample hard; production default is 200 ms
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from r9700_vllm.utils import prof  # noqa: E402

fail = 0


def ck(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name} {detail}")
    if not cond:
        fail += 1


buf = io.StringIO()
h = logging.StreamHandler(buf)
h.setLevel(logging.INFO)
prof.logger.addHandler(h)
# outside the server, vllm's logger defaults above INFO and would silently drop every line this test checks for
prof.logger.setLevel(logging.INFO)
logging.getLogger().setLevel(logging.INFO)


def walk(model, per_visit=0.001):
    """The stock post-load pass shape: one named_modules loop with work between the yields."""
    n = 0
    for _name, _m in model.named_modules():
        n += 1
        time.sleep(per_visit)
    return n


model = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(200)])
assert torch.cuda.is_available(), "the probe measures the device path; it needs a GPU"

# ---- armed by default, samples on both clocks, cheap ----------------------------------------------------------------
prof.start_pwal()
visits = walk(model)
prof.stop_pwal()
st = prof._PWPR
ck("the probe sampled", st["n"] >= 5, f"({st['n']} samples over {visits} visits)")
ck("both halves recorded", len(st["sub"]) == len(st["tot"]) and len(st["sub"]) + len(st["sync_only"]) == st["n"],
   f"({len(st['sub'])} submit+sync, {len(st['sync_only'])} floor)")
ck("submit+sync latencies are positive", all(x > 0 for x in st["tot"]),
   f"(p50 {prof._pct(st['tot'], 0.5):.3f} p95 {prof._pct(st['tot'], 0.95):.3f} max {max(st['tot']):.2f} ms)")
ck("the submission half is finite",
   all(0.0 <= x < 1e9 for x in st["sub"]),
   f"(p50 {prof._pct(st['sub'], 0.5):.3f} p95 {prof._pct(st['sub'], 0.95):.3f} ms)")
ck("floor samples (sync with nothing submitted) are recorded",
   len(st["sync_only"]) >= 1 and all(x >= 0 for x in st["sync_only"]),
   f"(p50 {prof._pct(st['sync_only'], 0.5):.3f} ms)")
ck("the host loop is timed once per sample", len(st["host"]) == st["n"] and all(1.0 < x < 1e6 for x in st["host"]),
   f"({len(st['host'])} samples, p50 {prof._pct(st['host'], 0.5):.1f} ns/iter)")
ck("probe cost stays small", st["cost"] / max(1, st["n"]) < 0.010,
   f"({1000 * st['cost'] / max(1, st['n']):.2f} ms per sample)")
log = buf.getvalue()
ck("the report names both halves", "probe:" in log and "submit ms" in log and "submit+sync ms" in log)
ck("the report names the host loop", "host loop ns/iter" in log,
   f"({[l.split('probe: ')[1] for l in log.splitlines() if 'host loop' in l][:1]})")

# ---- interval spacing is respected --------------------------------------------------------------------------------
fast = st["n"]
os.environ["R9K_PWAL_PROBE_MS"] = "100000"     # effectively never sample again
prof.start_pwal()
walk(model)
prof.stop_pwal()
ck("a huge interval samples at most once (the first visit always probes)", prof._PWPR["n"] <= 1,
   f"({prof._PWPR['n']} samples)")
ck("spacing worked earlier", 5 <= fast <= visits, f"({fast} samples for {visits} visits)")

# ---- budget stops it, honestly ------------------------------------------------------------------------------------
os.environ["R9K_PWAL_PROBE_MS"] = "1"          # back to sampling every visit for the budget check
prof.start_pwal()
prof._PWPR["budget"] = 4
walk(model, per_visit=0.002)
prof.stop_pwal()
ck("budget bounds the sample count", prof._PWPR["n"] <= 5, f"({prof._PWPR['n']} samples)")
ck("budget stop is reported", "budget" in (prof._PWPR["why"] or ""), f"({prof._PWPR['why']})")

# ---- off switch ---------------------------------------------------------------------------------------------------
os.environ["R9K_PWAL_PROBE"] = "0"
prof.start_pwal()
walk(model)
prof.stop_pwal()
ck("R9K_PWAL_PROBE=0 arms nothing", prof._PWPR["n"] == 0 and "disabled" in prof._PWPR["why"],
   f"({prof._PWPR['why']})")
ck("off clears every sample list, host included", not any(prof._PWPR[k] for k in ("sub", "tot", "sync_only", "host")),
   f"({ {k: len(prof._PWPR[k]) for k in ('sub', 'tot', 'sync_only', 'host')} })")
os.environ["R9K_PWAL_PROBE"] = "1"

# ---- a failing probe cannot break the pass --------------------------------------------------------------------------
real = torch.cuda.current_device
torch.cuda.current_device = lambda: (_ for _ in ()).throw(RuntimeError("device exploded"))
try:
    prof.start_pwal()
    n = walk(model)
    prof.stop_pwal()
    ck("the pass survives a broken probe", n == len(list(model.named_modules())), f"({n} visits)")
    ck("the failure is reported", "probe failed" in (prof._PWPR["why"] or ""), f"({prof._PWPR['why']})")
except Exception as e:  # noqa: BLE001
    ck("the pass survives a broken probe", False, repr(e))
finally:
    torch.cuda.current_device = real

# ---- percentile helper --------------------------------------------------------------------------------------------
ck("_pct on empty is 0", prof._pct([], 0.95) == 0.0)
ck("_pct picks ordered elements", prof._pct([3.0, 1.0, 2.0], 0.5) == 2.0 and prof._pct([3.0, 1.0, 2.0], 0.99) == 3.0)

# ---- the window wrapper is restored even after all that -----------------------------------------------------------
ck("named_modules restored", torch.nn.Module.named_modules is not None
   and "timed" not in repr(getattr(torch.nn.Module, "named_modules")))

print("FAIL" if fail else "PASS", f"({fail} failing)")
sys.exit(1 if fail else 0)
