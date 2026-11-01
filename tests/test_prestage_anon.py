#!/usr/bin/env python3
"""tests/test_prestage_anon.py - the anon prestage/pin helper must be honest, cheap and unable to break a launch.

Run: ~/venv-r9700/bin/python3 tests/test_prestage_anon.py
"""
import logging
import os
import resource
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(message)s")

from r9700_vllm.utils import pin  # noqa: E402

fail = 0


def ck(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name} {detail}")
    if not cond:
        fail += 1


def _locked_kib():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmLck:"):
                return int(line.split()[1])
    return -1


def _maps_line(start_hex, end_hex):
    s, e = int(start_hex, 16), int(end_hex, 16)
    with open("/proc/self/maps") as f:
        for line in f:
            m = pin._MAPS.match(line.rstrip("\n"))
            if not m:
                continue
            if int(m.group(1), 16) == s and int(m.group(2), 16) == e:
                return m
    return None


# ---- range selection ----------------------------------------------------------------------------------------------
# A bare interpreter has nothing but 4 KiB arenas (largest anon mapping measured: 2 MiB), so the test allocates the
# big anonymous region it wants to exercise -- which is also what a worker's allocator segments look like.
_buf = bytearray(256 << 20)
for _o in range(0, len(_buf), 4096):
    _buf[_o] = 1
rs = pin._anon_private_ranges()
ck("finds private anon regions >= 8 MiB", len(rs) >= 1, f"({len(rs)} found)")
ck("sees the 256 MiB region the test allocated", any(e - s >= 200 << 20 for s, e in rs),
   f"(biggest {max((e - s for s, e in rs), default=0) >> 20} MiB)")
ck("sorted biggest first", all(rs[i][1] - rs[i][0] >= rs[i + 1][1] - rs[i + 1][0] for i in range(len(rs) - 1)))
bad = [r for r in rs if (lambda m: m is None or m.group(4) != "0" or m.group(5).strip())(_maps_line(format(r[0], "x"), format(r[1], "x")))]
ck("no file-backed or named mapping slips through", not bad, f"({len(bad)} bad)")
ck("all sizes >= _MIN_REGION", all(e - s >= pin._MIN_REGION for s, e in rs))

# ---- off by default -----------------------------------------------------------------------------------------------
os.environ.pop("R9K_PWAL_PRESTAGE", None)
ck("off when R9K_PWAL_PRESTAGE unset", pin.prestage_anon() == {"mode": "off"})
os.environ["R9K_PWAL_PRESTAGE"] = "bogus"
ck("unknown mode is treated as off", pin.prestage_anon() == {"mode": "off"})

# ---- populate: madvise, so nothing gets locked --------------------------------------------------------------------
os.environ["R9K_PWAL_PRESTAGE"] = "populate"
os.environ["R9K_PWAL_GIB"] = "0.5"
lck0 = _locked_kib()
st = pin.prestage_anon("test populate")
ck("populate touched something", st["bytes"] > 0 and st["ranges"] > 0, f"({st['bytes'] / 2**20:.0f} MiB)")
ck("populate covers the big region first", st["bytes"] >= 200 << 20, f"({st['bytes'] / 2**20:.0f} MiB)")
ck("populate reports a time", st["seconds"] >= 0.0, f"({st['seconds'] * 1000:.1f} ms)")
ck("populate does not lock anything", _locked_kib() == lck0, f"({lck0} -> {_locked_kib()} KiB)")
ck("populate budget respected", st["bytes"] <= int(0.5 * 2**30), f"({st['bytes']} bytes)")
ck("populate honest about errors", st["first_error"] is None or st["errors"] > 0)

# ---- lock: mlock, so VmLck rises (or the kernel says no, honestly) -------------------------------------------------
os.environ["R9K_PWAL_PRESTAGE"] = "lock"
os.environ["R9K_PWAL_GIB"] = "0.125"  # 32 MiB: small enough for any RLIMIT_MEMLOCK
lck1 = _locked_kib()
st = pin.prestage_anon("test lock")
lck2 = _locked_kib()
soft = resource.getrlimit(resource.RLIMIT_MEMLOCK)[0]
allowed = soft == resource.RLIM_INFINITY or soft >= 32 << 10
ck("lock either worked or reported why", (lck2 >= lck1 + (16 << 10)) or st["errors"] > 0,
   f"(VmLck {lck1} -> {lck2} KiB, errors {st['errors']} {st['first_error']}, soft {soft})")
if allowed and st["errors"] == 0:
    ck("lock raised VmLck by roughly the requested amount", lck2 >= lck1 + (16 << 10), f"({lck1} -> {lck2} KiB)")

# ---- never raises, whatever the environment says ------------------------------------------------------------------
os.environ["R9K_PWAL_GIB"] = "not-a-number"
try:
    st = pin.prestage_anon("test junk budget")
    ck("junk budget falls back to a default instead of raising", st["bytes"] > 0)
except Exception as e:  # noqa: BLE001
    ck("junk budget falls back to a default instead of raising", False, repr(e))

print("FAIL" if fail else "PASS", f"({fail} failing)")
sys.exit(1 if fail else 0)
