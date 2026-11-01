"""prof.slowcall: a load-path call that takes longer than the threshold names itself, with its CPU split.

Run: ~/venv-r9700/bin/python3 tests/test_slowcall.py
Why: per-module load attribution stops at a layer, and the tail we are chasing is one call inside one layer
(tail48: `layers.0.mlp.shared_expert.gate_up_proj`, 43.57 s wall against 87.1 s of user CPU). The only way to read
that number is to know which call it was, on which device, burning how much CPU. So the decorator has to (a) fire
only past its threshold, (b) carry the CPU/fault split that distinguishes "executed" from "waited", and (c) cost
nothing when profiling is off."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from r9700_vllm.utils import prof  # noqa: E402

prof.ON = True
lines = []
_real_info = prof.logger.info


def _cap(fmt, *a):
    lines.append((fmt % a) if a else fmt)
    return _real_info(fmt, *a)


prof.logger.info = _cap

ok = True


def check(name, cond, detail=""):
    global ok
    print(f"  {name:<58} {'ok' if cond else 'FAIL'} {detail}")
    ok = ok and bool(cond)


@prof.slowcall(thresh=0.05)
def slow_burn(w):
    time.sleep(0.08)                      # wall over the threshold, deterministically, on any machine
    return w * 1.5                        # ... plus a little real CPU for the split to be non-zero


@prof.slowcall(thresh=0.05)
def fast(w):
    return w.clone()


w = torch.randn(64, 128)
lines.clear()
slow_burn(w)
check("a slow call logs one line", len(lines) == 1, str(lines)[:120])
line = lines[0] if lines else ""
check("it names the function", "slow_burn" in line)
check("it gives wall and the user/sys CPU split", " cpu " in line and " u/s" in line, line[:100])
check("it reports page faults", "minor" in line and "major" in line)
check("it describes the tensor argument (device/dtype/shape)", "cpu float32 [64, 128]" in line, line[-80:])
check("it says the layout only when it is odd", "NONCONTIG" not in line)

lines.clear()
fast(w)
check("a fast call logs nothing", not lines, str(lines)[:120])

lines.clear()
slow_burn(w.t())                                  # non-contiguous view
check("a transposed tensor is flagged NONCONTIG", "NONCONTIG" in (lines[0] if lines else ""), str(lines)[:120])


@prof.slowcall(thresh=0.0)
def always():
    return 1


check("thresh=0 fires on any call", always() == 1 and len(lines) >= 1)

prof.ON = False


@prof.slowcall
def untouched(w):
    return w


prof.ON = True


@prof.slowcall
def also_wrapped(w):
    return w


check("with profiling off the function is not wrapped", not hasattr(untouched, "__wrapped__")
      and untouched(3) == 3)
check("with profiling on it is wrapped (and still returns its value)", hasattr(also_wrapped, "__wrapped__")
      and also_wrapped(3) == 3)

# ---- prof.splitcall: one call measured as issuing + waiting, and R9K_SPLIT=0 removing the sync entirely --------
if torch.cuda.is_available():
    os.environ["R9K_SPLIT_MS"] = "1"

    @prof.splitcall("matmul")
    def gpu_work(n):
        a = torch.randn(n, n, device="cuda")
        return a @ a

    gpu_work(1024)
    spl = [l for l in lines if "split matmul" in l]
    check("the split names issuing and waiting separately", bool(spl) and "issuing + " in spl[-1],
          (spl[-1:] or [""])[0][-95:])
    lines.clear()
    gpu_work(2048)
    spl2 = [l for l in lines if "split matmul" in l]
    _nums = [float(x) for x in __import__("re").findall(r"([-+]?[0-9]*\.?[0-9]+) s (?:issuing|waiting)", spl2[-1])] if spl2 else []
    check("a bigger call reports, and neither half is negative", bool(spl2) and len(_nums) == 2
          and all(x >= 0 for x in _nums), (spl2[-1:] or [""])[0][-95:])
    lines.clear()
    os.environ["R9K_SPLIT"] = "0"
    gpu_work(1024)
    check("R9K_SPLIT=0 adds no sync and no line", not [l for l in lines if "split matmul" in l], str(lines)[:120])
    os.environ["R9K_SPLIT"] = "1"
    os.environ["R9K_SPLIT_MS"] = "50"
    prof.split(0.0002, 0.0002, "quick")
    check("below R9K_SPLIT_MS the split is quiet", not [l for l in lines if "split quick" in l], str(lines)[:120])
    prof.slow_note(0.2, "fold guard: %.2f s issuing + %.2f s waiting", 0.01, 0.2)
    check("the fold note now fires at 0.2 s (it used to need 1 s)", "fold guard" in "".join(lines), str(lines)[:120])
else:
    check("splitcall needs a device to be meaningful", True, "(skipped: no cuda)")

print(f"slowcall tests: {'PASS' if ok else 'FAIL'}")
raise SystemExit(0 if ok else 1)
