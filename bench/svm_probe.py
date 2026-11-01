"""Is the load's multi-second copy_ stall caused by the big hipHostRegister'd PLE table?

The load evidence (inst11-14, notes/native-4xr9700.md): ~15 copies per rank take 94-104 s of the ~136 s of copy
wall, the slowest is 52 s for a **0.2 MiB** copy, all four ranks are blocked in the same window, the GPUs read
gpu_busy_percent=3% during it, cached/swap/MemAvailable are flat (so it is not reclaim), and the same 0.2 MiB copy
measures 0.049 ms in isolation (bench/h2d_probe.py). What is different in the real load is that each rank has
9.7 GiB of host memory registered with hipHostRegister (the int6 PLE table, utils/hostmem.py) and the driver does
SVM bookkeeping for it -- the same svm_range_validate_and_map path that later deadlocked the box
(notes/load-oom-svm-wedge.md).

This script measures that in isolation: optionally register N GiB of host memory, then time a run of small
pageable H2D copies and report the latency distribution. Run it once alone and once with four copies at a time to
see whether the pinned region is what turns a 0.05 ms copy into a multi-second one.

    usage: PIN_GIB=9.7 COPIES=2000 DEV=0 ~/venv-r9700/bin/python3 bench/svm_probe.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from r9700_vllm.utils import hostmem                                    # noqa: E402

DEV = f"cuda:{os.environ.get('DEV', '0')}"
PIN_GIB = float(os.environ.get("PIN_GIB", "0"))
COPIES = int(os.environ.get("COPIES", "2000"))
CPY_BYTES = int(os.environ.get("CPY_BYTES", "204800"))                 # one expert slice, as _load_w13 copies


def latencies(src, dst, n: int) -> list[float]:
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        dst.copy_(src)
        out.append(time.perf_counter() - t0)
    return out


def pct(v, p):
    return sorted(v)[min(len(v) - 1, int(len(v) * p))]


def main() -> None:
    torch.cuda.init()
    hold = None
    if PIN_GIB > 0:
        t0 = time.perf_counter()
        hold = hostmem.pinned_empty((int(PIN_GIB * 2**30),), torch.uint8)   # exact-size hipHostRegister
        torch.cuda.synchronize()
        print(f"  pid={os.getpid()} dev={DEV} registered {PIN_GIB} GiB in {time.perf_counter() - t0:6.2f} s "
              f"(pinned={hold.is_pinned()})")
    src = torch.empty(CPY_BYTES, dtype=torch.uint8, device="cpu")
    src.fill_(1)
    dst = torch.empty(CPY_BYTES, dtype=torch.uint8, device=DEV)
    latencies(src, dst, 20)
    v = latencies(src, dst, COPIES)
    print(f"  pid={os.getpid()} {COPIES} x {CPY_BYTES / 2**20:.1f} MiB pageable copies: "
          f"p50 {pct(v, 0.5) * 1e3:7.3f} ms  p90 {pct(v, 0.9) * 1e3:7.3f} ms  p99 {pct(v, 0.99) * 1e3:8.3f} ms  "
          f"max {max(v) * 1e3:9.3f} ms  total {sum(v):7.2f} s")


main()
