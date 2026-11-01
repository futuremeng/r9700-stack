"""What does a CPU->GPU copy actually cost on this box, pageable vs pinned?

The load profile says 94% of the target pass sits in `expert_data.copy_(loaded_weight)` inside
vllm/model_executor/layers/fused_moe/routed_experts.py, and the copy probe says the time is in a handful of 0.2 MiB
copies that block 5-50 s (inst12/13), not in the 148k ordinary ones (0.18 ms each). That is the signature of the
pageable host-to-device path: every call bounces through a driver staging buffer, four ranks do it at once, and
the queue shows up in the caller as a stall. This script measures the two paths at the sizes the loader actually
uses, so the fix (pinned staging + fewer, larger copies) can be priced before writing it.

    usage: ~/venv-r9700/bin/python3 bench/h2d_probe.py [device]        # run 4 in parallel for the 4-rank case
"""
import os
import sys
import time

import torch

DEV = f"cuda:{sys.argv[1] if len(sys.argv) > 1 else os.environ.get('DEV', '0')}"
SIZES = [(204800, "0.2 MiB (one expert slice)"), (1 << 20, "1 MiB"), (16 << 20, "16 MiB"), (64 << 20, "64 MiB")]


def bench(nbytes: int, pinned: bool, reps: int = 12) -> float:
    src = torch.empty(nbytes, dtype=torch.uint8, device="cpu", pin_memory=pinned)
    if not pinned:
        src.fill_(1)                              # fault the pages in, like a touched mmap page would be
    dst = torch.empty(nbytes, dtype=torch.uint8, device=DEV)
    for _ in range(2):                            # warm
        dst.copy_(src)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        dst.copy_(src)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps
    del src, dst
    return dt


def main() -> None:
    print(f"{os.getpid()} on {DEV}: {torch.cuda.get_device_name(DEV)}")
    for nbytes, label in SIZES:
        page = bench(nbytes, False)
        pin = bench(nbytes, True)
        print(f"  {label:26s} pageable {page * 1e3:8.3f} ms ({nbytes / page / 2**30:6.2f} GiB/s)   "
              f"pinned {pin * 1e3:8.3f} ms ({nbytes / pin / 2**30:6.2f} GiB/s)   speedup {page / pin:5.1f}x")


main()
