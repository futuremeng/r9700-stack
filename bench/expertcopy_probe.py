#!/usr/bin/env python3
"""Is the experts weight-load cost per-call overhead or bandwidth?

The target pass issues 147k pageable H2D copies of ~106 KiB and ~51 KiB at 0.33-0.52 GB/s (bench/prof copy classes).
This reproduces the shape -- one fused CUDA destination [E, N, K], E=512 experts written one at a time -- and then the
same bytes as one big copy, plus a pinned-staging variant. Run it with the GPUs free (after bench/gate.sh):

    LD_LIBRARY_PATH=$HOME/opt/openmpi/usr/lib/x86_64-linux-gnu ~/venv-r9700/bin/python3 bench/expertcopy_probe.py
"""
import time

import torch

E, N, K = 512, 4, 1360            # 512 experts x 4 rows x 1360 bytes = 106 KiB a slice, 54 MiB a layer
dev = torch.device("cuda")


def timeit(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def gib(x):
    return x.numel() * x.element_size() / 2**30


for label, k in (("w13-shaped (106 KiB/expert)", K), ("w2-shaped (51 KiB/expert)", K // 2)):
    nbytes = N * k
    src = torch.randint(0, 255, (E, N, k), dtype=torch.uint8)          # pageable host, contiguous
    dst = torch.empty(E, N, k, dtype=torch.uint8, device=dev)          # the fused parameter
    total = gib(src)

    dt = timeit(lambda: [dst[e].copy_(src[e]) for e in range(E)])
    print(f"{label:<30} per-expert  : {dt * 1000:8.1f} ms  {total / dt:6.2f} GiB/s  ({E} copies)")
    dt = timeit(lambda: dst.view(-1).copy_(src.view(-1)))
    print(f"{label:<30} one copy    : {dt * 1000:8.1f} ms  {total / dt:6.2f} GiB/s")
    dst2 = torch.empty(E, N, k, dtype=torch.uint8, device=dev)
    dt = timeit(lambda: dst2.copy_(src.to(dev, non_blocking=False)))
    print(f"{label:<30} tensor.to   : {dt * 1000:8.1f} ms  {total / dt:6.2f} GiB/s")
    del dst, dst2, src

# the strided-source class: 49,248 copies of 51 KiB whose source is a strided view of a fused checkpoint tensor
E2, N2, K2 = 512, 4, 680
full = torch.randint(0, 255, (E2, N2, 2 * K2), dtype=torch.uint8)     # [E, N, 2K]: take the left half of each row
strided = full[:, :, :K2]                                            # non-contiguous source, same bytes as dst
dst = torch.empty(E2, N2, K2, dtype=torch.uint8, device=dev)
total = gib(strided)
dt = timeit(lambda: [dst[e].copy_(strided[e]) for e in range(E2)])
print(f"{'strided source (51 KiB)':<30} per-expert  : {dt * 1000:8.1f} ms  {total / dt:6.2f} GiB/s")
dt = timeit(lambda: dst.copy_(strided.contiguous()))                 # materialise once, then one copy
print(f"{'strided source (51 KiB)':<30} contig+one  : {dt * 1000:8.1f} ms  {total / dt:6.2f} GiB/s")
