#!/usr/bin/env python3
"""What does the draft pass's bf16->MXFP4 requant actually cost per element?

The load profile charges 4.4-44 s (instance-dependent) to `quantize_mxfp4_search` over the MTP experts. This runs the
same function on the same row width (2560) at a few sizes and reports GiB-of-bf16 per second, so the load-time number
can be compared against what the kernel alone can do. Skips any chunk that does not fit in the free VRAM, which
matters when it is run next to a live server.

    LD_LIBRARY_PATH=$HOME/opt/openmpi/usr/lib/x86_64-linux-gnu ~/venv-r9700/bin/python3 bench/mtpquant_probe.py
"""
import sys
import time

import torch

sys.path.insert(0, ".")
from r9700_vllm.quant.nvfp4 import quantize_mxfp4_search  # noqa: E402

K = 2560
free, total = torch.cuda.mem_get_info()
print(f"device 0: {total / 2**30:.1f} GiB total, {free / 2**30:.1f} GiB free")
for rows in (1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18):
    need = rows * K * 4 * 4                       # fp32 input + a few fp32 intermediates
    if need > 0.8 * free:
        print(f"{rows:>7} rows: skipped, needs ~{need / 2**30:.1f} GiB")
        continue
    w = torch.randn(rows, K, device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    quantize_mxfp4_search(w)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    gib_bf16 = rows * K * 2 / 2**30               # what the checkpoint holds for these rows
    print(f"{rows:>7} rows: {dt * 1000:8.1f} ms   {gib_bf16 / dt:7.2f} GiB-of-bf16/s")
    del w
    torch.cuda.empty_cache()
