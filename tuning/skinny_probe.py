#!/usr/bin/env python3
"""Is upstream vLLM #62799 (Triton skinny GEMM for 6-128 rows on RDNA4) worth anything to this stack?

The PR's claim, measured by its author on one R9700 (gfx1201): BF16/FP16 dense GEMMs with 6-128 rows fall through
`wvSplitK` (1-5 rows, ~600 GB/s) to hipBLASLt, which runs those weight-bound shapes at 420-470 GB/s, and a Triton
kernel for the band buys +28-29% output throughput with DSpark at conc 1-16.

This stack does not run those shapes on hipBLASLt by default: the launcher serves with R9K_HC_FP8_DECODE=1 /
R9K_HC_FP8_DECODE_MAXM=255 (the hyper-connection down 336x10240 and up 10240x320 GEMMs run per-token fp8 on our
WMMA kernels in decode) and R9K_ROUTER takes the 512x2560 router gate up to max_m (default 32). So the question is
narrow: for the (M x 2560) x [512, 2560]^T router shape, 51 calls a step, what does each path cost at the M a
serving launch actually puts on the GPU (M = tokens a step, so ~3 at conc 1, ~2x streams at conc 16/32 with MTP=2),
and what would a 500-600 GB/s streaming kernel save?

Graph-timed like the kernel gates (the same M, N, K as the served op; the served op is the *transposed* weight, as
in ReplicatedLinear: y = x @ w.T). usage: skinny_probe.py [N K ...] [--ms 1,4,8,...] [--reps 50]

Read the answer as a bound, not a promise: saved-per-step = calls_per_step x (t_now - t_best), and the step it has
to fit inside is the one bench/itl.py measures on the live launch.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from r9700_vllm.router import router_gemm  # noqa: E402


def time_graph(fn, reps: int) -> float:
    """Median of `reps` replays of a captured graph, in us. Warm-up runs outside the capture."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); g.replay(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3)
    return statistics.median(ts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shapes", nargs="*", default=["512 2560", "336 10240", "10240 320"],
                    help="N K pairs (weight is [N, K], the op computes x @ w.T)")
    ap.add_argument("--ms", default="1,2,4,8,16,24,32,48,64,96,128")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--calls", type=int, default=51, help="calls per decode step, for the bound")
    ap.add_argument("--step-ms", type=float, default=13.0, help="step time to divide the saving by")
    a = ap.parse_args()

    ms = [int(m) for m in a.ms.split(",")]
    print(f"device {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    from r9700_vllm import router as RT
    print(f"r9k_router: available={RT.available()} kernel max_m={RT.lib().r9k_router_max_m() if RT.available() else '-'} "
          f"op default MAX_M={RT.MAX_M} split={RT.SPLIT}")
    print(f"bound uses --calls {a.calls}/step against a {a.step_ms} ms step\n")
    for spec in a.shapes:
        n, k = (int(v) for v in spec.split())
        w = (torch.randn(n, k, generator=torch.Generator().manual_seed(0)) * 0.02).to(torch.bfloat16).cuda()
        wb = w.element_size() * w.numel() / 1e6
        print(f"### x[M,{k}] x w[{n},{k}]^T   weight {wb:.2f} MB")
        head = f"{'M':>4} {'torch.matmul us':>16} {'GB/s':>7} {'mm fp32-out us':>17} {'GB/s':>7} {'r9k router us':>16} {'GB/s':>7}"
        print(head)
        for m in ms:
            x = torch.randn(m, k, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16).cuda()
            rows = []
            for name, fn in (("matmul", lambda: torch.matmul(x, w.t())),
                             ("mm-fp32", lambda: torch.mm(x, w.t(), out_dtype=torch.float32)),
                             ("r9k-router", lambda: router_gemm(x, w, out_bf16=True, max_m=max(m, 1)))):
                try:
                    us = time_graph(fn, a.reps)
                except Exception as ex:
                    print(f"{m:>4} {name}: {type(ex).__name__}: {str(ex)[:60]}")
                    us = float("nan")
                rows.append((us, wb * 1e6 / (us * 1e-6) / 1e9 if us == us else float("nan")))
            cells = "".join(f"{u:>16.1f} {g:>7.0f}" for u, g in rows)
            best = min(u for u, _ in rows if u == u)
            hip = rows[0][0]
            saved = (hip - best) * a.calls / 1e3
            print(f"{m:>4}{cells}   saved/step {(hip - best) * a.calls:>9.0f} us = {100 * saved / a.step_ms:>5.2f}% of the step")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
