"""Correctness of the batched MoE H2D copies (r9700_vllm/moe/h2dbatch.py). Run with one GPU visible:

    R9K_MOE_BATCH_H2D=1 ~/venv-r9700/bin/python3 tests/test_moe_h2dbatch.py

The optimization replaces ~10^5 per-expert pageable copies with a handful of large ones, so it must be byte-exact
against the plain path, including the cases that break a run (non-contiguous source, a foreign copy into the same
storage, a run larger than the staging buffer).
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("R9K_MOE_BATCH_H2D", "1")

from r9700_vllm.moe import h2dbatch  # noqa: E402

ok = True


def check(what, cond, detail=""):
    global ok
    print(f"  {'ok  ' if cond else 'FAIL'} {what}{'' if cond else '  ' + detail}")
    ok = ok and cond


E, N, K = 512, 4, 1360                      # the w13 shape of this checkpoint's fused experts (106 KiB a slice)
torch.manual_seed(7)
src = torch.randint(0, 256, (E, N, K), dtype=torch.uint8)

if not h2dbatch.install():
    print("FAIL: h2dbatch did not install")
    sys.exit(1)

# 1. the stock pattern: one copy_ per expert, in order
plain = torch.zeros(E, N, K, dtype=torch.uint8, device="cuda")
t0 = time.perf_counter()
for e in range(E):
    plain[e].copy_(src[e])
t_plain = time.perf_counter() - t0

batched = torch.zeros(E, N, K, dtype=torch.uint8, device="cuda")
t0 = time.perf_counter()
for e in range(E):
    batched[e].copy_(src[e])
h2dbatch.flush_all("test: one fused tensor")
t_batch = time.perf_counter() - t0

check("per-expert pattern is byte-exact", bool((batched == plain).all()))
check("batching is faster here", t_batch < t_plain, f"plain {t_plain * 1000:.1f} ms vs batched {t_batch * 1000:.1f} ms")

# 2. run broken by a non-contiguous source (the `srcstrided` class): must pass through and still be exact
half = torch.randint(0, 256, (E, N, 2 * K), dtype=torch.uint8)
strided = half[:, :, :K]
plain2 = torch.zeros(E, N, K, dtype=torch.uint8, device="cuda")
for e in range(E):
    plain2[e].copy_(strided[e])
batched2 = torch.zeros(E, N, K, dtype=torch.uint8, device="cuda")
for e in range(E):
    batched2[e].copy_(strided[e])     # not eligible (strided) -> plain copy_, but must flush any pending run
h2dbatch.flush_all("test: strided source")
check("strided-source pattern is byte-exact", bool((batched2 == plain2).all()))

# 3. interleaving into one storage (a gate/up pair written alternately) and a foreign copy mid-run
param = torch.zeros(E, N, K, dtype=torch.uint8, device="cuda")
other = torch.randint(0, 256, (E, N, K), dtype=torch.uint8)
for e in range(E):
    param[e].copy_(src[e])
    if e % 7 == 0:
        param[e].copy_(other[e])      # different source, same storage: eligible, so it is staged too
h2dbatch.flush_all("test: interleaved writes")
ref = src.clone()
for e in range(0, E, 7):
    ref[e] = other[e]
check("interleaved writes are byte-exact", bool((param == ref.to(param.device)).all()))

# 4. nothing may be left staged after flush_all
check("no staged bytes pending", h2dbatch._RUN is None)
h2dbatch.flush_all("test: second flush is a no-op")

torch.cuda.synchronize()
print(f"  {E} slices: plain {t_plain * 1000:.1f} ms, batched {t_batch * 1000:.1f} ms")
t = h2dbatch._TOT
print(f"  counters: {t['calls']} staged, {t['batches']} copies, {t['bytes'] / 2**20:.0f} MiB, "
      f"{t['stage_s']:.3f} s staging, {t['copy_s']:.3f} s device")
print("PASS" if ok else "FAILURES")
sys.exit(0 if ok else 1)
