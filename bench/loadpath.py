#!/usr/bin/env python3
"""Where loading expert weights goes: one real fused MXFP4 expert tensor, four ways.

The server spends 120.8 s a rank moving 59.8 GiB of expert tensors (R9K_LOAD_PROF=1, see
notes/native-4xr9700.md) -- about 0.5 GB/s. In this checkpoint the experts of a layer are fused into one tensor
per projection (`layers.N.mlp.experts.gate_up_proj_packed`, 512 experts x 1280 x 2560 u8-packed = 839 MB), and a
TP4 shard is a *column* slice of every expert, so the obvious read is strided. Measure the alternatives on the
checkpoint's own bytes:

  A  mmap fault-in            safetensors -> torch tensor (CPU pages; the cost vLLM's iterator hides)
  B  bulk H2D                the whole tensor in one copy
  C  per-expert strided host slice + H2D per expert   (what a shard-first loader ends up doing)
  D  bulk H2D, then the same slice on device          (host-first, shard later)

Run it with the server down (it wants ~1.1 GiB of VRAM) and 4 GPUs idle. Usage: loadpath.py [TENSOR] [REPS]
"""
import sys, time
import torch
from safetensors import safe_open

CKPT = "/home/jiulu/models/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ"
NAME = sys.argv[1] if len(sys.argv) > 1 else "model.language_model.layers.0.mlp.experts.gate_up_proj_packed"
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
IDX = {"model.language_model.layers.0.mlp.experts.gate_up_proj_packed": "model-00009.safetensors"}.get(NAME)
if IDX is None:
    import json
    IDX = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"][NAME]

torch.cuda.init()
with safe_open(f"{CKPT}/{IDX}", framework="pt") as sf:
    t0 = time.perf_counter()
    w = sf.get_tensor(NAME)
    a = time.perf_counter() - t0
E, N, K = w.shape
gib = w.numel() * w.element_size() / 2**30
print(f"{NAME}: dtype {w.dtype} shape {tuple(w.shape)} = {gib:.2f} GiB")

# C: per-expert strided host slice -> H2D per expert. TP4 splits the intermediate dim: 4 columns per expert here.
tp = 4
n_local = N // tp
dev_full = torch.empty_like(w, device="cuda")
dev_shard = torch.empty((E, n_local, K), dtype=w.dtype, device="cuda")


def reps(fn):
    ts = []
    for _ in range(REPS):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return min(ts)


print(f"A  mmap fault-in           {a:6.2f} s   {gib/a:6.2f} GiB/s   (first touch, cold)")
b = reps(lambda: dev_full.copy_(w))
print(f"B  bulk H2D whole tensor   {b:6.2f} s   {gib/b:6.2f} GiB/s")


def per_expert_strided():
    for e in range(E):                      # host slice per expert, then a small H2D per expert
        dev_shard[e].copy_(w[e, :n_local, :])


c = reps(per_expert_strided)
print(f"C  per-expert strided     {c:6.2f} s   {E * n_local * K * w.element_size() / 2**30 / c:6.2f} GiB/s of "
      f"useful bytes ({c / b:.1f}x slower than bulk)")


def device_slice():
    dev_full.copy_(w)                       # one contiguous copy...
    dev_shard.copy_(dev_full[:, :n_local, :])   # ...then the strided slice where it is cheap


d = reps(device_slice)
print(f"D  bulk H2D + slice on GPU {d:6.2f} s   ({d / b:.1f}x bulk)")
