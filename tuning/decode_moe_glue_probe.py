"""What the MoE layer costs *around* its two GEMMs, at this checkpoint's shapes, graph-timed.

The point is the ceiling of a "glue" project: at top-22 of 512 experts, everything that is not the two grouped GEMMs
(router GEMM, grouped_topk, the align/sort tables, the two per-row fp8 quants, the activation quant, moe_sum, the two
bf16 latent projections, the layer norm) is a candidate for fusion or deletion. Compare the total against the two
GEMMs, which `tuning/decode_moe_sweep.py` measures at 412 us a layer for 1760 rows (16 streams x 5 tokens x top-22).

Shapes are this model's, not Flash-Next's: hidden 4096, moe_latent_size 1024 (so the experts work in a 1024-wide
latent space -- `fc1_latent_proj [1024,4096]` and `fc2_latent_proj [4096,1024]` are bf16 and *replicated* across TP),
512 experts, top-22, relu2 (ungated), per card at TP4 gate_up 672x1024 and down 1024x672, grouped_topk with
n_group=1, topk_group=1, routed_scaling_factor 5.0 and an e_score_correction_bias.

    LD_LIBRARY_PATH=$HOME/opt/openmpi/usr/lib/x86_64-linux-gnu \
      ~/venv-r9700/bin/python3 tuning/decode_moe_glue_probe.py 16 4

Run it with the engine stopped: measuring next to a live engine measures contention.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("R9K_LIB", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                              "r9700_vllm", "kernels", "libr9k.so"))
from r9700_vllm.kernels import moe as K  # noqa: E402

torch.cuda.init()
dev = torch.device("cuda")
bf = torch.bfloat16

S = int(sys.argv[1]) if len(sys.argv) > 1 else 16
SPEC = int(sys.argv[2]) if len(sys.argv) > 2 else 4
M = S * (1 + SPEC)                       # rows a step
E, TOPK, HID, LAT = 512, 22, 4096, 1024
N1, K1, N2, K2 = 672, 1024, 1024, 672    # per card at TP4 (gate_up 672x1024, down 1024x672)
numel = M * TOPK
g = torch.Generator(device=dev).manual_seed(0)


def time_graph(fn, iters=10, reps=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(iters):
                fn()
        run = graph.replay
    except Exception as e:
        print(f"  (not graph-capturable: {type(e).__name__}: {e}; eager timing)")
        run = lambda: [fn() for _ in range(iters)]  # noqa: E731
    best = []
    for _ in range(3):
        st, en = torch.cuda.Event(True), torch.cuda.Event(True)
        st.record()
        for _ in range(reps):
            run()
        en.record()
        torch.cuda.synchronize()
        best.append(st.elapsed_time(en) * 1e3 / (iters * reps))
    return min(best)


x = torch.randn(M, HID, device=dev, dtype=bf, generator=g)
gate_w = torch.randn(E, HID, device=dev, dtype=bf, generator=g) * 0.02
bias = torch.randn(E, device=dev, dtype=torch.float32, generator=g) * 0.1
logits = torch.randn(M, E, device=dev, dtype=bf, generator=g)
fc1_w = torch.randn(LAT, HID, device=dev, dtype=bf, generator=g) * 0.02
fc2_w = torch.randn(HID, LAT, device=dev, dtype=bf, generator=g) * 0.02
lat = torch.randn(M, LAT, device=dev, dtype=bf, generator=g)
gate_up = torch.randn(numel, N1, device=dev, dtype=bf, generator=g)
down_out = torch.randn(numel, N2, device=dev, dtype=bf, generator=g)
moe_out = torch.empty(M, N2, device=dev, dtype=bf)
norm_w = torch.ones(HID, device=dev, dtype=bf)
tw = torch.rand(M, TOPK, device=dev, dtype=torch.float32, generator=g)

res = {}

# 1. the router GEMM (GateLinear, bf16, replicated): [M,4096] x [512,4096]
res["router gate GEMM (bf16)"] = time_graph(lambda: torch.nn.functional.linear(x, gate_w))

# 2. grouped_topk with the score-correction bias (n_group=1, topk_group=1, scaling 5.0)
topk_ids, topk_w = None, None
try:
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import grouped_topk

    def _rt():
        return grouped_topk(x, logits, TOPK, True, 1, 1, "sigmoid", 5.0, bias)

    res["grouped_topk (512, top-22)"] = time_graph(_rt)
    # grouped_topk returns (topk_weights, topk_ids) -- getting this backwards feeds float weights to the align kernel,
    # which fails with "moe_align_block_size_kernel not implemented for 'Float'".
    topk_w, topk_ids = _rt()
except Exception as e:
    print(f"  grouped_topk failed: {type(e).__name__}: {e}")
    topk_ids = torch.randint(0, E, (M, TOPK), device=dev, dtype=torch.int32, generator=g)
    topk_w = tw

# 3. the align/sort tables over 512 experts
blk = K.MOE_BLOCK * K.pick_mt(numel, E)
try:
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

    res["moe_align_block_size"] = time_graph(
        lambda: moe_align_block_size(topk_ids, blk, E, None))
except Exception as e:
    print(f"  moe_align_block_size failed: {type(e).__name__}: {e}")

# 4. the two per-row fp8 quants and the activation quant
try:
    res["quant_rows_fp8 (M x 4096)"] = time_graph(lambda: K.quant_rows_fp8(x))
except Exception as e:
    print(f"  quant_rows_fp8 failed: {type(e).__name__}: {e}")
try:
    res["relu2_quant_fp8 (numel x 672)"] = time_graph(lambda: K.relu2_quant_fp8(gate_up))
except Exception as e:
    print(f"  relu2_quant_fp8 failed: {type(e).__name__}: {e}")

# 5. moe_sum over top-22 of the 1024-wide latent outputs
try:
    res["moe_sum (22 x 1024 -> 1024)"] = time_graph(
        lambda: K.moe_sum(down_out.view(M, TOPK, N2), None, moe_out))
except Exception as e:
    print(f"  libr9k moe_sum failed: {type(e).__name__}: {e}")
    try:
        from vllm.model_executor.layers.fused_moe.fused_moe import moe_sum
        res["moe_sum (vllm)"] = time_graph(lambda: moe_sum(down_out.view(M, TOPK, N2), moe_out))
    except Exception as e2:
        print(f"  vllm moe_sum failed too: {type(e2).__name__}: {e2}")

# 6. the bf16 latent shrink/expand, replicated across TP: they stream 16.8 MB a layer a step
res["fc1_latent bf16 (M,4096)->1024"] = time_graph(lambda: torch.nn.functional.linear(x, fc1_w))
res["fc2_latent bf16 (M,1024)->4096"] = time_graph(lambda: torch.nn.functional.linear(lat, fc2_w))

# 7. one layer norm (88 a step across the model)
try:
    from vllm import _custom_ops as vops
    norm_out = torch.empty_like(x)
    res["rms_norm (M x 4096)"] = time_graph(lambda: vops.rms_norm(norm_out, x, norm_w, 1e-5))
except Exception as e:
    print(f"  rms_norm failed: {type(e).__name__}: {e}")

print(f"\nM={M} rows (S={S}, spec={SPEC}) E={E} topk={TOPK} numel={numel} per card TP4")
tot = 0.0
for name, us in res.items():
    print(f"  {name:<32} {us:8.1f} us")
    tot += us
print(f"  {'TOTAL MoE glue a layer':<32} {tot:8.1f} us   x40 = {tot * 40 / 1e3:6.2f} ms/step")
print(f"  for reference: the two grouped GEMMs at this numel are 412 us a layer (tuning/decode_moe_sweep.py)")
