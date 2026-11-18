"""What one Nemotron-H Mamba2 layer costs at decode, measured per kernel with this checkpoint's shapes.

Shapes come from the checkpoint header (not guessed): in_proj [18560,4096] FP8 with a per-tensor input_scale
(static W8A8), conv1d over 10240 channels (= 8192 inner + 1024 B + 1024 C: n_groups 8 x d_state 128), out_proj
[4096,8192] FP8, 128 heads x 64, d_state 128, conv kernel 4. TP=4 -> per card: heads 32, groups 2, xBC 2560,
in_proj N 4640, out_proj [4096, 2048]. The decode batch with MTP=4 is S x (1+4) rows.

Timed with CUDA graphs (10 calls per graph) so launch overhead is out of the number, like tuning/decode_moe_sweep.py.

Run it as the operator does, with the OpenMPI libraries on the loader path (torch will not import without them), and
with the GPUs to itself -- run it with the engine stopped, or the numbers measure contention rather than kernels:

    LD_LIBRARY_PATH=$HOME/opt/openmpi/usr/lib/x86_64-linux-gnu \
      ~/venv-r9700/bin/python3 tuning/decode_mamba_probe.py 16 4
"""
import sys

import torch

torch.cuda.init()
dev = torch.device("cuda")
bf, f32 = torch.bfloat16, torch.float32

S = int(sys.argv[1]) if len(sys.argv) > 1 else 16      # sequences in flight
SPEC = int(sys.argv[2]) if len(sys.argv) > 2 else 4    # speculative tokens
M = S * (1 + SPEC)                                     # rows through the projections
TP = 4
H_IN, H_ID, D_STATE, GROUPS, KERN = 128 // TP, 64, 128, 8 // TP, 4
INNER, NC = (128 * 64) // TP, GROUPS * D_STATE         # 2048 x, 256 B, 256 C per card
XBC = INNER + 2 * NC                                   # 2560
N_IN, N_OUT = 18560 // TP, 4096                        # in_proj rows, out_proj rows
SLOTS = 64
g = torch.Generator(device=dev).manual_seed(0)


def time_graph(fn, iters=10, reps=30):
    """Graph-timed: capture `iters` calls, replay `reps` times, report us per call.
    Falls back to plain event timing if the kernel cannot be captured (a capture-hostile path is itself news)."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(iters):
                fn()
        torch.cuda.synchronize()
        run = graph.replay
    except Exception as e:
        print(f"  (not graph-capturable: {type(e).__name__}: {e}; timing eagerly instead)")
        run = lambda: [fn() for _ in range(iters)]
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


# ---- fp8 projections (static per-tensor scales, as the checkpoint declares) -------------------------------
a = torch.randn(M, 4096, device=dev, dtype=f32).to(torch.float8_e4m3fn)
w_in = torch.randn(N_IN, 4096, device=dev, dtype=f32).to(torch.float8_e4m3fn)
s_in = torch.ones((), device=dev, dtype=f32)
x_in = torch.randn(M, INNER + 2 * NC, device=dev, dtype=bf)          # in_proj output feeds conv
h_out = torch.randn(M, INNER, device=dev, dtype=bf)
w_out = torch.randn(N_OUT, INNER, device=dev, dtype=f32).to(torch.float8_e4m3fn)
out = torch.empty(M, N_OUT, device=dev, dtype=bf)

t_in = time_graph(lambda: torch._scaled_mm(a, w_in.t(), scale_a=s_in, scale_b=s_in, out_dtype=bf))
t_out = time_graph(lambda: torch._scaled_mm(h_out.to(torch.float8_e4m3fn), w_out.t(),
                                            scale_a=s_in, scale_b=s_in, out_dtype=bf))

# ---- conv1d update + selective state update (vLLM's own Triton kernels) ----------------------------------
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update

conv_w = (torch.randn(XBC, 1, KERN, device=dev, dtype=bf) * 0.2).squeeze(1)   # the op wants (dim, width)
conv_b = torch.randn(XBC, device=dev, dtype=bf) * 0.1
conv_state = torch.randn(SLOTS, XBC, KERN, device=dev, dtype=bf)
idx = torch.arange(S, device=dev, dtype=torch.int32)
nacc = torch.full((S,), 1 + SPEC, device=dev, dtype=torch.int32)
qsl = torch.arange(0, (S + 1) * (1 + SPEC), 1 + SPEC, device=dev, dtype=torch.int32)


def conv_call():
    causal_conv1d_update(x_in, conv_state, conv_w, conv_b, None, conv_state_indices=idx,
                         num_accepted_tokens=nacc, query_start_loc=qsl, max_query_len=1 + SPEC)


try:
    t_conv = time_graph(conv_call)
except Exception as e:
    t_conv = float("nan")
    print(f"  conv1d_update failed: {type(e).__name__}: {e}")

ssm_state = torch.randn(SLOTS, H_IN, H_ID, D_STATE, device=dev, dtype=f32)   # (slots, heads, head_dim, d_state)
dt = torch.rand(M, H_IN, H_ID, device=dev, dtype=f32) * 0.1
A = -torch.rand(H_IN, H_ID, D_STATE, device=dev, dtype=f32) - 0.5
Bc = torch.randn(M, GROUPS, D_STATE, device=dev, dtype=bf)
Cc = torch.randn(M, GROUPS, D_STATE, device=dev, dtype=bf)
D = torch.randn(H_IN, H_ID, device=dev, dtype=bf)
dt_bias = torch.randn(H_IN, H_ID, device=dev, dtype=f32) * 0.1
h_ssm = h_out.view(M, H_IN, H_ID)
ssm_out = torch.empty(M, H_IN, H_ID, device=dev, dtype=bf)
# the plain (non-ReplaySSM) path keeps a 2-slot state window per sequence, one column per verification step
state_idx = torch.arange(S, device=dev, dtype=torch.int32).unsqueeze(1).repeat(1, 2).contiguous()


def ssm_call():
    selective_state_update(ssm_state, h_ssm, dt, A, Bc, Cc, D, dt_bias, None, dt_softplus=True,
                           state_batch_indices=state_idx, dst_state_batch_indices=state_idx,
                           out=ssm_out, num_accepted_tokens=nacc, cu_seqlens=qsl, is_blackwell=False)


try:
    t_ssm = time_graph(ssm_call)
except Exception as e:
    t_ssm = float("nan")
    print(f"  selective_state_update failed: {type(e).__name__}: {e}")

tot = t_in + t_out + t_conv + t_ssm
print(f"S={S} spec={SPEC} rows={M} per card TP{TP}")
for name, us in (("in_proj fp8 GEMM", t_in), ("conv1d update", t_conv),
                 ("selective_state_update", t_ssm), ("out_proj fp8 GEMM", t_out)):
    print(f"  {name:<24} {us:8.1f} us")
print(f"  {'TOTAL a mamba layer':<24} {tot:8.1f} us   x40 layers = {tot * 40 / 1e3:6.2f} ms/step")
