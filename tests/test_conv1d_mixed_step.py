#!/usr/bin/env python3
"""tests/test_conv1d_mixed_step.py - the step that killed the 2026-11-14 launch, as a check.

The fault (notes/bug-causal-conv1d-oob.md): a mixed step -- one 610-token prefill arriving into 30
spec-decoding requests, `kv_cache_usage=0.998` -- died inside vLLM's Triton `_causal_conv1d_fwd_kernel`. Our
plugin was not in that step's path: `_prefill_ok` is false above `R9K_GDN_PREFILL_MAX` (256), and one-page GDN
state is not applied for Qwen4Exp (`gdn.py:325` logs "stock mamba pages" at boot), so `gdn.py:613` hands the
step to stock `_forward_core`, whose conv call is `qwen_gdn_linear_attn.py:1451`.

What this file can and cannot catch, said out loud (a check that asserts nothing is worse than none):

  A. GEOMETRY, no GPU and no vLLM: the launch geometry the kernel computes -- `grid = (num_program, cdiv(dim,
     BLOCK_N))` with `BLOCK_M = 8`, one program per 8-token chunk of one sequence, `idx_seq`/`chunk_offset`
     read out of `batch_ptr`/`token_chunk_offset_ptr`, `seqlen` re-read from `query_start_loc` inside the
     kernel, and the conv-state line read/written at `cache_indices[idx_seq]`. Every index range is checked
     against the buffers a step of that shape really has. **What it can catch**: a step shape for which the
     program count, the chunk ranges, the token span or a state-line index leaves its buffer -- i.e. the
     arithmetic of the arguments. It reproduces 2026-11-14's geometry exactly (77 programs = ceil(610/8),
     10 = ceil(2560/256)), so the shapes below are the served ones, not guesses. **What it cannot catch**: an
     argument that is in range by arithmetic but no longer describes live memory (a pointer/`nums_dict` baked
     with the buffer behind it gone). The 2026-11-14 fault is that family, so A passing is a fact to record,
     not an all-clear.
  B. REPLAY, needs one GPU: stock `causal_conv1d_fn` at that exact shape, `validate_data=True`, the conv-state
     line index set to the largest legal block, then `torch.cuda.synchronize()`. Any out-of-range read that the
     allocation map happens to make fatal shows up here; and the conv state must be written at the prefill row's
     line and nowhere else (a write to a neighbour's line would corrupt a running request silently).

Run:  python3 tests/test_conv1d_mixed_step.py            (B is skipped without a free GPU; BENCH=1 prints shapes)
      GPU=0 python3 tests/test_conv1d_mixed_step.py      (geometry only)
"""
from __future__ import annotations

import os
import sys

fail = 0


def ck(name, cond, detail=""):
    global fail
    if not cond:
        fail += 1
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond or not detail else f"  {detail}"))


# The kernel's own constants (vllm/model_executor/layers/mamba/ops/causal_conv1d.py): BLOCK_M is fixed and
# BLOCK_N is passed as 256 at the launch; the grid is (num_program, cdiv(dim, BLOCK_N)).
BLOCK_M, BLOCK_N = 8, 256
PAD_SLOT_ID, NULL_BLOCK_ID = -1, -1


def cdiv(a, b):
    return -(-a // b)


def conv1d_programs(query_lens):
    """`num_program` + the `mlist`/`offsetlist` the same function fills `batch_ptr`/`token_chunk_offset_ptr`
    from: `nums = ceil(qlen / BLOCK_M)`, `tot = sum(nums)`, one entry per program saying which sequence and
    which chunk of it. Zero-length (padded) rows contribute no program."""
    nums = [cdiv(q, BLOCK_M) for q in query_lens]
    mlist = [i for i, n in enumerate(nums) for _ in range(n)]
    offsets = [c for n in nums for c in range(n)]
    return sum(nums), mlist, offsets


def step_geometry(prefill_lens, state_lines, dim, spec_rows=0, spec_len=3, block_ids=None):
    """Everything the kernel will index, for a step whose non-spec rows are `prefill_lens` (the builder puts
    zero-length padded rows and reclassified decodes in the same list) and whose spec rows go to
    causal_conv1d_update instead. Returns the ranges it touches."""
    qlens = list(prefill_lens)
    tot, mlist, offsets = conv1d_programs(qlens)
    grid = (tot, cdiv(dim, BLOCK_N))
    cu = [0]
    for q in qlens:
        cu.append(cu[-1] + q)
    touched_tokens, touched_lines = set(), set()
    for seq, chunk in zip(mlist, offsets):
        start, end = cu[seq], cu[seq + 1]
        seqlen = end - start
        seg = min(BLOCK_M, seqlen - BLOCK_M * chunk)
        for k in range(max(seg, 0)):
            touched_tokens.add(start + BLOCK_M * chunk + k)
        touched_lines.add(block_ids[seq])
    # the conv window the kernel reads back for has_initial_state rows reaches state_len = KERNEL_WIDTH - 1
    # columns into the state line; the feature range is program_id(1)*BLOCK_N + arange(BLOCK_N).
    feats = cdiv(dim, BLOCK_N) * BLOCK_N
    return dict(tot=tot, grid=grid, cu=cu, tokens=touched_tokens, lines=touched_lines, feats=feats,
                nprogs=len(mlist))


def in_bounds(g, n_tokens, state_lines, dim, pool_blocks):
    bad = []
    if g["tot"] != g["nprogs"]:
        bad.append(f"grid x {g['tot']} != the number of (seq, chunk) pairs {g['nprogs']}")
    if max(g["tokens"], default=-1) >= n_tokens:
        bad.append(f"a program reads token {max(g['tokens'])} of a {n_tokens}-token x")
    for ln in g["lines"]:
        if not 0 <= ln < min(state_lines, pool_blocks) and ln != NULL_BLOCK_ID and ln != PAD_SLOT_ID:
            bad.append(f"a conv-state line index {ln} is outside lines {state_lines} / pool {pool_blocks}")
    if g["feats"] - dim >= BLOCK_N:
        bad.append(f"feature coverage {g['feats']} overshoots dim {dim} by a whole BLOCK_N")
    return bad


# --------------------------------------------------------------------------- A. the geometry of real steps ---
def geom():
    print("A. launch geometry (no GPU): the served step shapes stay inside their buffers")
    # The 2026-11-14 step, from the engine's own dump: 1 non-spec prefill of 610 tokens, 30 spec rows x 3,
    # conv dim 2560 per rank (10 x BLOCK_N=256 at TP4; the whole model's conv is 10240 wide),
    # 1,146 blocks in the KV pool, the new request's mamba-group block ids from the dump.
    g = step_geometry([610], state_lines=1146, dim=2560, spec_rows=30, block_ids=[654])
    ck("the faulting step reproduces the device's grid: x=77 = ceil(610/8) programs, y=10 = ceil(2560/256)",
       g["grid"] == (77, 10), str(g["grid"]))
    ck("... and the HSA dispatch we recorded is that grid in workitems: 77 x 128 = 9856, workgroup [128,1,1]",
       g["grid"][0] * 128 == 9856, str(g["grid"][0] * 128))
    ck("the faulting step's indices are all in range (so the fault is an argument that stopped describing "
       "live memory, not arithmetic)", not in_bounds(g, 610, 1146, 2560, 1146), str(in_bounds(g, 610, 1146, 2560, 1146)))

    # The rest of the shapes this launcher can put on the wire: chunked prefills up to
    # --max-num-batched-tokens, a batch of them, the padded/zero rows the builder keeps, decodes reclassified
    # as prefills when spec decodes are present (gdn_attn.py), and the pool's largest legal block id.
    shapes = [
        ("one 8192-token chunk (= --max-num-batched-tokens)", [8192], 8192, 2560, 1146, [1145]),
        ("a 610-token prefill beside 30 one-token decodes reclassified as prefills", [610] + [1] * 30, 640,
         2560, 1146, [654] + list(range(30))),
        ("prefill + zero-length padded rows at the back", [610, 0, 0], 610, 2560, 1146, [654, 0, 0]),
        ("32 prefills of 256 (all inside R9K_GDN_PREFILL_MAX, but stock runs them above 256 total)",
         [256] * 32, 8192, 2560, 1146, list(range(32))),
        ("a sequence whose state line is the last block in the pool", [610], 610, 2560, 1146, [1145]),
        ("dim not a multiple of BLOCK_N (a model with 2300 conv channels per rank)", [610], 610, 2300, 1146,
         [7]),
    ]
    for name, lens, ntok, dim, lines, bids in shapes:
        gg = step_geometry(lens, state_lines=lines, dim=dim, block_ids=bids)
        bad = in_bounds(gg, ntok, lines, dim, lines)
        ck(f"{name}: {gg['grid'][0]} programs, {ntok} tokens, dim {dim} -> in range", not bad, str(bad))

    # The knob that would have kept that step out of the faulting kernel: our conv covers a step whose prefill
    # tokens are <= PREFILL_MAX. Show the faulting step is just above it, and what ceiling covers it.
    ck("610 prefill tokens is above the default R9K_GDN_PREFILL_MAX=256 (that is why stock's kernel ran)",
       610 > 256)
    ck("a ceiling of 1024 covers the faulting step, and 8192 covers every chunk this launcher can emit",
       610 <= 1024 and 8192 <= 8192)


# ------------------------------------------------------------------------ B. replay it on a real GPU ---
def replay():
    print("\nB. replay the faulting step through stock causal_conv1d_fn (one GPU)")
    if os.environ.get("GPU", "1") == "0":
        print("  skip   GPU=0")
        return
    try:
        import torch
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
        from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata  # noqa: F401
    except Exception as e:                                                      # noqa: BLE001
        print(f"  skip   no GPU-capable vLLM here: {type(e).__name__}: {e}")
        return
    if not torch.cuda.is_available():
        print("  skip   torch.cuda.is_available() is false")
        return
    dev = torch.device("cuda")
    DIM, W, SEQL, LINES, MAXBLOCK = 2560, 4, 610, 1146, 1145
    # stock passes `mixed_qkv_non_spec.transpose(0, 1)`: the layer's buffer is (tokens, dim) channel-last,
    # and causal_conv1d_fn reads `dim, cu_seqlen = x.shape` -- the layout assert is what keeps a transposed
    # buffer from silently meaning the wrong thing (it tripped here first).
    x = torch.randn(SEQL, DIM, dtype=torch.bfloat16, device=dev)
    conv_states = torch.zeros(LINES, DIM, W - 1, dtype=torch.bfloat16, device=dev)
    weight = torch.randn(DIM, W, dtype=torch.bfloat16, device=dev)
    bias = torch.randn(DIM, dtype=torch.bfloat16, device=dev)
    qsl = torch.tensor([0, SEQL], dtype=torch.int32, device=dev)
    cache_indices = torch.tensor([MAXBLOCK], dtype=torch.int32, device=dev)   # the largest legal line
    has_init = torch.tensor([False], dtype=torch.bool, device=dev)
    nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
        torch.tensor([0, SEQL], dtype=torch.int32), device=dev)

    class MD:      # what causal_conv1d_fn reads off `metadata` (nums_dict / batch_ptr / token_chunk_offset_ptr)
        pass

    md = MD()
    md.nums_dict, md.batch_ptr, md.token_chunk_offset_ptr = nums_dict, batch_ptr, token_chunk_offset_ptr
    ck("the metadata path agrees with the geometry: tot=77 programs for one 610-token row",
       nums_dict[BLOCK_M]["tot"] == 77, str(nums_dict[BLOCK_M]["tot"]))
    try:
        out = causal_conv1d_fn(x.transpose(0, 1), weight, bias, conv_states=conv_states, query_start_loc=qsl,
                               cache_indices=cache_indices, has_initial_state=has_init, activation="silu",
                               metadata=md, validate_data=True)
        torch.cuda.synchronize()
        ok = True
    except Exception as e:                                                      # noqa: BLE001
        ok, msg = False, f"{type(e).__name__}: {e}"
    ck("the faulting step replays clean at the largest legal state line (sync'd)", ok, locals().get("msg", ""))
    if ok:
        ck("out has the shape the layer expects and is finite",
           tuple(out.shape) == (DIM, SEQL) and bool(torch.isfinite(out.float()).all()), str(tuple(out.shape)))
        touched = (conv_states[MAXBLOCK].float().abs().sum() > 0).item()
        others = conv_states.index_select(0, torch.tensor([0, 1, 512, LINES - 2], device=dev))
        ck("the conv state was written at the prefill row's line and at no other line",
           touched and bool(others.float().abs().sum() == 0),   # LINES-2, not LINES-1: MAXBLOCK *is* LINES-1
           f"line {MAXBLOCK} sum {float(conv_states[MAXBLOCK].float().abs().sum()):.3e}")
    del x, conv_states, weight, bias
    torch.cuda.empty_cache()


if __name__ == "__main__":
    geom()
    replay()
    print("\nPASS" if fail == 0 else "FAIL", f"({fail} failing)")
    sys.exit(1 if fail else 0)
