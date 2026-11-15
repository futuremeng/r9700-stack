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
  C. BOUNDS, needs vLLM importable but no GPU work: `r9700_vllm.models.gdn._bounds_problems`, the host range
     check that `R9K_GDN_DEBUG=1` now runs on the STOCK fallback path (gdn.py `_fallback_check`) -- the path the
     faulting step took and that nothing used to check. It covers the two walks that kernel can do: a conv-state
     line index outside the cache, and a token span outside the rows of x. Includes the guard firing on CPU
     tensors (one logger.error line, quiet when in range, layer 0 only) and a source check that the call sits
     before stock's `_forward_core`.
  D. THAT THE DIAGNOSTIC DOES NOT MOVE THE THING IT MEASURES, source level plus a unit check on `gdn._dbg_here`:
     `R9K_GDN_DEBUG=1` pays in host syncs, so it has to spend them where it loses no coverage. On this model (36
     GDN layers, from config.json's `layer_types`) the debug block would otherwise run on all of them -- 3
     synchronize() calls per layer per step, 108 per step -- which drains the launch queue and re-times the
     prefill/decode overlap a mixed step is being chased for. The checks read the same per-step metadata on every
     layer, so they run on layer 0 unless `R9K_GDN_DEBUG_ALL=1` asks for all of them.
  E. THE SECOND FAULT'S STEP, needs vLLM (the bounds part needs no GPU) and the GPU replay needs one card: the
     2026-11-15 05:11:18 step as its own arithmetic -- 23 spec rows of 3 tokens plus two new prefills of 160 and
     156 tokens landing on pages allocated in that step -- bounded, replayed through stock's conv update + conv
     fn, and `torch.cat`'d at the step's own q/k/v widths. Same limit as A and B: an index in range by arithmetic
     that no longer names live memory is out of reach of all of it.

Run:  python3 tests/test_conv1d_mixed_step.py            (A always; B and C skip if vLLM/GPU is missing)
      GPU=0 python3 tests/test_conv1d_mixed_step.py      (no GPU replay)
  On the native box, arm C needs the OpenMPI runtime the launcher itself puts on LD_LIBRARY_PATH
  (serve/native-flashnext.sh:60): LD_LIBRARY_PATH=$HOME/opt/openmpi/usr/lib/x86_64-linux-gnu \
  ~/venv-r9700/bin/python tests/test_conv1d_mixed_step.py
"""
from __future__ import annotations

import os
import pathlib
import re
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


# ------------------------------------------------- C. the fallback bounds check the fault would have hit ---
def bounds():
    print("\nC. the stock-path bounds check (needs vLLM importable, no GPU work)")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from r9700_vllm.models.gdn import _bounds_problems as bp
    except Exception as e:                                                      # noqa: BLE001
        print(f"  skip   r9700_vllm.models.gdn not importable here: {type(e).__name__}: {e}")
        return
    # The 2026-11-14 step's own indices: the new request's block_ids from the engine dump, the conv-state cache
    # is one line per KV block (1,146), and 610 non-spec tokens inside a 700-token step.
    dump = [1071, 125, 596, 56, 314, 840, 1041, 277, 1116, 989, 1008, 833, 654, 185]
    ck("the faulting step's own block ids pass: max 1116 inside 1146 conv-state lines",
       bp(dump, 1146, 610, 700) == [], str(bp(dump, 1146, 610, 700)))
    ck("a line index one past the cache is named (that is the walk the fault is)",
       any("leaves the conv-state cache" in s for s in bp([1146], 1146, 610, 700)), str(bp([1146], 1146, 610, 700)))
    ck("a garbage index far past it is named too (the page-aligned read we saw)",
       any("max 53761" in s for s in bp([53761], 1146, 610, 700)), str(bp([53761], 1146, 610, 700)))
    ck("the two padding ids are legal: PAD_SLOT_ID -1 and NULL_BLOCK_ID 0 are skipped by the kernels",
       bp([-1, 654, 0, -1], 1146, 610, 700) == [], str(bp([-1, 654, 0, -1], 1146, 610, 700)))
    ck("... and that is documented, not assumed: block 0 is indistinguishable from padding to the conv kernel",
       bp([0], 1146, 610, 700) == [])
    ck("a step whose non-spec tokens exceed the rows of x is named (the other walk in that kernel)",
       any("rows of x" in s for s in bp([7], 1146, 701, 700)), str(bp([7], 1146, 701, 700)))
    # The two span rules, added for the SECOND fault (its named kernel was a torch.cat, which reads a RANGE).
    ck("the two partitions of one step counted twice is named (non-spec 316 + spec 69 > 380 act)",
       any("the two partitions of one step" in s for s in bp([7], 1146, 316, 380, spec_cu_last=69)),
       str(bp([7], 1146, 316, 380, spec_cu_last=69)))
    ck("... and the same numbers are clean when act is the honest 385",
       bp([7], 1146, 316, 385, spec_cu_last=69) == [], str(bp([7], 1146, 316, 385, spec_cu_last=69)))
    ck("num_actual_tokens above the rows of the buffer we actually hand over is named (385 act, 384 rows)",
       any("rows of the tensors handed to stock" in s for s in bp([7], 1146, 316, 385, rows=384)),
       str(bp([7], 1146, 316, 385, rows=384)))
    ck("no prefill rows: nothing to check, nothing reported", bp([], 1146, 0, 90) == [])
    ck("the check reports several problems at once (one line is enough to read in the log)",
       len(bp([1146], 1146, 4000, 700)) == 2, str(bp([1146], 1146, 4000, 700)))

    # The wired-up guard, not just the helper: R9K_GDN_DEBUG=1 must actually call it on the fallback path, and
    # it must speak with one logger.error line naming the index. CPU tensors only -- the check's whole cost is
    # host syncs, which is why it sits behind DEBUG.
    import torch
    from r9700_vllm.models import gdn as G

    real_logger = G.logger
    seen = []

    class FakeLog:
        def error(self, fmt, *a):
            seen.append(fmt % a)

        def __getattr__(self, _n):
            return lambda *a, **k: None

    class FakeLayer:
        prefix = "model.layers.0.gdn"
        kv_cache = [torch.zeros(1146, 2560, 3, dtype=torch.bfloat16)]

    class FakeMd:
        num_prefills, num_decodes, num_spec_decodes, num_actual_tokens = 1, 0, 30, 700
        non_spec_query_start_loc = torch.tensor([0, 610], dtype=torch.int32)
        spec_query_start_loc = torch.tensor([0, 90], dtype=torch.int32)
        spec_state_indices_tensor = None

    G.logger = FakeLog()
    try:
        md = FakeMd()
        md.non_spec_state_indices_tensor = torch.tensor(dump[:9] + [654], dtype=torch.int32)
        G._fallback_check(FakeLayer(), md)
        ck("R9K_GDN_DEBUG on the stock fallback: an in-range step stays quiet (no log spam per step)",
           seen == [], str(seen[:2]))
        md.non_spec_state_indices_tensor = torch.tensor([654, 1146], dtype=torch.int32)
        G._fallback_check(FakeLayer(), md)
        ck("... and one out of range is named, with the value and the cache size",
           len(seen) == 1 and "leaves the conv-state cache" in seen[0] and "1146" in seen[0], str(seen)[:200])
        seen.clear()
        layer = FakeLayer()
        layer.prefix = "model.layers.7.gdn"
        md.non_spec_state_indices_tensor = torch.tensor([9999], dtype=torch.int32)
        G._fallback_check(layer, md)
        ck("the helper checks whichever layer it is handed -- the CALLER decides who pays, through _dbg_here. "
           "It used to filter here on a hardcoded .layers.0., and that is why the 2026-11-15 06:55 hit named the "
           "dying layer (layers.1, stage counts 615/614/614) and still could not say what was wrong with its "
           "indices: the one layer that mattered had never been looked at",
           len(seen) == 1 and "leaves the conv-state cache" in seen[0], str(seen)[:200])
        ck("the helper's own body carries no layer test at all (its docstring may tell the story; its code may not)",
           '.layers.0." not in self.prefix' not in open(str(pathlib.Path(__file__).resolve().parent.parent /
                                                         "r9700_vllm" / "models" / "gdn.py")).read()
                                                        .split("def _fallback_check")[1].split("def _debug_check")[0])
        seen.clear()
        layer = FakeLayer()
        md.non_spec_state_indices_tensor = torch.tensor([654], dtype=torch.int32)
        md.num_actual_tokens = 700
        G._fallback_check(layer, md, rows=699)              # one row fewer than the metadata claims
        ck("and the wired-up guard takes the rows of the real buffer, so a metadata claiming one row more is "
           "named before stock's cat reads it",
           len(seen) == 1 and "rows of the tensors handed to stock" in seen[0] and "699" in seen[0], str(seen)[:220])
    finally:
        G.logger = real_logger

    src = open(os.path.join(os.path.dirname(__file__), "..", "r9700_vllm", "models", "gdn.py")).read()
    fb = src.split("dbg = _dbg_here(self) and not torch.cuda.is_current_stream_capturing()")[-1]
    ck("the guard is on the path the faulting step took: it runs before stock _forward_core is called, and it is "
       "handed the rows of the buffer being handed over",
       "_fallback_check(self, md, rows=qkvz.shape[0])" in fb and "self._forward_core(" in fb
       and fb.index("_fallback_check(self, md, rows=qkvz.shape[0])") < fb.index("self._forward_core("))

    # D. the diagnostic's own cost, kept where it loses nothing (see the docstring above).
    print("\nD. R9K_GDN_DEBUG spends its syncs on layer 0, not on every GDN layer")
    ck("both debug blocks ask _dbg_here, and no block asks bare DEBUG",
       src.count("dbg = _dbg_here(self) and not torch.cuda.is_current_stream_capturing()") == 2
       and "dbg = DEBUG" not in src, f"_dbg_here sites: {src.count('dbg = _dbg_here(self)')}")
    body = src.split("def gdn_core(")[1].split("def _trace(")[0]
    syncs = [i for i in range(len(body)) if body.startswith("torch.cuda.synchronize()", i)]
    unguarded = [body[max(0, i - 260):i] for i in syncs if "if dbg:" not in body[max(0, i - 260):i]]
    ck(f"every sync inside gdn_core sits under an `if dbg:` ({len(syncs)} of them) -- an unguarded one is paid "
       "on every step whether or not anyone is diagnosing",
       bool(syncs) and not unguarded, f"{len(syncs)} syncs, {len(unguarded)} unguarded")
    import torch
    from r9700_vllm.models import gdn as G

    class Lay:
        def __init__(self, n):
            self.prefix = f"model.layers.{n}.gdn"

    d0, d1 = G.DEBUG, G.DEBUG_ALL
    try:
        G.DEBUG, G.DEBUG_ALL = True, False
        ck("with DEBUG on, layer 0 pays and layer 7 does not (the metadata is per step, so one layer sees the "
           "whole step)", G._dbg_here(Lay(0)) and not G._dbg_here(Lay(7)))
        G.DEBUG_ALL = True
        ck("R9K_GDN_DEBUG_ALL=1 is the escape hatch, and it is opt-in",
           G._dbg_here(Lay(7)) and G._dbg_here(Lay(22)))
        G.DEBUG, G.DEBUG_ALL = False, True
        ck("and none of it costs anything with DEBUG off, whatever DEBUG_ALL says", not G._dbg_here(Lay(0)))
    finally:
        G.DEBUG, G.DEBUG_ALL = d0, d1


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


# ------------------------------- E. the SECOND fault's step: two prefills arriving into 23 spec rows ---------
def mixed():
    """2026-11-15 05:11:18, from the engine's own SchedulerOutput dump (logs/setup-044236.log:10254).

    23 spec-decoding requests at 3 tokens each (one real token + two MTP drafts, every draft still -1), plus TWO
    NEW requests of 160 and 156 prefill tokens: `total_num_scheduled_tokens=385`, `kv_cache_usage=0.5913`,
    `num_running_reqs=25`, `num_waiting_reqs=0`, `new_block_ids_to_zero=[246, 464]`. Both new requests report
    `num_computed_tokens=0`, so neither got a prefix hit: 316 of the 385 tokens are prefill, which is above
    `R9K_GDN_PREFILL_MAX=256`, so `_prefill_ok` is false and the step went to stock's `_forward_core` -- the same
    door the 11-14 fault went through. The kernel the driver named this time is
    `at::native::CatArrayBatchedCopy_contig` (a `torch.cat`), grid `[13312, 3, 1]`, not the conv.

    Arm A bounded the 11-14 arithmetic (one 610-token prefill into 30 spec rows). This is different arithmetic and
    had never been bounded: TWO prefills, landing on GDN state pages allocated **in this very step** (that is what
    `new_block_ids_to_zero` says), sharing the step with 23 spec rows. What it cannot catch is unchanged from A:
    an argument that is in range by arithmetic and no longer describes live memory -- for both faults, the
    surviving suspect. So the finding here is again a negative, and negatives are what narrows this.
    """
    print("\nE. the 2026-11-15 05:11 step: two prefills (160, 156) arriving into 23 spec rows of 3 tokens")
    from r9700_vllm.models.gdn import _bounds_problems as bp

    LINES = 1146                       # conv-state lines at this recipe (arm B measures the same launch shape)
    # Which group of the six is the GDN state, and which pages this step had to zero, read out of the dump itself
    # rather than asserted from memory: every new request carries six block-id groups, the LAST one is the mamba
    # group (one page per request), and `new_block_ids_to_zero` is the engine saying which pages it zeroed for them.
    dline, pages, zeroed = "", [], []
    for cand in (os.path.expanduser("~/logs/setup-044236.log"), "/home/jiulu/logs/setup-044236.log"):
        if os.path.exists(cand):
            with open(cand, errors="replace") as fh:
                for ln in fh:
                    if "Dumping scheduler output" in ln:
                        dline = ln
                        break
            break
    if dline:
        for m in re.finditer(r"block_ids=\((.*?)\),num_computed_tokens", dline):
            groups = re.findall(r"\[([^\]]*)\]", m.group(1))
            pages.append([int(x) for x in groups[-1].split(",") if x.strip()])
        z = re.search(r"new_block_ids_to_zero=\[([^\]]*)\]", dline)
        zeroed = [int(x) for x in z.group(1).split(",")] if z else []
        ck(f"the dump read here: {len(pages)} new requests, mamba group = {[p for p in pages]}, "
           f"new_block_ids_to_zero = {zeroed} -- the last of the six groups is the GDN page, and it is the page "
           "the step zeroed",
           len(pages) == 2 and all(len(p) == 1 for p in pages) and sorted(sum(pages, [])) == sorted(zeroed),
           f"pages {pages}, zeroed {zeroed}")
        ck("both new requests start with nothing computed (no prefix hit), which is what puts 316 prefill tokens "
           "in one step",
           len(re.findall(r"num_computed_tokens=0,", dline)) >= 2,
           str(re.findall(r"num_computed_tokens=\d+,", dline)[:4]))
        ck("and 23 of the 25 running requests are spec rows with 2 draft tokens each (the dump's own -1, -1)",
           dline.count("[-1, -1]") == 23, f"{dline.count('[-1, -1]')} spec rows in the dump")
    else:
        print("  skip   the 05:11 dump is not on this box (~/logs/setup-044236.log): the shapes below are the ones "
              "read from it on 2026-11-15, not re-read now")
    ck("the step's own token arithmetic: 385 = 23x3 + 160 + 156, and 316 of them are prefill -- above 256, so "
       "stock's path (the 11-14 door, again)", 23 * 3 + 160 + 156 == 385 and 160 + 156 == 316 > 256)
    lines_new = sum(pages, []) or [246, 464]          # from the dump when it is on this box, else as read on 11-15
    ck("the two walks of the guard, on THIS shape: both new pages inside 1146 lines, 316 non-spec tokens inside "
       f"385 rows of x -- nothing to report ({lines_new})",
       bp(lines_new, LINES, 316, 385) == [], str(bp(lines_new, LINES, 316, 385)))
    ck("the spans of THIS step add up under the new rules: 316 non-spec + 69 spec = the 385 the dump reports, and "
       "385 rows exist",
       bp(lines_new, LINES, 316, 385, spec_cu_last=23 * 3, rows=385) == [],
       str(bp(lines_new, LINES, 316, 385, spec_cu_last=23 * 3, rows=385)))
    ck("and the check still bites on this shape if the step is read wrongly (one token too many in the span)",
       any("rows of x" in s for s in bp(lines_new, LINES, 386, 385)), str(bp(lines_new, LINES, 386, 385)))
    ck("a page index one line past the cache is still the fault it was for the 11-14 step",
       any("leaves the conv-state cache" in s for s in bp(lines_new + [LINES], LINES, 316, 385)))
    print("  note: the 23 spec rows' own line indices are NOT in the dump (only new requests carry block_ids in "
          "it), so what is bounded here is the two new pages and the token spans -- stated, not glossed.")

    if os.environ.get("GPU", "1") == "0":
        print("  skip   GPU=0")
        return
    try:
        import torch
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
        from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata
    except Exception as e:                                                      # noqa: BLE001
        print(f"  skip   no GPU-capable vLLM here: {type(e).__name__}: {e}")
        return
    if not torch.cuda.is_available():
        print("  skip   torch.cuda.is_available() is false")
        return
    dev, DIM, W = torch.device("cuda"), 2560, 4      # qkv_size at TP4 (key 2048*2 + value 6144)/4, conv width 4
    bf = torch.bfloat16
    # the spec half: 23 rows of 3 tokens, each on its own legal conv-state line, 2 of the 2 drafts accepted
    # (this launch's measured mean acceptance length is 2.38-2.59, so 2 is the ordinary case rather than the good one)
    xs = torch.randn(69, DIM, dtype=bf, device=dev)
    states = torch.zeros(LINES, DIM, W - 1, dtype=bf, device=dev)
    weight = torch.randn(DIM, W, dtype=bf, device=dev)
    bias = torch.randn(DIM, dtype=bf, device=dev)
    cu = torch.arange(0, 70, 3, dtype=torch.int32, device=dev)
    sidx = torch.arange(1, 24, dtype=torch.int32, device=dev)
    acc = torch.full((23,), 2, dtype=torch.int32, device=dev)
    # the prefill half: two rows of 160 and 156 tokens on the pages this step allocated (246, 464), no initial state
    xp = torch.randn(316, DIM, dtype=bf, device=dev)
    qsl = torch.tensor([0, 160, 316], dtype=torch.int32, device=dev)
    cidx = torch.tensor((sum(pages, []) or [246, 464]), dtype=torch.int32, device=dev)
    has = torch.tensor([False, False], dtype=torch.bool, device=dev)
    nums, bptr, toff = compute_causal_conv1d_metadata(torch.tensor([0, 160, 316], dtype=torch.int32), device=dev)

    class MD:
        pass

    md = MD()
    md.nums_dict, md.batch_ptr, md.token_chunk_offset_ptr = nums, bptr, toff
    ok, msg = True, ""
    try:
        upd = causal_conv1d_update(xs, states, weight, bias, "silu", conv_state_indices=sidx,
                                   num_accepted_tokens=acc, query_start_loc=cu, max_query_len=3,
                                   validate_data=False, out=torch.empty_like(xs))
        torch.cuda.synchronize()
        pre = causal_conv1d_fn(xp.transpose(0, 1), weight, bias, conv_states=states, query_start_loc=qsl,
                               cache_indices=cidx, has_initial_state=has, activation="silu", metadata=md,
                               validate_data=True)
        torch.cuda.synchronize()
    except Exception as e:                                                      # noqa: BLE001
        ok, msg = False, f"{type(e).__name__}: {e}"
    ck("the mixed step replays: spec conv update (23 rows x 3) then the prefill conv (160+156) on the pages this "
       "step allocated, sync'd between the two", ok, msg)
    if ok:
        ck("the spec rows' lines and the two new lines are all that moved; neighbours untouched",
           bool(states[246].float().abs().sum() > 0) and bool(states[464].float().abs().sum() > 0)
           and bool(states[24].float().abs().sum() > 0)
           and bool(states[0].float().abs().sum() == 0 and states[100].float().abs().sum() == 0))
        # the cat the driver named: a contiguous torch.cat over the step's own q/k/v widths at TP4
        # (qkv_size 2560 = q 512 + k 512 + v 1536). A synthetic cat cannot reproduce a stale-pointer read, and
        # this does not claim to; what it establishes is that this step's sizes are not themselves a fault.
        q, k = torch.randn(385, 512, dtype=bf, device=dev), torch.randn(385, 512, dtype=bf, device=dev)
        v = torch.randn(385, 1536, dtype=bf, device=dev)
        cat = torch.cat([q, k, v], dim=-1)
        torch.cuda.synchronize()
        ck("the same step's torch.cat at q/k/v widths (512/512/1536, the CatArrayBatchedCopy_contig family) is "
           "in shape and finite -- a synthetic cat cannot catch a stale buffer, and does not claim to",
           tuple(cat.shape) == (385, 2560) and bool(torch.isfinite(cat.float()).all()), str(tuple(cat.shape)))
    del xs, xp, states, weight, bias, q, k, v
    torch.cuda.empty_cache()


if __name__ == "__main__":
    geom()
    bounds()
    replay()
    mixed()
    print("\nPASS" if fail == 0 else "FAIL", f"({fail} failing)")
    sys.exit(1 if fail else 0)
