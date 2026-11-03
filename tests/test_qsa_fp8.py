"""r9k QSA fp8-e4m3 main KV (kernels/r9k_qsa.hip r9k_qsa_attn_f8) against the same math in torch.

Three numbers per case, printed as a table and asserted:

  kernel-vs-ref (bf16)   our bf16 attention vs a torch reference over the same bf16 K/V   (the existing path)
  kernel-vs-ref (fp8)    our fp8 attention vs a torch reference over the SAME fp8-quantized K/V
                         -> this is the one that says the fp8 read path is CORRECT: it must be as tight as bf16
  quant (bf16 vs fp8)    both kernels, each against its own reference -> the accuracy question, isolated
                         (synthetic data: a peaked softmax on random keys flips winners, so this column is an upper
                          bound, not the model's number -- that one comes from bench/quality.py. On the shapes here
                          e4m3 costs ~5% per K/V element and 0.10-0.21 of max output on random keys, which is
                          exactly why the gate is not optional.)

Case mix: GQA 6 and 12 (the TP4 rank sees 6 q heads / 1 kv head; the 27B sees 12), rows 1..256 (decode through a
prefill chunk), nsplit 1 and >1 (the split+merge path), page 448 (what a bf16 cache picks) and 896 (what an fp8
cache picks once the attention page has to stay >= the mamba page), descale 1.0 and 0.5 (a non-unity scale must
descale K and V, not one of them).

Run: /home/jiulu/venv-r9700/bin/python3 tests/test_qsa_fp8.py
"""
import itertools
import os
import sys

import torch

from r9700_vllm.attn.qsa import sparse_attention

DEV = "cuda"
HD = 256
GS = 4                                   # indexer_compress_ratio: the kernel's group size


def quant(t, scale):
    """What the fused writer does to K/V when the destination cache is fp8 (round to e4m3 of the scaled value)."""
    if scale == 1.0:
        return t.to(torch.float8_e4m3fn)
    return (t / scale).to(torch.float8_e4m3fn)


def reference(q, kc, vc, idx, bt, t2r, qsl, seqlens, page, desc):
    """Plain attention over each row's expanded token list, in fp32, on the dequantized cache."""
    rows, hq, _ = q.shape
    hkv = kc.shape[2]
    gqa = hq // hkv
    kd, vd = desc
    kc = kc.float() * kd if kc.dtype == torch.float8_e4m3fn else kc.float()
    vc = vc.float() * vd if vc.dtype == torch.float8_e4m3fn else vc.float()
    out = torch.zeros_like(q, dtype=torch.float32)
    for r in range(rows):
        req = int(t2r[r])
        if req < 0:
            continue
        p = int(seqlens[req]) - (int(qsl[req + 1]) - int(qsl[req])) + (r - int(qsl[req]))
        sel = idx[r]
        sel = sel[(sel >= 0) & (sel <= p)].long()
        if sel.numel() == 0:
            continue
        blk = bt[req][(sel // page).long()]
        tok = (sel % page).long()
        k = kc[blk, tok]                                        # [n, hkv, HD] (advanced indexing, per token)
        v = vc[blk, tok]
        for h in range(hq):
            s = (q[r, h].float() @ k[:, h // gqa].T) * (1.0 / HD ** 0.5)
            w = torch.softmax(s, dim=-1)
            out[r, h] = w @ v[:, h // gqa]
    return out.to(torch.bfloat16)


def make_case(rows, hq, hkv, page, ctx, topk, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(rows, hq, HD, generator=g).to(DEV, torch.bfloat16)
    # One shared block pool, reused by every row: overlapping block tables are legal for a numerics test, and the
    # alternative (a private pool per row) would put ~9 GB of cache on the card at rows=2048.
    nb = (ctx + page - 1) // page + 8
    kc = torch.randn(nb, page, hkv, HD, generator=g).to(DEV, torch.bfloat16) * 30.0   # K magnitudes ~O(90)
    vc = torch.randn(nb, page, hkv, HD, generator=g).to(DEV, torch.bfloat16)
    perm = torch.randperm(nb, generator=g)[:nb].to(DEV)
    kc, vc = kc[perm], vc[perm]
    bpp = (ctx + page - 1) // page
    bt = torch.arange(bpp, device=DEV).repeat(rows, 1).int()
    qsl = torch.arange(0, rows + 1, device=DEV, dtype=torch.int32)
    seqlens = torch.full((rows,), ctx, device=DEV, dtype=torch.int32)
    t2r = torch.arange(rows, device=DEV, dtype=torch.int32)
    # every row selects `topk` whole groups (the top-k list is expanded, -1 padded)
    ngroups = ctx // GS
    gi = torch.stack([torch.randperm(ngroups, generator=g)[:topk] for _ in range(rows)])
    tok = (gi.unsqueeze(-1) * GS + torch.arange(GS).view(1, 1, -1)).reshape(rows, -1)
    idx = torch.cat([tok, torch.full((rows, 3), -1, dtype=torch.int32)], dim=1).sort(dim=1).values.to(DEV).int()
    return q, kc, vc, idx, bt.to(DEV), qsl, seqlens, t2r


CASES = [(r, hq, hkv, page, ctx, topk, ns, desc) for (r, hq, hkv, page, ctx, topk, ns, desc) in itertools.chain(
    [(1, 6, 1, 448, 32768, 512, 0, (1.0, 1.0)),
     (1, 6, 1, 896, 32768, 512, 0, (1.0, 1.0)),
     (3, 6, 1, 896, 65536, 512, 0, (0.5, 0.25)),
     (8, 12, 2, 896, 32768, 512, 0, (1.0, 1.0)),
     (16, 6, 1, 896, 8192, 128, 1, (1.0, 1.0)),
     (64, 6, 1, 448, 8192, 128, 1, (0.5, 1.0)),
     (256, 6, 1, 896, 8192, 128, 1, (1.0, 1.0)),
     (2048, 6, 1, 896, 8192, 256, 1, (1.0, 1.0))])]

print(f"{'rows':>5} {'GQA':>4} {'page':>5} {'nsplit':>6} {'desc':>11} {'bf16 rel':>10} {'fp8 rel':>10} "
      f"{'quant rel':>10} {'fp8/bf16 ms':>12}")
bad = 0
for rows, hq, hkv, page, ctx, topk, ns, desc in CASES:
    q, kc, vc, idx, bt, qsl, seqlens, t2r = make_case(rows, hq, hkv, page, ctx, topk)
    kd, vd = desc
    o_b = torch.zeros_like(q)
    sparse_attention(q, kc, vc, idx, bt, t2r, qsl, seqlens, o_b, nsplit=ns or None, descale=(1.0, 1.0))
    kc8, vc8 = quant(kc, kd), quant(vc, vd)
    o_f = torch.zeros_like(q)
    sparse_attention(q, kc8, vc8, idx, bt, t2r, qsl, seqlens, o_f, nsplit=ns or None, descale=desc)

    nrows = min(rows, 256)                                      # the torch reference is a Python loop: spot-check
    ref_b = reference(q[:nrows], kc, vc, idx[:nrows], bt[:nrows], t2r[:nrows], qsl, seqlens, page, (1.0, 1.0))
    ref_f = reference(q[:nrows], kc8, vc8, idx[:nrows], bt[:nrows], t2r[:nrows], qsl, seqlens, page, desc)

    def rel(a, b):
        d = (a.float() - b.float()).abs().max().item()
        return d / max(b.float().abs().max().item(), 1e-6)

    r_bf, r_f8 = rel(o_b[:nrows], ref_b), rel(o_f[:nrows], ref_f)
    q_rel = rel(ref_f, ref_b)                                   # what e4m3 itself costs, kernel aside
    def gpu_ms(k_c, v_c, dsc):
        for _ in range(2):
            sparse_attention(q, k_c, v_c, idx, bt, t2r, qsl, seqlens, torch.zeros_like(q), nsplit=ns or None,
                             descale=dsc)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(10):
            sparse_attention(q, k_c, v_c, idx, bt, t2r, qsl, seqlens, torch.zeros_like(q), nsplit=ns or None,
                             descale=dsc)
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / 10
    mb, mf = gpu_ms(kc, vc, (1.0, 1.0)), gpu_ms(kc8, vc8, desc)
    flag = ""
    if r_f8 > max(2e-2, 3 * r_bf) or r_bf > 2e-2:
        flag = "  <-- fp8 read path WRONG"
        bad += 1
    print(f"{rows:>5} {hq // hkv:>4} {page:>5} {ns or 'auto':>6} {str(desc):>11} {r_bf:>10.2e} {r_f8:>10.2e} "
          f"{q_rel:>10.2e} {mf / mb:>12.3f}{flag}")
print("FAIL" if bad else "PASS (fp8 kernel tracks its own reference as tightly as bf16 does)")
sys.exit(1 if bad else 0)
