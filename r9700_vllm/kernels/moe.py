"""Python bindings for libr9k's grouped MXFP4 x FP8 MoE GEMM (gfx1201).

Weights come from a compressed-tensors ``mxfp4-pack-quantized`` checkpoint: per expert
``weight_packed [N, K/2] uint8`` (element 2i in the low nibble, 2i+1 in the high nibble, e2m1 codes) and
``weight_scale [N, K/32] uint8`` (E8M0). ``prepare_mxfp4_weights`` converts them once at load to the layout
the kernel reads; see kernels/r9k_moe_mxfp4a8.hip for the format.
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass

import torch

from ..utils import prof

GROUP = 32
MOE_BLOCK = 16

_LIB = None


def lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        path = os.environ.get("R9K_LIB") or os.path.join(os.path.dirname(__file__), "libr9k.so")
        L = ctypes.CDLL(path)
        L.r9k_moe_mxfp4a8.restype = ctypes.c_int
        L.r9k_moe_mxfp4a8.argtypes = [ctypes.c_long] * 12 + [ctypes.c_int] * 9 + [ctypes.c_long]
        if hasattr(L, "r9k_moe_mxfp4a8_f"):
            L.r9k_moe_mxfp4a8_f.restype = ctypes.c_int
            L.r9k_moe_mxfp4a8_f.argtypes = [ctypes.c_int] + [ctypes.c_long] * 12 + [ctypes.c_int] * 9 + [ctypes.c_long]
        L.r9k_moe_nvfp4a8.restype = ctypes.c_int
        L.r9k_moe_nvfp4a8.argtypes = [ctypes.c_long] * 12 + [ctypes.c_int] * 9 + [ctypes.c_long]
        L.r9k_moe_4bit_prefill.restype = ctypes.c_int
        L.r9k_moe_4bit_prefill.argtypes = [ctypes.c_int] + [ctypes.c_long] * 12 + [ctypes.c_int] * 6 + [ctypes.c_long]
        L.r9k_moe_prefill_bm.restype = ctypes.c_int
        L.r9k_moe_prefill_bm.argtypes = [ctypes.c_int]
        L.r9k_moe_prefill_bk.restype = ctypes.c_int
        L.r9k_moe_prefill_bk.argtypes = [ctypes.c_int]
        L.r9k_quant_rows_fp8.restype = ctypes.c_int
        L.r9k_quant_rows_fp8.argtypes = [ctypes.c_long] * 3 + [ctypes.c_int] * 3 + [ctypes.c_long]
        L.r9k_silu_mul_quant_fp8.restype = ctypes.c_int
        L.r9k_silu_mul_quant_fp8.argtypes = [ctypes.c_long] * 3 + [ctypes.c_int] * 2 + [ctypes.c_long]
        L.r9k_quant_rows_fp8_gate.restype = ctypes.c_int
        L.r9k_quant_rows_fp8_gate.argtypes = [ctypes.c_long] * 5 + [ctypes.c_int] * 4 + [ctypes.c_long]
        assert L.r9k_moe_block() == MOE_BLOCK
        if hasattr(L, "r9k_fold_supported"):          # older libr9k.so builds have no folded kernels
            L.r9k_fold_supported.restype = ctypes.c_int
        if hasattr(L, "r9k_moe_4bit_prefill_at"):     # ... or no A-tiled kernels
            L.r9k_moe_4bit_prefill_at.restype = ctypes.c_int
            L.r9k_moe_4bit_prefill_at.argtypes = [ctypes.c_int] + [ctypes.c_long] * 12 + [ctypes.c_int] * 7 + [ctypes.c_long]
            L.r9k_moe_atiled_bm.restype = ctypes.c_int
            L.r9k_moe_atiled_bm.argtypes = [ctypes.c_int]
            L.r9k_quant_rows_fp8_tiled.restype = ctypes.c_int
            L.r9k_quant_rows_fp8_tiled.argtypes = [ctypes.c_long] * 3 + [ctypes.c_int] * 3 + [ctypes.c_long]
        _LIB = L
    return _LIB


def fold_supported() -> bool:
    L = lib()
    return hasattr(L, "r9k_fold_supported") and bool(L.r9k_fold_supported())


def atiled_supported() -> bool:
    return hasattr(lib(), "r9k_moe_4bit_prefill_at")


def _stream() -> int:
    return torch.cuda.current_stream().cuda_stream


@dataclass
class Mxfp4Experts:
    wq: torch.Tensor    # [E, N/16 * K/16 * 32] int32, fragment order
    wsr: torch.Tensor   # [E, K/32 + 1, N] uint8: E8M0 exponents K-block-major, last row = per-row reference
    N: int
    K: int
    fold: bool = False  # folded-exponent kernels (fold_decide() per tensor at weight prep; layout is unchanged)

    @property
    def estride(self) -> int:
        return (self.K // GROUP + 1) * self.N


@dataclass
class Nvfp4Experts:
    """Native NVFP4 (e2m1 + e4m3 scale per 16 + fp32 global per row), same fragment-order codes as MXFP4."""
    wq: torch.Tensor    # [E, N/16 * K/16 * 32] int32, fragment order
    ws: torch.Tensor    # [E, K/16, N] uint8 (e4m3 bits), K-step-major
    wg: torch.Tensor    # [E, N] fp32 per-row global multiplier (1 / CT's stored divisor)
    N: int
    K: int


def prepare_nvfp4_weights(packed: torch.Tensor, scale16: torch.Tensor, row_mult: torch.Tensor) -> Nvfp4Experts:
    """packed [E, N, K/2] u8, scale16 [E, N, K/16] e4m3, row_mult [E, N] fp32 -> Nvfp4Experts."""
    E, N, Kh = packed.shape
    K = Kh * 2
    assert N % 16 == 0 and K % 16 == 0, (N, K)
    assert scale16.shape == (E, N, K // 16), (scale16.shape, (E, N, K // 16))
    ws = scale16.view(torch.uint8).transpose(1, 2).contiguous()
    return Nvfp4Experts(permute_fragments(packed.view(torch.uint8)), ws,
                        row_mult.float().reshape(E, N).contiguous(), N, K)


@prof.slowcall
def permute_fragments(packed: torch.Tensor) -> torch.Tensor:
    """[E, N, K/2] uint8 checkpoint order -> [E, N/16*K/16*32] int32 fragment order.

    Slot l of tile (nt, ks) holds W[nt*16 + (l & 15)][ks*16 + 8*(l >> 4) .. +8] (4 packed bytes).

    That destination is not a choice: `v_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` requires lane l of a wave to hold
    row (l & 15) and k-bytes 8*(l >> 4)..+8 of the B fragment. Permuting the weight into exactly that order on
    the host is the only way a wave can then read its fragment as 32 contiguous dwords, so any correct
    implementation for this builtin produces the same bytes. (An earlier docstring described this as "identical
    to libr4d mxfp4_layout.permute_w" -- true, and true of anyone else's too, because the hardware fixes it.
    See notes/independence.md.)
    """
    E, N, Kh = packed.shape
    K = Kh * 2
    nt, ks = N // 16, K // 16
    w = packed.reshape(E, nt, 16, ks, 2, 4)            # [E, nt, r, ks, h, 4 bytes]
    w = w.permute(0, 1, 3, 4, 2, 5).contiguous()       # [E, nt, ks, h, r, 4] -> lane = h*16 + r
    return w.reshape(E, -1).view(torch.int32)


def prepare_mxfp4_weights(packed: torch.Tensor, scale: torch.Tensor) -> Mxfp4Experts:
    E, N, Kh = packed.shape
    K = Kh * 2
    assert N % 16 == 0 and K % GROUP == 0, (N, K)
    assert scale.shape == (E, N, K // GROUP), (scale.shape, (E, N, K // GROUP))
    return Mxfp4Experts(permute_fragments(packed.view(torch.uint8)), pack_scales(scale), N, K)


@prof.slowcall
def pack_scales(scale: torch.Tensor) -> torch.Tensor:
    """[E, N, K/32] E8M0 -> [E, K/32 + 1, N]: transposed block exponents plus the per-row max as the last row."""
    scale = scale.view(torch.uint8)
    E, N, nb = scale.shape
    out = torch.empty((E, nb + 1, N), dtype=torch.uint8, device=scale.device)
    out[:, :nb].copy_(scale.transpose(1, 2))
    out[:, nb] = scale.max(dim=2).values
    return out


# ------------------------------------------------------------------------------------ folded-exponent MXFP4
# The kernels can fold each block's E8M0 into the e2m1 -> e4m3 unpack (kMag table: value * 2^-d, d = row reference
# exponent - block exponent) and apply only 2^(ref - 127) per row in the epilogue: no per-group fp32 scaling in the
# inner loop. e4m3 reaches 2^-9 below the row's largest block, so the fold is
#   exact     for d <= FOLD_EXACT (8): every e2m1 value (0.5 .. 6) is representable, subnormals included;
#   rounded   for 9 <= d <= FOLD_REACH (12): mantissa bits fall off the bottom of the subnormal range (kMag rounds);
#   flushed   for d >= 13: the block reads as zero (its weights are < 2^-12 of the row's largest block).
# A flushed weight errs by at most 2^-10 * 2^ref, i.e. < 1/6000 of the row's largest weight; the damage is bounded
# per weight but a tensor whose rows are dominated by such blocks would silently lose them, so folding is decided
# per weight tensor from the d distribution (fold_stats) against explicit thresholds (fold_ok), logged at load.
# R9K_FOLD: 0 (default) never fold; 1 fold when the tensor passes the guard; force fold regardless (measurements).
FOLD_EXACT, FOLD_REACH = 8, 12
FOLD_MODE = os.environ.get("R9K_FOLD", "0").lower()
FOLD_MAX_INEXACT = float(os.environ.get("R9K_FOLD_MAX_INEXACT", "0.05"))   # max fraction of blocks with d > 8
FOLD_MAX_FLUSH = float(os.environ.get("R9K_FOLD_MAX_FLUSH", "0.001"))      # max fraction of blocks with d > 12
# Short-K tensors are not FMA-bound (few slabs, weight-stream / launch bound) and the fold's per-slab scale load +
# lane permutes cost more than the FMAs they remove: measured on the Flash-Next routed down GEMM (K=320, cfg 11:
# 1010 -> 1048 us @2048 tokens, 1659 -> 1696 @4096), while every K >= 2560 shape gained 6-16%.
FOLD_MIN_K = int(os.environ.get("R9K_FOLD_MIN_K", "1024"))
# The guard's walk is a load cost, not a free one: with R9K_LOAD_PROF=1 the timer charges `kernels fold_stats`
# 8.1-8.5 s a rank on Flash-Next (196 tensors, 965 M scale blocks a rank = 965 MiB of uint8, and the int16
# difference makes it some GiB more on the device). Much of that is waste: the K < FOLD_MIN_K class -- 98 of the 196
# tensors, a third of the blocks -- is refused by shape and needs no walk at all, and a question as coarse as "is
# p_flush under 0.1%" does not need 13.1 M blocks when the answer is nowhere near the line (every Flash-Next tensor
# measured: max_d <= 6 against FOLD_EXACT = 8, p_flush exactly 0 against 1e-3). So the guard samples at most
# R9K_FOLD_SAMPLE blocks per tensor -- a uniform random set of blocks, deterministic from a fixed seed so all four
# ranks sample the same blocks and cannot disagree -- and re-walks a tensor exactly when the sample comes within
# R9K_FOLD_MARGIN of a threshold or sees a single flushed block, so a borderline call is never made on a sample.
# 0 turns sampling off (the old exact walk).
FOLD_SAMPLE = int(os.environ.get("R9K_FOLD_SAMPLE", str(1 << 20)))
FOLD_MARGIN = float(os.environ.get("R9K_FOLD_MARGIN", "3"))
FOLD_SAMPLE_SEED = 0x5EED


@dataclass
class FoldStats:
    blocks: int
    hist: list            # hist[d] = blocks with d = ref - block exponent (d clamped to 0..16; 16 = ">15")
    max_d: int
    K: int = 0            # the GEMM's K (folding is not worth it below FOLD_MIN_K)
    total: int = 0        # blocks in the tensor; blocks < total means the histogram is a sample
    exact: bool = True    # False when a sample decided it and no exact re-walk followed

    @property
    def p_inexact(self) -> float:
        return sum(self.hist[FOLD_EXACT + 1:]) / max(1, self.blocks)

    @property
    def p_flush(self) -> float:
        return sum(self.hist[FOLD_REACH + 1:]) / max(1, self.blocks)

    @property
    def mean_d(self) -> float:
        return sum(d * n for d, n in enumerate(self.hist)) / max(1, self.blocks)

    def __str__(self):
        return (f"K={self.K} blocks={self.blocks}"
                + (f"/{self.total} sampled" if self.total > self.blocks else "")
                + f" mean_d={self.mean_d:.2f} max_d={self.max_d} "
                f"inexact(d>{FOLD_EXACT})={100 * self.p_inexact:.3f}% flushed(d>{FOLD_REACH})={100 * self.p_flush:.4f}%")


def warm_post_load(device: torch.device | None = None) -> float:
    """Run each post-load helper once, on a coin-sized dummy, before the real pass is timed. Returns the seconds it
    took.

    Why: in both tailed instances the whole window sits in one visit, and it is the same visit both times --
    `language_model.model.layers.0.mlp.shared_expert.gate_up_proj`, which is the **first tensor the fold guard ever
    runs in the process** (43.57 s / 87.1 CPU-s in tail48, 69.30 s / 141.4 CPU-s in tail57). Inside it `_fold_hist`
    costs 4.8-9.9 s and 10-20 CPU-seconds on a 26 KB tensor, the thread is in state R with `syscall running`, and
    nothing is swapped or faulted in worth naming. That is the shape of first-call work on the device path (first
    kernel image, first allocation of that size, first queue wakeup) landing inside the window we measure. Warming
    it outside the window is the experiment: if the tail goes away the term was first-call cost; if it does not,
    it was not, and that is just as useful a fact."""
    import time
    from .fp8 import quantize_rows_fp8
    dev = device or torch.device("cuda", torch.cuda.current_device())
    t0 = time.perf_counter()
    N, K = 16, 2560                                        # the shape class of the first guarded tensor
    permute_fragments(torch.zeros((1, N, K // 2), dtype=torch.uint8, device=dev))
    wsr = pack_scales(torch.zeros((1, N, K // 32), dtype=torch.uint8, device=dev))
    _fold_hist(wsr, 64 << 20, (None, None))                # first zeros/bincount/.cpu() on this device
    quantize_rows_fp8(torch.zeros((16, 16), dtype=torch.bfloat16, device=dev))
    return time.perf_counter() - t0


def fold_k(wsr: torch.Tensor) -> int:
    """The GEMM's K that a packed [E, K/32 + 1, N] scale tensor belongs to -- shape only, and shape alone is all the
    short-K refusal needs: it is the one guard reason that costs nothing to answer."""
    return (wsr.shape[1] - 1) * GROUP


@prof.slowcall
def _fold_hist(wsr: torch.Tensor, chunk: int, idx: tuple) -> list:
    """The 17-bin d histogram over the sampled blocks `idx` = (flat block index, its column); (None, None) = every
    block. Accumulated on the tensor's own device, so the host synchronises once per tensor, not once per chunk."""
    E, nb1, N = wsr.shape
    flat, col = idx
    hist = torch.zeros(17, dtype=torch.int64, device=wsr.device)
    if flat is None:
        step = max(1, chunk // (nb1 * N))
        for e0 in range(0, E, step):
            w = wsr[e0:e0 + step]
            d = (w[:, -1:, :].to(torch.int16) - w[:, :-1, :].to(torch.int16)).clamp_(0, 16)
            hist += torch.bincount(d.reshape(-1), minlength=17)
    else:
        flat, col = flat.to(wsr.device), col.to(wsr.device)
        step = max(1, chunk // max(2, 2 * flat.numel()))        # two gathers and an int16 difference per block
        for e0 in range(0, E, step):
            w = wsr[e0:e0 + step]
            b = w.reshape(w.shape[0], -1).index_select(1, flat)         # the sampled blocks
            r = w[:, -1, :].index_select(1, col)                        # the reference exponent of their rows
            hist += torch.bincount((r.to(torch.int16) - b.to(torch.int16)).clamp_(0, 16).reshape(-1), minlength=17)
    return hist.cpu().tolist()


def _fold_sample_idx(wsr: torch.Tensor, budget: int):
    """Which blocks to walk when the tensor is bigger than the budget: a uniform random sample of one expert's
    (block, column) plane, the same set for every expert, from a fixed seed.

    Uniform over blocks rather than whole columns because the block is what the guard counts: 1 M blocks drawn as 8%
    of the columns can walk straight past a channel that is entirely bad, while 1 M blocks spread over the plane see
    a 0.1%-of-blocks failure ~always. Fixed seed because four ranks sample the same tensor and must not reach
    different decisions."""
    E, nb1, N = wsr.shape
    plane = (nb1 - 1) * N
    nsel = max(1, min(plane, budget // max(1, E)))
    if nsel >= plane:
        return None, None
    gen = torch.Generator(device="cpu").manual_seed(FOLD_SAMPLE_SEED)
    flat = torch.randperm(plane, generator=gen)[:nsel].sort().values.to(torch.int64)
    return flat, torch.remainder(flat, N)


def _fold_near(st: FoldStats) -> bool:
    """True when a sampled guard is close enough to a threshold that only the exact walk may decide it. One flushed
    block in the sample also escalates: with a 0.1% threshold and a million-block sample, "the sample saw one" says
    nothing, while "the sample saw none" would be read as "far under 0.1%"."""
    if not FOLD_MARGIN or st.exact:
        return False
    if sum(st.hist[FOLD_REACH + 1:]):
        return True
    for p, thr in ((st.p_inexact, FOLD_MAX_INEXACT), (st.p_flush, FOLD_MAX_FLUSH)):
        if thr / FOLD_MARGIN < p <= thr * FOLD_MARGIN:
            return True
    return False


@prof.timed("kernels fold_stats")
def fold_stats(wsr: torch.Tensor, chunk: int = 64 << 20, sample: int | None = None) -> FoldStats:
    """Distribution of d = row reference exponent - block exponent over a packed [E, K/32 + 1, N] scale tensor
    (pack_scales layout; the last row is the reference). Chunked over experts, any device. Samples at most
    R9K_FOLD_SAMPLE blocks (``sample`` overrides; 0 walks every block) and re-walks exactly when the sample lands
    near a threshold, so a borderline tensor is never decided on a sample."""
    budget = FOLD_SAMPLE if sample is None else sample
    E, nb1, N = wsr.shape
    total = E * (nb1 - 1) * N
    idx = _fold_sample_idx(wsr, budget) if budget and total > budget else (None, None)
    h = _fold_hist(wsr, chunk, idx)
    nwalked = total if idx[0] is None else E * idx[0].numel()
    st = FoldStats(nwalked, h, max(i for i, n in enumerate(h) if n) if any(h) else 0, (nb1 - 1) * GROUP, total,
                   exact=idx[0] is None)
    if _fold_near(st):
        h = _fold_hist(wsr, chunk, (None, None))
        st = FoldStats(total, h, max(i for i, n in enumerate(h) if n) if any(h) else 0, (nb1 - 1) * GROUP, total)
    return st


def fold_ok(st: FoldStats) -> bool:
    return fold_why(st) == ""


def fold_why(st: FoldStats) -> str:
    """'' when the tensor may fold, else the reason it stays on the exact kernels."""
    if st.p_inexact > FOLD_MAX_INEXACT or st.p_flush > FOLD_MAX_FLUSH:
        return "exceeds R9K_FOLD_MAX_INEXACT/FLUSH"
    if st.K < FOLD_MIN_K:
        return f"K={st.K} < R9K_FOLD_MIN_K={FOLD_MIN_K}: no gain on short K"
    return ""


_FOLD_LOG: list[tuple[str, bool, FoldStats]] = []


@prof.timed("kernels fold_decide")
def fold_decide(wsr: torch.Tensor, name: str = "", log=None) -> bool:
    """Whether the MXFP4 tensor `wsr` (pack_scales layout) is served with the folded-exponent kernels: R9K_FOLD off
    -> False without looking; on -> fold_ok(fold_stats); force -> True. Every decision is logged (with the stats and
    running totals) so a checkpoint that folds badly is visible at load rather than silently degraded."""
    if FOLD_MODE in ("", "0", "off", "no"):
        return False
    if not fold_supported():
        return False
    k = fold_k(wsr)
    if k < FOLD_MIN_K and FOLD_MODE != "force":
        st = FoldStats(0, [0] * 17, 0, k)                    # the cheap refusal: no bytes walked
        why = f"K={k} < R9K_FOLD_MIN_K={FOLD_MIN_K}: no gain on short K (scales not walked)"
        fold = False
    else:
        st = fold_stats(wsr)
        why = fold_why(st)
        fold = not why or FOLD_MODE == "force"
    _FOLD_LOG.append((name, fold, st))
    n_on = sum(1 for _, f, _ in _FOLD_LOG if f)
    msg = (f"r9700: fold {'ON ' if fold else 'OFF'} {name or 'mxfp4'}: {st}"
           f"{'' if not why else ' (' + why + (', forced)' if fold else ')')}"
           f" [tensors so far: {n_on} folded / {len(_FOLD_LOG) - n_on} exact]")
    (log or _default_log)(msg)
    return fold


def fold_summary() -> str:
    on = [x for x in _FOLD_LOG if x[1]]
    off = [x for x in _FOLD_LOG if not x[1]]
    worst = max(_FOLD_LOG, key=lambda x: x[2].p_flush, default=None)
    return (f"fold: {len(on)} tensors folded, {len(off)} kept exact"
            + (f"; worst flush {100 * worst[2].p_flush:.4f}% ({worst[0]})" if worst else ""))


def _default_log(msg: str) -> None:
    try:
        from vllm.logger import init_logger
        init_logger("vllm." + __name__).info(msg)
    except Exception:
        print(msg)


def quant_rows_fp8(x: torch.Tensor, tiled: bool = False):
    """bf16 [M, K] -> (e4m3fn [M, K], fp32 [M]) with a per-row dynamic scale.

    tiled: the fp8 bytes in the WMMA-fragment-tiled layout the A-tiled prefill kernel reads (16*ceil(M/16) rows;
    fragment (row//16, k//16) = 256 contiguous bytes, see r9k_quant_rows_fp8_tiled); returned as a [16*Mt, K]
    e4m3fn tensor whose bytes are NOT row-major (only moe_gemm(..., a_tiled=True) may consume it). Same values and
    scales as the row-major quant, only the store addresses differ."""
    assert x.dtype == torch.bfloat16 and x.dim() == 2 and x.stride(1) == 1
    M, K = x.shape
    s = torch.empty((M,), dtype=torch.float32, device=x.device)
    if tiled:
        mt = (M + 15) // 16
        q = torch.empty((mt * 16, K), dtype=torch.float8_e4m3fn, device=x.device)
        rc = lib().r9k_quant_rows_fp8_tiled(x.data_ptr(), q.data_ptr(), s.data_ptr(), M, K, x.stride(0), _stream())
        if rc:
            raise RuntimeError(f"r9k_quant_rows_fp8_tiled failed ({rc}) M={M} K={K}")
        return q, s
    q = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x.device)
    rc = lib().r9k_quant_rows_fp8(x.data_ptr(), q.data_ptr(), s.data_ptr(), M, K, x.stride(0), _stream())
    if rc:
        raise RuntimeError(f"r9k_quant_rows_fp8 failed ({rc})")
    return q, s


def quant_rows_fp8_gate(x: torch.Tensor, wg: torch.Tensor, tiled: bool = False):
    """quant_rows_fp8 plus the shared expert's gate in the same launch: (q, s, g) with g [M] fp32 =
    bf16(sigmoid(bf16(x @ wg))) -- stock's F.sigmoid(expert_gate(x)) values -- for the down GEMM's row fold."""
    assert x.dtype == torch.bfloat16 and x.dim() == 2 and x.stride(1) == 1
    assert wg.dtype == torch.bfloat16 and wg.is_contiguous() and wg.numel() == x.shape[1]
    M, K = x.shape
    s = torch.empty((M,), dtype=torch.float32, device=x.device)
    g = torch.empty((M,), dtype=torch.float32, device=x.device)
    rows = (M + 15) // 16 * 16 if tiled else M
    q = torch.empty((rows, K), dtype=torch.float8_e4m3fn, device=x.device)
    rc = lib().r9k_quant_rows_fp8_gate(x.data_ptr(), q.data_ptr(), s.data_ptr(), wg.data_ptr(), g.data_ptr(), M, K,
                                       x.stride(0), 1 if tiled else 0, _stream())
    if rc:
        raise RuntimeError(f"r9k_quant_rows_fp8_gate failed ({rc}) M={M} K={K} tiled={tiled}")
    return q, s, g


def tile_fp8_ref(q: torch.Tensor) -> torch.Tensor:
    """Pure-torch re-layout of a row-major e4m3 [M, K] into the fragment-tiled [16*ceil(M/16), K] buffer (tests):
    fragment (mt, ks) -> 32 lanes x 8 bytes, lane l = row mt*16 + l%16, bytes ks*16 + 8*(l//16) .. +8."""
    M, K = q.shape
    mt = (M + 15) // 16
    b = torch.zeros((mt * 16, K), dtype=torch.uint8, device=q.device)
    b[:M] = q.view(torch.uint8)
    b = b.reshape(mt, 16, K // 16, 2, 8).permute(0, 2, 3, 1, 4)     # [mt, ks, half, row, 8] -> lane = half*16 + row
    return b.contiguous().reshape(mt * 16, K).view(torch.float8_e4m3fn)


def silu_mul_quant_fp8(gu: torch.Tensor):
    """bf16 [M, 2I] (gate | up) -> (e4m3fn [M, I], fp32 [M]): silu(gate) * up with a per-row dynamic scale."""
    assert gu.dtype == torch.bfloat16 and gu.is_contiguous() and gu.shape[1] % 2 == 0
    M, I2 = gu.shape
    q = torch.empty((M, I2 // 2), dtype=torch.float8_e4m3fn, device=gu.device)
    s = torch.empty((M,), dtype=torch.float32, device=gu.device)
    rc = lib().r9k_silu_mul_quant_fp8(gu.data_ptr(), q.data_ptr(), s.data_ptr(), M, I2 // 2, _stream())
    if rc:
        raise RuntimeError(f"r9k_silu_mul_quant_fp8 failed ({rc})")
    return q, s


# Prefill (LDS-tiled, large-M) kernel tile table: cfg -> (rows per routing block BM, columns BN); mirrors
# kPfCfgs in r9k_moe_mxfp4a8.hip (the kernel is the source of truth: prefill_block() asks it).
PREFILL_TILES = {0: (256, 64), 1: (256, 128), 2: (128, 128), 3: (128, 64), 4: (128, 64), 5: (256, 32),
                 6: (64, 64), 7: (256, 128), 8: (256, 128), 9: (256, 128), 10: (128, 128), 11: (64, 64),
                 12: (256, 128), 13: (128, 128), 14: (64, 64), 15: (64, 128), 16: (256, 128), 17: (64, 128),
                 18: (64, 64), 19: (64, 128)}
PREFILL_MIN_M = int(os.environ.get("R9K_PREFILL_MIN_M", "128"))   # dense calls at M >= this use the prefill path
# Untuned shapes: 256x128 BK=64 double-buffered (cfg 12) from M >= 512, 128x128 double-buffered (cfg 10) below (grid fill).
PREFILL_DEFAULT = int(os.environ.get("R9K_PREFILL_CFG", "12"))
PREFILL_DEFAULT_SMALL = int(os.environ.get("R9K_PREFILL_CFG_SMALL", "10"))


def prefill_block(cfg: int) -> int:
    """Rows per routing block (moe_align_block_size block) of prefill tile `cfg`."""
    bm = lib().r9k_moe_prefill_bm(cfg)
    if bm <= 0:
        raise ValueError(f"unknown prefill cfg {cfg}")
    return bm


def prefill_bk(cfg: int) -> int:
    """K slab of prefill tile `cfg`; the kernel needs K % BK == 0 (cfgs 8-11, 17 are BK=32; the rest BK=64)."""
    bk = lib().r9k_moe_prefill_bk(cfg)
    if bk <= 0:
        raise ValueError(f"unknown prefill cfg {cfg}")
    return bk


def is_prefill_cfg(cfg) -> bool:
    return bool(cfg) and cfg[0] == "P"


# A-tiled prefill kernel (r9k_moe_4bit_prefill_at: fragment-tiled activations straight from global, weights through
# LDS, folded or with fp32-per-group scaling like the LDS-A kernel). cfg -> (BM, BN, BK); mirrors kAtCfgs in the
# kernel (atiled_block() asks it). Selected by pick_cfg as ("A", cfg) for dense MXFP4 GEMMs at prefill M, folded or
# not, when R9K_ATILED=1 and K % BK == 0; the caller then quantizes with quant_rows_fp8(x, tiled=True) and runs
# moe_gemm(..., a_tiled=True). Everything else keeps the LDS-A tiles.
ATILED_TILES = {0: (256, 64, 64), 1: (256, 128, 64), 2: (256, 64, 128), 3: (256, 128, 64), 4: (128, 128, 64),
                5: (128, 64, 64), 6: (256, 128, 64), 7: (256, 64, 64), 8: (128, 128, 64)}
ATILED = os.environ.get("R9K_ATILED", "1") == "1"
# Untuned shapes, by M band (measured 2026-09-20 on the five served 27B shapes, steady state, vs the folded LDS-A
# tiles: 0.83-0.92x at M >= 2048 on the 256x128 8-wave tile, 0.74-0.83x at 512 on 256x64, 0.74-0.86x at 128-256
# on 128x64; the 256-row tiles lose at M=128 where half the tile is padding).
ATILED_DEFAULT = int(os.environ.get("R9K_ATILED_CFG", "1"))              # M >= 2048: 256 x 128, 8 waves
ATILED_DEFAULT_MID = int(os.environ.get("R9K_ATILED_CFG_MID", "0"))      # 512 <= M < 2048: 256 x 64
ATILED_DEFAULT_SMALL = int(os.environ.get("R9K_ATILED_CFG_SMALL", "5"))  # M < 512: 128 x 64
# The tiled quantizer has no scalar fallback (r9k_quant_rows_fp8_tiled returns -3 past the vectorized kernel's
# reach), so a K it cannot serve must not pick this path at all: RQ_THREADS * RQ_VEC * 8 in the kernel.
ATILED_MAX_K = 20480


def atiled_block(cfg: int) -> int:
    bm = lib().r9k_moe_atiled_bm(cfg)
    if bm <= 0:
        raise ValueError(f"unknown A-tiled cfg {cfg}")
    return bm


def is_atiled_cfg(cfg) -> bool:
    return bool(cfg) and cfg[0] == "A"


def pick_atiled(N: int, K: int, M: int, fold: bool = True) -> tuple[str, int] | None:
    """("A", cfg) for a dense MXFP4 GEMM at M >= PREFILL_MIN_M when the tiled path is on and legal. fold only says
    which kernel variant the caller will run: the tile comes from the tuned "mxfp4_at" rows either way (measured
    2026-09-22: the fold-tuned tile is also the best non-folded tile on every served shape and band, within 3%),
    else the M-band defaults."""
    if not ATILED or M < PREFILL_MIN_M or K % 64 or K > ATILED_MAX_K or not atiled_supported():
        return None
    from .tuned import lookup
    t = lookup("mxfp4_at", N, K, M)
    if t and is_atiled_cfg(t) and K % ATILED_TILES[t[1]][2] == 0:
        return ("A", t[1])
    cfg = ATILED_DEFAULT if M >= 2048 else (ATILED_DEFAULT_MID if M >= 512 else ATILED_DEFAULT_SMALL)
    return ("A", cfg) if K % ATILED_TILES[cfg][2] == 0 else None


def moe_gemm(a_q: torch.Tensor, a_s: torch.Tensor, w: Mxfp4Experts, out: torch.Tensor,
             sorted_ids: torch.Tensor, expert_ids: torch.Tensor, ntpp: torch.Tensor, numel: int,
             a_row_div: int, topk_w: torch.Tensor | None = None, WV: int = 2, SK: int = 4, NPW: int = 2,
             num_experts: int | None = None, MT: int = 1, ldsa: bool = False, prefill: int | None = None,
             a_tiled: bool = False, fuse: int = 0):
    """out[r, :] = dequant(a_q[r // a_row_div]) @ W[expert(r)]^T (* topk_w[r]) for every routed flat row r.

    sorted_ids/expert_ids must come from moe_align_block_size(block_size=16*MT). MT M-tiles per workgroup share
    each weight fragment (use MT>1 when experts see many rows: prefill / wide batches). ldsa: stage the A tile
    through LDS (MT > 1 only; a tuned per-config choice, passed to the kernel as MT | 8).
    prefill: tile cfg of the LDS-tiled large-M kernel (r9k_moe_4bit_prefill) instead; the tables must then come
    from moe_align_block_size(block_size=prefill_block(cfg)) and WV/SK/NPW/MT/ldsa are ignored.
    The grid covers at most ceil(numel/BLK) + min(numel, E) blocks -- the most moe_align_block_size can fill
    with numel routed rows over E experts -- instead of the buffer's worst-case capacity; host-known, so the
    launch stays cudagraph-safe.
    a_tiled: a_q is the fragment-tiled buffer of quant_rows_fp8(tiled=True) in sorted-position order (dense:
    identity tables) and `prefill` is an A-tiled cfg (atiled_block(cfg) rows per block); MXFP4 only, w.fold picks
    the folded or the fp32-per-group variant."""
    E = num_experts if num_experts is not None else w.wq.shape[0]
    if fuse:
        # fuse 1: gate_up with silu(g)*u in the epilogue -> out fp32 [numel, N/2]; fuse 2: down with the fp32 A rows
        # quantised on load (a_s unused). Decode direct path (MT == 1), MXFP4 experts (kernels/r9k_moe_mxfp4a8.hip).
        assert isinstance(w, Mxfp4Experts) and MT == 1 and prefill is None and not a_tiled
        assert a_q.shape[1] == w.K and (out.dtype == torch.float32 and out.shape[1] == w.N // 2 if fuse == 1
                                        else out.dtype == torch.bfloat16 and out.shape[1] == w.N)
        if fuse == 2:
            assert a_q.dtype == torch.float32 and a_row_div == 1
        blk = MOE_BLOCK
        max_blocks = min(expert_ids.numel(), (numel + blk - 1) // blk + min(numel, E))
        rc = lib().r9k_moe_mxfp4a8_f(
            fuse, a_q.data_ptr(), a_s.data_ptr() if a_s is not None else 0, w.wq.data_ptr(), w.wsr.data_ptr(),
            w.wsr.data_ptr() + (w.K // GROUP) * w.N, out.data_ptr(),
            sorted_ids.data_ptr(), expert_ids.data_ptr(), ntpp.data_ptr(),
            topk_w.data_ptr() if topk_w is not None else 0,
            w.estride, w.estride, max_blocks, numel, a_row_div, w.K, w.N, WV, SK, NPW, 1 | (16 if w.fold else 0),
            _stream())
        if rc:
            raise RuntimeError(f"r9k_moe_mxfp4a8_f failed ({rc}) fuse={fuse} N={w.N} K={w.K} WV={WV} SK={SK} NPW={NPW}")
        return out
    assert out.dtype == torch.bfloat16 and out.shape[1] == w.N and a_q.shape[1] == w.K
    if a_tiled:
        assert prefill is not None and isinstance(w, Mxfp4Experts), "a_tiled: MXFP4 prefill only"
        blk = atiled_block(prefill)
        max_blocks = min(expert_ids.numel(), (numel + blk - 1) // blk + min(numel, E))
        rc = lib().r9k_moe_4bit_prefill_at(
            1 if w.fold else 0, a_q.data_ptr(), a_s.data_ptr(), w.wq.data_ptr(), w.wsr.data_ptr(),
            w.wsr.data_ptr() + (w.K // GROUP) * w.N, out.data_ptr(),
            sorted_ids.data_ptr(), expert_ids.data_ptr(), ntpp.data_ptr(),
            topk_w.data_ptr() if topk_w is not None else 0,
            w.estride, w.estride, max_blocks, numel, a_row_div, a_q.shape[0] // 16, w.K, w.N, prefill, _stream())
        if rc:
            raise RuntimeError(f"r9k_moe_4bit_prefill_at failed ({rc}) fold={w.fold} N={w.N} K={w.K} cfg={prefill}")
        return out
    if prefill is not None:
        blk = prefill_block(prefill)
        max_blocks = min(expert_ids.numel(), (numel + blk - 1) // blk + min(numel, E))
        if type(w).__name__ == "Fp8Weight":           # kernels/fp8.py: e4m3 fragments + one fp32 scale a column
            rc = lib().r9k_moe_4bit_prefill(
                3, a_q.data_ptr(), a_s.data_ptr(), w.wq.data_ptr(), 0, w.ws.data_ptr(), out.data_ptr(),
                sorted_ids.data_ptr(), expert_ids.data_ptr(), ntpp.data_ptr(),
                topk_w.data_ptr() if topk_w is not None else 0, 0, w.N,
                max_blocks, numel, a_row_div, w.K, w.N, prefill, _stream())
            if rc:
                raise RuntimeError(f"r9k_moe_4bit_prefill (fp8) failed ({rc}) N={w.N} K={w.K} cfg={prefill}")
            return out
        nv = isinstance(w, Nvfp4Experts)
        rc = lib().r9k_moe_4bit_prefill(
            1 if nv else (2 if w.fold else 0), a_q.data_ptr(), a_s.data_ptr(), w.wq.data_ptr(),
            w.ws.data_ptr() if nv else w.wsr.data_ptr(),
            w.wg.data_ptr() if nv else w.wsr.data_ptr() + (w.K // GROUP) * w.N, out.data_ptr(),
            sorted_ids.data_ptr(), expert_ids.data_ptr(), ntpp.data_ptr(),
            topk_w.data_ptr() if topk_w is not None else 0,
            (w.K // 16) * w.N if nv else w.estride, w.N if nv else w.estride,
            max_blocks, numel, a_row_div, w.K, w.N, prefill, _stream())
        if rc:
            raise RuntimeError(f"r9k_moe_4bit_prefill failed ({rc}) nv={nv} N={w.N} K={w.K} cfg={prefill}")
        return out
    blk = MOE_BLOCK * MT
    max_blocks = min(expert_ids.numel(), (numel + blk - 1) // blk + min(numel, E))
    MT = MT | (8 if ldsa and MT > 1 else 0)
    if isinstance(w, Nvfp4Experts):
        rc = lib().r9k_moe_nvfp4a8(
            a_q.data_ptr(), a_s.data_ptr(), w.wq.data_ptr(), w.ws.data_ptr(), w.wg.data_ptr(), out.data_ptr(),
            sorted_ids.data_ptr(), expert_ids.data_ptr(), ntpp.data_ptr(),
            topk_w.data_ptr() if topk_w is not None else 0,
            (w.K // 16) * w.N, w.N, max_blocks, numel, a_row_div, w.K, w.N, WV, SK, NPW, MT, _stream())
        if rc:
            raise RuntimeError(f"r9k_moe_nvfp4a8 failed ({rc}) N={w.N} K={w.K} WV={WV} SK={SK} NPW={NPW} MT={MT}")
        return out
    rc = lib().r9k_moe_mxfp4a8(
        a_q.data_ptr(), a_s.data_ptr(), w.wq.data_ptr(), w.wsr.data_ptr(),
        w.wsr.data_ptr() + (w.K // GROUP) * w.N, out.data_ptr(),
        sorted_ids.data_ptr(), expert_ids.data_ptr(), ntpp.data_ptr(),
        topk_w.data_ptr() if topk_w is not None else 0,
        w.estride, w.estride, max_blocks, numel, a_row_div, w.K, w.N, WV, SK, NPW, MT | (16 if w.fold else 0),
        _stream())
    if rc:
        raise RuntimeError(f"r9k_moe_mxfp4a8 failed ({rc}) N={w.N} K={w.K} WV={WV} SK={SK} NPW={NPW} MT={MT}")
    return out


def pick_cfg(N: int, K: int, group: int = GROUP, M: int | None = None, kind: str | None = None,
             fold: bool = False) -> tuple[int, ...]:
    """A legal (WV, SK, NPW[, MT[, LDSA]]) for any N % 16 == 0, K % group == 0 (32 MXFP4, 16 NVFP4): deep split-K for long K (weight streaming with few
    N tiles per wave), shallow for short K. Mirrors the tuned Flash-Next defaults (2,4,2) @K=2560, (4,2,1) @K=320.
    Large M (dense prefill) returns ("P", cfg): the LDS-tiled prefill kernel (tuned.json rows ["P", cfg], else
    PREFILL_DEFAULT from M >= PREFILL_MIN_M when K % 64 == 0); or ("A", cfg), the A-tiled kernel, for MXFP4
    (folded or not: fold only selects the kernel variant) when R9K_ATILED is on (pick_atiled)."""
    if kind == "mxfp4" and M:
        t = pick_atiled(N, K, M, fold)
        if t:
            return t
    if kind and M:
        from .tuned import lookup
        t = lookup(kind, N, K, M)
        if t:
            return t
        if M >= PREFILL_MIN_M and K % 64 == 0:      # the untuned defaults are BK=64 tiles
            return ("P", PREFILL_DEFAULT if M >= 512 else PREFILL_DEFAULT_SMALL)
    for WV, SK, NPW in ((2, 4, 2), (4, 2, 1), (2, 5, 2), (4, 1, 1)):
        if K % (SK * group) == 0 and (K >= 1024 or SK <= 2):
            return WV, SK, NPW
    return 4, 1, 1


def legal_cfg(cfg: tuple[int, ...], N: int, K: int, group: int = GROUP) -> tuple[int, int, int]:
    """cfg's (WV, SK, NPW) if the kernel accepts it for this shape (K % (SK*group) == 0), else pick_cfg's.
    The fixed decode configs are tuned for Flash-Next at TP=2 (down K=320); TP=4 shards the down GEMM to K=160,
    where SK=2 is rejected (-2) and only SK in {1, 5} fit."""
    WV, SK, NPW = cfg[:3]
    if K % (SK * group) == 0 and WV * SK * 32 <= 1024:
        return WV, SK, NPW
    return pick_cfg(N, K, group)[:3]  # type: ignore[return-value]


def pick_mt(numel: int, num_experts: int) -> int:
    """M tiles per routing block from the host-known row count: rows per touched expert >= 32 -> 4, >= 16 -> 2."""
    per = numel / max(1, min(num_experts, numel))
    return 4 if per >= 32 else (2 if per >= 16 else 1)


MOE_PREFILL_CFG = int(os.environ.get("R9K_MOE_PREFILL_CFG", "17"))     # -1: routed MoE never uses the prefill tile
MOE_PREFILL_CFG_GATE_UP = int(os.environ.get("R9K_MOE_PREFILL_CFG1", "17"))   # -1: gate_up stays on the MT kernel


def pick_moe_prefill(MT: int, K_down: int, gate_up: bool = False) -> int | None:
    """Prefill tile for the routed-MoE GEMMs of a step whose routing block is 16*MT rows (None: keep the MT kernel).
    Both tiles share block 64 with MT=4, so one moe_align_block_size table serves both GEMMs. Since the epilogue
    reads its per-row scales from LDS (2026-09-28) the 64x128 BK=32 double-buffered tile (cfg 17) is the best of
    the block-64 tiles for both GEMMs at both TP sizes (tuning/prefill_moe_bench.py, folded, 4096 tokens):
      * TP=2: gate_up 640x2560 1531 us (cfg 15: 1638, MT kernel: 2085); down 2560x320 1169 (cfg 11: 1270, MT 2719)
      * TP=4: gate_up 320x2560  899 us (cfg 15:  980, MT kernel:  969); down 2560x160  811 (cfg 11:  855, MT 1982)
    BK=32 also admits K=160 (TP=4 down), which the BK=64 tiles reject."""
    cfg = MOE_PREFILL_CFG_GATE_UP if gate_up else MOE_PREFILL_CFG
    if cfg < 0 or 16 * MT != prefill_block(cfg) or K_down % prefill_bk(cfg):
        return None
    return cfg



def moe_sum(down: torch.Tensor, shared: torch.Tensor | None, out: torch.Tensor) -> torch.Tensor:
    """out[M, N] = sum over top-k of down[M, topk, N] (+ shared[M, N]) in fp32, one bf16 rounding
    (kernels/r9k_moe_sum.hip). Replaces ops.moe_sum, the finalize copy and the runner's shared add."""
    M, topk, N = down.shape
    assert down.is_contiguous() and out.shape == (M, N) and out.dtype == torch.bfloat16 and out.stride(1) == 1
    L = lib()
    if not hasattr(L, "_moe_sum_bound"):
        L.r9k_moe_sum.restype = ctypes.c_int
        L.r9k_moe_sum.argtypes = [ctypes.c_long] * 3 + [ctypes.c_int] * 3 + [ctypes.c_long] * 3
        L._moe_sum_bound = True
    rc = L.r9k_moe_sum(down.data_ptr(), shared.data_ptr() if shared is not None else 0, out.data_ptr(), M, topk, N,
                       shared.stride(0) if shared is not None else 0, out.stride(0), _stream())
    if rc:
        raise RuntimeError(f"r9k_moe_sum failed ({rc}) M={M} topk={topk} N={N}")
    return out


def align_block_size_ref(topk_ids: torch.Tensor, num_experts: int, block: int = MOE_BLOCK):
    """Pure-torch equivalent of vLLM's moe_align_block_size (for tests): pads each expert's rows to a
    multiple of `block`, pad value = numel. Returns (sorted_ids, expert_ids, ntpp) sized to capacity."""
    flat = topk_ids.flatten().to(torch.int64)
    numel = flat.numel()
    cap = numel + num_experts * (block - 1)
    cap = (cap + block - 1) // block * block
    sorted_ids = torch.full((cap,), numel, dtype=torch.int32)
    expert_ids = torch.full((cap // block,), -1, dtype=torch.int32)
    pos = 0
    for e in range(num_experts):
        rows = (flat == e).nonzero().flatten()
        if rows.numel() == 0:
            continue
        n = (rows.numel() + block - 1) // block * block
        sorted_ids[pos:pos + rows.numel()] = rows.to(torch.int32)
        expert_ids[pos // block:(pos + n) // block] = e
        pos += n
    return sorted_ids, expert_ids, torch.tensor([pos], dtype=torch.int32)
