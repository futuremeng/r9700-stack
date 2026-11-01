"""The fold guard's sampling (R9K_FOLD_SAMPLE / R9K_FOLD_MARGIN): that the short-K refusal walks nothing, that a
sample is a subset of the same estimator, that a tensor near a threshold (or whose sample saw any flushed block) is
decided on the exact walk instead, that badness concentrated in a few channels is still caught, and that every device
samples the same blocks -- four ranks must not reach different decisions on one tensor.

Run: LD_LIBRARY_PATH=$HOME/opt/openmpi/usr/lib/x86_64-linux-gnu python3 tests/test_fold_sample.py
"""
import time

import torch

from r9700_vllm.kernels import moe as K

ok = True
E, NB1, N = 128, 81, 1280                           # Flash-Next's routed w13: K = (nb1-1)*32 = 2560, 13.1 M blocks
PLANE = (NB1 - 1) * N                               # blocks in one expert
SAMPLED = K.FOLD_SAMPLE


def say(name, good, extra=""):
    global ok
    ok &= bool(good)
    print(f"  {name:74s} {'ok' if good else '<-- FAIL'} {extra}", flush=True)


def packed(dfun=None, e=E, nb1=NB1, n=N):
    """A packed [e, nb1, n] scale tensor whose block exponents sit d = ref - block below a reference of 120, with
    d = dfun(flat block index). Built directly: that is the layout fold_stats reads."""
    plane = (nb1 - 1) * n
    f = torch.arange(plane, dtype=torch.int64)
    d = torch.zeros(plane, dtype=torch.int16) if dfun is None else dfun(f).to(torch.int16)
    w = torch.empty((e, nb1, n), dtype=torch.uint8)
    w[:, :-1, :] = (120 - d).clamp(1, 254).to(torch.uint8).reshape(nb1 - 1, n)
    w[:, -1, :] = 120
    return w


def every(k, d):
    """d at every k-th block of the plane -- so exactly a 1/k fraction of the blocks, uniform over columns and rows."""
    return lambda f: (f % k == 0) * d


def cols(colset, d, n=N):
    """d for every block of the given columns: badness concentrated in output channels rather than spread out."""
    cs = torch.tensor(list(colset), dtype=torch.int64)
    return lambda f: torch.isin(torch.remainder(f, n), cs).to(torch.int16) * d


real_stats = K.fold_stats                           # the walk itself, callable with sample= explicitly

# ---- 1. short-K: refused from the shape, and the tensor's bytes are never read
wsr_short = packed(e=2, nb1=6, n=64)                # K = (6-1)*32 = 160 < FOLD_MIN_K
calls, msgs = [], []
K.fold_stats = lambda *a, **kw: (calls.append(1), real_stats(*a, **kw))[1]
saved_mode = K.FOLD_MODE
K.FOLD_MODE = "1"
got = K.fold_decide(wsr_short, "short-K", log=msgs.append)
say("short-K (K=160) refused without walking the scales", got is False and not calls,
    (msgs[0].rsplit(": ", 1)[-1] if msgs else "")[:74])
K.FOLD_MODE = "force"
say("R9K_FOLD=force still folds a short-K tensor", K.fold_decide(wsr_short, "forced", log=lambda m: None) is True)
K.FOLD_MODE, K.fold_stats = saved_mode, real_stats

# ---- 2. a sample is a subset of the same estimator
wsr = packed()                                      # all clean
st_full, st_smp = real_stats(wsr, sample=0), real_stats(wsr)
say("the big tensor is sampled, the sample respects the budget, and the clean one folds either way",
    st_full.total == E * PLANE == 13107200 and st_full.blocks == st_full.total and st_full.exact
    and SAMPLED * 0.9 < st_smp.blocks <= SAMPLED and K.fold_ok(st_smp) and not st_smp.exact,
    f"{st_smp.blocks} of {st_smp.total} blocks")
say("sampled max_d / mean_d match the exact walk on clean data",
    st_full.max_d == st_smp.max_d == 0 and st_full.mean_d == st_smp.mean_d == 0.0, str(st_smp))

# ---- 3. the sample is deterministic and spread over the whole plane (four ranks, same blocks)
(f1, c1), (f2, c2) = K._fold_sample_idx(wsr, SAMPLED), K._fold_sample_idx(wsr.clone(), SAMPLED)
say("same shape -> same blocks (fixed seed)", torch.equal(f1, f2) and torch.equal(c1, c2) and f1.dtype == torch.int64,
    f"{f1.numel()} of {PLANE} blocks per expert")
say("the sample spans both axes, not one run of blocks",
    int(f1.min()) < 0.02 * PLANE and int(f1.max()) > 0.98 * PLANE and len(set((f1 % N).tolist())) > 0.05 * N
    and len(set((f1 // N).tolist())) == NB1 - 1,
    f"{len(set((f1 % N).tolist()))} of {N} columns, {len(set((f1 // N).tolist()))} of {NB1 - 1} block rows")

# ---- 4. near a threshold, or any flushed block in the sample: the exact walk decides
thr = K.FOLD_MAX_FLUSH
st = real_stats(packed(every(round(1 / 0.002), 14)))         # 0.2% of blocks flushed, against a 0.1% threshold
say("a 0.2%-flushed tensor: the sample sees it and re-walks every block", st.exact and st.blocks == st.total, str(st))
say("and it is refused", not K.fold_ok(st))

st_at = real_stats(packed(every(900, 14)))                   # p_flush 0.11%: just over the threshold
say("p_flush just over the threshold: refused", not K.fold_ok(st_at) and st_at.exact, f"p_flush={st_at.p_flush:.5f}")

st_in = real_stats(packed(every(17, 10)))                    # d=10: rounded (inexact), not flushed; 5.9% vs a 5% line
say("inexact 6% (threshold 5%) refused on the exact walk", st_in.exact and not K.fold_ok(st_in), str(st_in))

st_under = real_stats(packed(every(1400, 14)))               # 0.071% flushed, under 0.1% but inside the margin band
say("inside the margin band under the threshold: escalated, then allowed", st_under.exact and K.fold_ok(st_under),
    f"p_flush={st_under.p_flush:.5f} vs thr {thr}")

st_far = real_stats(packed(every(70, 10)))                    # inexact 1.4%: under the 5% line, outside the 3x band
say("a fraction under the line and outside the band is decided on the sample, with no exact re-walk",
    not st_far.exact and K.fold_ok(st_far), str(st_far))
nper = st_smp.blocks                                          # blocks sampled from the whole tensor
say("the sample is big enough that a tensor sitting AT the threshold is almost surely seen",
    nper * thr >= 100, f"{nper} blocks x thr {thr} = {nper * thr:.0f} expected flushed hits -> "
    f"detection is certain in practice; that is what makes the any-hit rule enough")

tiny = real_stats(wsr, sample=1000)
say("a 1000-block budget still samples whole blocks per expert", tiny.blocks == E * max(1, 1000 // E),
    f"{tiny.blocks} blocks")
f_hit = int(K._fold_sample_idx(wsr, 1000)[0][0])             # a flushed block exactly where that sample looks
st_hit = real_stats(packed(lambda f: torch.where(f == f_hit, torch.tensor(14), torch.tensor(0))), sample=1000)
say("one flushed block seen in a 1000-block sample escalates to the exact walk",
    st_hit.exact and st_hit.blocks == st_hit.total and K.fold_ok(st_hit),
    f"block {f_hit}: {st_hit.blocks} blocks walked, p_flush={st_hit.p_flush:.2e}")

# ---- 5. badness concentrated in a few channels is not missed by a block-uniform sample
say("a whole column at d=14 (0.078% of blocks): escalated, allowed on the exact count",
    (lambda s: s.exact and K.fold_ok(s) and 0 < s.p_flush < thr)(real_stats(packed(cols([7], 14)))),
    f"p_flush={real_stats(packed(cols([7], 14))).p_flush:.5f}")
say("three channels at d=14 (0.23%) refused",
    not K.fold_ok(real_stats(packed(cols([7, 500, 1000], 14)))))
say("a tenth of the channels at d=14 (7.8%) refused",
    not K.fold_ok(real_stats(packed(cols(range(0, N, 10), 14)))))

# ---- 6. every device samples the same blocks and counts the same histogram
if torch.cuda.is_available():
    w = packed(every(round(1 / 0.002), 14))
    a, b = real_stats(w.cpu()), real_stats(w.cuda())
    say("CPU and CUDA walks agree exactly", a.hist == b.hist and a.blocks == b.blocks and a.max_d == b.max_d,
        f"{a.blocks} blocks")
    ga, gb = K._fold_sample_idx(w, 4096), K._fold_sample_idx(w.cuda(), 4096)
    say("and they sample the same blocks", torch.equal(ga[0], gb[0].cpu()) and torch.equal(ga[1], gb[1].cpu()))

# ---- 7. what the guard costs a rank, on the shapes the load actually walks (965 M blocks, 8.1-8.5 s measured)
print("  the load's own guard, one rank's worth of tensors (Flash-Next):")
tot = {"exact": 0.0, "new": 0.0}
for (e, nb1, n, nt) in ((E, NB1, N, 49), (E, 6, 10240, 49)):    # routed w13 (K=2560) and w2 (K=160)
    w = packed(e=e, nb1=nb1, n=n).cuda()
    t0 = time.perf_counter(); s0 = real_stats(w, sample=0); t_full = time.perf_counter() - t0
    t0 = time.perf_counter(); s1 = real_stats(w); t_smp = time.perf_counter() - t0
    tot["exact"] += t_full * nt
    tot["new"] += (t_smp if s0.K >= K.FOLD_MIN_K else 0.0) * nt   # short-K: shape-refused, nothing walked now
    print(f"    K={s0.K:<5} {s0.total / 1e6:6.1f} M blocks x{nt}: exact {t_full * 1e3:7.1f} ms   "
          f"now {t_smp * 1e3:6.1f} ms   decision {'same' if K.fold_ok(s0) == K.fold_ok(s1) else '<-- DIFFERS'}"
          + ("" if s0.K >= K.FOLD_MIN_K else "   (shape-refused, nothing walked)"))
print(f"  guard, per rank: {tot['exact']:.2f} s exact -> {tot['new']:.2f} s as shipped "
      f"({tot['exact'] / max(0.01, tot['new']):.0f}x); the load timer said 8.1-8.5 s for all 196 tensors")
print("\nfold sampling tests:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
