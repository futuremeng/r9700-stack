# `r9k_router_gemm_kernel<8, 2, true>` faults at conc 8: the MoE gate kernel the fault names, and what turns on it

Two launches of the standing 262k recipe died inside `bench/quality.py` (GSM8K at conc 8) on 2026-11-08, ~20 s into
the phase, and the device named the kernel:

| # | time | device error | dispatch |
|---|---|---|---|
| 1 | 11:34:59 | `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` → `hipErrorIllegalAddress` (Worker_TP0) | `grid=[32768,1,1] workgroup=[256,1,1]` |
| 2 | 11:49:25 | `HSA_STATUS_ERROR_MEMORY_FAULT` → `hipErrorIllegalAddress` (Worker_TP0) | `grid=[32768,1,1] workgroup=[256,1,1]` |

Both print, repeatedly, from `rocdevice.cpp:3948`:

```
[host: jiulu-9700, GPU index: 0, faulting addr: 0x7d94e403f000,
 kernel: void (anonymous namespace)::r9k_router_gemm_kernel<8, 2, true>(un ...]
```

`workgroup=[256,1,1]` is `NT = 32 * WAVES = 256` and `grid_x` in an HSA dispatch is total workitems, so
32768 / 256 = **128 blocks** = `E / EPB` with `EPB = WAVES / SPLIT = 4` → **E = 512, SPLIT = 2, MT = 8**: exactly
the Flash-Next MoE gate (`K=2560, E=512`) through `r9k_router_gemm` (`kernels/r9k_router.hip`, added by `d1a5a9a`
2026-09-28 at `R9K_ROUTER_MAX_M` default 8). The faulting address is page-aligned, which is the shape of a read
walking off the end of a mapping, not a wild index.

## Which traffic reaches that kernel, and why the crash lands on conc 8 and not on a probe32 wave

`MT=8` means the launch happened with `M <= 8`. With MTP=2 the target pass runs `requests x (1 + 2)` rows, so 32
streams is 96 rows and 8 streams is 24 -- both above the knob, both fall back to `torch.mm`. The pass that runs one
row **per request** is the drafter, so at conc 8 the draft step is exactly `M = 8`, which is inside the kernel. That
is the whole pattern:

- `probe32` (32 and 16 in flight) and the 60k/120k/180k ladder ran **clean** in both crashed arms;
- `bench/quality.py 100 8` -- 8 requests in flight, so the drafter runs M=8 -- killed both, 20 s in, at
  `Running: 8 reqs`, acceptance length 2.69 and 2.7;
- `sanity_stress` at conc 33/48/64 never enters the kernel (M=33/48/64 > 8) and it is the phase that would have
  followed.

## Mitigation, measured rather than asserted

`R9K_ROUTER_MAX_M=0` makes `_fits` false for every M, so the gate goes to `torch.mm(x, w.t(), out_dtype=fp32)` +
cast. The whole protocol then passes (arm A1, 2026-11-08, PC=0, `R9K_ARN_CALIBRATE=0`, `~/venv-r9700`, tree =
v0.4.3 merged, `libr9k.so` rebuilt from that tree):

| | A0 -- router kernel ON | A1 -- `R9K_ROUTER_MAX_M=0` |
|---|---|---|
| probe32 @32x300 | 1,518.1 / 1,540.7 / 1,539.5 | 1,561.5 / 1,556.5 / 1,571.0 |
| probe32 @32x1200 | 1,790.9 / 1,790.9 | 1,786.3 / 1,792.4 |
| probe32 @16x300 | 1,099.8 / 1,115.2 / 1,091.2 | 1,089.2 / 1,104.5 / 1,090.5 |
| power at busy==100 | median 299 W (n=140) | median 300 W, max 315 W (n=128) |
| ITL conc 1 / 2 / 4 | 12.9 / 14.7 / 18.5 ms | **13.2** / 15.0 / 18.5 ms |
| ladder 60k / 120k / 180k prefill | 8,600 / 7,908 / 7,342 tok/s | 8,563 / 7,910 / 7,314 |
| GSM8K 100 + needles | **serve died 20 s in** (twice) | **97.0 % (97/100), 33,834 prompt tokens in 42 s, needles 3/3** |
| strict sanity 33 / 48 / 64 + long prefills first | never reached | **0 bad of 1,980 / 2,880 / 3,840, and 0 of 330** |

So the knob is worth nothing at aggregate (A1 is inside the launch spread of A0 at every probe32 protocol, and 2 %
*better* at 32x300, which is the spread) and about **+0.3 ms of single-stream ITL (~2 %)**, against a box that dies
mid-request without it. The kernel itself was measured at the time it landed as ~19.6 -> ~5 us over 51 calls a step,
which is where the ~2 % comes from.

## What the kernel's own addressing rules out

Reading `r9k_router_gemm_kernel`: loads are `wp + c*8` with `c < nch = (K/SPLIT)/8` (so inside the weight row),
`xp + mr*sx + c*8` with `mr = m < M ? m : M-1` (clamped to the last real row), stores guarded by `live` (`e < E`)
and `lane < M`, and `_fits` on the python side requires `K % (8*split) == 0`, contiguous rows and strides `% 8 == 0`.
Every index is inside the buffers **as described by the arguments passed**. A page-aligned fault on a kernel whose
indices are in range points at arguments that no longer describe live memory -- a pointer baked somewhere with the
buffer behind it gone -- rather than at arithmetic in the kernel body.

Leading hypothesis, not established: the op is registered as a custom op with a fake impl
(`_router_gemm_op` / `_router_gemm_fake` in `r9700_vllm/router.py`) so it survives `torch.compile`, and the serving
launch runs it inside captured HIP graphs; a captured `w` (or `x`) pointer replayed after the tensor behind it was
freed or replaced faults exactly this way. This repo already knows weights can be freed under a sibling path --
`hc.py: down_silu` carries the comment "the bf16 weight may be freed" and keeps `n_out` for it. Secondary
possibility: the read was always slightly out of range and only became a fault because the allocation map moved --
the fp8 main KV cache (default since [kv-fp8-qsa.md](kv-fp8-qsa.md)) and multimodal profiling both changed what
sits next to what, and the last clean GSM8K runs of this kernel (2026-11-02/03) predate both.

## What would settle it, in order of cost

1. `EAGER=1` with the router ON, then `bench/quality.py 100 8`. Survives -> the graph capture/replay is implicated,
   and `EAGER=1` is a second (expensive) mitigation; dies -> it is the shapes/arguments, independent of graphs.
2. A conc-8 replay loop against a launch with the router ON (`bench/quality.py` alone, or a 20-line drafter-shaped
   loop at M=8) to get the fault on demand instead of at 1-in-2 launches.
3. `tests/test_router_r9k.py` extended to the served shape through a capture/replay of the op (E=512, K=2560,
   split 2, M=1..8), with the weight allocated, the graph captured, the weight released, and the graph replayed --
   the hypothesis in (1) as a unit test.
4. Log `mlp.gate.weight.data_ptr()` (all 51 MoE gates plus the MTP draft layer's) at load and at the first GSM8K
   step: if a pointer moved, the baked-pointer hypothesis is proven and the fix is on the plugin side, not in the
   kernel.

## What is NOT the cause

Ruled out from the journal for both crashes: no reboot (uptime 3 d 8 h), no OOM-killer, no MCA/MCACOD today (the
sick-L2 workaround is intact: cpu17/113 still offline, `r9700-offline-core.service` enabled), no platform reset and
no PCIe AER storm at the crash minutes (the day's 247 CPER lines are the usual riser-chain correctables, last at
10:10). The kernel also logged the known `amdgpu_amdkfd_restore_userptr_worker hogging CPU` line at 11:35:03 -- the
signature of the SVM/mm-notifier family in [load-oom-svm-wedge.md](load-oom-svm-wedge.md), which is why the first
pass of this investigation went looking there; the device report names the router kernel, and the crash is a device
fault, not a wedge.

## Box state and the effect on the 0.32-vs-0.33 comparison

The comparison in [vllm-0.33.md](vllm-0.33.md) is **paused with arm A0 half-taken and arm A1 complete**: A0 (router
ON) is the clean 0.32 baseline up to the quality gate -- load 85.6 s, KV 978,957 tokens / 3.73x, probe32 and the
ladder inside 1-4 % of the recorded 2026-11-02 bands, ITL 12.9/14.7/18.5 -- and A1 (router OFF) is the first full
pass of the whole protocol on this box since the merge. The 0.33 arms (B: `~/venv-r9700-033`, C: a `+rocm100`
venv) are unmeasured, and until the router question is closed they have to run with the same
`R9K_ROUTER_MAX_M` value on both sides to be comparable at all.
