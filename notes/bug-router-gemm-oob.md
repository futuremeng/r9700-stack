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

`MT=8` means the launch happened with `M <= 8` (the C++ dispatcher pads rows 5..8 to `MT=8`). With MTP=2 the target
pass runs `requests x (1 + 2)` rows, so 32 streams is 96 rows and 8 streams is 24 -- both above the knob, both fall
back to `torch.mm`. The pass that runs one
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
sits next to what, and the earlier faults this kernel has (2026-11-04, below) predate the fp8 KV default by a day either way.

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

## This is not the first time: 2026-11-04 named the same kernel, and this same cap was already the discriminator

[launch-methodology.md §12](launch-methodology.md) records two `hipErrorIllegalAddress` deaths on **2026-11-04**
(07:53 with the prefix cache off, 14:01 with it on), each naming `r9k_router_gemm_kernel<8, 2, true>`, on a decode
step of **6 rows** (2 streams x (1 + 2 MTP drafts)) at up to 203,257 computed tokens. The experiment that followed
was this cap at 4 -- `R9K_ROUTER_MAX_M=4`, session `r9700-native-TEST-routermaxm4` -- which carried real traffic for
**9 h 37 m with zero illegal-access lines and zero EngineDeadError** (329,983 drafts, 1.317 accepted per draft), and
was then reverted to the default `M <= 8` on purpose, with the note writing the rule: *"if the fault comes back in
`r9k_router_gemm_kernel<8, 2, true>`, the discriminator did not discriminate, and the next move is serialization,
not another cap."*

Today's two deaths are exactly that come-back. So the honest reading of the cap evidence is now:

- faults at the default cap 8: **four** (2026-11-04 x2, 2026-11-08 x2) -- and the 11-08 pair happened in a 20 s
  GSM8K burst at *short* context, so the deep-context framing of 11-04 was not a requirement: what the two pairs
  share is an `M` in 5..8 (6 target rows at conc 2 then, 8 draft rows at conc 8 now), not a context depth;
- clean at cap 4: **9 h 37 m** of live traffic, plus every `M <= 4` step this box has ever run (the cap only moves
  which instantiation exists, and `M <= 4` never selects `MT=8`);
- clean at cap 0: today's full-protocol arm A1.

Per §12's own rule, if a launch with the cap at 4 faults in this kernel the next move is `AMD_SERIALIZE_KERNEL=3` --
the driver's own suggestion, which names the faulting kernel instead of the first innocent reader that trips over a
poisoned page -- not a third cap.

## What the recipe now carries (baked 2026-11-08)

`serve/native-flashnext.sh` sets `: ${R9K_ROUTER_MAX_M:=4}` and exports it, so the standing recipe -- not an
operator's memory -- holds the cap, and the launcher's env snapshot (which is taken after the last default it sets)
records it in the recipe file; `tools/recipe-check.py` compares `R9K_*` by value, so a live serve at cap 4 and a tree
at cap 4 agree, and `R9K_ROUTER_MAX_M=8` in an operator's shell is visible as a test config. The library default in
`r9700_vllm/router.py` moved to 4 as well, so the container path inherits the same choice rather than only the
native launcher. Cap 4 rather than cap 0 because cap 4 is the value with 9 h 37 m of clean traffic behind it, it
keeps our kernel on the 1-2 stream steps where its ~0.3 ms lives, and it excludes the faulting `MT=8` instantiation
just as completely.

Verified on the launch that carries it (`~/logs/relaunch-262k-cap4.log`, 12:37, `bench/relaunch.sh 262k-cap4`):

- boot proof, all of it rather than `tail -1` per [launch-methodology.md §12](launch-methodology.md): four ranks x
  `r9k router GEMM installed on 48 gates (M <= 4, split 2)` and four x `installed on 1 gates` (the MTP draft model
  has one gate);
- `tr '\0' '\n' < /proc/<pid>/environ | grep '^R9K_ROUTER_MAX_M='` -> `R9K_ROUTER_MAX_M=4`, and
  `ENVRECIPE=1` from the launcher itself answers `R9K_ROUTER_MAX_M = 4`, so the cap is in the recipe rather than in
  somebody's shell;
- fingerprints of that launch: load 72.6 s, KV 940,726 tokens / 3.59x at 262,144;
- **the phase that killed both cap-8 arms**: `bench/quality.py 100 8` -> **97.0 % (97/100), 33,209 prompt tokens in
  40 s, needles 3/3**, engine still up afterwards, and the launch log holds 0 `illegal memory access` /
  `HSA_STATUS_ERROR` lines;
- the cost of the cap, three-way on single-stream latency: `bench/itl.py 1 2 4` gives **13.0 / 15.1 / 18.4 ms** at
  cap 4 against 12.9 / 14.7 / 18.5 at cap 8 (A0) and 13.2 / 15.0 / 18.5 at cap 0 (A1). Cap 4 recovers most of the
  conc-1 step (3 rows stay on ours) and matches cap 0 from conc 2 up, exactly as the row arithmetic says
  (2 streams x 3 = 6 rows > 4).

## What is NOT the cause

Ruled out from the journal for both crashes: no reboot (uptime 3 d 8 h), no OOM-killer, no MCA/MCACOD today (the
sick-L2 workaround is intact: cpu17/113 still offline, `r9700-offline-core.service` enabled), no platform reset and
no PCIe AER storm at the crash minutes (the day's 247 CPER lines are the usual riser-chain correctables, last at
10:10). The kernel also logged the known `amdgpu_amdkfd_restore_userptr_worker hogging CPU` line at 11:35:03 -- the
signature of the SVM/mm-notifier family in [load-oom-svm-wedge.md](load-oom-svm-wedge.md), which is why the first
pass of this investigation went looking there; the device report names the router kernel, and the crash is a device
fault, not a wedge.

## Box state and the effect on the 0.32-vs-0.33 comparison

The comparison in [vllm-0.33.md](vllm-0.33.md) is **paused with arm A0 half-taken and arm A1 complete**. A0 (cap 8)
is the clean 0.32 baseline up to the quality gate -- load 85.6 s, KV 978,957 tokens / 3.73x, probe32 and the ladder
inside 1-4 % of the recorded 2026-11-02 bands, ITL 12.9 / 14.7 / 18.5. A1 (cap 0) is the first full pass of the whole
protocol on this box since the merge: GSM8K 97/100, needles 3/3, strict sanity 0 bad of 1,980 / 2,880 / 3,840 plus 0
of 330, and a 1504 s soak at 16 clients with **753 ok, 0 errors, 0 context-length rejections**.

The box now stands at the baked cap 4 (`~/logs/relaunch-262k-cap4.log`: load 72.6 s, KV 940,726 tokens / 3.59x,
health OK, GSM8K 97/100 and needles 3/3 run against it). That is also what makes the paused 0.33 arms (B:
`~/venv-r9700-033`, C: a `+rocm100` venv) comparable when they do run -- both sides inherit the same cap from the
launcher instead of from a shell.
