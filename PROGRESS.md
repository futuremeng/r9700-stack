## 2026-11-03 (14:34-15:20): merge upstream v0.4.0 (187 commits), make the fp8 QSA main KV this box's default, and put the release page back on the generator

* Saved the fp8 work as `b13d63d` first, then `git fetch upstream` + merge: **187 upstream commits, 54 local, and only
  two conflicts** -- `kernels/build.sh` and `README.md`. Upstream had touched neither `r9k_qsa.hip` nor
  `r9700_vllm/attn/qsa.py`, and `PROGRESS.md` is fork-only, which is why the merge was small where it mattered.
  Upstream brings DeepSeek-V4.1-Flash EXL3 (loader, moe/linear/mhc kernels, `serve/ds41-exl3.sh`), the ar8 8-rank
  collective, dsv41 decode attention, and the v0.4.0 tag.
* `build.sh`: upstream's export list (the `r9k_mhc_*`, `r9k_moe_exl3_*`, `r9k_exl3_dq_tiles` and ar8/two-shot extras)
  with `r9k_qsa_attn_f8` inserted after `r9k_qsa_attn`, kept on this fork's single-`nm`-pass form because the pipe
  form is the pipefail flake recorded below. Merged build: 7,033,544 bytes, 96 `r9k_` exports, all checks pass, and
  `tests/test_qsa_fp8.py` still passes on the merged .so (rel 3.1e-3-3.9e-3, fp8/bf16 0.92-0.99).
* `README.md`: composed rather than picked -- upstream's v0.4.0 hero, graphic, four-model list, DeepSeek speed table,
  `ds41-exl3.sh` line and EXL3 checkpoint recipe, this fork's nav + setup.sh callout + "Four cards, the numbers worth
  knowing" + the reading guide + the uncapped-32k / full-262k / real-traffic subsections + the Monitor pages + the
  docker-free launcher path. The four release tables got their H3 back ("four, two and eight cards") so the release
  config and the three native-box subsections sit at one depth.
* `tests/test_readme.py` was pinned to the v0.3.8 heading and figure and failed the merge (2 checks). It now checks
  what is true: the README's release figure is the current release's and used once, the hand-drawn v0.4.0 page names
  v0.4.0 and its 3 numeric texts all appear in the README, and its smallest type is **measured, 11.0 px on a
  1200-unit canvas = 9.0 px in github's ~980 px column** -- the reason `docs/summary-page.py` exists. The generated
  page and its PNG stay in the tree and still have to re-draw clean text-for-text, but they are not back in the
  README until the generator carries v0.4.0's numbers (the DeepSeek row). That is the open step, written in the test.
* **fp8 main KV is now the launcher's default** (`serve/native-flashnext.sh`, fork-local: upstream has no
  native-flashnext.sh). `R9K_QSA_KVF8=1` + `KVDTYPE=fp8_e4m3` with the measurement next to the knob, and the launcher
  refuses `R9K_QSA_KVF8=0` with an fp8 `KVDTYPE` because stock QSA raises on it -- verified: exit 1 with the bf16
  rollback line (`R9K_QSA_KVF8=0 KVDTYPE= bash serve/native-flashnext.sh`), no process started. The README's context
  row and the 262k table now carry both numbers (1,009,899 fp8 / 540,202 bf16).
* First launch on the merged tree with the default: registers `patch:qsa_kvfp8` next to upstream's `quant:exl3` and
  `patch:dsv41_rdna4`, fp8_e4m3 on four ranks, block 880, **1,009,899 tokens / 3.85x**, init 21.8 s (0.52 s compile).
  Gates (`~/logs/merged-gate.log`): `quality.py 300 8` 95.0% (285/300 -- inside the 95.0-96.0% instance band both
  configs showed today), needle 3/3 and cold+warm 8/8 to 250,078 tokens, `probe32` @32x300 1,511-1,563,
  @32x1200 1,789/1,796, @16x300 1,090-1,114, 300 W a card, and **accepted per draft 1.548 against 1.549 pre-merge**:
  upstream's 187 commits did not move this model's numerics or its throughput. Box left running this launch.
* Not done: nothing pushed (origin/master is still where it was; the merge and the fp8 work are 2 commits + this one
  on local `master`), and the generated release page still awaits v0.4.0's numbers.

* **README data and figure carry v0.4.0 now** (15:05-15:20). The numbers: the four-cards block and the 262k table name
  the fp8 launch's absolutes rather than a percentage -- aggregate **1,792** at @32x1200 (1,789 / 1,796 against bf16's
  1,817 / 1,831), prefill **2,614 / 7,886 / 9,702 / 8,965 / 8,282 tok/s at 485 / 1,976 / 7,792 / 31,131 / 77,733
  tokens**, pools 1,009,899 against 540,202. The figure: the hero points at the generated page again. `docs/summary-page.py`
  gained the fourth model (DeepSeek-V4.1-Flash: 552B MoE, EXL3 2.9 bpw, TP8+EP, 158 tok/s on code at its "-500 MHz"
  gate), a per-bar gate label, a 2x2 card grid (`COLS` -- four names at 19 px do not fit four across at 276 px, so two
  across at 421), the fp8 facts column, and a **version-free output name**: `docs/r9700-stack-page.svg` (+ a
  2000 x 2532 PNG for zooming), because the version is data in the script and the README's pointer should not age at
  the next tag. It draws 1000 x 1266 with no OVERFLOW. Why that page and not the tag's own graphic is measured, not
  asserted: upstream's hand-drawn `docs/r9700-stack-v0.4.0.svg` floors at 11.0 px on a 1200-unit canvas = **9.0 px**
  rendered in github's ~980 px column; this one floors at 14 px on 1000. `tests/test_readme.py` checks that shape --
  the README carries exactly one release-page figure, the caption names the script, every numeric text on the page
  appears in the README (7 today), four bars each naming its gate, re-draw reproduces the committed file text for text,
  PNG at least 2x the canvas -- and keeps the hand-drawn page's measurement in the output. `docs/summary-image.py`'s
  cross-reference to the old filename fixed. `test_readme.py` and `test_lockup.py` PASS.

## 2026-11-03 (afternoon): fp8 QSA main KV -- the writer was already fp8, so the whole ask is three guards and one read path

* Reopened `PROGRESS.md:896` ("fp8 KV is refused by stock QSA"). Reading `vllm models/qwen4_exp/amd/`: the **write**
  side already does e4m3 -- the fused QSA prepare Triton kernel stores K/V through `_to_dst_dtype()` (`tl.float8e4nv`,
  `amd/ops/qsa_prepare.py:123-140,268`) with descales `attn._k_scale_float` / `._v_scale_float`
  (`amd/indexer_qsa.py:323-326`) -- and upstream's **NVIDIA reader** already reads fp8 with those same scales
  (`nvidia/qsa.py:306`). On the ROCm side the walls are three `raise`s (`amd/qsa.py:113,190,273`), our own guard
  (`attn/qsa.py:138`) and a bf16-only `r9k_qsa.hip`. Full write-up: **`notes/kv-fp8-qsa.md`**.
* Prize, from the running instance's own log (block 448, mamba pad 1.98%, `7.22 GiB` a card, `540,202 tokens`,
  `2.06x` at 262,144): the main KV is 12 full-attention layers x 1,024 B/token = **12,288 of the 14,348 B/token/card**
  (85.6%), so e4m3 there puts the pool at **~0.96-1.02M tokens** and 262k-context concurrency at **~3.7-3.9x**. The
  block size moves 448 -> 896 (attention page must stay >= the 449,828 B GDN page); decode/prefill gain ~nothing
  (2,048 tokens x 1,024 B x 12 layers is ~1% of GDDR7) -- **this is a capacity change, not a speed one**.
* Built (offline; the box served the whole session, nothing was restarted): `r9k_qsa.hip` attention kernel templated
  on the KV element type -- fp8 changes only the two stage loads (16 B of 16 fp8 for K, `global_load_tr_b64_v2i32` for
  V^T, `fp8x8_to_bf16` on the way in) and keeps the **bf16 LDS layout, occupancy and WMMA pipeline identical**; K's
  descale folds into `scale_log2`, V's into the output normalizer (correct through split+merge too). New entry
  `r9k_qsa_attn_f8`. `attn/qsa.py` dispatches by dtype and reads the descale once per layer from the same
  `_k_scale_float` / `_v_scale_float` the writer used. `compat/qsa_kvfp8.py` (+ `gate.py`, plugin registration) builds
  the layer with `cache_dtype` temporarily reported as bf16 -- which also keeps the **side** caches bf16 -- then sets
  the layer's own dtype to fp8 for `get_kv_cache_spec()`. `tests/test_qsa_fp8.py` separates kernel correctness from
  quantization error per case (GQA 6/12, rows 1-2048, split/merge, page 448/896, descale 1.0/0.5).
* **Measured on this box in the 13:23-13:52 idle window** (two fresh instances, box left on bf16). Kernel test:
  the fp8 read path is as tight as bf16 (rel 1.7e-3-3.9e-3 across rows 1-2048, page 448/896, nsplit auto/1,
  descale 1.0/0.5/0.25) and **7-11% faster**. Serving: block size **448 -> 880** (mamba pad 1.98% -> 0.16%),
  `GPU KV cache size` **540,202 -> 1,021,931 tokens (+89.2%)**, `Maximum concurrency for 262,144` **2.06x -> 3.90x**.
  **needle 8/8 including 250,078-token prompts** (depth 105,747). GSM8K no-think 300 @8: 95.3% vs 96.0% bf16 --
  two questions, no detectable difference at that n. `probe32`: @32x1200 1,785/1,802 vs 1,808/1,811 (-0.7%),
  @32x300 1,528-1,576 vs 1,594-1,596, @16x300 1,122 vs 1,131; accepted/draft 1.549 vs 1.604 (-3.4%); power still a
  flat 300 W. One real cost: the fp8 cache dtype invalidates the AOT compile (**32.1 s** vs 0.53 s on the first
  launch). Table and reading: `notes/kv-fp8-qsa.md`.
* **Second window, paired arms** (13:51-14:00 bf16 then 14:06-14:15 fp8, one fresh instance each, `bench/quality.py
  800 8` + `needle_long` + `prefill.py` + `probe32` in the same order, logs `~/logs/gatepair-{bf16,kvf8}.log`):
  GSM8K no-think 800 @8 **96.0% fp8 against 95.6% bf16**; needle 3/3 and 8/8 cold+warm out to 250,078 tokens in both
  arms; **prefill flat to +-0.5% at 483 / 1,964 / 7,777 / 31,147 / 77,811 tokens** -- so half the KV bytes buys nothing
  on the prefill side, as predicted. `probe32` pooled over two instances per arm: @32x1200 1,814 -> 1,787 (-1.5%),
  @32x300 1,612 -> 1,543 (-4.3%), @16x300 1,141 -> 1,091 (-4.4%). The kernel is *faster*, so the gap is MTP acceptance:
  accepted tokens per draft 1.604 / 1.620 (two bf16 instances) against 1.549 (fp8), i.e. **2.61 -> 2.55 tokens per step,
  -2.3%**, which is the whole 1,200-token-wave gap.
* HumanEval 164 (greedy, thinking on): **fp8 161/164 = 98.17% against bf16 159/164 = 96.95%**; McNemar b=2, c=0,
  exact p=0.50. `bench/humaneval.py` now reads `~/data/HumanEval.jsonl.gz` when present (the GSM8K cache trick -- a
  GitHub fetch died mid-gate and cost a minute of an arm). `bench/ab.sh`, the formal randomized pair, **cannot run
  here**: it drives `serve/serve.sh` and greps `docker ps` for `vllmstock`, and this box has no docker -- which is why
  `serve/native-flashnext.sh` exists.
* Net: **+87% KV (2.06x -> 3.85x at 262k) for no measurable quality cost and 1.5-4% of decode**, plus one 32 s recompile
  on the first fp8 launch (the second hit the AOT cache: 20.9 s init, 0.52 s compile). For this box's ledger -- p99
  prompt 245,792, KV usage seen at 82% under three streams -- that trade is worth taking; flip with
  `R9K_QSA_KVF8=1 KVDTYPE=fp8_e4m3`. **Box left on bf16** pending that call (the call was taken at 14:41, entry above). Scales are still the 1.0 defaults, and a
  calibration pass for the 12 main layers is the obvious place to claw the -2.3% acceptance back.
* A build.sh landmine, fixed on the way: the export check's `nm -D "$OUT" | grep -q` under `set -o pipefail` fails a
  **clean** build at random (`grep -q` closes the pipe -> nm SIGPIPE 141 -> reads as MISSING EXPORT); it killed two
  builds in a row on different symbols and passed when the same loop was re-run on the same file. Now one `nm` pass
  into a temp file.
* libr9k.so in both places now carries `r9k_qsa_attn_f8`; the bf16 production instance relaunched on it and read the
  same pool and throughput as before, so the templated kernel costs the bf16 path nothing. Previous builds kept as
  `libr9k.so.pre-f8`. Box left running **bf16 production config**.

## 2026-11-01 (end): queue knobs on wall time -- MWAITX=0 costs 3x, HWQ=2 neutral -- and shape (b) now has no marker

* Wall-based re-run, one launch per arm: default **5.3-5.5 s**, `HWQ=2` **5.2-5.7 s** (neutral), `MWAITX=0`
  **15.5-16.4 s** (~3x worse). Serve defaults stay (`HWQ=1 MWAITX=1`); n=1 per arm, but the sign matches the earlier
  attributed-sum run. The ~65 s stall in tail83's `HWQ=2` arm was therefore a shape-(a) event, not a queue-count effect.
* The elevated regime **lifted on its own** between tail90 and tail91 after ~15 consecutive launches -- it lasts hours,
  and is not a property of the code or the checkpoint. tail97: load 70.5 s (fastest today), KV 494,351, windows 5.5-
  5.7 s, GSM8K 98/100, needles 3/3, health 200, left running.
* Final correction to C16/C17: the window's major-fault count (9k then, 148-153k now) and the swap at window start
  (512-585 then, 897-938 now) are the **same in clean and elevated launches measured on wall time** -- they were
  tracking the day phase, not the regime. Shape (b) now has no marker in anything measured: gpu busy, thread state,
  io/memory pressure, swap, faults, KV pool, warm-up cost.

## 2026-11-01 (late night): re-measured on wall time the two regimes survive -- 5 s vs 13 s -- but the old magnitudes are void

* New `bench/window_wall.py`: recovers each window's wall duration from the log timestamps of the window's opening and
  closing lines, beside what the profiler printed. Validated against the direct measurement (final46: 5 s retro, 5.1-
  5.3 s reported).
* Three launches with the fixed instrument: **wall 5.0-5.1 s (tail88), 12.1-13.0 s (tail89), 12.6-13.3 s (tail90)**,
  attributed 11.5 / 12.3 / 54 s -- nesting factor 1.0x, 2.3x, 4.0x independent of the regime.
* So shape (b) (the whole pass ~2.5x slower, uniformly on all four ranks) is a real state of the box, and the regime
  comparisons of C13-C19 hold directionally. But pre-fix **magnitudes** are void: tail77's "86 s" was 16 s of wall,
  tail83's "440 s" was 67 s, and attributed is not a bound either way (tail87 printed 10-11 s against 17-18 s of
  wall, because time outside the yield-to-yield gaps is never counted).
* Instance state: default config (warm-up on, HWQ=1, MWAITX=1), latest launch tail90 up; today's cleanest load was
  final46 at 5.1 s windows / 98.3 s total, and final45 measured GSM8K 100/100 with needles 3/3.

## 2026-11-01 (night): queue/wakeup A/B negative (defaults win); the window number was an attributed sum, not wall; warm-up is not a guarantee

* `HWQ` / `MWAITX` are already parameterised in `serve/native-flashnext.sh`, so the A/B was env-only (worker
  `environ` verified). Four launches: `MWAITX=0` 10.9-12.3 s, `HWQ=2` 11.3-11.6 s + a ~65 s stall and a 74.5 s draft
  pass, `HWQ=2 MWAITX=0` 11.3-12.6 s, **control (`HWQ=1 MWAITX=1`) 9.9-10.6 s -- the fastest arm**. Both queue
  hypotheses closed; the serve script's defaults stay.
* Instrumentation correction: `stop_pwal` printed the **sum of per-class gaps**, not wall. tail83 read `440.5 s over
  33691 visits` against **67 s of wall** (timer opened 14:21:20, closed 14:22:27). Now: `t0` stamped in `start_pwal`,
  the line reads `67.4 s of wall (440.5 s attributed)`, with a warning when attributed > wall; `bench/ab_report.py`
  shows both. Clean-run numbers (4.4-5.9 s) equal wall; the tall pre-fix numbers (45-86 s) are attributed sums.
* The warm-up is not a guarantee: tail83 warmed in 204 ms and still spent essentially all 67 s inside one
  `_fold_hist` (watchdog every 2 s: `state=R wchan=0 syscall=running`). "7/7 warmed launches with no window >20 s"
  stands as those seven launches, not as a promise -- ten warmed launches since, one recurrence.
* After today: shape (a) suppressed (not eliminated) by the warm-up; shape (b) -- a stable 10-13 s floor across the
  whole pass, now in its 10th consecutive launch, vs 4.4-4.8 s earlier in the day -- has no demonstrated cause.
  Closed as causes: cross-rank queueing, the known MES bug, swap/paging (populate A/B), prestage readahead, queue
  count and MWAITX.

## 2026-11-01 (end): populate-vs-control A/B closes the swap family -- paging marks shape (b), it does not cause it

* New: `r9700_vllm/utils/pin.py` (`R9K_PWAL_PRESTAGE=populate|lock`, `R9K_PWAL_GIB=<n>`, off by default) faults in or
  pins the worker's biggest private anon mappings before the post-load window, and reports its own cost. Unit test
  `tests/test_prestage_anon.py` (16 checks: range selection, off-by-default, populate locks nothing, lock raises
  VmLck or honestly reports why, junk budget cannot raise).
* First run was mis-budgeted and the log exposed it: `0.3 s on prestage anon (populate): 1 ranges, 8.00 GiB` -- the
  evicted ~1 GiB was not in that mapping. Over the whole set (105-108 ranges, 13.25 GiB) it costs **6.8-10.1 s**.
* Result: swap at window start 911-1094 → **624-636 MiB**, majflt inside the window 151-155k → **90,723** (-40%),
  and the window **does not move** (11.2-12.9 s vs 10.4-11.5 s controls, same 24-34 s outliers in both arms).
* With io pressure measured at ~1 s inside 30-70 s windows, that is two independent readings saying the page-ins are a
  marker of the slow regime rather than its cost. The "keep it out of swap" family (populate, mlock, draining swap)
  is closed as a fix; the fault counts stay useful as a diagnostic of which regime a launch landed in.
* Left standing after today: warm-up fixes shape (a) (7/7 warmed launches with no window >20 s; clean window 5.4-5.9
  → 4.4-4.8 s). Shape (b) -- 2-3x across the whole pass, host threads in R, device idle -- has no demonstrated cause.

## 2026-11-01 (final, later): prestage A/B is a clean negative -- and the two regimes differ by what the pass pages in, 18:1

* New tool `bench/ab_report.py`: windows, major faults inside the windows, worker swap at window start, draft-pass
  wall, box disk reads, load and KV -- one line per launch, so an A/B is one command.
* Four alternating launches with `R9K_PRESTAGE_MTP=0/1/0/1`: windows (11-13 s, one 35 s), majflt (~147-152k) and
  swap (~0.9 GiB) are the same in both arms -- **prestage is not the eviction source**. Its benefit is real but
  cache-dependent: off costs 16.1 s of draft requant when the draft bytes are cold, 6.5 s when the previous launch
  already left them in the page cache. Keep it.
* Confound worth remembering on this box: **page cache carries across launches**, which can erase one arm of a paired
  test. (Same reason the notes' 5.5-8 s prestage number was measured on gated relaunches.)
* Per worker inside the window: clean 506-588 → 495-577 MiB swapped, ~12 MiB paged in, ~2.2k major faults; elevated
  969-992 → 590-621 MiB, ~380 MiB paged in, ~40k major faults, 1.5 GiB swap traffic box-wide. 18:1, no overlap.
* But the box state at the end of the weight stream is identical in both regimes (self rss/swap 12.0/0.9 GiB,
  SwapFree 2 GiB): same amount swapped, different amount touched. So the operational lever (drain swap) is weak and
  the process-side ones (keep the pass's destination pages resident; shrink the anon alive during the stream) remain,
  with `mlock` available up to `ulimit -l` = 11.7 GiB per process.

## 2026-11-01 (final): the diffuse slow window IS a paging event -- major faults inside the window separate the shapes 17:1

* final44 launched with the warm-up on and still took 11.1-11.8 s a rank, and its census said why: each worker went
  `VmSwap 921 -> 586 MiB` with **+38,158 major faults** in an 11 s window; the box swapped in ~1.1 GiB while the four
  ranks did it.
* Summing major faults over the four workers' windows: clean warmed runs **8.8-9.0k**; shape (a) (tail57, one 73.8 s
  visit) **16.1k**; diffuse slow runs (tail66/67/68, final44) **150-162k**. No overlap, 17:1.
* Correction to C13 ("the swap reading is dead"): I had compared VmSwap *at window start* (578 vs 616 MiB, no
  separation). The separating variable is faults *during* the window. Shape (a) is not a paging event; shape (b) is.
* That makes the lever host-memory peak, not the device: workers end the load with 0.6-0.9 GiB of themselves in an
  8 GiB swap that is never drained between launches, and the elevated launches all came late in a day of ~25
  relaunches. Next candidates: release host staging right after each tensor is placed; keep the post-load
  destination buffers resident; drain/enlarge swap between heavy launches.

## 2026-11-01 (later night): two shapes of tail; the device is idle while the host waits; a heartbeat does not help

* New instrument: `gpu_busy_percent` from sysfs (no root, no ROCm tooling) sampled at 0.25 s -- the window census now
  prints it as a sparkline -- and `_fold_hist` now splits *issuing* (host) from *waiting* (device) via
  `prof.slow_note`; `slowcall` also samples gpu busy around a slow call.
* **tail66 (74.9 / 21.8 / 12.8 / 11.8 s) is a second shape**: no single call over 2 s, every class inflated together
  (`R9kQwenGatedDeltaNetAttention` 41.14 s/864 visits, `layers.26.linear_attn` 24.29 s), and the max-over-GPUs busy
  trace `####+-.....#...#-.###..........--###.-` with **1.25-2.5 s stretches where no GPU is computing** while the
  rank sits in state R. So: (a) one enormous visit at the first guarded tensor (tail48/57, warm-up suppresses it),
  (b) the whole pass 4-8x slower (tail66).
* `R9K_PWAL_BEAT=1` (one trivial kernel every 50 ms on the rank's device during the window, warm-up off) is a
  **negative**: tail67 11.6-12.2 s, tail68 50.4 / 38.6 / 35.6 / 25.7 s. Caveat that matters: the beat thread managed
  83-136 submissions in a 25-50 s window (~4 Hz, not 20) because the load thread holds the GIL inside C calls.
* The swap column now separates the elevated instances: 982 / 931 / 931 MiB of each worker swapped at window start for
  tail66/67/68 versus 512-607 MiB in the clean warmed rounds -- though tail57 (578 MiB, 73.8 s) blocks calling it a
  rule. Swap on this box is 8 GiB and is never drained between launches, and the elevated launches came late in a day
  of relaunches.

## 2026-11-01 (end of day): warming the post-load path took the tail away -- 7 launches, no window over 20 s, and the clean window fell from 5.6 s to 4.6 s a rank

* The swap reading died with the KV one: a worker starts its post-load window with 512-750 MiB of itself in swap
  whether the instance is clean (tail61-65: 512-607 MiB) or tailed (tail57, 73.8 s window: 578 MiB).
* What survived is *where* the tail sits -- the visit of the first tensor the fold guard runs -- so the experiment
  was `warm_post_load()` (kernels/moe.py, `R9K_PWAL_WARM=1`, default on): `permute_fragments`, `pack_scales`,
  `_fold_hist`, `quantize_rows_fp8` once on coin-sized dummies, 48-184 ms, right before `start_pwal`.
* Seven warmed launches: per-rank windows **4.4-4.8 s** (was 5.4-5.9 -- the clean instances were paying ~1 s of
  first-call cost too), outliers 7.5 s and 8.4 s once each, and **no window over 20 s** where 5 of the previous 13
  launches had carried a 43-74 s single visit. Load totals 71.1-105.2 s, new best 71.07 s.
* Getting there cost four launches: `torch.device("cuda1")` is not a valid device string and the exception took the
  workers down. The call is now `try/except`-guarded (a warm-up may only ever cost time) and tested exactly as the
  server calls it (`warm_post_load()` no-argument on the current CUDA device).
* Instance up: tail65, load 71.1 s, KV **495,160**, GSM8K 98/100, needles 3/3.
* Still unexplained: what the device was doing while the host waited -- one call queueing 44-69 s on a 26 KB tensor
  with two host threads spinning (`syscall running`, state R), no swap, no MES timeout, no memory pressure, no
  cross-rank queueing. The warm-up removes the waiting, not the cause.

## 2026-11-01 (night): the known MES bug is ruled out by the kernel log; the census now samples swap, and `slowcall` says the slow calls are waits, not work

* We are in group `adm`, so `/var/log/kern.log` is readable: **zero `MES(...) failed to respond` lines today**, on
  launches that carried 45-69 s tails. Notable because this is the known-MES box (`notes/mes-timeouts.md`) and this
  guest does *not* have the documented `amdgpu.mes_log_enable=1` (cmdline is only `amdgpu.ppfeaturemask=0xffffffff`,
  `/sys/module/amdgpu/parameters/mes_log_enable` = 0) -- worth fixing anyway, but it is not the tail.
* What the kernel does log near the tails: `workqueue: amdgpu_amdkfd_restore_userptr_worker [amdgpu] hogged CPU for
  >10000us` at 07:35 (1027 times) and 11:01 (2051 times) -- the driver's worker that re-maps GPU visibility of
  **host** pages. That cost lands in a kworker, not in our process, which matches the census shape (our sys CPU
  ~2%, no PSI-memory stall, thread `R` with `wchan (running)`).
* tail51 (68.7 / 45.5 / 5.6 / 6.3 s, load 98.5 s, KV 476,551) is where `prof.slowcall` fired, and the named calls
  are far too small to be work: `_fold_hist` 4.82-5.45 s / 9.8-11.1 CPU-s on a **26 KB** tensor
  (`cuda:N uint8 [1, 81, 320]`), `quantize_rows_fp8` 7.11 s on `[10240, 320]`, 6.19 s on another rank, 2.28 s on
  `[336, 10240]`. A handful of launches each, while the same code costs 0.4 s on the rank next to it.
* Census now also samples our `VmSwap`, our minor/major faults and the box's `pswpin`/`pswpout` across the window,
  and `/proc/<tid>/syscall` (self-tested: `clock_nanosleep` for a sleeping thread, `(user)` for one running).
  The box: 8 GiB swap, ~4 GiB in use, 273 M pages swapped in and 318 M out over its life.

## 2026-11-01 (evening, latest): the serialization A/B kills device queueing as the explanation too; the instrument now reaches the single call

* `R9K_PWAL_SERIAL=1` (cross-rank file lock across the measured window): tail46 and tail47 gave 5.5-5.9 s windows on
  every rank with queue waits 0 / 11-15 / 14-19 / 23 s and the same 75.5 s load total -- looked like a fix.
* tail48 refuted it: **48.2 s window with queue wait 0.00 s** -- that rank had the device to itself from the start,
  and the three ranks that ran after it, also alone, took 5.8-5.9 s. So the tail is not ranks queueing behind
  each other. (Knob stays, default off; it is a measured negative.)
* What that window was: `MergedColumnParallelLinear` 44.24 s wall / **88.11 s user CPU**, one visit
  `layers.0.mlp.shared_expert.gate_up_proj` 43.57 s / 87.08 s, sys CPU 0.8 s for the whole window, faults ~13 k
  major, census R=27 of 29 samples with `wchan (running)`, memory pressure 0.0 s stalled. User CPU exceeding wall
  means several threads burned it. Module-level attribution has hit its limit here.
* New instrument: `prof.slowcall` logs any single load-path call over `R9K_SLOWCALL` (default 2 s) with the
  arguments' device/dtype/shape/layout, wall, user/sys CPU and minor/major faults -- wrapped around
  `permute_fragments`, `pack_scales`, `_fold_hist`, `quantize_rows_fp8`. The window census now also samples
  `/proc/<tid>/syscall` (`(user)` when the thread is executing, otherwise the syscall it is parked in), which tells
  spinning from sleeping.
* `tests/test_slowcall.py` covers it (fires past the threshold, names the call, carries the CPU/fault split, flags
  NONCONTIG, silent when fast, no wrapper at all when profiling is off) and caught a real bug: `prof` already had a
  module-level `_SLOW` (copy outliers), so the new counter shadowed it -- renamed `_SLOWCALL`.

## 2026-11-01 (evening, later): caught a slow window with its brackets -- the slow rank burns user CPU, not system CPU, and hardly faults

* New tool `bench/catch_tail.sh`: relaunch until a window >= 8 s appears. Round 1 clean (5.5 s x4), round 2 (tail45)
  landed a bad one: **5.7 / 21.8 / 51.0 / 57.0 s**, KV pool 468,865 (smallest so far), and it served fine anyway
  (GSM8K 97/100, needles 3/3).
* The rusage brackets settle what the slow ranks are doing: for `GatedResidual`, wall 0.44 s / user 0.72 s on the
  fast rank against wall 50.66 s / user 53.06 s on the slowest, sys CPU ~2% of it, ~2.5 k major faults (10 MiB) and
  **0.0 s** of memory pressure across the whole 57 s window. Single visits: `layers.41.mlp_hyper_connection`
  30.76 s (31.96 s user), `layers.5.attn_hyper_connection` 32.52 s (33.85 s user).
* So the tail is not a slow module, not the memory side, not the disk: same ops, same tensors, 75x the user CPU --
  the thread is busy-waiting on a device that three other ranks are driving (the serve script even sets
  `HSA_ENABLE_MWAITX=1`, and the census sees state R with `wchan (running)`, leaf frames in `quantize_rows_fp8`).
* Next: `R9K_PWAL_SERIAL=1` (added) holds a cross-rank file lock across exactly that window, so the four ranks do
  their post-load repacks one at a time. If the reading is right, the 50 s windows and the shrunken KV pool go away;
  the cost, if any, is a few seconds on the load total.

## 2026-11-01 (later still): a slow post-load window is CPU work, not a wait; census + per-visit rusage are now in place, and the tail/KV-pool separation is complete

* `prof`: `start_pwsamp()` samples the load thread across the whole post-load window (leaf frame, kernel state R/S/D,
  wchan, PSI memory+io) and `stop_pwal` prints the census; every module visit is now also bracketed by `getrusage`
  (wall, user CPU, sys CPU, minor/major faults), because a wall-time column cannot tell a slow module from a
  starved one.
* final42, where all four ranks were slow (13.1 / 25.5 / 36.5 / 41.7 s): thread state **R=14-21, D=3-10, S=1**,
  memory pressure **0.0 s stalled** in a 36 s window (avg10 0.6%), io pressure 0.7 s. Not a lock, not reclaim, not
  the disk -- the window is CPU work, done slowly because four ranks do it at once on a box with 6 GiB swapped out.
* Instrument caveats worth keeping: the Python sampler only reaches 0.7-1.1 Hz, not the 4 Hz it asks for, because it
  needs the GIL that the load thread holds inside C calls; and rusage is per-process, so a class can show more CPU
  than wall when that worker's other threads were busy too.
* final43, clean: windows 5.4-5.7 s on all four ranks, load 77.7 s, `kernels fold_stats` 0.4 s (96 calls),
  **KV pool 497,588 tokens** (largest yet), GSM8K 97.0% (194/200), needles 3/3.
* Tail vs KV pool now separates perfectly over ten instances: every instance with a tail has a pool of 490,306 or
  fewer, every clean one 492,329 or more (468,865 costs 5.7% of concurrency against 497,588).
* Still missing: a tailed instance *with* the rusage brackets -- final41/42 had the attribution but not the brackets.
  4 of the last 10 launches had a tail, so the catch is a couple of relaunches away.

## 2026-11-01 (late): the fold guard cost 8 s a rank -- cut to 0.4 s; the load total and the post-load tail did not move

* `kernels/moe.py`: the guard now refuses short-K tensors from the shape (`fold_k`, no bytes touched: 98 of the 196
  tensors a rank), samples at most `R9K_FOLD_SAMPLE`=1 M blocks per tensor (uniform over the (block, column) plane,
  fixed seed so four ranks sample the same blocks), re-walks a tensor exactly when the sample lands within
  `R9K_FOLD_MARGIN` (3x) of a threshold or sees any flushed block, and syncs once per tensor instead of per chunk.
* Measured in the load, not in isolation: `kernels fold_stats` **8.1-8.5 s a rank over 192 calls (warm4) -> 0.4 s
  over 96 calls (final41)**. Decisions are unchanged (392 fold ON / 392 fold OFF both ways; sampled mean_d 0.47-0.48
  == exact; sampled max_d 4-5 is a lower bound on 5-6 by construction), GSM8K 98/100, needles 3/3.
* New `tests/test_fold_sample.py` killed my first design: sampling whole columns (8% of them) can walk past a
  channel that is entirely bad. The guard counts blocks, so the sample has to be over blocks.
* **The 8 s was real work and it was not on the critical path**: final41 loaded in 79.31 s (inside the day's
  67.9-115.3 s spread) and two ranks still spent 29.4 s / 55.9 s in the post-load window that others finish in 4.6 s.
* The tail changed shape, which is the clue for the next step: warm4's slow rank was stuck in one call (the watchdog
  caught `fold_stats`); final41's slowest rank spread 48.6 s over 2460 `GatedResidual` visits, 35.4 s of it inside a
  single `layers.29.mlp_hyper_connection` visit -- and the watchdog saw nothing, because it only reports a frame it
  finds unchanged for 2 s. Next instrument: sample the load thread's frame **and kernel state** across the whole
  window, and count each module's slow visits directly.
* Tail instances hold the three smallest KV pools measured (468,865 / 478,979 / 490,306 against 492,329+ for every
  clean instance): the slow window and a smaller pool look like one memory-side event, and a smaller pool is itself
  a 5% concurrency loss.

## 2026-11-01 (evening): the post-load window is named per module; warm-ups kill one term, the fold guard is the cost that is really there

* New tool: `prof.start_pwal()` / `stop_pwal()` attributes the stream -> draft window to module classes by timing the
  gaps between `named_modules` yields, so blocked time counts and no stock loop body is duplicated.
* It named one term exactly: `layers.0.attn_hyper_connection` costs 1.57 s in a fast rank and **17.38 s** in a slow
  one, with the watchdog catching `fp8.py:65 quantize_rows_fp8` at `wchan=lock_mm_and_find_vma`. The quantization is
  not slow (isolated: 16 MiB of bf16 in 2 ms, 16-22 GiB/s) -- the *first call in the process* costs 0.107 s.
  `kernels/fp8.py:warmup()` + `quant/nvfp4.py:warmup()` at stream start (`R9K_QUANT_WARMUP=1`, 0.33 s, 512x512) remove it.
* **warm4 falsified the first-call explanation for the window as a whole** (post-load 5.2 / 6.2 s on two ranks and
  **59.0 / 61.8 s** on two others, `layers.0.mlp.shared_expert.gate_up_proj` 54.49 s against 1.89 s on a fast rank,
  watchdog `moe.py:202 fold_stats` `state=R`), and the next instance (final40) was clean: 4.9 / 4.9 / 5.0 / 5.1 s on
  four ranks, 75.51 s load, GSM8K 98/100, needles 3/3. Of the 8 instances that measured the window, 2 had a tail --
  and those two (warm4, pwal3) also have the two smallest KV pools ever recorded (490,306 / 468,865 against
  492,329+): tail and KV shrink look like the same memory-side event, not a fixed cost.
* What *is* there every time, and big: the fold guard. `serve/native-flashnext.sh` sets `R9K_FOLD:=1`, so every rank
  histograms every MXFP4 tensor's packed scales: 196 tensors, **965 M blocks a rank** (965 MiB of int8) for a
  statistical guard, and warm4's timer dump charges it **8.1-8.5 s a rank** (`load timer: kernels fold_stats`) --
  114 M blocks/s, i.e. host-side traffic, not device work. It also sits inside the post-load window for the dense
  tensors, which is where warm4's slow ranks were stuck. `prof.report()` now runs at target-end and post-load-end,
  because a server never exits and the timer's 15 s dump only fires if a timed call lands on it (final40 printed none).
* Load totals across the day, one config: **67.85 / 71.89 / 72.61 / 73.03 / 74.63 / 75.18 / 75.51 / 75.80 / 79.71 /
  80.76 / 90.77 / 96.82 / 115.30 s**; serving numbers never moved with them.

## 2026-11-01 (later still): the load's read path is fault-bound but readahead-proof; the leftover is draft-pass variance

* `bench/mmapread_probe.py` (cold shard, `fadvise(DONTNEED)` to reset, no root needed): streaming `pread` **3.49 GiB/s**,
  mmap touching one byte per 4 KiB page **0.57 GiB/s** (= exactly what a rank's expert placement measures), 4 processes
  on the same file **1.01 GiB/s each**, and `fadvise(WILLNEED)` + the same mmap pass **4.12 GiB/s**. mmap faults get no
  readahead; queueing it works.
* **C5: stream-ahead readahead in the loader** (`utils/prestage.py`, `R9K_PRESTAGE_AHEAD=<bytes>`: while the consumer
  places tensor *i*, WILLNEED the next N bytes of tensors in header order; 0.15 s of fadvise wall a rank) --
  **no effect on the target pass**: place 47.6 s / experts 35.8 s / 115.5 GiB read vs 47.8 / 35.9 / 115.2 without
  (ah256 vs ah0). The four ranks already share the cache, and 110.6 GiB does not fit in it, so pages are read once,
  used once, evicted: nothing left to overlap. Off by default.
* The remaining load variance is in **two windows, neither of them the requant**: seven instances of one config give
  **67.85 / 72.61 / 73.03 / 74.63 / 75.80 / 79.71 / 96.82 / 115.30 s**. Snaps now separate them (`target end` ->
  `prestage mtp end` -> `draft start` -> `draft end`): the **post-load window** (the target model's
  `process_weights_after_loading` plus the draft loader opening its shards) is **6 s in final38 and 31 s in final39**,
  and the **draft generator** is 5-7 s usually, 48.3 s in ah0's rank0. Around the draft window (per rank): `majflt`
  +45k to +52k, `Cached` +6 GiB, `VmSwap` 0.8 -> 0.5 GiB, but **disk reads only +0.9 to +3.2 GiB** -- same faults, 135
  us each in the fast arm and 1.0 ms each in the slow one. Stack samples (685 for a 6 s pass = 1.4 s CPU) land on
  `_load_w13` 13%, `cuda.synchronize` 11.7%, `_load_w2` 10.7%, `mtp_weights` 9.9%, `_encode` 8%: the pass also places
  the draft's own 512-expert MoE, and the slow cases are blocked rather than computed.
* Added counters to make that visible: `prof.snap` now reports per-process `VmRSS/VmSwap/VmPin`, and the stack sampler
  can be reset and re-armed per window (`prof.reset_stacks`, `draft start`/`draft end` snaps).
* Load path stands at **67.9-79.7 s typical** (from 174.8/189.6 s before late pinning), with the draft pass as the one
  unbounded term. Both new knobs stay in the tree, off by default, with their measurements in the notes.
* **C6, neutral**: `R9K_CACHE_WINDOW=<MiB>` hands consumed shard bytes back with `fadvise(DONTNEED)` as the stream
  passes (8 GiB window: load 75.8 s, `rankRdTot` 113.8 GiB so nobody is re-reading, `maxSwap` still 8191 MiB). The
  swap use is structural, not a squeeze: no cgroup limit anywhere (`memory.max`/`high`/`swap.max` = `max` on
  `session-2.scope`), `memory.current` 85.6 GiB of 93.8 GiB, and the anon set is the four 9.69 GiB PLE tables plus
  ~6 GiB a rank before a single checkpoint page is mapped. Same reason `vm.swappiness=1` was inert.
* Online instance now: **final39** -- load **96.82 s** (place 47.3, experts 36.6, draft 5.0-6.4 s: the extra 31 s went
  into the post-load window), GSM8K **99/100**, needles 3/3, KV 496,374, probes `@32 x300` 1577.2-1593.1,
  `@32 x1200` 1818.4-1825.4, `@16 x300` 1110.0-1129.5 tok/s, 300 W median at busy==100. The instance before it,
  **final38** (identical config), loaded in **71.89 s** with GSM8K **100/100** and probes `@32 x300` 1578.3-1621.9,
  `@32 x1200` 1807.5-1816.0, `@16 x300` 1117.4-1151.3. Serving is insensitive to which way the load went.

## 2026-11-01 (later): the experts term is source-side reads (C4, negative); throughput baseline locked on the online instance

* Anatomy of what is left after late pinning (final31, per rank): the experts term is **148,998 `copy_` calls,
  27.3 GiB, 45.7 s** -- 98,496 x 106 KiB contiguous (19.3 s, w13), 49,248 x 51 KiB strided (15.2 s, w2), 64 x 151 MiB
  cpu->cpu PLE fills (8.5 s). `bench/expertcopy_probe.py` made that look like per-transfer overhead (512 resident
  106 KiB slices: 75 ms; one combined copy: 10.7 ms; `tensor.to(device)`: 0.3 ms).
* **C4 built and measured negative**: `r9700_vllm/moe/h2dbatch.py` stages runs of adjacent destination slices into a
  64 MiB pinned buffer and issues one `copy_` per run (`tests/test_moe_h2dbatch.py`: byte-exact for the per-expert,
  strided and interleaved patterns, 2.8x faster on resident data). On the real load it is **slower: 80.6 s vs
  73.6 s** (h2db36 vs final31). Its own counters: 98,496 slices -> 98,496 copies (runs never join, stock writes the
  w1 and w3 halves alternately into two storages), and **the host staging read alone costs 18.35 s / 10.0 GiB
  (0.56 GB/s) against 19.56 s (0.51 GB/s) for the copies it replaces** -- same bytes at the same rate, so the cost is
  touching the mmap'd pages, and the H2D part is only ~5 s. Left in the tree off by default (`R9K_MOE_BATCH_H2D=0`).
* Where the remaining ~35 s actually is: a rank *walks* ~110 GiB of shard pages to place 27.3 GiB of weights. No
  amount of copy batching, pinning or faster transfer paths touches that; only reading fewer bytes would.
* **Throughput baseline now recorded as a fixed protocol** on the instance left online (final37, same config as
  final31): `@32 x300` **1572.4-1575.1**, `@32 x1200` **1818.6-1831.8**, `@16 x300` **1112.6-1126.3** tok/s, 299 W
  median per card at busy==100. Three-sample spread on `@32 x300` is 2.7 tok/s (0.2%). Older MTP=2 aggregates quoted
  without a protocol should be read as soft.
* Still open: E1 prestage was *not* re-tested A/B under late pinning (its headroom is now small -- the whole draft
  pass is already 5.4-6.7 s, of which the MXFP4 search is 1.5-1.7 s); a single-pass expert reader; `MemorySwapMax`
  capping if more swap is ever wanted; `@16 x200` vs the recorded 1,265.

## 2026-11-01 (end of day): swap capacity A/B says *less* swap is better for this load; sampler tooling fixed

* Added a 24 GiB swapfile (SwapTotal 8 -> 32 GiB) and re-ran late pinning both ways: **cold 68.7 -> 103.2 s, warm
  72.3/72.5 -> 98.3 s**, `hipHostRegister` 5-8 s -> **20-30 s**, swap peak 8.2 GiB (saturated) -> 25-28 GiB. With room
  to move, the kernel parks the not-yet-registered PLE tables in swap and that debt is paid back at registration.
  So the 8 GiB swap was acting as a **budget that protects the load**, and my recommendation to widen it was wrong --
  measured twice, then reverted (`swapoff`, `rm`, fstab cleaned) and confirmed: **final31 = 73.6 s, `>1s` copies 0**.
* Cleaner metric came out of it: the four workers' total disk reads per load, **109.5-118.6 GiB at 8 GiB swap** against
  **134.8-137.5 GiB at 32 GiB swap** (+17%) -- same stalls either way (zero), just more page-in work.
* Two of my own measurement bugs found and fixed, both of which had produced numbers I quoted:
  13 `loadsample.sh` instances were still running (one continuous CSV, so a whole-file `max swap` reported 25 GiB on
  an 8 GiB box), and samplers started before the diskstats partition fix double-count `disk_rd_mbs`/`disk_wr_mbs`.
  New: `bench/loadstats.py` (per-instance windows taken from the launch log's own timestamps) and a `flock` in
  `bench/loadsample.sh` so only one sampler can run. The corrected per-instance table is in `notes/native-4xr9700.md`.
* Open for next time: E1 prestage re-tested under late pinning; the draft pass's 3.4-48.8 s swing (data movement into
  the draft, the search itself is 1.5-1.7 s of it); and if more swap is ever wanted, cap vLLM's own swap budget
  (e.g. `MemorySwapMax=8G` in a slice) rather than widening what the load may use.

## 2026-11-01 (later still): cold-cache A/B done, late pinning is now the default

* The operator ran `sync; echo 3 > /proc/sys/vm/drop_caches` before two launches (`Cached` 0.06 GiB both times), which
  is the cleanest pair so far: **cold + late pin 68.7 s** (place 47.3, experts 37.7, `>1s` copies **0**) vs **cold +
  early pin 114.5 s** (place 96.6, experts 55.7, 1-5 copies of 50.8-52.5 s each).
* Late pinning does **not** reduce I/O: duplicate reads are 23.6-26.7 GiB per rank in both. It removes the waiting.
  And the shared page cache already holds read amplification at **~1.07x** of the checkpoint for four ranks (118.7 GiB
  read by the four workers on a cold cache) -- so the worry that each rank streams the whole file four times was wrong.
* Cold+early (114.5 s) is far better than warm+early (174.8-189.6 s): the pathological window needs a **full** page
  cache to show up, because that is when the kernel starts swapping instead of shedding cache.
* `R9K_PLE_PIN_LATE` now defaults to on (five instances with it: 68.7 / 72.3 / 72.5 / 118.6 s; three without:
  114.5 / 174.8 / 189.6 s). Final serving instance late27: load **72.27 s**, GSM8K 99/100, needles 3/3, KV 496,374
  tokens, `@32 x300` 1543-1590, `@32 x1200` 1810-1815, `@16 x300` 1107-1132 tok/s, 299 W median at busy==100.
* Open: swap headroom (late pinning uses all 8 GiB of swap during the load; needs root), E1 prestage re-tested under
  late pinning, and the draft pass's 3.4-48.8 s swing (data movement into the draft, not the requant -- the search is
  1.5 s of it).

## 2026-11-01 (later): the load's stalls are swap I/O -- found by nine instrumented launches

* `Tensor.copy_` accounting (`R9K_LOAD_COPIES=1`): the target pass makes **148,998 copy_ calls per rank writing
  27.3 GiB and spending 136-163 s**. By call size: 148k calls under 1 ms cost only 22-30 s, while **12-25 calls
  over 1 s cost 83-121 s**. So the load's variance is ~15 pathological copies, each 0.2 MiB, of
  `mlp.experts.{gate_up,down}_proj_packed` inside stock `_load_w13`/`_load_w2`, slowest 59 s, and all four ranks
  stall in the same window.
* Hypotheses killed one instrument at a time: **PCIe** (`bench/h2d_probe.py`: 0.2 MiB pageable copy = 0.049 ms;
  16-64 MiB = 27-47 GiB/s; pinned only 1.3-1.6x better), **the pins themselves** (`bench/svm_probe.py`: 4 processes
  each registering 9.69 GiB and hammering 0.2 MiB copies give p99 0.03 ms, no stalls), and **page-ins by the
  copying thread** (getrusage around the slow calls: minflt+4..9, majflt+2..26 for a 55 s call).
* Where it waits (watchdog on `/proc/self/task/<pid>/{wchan,syscall,stat}`): `state=D wchan=lock_mm_and_find_vma`,
  `state=D wchan=folio_wait_bit_common`, `state=S wchan=futex_do_wait syscall=202`. And who owns the device then
  (sampler with PSI-io + diskstats + per-worker `read_bytes`): **gpu busy 3 %, disk 1.2-1.5 GB/s read AND
  1.2-1.5 GB/s write, worker read_bytes 0 MiB/s, swap 5611->5975 MiB, PSI io 5-7 %**. The kernel's swap churn owns
  the NVMe; vLLM is not the reader.
* Mechanism: 4 x 9.69 GiB of registered (unswappable) host memory allocated before the first byte is read + ~15 GiB
  anon + ~46 GiB cache on a 93.8 GiB box with 8.39 GiB swap -> the kernel writes ~5.9 GiB of swap at ~1.4 GB/s on
  the device that holds the mmap'd checkpoint, and every copy that touches a non-resident page queues behind it,
  GPU idle. That is the 115.7-186.1 s spread (inst18/19/20: 181.7 / 151.3 / 174.5 s).
* Ran the two candidates: **C1 `vm.swappiness=1` is inert** (the kernel still wrote the same 5.9 GiB of swap and the
  `>1s` bucket was unchanged) -- which makes sense, because the 38.8 GiB is *registered* and therefore not in the
  anon-reclaim set the knob steers. **C2 late pinning works**: with `R9K_PLE_PIN_LATE=1` the PLE table is allocated as
  ordinary memory, filled during the stream and registered at the end of the target stream (before the first
  forward); the `>1s` copy bucket goes to **zero on all four ranks**, place falls from 151.6-169.9 s to **47-49 s**,
  experts from 113-138 s to **37.3-37.4 s**, and the total from 174.8/189.6 s to **118.6/72.5 s** (late23/late24).
  Cost: `hipHostRegister` 4.7 -> 6.4-8.2 s and swap now peaks at all 8 GiB during the load (the tables became the
  pressure valve); MemAvailable trough 36.2 GiB, no OOM, no new blocked tasks.
* Runtime after late pinning is unchanged: KV cache 496,374 tokens (late23's 487,879 was drift -- that closes the
  earlier -1.2% question), GSM8K 98/100, needles 3/3, `@32 x300` 1574, `@32 x1200` 1804, `@16 x300` 1134 tok/s,
  300 W median at busy==100.
* With stalls gone the load is legible at last: place 47.1 s is experts 37.4 s + other 8.6 s, and the copy histogram is
  147,366 calls under 1 ms for 29.7 s -- real work, no anomaly. The draft pass's scary-looking "44.1 s bf16->MXFP4
  requant" is **not** the requant: split, the search is 1.5-1.6 s over 5 calls and to-GPU 0.7 s; a standalone
  `quantize_mxfp4_search` (`bench/mtpquant_probe.py`) runs at 4.3 GiB-of-bf16/s plus 0.39 s of launch. The 44 s was
  `x.cuda()` paging the draft's bf16 in from a swapping device. Last big swing left = the draft pass (3.5-48.8 s).
* Corrected my own earlier number: the sampler summed `/proc/diskstats` whole-disk *and* partition lines, so the
  "1.2-1.5 GB/s read + write" in the stall window was 2x too high (~0.6-0.75 GB/s each). Conclusion unaffected.
* Next: re-test **E1 prestage with late pinning** (it failed before because the MTP span was evicted during
  consumption; now the tables are swappable so the readahead may survive), a **cold-cache** A/B
  (`sync; echo 3 > /proc/sys/vm/drop_caches`, needs root) for the first-boot number, and more swap (needs root),
  since late pinning now intentionally uses all 8 GiB.
* Tools fixed along the way, each of which had silently wrong output before: the stack sampler charged the handler
  itself as the leaf of every sample (`sys._current_frames()` returns the handler's frame for the current thread)
  and charged idle threads to the denominator; `sys._current_frames()` keys are CPython ids (pthread handles), not
  kernel TIDs, so `/proc/self/task/<tid>/` needs `os.getpid()` instead; and the footprint sampler's
  `pgrep -f 'VLLM::Worker'` child matched `serve/native-flashnext.sh`'s own guard and refused a launch that had
  just passed the gate.

## 2026-11-01 (same day, five gated relaunches): what the 116-155 s cold start actually is

* Reloading five times behind `bench/gate.sh` (new: `bench/relaunch.sh` prints the fingerprints -- load seconds,
  place breakdown, KV pool, spec config, RAM/swap at the moment the load finishes) shows the **total is the wrong
  metric**: **115.7 / 120.6 / 138.6 / 149.9 / 155.3 s**, built from a target `experts` term that swings
  **32.8-108.0 s** and a draft pass that swings **5.5-76 s**, in opposite directions. So my "page cache is worth
  4%" from earlier today is corrected: cache state moves the parts hard and the sum little.
* New counters did it, not the old ones. `snap()` now logs `Cached` / `SwapFree` / `majflt` / `minflt` /
  **`rchar` / `read_bytes`** / PSI-memory at each phase boundary. The decisive number: **23.0-28.3 GiB of block
  reads a rank during the target pass** (0.0 at start) with `Cached` pinned at 38-46 GiB against a 110.6 GiB
  checkpoint -- the ranks stream the whole file, take ~50 GiB of anon (9.7 GiB of it the pinned int6 PLE table),
  and evict the pages they are streaming, so ~1/4 of it comes off disk again.
* Two counters that would have misled: `rchar` is only 0.5-0.6 GiB (safetensors is mmap, not `read()`), and
  `majflt` ~130-150k a rank is ~0.5 GiB of anon swap-in, not checkpoint traffic -- page-cache misses never show up
  there. PSI memory grows just **~2.7 s** across the entire target pass, so the seconds are not spent stalled in
  reclaim, even though every load writes **~5 GiB of swap** (`SwapFree` 7 -> 2 GiB): the same precondition that
  made the amdgpu SVM deadlock reachable.
* The draft pass iterates **2182 tensors / 110.6 GiB a rank** to consume a few GiB of bf16 MTP MLP weights and
  requantize them; 5-8 s when those bytes are still resident, 60-76 s when they are not, and inside one instance
  one rank paid 5.2 s while another paid 47.7 s (they race). KV pool 496,374 tok is identical on every instance,
  so it is not the variable.
* **E1 (prestage the draft's byte ranges) is negative, and died on its own instrument.** The draft consumes 34 MTP
  tensors, 4.86 GiB, sitting in **3 contiguous spans** (4.82 GiB of them one span of model-00020.safetensors:
  `mtp.layers.0.mlp.experts.gate_up_proj` 3.2 GiB bf16 + `down_proj` 1.6 GiB). fadvise(WILLNEED) queues them in
  0.01 s but reads **0.00 GiB on all four ranks** -- at the end of the target pass the span is still resident. The
  duplicate read still shows up per rank *during* the draft window (+1.7 to +5.7 GiB): pages are evicted while they
  are being consumed, which readahead cannot prevent. Control vs treatment: 149.8 s (off) vs 186.1 / 161.9 s (on),
  and the difference lives in the target pass, not the draft. Code stays as a 0.01 s probe.
* **Footprint time series (`bench/loadsample.sh`) says where the pressure comes from:** worker RSS goes 6.3 ->
  45.9 GiB in 5 s **before the checkpoint stream opens** (the pinned int6 PLE table, 9.7 GiB a rank, is allocated
  at model construction), the pass then runs at `MemAvailable` 40.8 GiB with `Cached` 37-41 GiB (available ~=
  cache), RSS oscillates 48-68 GiB (up to 19.8 GiB a worker against 12.5 steady), and swap jumps 0.1 -> 5.4 GiB
  inside 11 s mid-pass. `Mlocked` shows only 25.9 MiB -- `hipHostRegister` is not counted there.
* **That rules the usual suspects out for the experts term** (92-121 s per rank for 59.8 GiB = 0.5-0.65 GB/s):
  re-reads are only 0.2 GB/s a rank against a 5.8 GB/s raw checkpoint stream; H2D is 19.65 GiB a rank (~4-10 s for
  all four); PSI stall is ~2.7 s across the whole pass. What is left is the transformation: 59.8 GiB in, 19.65 GiB
  of model out per rank, ~3x rewriting at ~0.5 GB/s. Next measurement: split the experts `place` into plugin repack
  (`_prep_mxfp4`/fold) vs stock `weight_loader` copy vs H2D submit -- if it is the repack, the fix is a faster
  repack (or one fold per checkpoint instead of one per rank), not an I/O fix.

## 2026-11-01 (after the reboot): the throughput baseline is protocol-defined; the envelope is a flat 300.0 W

* Post-reboot health on the same tree: **GSM8K 98/100, needles 3/3**, single-stream decode 169.4-181.1 (median
  **171.6** against the recorded 170.4), speculative config confirmed live as `{'method': 'mtp',
  'num_speculative_tokens': 2}` with acceptance **1.49 accepted a draft** (pos0 82.6%, pos1 66.3%), GPU KV cache
  **496,374 tok** / 15.15x (recorded 502,442, -1.2%). Boot-cold load 115.67 s vs warm 120.60 s (page cache: 4%).
* A 4-5% @16 gap against the recorded 1,265 sent me into the protocol, which was never pinned down. With
  `bench/probe32.sh` it repeats to **+-0.5%**, but **`max_tokens` moves @32 aggregate by 15%**: 32 in flight,
  300-token waves -> **1,598** (1,586.5-1,600.1, 51.8 a stream); 1,200-token waves -> **1,823** (58.1); 16 in
  flight, 300-token -> **1,107** (72.6); 16 in flight at `bench.py run`'s 200 tokens -> **1,198** (79.1). The
  length effect even flips sign between 16 and 32 streams. Only the last row is comparable with the recorded
  1,265: **-5%**, and most of the scary -9% at @32 is the recorded table behaving like long waves.
* The power envelope is not the cause, and suspecting it was a measurement error of my own: sampling power around a
  6 s wave read 258 W median. Gated on `gpu_busy_percent == 100` (140 samples) all four cards sit flat at
  **299 W median / 300.0 W max**, 44-47 C, busy 100 -- the same ~300 W ceiling the note records -- and this guest
  has no powercap installed at all (`/usr/local/sbin` empty, no `r9700-*` units; the 225 W cap is the production
  guests' per `host/README.md`).
* New rule for every throughput number: state `conc` **and** `max_tokens` **and** the prompt, and gate power
  sampling on busy==100. Historical aggregates without `max_tokens` are soft.

## 2026-11-01 (night): the prefetch question closes as a no-op, and the A/B wedges the box (OOM -> amdgpu SVM)

* **The prefetch knobs are inert on this stack.** `weight_utils.py:1115` is `should_prefetch = strategy ==
  "prefetch"`; with no `--safetensors-load-strategy` it auto-enables only on NFS/Lustre *and* when the checkpoint
  fits 90% of available RAM. Every rank logs `EXT4. Checkpoint size: 110.60 GiB. Available RAM: 40.74 GiB.` +
  `Auto-prefetch is disabled ...`, so `--safetensors-prefetch-num-threads 1` and the default arm ran **identical
  code**. Forcing the strategy would stage later shards into RAM that is already 70 GiB short of the 110.6 GiB
  each rank streams; vLLM warns `This may cause out-of-memory errors`. Not attempted.
* **Page cache is worth 4%, not 2.02x.** The 208.4 s vs 103.2 s pair came from inside a degrading machine. Re-run
  after the reboot with a verified teardown gate (`bench/gate.sh`): boot-cold `Cached` 0.35 GiB -> **load 115.67 s,
  place 98.0 s, experts 87.8 s**; warm `Cached` 39.3 GiB -> **120.60 s / 100.7 s / 88.2 s**. One healthy load takes
  ~50 GiB of anon across the four ranks *before reading a shard* (`Available RAM: 39.8 GiB` at loader start) and
  always leaves **4.2 GiB of swap used**; verified teardown gives it all back (MemAvailable 87-91 GiB, SwapFree
  8.0/8.0). Warm cache slows the shard stream (10.1 s/it vs 2.7 s) and buys nothing -- that direction is closed.
  `bench/prefetch_ab.sh` is neutered (reasoning in its header); `bench/coldstart_ab.sh` + `bench/gate.sh` are the
  only sanctioned way to relaunch.
* **The box died for it, and the cause is not vLLM.** `STOP` returns before teardown is verified, so run 3 of the
  loop started a 4-rank load while the previous 53.7 GiB of worker RSS (10.6/11.8/18.4/12.9) was still mapped --
  vLLM was already at `Available RAM: 40.74 GiB`, buff/cache 47 GiB, `SwapFree` down to 1.5 GiB of 8. The OOM
  killer fired (`/proc/vmstat oom_kill 8`) and SIGKILLed the two launcher wrappers plus the agent (session file
  ends mid-tool-call, no error block -- that is the OOM signature), then **kswapd0 wedged on the amdgpu SVM
  mmu-notifier**: `svm_range_cpu_invalidate_pagetables+0x7b [amdgpu] <- __mmu_notifier_invalidate_range_start`,
  `blocked on a mutex likely owned by task kworker/185:3`, while another task sat in
  `amdgpu_hmm_range_get_pages+0xf3`. Reclaim needs the mutex a HMM-validating worker holds; the worker needs
  reclaim. `sshd`, the driver `bash` and a dozen kworkers queued behind it, load average 62.
* Unkillable: after `kill -9`, workers stayed `D` in `exit_mm`, the driver stayed `D` in
  `svm_range_cpu_invalidate_pagetables`, VRAM held 19.4 GiB/GPU, load frozen at 1/23 shards for 73 minutes,
  `kswapd0` still `D` at 02:55. A GPU reset cannot free a kernel mutex -- reboot is the only exit. Full write-up
  and the launch rules (no launch unless no `vllm serve|VLLM::` process and `MemAvailable >= 55 GiB`; watch
  `SwapFree`; poll with short calls so an agent OOM costs a call, not a session) in **`notes/load-oom-svm-wedge.md`**.

## 2026-11-01: draft depth is a concurrency question; the native default becomes MTP=2

* Measured on two fresh instances (3 x 300-token samples at 32 in flight each, plus ITL, plus the quality gate):
  **MTP=2 median 1,755 tok/s aggregate at 57.6 tok/s a stream, against 1,488 at 48.8 for MTP=3** -- +18% both
  ways, with median ITL at 32 streams **37.1 ms against 49.8**. It costs single-stream decode: 197.6 -> 179.8
  tok/s (bench.py's decode-only metric 180.1 -> 170.4). GSM8K 98/100 and needles 3/3 at MTP=2, and it frees
  10,922 KV tokens (491,520 -> 502,442).
* Why: a decode step is NSEQ x (MTP+1) rows (128 -> 96 at NSEQ=32) while the third draft token only adds
  ~0.30 accepted tokens a step (2.52 -> 2.22 at one stream). Above ~16 streams the cards sit on the ~300 W
  envelope, so rows are the scarce resource; with one stream they are free, which is why MTP=3 stays the
  single-interactive-stream setting.
* Method notes for every future A/B here: @32 samples are bimodal (1,54x vs 1,75x, tracking 2.0 vs 3.1 GHz DPM
  states) so take medians of >=3; and low-concurrency aggregates swing 10% (first @4 328, repeat 471), so A/B on
  @8/@16/@32 with ITL as the tie-breaker.
* **MTP=1 completes the curve** -- @32 median 1,638 (below MTP=2's 1,755), single-stream 146.2, but the best ITL
  of the three (11.9 / 17.6 / 29.8 ms at 1 / 8 / 32) and the biggest KV pool (527,085). Aggregate peaks at depth 2.
  **NSEQ=48 with MTP=2 is a negative control**: @32 1,794 (inside the spread), @16 1,252, ITL unchanged, and 48 in
  flight reads 1,811 at only 39-40 tok/s a stream -- `--max-num-seqs 32` stands.
* Also today: the quality gate caches GSM8K locally (GitHub reach is intermittent here, and a mid-A/B gate died
  on a 60 s fetch), the launcher refuses to start on top of leftover ranks (`STOP=1` / `FORCE=1`), NSEQ/NBT
  defaults are TP-aware, `harness.py --conc`, and bench reports per-stream speed.

* **Cold-start anatomy.** `Model loading took` spans 143.8-179.9 s for the same config (page-cache state). Per
  rank the safetensors pull is 0.2 s and the consumer side is 77-156 s, of which **120.8 s is the 192 expert
  tensors (59.8 GiB a rank)**; **everything the plugin does at load adds up to 1.3 s** (fold_stats 0.4,
  `_prep_mxfp4` 0.4, permute/pack_scales ~0), and moving bytes is cheap: a real 0.78 GiB fused expert tensor does
  bulk H2D in 0.02 s (47 GiB/s), a per-expert strided TP slice only 2.5x slower. The minutes are four ranks each
  iterating the whole 110.6 GiB against one 5.8 GB/s disk. `local_expert_ids` cannot help at TP (a TP shard is a
  column slice of every expert) and `--load-format fastsafetensors` fails on ROCm outright. Corrected: the fold
  guard costs 0.4 s, not the 27 s the earlier `R9K_FOLD=0` ablation implied (different cache states).

## 2026-10-31: the stack, installed natively on a second 4x R9700 box

* No docker, no root, ROCm 10.1 host packages: the release tree runs in a **venv** (`vllm 0.32.0+rocm723`,
  `torch 2.13.0+git733fca1`) with `serve/native-flashnext.sh` as the launcher. The host needed three
  unpacked .debs and no privileges: ROCm's `-dev` headers for the libr9k build, OpenMPI 4 runtime for the
  ROCm 7.2-era wheels' torch, and the wheel's own `libamd_smi.so` reachable by its loader (the system
  ROCm 10.1 one lacks `amdsmi_set_gpu_clk_range`).
* **`R9K_MTP_QUANT_ROWS`**: rev 6fe3e746's bf16 fused MTP experts requant chunked at 2^18 rows = 2.68 GiB
  of fp32 a copy on the device. On this checkpoint it did not fit next to 19.2 GiB of weights (tried to
  allocate 5 GiB at 27.85 GiB in). Chunk count is now a knob; 32768 rows loads clean.
* Gates on a box that has never seen this code (Gen5 x16 host, four PLX switches, uncapped cards):
  integration vs **vLLM 0.32.0** ALL OK, kernel gates ALL OK, GSM8K 97/100, needles to 30.4k tokens.
* Flash-Next TP4: 19.16 GiB weights a GPU, KV 554,982 tokens, **186.7 tok/s single-stream (14.3 ms a step,
  2.68 tokens a step), 841 / 1,164 tok/s aggregate at 8 / 16, 7,702 / 9,633 / 8,963 tok/s prefill at
  2k / 8k / 31k.** Same step time as the 210 W gate, less draft acceptance (56.1% vs 62.2%). Against the
  Strata IQ3_S server on the same hardware the same afternoon: ~2x decode, 2.3-4x prefill, ~6x at 16 up.
  Full detail: `notes/native-4xr9700.md`.

* **P2P is not "~nothing" on a Gen5 host.** Negative control at TP4 (`R9K_ARN=0`, RCCL only): decode 187.0 ->
  155.1 tok/s, ITL 14.1 -> 16.3 ms, conc-8 844 -> 750, prefill 7.8k 9,686 -> 8,983. The 2026-09-18 finding
  ("P2P gives ~nothing at TP2") is a Gen3 box with the experts offloaded: there the step belongs to host expert
  traffic, not collectives. Here collectives are ~2.2 ms of a 16.3 ms step; the residual bound is ~1 ms.
* **Prefill chunk 4096 -> 8192** on the native TP4 launcher: +5.4% at 7.8k, +3.5% at 31k, +4.4% at conc 16,
  decode flat, costs 64k KV tokens. Cold start here is ~236 s, of which the safetensors + conversion pass is
  144 s while the disk can do 5.8 GB/s (O_DIRECT) -- page-cache thrash and per-tensor conversion, not IO.
* **`max_num_seqs` 16 -> 32** on the same launcher: 32 clients in flight 1,216 -> **1,433 tok/s** aggregate
  (+18%); at 16 slots they buy only +5% over 16 because they queue in two waves, and 48 is worse (1,366).
  Single-stream decode and long-context prefill do not move; the gate (GSM8K 97/100, needles 3/3, 64 sanity
  answers at conc 16) is clean.

* **Not adopted: `R9K_AR4_SDMA=1`** on a Gen5 host (@32 1,440 -> 1,402, @8 844 -> 822, decode unchanged) -- the
  2026-10-07 DMA-push result holds when the uplink is already saturated. **Load floor**: dummy 8.8 s, real 143.8 s,
  116.5 with `R9K_FOLD=0` (the fold repack is ~27 s), disk alone 19 s at 5.8 GB/s -- the rest is per-tensor repack
  under page-cache pressure, four ranks at once, with 38.7 GiB of RAM already pinned before the load starts.

* **What caps concurrency on this box is the power envelope.** VRAM is the binding resource for the model
  (31.3-31.4 of 31.86 GiB a card); CPU is flat at 5.4% of 192 cores from 1 to 32 streams (each rank 2.1 cores,
  EngineCore 1.00 spinning, API 0.05); at 16-32 streams the cards sit at ~300 W and the DPM pulls sclk from
  3.3 GHz to 2.0-2.3 GHz. Per joule: 0.21 tok/J single-stream -> 1.30 at 32 streams.

* **Per-stream rate added to the bench** (`bench.py` prints it, `harness.py` records `streamN_tok_s`): reading
  only the aggregate hid the real knee. 300-token outputs: 1 stream 193 tok/s; 8 -> 900 aggregate but **119.6 a
  stream**; 16 -> 1,203 / 79.8; **32 -> 1,537 / 50.5 (spread 48-54)**; 48 -> 1,454 / 48.2 (spread 30-53); 64 ->
  1,532 / 34.8. The box saturates at ~1,530 while every client keeps getting slower and the spread opens up past
  32 in flight -- the `NSEQ=32` choice holds on fairness as well as throughput.

* **Two-card probe on 93.8 GiB of host RAM (the RAM question):** TP2 with the biggest offload that fits
  (`OFFLOAD_GB=12` a rank) boots but is boxed in -- load 327.5 s swapping 4.1 -> 6.1 GiB, `Available KV cache
  memory` only **2.18 GiB (134,795 tokens, 4.11x at 32k)** against 491,520 at TP4, and decode 62 tok/s warm
  decaying to 9.5 within minutes (swap traffic during decode was 60 pages, so it is the streamed-expert path,
  not disk). Aggregate 147 @8 / 135 @16 with 18.8 / 12.5 a stream; prefill 4,190 / 4,582. Host memory adds up
  exactly as `38.7 (PLE int6, TP-independent) + TP x OFFLOAD_GB + TP x 4.3 + page cache`: at TP2 that is 92.5 GiB
  touched with only 12 a rank. **The repo's TP2 default (34 a rank) needs ~135 GiB -- 192 GiB is the comfortable
  size.** Four cards fits (30+ GiB spare in steady state) but the load still swaps ~4.1 GiB even from a clean
  start, and starting a second server on top of a live one cost **171.3 vs 186.4 tok/s** single-stream -- which
  is what the launcher's new pre-flight guard (`STOP=1` / `FORCE=1`) is for. Four-card's ceiling stays VRAM plus
  the ~300 W envelope. Also: SIGTERM does not stop the ranks -- `kill -9` the PIDs from `rocm-smi --showpids`.

## 2026-10-07 (evening): prefill kernels, the KV cache, and two things that did not pay

- **Fused fp8 hyper-connection up GEMM + gate mix** on the tiled WMMA kernel: 268-298 us vs the 665-674 us bf16
  pair a layer at 4096 rows; +4% 8k prefill in serving (7,044-7,069 vs 6,769), decode untouched, GSM8K paired
  p = 0.86 vs the record, HumanEval 160/164. Now default on Flash-Next (`R9K_HC_FP8=r9k`). Bug on the way: the
  fp8 copies were quantized at construction, before the checkpoint loaded -- single-stream decode looked fine
  (bf16 below 256 rows) while every batched or long prefill was garbage. Quantize after `load_weights`.
- **KV cache 279k -> 475k tokens on TP4** from three facts: vLLM's profiling run counts torch.compile transients
  as activation (3.83 GiB; the model's own peak is 0.6 GiB, `compat/memsnap.py`), the cudagraphs for 384-2048-token
  chunks cost 1.4 GiB a card for no soak-measured throughput, and 0.98 utilization leaves 1 GB at the soaked peak.
  Defaults: 0.98 + graphs to 256 (442-475k, the estimate varies by launch); `KVMEM=7.0` pins 475k; 7.5 (509k) ran
  a clean soak 440 MiB from the top and is the edge, not a setting. A 2048-token chunk: 311k tokens, -8% prefill.
- **All-reduce pipelining by row parts**: correct, gated, and a net loss (-5.5% / -6% prefill): the ar4 kernels,
  fused or DMA pushes, hold CUs while they wait on the uplink (4-rank stand-in: stock 5.15 ms, best 4.67, ideal
  3.68). Shelved. **Exact block-scaled fp8 tiled GEMM**: correct, 15-25% slower than vLLM's Triton. Opt-in.
- v0.2.6 validation on the release tree passed in full (BetterBench 162.8, three soaks 0 errors, gates, comm
  tests); the publish is on hold by decision. Full numbers: `notes/prefill-fp8-pipe.md`.

## 2026-09-18 (overnight session)

### Findings
- **P2P gives ~nothing at TP2 on Flash-Next** (clean A/B, same flags): single 69.7 vs 69.5 tok/s, agg within noise.
  Expert offload over PCIe dominates, not all-reduce. (bench/ab-p2p.log, bench/ab-nop2p.log)
- **Host->GPU link on the .100 box is PCIe Gen3** behind PLX PEX 8747 switches: 14 GB/s per GPU, 27.8 GB/s both.
  A Gen4 CPU-direct box roughly doubles it.
- **Kernel ablation of tcclaviger/vllm:dev** (ablation/run-ablation.sh; results ablation/summary.tsv):
  see table below. `clav_attn` is dead weight for this model (Triton attention equal); `fp8hip`, QSA r4d kernels,
  GDN HIP and the clav glue kernels each matter 5-13%.
- **Public prior art found**: davetha/r9700-lru-expert-cache (Apache-2.0: device-side LRU expert cache kernels +
  tests, 93-139 tok/s single-stream on the same 2x R9700 + model) and StillDeadcode/libr4d (gfx1201 GEMMs, AR,
  GDN; no license yet). The only truly private hard kernels are tcclaviger's MoE grouped GEMM and QSA sparse
  attention. Notes: notes/kernel-inventory.md, notes/ggz14-libr4d-review.md, notes/tcclaviger-fork-analysis.md,
  notes/upstream-0.29-rocm10.md.
- **Stock target exists**: `vllm/vllm-openai-rocm:nightly-rocm100-<sha>` = vLLM main + torch 2.12 rocm10 +
  triton 3.8 + hipcc, gfx1201 wheels. Pinned dee37d89.

### Built (all in this repo, compiled for gfx1201 with the image's hipcc, hostcall-free)
- `kernels/r9k_moe_mxfp4a8.hip` — grouped MXFP4 x FP8 MoE GEMM (derived from libr4d's dense kernel: fragment-order
  weights, folded-exponent unpack, fp8 WMMA, in-block split-K), MoE routing via vLLM's moe_align_block_size
  output, router weight folded in the epilogue, MT=1/2/4 M-tiles per block for prefill. + per-row fp8 quant.
- `kernels/r9k_ple.hip` — PLE int6 fused-row gather+dequant (reads the table from pinned host memory via UVA).
- `kernels/third_party/davetha/r4d_lru.hip` — vendored LRU expert cache kernels (Apache-2.0).
- `r9700_vllm/` plugin (vllm.general_plugins entry point), all hooks ROCm-gfx12-only and idempotent:
  - `moe/ct_mxfp4.py` — compressed-tensors MXFP4 MoE -> our experts (stock picks CUDA-only Marlin); in-place,
    UVA-aware weight re-layout.
  - `moe/experts.py` — `R9700Mxfp4Experts` (vLLM modular experts class).
  - `moe/cache.py` — per-layer VRAM slot arena + UVA host store + LRU (davetha kernels), warm start from the
    checkpoint's routing profile; two-pass hot/cold GEMM. `R9K_EXPERT_CACHE_GB` per rank.
  - `ple/int6.py` — int6 PLE table in pinned host memory (UVA) instead of stock's bf16-in-VRAM (51 GB/rank).
  - `linear/mxfp4.py` — dense MXFP4 linears (shared experts) on our kernel instead of per-call dequant emulation.
  - `spec/mtp_rocm.py` — MTP k>1 on ROCm (QSA metadata onto the spec-decode allowlist, cf. PR #55292).
- `docker/Dockerfile` (stock nightly + plugin), `serve/serve.sh` (was serve-stock-fn.sh), `bench/bringup-tests.sh`,
  `tests/` (MoE GEMM vs exact dequant, PLE vs reference, cache bit-identity + LRU invariants).

### Stock vLLM (ROCm 10 nightly) + plugin: bring-up and tuning log (2026-09-18 morning)
All unit tests pass on gfx1201 (MoE GEMM, PLE, cache bit-identity + LRU invariants, fp8 GEMM exactness).
Bring-up fixes needed (all in the plugin / launcher, no vLLM source patches):
- hostcall-free RCCL rebuilt IN the nightly image + hostcall-patched vLLM _rocm_C/_C_stable_libtorch (emulated switch)
- CT: fp8 groups inherit global `mxfp4-pack-quantized` format -> set float-quantized; `weight_scale_inv` -> `weight_scale`;
  drop fork q/k/v scales + fork MTP q4 head; MTP MLP fp8 block-128 can't shard 640/2 -> re-quantized to MXFP4 at load
- torch pinned memory rounds to 2^k (OOM) -> hipHostRegister exact-size pinning (PLE, cache, stock UVA offload)
- `--language-model-only` (ViT SDPA hipErrorInvalidValue on gfx1201/torch 2.12)
- stock CT W4A4 asks kMxfp4Dynamic -> our dense kernel must accept it (else per-call emulation)
- **HSA_ENABLE_IPC_MODE_LEGACY=0** (nightly sets 1 -> hipIpcGetMemHandle fails -> no P2P)
- **GPU_MAX_HW_QUEUES=1**: cross-stream waits inside HIP graphs were stalling ~50x/step (129 -> 25 ms ITL at bs1)
- libr4d 2-rank P2P all-reduce instead of RCCL (69 us -> ~3 us per call); vLLM's own custom AR = garbage on gfx1201

| config | single | @4 | @8 | @16 | pf 2k | GSM8K | needle |
|---|---|---|---|---|---|---|---|
| tcclaviger:dev (reference) | 82.5 | 81.1 | 146.4 | 145.6 | 2542 | 97/100 | 3/3 |
| stock+plugin eager, no cache, no MTP | ~5 | | | | | 39/40 | 3/3 |
| + graphs, cache 200, HWQ=1 | 38.3 | 114.6 | 120.0 | 115.4 | 333 | | |
| + MTP-3 (accept 2.7-2.8) | 57.3 | 98.8 | 135.4 | 139.5 | 503 | | |
| + fp8 LM heads + fp8 HC linears (cache 180) | 62.1 | 100.3 | 134.0 | 127.8 | 480 | 95/100 | 3/3 |
| + libr4d all-reduce | **68.0** | 105.8 | **139.5** | **139.6** | 487 | | |

| **VM100 256 GB RAM: all experts host-resident, LRU cache 270 slots on all 48 layers** | **75.3** | 120.5 | **193.8** | **194.7** | | **99/100** | **3/3** |

| same + Fable review P0 guards (commit 0f8af02) | 78.1 | | 203.4 | 187.9 | | 98/100 | 3/3 |

Prefill investigation (2026-09-18 pm): warm prefill ~1,550 tok/s @2k, ~2,050 @8k-31k. Profile of a 31k prompt:
43% of GPU time is expert gathers at ~14.5 GB/s = the Gen3 link. Every 4096-token chunk routes to nearly all 512
experts/layer, so all ~240 non-resident experts cross PCIe once per chunk. Staging cold experts into VRAM
(R9K_STAGE_COLD=1, bit-exact, tested) removes duplicate reads but gives no measurable prefill gain; bigger chunks do
(NBT=8192 + 240 slots: +25% @31k, +11% @8k, -10% @2k, KV only 43k tokens) -> trade-off, not default.
Real prefill fixes: Gen4 host link (~2x), more VRAM for experts (4 cards), or CPU-side compute of rarely-hit experts.
BENCH CAVEAT: bench.py runs are single-shot and confounded by compile-cache state (servers that loaded a cached
AOT graph (140 s startup) ran @4 at ~182 tok/s; fresh-compile starts (~470 s) ran ~118-121 with the same code).
Need a repeated-run harness before trusting <15% differences.

Launch (defaults now in serve/serve.sh, MTP=3 default): `OVERLAYS=emulated-switch serve/serve.sh` then `python3 ~/warmup.py`
(= OFFLOAD_GB=34, UTIL=0.94, R9K_EXPERT_CACHE_SLOTS=270, fp8 target+draft LM heads). 320 slots leaves no KV room.
Opt-ins measured and left off: R9K_FP8_BLOCK=rowwise|block (acceptance drop / ~1 ms), R9K_FP8_LINEARS=hyper_connection
(acceptance drop), R9K_DRAFT_LMHEAD=mxfp4 (throughput +, single -).

Remaining gaps: short-prompt prefill (host read-through of offloaded experts over Gen3 PCIe), LRU miss traffic
(~7.5 ms/step), untuned Triton fp8-block GEMMs (~4.7 ms/step); VM100 RAM upgrade would allow a full cache.
Correction: the "+ fp8 HC linears" row above ran on a stale torch.compile cache that still used the bf16 HC path
(Dynamo cannot trace our ctypes kernels; they are now torch custom ops, and the compile cache is keyed by R9K_* knobs):
that gain is the fp8 LM heads alone. With fp8 HC + row-wise fp8 block projections truly active: 58.5 single, MTP
acceptance 2.64 (vs ~3.0) -> worse; being A/B'd separately.
Correction: the earlier "P2P gives nothing" A/B was flawed (tcclaviger's r4d AR kept using P2P IPC in both arms).

### 2026-09-18 afternoon: extension points, overlay split, harness, NVFP4
- **Plugin on official extension points** (a529442): `R9kCompressedTensorsConfig` registered over
  "compressed-tensors"; `ModelRegistry` subclasses for Qwen4Exp CG/CausalLM/MTP; `R9700Platform` via
  `vllm.platform_plugins` -> `R9kCommunicator` (libr4d AR). One class patch left (MTP allowlist, version-gated).
  `tests/test_integration.py` covers every hook. Gate (MTP-3, TP2): 203.9 tok/s @8, 189.0 @16, prefill 8k 2727 tok/s,
  GSM8K 98/100, needle 3/3 -- parity with pre-refactor.
- **Overlay split** (cbffc52): `overlay/emulated-switch/` (hostcall-free RCCL, vLLM .so hostcall patcher, scanners),
  `serve/serve.sh` stock by default, `OVERLAYS=emulated-switch` on VM100. MTP=3 now the default.
- **Harness**: `bench/harness.py` (nonce prompts, streamed TTFT vs decode, repeats -> median/spread) + `bench/ab.sh`
  (interleaved A/B with restarts).
- **NVFP4**: (a) load-time NVFP4->MXFP4 conversion (dense + MoE; exact dequant, 2-candidate E8M0 search). Weight error
  sits at the double-rounding floor, ~1.55x NVFP4's own (synthetic: 0.095 -> 0.149). Qwen3.8-27B-NVFP4: GSM8K-200
  96.5% (GGZ14's conversion run on 09-16: 97.0%), decode 29.7 tok/s, 8-conc 210.7 tok/s.
  (b) native NVFP4 kernel variant (`r9k_moe_nvfp4a8`, template NV): WMMA on unscaled e2m1, per-(row, 16-K) e4m3 scale
  applied to the partial (8 FMAs/WMMA), per-row global in the epilogue -- exact weights; default for dense NVFP4
  (`R9K_NVFP4=native|mxfp4`). NVFP4 MoE layers still convert (expert cache carries MXFP4-layout scales).

### 2026-09-19/20: Qwen3.8-27B-NVFP4 speed work (target: GGZ14 radiance on the same box)

**Baseline discipline.** `~/bb-prod.log` (181.5 combined) is the PRODUCTION box (192.168.0.155, Quark MXFP4) --
not comparable. The same-box GGZ14 radiance NVFP4 baseline is `~/bb-full.log` (2026-09-16) and a 2026-09-19 rerun:
**combined decode 196.5 t/s, conc 177/294/428/549, update p50 23.5 ms, TTFT 65 ms, prefill ~4.8k t/s**, GSM8K 97.0,
HumanEval 98.2. Always check `results.json` `env.endpoint` before quoting a number.

**Our 27B config** (serve/serve.sh): `R9K_AR_QUANT=1 VLLM_KV_CACHE_LAYOUT=LBHNC KVMEM=9 R9K_BF16_TO_MXFP4=in_proj_ba
R9K_FP8_TO_MXFP4=1 MODEL=/models/Qwen3.8-27B-NVFP4 OFFLOAD_GB=0 MTP= DRAFT=/models/Qwen3.8-27B-DFlash2-FP8 SPEC=7
ATTN=CUSTOM DRAFT_ATTN=CUSTOM CHAT_TEMPLATE=qwen-fixed NSEQ=8 OVERLAYS=emulated-switch`.

| step | decode single | conc-8 | prefill 9k | ms/step |
|---|---|---|---|---|
| start (no spec, stock attn) | 29.7 | 211 | 1374 | - |
| + DFlash2 spec7 (drafter fp8 fix) | 69.7 | 295 | - | 43 |
| + TRITON_ATTN (target+drafter), fp8->MXFP4, tuned tiles | 103 | 314 | - | 27.4 |
| + split-KV verify attention (CUSTOM), quant kernel, drafter on libr9k | 115 | 405 | 1340 | 25.8 |
| + GGZ14 chat template (acceptance +14%) | 118 | 416 | - | 25.3 |
| + libr4d prefill attention (LBHNC + fp8 descales) | - | - | 2048 | - |
| + wht6 compressed all-reduce (>=128 KB, opt-in) | - | - | 2250 | - |
| + large-M prefill GEMM (Fable pass 1+2) | 114 | 437 | 3450 | 26.0 |
| + NVFP4->MXFP4 conversion at load (R9K_NVFP4=mxfp4) | 124 | 472 | 3740 | 25.5 |
| + folded-exponent MXFP4 (R9K_FOLD=1, bit-exact here) | 114 | 487 | **3932** | 25.3 |

Quality: GSM8K-200 96.5-97.0%, HumanEval 96.3%, 8/8 concurrent sanity answers.

**What moved the needle, in order:** speculative decoding (drafter fp8 dequant fix in models/dflash.py), the chat
template (acceptance: code 3.34 -> 4.49 tok/step), libr4d prefill attention (2413 -> 93 ms per 9k prompt), the
large-M GEMM (68-84 -> 117-142 TFLOPS), split-KV verify attention, fp8->MXFP4 requant, wht6 all-reduce.

**Measured dead ends:** unpadded drafter batch + sync scheduling (needs GGZ14's vLLM patch; 29.1 ms/step on stock);
drafter in W4 (-0.5 ms/step but -7% acceptance); LDS-staged A for the decode kernel (never wins once its gate is
correct); NT loads on MT>1 (Flash-Next prefill -24%); fp8->MXFP4 on Flash-Next (no gain, it is expert-traffic bound).

**Flash-Next on the same code:** decode 81 / conc-8 206 / prefill 2046, MTP acceptance 2.69 -- unchanged or better
at every step (the prefill GEMM gave conc-8 +13%).

**Gotchas found the hard way:** vLLM's memory profile underestimates once load-time requant/merges are on (use
KVMEM=); a stale `tuned.json` sync silently reverted the M=32/64 rows; LDS-A with WV=1,MT=4 staged half its tile
(NaN at batch 64) -- tests/test_tuned_cfgs.py now gates every tuned row.


**FINAL BetterBench 2026-09-20** (20 passes, same box, same settings as the GGZ14 baseline), ours vs GGZ14:
combined decode **189.3 vs 196.5 (96%)**, conc 165/274/404/522 vs 177/294/428/549 (93-95%), prefill sweep
3838/3902/3843/3649 vs 4776/4950/4906/4745 (~79%), step p50 24.4 vs 23.5 ms, TTFT 99 vs 65 ms, per-category
acceptance at parity (code 4.69 vs 4.64, file_edit 6.10 vs 6.03). Serve config: `R9K_FOLD=1 R9K_NVFP4=mxfp4
R9K_AR_QUANT=1 R9K_FP8_TO_MXFP4=1 R9K_BF16_TO_MXFP4=in_proj_ba VLLM_KV_CACHE_LAYOUT=LBHNC KVMEM=9 ATTN=CUSTOM
DRAFT_ATTN=CUSTOM CHAT_TEMPLATE=qwen-fixed DRAFT=.../Qwen3.8-27B-DFlash2-FP8 SPEC=7`.
Flash-Next on the same build: single 84.5, conc-8 206, prefill 2165, MTP acceptance 2.75 (all >= its pre-week numbers).

**Remaining prefill gap is GEMM only** (9k prompt, GPU-busy): ours 1361 ms / 768 calls vs GGZ14 930 ms / 484;
all-reduce 373 vs 360, attention 92 vs 58, glue comparable. Their edge: activations arrive WMMA-fragment-tiled
(256 B per 16x16 fp8 fragment) from their fused norm+quant, so their GEMM never stages A through LDS.

### 2026-09-20 evening: fragment-tiled activations (closes most of the prefill GEMM gap)

The prefill GEMM now takes its activation in WMMA-fragment layout (`r9k_quant_rows_fp8_tiled`: fragment (mt, ks)
= 256 contiguous bytes, lane l holding A[mt*16 + l%16][ks*16 + (l/16)*8 .. +8]) and loads each A fragment with one
SADDR `global_load_b64` straight into the register the WMMA consumes -- no LDS A slab, no per-fragment `ds_read`,
no staging stores, LDS down from 2x26 KB to 2x8 KB on the 256x128 tile. The quantizer writes that layout instead
of row-major at the same cost (+/-1%), so it is a re-layout and not an extra pass. W keeps the pass-3 folded
unpack through LDS. Output is bit-identical to the folded LDS-A kernel (same accumulation order), gated by
`tests/test_atiled_4bit.py`. Dense path only: `pick_cfg(..., fold=True) -> ("A", cfg)` at M >= 128 when
`R9K_ATILED=1` (default on), folded MXFP4, K % BK == 0 and K <= 20480; everything else keeps the LDS-A tiles.

Kernel-level (min of 5 interleaved, warmed): 0.74-0.92x of the tuned folded tile across the five served 27B
shapes at M 128-4096, i.e. ~150 -> 175-215 TF/s. Served, 27B, three probes per leg on two independent server
starts (cv < 0.1%):

| prompt | `R9K_ATILED=0` | `R9K_ATILED=1` | gain |
|---|---|---|---|
| 8.97k tok | 3936 tok/s (2281 ms) | **4434 tok/s** (2021 ms) | **+12.7%** |
| 20.7k tok | 3772 tok/s (5477 ms) | **4209 tok/s** (4919 ms) | **+11.6%** |

So 9k prefill is now **~93% of same-box GGZ14** (4776-4950), up from ~79%. Decode is untouched, as the M >= 128
gate implies: single 113.2/116.4 (on) vs 115.2/118.7 (off), conc-8 483.6 vs 483.8, GSM8K-500 96.40% both legs;
Flash-Next unchanged (84.6 / 205.1 / prefill 2151 / accept 2.73). A first conc-8 pair read 472.6 vs 503.8 with
non-overlapping ranges at n=2 -- re-measured on fresh servers it was 483.6 vs 483.8, i.e. run-to-run noise, which
is what the code says it must be (`pick_atiled` returns None below M=128, so the decode path is byte-identical).

Routed MoE deliberately not converted: those GEMMs are weight/L2-stream bound (gate_up moves the 420 MB expert set
at 397 GB/s), so a gather+tile pass would add ~25% of the weight traffic to a kernel whose A staging is not the
limit. Argued, not measured -- the kernel already accepts sorted-position tiled A, only the producer is missing.

**The 768-vs-484 launch-count gap was not apples to apples.** GGZ14 chunks at 8192 (2 chunks on a 9k prompt to our
3 at NBT=4096) and keeps MLP layers 56-63 on fp8, while `R9K_FP8_TO_MXFP4=1` converts them -- so their 930 ms
excluded 16 GEMMs per chunk that ours included. Untried from this: `NBT=8192` (check the libr4d AR max message
size at 8192x5120x2 = 80 MiB) and leaving layers 56-63 on the fp8 kernel.

### 2026-09-21: independence from libr4d (see notes/independence.md)

Both runtime dependencies replaced by our own kernels: paged attention (`kernels/r9k_attn.hip`, now the default)
and the 2-rank all-reduce (`kernels/r9k_ar.hip` + `r9k_ar_wht.hip`, `R9K_AR_IMPL=r9k`). A fully libr4d-free 27B
runs at **~11% below prefill and ~4% below decode** with identical GSM8K-500, against -55% prefill if the
dependency were simply dropped. Attention alone is 99.8% of libr4d; the whole remaining gap is the all-reduce.

**Prefill knobs measured and rejected** (both fell out of the finding that the "768 vs 484 GEMM launches" gap was
never apples to apples -- GGZ14 chunk at 8192 and keep MLP layers 56-63 on fp8):
- `NBT=8192`: no gain at 9k (4395 vs 4392) and the engine dies with HTTP 500 at a 20.7k prompt. Rejected.
- `R9K_FP8_TO_MXFP4=0` (keep layers 56-63 on fp8): prefill 3456 vs 4392 (-21%), decode 94.7 vs 115.1, step
  30.7 vs 25.1 ms. Much worse -- converting those layers is a clear win and the existing default is right.

### 2026-09-21: TTFT investigated and closed

101 ms vs the reference stack's 65 ms. Decomposed to **~80 ms fixed + 0-25 ms step-boundary wait**; the fixed part
is prefill (queue time 0.0 ms, prefill time ~78 ms for an *8-token* prompt), and prefill runs **eager**: 32.2 ms
GPU busy against ~78 ms wall, ~1,950 launches at ~24 us of gap each. `cudagraph_mode` already defaults to
FULL_AND_PIECEWISE so those launches survive capture; `FULL` is 55 ms *worse*. Ruled out by measurement:
speculative decoding (helps by 37 ms), chat-template rendering, HSA_ENABLE_MWAITX, GPU_MAX_HW_QUEUES, the API
server. Closed as upstream behaviour -- full detail and the one remaining lever in notes/picking-up.md.
(**Reopened and root-caused 2026-10-02**, below: the eager launches were stock's chunked GDN prefill core, 48 times
a step. Not upstream behaviour we had to accept.)

Added `CGMODE=` and `MWAITX=` knobs to serve/serve.sh while investigating.

### 2026-09-22: power cap is a fixed condition, not a knob

Raising the 225 W cap is closed permanently. Every reference measurement (libr4d, GGZ14) was taken at 225 W, so
uncapping ours would invalidate the comparison in the same way the `bb-prod.log` baseline mix-up did; and Brian
runs capped in production, so an uncapped number is one he would never see. Prefill being clock-limited at
2.40 GHz is therefore a property of the target, not an opportunity.

**Measurement discipline (learned the hard way this week):**
- ALWAYS check `results.json` `env.endpoint` before quoting a baseline: ~/bb-prod.log is the PRODUCTION box.
- Never start a run while another is live (two servers on the same GPUs produced 30% CVs and nonsense numbers);
  `pgrep -f "[g]sm8k|[b]atch|[q]fn"` + `docker ps` before starting.
- Cold single-shot kernel timings overstate by ~15%: the card sits at 2.2-2.4 GHz / 222 W under sustained load
  (decode sees 2.82 GHz). Use tuning/prefill_ab.py (warmed, interleaved).
- GSM8K-500 CANNOT separate these configs: six runs scatter 95.6-97.4% with no consistent ordering. Quality claims
  need several thousand questions or a different eval.
- conc-8 (`agg8_tok_s`) at n=2 can separate identical builds by 6% with non-overlapping ranges: the prefill-probe
  words are ~6.9 tokens each (`w<random>`), so WORDS=1300 is the ~9k-token prompt and WORDS=6000 exceeds the 32k
  context and 400s. Before believing a decode delta, ask whether the code even has a path to it, then re-measure.
- Kernel gates that call the inner op miss served-path breakage: exercise r9700_vllm/ops.py's public wrappers too.
- `DRYRUN=1` only prints the assembled docker command: it does NOT source the overlays, so a launcher missing
  `OVERLAYS=` passes DRYRUN and then fails at startup. serve/27b.sh and serve/flashnext.sh shipped that way and
  every RCCL collective died with HIP "the operation cannot be performed in the present state" at enqueue.cc:2119
  (comm init completes, single-GPU compute is fine, all transports fail alike) -- the signature of the stock,
  hostcall-carrying RCCL on the .100 PLX box. Both launchers now default `OVERLAYS=${OVERLAYS-emulated-switch}`.
  A new launcher is not verified until it has actually served a request.
### 2026-09-23: 45+48 on one PLX switch, switch-local P2P, and what routing costs in serving
Third card installed; VM100 now runs **45 + 48, both on the same PEX 8747** (was 45 + c6 on separate switches).
Host work is in `host/README.md`: a second card on one switch gets no BAR0 without `r9700_chainfix` (DKMS), and
peer traffic only stays in the switch with ACS redirect off **and** guest GPU addresses equal to host addresses.
- Synthetic, 45<->48 at 256 MB: both-ways peer copy 12.74 -> **25.26 GB/s**; RCCL all-reduce busbw 5.09 -> 10.08.
- **Full BetterBench 20-pass, `serve/27b.sh` (switch-local P2P) vs the 2026-09-22 baseline (45 + c6): parity.**
  Step 24.4 ms both; conc 166/280/379/510 vs 166/270/394/521; prefill 4,123/4,153/4,048/3,812 vs
  4,099/4,143/4,043/3,809 (2k/8k/16k/32k); per-category decode all within noise (`betterbench compare`).
- **Live A/B on one running server, ACS redirect off / on / off** (`host/acsab.sh`; prefill sweep + quick decode):

| | pf 2k | pf 8k | pf 16k | pf 32k | step p50 | step p99 |
|---|--:|--:|--:|--:|--:|--:|
| switch-local (off) | 4,167 | 4,187 | 4,070 | 3,821 | 24.47 ms | 26.62 ms |
| **via CPU (on)** | **3,741** | **3,770** | **3,683** | **3,491** | **25.29 ms** | **27.37 ms** |
| switch-local (off, repeat) | 4,121 | 4,149 | 4,047 | 3,810 | 24.49 ms | 26.50 ms |

  Routing peer traffic through the CPU costs **9-10% prefill and +3.3% decode step time**; the repeat arm
  returns to within ~1%, so it is not drift. Decode tok/s medians moved with speculative acceptance (199/200/203)
  and are not the signal here; step time is. No IOMMU/AER faults in any arm. Raw results: `~/bb0923/` on VM100.
- So two cards per switch cost nothing **only** with switch-local routing. For 4 cards on two switches the
  in-bank hops need it too, which means matching guest addresses for both banks in one VM (not yet solved:
  OVMF packs its 64-bit window contiguously and the two banks' host apertures are ~0.5 TB apart).
- Correction carried forward: the 2026-09-18 Flash-Next "P2P gives nothing" A/B is invalid (P2P IPC in both arms).

### 2026-09-24: Flash-Next at TP=4 on four cards (target: tcclaviger 29.04.4 on the same box)

Rob's published Flash-Next TP4/MTP4 run (163.8 BetterBench decode, 687 @16, 7.2k prefill) is on his hardware; his
image on OUR box (`~/serve-rob-tp4.sh`, hostcall-patched copies in `~/p2p-patched-2904`) is the real target. All
numbers below are the 1-minute probe (`~/probe.py`: 4 mixed prompts greedy / conc 8 and 16 / one 8.7k prefill);
BetterBench is reserved for a finalist. Step time is single-request MTP-3 (`~/tp4tune.sh` step.txt).

| config | step ms | dec 1 | c8 | c16 | prefill 8k |
|---|--:|--:|--:|--:|--:|
| ours, start of day (TP=4, RCCL all-reduce, stock QSA) | 28.1 | -- | 299* | 508* | 3,767* |
| + our 4-rank P2P all-reduce | 22.9 | 140 | 299 | 508 | 3,767 |
| + our QSA sparse attention | 22.9 | 139 | 479 | 770 | 4,689 |
| + hybrid block-fp8 + mxfp4 heads + fused qk-norm/rope | **21.6** | **148** | **501** | **770** | **4,688** |
| Rob's image (29.04.4, MTP-4, fp8 KV, expert offload) | 20.8 | 169 | 503 | 769 | 6,446 |

(*) measured after the all-reduce change; the RCCL baseline's probe was not run.

- **N-rank P2P all-reduce** (`kernels/r9k_ar.hip` `r9k_ar_oneshot_nrank` / `r9k_ar_twoshot_nrank`, `comm/r9k_ar.py`
  `R9kAllReduceN`, installed for TP>2): one-shot <= 16 KB, two-shot <= 512 KB, RCCL above. Graph-timed at 20 KB
  11 us vs RCCL 73; 320 KB 73 vs 102. Decode step 28.1 -> 22.9 ms. At prefill sizes (20 MB) the exact two-shot
  loses to RCCL's ring (4.05 vs 2.97 ms even at 256 blocks): the all-to-all push saturates the inter-switch uplink.
  Only compression or a topology-aware schedule would win there; parked.
- **Our QSA sparse attention** (`kernels/r9k_qsa.hip`, `attn/qsa.py`, bound per layer; `R9K_QSA=stock` reverts).
  Reformulation: a row's selection is a union of whole 4-token groups plus the causal tail, so each row becomes a
  bitmap over groups and a tile of 16 rows walks the UNION of its groups, staging K/V once per tile (the stock
  Triton kernel gathers ~2 MB per row per layer). Bit-for-bit the same attended set; fp32 online softmax either
  way. 3.6x on the 12k prefill chunk (14.0 -> 3.9 ms), 13x on a 4k prefill, 2.4x decode; `tests/test_qsa_r9k.py`.
  Bug found on the way and worth remembering: a row that has met no valid key yet (m = -inf) riding along another
  row's rescale computes exp2(-inf - -inf) = NaN; impossible in causal attention (every row sees key 0 first),
  routine in QSA. Guarded.
- **Decode profile, ours vs Rob (torch profiler, `~/step-audit.py` per-layer):** GPU time per layer is at parity
  (415 vs 462 us, his including ~62 us of host expert streaming); we launch 78 kernels per layer to his 37. Enabling
  stock's fused qk-norm/rope Triton kernel on ROCm (`R9K_FUSED_QKROPE`, off in stock only because of an
  `is_cuda()` check; `tests/test_fused_qk_rope.py`) removed ~30 launches per QSA layer for ~0.06 ms/step: launch
  count is worth far less than assumed. The remaining eager glue is the indexer's norm + rope; not worth fusing.
- **Knobs that paid:** `R9K_FP8_BLOCK=block` (exact block scales, our split-K GEMM) at decode widths with stock's
  Triton kernel above M=64 (`R9K_FP8_BLOCK_MAXM`); the dispatch lives INSIDE a custom op (`r9700.fp8_block_dispatch`,
  layer registry keyed by registration index) because a Python branch on M in compiled code is resolved once at the
  M=4096 profile pass and then serves decode with the wrong kernel, and because `id(layer)` baked into the AOT
  artifact differs per rank process. mxfp4 LM heads (`R9K_TARGET_LMHEAD` / `R9K_DRAFT_LMHEAD`) halve the head's bytes.
- **Knobs that did not:** MTP-4 (acceptance up, step +1.7 ms, net loss); fp8 or mxfp4 for the hyper-connection
  mixers (latency-bound at these sizes; the merged down+inject matrix is 324 rows and fails the %16 check anyway);
  MoE decode config sweep for TP=4 shapes (`tuning/decode_moe_sweep.py`: served configs within 2-11% of best;
  gate_up now keyed by N in `moe/experts.py`); fp8 KV is refused by stock QSA ("requires a BF16 main KV cache").
- **Quality gate** (`bench/eval.py`, 300 GSM8K, conc=1): reference 96.0% vs candidate 97.0%, McNemar p=0.51, sanity
  8/8 both. Long-chain paired eval still to run before the candidate becomes the default.
- **vLLM 0.30** (`nightly-rocm100-e975732`, torch 2.12): whole stack runs unchanged (`r9700/vllm:dev030`,
  `~/p2p-patched-030`); probe at parity with the 0.29-era nightly.
- **Full BetterBench (20 passes), final default vs Rob's image, same box:** decode 125.1 vs 134.1; step p50 21.22
  vs 20.44 ms; TTFT p50 137 vs 145 ms; prefill 2,853 / 4,637 / 4,701 / 4,614 vs 5,279 / 5,711 / 5,977 / 6,106
  (2k / 8k / 16k / 32k); concurrency 117 / 193 / 283 / 374 / 478 vs 126 / 197 / 303 / 427 / 542 (1 / 2 / 4 / 8 / 16).
  The greedy probe overstated concurrency parity (-12% under sampling); short prefill (2k) is the worst ratio.
- **2k prefill profile (2,142 tokens, rank 0): our all-reduce is 387 ms of 671 ms GPU time (222 RCCL calls at
  1.73 ms for 11 MB messages); Rob's 116 ms (0.58 ms per call, compressed two-shot).** Everything else is within
  ~50 ms of his. The short-prefill 2x is the all-reduce alone.
- Prefill gap that remains (12.7k tokens, rank 0): all-reduce 1,019 ms (RCCL) vs Rob's 607 (compressed two-shot);
  attention now ~250 vs his 159; MoE/dense/elementwise ~180 ms combined.

### 2026-09-25: compressed hierarchical all-reduce, vLLM 0.30 default

- **Base image switched to the vLLM 0.30 nightly** (`nightly-rocm100-e975732`, torch 2.12). Final-config probe on
  it: 21.60 ms/step, 148.0 / 484 / 752 / 4,685 -- identical to the dee37d89 image. On VM100 `r9700/vllm:dev` is now
  the 0.30 build (old image kept as `dev-0918`), `~/p2p-patched-nightly` links to the 0.30 patched extensions.
- **`kernels/r9k_ar4.hip` + `comm/r9k_ar4.py`: the all-reduce from notes/ar4-plan.md.** Hierarchical 2x2 over the
  switch pairs, Walsh-Hadamard 4/6-bit wire fused into the push kernels (two elements per lane; no LDS in the
  transform), rotated-domain sums, seven launches, four handshakes, bit-identical outputs on all ranks.
  `tests/test_ar4.py` (torchrun, 4 ranks): PASS at 4 and 6 bits, graph replay interleaved with the exact kernels.

| message | RCCL | ar4 4-bit | ar4 6-bit | exact two-shot |
|---|--:|--:|--:|--:|
| 1 MB | 204 us | 118 | 143 | 222 |
| 5 MB | 773 | 371 | 495 | 1,194 |
| 11 MB | 1,640 | **739** | 1,009 | 2,723 |
| 21 MB | 3,107 | **1,352** | 1,863 | -- |

  Rob's kernel: ~580 / ~1,200 us at 11 / 21 MB. Relative RMS error on a heavy-tailed input: 0.12 (4-bit), 0.027
  (6-bit); an already-quantised input round-trips to bf16 rounding.
- **Serving (Flash-Next TP=4 probe):** 8k prefill 4,685 -> **5,621 tok/s with 4-bit (+20%)**, 5,319 with 6-bit;
  decode 147-149, c8 497-506, c16 738-768 -- unchanged within noise. Prefill is now 87% of Rob's image (from 73%).
- **Quality (300 GSM8K, conc=1, same image):** default 96.67% vs 4-bit 96.00%, discordant 4/2, McNemar p=0.69: no
  detectable difference. The 800-question chain-of-thought paired eval decides whether R9K_AR4 becomes the default
  (opt-in `R9K_AR4=1` until then; `R9K_AR4_BITS=6` is the conservative wire).

### 2026-09-25 (cont.): the short-prefill floor, root-caused

BetterBench 2k prefill was 2,779 tok/s against Rob's 5,279 while 8k+ was within 3-8%. A TTFT-vs-length sweep
(`~/pfsweep.py`) showed TTFT quantized: ~285 ms for anything up to ~700 tokens, ~563 ms for 1.3k-3.1k tokens
regardless of length, then linear at ~5,650 tok/s. Two mechanisms, found with the torch profiler's CPU-side trace:

- **The prefill forward pass is CPU-bound at ~285 ms.** For a 672-token prefill the CPU issued ops for 407 of the
  426 ms span: 20,230 CPU ops per forward (compiled-graph pieces 267 ms, MoE Python 96, GDN 77, 351 unquantized
  GEMM dispatches 54, 101 all-reduces 32). Decode hides this behind cudagraphs; prefill runs the compiled graph
  eagerly. The rank skew this creates is what the all-reduce kernels spin on (ar4_pack_push at 498 us per call in
  the profile against ~40 us in the bench). Rob's stack has the same floor: his linear rate is 5,711 tok/s.
- **Prompts were split at the mamba block size.** With prefix caching on (default), the hybrid model runs the mamba
  cache in "align" mode and vLLM's scheduler aligns every chunk end to the mamba block size, so a 1,302-token prompt
  ran as two forward passes (202 all-reduces, 98 MoE calls, 26 QSA calls: exactly 2x the 672-token run), each paying
  the CPU floor. `--mamba-block-size` is ignored in align mode.

Fixes measured (TTFT, tokens -> ms): 1,325: 563 -> 313; 2,166: 564 -> 373; 3,086: 564 -> 512 (6,022 tok/s).

| | before | `--block-size 4096` | **`--no-enable-prefix-caching`** |
|---|--:|--:|--:|
| 1,325 tokens | 563 ms | 296 | 313 |
| 3,086 tokens | 564 ms | 510 | 512 |
| KV cache capacity | 296k tokens | 99k (mamba pages padded 9x) | 296k |

`serve/flashnext.sh` now defaults `PREFIX_CACHE=0` (serve.sh knob; `=1` restores prefix reuse across requests --
a serving-behaviour change, flagged to Brian). **Full BetterBench of this default:** decode 125.1, step p50 21.12 ms,
TTFT 144 ms, prefill 5,267 / 5,702 / 5,723 / 5,530 (2k / 8k / 16k / 32k), concurrency 116 / 190 / 277 / 370 / 463.
Prefill is at parity with Rob's image through 8k (5,279 / 5,711) and 4-9% behind at 16-32k; decode -7%, concurrency
-13..-15% under sampling remain. Removing the CPU floor itself (capturing prefill chunks in cudagraphs
at 1024/2048/4096) is being measured; it is what would pass Rob at short prompts.

### 2026-09-25: final default and where it stands

Default now: our QSA attention, N-rank + compressed (4-bit) all-reduce, hybrid block-fp8, mxfp4 LM heads, fused
qk-norm/rope, prefix caching off, prefill cudagraphs to 2048 tokens, vLLM 0.30. Full 20-pass BetterBench, same box
and checkpoint as Rob's image (tcclaviger 29.04.4, MTP-4, fp8 KV, expert offload):

| | ours (2026-09-24 morning) | **ours now** | Rob's image |
|---|--:|--:|--:|
| decode score | 125.1 | 125.0 | 134.1 |
| step p50 | 21.22 ms | 21.15 ms | 20.44 ms |
| TTFT p50 | 137 ms | **104 ms** | 145 ms |
| prefill 2k / 8k / 16k / 32k | 2,853 / 4,637 / 4,701 / 4,614 | **5,106 / 5,624 / 5,726 / 5,532** | 5,279 / 5,711 / 5,977 / 6,106 |
| concurrency 1 / 2 / 4 / 8 / 16 | 117 / 193 / 283 / 374 / 478 | **118 / 197 / 292 / 433 / 543** | 126 / 197 / 303 / 427 / 542 |

Concurrency at parity, TTFT 28% better, prefill within 1.5-9%, single-stream decode 7% behind (his MTP-4 vs our
MTP-3). The same full run with MTP-4: decode 130.5, step p50 22.48 ms, concurrency 122 / 197 / 294 / 402 / 466 --
+4.4% single-stream for -14% at 16 concurrent, so MTP-3 stays the default and `MTP=4` is the single-stream knob. Every lossy default passed two null paired evals at
conc=1 (300 short-answer, 800 chain-of-thought); the served config's sanity check passes.

### 2026-09-28: QSA indexer scoring kernel (r9k_qsa_score)

The 4k-chunk profile of 2026-09-25 (final stack vs Rob, `~/chain-pf4k.out`) put our GPU-busy at 731 ms vs his 625:
all-reduce 149 vs 200 (ours now faster), MoE 177 vs 141, elementwise 161 vs 97, QSA 96 vs 46. The QSA gap was
one kernel: vLLM's Triton `_qsa_mqa_paged_kernel` (indexer scoring) at 80 ms per chunk, 43 calls x 1.85 ms. It
launches a program per (row, 32 columns) over the whole context capacity, re-reads the compressed keys once per
row, and writes every column of an fp32 [rows, capacity] logits buffer (128 MB per layer at 4096 rows), although
`top_k_per_row_decode` reads row r only up to its visible count (position / 4).

`kernels/r9k_qsa_score.hip` (independent implementation from the stock kernel's documented semantics, same WMMA
pipeline as r9k_attn): one workgroup = 16 query rows x one column part; K[64 x 128] staged once per tile through
the page table, per head S^T = K . Q_h^T on 8 WMMA 16x16x16 bf16, relu, sum over heads, scale; the tile stops at
its last visible column and leaves the rest of the logits row untouched. Few-row batches (decode, MTP verify)
split the column range across workgroups (R9K_QSA_SCORE_SPLITS). `attn/qsa_score.py` mirrors the stock
`qsa_select_paged_tokens` (same 128 MB chunking, stock top-k and expansion) and is bound as `_select` on every
`QSAIndexer` instance from `qsa.install`; `R9K_QSA_SCORE=stock` keeps vLLM's kernel. The indexer runs inside
vLLM's `qwen4_exp_qsa_with_output` custom op, so the ctypes launch is outside torch.compile and cudagraph-safe.

`tests/test_qsa_score.py` (single GPU, 15 cases: prefill tiles, chunked prefill to 32k, mixed batches with
padding rows and forced splits, decode, MTP verify, tiny contexts, page sizes 4/16/64, 1/2/4/8 heads): logits on
every visible column vs an fp32 reference, ours 1.4-4.2e-7 relative vs stock 1.4e-7-1.1e-6; visible counts equal;
selected token lists identical except where the differing blocks tie with the k-th best reference score
(4041/4096 rows identical on the random 4k case, the rest ties). Timing, 4 heads, page 16, scorer alone / end to
end (score + top-k + expand):

| shape | stock | **ours** | stock e2e | **ours e2e** |
|---|--:|--:|--:|--:|
| 4k chunk, ctx 12288 | 2,430 us | **752** | 2,547 | **1,049** |
| 4k chunk, ctx 32768 | 6,561 | **2,029** | 7,253 | **2,511** |
| 2k prefill | 210 | **65** | 273 | **184** |
| decode 1 row, ctx 8000 | 79 | **38** | 214 | **180** |
| MTP 16 reqs x 4 rows | 82 | **57** | 219 | **179** |

Expected serving effect: ~-65 ms per 4k chunk at 12k context (~9% of GPU-busy), more at 32k. Serving A/B by probe,
same day, TP=4 (`~/tp4tune/{qsc-r9k,qsc-stock}`): decode 148.5 -> 149.6 tok/s, c8 556 -> 533 (probe spread on c8
across earlier identical configs is 517-562), c16 768 -> 770, **prefill 8k 5,549 -> 6,188 tok/s (+11.5%)**. Full
20-pass BetterBench of the new default (`~/tp4tune/bb-final5`; same box, checkpoint and settings as bb-final4):

| | bb-final4 (stock scorer) | **bb-final5 (r9k scorer)** | Rob's image |
|---|--:|--:|--:|
| decode score | 125.0 | 123.2 | 134.1 |
| step p50 | 21.15 ms | 21.27 ms | 20.44 ms |
| TTFT p50 | 104 ms | 103 ms | 145 ms |
| prefill 2k / 8k / 16k / 32k | 5,106 / 5,624 / 5,726 / 5,532 | **5,607 / 6,281 / 6,406 / 6,158** | 5,279 / 5,711 / 5,977 / 6,106 |
| concurrency 1 / 2 / 4 / 8 / 16 | 118 / 197 / 292 / 433 / 543 | 117 / 193 / 311 / 437 / 561 | 126 / 197 / 303 / 427 / 542 |

Prefill +10-12% at every length and now ahead of Rob's image at all four (+6 / +10 / +7 / +1%); concurrency at
parity or better (the probe's c8 dip was noise); decode -1.4%, inside the spread of identical earlier runs
(123.1-125.1). Remaining gap to Rob: single-stream decode (his MTP-4) and step p50. Numerics: scores differ from stock only in fp32 summation order, so
selections differ only at exact ties; no paired eval needed. Commit fd950d4 (local).

### 2026-09-28 (cont.): re-profiled, the TP=4 down GEMM, and the PLE transposes

4k-chunk profile with the r9k scorer (`~/tprof-ours-prefill4kb`): GPU-busy 731 -> 653 ms (Rob 625); QSA 96 -> 19 ms
(Rob 46). Remaining per family: elementwise 160 vs 97, MoE 176 vs 141, dense GEMM 125 vs 118, all-reduce 148 vs 200,
GDN 14.5 vs 20. Per kernel:

- **Down GEMM at TP=4 ran on the decode kernel.** Per rank K=160; `r9k_moe_4bit_prefill` and `pick_moe_prefill`
  required K % 64 although the BK=32 tiles (cfgs 8-11) walk K as whole 32-slabs. 49 calls x 1.5 ms = 73 ms per
  chunk. Guard is now per tile (`r9k_moe_prefill_bk`), plus a 64x128 BK=32 tile (cfg 17). Bench (E=512, top-10,
  4096 tokens, `tuning/prefill_moe_bench.py --tp 4`): down 1,946 us (old) -> 1,222 (cfg 11) / 1,250 (cfg 17);
  folded variants slower (1,328 / 1,429); cfg 11 stays the default. Still only 27 TFLOPs: the epilogue writes 2-byte
  scattered elements (a wave's store covers two 32 B row fragments) and the down output is 210 MB per chunk --
  ablation queued (`-DR9K_PF_ABL=8`, no epilogue stores) to size an LDS-staged epilogue.
- **`torch.zeros` of the down buffer** (168 MB, 260 us x 49 = 10.6 ms per chunk) dropped when there is no expert
  map: the hot and cold passes write every routed row (`R9K_MOE_ZERO_DOWN=1` restores it).
- **PLE short conv: 27.5 ms per chunk, of which 23.4 ms are six ATen strided copies** -- the two
  `transpose(1, 2).contiguous()` around `F.conv1d` on [prefills, 4096, 10240] bf16 at ~14 GB/s (5.9 ms each; the
  depthwise conv itself is 1.1 ms). `kernels/r9k_transpose.hip` (64x64 LDS tiles, 16 B coalesced both ways) +
  `ple/short_conv.py`: the stock prefill method with the two transposes on our kernel, bound per PLE layer
  (`R9K_PLE_CONV=stock` keeps vLLM's). Test vs the stock method (bit-identical output and conv state) pending.
- Also seen: `hc_combine_norm` 575 us x 97 (Rob ~420), `hc_gate_mix` 322 us x 100 (Rob ~270) -- stock Triton, one
  program per row, 512-wide blocks; a HIP rewrite is the next elementwise item. `aten::mul` in the compiled MoE
  graph 2.8 ms; `w8a8_triton_block_scaled_mm` 34 ms (Rob's fp8 GEMM 29).

Serving probe (TP=4) after the down tile + no zero fill: decode 149.6 -> 148.8, c8 533 -> 550, c16 770 -> 783,
**prefill 8k 6,188 -> 6,639 tok/s** (+7.3%; +20% over the stock-scorer 5,549 of the same morning). Commit 115ae91.

**Prefill epilogue root cause (ablations, cfg 11 down GEMM, TP=4, 4096 tokens):** 1,191 us as is; 976 with the
stores skipped; **447 with no epilogue at all**. The epilogue gathered `As[row]` and `topk_w[row]` per output
element, re-read for every column tile: ~45% of the K=160 kernel. Both prefill kernels now build a per-row scale
table in LDS next to `sRow` once per routing block. Single-buffered BK=32 tiles (cfgs 18/19, half the LDS) had
been tried first and changed nothing, which ruled out occupancy. With the fix, folded weights, 4096 tokens:

| routed GEMM | MT kernel | cfg 11 / 15 (old defaults) | **cfg 17 (64x128 BK=32)** |
|---|--:|--:|--:|
| TP=4 down 2560x160 | 1,982 us | 855 | **811** |
| TP=4 gate_up 320x2560 | 969 | 980 | **899** |
| TP=2 down 2560x320 | 2,719 | 1,270 | **1,169** |
| TP=2 gate_up 640x2560 | 2,085 | 1,638 | **1,531** |

cfg 17 is the default for both GEMMs (`R9K_MOE_PREFILL_CFG` / `_CFG1` still override). Remaining epilogue cost is
the 2-byte scattered C stores (~200-350 us of the 811); staging the C tile through LDS for 16 B row stores is the
next step there. `tests/test_prefill_4bit.py`, `test_atiled_4bit.py`, `test_cache_moe.py` pass.

**PLE short conv:** `r9k_transpose16` 3,121 -> 444 us on the [1, 4096, 10240] transpose; the prefill method
26.5 -> 4.1 ms per call, output and conv state bit-identical to stock on one long prefill, a mixed batch with a
decode prefix, tiny lengths, NULL-block and empty-state cases (`tests/test_ple_conv.py`).

**Hyper-connection kernels (`kernels/r9k_hc.hip`, `hc.py`):** one 128-thread workgroup per (row, stream) for
combine_norm (combine result held as packed bf16 between the two passes; the first cut, one wave per stream with
fp32 registers, was register-bound at 558 us and, worse, 12.5 us vs stock's 3.7 in graph replay at decode widths --
it cost 0.8 ms/step in the first serving probe), one wave per (row, 256 columns) for gate_mix. Graph-timed, 4096
rows: combine_norm 607 -> 469 us, gate_mix 318 -> 310; at 1-64 rows stock is at the launch floor (3.4 us) and ours
0.5-2 us behind, so rows below `R9K_HC_MIN_ROWS` (256; gate_mix 1024) stay on stock -- decode never sees ours.
Combine output bit-identical to stock, norm output at stock's own bf16 error (`tests/test_hc_r9k.py`). Installed by
binding over `hyperconnection.py`'s imported names as torch.ops.r9700 custom ops before tracing (they sit inside the
compiled graph; the op signatures need full type annotations or the engine refuses to start).

**Round-2 serving result (2026-09-28, TP=4).** Probe after all of the above: decode 150.1, c8 521, c16 794,
prefill 8k 7,287 tok/s (5,549 with the stock scorer this morning: +31%). Full 20-pass BetterBench, same box,
checkpoint and settings (`~/tp4tune/bb-final6`):

| | bb-final4 (stock QSA scorer) | bb-final5 (r9k scorer) | **bb-final6 (round 2)** | Rob's image |
|---|--:|--:|--:|--:|
| decode score | 125.0 | 123.2 | **125.4** | 134.1 |
| step p50 | 21.15 ms | 21.27 ms | **21.06 ms** | 20.44 ms |
| TTFT p50 | 104 ms | 103 ms | **100 ms** | 145 ms |
| prefill 2k / 8k / 16k / 32k | 5,106 / 5,624 / 5,726 / 5,532 | 5,607 / 6,281 / 6,406 / 6,158 | **6,308 / 7,415 / 7,537 / 7,254** | 5,279 / 5,711 / 5,977 / 6,106 |
| concurrency 1 / 2 / 4 / 8 / 16 | 118 / 197 / 292 / 433 / 543 | 117 / 193 / 311 / 437 / 561 | 118 / 193 / 308 / 442 / 550 | 126 / 197 / 303 / 427 / 542 |

Prefill +19 / +30 / +26 / +19% over Rob's image at 2k / 8k / 16k / 32k (+24-31% over this morning's default);
concurrency at parity; single-stream decode unchanged (-6.5% vs his MTP-4). Sanity check below.

**Staged C stores in the prefill epilogues.** The WMMA D layout leaves each lane 8 rows of one column, so the
direct store was 2 bytes per lane. Each wave now writes its TM*16 x 16-column strip into LDS (the slab buffers,
free after the K loop) and stores whole 16 B row chunks. Whole-wave-tile staging (JS = TN) needs more LDS than the
slabs on cfg 17 and measured worse (down 745 us) than one strip at a time (690), so `PF_STAGE_MAX` = 16 KB keeps
cfg 17 at one strip. TP=4 folded 4096 tokens: down 811 -> 690 us (1,932 on the decode kernel this morning), gate_up
899 -> 883; TP=2: down 1,169 -> 1,077, gate_up 1,531 -> 1,510. A first cut put the stage past the slabs but left
the row tables at the slab end, so cfg 17's stage overwrote them (memory fault): the tables now sit past
max(slabs, stage). Probe: decode 150.6, c8 522, c16 794, prefill 8k 7,346 tok/s.

**Re-profile after round 2 (`~/tprof-ours-prefill4kc`, 4189-token prompt, TP=4):** GPU-busy **548 ms vs Rob's
625** (731 this morning), span 602 vs 666; 8,051 kernel launches vs his 5,393.

| family | ours | Rob | biggest items |
|---|--:|--:|---|
| all-reduce | 146.6 | 200.0 | ar4_pack_push 58.2 x202, push2 27.1, reduce_pack_push 26.6, decode 8.3 x404 |
| dense GEMM | 127.1 | 117.9 | `_w8a8_triton_block_scaled_mm` 34.1 x196 vs his fp8hip 28.7; hipBLASLt bf16 equal in total |
| MoE | 118.3 | 141.1 | cfg 17 gate_up 34.8 + down 28.2 vs his m64 pair 101.7; moe_sum 21.4 vs 23.7 |
| elementwise | 112.5 | 97.0 | hc_combine_norm 41.2 vs 41.4 (equal now), hc_gate_mix 32.7 vs 26.9, misc glue +9 |
| QSA | 18.6 | 46.2 | |
| GDN | 14.4 | 20.3 | |

Kernel-side gaps are single digits now. The largest remaining item on eager 4k chunks is the 54 ms of launch gaps
(span - busy; Rob 41 ms), i.e. the CPU floor: cudagraph capture of prefill chunks above 2048 tokens was rejected
on 2026-09-25 for padding cost, when a chunk's GPU time was 25% higher than now -- re-measured and **rejected
again** (`~/chain-cg4k.out`, same day, TTFT ms): 2166 tokens 293 -> 323, 3086 394 -> 435 (padding to the next
capture size), 6076 798 -> 797, 12.7k 1731 -> 1726, 25k 3514 -> 3505 (no gain: the eager 4k chunk is not
CPU-bound in practice), KV capacity 258,389 -> 176,128 tokens. Capture sizes stay dense to 2048.

Current TTFT sweep (tokens: ms / tok/s): 236: 86 / 2,736; 695: 134 / 5,184; 1,325: 228 / 5,800; 2,166: 293 / 7,391;
3,086: 394 / 7,825; 6,076: 798 / 7,612; 12,746: 1,731 / 7,365; 25,367: 3,514 / 7,219 (2026-09-25: 287 / 284 / 259 /
564 / 564 / 1,075 / 2,248 / 4,509 ms).

Remaining elementwise glue after the hc kernels (38.6 ms vs Rob's ~29): `r9k_silu_mul_quant_fp8` 6.4 ms x102,
`__amd_rocclr_copyBuffer` 5.0 ms x244, stock `per_token_group_quant_8bit` 4.2 x196 (input quant of the block-fp8
GEMM), `r9k_quant_rows_fp8` 5.4, `aten::mul` in the compiled MoE graph 2.9 -- nothing above 1% of the chunk.

### 2026-09-28 round 3: decode -- where the 21 ms/step goes

Target: decode 125.4 vs Rob's 134.1 (step 21.06 vs 20.44 ms, MTP-3 vs his MTP-4). Decode profiles at MTP-3, MTP-4 and
of Rob's image (`~/tprof-{ours-decode3,ours-decode4,rob-decode}`, 256-token generation, rank 0; per-step scripts
`~/dec-step.py`, `~/dec-gaps.py`, `~/dec-eager.py`, `~/dec-tail.py` on VM100):

| per step | ours MTP-3 | ours MTP-4 | Rob MTP-4 |
|---|--:|--:|--:|
| kernel nodes | 3,254 | 3,363 | 1,639 |
| GPU busy (ms) | 15.8 | 17.1 | 18.1 |
| unprofiled step (ms) | 21.06 | 22.5 | 20.8 |

**Our GPU time per step is lower than Rob's; the deficit is the node count.** The target forward is one HIP graph
(2,823 nodes); the gaps sit inside its replay, not at eager launches (97% of kernels come from `hipGraphLaunch`,
the CPU runs ~30 ms ahead). Rob's gaps are a uniform ~5.6 us per node under the profiler; ours bunch after runs
of tiny ATen nodes. A microbench of graph replay on this ROCm (`~/graphnode-bench.py`) puts a tiny dependent node
at 2.85 us (eager launch 8.7), and unprofiled arithmetic gives ~1.6 us of non-busy time per node for both stacks
(ours 5.2 ms over 3,254, Rob 2.4 over 1,639). `hipGraphLaunch` of the 2,823-node graph also costs the CPU ~6 us
per node under the profiler (16.8 ms per call), so bursts of small nodes starve the GPU. Either way: fewer, fatter
nodes. Where our extra ~1,600 nodes per step come from (48 layers + 3 MTP passes):

- QSA indexer glue (~30 per QSA layer, 15 passes): GemmaRMSNorm takes ir.ops.rms_norm's native path (fp32
  `1 + w` weight, ~10 ATen launches per norm, two norms), and the 1-D neox rope runs `ApplyRotaryEmb.forward_static`
  (~9 launches; flash_attn's Triton rotary is not in the image). ~400 nodes.
- Hyper-connection per block: two wvSplitK skinny GEMMs (the merged 336x10240 down+inject at 14.4 us, the 10240x320
  up at 12.2 us: 6.5 MB each at ~500 GB/s) plus Triton `hc_silu`, `hc_gate_mix`, `hc_combine_norm`. Rob runs one
  `hc_fused_kernel` (38 us). ~200 excess nodes, and ~26 us of GEMM per block that a streaming kernel could halve.
- Shared expert as a separate dense path: two mxfp4 GEMMs on the default (2,4,2) config = 5 workgroups for the
  320-wide gate_up (18 us for 0.4 MB), two row quants, Triton silu-mul, a 6 us wvSplitK for the [1 x 2560] expert
  gate, sigmoid, mul. ~250 nodes.
- GDN layer glue (conv update, cat, index copies, elementwise) ~10 per layer; MoE align/count/topk 3 per layer;
  10 NCCL all-gathers per step (the MTP head's `fc_embedding` / `fc_hidden` are built with `gather_output=True`,
  2 per draft at ~47 us, plus the logits all-gather per head: 79 us drafts, 199 us target). Rob pays 5 x 170 us
  through libr4d.
- GPU-time items on top: the router gate is a plain bf16 `ReplicatedLinear` -> hipBLASLt picks a 16x16x32 tile
  (19.6 us x 51 for 2.6 MB); the TP=4 block-fp8 and dense shapes were all untuned (`tuned.json` only had TP=2 and
  27B shapes); block-fp8 projections run at ~480 GB/s whatever the config (kernel-bound, 22 + 10 us per layer).

**Done, in order, each measured with the 1-minute probe (`~/tp4tune.sh`, TP=4, MTP-3):**

1. `tuned.json` for the TP=4 shapes (`tuning/tune_dense.py` SHAPES extended): mxfp4 320x2560 11.2 -> 6.7 us,
   2560x160 6.9 -> 5.6, LM head 62080x2560 156 -> 143; fp8block 4096x2560 23.0 -> 21.9, 2560x1536 9.5 -> 8.3,
   3584x2560 20.9 -> 19.7 (M=4, graph-timed).
2. **Router GEMM** (`kernels/r9k_router.hip`, `r9700_vllm/router.py`): bf16 x bf16 -> fp32 accumulate, one wave per
   (expert, K split), rounded once to bf16 as hipBLASLt's bf16 output is (differs only in summation order;
   `tests/test_router_r9k.py`: top-10 agreement 1.0, flips only on near-ties). Bound as the forward of every
   `mlp.gate` (48 + MTP). 5.4 us at 4 rows vs 7.5 warm / 19.6 served; above 8 rows it loses to hipBLASLt (re-reads
   the activations per expert), so the op switches to `torch.mm(out_dtype=fp32)` + cast at run time. Split 4/8 at
   4 rows is anomalously slow (30 us) -- unexplained, split 2 is the default.
   Probe `rt-r9k` (1 + 2): step 21.37 -> 20.93 ms, decode 150.6 -> 154.9 tok/s, c8 522 -> 525, c16 794 -> 843,
   prefill 8k 7,346 -> 7,324.
3. **QSA indexer glue** (`kernels/r9k_norm_rope.hip`, `r9700_vllm/attn/indexer_glue.py`): one launch per
   (Gemma norm [+ rope]) per head set, fp32 math with the native path's rounding points, neox rope on bf16 cos/sin
   with the three bf16 roundings of `forward_static`, and the MRoPE section rule of `_triton_mrope_forward`
   (interleaved or concatenated, [3, T] positions) so both q (1-D or MRoPE positions) and k (MRoPE) are fused.
   Bound as `project_qk` / `normalize_compressed_keys` on every QSAIndexer (`R9K_QSA_GLUE=stock` reverts).
   `tests/test_indexer_glue_r9k.py`: bit-equal to the stock functions except fp32-order flips of the sum of squares.
   Graph-timed at 4 tokens: stock norm+rope 58.5 us -> 6.0; norm 34 -> 4.0. The first serving build still took
   stock's rope after our norm on 12 calls/step: `normalize_compressed_keys` gets its positions as `[:, 0]` of a
   `[T, 3]` buffer (1-D, element stride 3) and the fit check demanded unit stride. The kernel now takes a token
   stride for the positions (`ptok`; the MRoPE `[3, T]` transpose view with row stride 1 / token stride 3 is covered
   the same way), and the test has stride-3 / stride-5 / MRoPE stride-2 cases. Confirmed in serving: the round-3
   eval server logs no "not fused" warning, so both indexer paths are on the one-launch glue.
4. **Hyper-connection decode mix** (router kernel with an `hc_silu` epilogue on the merged down+inject GEMM, and
   `r9k_hc_up_mix` in `kernels/r9k_hc.hip`: up GEMM + sigmoid + gated mean over the four streams in one wave per
   four output dims, gate rounded to bf16 as the GEMM output was, streams summed in stock's order). Bound as
   `GatedResidual.mix` / `combine_and_mix` (`hc.install_mix`, `R9K_HC_MIX=stock` reverts); rows above 8 take
   stock's path inside the ops. `tests/test_hc_mix_r9k.py`. Cold microbench at 4 rows: stock down GEMM 17.6 us ->
   9.6 (split 8), up GEMM 9.4 -> up+mix 13.2 (weight rows read 64 B per lane-group; still slower than wvSplitK's
   up GEMM alone but it absorbs sigmoid + mean); the whole mix 48 -> 25.5 us.

**Probe with all three (`mix-r9k`, vs `rt-r9k` / this morning's `epi-r9k`):** step 19.64 ms (20.93 / 21.37),
decode 163.4 tok/s (154.9 / 150.6), c8 559 (525 / 522), c16 868 (843 / 794), prefill 8k 7,387 (flat). -8.1% step
time in the round. Rob's probe on this box: 169 / 503 / 769 / 6,446. Paired 800-question chain-of-thought eval
(round-2 default vs round 3) and the full BetterBench (`bb-final7`) ran after the hardware move (2026-09-29):

**bb-final7 (round 3, TP=4, MTP-3):** decode 139.1 tok/s, step p50 19.01 ms, TTFT p50 88 ms, prefill
6444 / 7514 / 7630 / 7349 at 2k / 8k / 16k / 32k, concurrency 131 / 216 / 324 / 457 / 598 at c1..c16, sanity 8/8.
Against bb-final6 (round 2): decode +11%, step -10%, prefill +2..+3%, c8 +3%, c16 +9%; single-stream decode is
ahead of Rob's image (134.1) for the first time.

**Paired eval: round 3 is WORSE.** 800 chain-of-thought questions at conc 1: qt-r2 (round-2 default) 98.00% vs
qt-r3 96.88%; discordant 10 (r2-only right) vs 1 (r3-only right), McNemar p = 0.012; identical output on 31.5%.
One of the three round-3 defaults perturbs the answers -- the router GEMM is the suspect (its near-tie flips change
expert selection; the glue and hc mix differ from stock only in fp32 summation order). Bisect queued
(`~/chain-bisect.sh`): router off / router alone / everything on (with round 4), each paired against qt-r2. Round 3
must not stay the default until the culprit is found or reverted (the knobs: `R9K_ROUTER=stock`,
`R9K_QSA_GLUE=stock`, `R9K_HC_MIX=stock`).

**Bisect (2026-09-29, `~/evalcmp.py` over every saved run):** router off (`qt-r3a`) 97.38%, router alone (`qt-r3b`)
97.38%, each 6 / 1 against qt-r2 (p = 0.125). But qt-r2's 98.00% is the outlier: the four earlier runs of the same
default (qt-ref / qt-new / qt-def030 / qt-ar4) sit at 97.50-97.62% and qt-r2 beats them 4 / 0, 4 / 1, 3 / 0 -- it
got four questions right that the default usually misses. Against those runs, r3a and r3b are 4 / 2 and 5 / 3, the
same as the runs against each other (5 / 4, 4 / 4): noise. Only the full round 3 (qt-r3, 96.88%) still leans worse
against the older runs (6 / 0, 7 / 1, 7 / 2), and one question (3195b2b45ffb) is wrong in all three round-3 legs and
in none of the five earlier runs. The one round-3 change no leg switched off is the TP=4 `tuned.json` entries
(1554e9a: block-fp8 dense shapes and the LM head take different split-K configs, so their fp32 sums round
differently). Queued: a replicate of round 3 (`qt-r3c`) and the round-2 config again (`qt-r2b`, the noise floor).
If qt-r3c repeats ~96.9% the tuned entries get their own leg; if it lands at 97.5% the round-3 verdict was noise.

Remaining node budget (eager profile with Python stacks, `~/dec-stack.py`, ~2,770 kernels/step after the round):
GDN spec-decode glue in `qwen_gdn_linear_attn.py` (cat / reshape / contiguous / zeros / copies, ~9 per layer x 36:
vLLM fuses only the non-speculative decode path), the shared expert's own quant + silu + expert-gate GEMM + sigmoid
+ mul (~5 per layer), `moe_align_block_size` + `count_and_sort` + `topk_softmax` + `moe_sum` + the top-k weight copy
(5 per layer, stock), the block-fp8 input quant (2 per layer), 10 NCCL all-gathers.

### 2026-09-28 round 4: GDN speculative-decode core in one launch

The largest remaining block of decode glue was the Gated DeltaNet layers (36 of 48). Stock vLLM has a fused CUDA op
for exactly the MTP decode step (`fused_gdn_decode_post_conv_mtp`: gating + recurrence + gated norm after the conv
update), but it is not built in the ROCm image, so `forward_cuda` runs the Triton recurrence inside glue: `b` / `a`
made contiguous, a zeroed core buffer, the q/k/v `cat`, the conv update, `fused_sigmoid_gating_delta_rule_update`,
the output copy, `layer_norm_fwd`, and three reshape copies -- 11 HIP-graph nodes per layer, ~400 per step (the
eager stack profile: `reshape` x108, `contiguous` x72, `cat` x36, `zeros` x36, copyBuffer x36 from
`qwen_gdn_linear_attn.py`).

`kernels/r9k_gdn.hip` `r9k_gdn_decode_mtp` (`r9700_vllm/models/gdn.py`): one workgroup per (sequence, value head)
holds the 128x128 fp32 state in registers (64 per thread, K halves paired by shuffle), runs the sigmoid gating and
the delta-rule recurrence over the sequence's spec tokens (state stored per token at its slot, sequences with a
null initial slot skipped, padding rows zeroed), and applies RMSNormGated (RMS over V through LDS, silu gate) in the
same launch. It reads the merged in_proj row and `ba` directly (any row strides), so nothing is copied. The layer's
forward becomes in_proj -> `torch.ops.r9700.gdn_core` -> out_proj; inside the op a pure spec-decode batch takes the
conv update + the kernel (2 launches), anything else (prefill, mixed batches, warmup) takes stock's core + norm.
`R9K_GDN_DECODE=stock` restores vLLM's forward. `tests/test_gdn_decode_r9k.py` checks it against vLLM's Triton
composition: output flips ~2e-4 of the elements (fp32 summation order), fp32 state within 1e-4, bf16 state by the
same flip rule, padding and skipped sequences zero.

**Shared expert in four launches** (`r9700_vllm/moe/shared.py`, `torch.ops.r9700.shared_expert`): the MoE block's
shared expert (a gated `Qwen3NextMLP`, 48 + MTP calls per step) ran as our dense MXFP4 gate_up (row quant + GEMM),
`SiluAndMul`, down (quant + GEMM), then the expert gate's bf16 `F.linear` (wvSplitK), `sigmoid` and `mul`: eight
launches. Now the row quant also computes the gate dot and its sigmoid (`r9k_quant_rows_fp8_gate`, stock's rounding
points: bf16 logit, bf16 sigmoid -- bit-equal to stock's gate on every tested row), the activation is the routed
MoE's `silu_mul_quant_fp8`, and the sigmoid is folded into the down GEMM's per-row epilogue (the router-weight
fold), so the output is rounded once. Large M takes the same prefill / A-tiled GEMM configs as the dense linear.
`tests/test_shared_expert_r9k.py`: TP=4 and TP=2 shapes at M = 1..1024, ours at least as close to the fp64
reference as stock (equal or better on 13 of 14 cases, within 0.7% on the other), 0.3-0.9% from stock in relative
norm (the activation's fp8 codes). `R9K_SHARED_EXPERT=stock` restores vLLM's forward.

**Unit timing (graph replay, idle GPU, `~/gates-gdn.out`), per layer call:**

| | stock | ours |
|---|---|---|
| GDN core, 1 sequence x 4 tokens (fp32 state) | 43.3 us (9 nodes) | 13.8 us (1) |
| GDN core, 4 sequences | 52.7 | 19.5 |
| GDN core, 16 sequences (fp32 / bf16 state) | 79.0 / 80.0 | 51.9 / 53.9 |
| shared expert, M = 1 / 4 / 16 | 44.1 / 46.1 / 46.7 (8 launches) | 18.3 / 18.5 / 18.9 (4) |

At batch 1 that is ~1.1 ms/step over the 36 GDN layers and ~1.2 ms over the 48 shared experts, against a 19.0 ms
step. At 16 sequences the GDN kernel is state-bandwidth bound (4 x 64 KB of state stores per head per layer, as
stock's); a split-V variant would help there if the concurrency numbers ask for it.

**Serving (first attempt: GPU memory fault on the first request).** Eager mode was fine and switching the fused
kernel off did not help, so the kernel was not the cause: vLLM lists its own `vllm::qwen_gdn_attention_core` in
`CompilationConfig._attention_ops`, the default piecewise-graph splitting list, so the GDN core runs eagerly at
every step of a piecewise prefill graph (its Triton kernels take that step's sequence layout). Our op replaced it
in the traced forward without that entry, so a prefill piece captured our op's kernels for the capture batch and
replayed them for every later one. `register_splitting_op()` now appends `r9700::gdn_core` to that class list at
plugin load (the install log reports whether the engine's `splitting_ops` has it). Rule for future plugin ops:
anything that replaces a splitting op must be added to that list. (Also caught on the way: Flash-Next's output
gate is `sigmoid`, and the first fit check only took silu, so the core silently sat out of one serving run; the
check now logs its reason.)

**Probe, GDN core only (`gdn-live`, shared expert stock) vs the round-3 default on the same build (`r4-stock`):**
decode 166.5 -> 170.3 tok/s (+2.3%, 3.09 -> 3.12 tok/step), c8 573 -> 561, c16 852 -> 865, prefill 8k 7407 ->
7171 (the last three within the probe's run-to-run band). Smaller than the ~1.1 ms/step the unit timing
suggested: in a captured decode graph the removed nodes were cheaper than in the eager profile.

**Probe, both fusions (`se-r9k`):** decode 190.9 tok/s (3.2 tok/step), step 17.16 ms, c8 623, c16 891, prefill 8k
7,233 -- vs `r4-stock` 166.5 / 19.3 / 573 / 852 / 7,407. The shared expert's four launches are worth more than the
GDN core's one: 48 calls per step against 36, and the router-weight fold removes the expert-gate GEMM outright.

**bb-final8 (round 4 = round 3 + GDN core + shared expert; TP=4, MTP-3, 20 passes):** decode **159.2 tok/s**, step
p50 **16.79 ms**, TTFT p50 94 ms, prefill 6,408 / 7,365 / 7,451 / 7,182, concurrency 151 / 231 / 350 / 478 / 635,
sanity 8/8. Against bb-final7: decode +14%, step -12%, concurrency +5..+15%, prefill -0.5..-2.4% (within the
run-to-run band; the GDN op now runs eagerly in the prefill pieces exactly as stock's did). Against Rob's image:
decode +19%, step -18%, TTFT 1.5x, prefill +18..+29%, concurrency +12..+20%.

**Quality, round 4 (`qt-r4c`, everything on, GDN core engaged):** 97.50% (780/800); paired against qt-r2 5 / 1,
McNemar p = 0.219, no detectable difference. Sanity 8/8.

**The round-3 question, closed (2026-09-29):** the round-2 configuration reruns at 97.88% one question apart
from its first run (qt-r2b vs qt-r2: 1 / 0, 87% identical outputs), so that build is near-deterministic, and the
round-3 family is consistently below it: 96.88 / 97.25 (round 3 twice), 97.38 / 97.38 (router off / router only),
97.00 with the round-2 tuning table (`qt-r3t`, 8 / 0, p = 0.008 -- so the TP=4 tuned entries are not the cause;
`test_tuned_cfgs` also passes on them), 97.50 with round 4 on top, 97.88 with the shared expert but the GDN core
off. Per component the drop is at the edge of detection (6 / 1, p = 0.125 each); combined it is real (8-10 / 1).
Read: the three round-3 decode fusions each perturb the numerics a little (fp32 order in the glue and mix, near-tie
routing flips in the router) and together cost ~0.5 point on this eval, which is inside the band the older builds
spanned (97.5-97.6) but below the round-2 build's 97.9-98.0. It is a knob, not a bug: `R9K_ROUTER=stock
R9K_QSA_GLUE=stock R9K_HC_MIX=stock` restores round-2 numerics at ~10% of decode. Shipped as-is for v0.2.0 with the
trade stated; the cleaner fix is per-fusion evals against a bit-reproducible baseline, one at a time.

*Superseded the same evening: on the full 1,319 questions no fusion, alone or together, costs anything (97.04% for
the default against 96.82% for the round-2 numerics, 5 / 8). The half point was the first-800 slice. See "the decode
fusions one at a time" below.*

### 2026-09-29: Flash-Next at TP2 with offloaded experts, on the 4-card box

Brian asked for a two-card benchmark with the experts in host RAM (the v0.1.0 way to run Flash-Next). Three
launches to get a fair one:

1. **GPUs 0,1 with the current defaults: no KV memory at all.** The 2048-token prefill graphs take 3.4 GiB per
   card at TP2 (1.8 at TP4) on top of the 15.8 GB expert cache; `CGSIZES=` (vLLM's default sizes) restores it.
2. **GPUs 0,1 with default graph sizes: 44k tokens of KV, and the quick probe collapsed at concurrency** (c8 54,
   c16 49 tok/s below the 81 single-stream; prefill 8k 1,379). 240 expert slots gave 105k tokens of KV and the same
   collapse, so KV was not it.
3. **One card per PLX switch (GPUs 0,2) fixed it:** 106 / 94 / 99 / 2,241 (single / c8 / c16 / prefill 8k) at 240
   slots, 114 / 132 / 120 / 2,464 at 270. The offloaded experts stream from pinned host memory down each switch's
   single Gen3 x16 uplink; two cards on one switch halve each other's read bandwidth, which is invisible when
   everything sits in VRAM (the TP4 numbers) and dominant when it does not. Also 154k tokens of KV on the split pair
   against 105k on the same-switch one at equal settings (the pair's P2P scratch differs).

**bb-qfn-tp2-offload (GPUs 0,2, OFFLOAD_GB=34/rank, 270 slots, CGSIZES default, otherwise v0.2.0 defaults; full
BetterBench 20 passes):** decode 93.8 tok/s, step p50 25.68 ms, TTFT p50 555 ms, prefill 2,099 / 3,163 / 3,434 /
3,329, concurrency 84 / 104 / 108 / 111 / 95, KV 71,859 tokens, sanity 8/8.

**The "concurrency loss" against 2026-09-20 was the harness, not the build.** The first write-up of this section
compared today's quick probe (c8 132) with the September-20 figure (conc-8 206) and called the difference an open
regression. The two were never the same measurement: 206 came from `bench/harness.py` (eight copies of ONE prompt,
a nonce apart, on the launcher's `NSEQ=4`), 132 from `~/probe.py` (two each of four different prompt types, code /
prose / json / math, at `NSEQ=16`). Four launches on GPUs 0,2, both harnesses against each server (`~/chain-hx.sh`,
`~/tp4tune/hx-*`):

| leg | step ms | probe single | probe c8 (mixed) | harness single | harness agg8 (one prompt) | harness prefill 2000w | KV tokens |
|---|---|---|---|---|---|---|---|
| v0.2.0 defaults, NSEQ=16 | 26.31 | 114.1 | 117.5 | 104.4 | 226.5 | 2,756 | 71,859 |
| v0.2.0 defaults, NSEQ=4 | 26.04 | 111.1 | 110.4 | 105.5 | 210.9 | 2,755 | 117,274 |
| September-20 knobs, NSEQ=4 | 32.61 | 92.2 | 93.6 | 86.1 | 207.2 | 2,164 | 124,625 |
| September-20 knobs, NSEQ=16 | 32.72 | 93.2 | 110.7 | 83.7 | 201.8 | 2,257 | 57,478 |
| 2026-09-20, as recorded | | | | 84.5 | 206 | 2,165 | |

("September-20 knobs" = prefix caching on, stock block-fp8, fp8 LM heads, stock QSA / scorer / qk-rope, and the
five decode fusions at stock, on today's build and image.) The old knobs reproduce the old numbers to within 2%,
so neither the move to four cards nor vLLM 0.30 changed this configuration. On the same harness v0.2.0 is +24%
single-stream, +27% prefill and -20% step time. The mixed probe reads c8 94-118 on every leg, old knobs included;
this morning's 132 was the top of that band.

**Why the mixed batch is slow: prompt diversity, not any one prompt** (`bench/mix.py`, conc 8, 256 tokens, median of
3, wall ms per step includes the prefill):

| 8 concurrent requests | v0.2.0: tok/s | tok/step | ms/step | Sept-20 knobs: tok/s | tok/step | ms/step |
|---|---|---|---|---|---|---|
| code x8 | 275 | 3.48 | 94 | 249 | 3.42 | 109 |
| prose x8 | 218 | 2.26 | 83 | 253 | 2.23 | 70 |
| json x8 | 338 | 3.86 | 92 | 439 | 3.83 | 69 |
| math x8 | 263 | 3.58 | 108 | 279 | 3.40 | 97 |
| harness prompt x8 | 223 | 2.60 | 95 | 244 | 2.64 | 84 |
| mixed (2 each of code / prose / json / math) | 119 | 3.14 | 213 | 112 | 3.03 | 219 |

Every prompt type runs at 218-439 tok/s on its own; two each of four types run at 112-119, at ordinary acceptance,
because the step itself takes 2.3x longer. With the experts in host RAM a step costs what it has to fetch. (First
reading, written before the counters existed: "a mixed batch routes to more distinct experts than the 270 slots
hold". Measured below: it does not -- it routes to 85 -- but it misses three times as many of them.) It is also why the
BetterBench concurrency curve for this configuration is flat from 2 requests up (104 / 108 / 111 / 95): its corpus
is mixed.

Two cautions from the same table. Homogeneous conc-8 does not separate the two builds: the harness prompt read
226.5 vs 201.8 in one pair of launches and 223 vs 244 in the next, so +-10% is launch-to-launch. And json x8 is
lower on v0.2.0 (338 vs 439, tight ranges within each launch, same acceptance, 92 vs 69 ms per step) -- one launch
each, so not a finding yet, but it is the first thing to repeat when this configuration gets its own pass.

**What bounds a mixed batch, measured (2026-09-29, same day).** The reading above was half right. Opt-in counters in
the expert cache (`R9K_EXPERT_CACHE_STATS=1`: device-side totals per cached layer of steps, distinct routed experts,
inserts, experts read through from host, steps over the insert threshold; a reader thread logs them every
`R9K_EXPERT_CACHE_STATS_SEC`; `tests/test_cache_stats.py` checks them against a host recount) on three launches,
rank 0, decode windows of `bench/mix.py -v`:

| 8 submitted (4 run, see below) | forward passes /s | ms per forward | distinct experts per layer | inserted | read through | miss | steps over threshold | host -> VRAM per forward |
|---|---|---|---|---|---|---|---|---|
| code x8 | 20.8-21.6 | 46-48 | 66-71 | 2.1-2.2 | 0.8-1.1 | 4.4-4.6% | 1.7-2.4% | 175-193 MiB |
| mixed (code / prose / json / math) | 10.9-11.6 | 86-92 | 85-87 | 10.3-11.1 | 1.4-1.7 | 14.0-14.5% | 2.3-2.6% | 714-751 MiB |
| one request (code) | 28-33 | 30-36 | 26-31 | 1.9-2.1 | 0-0.5 | 7.4-8.6% | 0-1% | 114-154 MiB |

1. **It is not the insert threshold** (with four requests running; with eight it is -- see "The full BetterBench
   of that candidate regressed" below). The LRU manager inserts nothing on a step that routes to more than
   `R9K_LRU_THRESH` x slots = 135 distinct experts; that was the suspect, and a mixed decode step is over it 2-3% of
   the time. Nor does a batch route to more experts than there are slots (85 of 270).
2. **It is the miss rate times the link.** A mixed batch misses 14% of its routed experts at every step, a
   single-type batch 4.5%. Each miss is 1.245 MiB per rank, and the copy kernel already runs at the link rate:
   `tuning/lru_gather_bench.py`, host -> VRAM, 11.4-13.5 GB/s from 4 inserts up on every launch grid (PCIe 3 x16).
   714-751 MiB per forward pass is 66-69 ms of the 86-92; a single stream still pays ~13 ms of its 36.
3. **The miss rate is the checkpoint's routing, not the policy.** Flash-Next routes diffusely: the busiest expert of
   a layer carries 1-2% of the routing, and 90% of it needs 216-365 of the 512 experts. By the routing profile
   shipped with the checkpoint the best static set of S experts per layer misses 24.0 / 17.4 / **13.3** / 9.9 / 7.1 /
   4.8 / 2.5% of the mass at S = 200 / 240 / **270** / 300 / 330 / 360 / 400 -- a mixed batch behaves like the global
   distribution, one prompt type like a much narrower one that the LRU follows. Re-splitting the same 12,960 slots
   across layers by marginal mass (195-367 per layer) only moves 13.3% to 12.8%: not worth a knob.
4. **A wider copy grid is worth ~8% on the mixed batch and nothing elsewhere:** `R9K_LRU_GATHER=64,16` against the
   default 8,16 is 15% faster at 13 inserts (1,262 vs 1,492 us) and equal at 2; served, mixed 106.5 -> 115.2 and
   113.9 tok/s (two launches), code x8 and single-stream unchanged. Candidate default, not yet flipped.

**And a second, separate limit: only four requests run at once.** Engine iterations against per-request steps
(`vllm:iteration_tokens_total_count` vs `vllm:spec_decode_num_drafts_total`, `--max-num-seqs 16`):

| submitted | tok/s | ms per iteration | requests per iteration |
|---|---|---|---|
| 1 / 2 / 4 / 8 x code | 100 / 166-181 / 265-285 / 257-275 | 35 / 38-42 / 47-50 / 50 | 0.99 / 1.9 / 3.8 / 3.7-3.9 |
| 1 / 2 / 3 / 4 mixed | 91 / 100 / 103 / 111 | 38 / 44 / 60 / 76 | 1.0 / 1.6 / 2.1 / 2.7 |
| 8 / 16 mixed | 111-116 / 113-119 | 94-95 / 100-103 | 3.4-3.5 / 3.6-3.8 |

With four short requests running the KV pool is 93.5% used (46.75% with two) and the others wait with reason
`capacity`: every request takes 18 blocks of a 77-block pool whatever its length (the per-request recurrent state),
and at TP2 with the experts cached the pool is what is left after 25.84 GiB of weights and 3.24 GiB of peak
activation: 0.87 GiB on these launches, 1.38 GiB (71,859 tokens, so six requests) on the BetterBench one. Prefix
caching on does not change it (same 4). So on this configuration concurrency above 4-6 buys nothing for any
workload, and for a mixed one concurrency buys little at all (91 -> 111 tok/s from 1 to 4 requests) because every
additional prompt type brings its own experts down the same link. That is the flat BetterBench curve (104 / 108 /
111 / 95 from 2 to 16), and it is a property of two cards on a PCIe 3 host, not a defect to fix in a kernel.

**The knobs, measured (2026-09-29 evening, `~/chain-knobs.sh`, `~/tp4tune/t2-*`).** Same harnesses on every leg;
every configuration after the first was launched twice and measured on the second launch (see 1.):

| leg | KV pool (tokens) | weights / peak act. / graphs (GiB) | single | code x8 | json x8 | mixed x8 | requests per iteration, code x8 | prefill 8k |
|---|---|---|---|---|---|---|---|---|
| defaults, NSEQ=16, first launch | 45,415 | 25.84 / 3.24 / 2.42 | 100.5 | 279 | 329 | 109 | 3.9 | 2,473 |
| the same, second launch | 71,859 | 25.41 / 3.16 / 2.26 | 101.2 | 283 | 347 | 118 | 3.9 | 2,439 |
| + copy grid 64,16 | 71,859 | 25.41 / 3.16 / 2.26 | 105.7 | 286 | 354 | 139 | 3.9 | 2,455 |
| + **NSEQ=8** | 120,724 | 25.41 / 2.23 / 1.53 | 102.4 | **401** | **535** | 130 | **7.6** | 2,393 |
| + NSEQ=8, NBT=2048 | 168,439 | 24.79 / 1.95 / 1.53 | 104.4 | 413 | 538 | 123 | 7.3 | 1,821 |
| + NSEQ=8, 240 slots | 202,931 | 23.84 / 2.23 / 1.53 | 96.4 | 327 | 470 | 93 | 7.6 | 2,221 |
| + NSEQ=8, UTIL=0.96 | 154,067 | 25.41 / 2.23 / 1.53 | 103.0 | 393 | 536 | 127 | 7.5 | 2,383 |

1. **The first launch of a configuration gets a smaller KV pool than every later one**: 45,415 against 71,859
   tokens (0.87 against 1.38 GiB), and the same on every other leg (94k -> 121k, 102k -> 168k, 155k -> 203k). The
   launch that compiles keeps 0.43 GiB more in "weights + non-torch" and a larger activation peak; the pool is sized
   from what is left. It is vLLM's accounting, not ours, and it is why the afternoon's fast legs (all first
   launches) saw four requests where the morning's BetterBench launch would have run six. Operationally: restart
   once after changing a knob. The launcher header says so.
2. **NSEQ=8 is the concurrency fix for one prompt type: +42% to +54% at conc-8** (283 -> 401, 347 -> 535), single
   stream unchanged. Graphs and activation for 16 sequences cost 1.66 GiB that went to requests which could not run;
   at 8 the pool holds all eight (7.6 per iteration against 3.9). `serve/flashnext.sh` now defaults `NSEQ=8`
   (`NSEQ=16` for TP4, as every TP4 number here was measured).
3. **The mixed batch does not move with any of it**: 118-139 tok/s on every leg that kept 270 slots, whatever runs
   at once. Its second-launch readings scatter more than the knobs differ (the same eight requests read 121-127 in
   the iteration harness on all four of those legs), so the copy grid's +17% in one harness and 0% in the other is
   one more reason to call it "up to +10%", which is what the kernel timing predicts. It costs nothing elsewhere and
   cannot change outputs (a copy; data checked in the bench, sanity 8/8), so `R9K_LRU_GATHER` now defaults to 64,16.
4. **240 slots: rejected.** 80k more tokens of KV, and -28% on the mixed batch, -18% / -12% on one prompt type, -6%
   single stream, -7% prefill: the misses cost more than the room buys, as the routing profile said (17.4 against
   13.3%).
5. **NBT=2048: rejected as a default.** 48k more tokens of KV for -24% prefill (2,393 -> 1,821) and nothing else. It
   is the knob for someone who needs the context more than the prompt speed.
6. **UTIL=0.96: ran clean, not adopted.** 33k more tokens and the same throughput, 8/8 sanity; but these legs never
   held a long context while eight requests ran, and the graph pool already sits outside the budget. It needs a
   soak at 32k before it can be a default.

**The full BetterBench of that candidate regressed, and the insert threshold was the cause after all.**
`bb-qfn-tp2-n8` (NSEQ=8, copy grid 64,16): single stream, step, TTFT and prefill equal to the morning's run, and
concurrency 84 / 104 / 115 / **77 / 76** against 84 / 104 / 108 / 111 / 95. Letting eight requests run removed the
queue (TTFT at 8 concurrent 7.1 s -> 1.1 s) and lowered the total. Counters under BetterBench's own concurrency
sweep (`~/lrubb.sh`, `~/statphase.py`, rank 0, bucketed by requests running):

| requests running | threshold 0.5 (135): distinct / inserted / read through / miss / steps over / ms per forward | threshold 0.99 (267): the same |
|---|---|---|
| 3-4 | 93 / 8.8 / 2.1 / 11.6% / 2.5% / 84 | 95 / 10.1 / 1.0 / 11.7% / 0.8% / 84 |
| 5-8 | **148** / 2.5 / **31.2** / 22.7% / **72%** / **207** | 152 / 20.8 / 3.3 / 15.8% / 3.0% / 144 |
| conc-4 / conc-8 aggregate | 112.8 / **75.1** | 114.4 / **109.2** |

Eight DIFFERENT requests route to ~150 distinct experts a layer. That is over 0.5 x 270 on 72% of steps, where the
manager inserts nothing and refreshes nothing, so the cache stops following the traffic and every step reads 31
experts a layer through from host: 2 GB and 207 ms per forward pass. Point 1 above ("it is not the insert
threshold") was true of what it measured -- four requests running, 85 distinct experts, 2-3% of steps over -- and
wrong as a general statement: the capacity limit had been hiding the threshold, and `bench/mix.py`'s mixed batch has
four distinct prompts where BetterBench has 29. Point 3 of the knob legs ("the mixed batch does not move") is
superseded for the same reason. `R9K_LRU_THRESH` now defaults to **0.99** (the cache tests pin 0.5, the value their
step sizes were written for).

**bb-qfn-tp2-rc2 = the v0.2.1 TP2-offload default** (NSEQ=8, copy grid 64,16, threshold 0.99; 270 slots, NBT 4096,
UTIL 0.94; GPUs 0,2; second launch, KV 120,724 tokens; full BetterBench, 20 passes), against the morning's run of
the v0.2.0 defaults:

| | v0.2.0 defaults (`bb-qfn-tp2-offload`) | v0.2.1 defaults (`bb-qfn-tp2-rc2`) | |
|---|---|---|---|
| single-stream decode | 93.8 tok/s | 97.4 | +4% |
| step p50 | 25.68 ms | 25.62 | = |
| TTFT p50 | 555 ms | 484 | -13% |
| prefill 2k / 8k / 16k / 32k | 2,099 / 3,163 / 3,434 / 3,329 | 2,208 / 3,558 / 3,839 / 3,793 | +5 / +12 / +12 / +14% |
| aggregate at 1 / 2 / 4 / 8 / 16 | 84 / 104 / 108 / 111 / 95 | 87 / 109 / 116 / 118 / 113 | +4 / +5 / +7 / +6 / +19% |
| per-request decode at 8 (median) | 20.6 tok/s | 17.1 | eight run instead of ~six |
| TTFT p50 at 8 / 16 concurrent | 7.1 s / 37.9 s | **0.95 s** / 27.3 s | |
| 8 requests, one prompt (code / json) | 283 / 347 | 434 / 577 | +53 / +66% |
| 8 requests, four prompt types | 118 | 166 | +40% |
| sanity | 8 / 8 | 8 / 8 | |

The prefill gain arrived with the threshold (the NSEQ=8 run without it had the morning's prefill to the percent);
which steps of a prefill it changes has not been isolated. Gates (`test_cache_moe`, `test_cache_stats`) pass on
the new defaults.

**What the cards are doing meanwhile** (amdgpu `gpu_busy_percent`, `mem_busy_percent`, hwmon power and clock at 1 Hz,
joined with the server's request counters; both cards of the pair, 225 W cap):

| phase | busy | memory busy | W per card | clock | W per card before the threshold fix |
|---|---|---|---|---|---|
| 1 request | 100% | 32% | 206 | 3,379 MHz | 203 |
| 2 requests | 100% | 26% | 199 | 3,385 | 197 |
| 3-4 requests | 100% | 21% | 179 | 3,399 | 175 |
| 5-8 requests | 100% | 15-16% | 168-171 | 3,402 | 150-152 |
| prefill, long prompts | 100% | 27% | 219 | 2,984 | 218 |
| TP4, 1 request (for scale) | 99% | 28% | 192 | 3,397 | |

A kernel is always running, and the more different requests run the LESS the cards compute: 206 W with one request,
170 W with eight. The rest of the time goes to experts arriving over the link. Only prefill sits at the cap with the
clock pulled down. Two cards with offloaded experts are link-bound with compute to spare; beyond these defaults
only three things move a mixed load: more VRAM for slots, a faster link, or routing that prefers resident experts
-- the last changes outputs and would need the full paired eval before it could be offered even as an option.

Placement rule, now measured both ways: **same switch for tensor parallel, one card per switch for offload.**
The launcher header says so; `GPUS=0,2` on this box.

### 2026-09-29: the decode fusions one at a time, on the full test set (the round-3 question, answered)

Item 1 of the list below, run the same day (`~/chain-qf.sh` / `chain-qf2.sh` on VM100, `~/qfcmp.py`): Flash-Next
TP4, **all 1,319 GSM8K test questions**, chain-of-thought, concurrency 1. Reference = round-2 numerics on the
current build (all five decode fusions at stock); one fusion on per leg; then the shipped default; then the
reference again.

| leg | accuracy | reference-only right | leg-only right | McNemar p (exact) | identical outputs |
|---|---|---|---|---|---|
| reference (`qf-base`) | 96.82% (1,277) | | | | |
| reference again (`qf-base2`) | 96.82% | 0 | 0 | 1.00 | 99.9% |
| router GEMM | 96.82% | 6 | 6 | 1.00 | 31.8% |
| indexer norm + rope glue | 96.97% | 0 | 2 | 0.50 | 87.2% |
| hyper-connection mix | 97.04% | 5 | 8 | 0.58 | 35.3% |
| GDN speculative-decode core | 96.97% | 5 | 7 | 0.77 | 31.9% |
| shared expert | 96.74% | 6 | 5 | 1.00 | 31.2% |
| **all five (the v0.2.0 default)** | **97.04%** | 5 | 8 | 0.58 | 29.9% |

1. **No fusion costs accuracy, alone or together.** The discordant pairs are balanced on every leg, and the shipped
   default is three questions ABOVE the round-2 numerics, which means nothing either.
2. **The reference is bit-reproducible** (1,318 of 1,319 outputs identical six hours apart, no discordant
   pair), so the noise floor of this comparison is zero and every discordant pair above is the fusion's doing: a
   fusion changes which marginal questions come out right, not how many.
3. **The "~0.5 point" of yesterday was the subset.** On the first 800 questions -- yesterday's eval -- the same
   eight runs read as they did then: reference 97.88% (1 / 0 against `qt-r2`), router 6 / 1, hc mix 6 / 2, GDN 5 / 2,
   shared expert 5 / 0, all five 97.50% and 5 / 1, exactly `qt-r4c`. The round-2 numerics score 98.0% on those 800
   and 95.0% on the other 519: they sit on the lucky side of the marginal questions of that slice, so anything that
   reshuffles marginal questions looks like a loss there and like a gain on the rest. Paired tests do not protect
   against that; a reference that scores unusually well on the slice is itself the outlier, which the bisect had
   already half seen ("qt-r2 is the lucky outlier") before settling on the wrong reading.
4. **Consequences.** The three `=stock` knobs buy nothing measurable and cost ~10% of decode; the quality
   statement of v0.2.0 (changelog, release notes, README) is corrected in v0.2.1; the eval for a change of default
   is from now on the full 1,319 questions (40 minutes at TP4), not the first 800.

### 2026-10-01: after v0.2.1 -- what the two-card prefill number means, and the UTIL soak

**Why prefill rose with the insert threshold: the benchmark's prompt, not prompts in general.** v0.2.1's BetterBench
prefill is 5-14% above v0.2.0's and the release said it had not been isolated. `bench/prefill_kinds.py` (new) sends
two kinds of prompt to the same server -- BetterBench's own filler, which is one paragraph's ~75 words shuffled, and
slices of real text (this repository's notes) -- with the cache counters on (TP2 + offload, GPUs 0,2, rank 0):

| prompt | threshold 0.5: tok/s | distinct / read through per layer | steps over | threshold 0.99: tok/s | distinct / read through | steps over |
|---|---|---|---|---|---|---|
| BetterBench filler, 8k | 3,342 | 312 / 125 | 100% | 3,672 (+10%) | 241 / 70 | 59% |
| BetterBench filler, 24k | 3,218 | 268 / 112 | 86% | 3,645 (+13%) | 269 / 76 | 63% |
| real text, 6.6k-8.4k | 2,467 | 247 / 99 | 55% | 2,288 (2,123-2,683) | 307 / 129 | 70% |
| real text, 20k-26k | 2,568 | 339 / 145 | 76% | 2,526 | 367 / 160 | 83% |

A 4,096-token chunk of the filler routes to 240-310 distinct experts a layer, which straddles 0.99 x 270 = 267: about
40% of its steps now insert, the cache learns the filler's experts, and a third fewer are read through. A chunk of
real text routes to 310-370 and stays over either threshold; its prefill does not move (the 8k pair differs by its
own scatter). So: **on two cards with offloaded experts real documents prefill at about 2,300-2,600 tok/s, some 30%
below the BetterBench figure, on v0.2.0 and v0.2.1 alike**, and the release's "+5..+14% prefill" is a statement about
narrow prompts. The four-card numbers do not have this dependence (nothing is fetched). README corrected.

**The soak found a memory leak in the released two-card default, not a UTIL question** (`bench/soak.py`, new: 8
clients for 15 minutes, each sending an 18k-30k-token slice of real text and asking for 256 tokens; VRAM from
`mem_info_vram_used` at 1 Hz). Flash-Next TP2 + offload, v0.2.1 defaults (UTIL 0.94, NSEQ 8, KV 120,724 tokens):

| | v0.2.1 as released | with the fix |
|---|---|---|
| requests served / failed | 77 / **369** | 111 / 0 |
| VRAM used per card | 30.5 -> 32.6 GiB in 13 minutes (the whole card), then `torch.OutOfMemoryError`, engine dead | 30.98 GiB flat for 15 minutes, one step to 31.64 as the queue drained |
| sanity afterwards | -- (server gone) | 8 / 8 |

Cause: `LayerCache._align_bufs` kept the align outputs of the fused LRU kernel per (rows, block) per layer, for good,
so that a captured graph would find them at the same address. But a prefill chunk shares the step's token budget
with whatever is decoding, so under real traffic nearly every eager step has a size nobody has seen, and each new
size left 28 MiB per rank behind (two 73k-entry int32 tables x 48 layers at a 4,096-token chunk). Every benchmark
here uses fixed prompt depths and a handful of shapes; none could show it. It has been there since the fused LRU
path went in, so v0.1.0 and v0.2.0 have it too. Fix: ONE buffer set per layer, sized by the largest step seen (in
serving the first call is vLLM's profile run at max_num_batched_tokens) and handed out as views; the fused kernel
rewrites every entry it hands back, so sharing is safe and the addresses a captured graph replays through never
move. If a larger step ever arrives the old set is kept and one twice the size takes over. (A first version kept
buffers only during graph capture and used temporaries otherwise; it soaked clean too, and was dropped for the
simpler scheme.) `tests/test_cache_shapes.py`: 900 eager shapes smallest-first hold five sets and under 100 KiB at
test scale; largest-first nothing grows and the address is one throughout; a captured graph replays 12 new routings
equal to an eager twin, also with eager steps of another size in between. The plugin's other shape-keyed caches
(`_MX_TABLES`, the QSA scratch, the staging area) are bounded by construction.

A longer and rougher soak of the fix (16 clients for 31 minutes, prompts of 270 to 28.5k tokens, so the queue
fills and drains continually): 291 served, 0 failed, sanity 8 / 8, VRAM at the same 31.64 GiB plateau from the
third minute to the end.

**The same soak on four cards** (TP4, everything in VRAM, the expert cache not in use): 236 served, 0 failed, 5
rejected for length, sanity 8 / 8, VRAM flat at 31.96 GiB per card for the whole run. First long-context soak of
the headline configuration; nothing found.

**UTIL=0.96: rejected.** With the leak fixed it ran 15 minutes at 32,555 MiB of 32,624 and died on a 150 MiB
allocation when the queue drained (82 served, 7 failed). The step the 0.94 run takes at the same moment is 640 MiB,
which 0.94 has (983 MiB free at the peak) and 0.96 does not. The 28% of extra KV is not available on this card.

**Rule from here on:** a release is soaked with mixed-length prompts on both configurations before it is tagged.
Fixed-depth benchmarks measure speed; they cannot see what accumulates.

### 2026-10-02: garbled answers on two cards -- a race in our own all-reduce

**Found by accident, by the sanity check.** After a prefill benchmark on the leak-fix build the 8-way sanity came
back 7 / 8 with `'45 Engine En'` and `'5 Engine Eng'`: the right digits, then one token (7943, " Engine") instead of
the end of the answer, for good. Every sanity before it had been a clean 8 / 8 -- and that check counts an answer
right if the number appears anywhere in it. `bench/sanity_stress.py` (new) asks the same questions strictly, in
rounds, optionally after long prefills, and prints every bad answer.

**It was not the leak fix** (all three buffer schemes showed it: 2 / 800, 2 / 800, 1 / 800 answers, always in the
first 8-way batch after a long prompt, always a late sequence) **and not the insert threshold** (3 of 60 rounds at
0.5, always the eighth request). The eighth request was the clue: with `NSEQ=8` a long request that has not left
yet makes the eighth of eight wait one step, and it then joins a batch of running decodes. Submitting more requests
than `max_num_seqs` does that on every round, with no long prompt at all:

| Flash-Next TP2 + offload, NSEQ=8, strict sanity | rounds with a bad answer | bad answers |
|---|---|---|
| 8 concurrent | 0 / 100 | 0 / 800 |
| 9 concurrent | 34 / 100 | 74 / 900 |
| 12 concurrent | 41 / 100 | 57 / 1,200 |
| 16 concurrent | 34 / 100 | 64 / 1,600 |

One knob per launch from there (9 concurrent x 100 rounds / 12 x 60; rounds with a bad answer):

| leg | 9 conc | 12 conc | |
|---|---|---|---|
| defaults | 34 | 41 (of 100) | |
| TP4, NSEQ=16 at 16 / 17 / 24 / 32 concurrent | 0 | 0 | 0 of 7,620 answers |
| TP4, NSEQ=8 | 0 | 0 | not the sequence limit as such |
| `EAGER=1` (no HIP graphs) | 0 | 0 | needs graph replay |
| GDN decode stock / all five decode fusions stock | 1 / 0 | 34 / 35 | not our decode kernels |
| `R9K_AR_QUANT=0` (exact all-reduce only) | 25 | 29 | not the compressed codec |
| `R9K_LRU_FUSED=0` | 8 | 6 | not the fused LRU kernel |
| a sparser capture-size list | **99** | **60 of 60** | deterministic: it is about how a mixed batch is padded and sized |
| GPUs 0,1 (one PLX switch) | 26 | 27 | not the card pair |
| no speculation (`MTP=` as an argument) | 0 | 0 | the draft passes supply the differently-sized messages |
| **RCCL all-reduce (`R9K_R4D_AR=0`)** | **0** | **0** | |
| **libr4d all-reduce (`R9K_AR_IMPL=r4d`)** | **0** | **0** | |
| Qwen3.8-27B, TP2, `serve/27b.sh` | 8 | 0 | the 27B has it too |

**The bug** (`kernels/r9k_ar.hip`, ours, the TP2 default since 2026-09-22). Each rank pushes its input into the
peer's scratch and sums the pair; the scratch is double-buffered so that a rank already on the next call cannot
overwrite what its peer is still reading. The half was chosen by `seq[b] & 1`, a counter PER BLOCK, and the launch
grid was the number of data blocks, which follows the message size (`n16 // 1400`, 4..24). A block that sat out a
small message kept its old parity, so on the next, larger message it wrote into the SAME half as the message
before -- and its slice starts inside that message's tail whenever the new slices are shorter (`per' * nb_prev <
n16_prev`: e.g. 28 rows = 6 blocks of 1,494 words, then 58 rows = 13 blocks of 1,428: block 6 starts at word 8,568
of a message that ends at 8,960). A rank one call ahead then overwrote the last rows of the previous message before
the other rank had summed them. From that step the two ranks disagreed on the hidden state of the LAST sequence in
the batch, and it decoded garbage for the rest of its life. The compressed path made it worse in a second way: it
shared the scratch with the exact path but kept its own counters, so the two paths could pick the same half for
consecutive calls at any time. It takes two back-to-back all-reduces of different sizes with the ranks slightly out
of step: the MTP draft passes next to a main forward whose size has just changed because a prompt joined, under graph
replay, where launches are microseconds apart. In lockstep it never shows, which is why every test passed (the
kernel's own test even exercised "a varying block count between calls" -- with a sync after each).

**The fix.** Every call advances every counter: the launch grid is fixed per (scratch, counters) pair (24 blocks at
TP2; 16 one-shot and 64 two-shot at TP4) and blocks beyond the data blocks do nothing but increment. All blocks
then agree on the half, which is what double buffering assumed all along; the 2-rank compressed path now uses the
exact path's counters. New entry points `r9k_ar_*_g(..., grid)`; the old ones remain for tests that keep one block
count. The N-rank kernels had the same per-block scheme (TP4 never showed it; its one-shot path has a constant
block count and its two-shot path varies only above ~40 tokens) and got the same fix.

`tests/test_ar_race.py` (torchrun, 2 ranks): one captured graph of small-then-larger pairs, replayed 1,500 times
with rank 1 held back. Old grid: 3 exact outputs wrong with every message exact, 6,008 wrong and 6,000 differing
between the ranks with the compressed path on. Fixed grid: 0 and 0. `test_ar_r9k`, `test_ar_nrank`, `test_ar4`
pass; the idle blocks cost nothing measurable (4-rank graph timings identical to the microsecond-level table).

**Serving, after the fix** (strict sanity, requests over `max_num_seqs`): TP2 + offload with the sparse capture list
0 of 1,620 (was 99 / 100 rounds); TP2 defaults 0 of 4,600 at 9 / 12 / 16 concurrent and 0 of 320 in the long-prompt
pattern; 27B 0 of 3,900; TP4 0 of 6,020. Probes unchanged on all three.

**What was affected.** Any two-card run on our all-reduce -- Flash-Next TP2 + offload and the 27B at TP2 -- from
2026-09-22 (v0.1.0) on, whenever a request joined a batch already in flight: more requests than `max_num_seqs`, or
requests arriving while others decode. One sequence of the batch, a few percent of the time. v0.2.1's `NSEQ=8`
made it common by letting eight run. Every quality eval here ran at concurrency 1 and could not see it; BetterBench
does not read answers. Four cards: no wrong answer in 15,000, but the same construction, now fixed.
On v0.2.0 / v0.2.1, `R9K_R4D_AR=0` (RCCL all-reduce) avoids it.

**Rules from this.** (1) The sanity check is strict now and part of every benchmark chain. (2) A release is
stressed at more requests than `max_num_seqs` and soaked with mixed lengths, on every configuration. (3) "Passes in
lockstep" is not evidence about a handshake; a protocol test must put the ranks out of step.

### 2026-10-02: out of memory on four cards under mixed-length concurrency -- the PLE short-conv prefill

**Found by the release soak.** The first validation pass of v0.2.2 ran BetterBench, the strict stress and a
mixed-length soak on every configuration. Four cards passed the first two (159.2 tok/s, 0 bad answers of 4,900) and
died in the third: 24 clients, prompts of 200 to 30k tokens, `NSEQ=16` -- 31 requests served, VRAM from 29.4 to
32.2 GiB in 40 seconds, `torch.OutOfMemoryError` on a 156 MiB allocation. The earlier four-card soak (8 clients,
all long prompts) had sat at 31.96 GiB and passed.

The allocation is in `ple/short_conv.py`, our drop-in for vLLM's `_short_conv_dilated_prefill_batched`, and the
cause is in the algorithm we mirror: it packs ALL prefills of a step into `[prefills, longest, 4*hidden]` and keeps
about six tensors of that size alive (zeros, transposed, history, conv output, activation, transposed back). One
long chunk next to many short prompts is therefore many times the step's real tokens. vLLM's memory profile runs
one sequence and sees 84 MiB per tensor; nothing budgets for the product.

| one step holds | stock: peak in this function | packed by length |
|---|---|---|
| 3,800 + 7 x 40 tokens | 3,055 MiB | 526 MiB |
| 3,000 + 15 x 70 tokens | 4,779 MiB | 432 MiB |
| 2,900 + 800 + 300 + 17 + 5 tokens (+ decode rows) | 1,498 MiB | 419 MiB |
| 3,000 + 900 + 60 tokens, one NULL state block | 958 MiB | 430 MiB |

**Fix:** group the step's prefills by length, longest first, each group at most 1.25x the step's own token count
(`R9K_PLE_CONV_SLACK`; a single sequence is always a group, so one prefill or prefills of similar length run exactly
as before, without the host sync the grouping needs). Each sequence's convolution reads only its own tokens and its
own state, so the output and the written-back conv state are the same bits: `tests/test_ple_conv.py` compares both
with stock on five skewed cases and bounds the peak.

It also explains two loose ends on two cards: the one-time +640 MiB step in the soaks of the leak fix (the first
step with several prefills of different lengths), and why `UTIL=0.96` died "when the queue drained" (the same step,
with 69 MiB free). Every release has had it -- through vLLM's own method in v0.1.0, through our override since
v0.2.0 -- on any configuration; four cards at 16 running sequences reach it first because they admit the most
prefills per step.

### 2026-10-02: v0.2.2 validated -- the checks every release gets from now on

On the release code (151 source files checksummed against the commit), every configuration: the 26 single-GPU gates
and the all-reduce suites (`test_ar_race` on both card pairs, `test_ar_r9k`, `test_ar_nrank`, `test_ar4`); then per
configuration a full BetterBench (20 passes), the 8-way sanity, the strict stress at more requests than
`max_num_seqs`, a mixed-length soak with VRAM at 1 Hz, and the strict stress after long prompts.

| | Flash-Next TP2 + offload | Flash-Next TP4 | 27B TP2 |
|---|---|---|---|
| decode / step p50 / TTFT p50 | 97.4 tok/s / 25.61 ms / 483 ms | 157.3 / 16.81 / 97 | 197.5 / 23.46 / 114 |
| prefill 2k / 8k / 16k / 32k | 2,210 / 3,558 / 3,839 / 3,800 | 6,476 / 7,348 / 7,445 / 7,180 | 4,190 / 4,191 / 4,072 / 3,837 |
| concurrency 1 / 2 / 4 / 8 / 16 | 87 / 109 / 116 / 116 / 116 | 151 / 239 / 355 / 473 / 624 | 174 / 280 / 413 / 519 |
| the release before | 97.4 / 25.62 / 484; 87-109-116-118-113 | 159.2 / 16.79 / 94; 151-231-350-478-635 | 196.6 / 23.43 / 116; 174-287-401-526 |
| strict stress over `max_num_seqs` | 0 bad of 2,500 (9, 16 conc) | 0 of 4,900 (17, 32) | 0 of 2,100 (9, 12) |
| mixed-length soak | 16 clients, 26 min: 230 served, 0 failed | 24 clients, 15 min: 450 / 0; 16 long-context clients, 10 min: 165 / 0 | 12 clients, 15 min: 232 / 0 |
| VRAM peak in the soak (of 32,624 MiB) | 30,948 | 30,210 | 32,505, once (see below) |
| strict stress after long prompts | 0 of 160 | 0 of 160 | 0 of 160 |
| sanity | 8 / 8 | 8 / 8 | 8 / 8 |

Speed is the release before's on all three, to within the run-to-run band. The short-conv packing took 0.7 GiB off
the two-card soak peak (31,640 -> 30,948: the one-time step is gone) and 1.75 GiB off four cards' (31,957 with only
eight clients -> 30,210 with twenty-four). Open after this: the 27B reached 32,505 MiB once during its soak before
torch trimmed its cache back to 28.3 GiB -- no error, but `KVMEM=9` leaves it little room; and
`tests/test_ar_quant.py` is a libr4d-backend test that spawns its own ranks and must not be run under torchrun.

### 2026-10-02: Qwen3.8-27B on two cards, a same-day baseline against the reference stack

Same two cards (GPUs 0,1, one PLX switch), same checkpoint, full BetterBench each, both sanity 8 / 8. Ours:
`serve/27b.sh` on vLLM 0.30. Reference: GGZ14 radiance 0.9.3 (vLLM 0.27.1), relaunched with `~/launch-ref27b.sh`.

| | ours | reference | |
|---|---|---|---|
| decode | 196.6 tok/s | 197.5 | 99.5% |
| step p50 | 23.43 ms | 23.15 | +1.2% |
| TTFT p50 | 116 ms | 65 | 1.8x slower |
| prefill 2k / 8k / 16k / 32k | 4,177 / 4,187 / 4,066 / 3,832 | 4,780 / 4,947 / 4,903 / 4,746 | 87 / 85 / 83 / 81% |
| concurrency 1 / 2 / 4 / 8 | 174 / 287 / 401 / 526 | 180 / 303 / 428 / 558 | 97 / 95 / 94 / 94% |
| KV cache | 210,782 tokens | 798,784 | 3.8x smaller |

Decode is at parity. What is left: prefill (the reference chunks at 8,192 and runs libr4d's all-reduce), time to
first token (prefill graphs up to 2,048 tokens took Flash-Next from 137 to 104 ms and have not been tried here),
concurrency, and KV capacity (ours is a fixed `KVMEM=9`; theirs is profiled at 0.95 and, by its size, 8-bit).

### 2026-10-02: time to first token -- the GDN prefill core (after v0.2.2, on master)

The 27B answered a short prompt in ~97 ms against the reference's 46, and on 2026-09-21 this log closed that as
upstream behaviour. It was not. Reopened with three measurements, each narrower than the last:

1. **TTFT against prompt length** (`~/pfsweep2.py`): ours 96 / 96 / 144 / 144 / 175 ms at 44 / 82 / 158 / 319 / 655
   tokens, the reference 46 / 67 / 94 / 130 / 163. Flat in the prompt length: a fixed cost, not compute. Prefill
   graphs up to 2,048 tokens (`CGSIZES`) moved only the 158-319 range (to 96 / 121) and cost 0.8 GiB; prefix caching
   off changed nothing. Neither is the cause.
2. **The server's own counters** (`bench/ttft_breakdown.py`): queue 0.0 ms, one engine iteration per request, and
   that one step ~90 ms where a decode step is 23-28. The reference: one step, 40 ms.
3. **A torch profile of that step with Python stacks**: 48 calls of `r9700::gdn_core`, each falling through to
   stock's prefill core -- `fused_post_conv_prep`, the chunked delta rule (`chunk_gated_delta_rule`: cumsum, KK^T,
   triangular solve, W/U, the state scan, the output: six Triton kernels), state gather and scatter, the gated norm.
   ~30 launches and 1.15 ms of wall per layer in a unit bench, the same at 16 tokens and at 2,048. 48 layers: ~55
   ms of a 97 ms step. The op runs eagerly between the pieces of the piecewise graph, so no capture size helps.

**The fix** (`kernels/r9k_gdn.hip`, `r9700_vllm/models/gdn.py`): for a step with few prefill tokens, the recurrence
token by token in one launch per layer -- `r9k_gdn_seq`, the decode kernel's loop with the state in registers,
any number of tokens, the final state stored once -- after the causal conv in one launch, `r9k_gdn_conv`. Both read
and write the batch's own rows through a row map, so a step that mixes prefills with running spec-decode sequences
needs no gather either (their half goes through the same kernel in its decode form; only stock's conv update still
gathers its input).

What the unit test holds (`tests/test_gdn_prefill_r9k.py`):
- against an fp64 token-by-token reference the output differs only in bf16 rounding flips (19-318 of 10^5-10^6
  elements), relative error 1-6e-5; **stock's chunked form sits at 5e-3 from the same reference** (state 3e-3). The
  chunked form does its solves on bf16 q/k; token by token in fp32 is simply closer to the definition.
- spec-decode sequences through the new kernel are bit-identical to `r9k_gdn_decode_mtp`; row-mapped runs are
  bit-identical to compact ones.
- the conv is bit-identical to stock's, output and state. That took finding out what stock computes: the Triton
  kernel multiplies bf16 by bf16, and on this backend the fp32 -> bf16 conversion inside that multiply
  **truncates** (the final store rounds to nearest). Half the outputs differed by an ulp until the products were
  truncated; found by enumerating the rounding choices against stock's output. Kept, because the decode-side conv
  is the same Triton arithmetic and the two should agree.

**It is not free on the GPU.** Unit bench, one sequence, Hv 24: stock 1.12 ms per layer at any length; ours 0.04 /
0.09 / 0.24 / 0.48 / 0.95 / 1.77 ms at 16 / 45 / 128 / 256 / 512 / 1,024 tokens (~1.8 us per token; the conv 0.03-0.15
against 0.42). In serving the crossover comes earlier than those numbers say, because stock's Python overlaps with
GPU work already queued: the step is max(launch time, GPU time), and once it is GPU-bound our extra 0.065-0.086 ms
per token (36 / 48 layers) is pure loss.

| first token, ms (client median) | 45 | 83 | 159 | 197 | 236 | 278 | 320 | 404 | 509 | 656 tokens |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 27B TP2, stock core | 99 | 98 | 148 | | | 148 | 148 | 148 | 150 | 172 |
| 27B TP2, ours at every length <= 512 | 49 | 61 | 81 | 85 | 92 | 122 | 128 | 142 | **161** | 172 |
| reference stack | 46 | 68 | 93 | | | | 125 | | | 158 |
| Flash-Next TP4, stock core | 87 | 87 | 88 | 87 | 88 | 91 | 92 | 107 | 112 | 131 |
| Flash-Next TP4, ours <= 256 | 44 | 58 | 73 | 82 | 87 | 90 | 91 | 107 | 112 | 132 |
| Flash-Next TP4, ours <= 512 (first try) | 46 | 60 | 74 | | | | **107** | | | 132 |

So the threshold (`R9K_GDN_PREFILL_MAX`, prefill tokens per step) is 256 by default -- Flash-Next on four cards
breaks even there -- and `serve/27b.sh` sets 400 (the 27B breaks even near 440). Flash-Next on two cards (offload,
no prefill graphs): 52 ms at 44 tokens; above its graph limit it runs eagerly as before (177 ms at 82 tokens).

Checked, in place of the release checklist (Brian's call: publish without the full suite): seven unit gates;
strict sanity with more requests than `max_num_seqs`, 100 rounds after short prefills and 10 after long ones, on
all three configurations, 0 bad answers of 3,850 (run at threshold 512, the wider coverage); three concurrency
probes per launch on Flash-Next TP4 with and without (c16 884-887 against 866-880: no change -- one early probe
that read 774 was a first-probe-after-launch number). **Not run:** GSM8K, a full BetterBench, the mixed-length soak.
The smoke test earned its keep once: with speculative decoding the conv state is wider than the conv's three
columns (the conv update's rolling window lives in the same tensor), the wrapper asserted exactly three, and the
server died at warm-up. The unit test now has that case.

What is left in a short first step (profile with stacks, 27B, 45 tokens, so every number is inflated ~1.7x):
GDN metadata built once per KV-cache group, 11 times, 14 ms (stock; the reference patches vLLM to share it); the
GDN layers still 0.6 ms each, most of it stock's conv launcher at the time (since replaced); the KV-cache update of
the 16 attention layers 6 ms; the drafter 4 ms.

### 2026-10-02: UTIL=0.96 on two cards, retested and adopted

Brian's question: could 0.96 work now that the other bugs are fixed? It was rejected on 2026-10-01 after the leak
fix but before the short-conv packing fix, and the allocation it died on (150 MiB with 69 MiB free, "when the
queue drained") was that third bug. Never retested since. Flash-Next TP2, GPUs 0,2, offload, master 7806509:

| | UTIL=0.94 | UTIL=0.96 |
|---|--:|--:|
| KV cache, first (compiling) launch | 95,429 tokens | 128,772 |
| KV cache, restarted launch | 121,299 | 155,791 (+28%) |
| VRAM, first launch, both soaks | | 30,376 -> 30,396 MiB, flat |
| VRAM, restarted launch, mixed soak | 30,400 -> 30,948 (25 min) | 31,066 -> 31,350 (first 5.7 min) |

First launch at 0.96: the soak that killed it (8 clients, 18k-30k tokens, 956 s) 86 served, 11 rejected for
length, 0 errors; mixed (16 clients, 200-30k tokens, 1,563 s) 227 served, 11 rejected, 0 errors; strict sanity
after 24k-token prefills 0 bad of 180; sanity 8 / 8; probe 117.0 tok/s, c8 188, prefill 8k 2,622 (speed unchanged).
The first launch has the smaller pool, so the case people run -- restarted -- was started as a short check and
**stopped at 5.7 minutes of the mixed soak on Brian's instruction (no more tests)**: peak 31,350 of 32,624 MiB
(1,274 free), no OOM or HIP error line, no request summary. At 0.94 all of the soak's growth came in its first
minute, which is why a short burst was the check; a full-length soak of the restarted launch is still owed.
Adopted as the default for TP2 with offload only (`serve/serve.sh`; `UTIL=0.94` restores).

### 2026-10-03: one state page per request for GDN under speculative decoding -- design

**What vLLM does today.** A hybrid model's linear-attention (GDN) state lives in the KV cache as "mamba pages". With
a drafter, vLLM gives every request `1 + num_speculative_tokens` state pages per GDN group (2 + spec with prefix
caching, "align" mode): the verify step writes the state after *each* candidate token into its own page, and the
next step reads the page of the last accepted token (`spec_state_indices[n, num_accepted - 1]`). Measured
(`~/kvcost.py`, 2026-10-03), the fixed cost per *running* request before it holds any context:

| configuration | per request | at full `max_num_seqs` |
|---|--:|--:|
| 27B TP2, SPEC=7 (shipped) | 15,661 token-equivalents = 7.4% of the pool | 8 running: 59% |
| 27B TP2, SPEC=5 / SPEC=3 | 5.6% / 3.8% | 45% / 31% |
| Flash-Next TP2 offload, MTP=3 | 10,394 = 8.1% | 8 running: 65% |
| Flash-Next TP4, MTP=3 | 6,152 = 2.4% | 16 running: 38% |

On the 27B a second, independent waste: the DFlash2 drafter's 5 attention layers make vLLM's group size 5, so the
16 target attention layers are padded to 20 (25% of attention bytes) and the 48 GDN layers to 50 -- upstream's
grouping heuristic (`_get_kv_cache_groups_uniform_page_size`), not addressed here. Tried and rejected: the
block-outermost layout (`VLLM_KV_CACHE_LAYOUT=BLHNC`, which our kernels could take) routes to the packed grouping
without padding, but that path sizes every block to the widest page (the GDN state) and asked 21.7 GiB for a 32k
request.

**The design: verify from one checkpoint, replay the accepted tokens at the start of the next step.** vLLM has
a cousin of this for Kimi-K3's KDA layers ("RecoverSSM": `num_speculative_blocks = 0`, a post-sampling commit hook
in the runner). Its kernels are not in this build, its flag is validated against Kimi architectures, and the flag
also stops piecewise graph capture above the decode width (`cudagraph_utils`: "recoverSSM cannot capture a dummy
query wider than its workspace"), which would cost Flash-Next its prefill graphs. So no runner hook: the layer does
the replay itself when the request next comes through.

1. **Page layout** (`get_state_shape` / `get_state_dtype` on our GDN layer class, which stock's `bind_kv_cache`
   turns into views): stock conv window `[C, 3 + spec]`, stock SSM state `[Hv, 128, 128]`, plus a RECORD of the last
   step's candidate inputs in two slots -- the conv'd q/k/v rows `[2, 1 + spec, C]` bf16, the gating inputs b, a
   `[2, 1 + spec, 2 Hv]` bf16 -- and a flag `[2]` int32 = {live rows | slot << 8, done counter}. `get_kv_cache_spec`
   sets `num_speculative_blocks = 0`. Per GDN group that is 1 page per request instead of 1 + spec; the record adds
   ~10% to a page (two slots).
2. **Verify** (`r9k_gdn_spec_verify`, the decode kernel's loop): if the page's flag says a record is live, replay
   the first `num_accepted` of its rows (vLLM hands the previous step's accepted count to this step) from the base
   state and store the result; then run the new candidates from that state, outputs through the gated norm, no
   state stored; write the new record into the other slot (sibling blocks of the same sequence may still be
   replaying from the live one) and, once every value head's block of the sequence has finished (per-page atomic
   counter), flip the flag. Stock's `causal_conv1d_update` keeps the conv window exactly as today: it only ever
   used page 0 and its own `num_accepted - 1` offset.
3. **Every path that touches a spec row is ours**: pure spec decode (in the FULL graph), prefill + spec, long prefill
   + spec (stock's chunked prefill for the prefill rows, gathered; our verify for the spec rows), and a batch of
   draft-less decodes (our verify with one candidate, so a live record is replayed first; the builder subclass
   keeps every row's accepted count for it). Prefill rows clear the flag (our kernels do it in-kernel; a stock
   fallback zeroes it from Python), so a recycled page never replays a stale record.
4. **Scope of this stage: mamba cache mode "none" (prefix caching off).** In "align" mode the runner's block-boundary
   copies pick a spec slot by `num_accepted - 1`, which no longer exists; supporting it means writing the boundary
   checkpoint from the replay. With prefix caching on the layer logs a warning and keeps vLLM's pages. On the 27B
   `PREFIX_CACHE=0` also measured 2.1 GiB less VRAM and 6% more KV (2026-10-02); Flash-Next already runs with it off.
   Knob: `R9K_GDN_STATE=onepage|stock`.

**Numerics.** Bit-identical to the slot design: the slot of the accepted token held `round(h)` after an fp32
recurrence from the rounded base; the commit recomputes the same fp32 recurrence from the same rounded base on the
same recorded inputs and rounds once. Outputs are the same kernel as today.

**Cost.** The replay is at most 1 + spec tokens per layer per request inside the verify launch; the one-page verify
without per-token state stores is *faster* than the slot kernel at 16 sequences (165 vs 281 us on 27B-TP2 heads,
replaying 8 then running 8). Memory per request per GDN group: 1 + spec pages -> 1.

**Expected.** 27B TP2 at 8 running: 59% -> ~13% of the pool pinned; Flash-Next TP2 65% -> ~16%; TP4 38% -> ~10%;
the drafter fits at 200k on one card (the 9.35 GiB need drops by the 7 spec pages, ~1.1 GiB at TP1, more than the
1.15 GiB that was missing).

**Validation.** Unit (`tests/test_gdn_onepage_r9k.py`): multi-step simulation vs the slot design with random
acceptance -- outputs and the replayed page bit-identical, including the step after a prefill, 1-candidate steps,
both state dtypes, the sigmoid gate, a row map; the conv-window shift identity for a future align-mode commit.
Serving: `kvcost`, strict sanity under overload on all configurations, probes vs shipped, VRAM peaks, then the
release checks (Brian's call).

### 2026-10-03: one state page per request -- what it measured, and a host crash during the tests

**Implemented as designed** (stage 1: `mamba_cache_mode` "none", i.e. `PREFIX_CACHE=0`; `R9K_GDN_STATE=stock` restores
vLLM's pages). The page is sized at the model level: `R9kQwen3_5*` override `get_mamba_state_shape_from_config`, and
`R9kQwen4Exp*` must override `get_mamba_specs_from_config` as well, because stock Qwen4Exp defines it and
`Platform._align_hybrid_block_size` prefers it when present -- the first Flash-Next launch fell back to a page sized
2x by the layer (`GDN page 851968 -> 1703936 (the model class did not size it)`), which this fixed. Unit test
`tests/test_gdn_onepage_r9k.py` passes (11 cases, bit-identical to the slot design).

| | stock pages | one page | |
|---|--:|--:|---|
| 27B, one card (KVMEM=3, prefix caching off): KV cache | 35,576 | 62,295 | +75% |
| 27B, one card: pages pinned per running request | 85 | 15 | |
| 27B, one card: c8 / c16 tok/s | 192 / 196 | 329 / 283 | |
| 27B, two cards (prefix caching off): KV cache | 223,329 | 367,494 | +64.5% |
| 27B, two cards: 8 running short requests, pool pinned | 59.4% | 11.5% | |
| 27B, two cards: decode mean / c8 / prefill 8k (3 probes each) | 206 / 534 / 4,055 | 211 / 586 / 3,900 | |
| Flash-Next, two cards (0,1; both restarted launches): KV cache | 154,641 | 183,202 | +18.5% |
| Flash-Next, two cards: pinned per running request (tok-eq.) / 8 running | 10,386 / 53.7% | 6,736 / 35.6% | -35% |
| Flash-Next, two cards: decode / c8 / c16 / prefill 8k (3 probes) | 93 / 119-122 / 95-96 / 1,690 | 93 / 121-122 / 92-96 / 1,678 | |
| Flash-Next, four cards (both first launches): KV cache | 256,682 | 279,564 | +8.9% |
| Flash-Next, four cards: pinned per running request / 8 running | 6,152 / 19.2% | 3,739 / 10.7% | -39% |
| Flash-Next, four cards: decode / c8 / c16 / prefill 8k | 194 / 568-597 / 893 / 7,230 | 195 / 592-621 / 896 / 6,780 | prefill -6% |

Strict sanity under overload: 27B one card 0 bad of 360; 27B two cards 0 of 990; Flash-Next two cards 0 of 540
(twice); Flash-Next four cards 0 of 1,020 (both modes). (A first one-page launch of the two-card configuration
compiles and shows 151,179; the table has the restarted launch, the normal way to run it.)
Attention block: 27B 1,808 tokens (stock 1,648), Flash-Next two cards 912 (832), four cards 464 (416): the
attention page grows because the mamba page now carries the 2 x (spec+1)-row record. **Cost: long prefill is 4-6%
slower with one-page** (27B 8k 4,055 -> 3,900; Flash-Next TP4 7,230 -> 6,780), presumably the bigger attention
block in the paged-attention prefill; the GDN path is unchanged for long prefills. Decode and concurrency are equal
or better everywhere.

**The c16 question.** The one-page two-card runs showed 636 / 417 / 444 tok/s at 16 concurrent against stock's
588-606. It was not the page design: ten further c16 runs on the same server measured 645-697, all above stock; GPU
power, clocks and preemptions were unchanged in a slow run; the slow runs coincided with other containers loading
models on the box. The 27B runs `max_num_seqs` 8, so c16 is two waves, and the prose prompts (lowest acceptance)
are always the stragglers (~41 tok/s each vs ~57 for code). Per-request detail: `~/c16diag2.py` on VM100.

**Host crash 02:21-02:25 UTC, and what it turned out to be.** With all five cards loading at once (27B TP2 on HIP
0,1, Flash-Next TP2 on HIP 2,3, a 27B reviewer on HIP 4), the second chain B card (guest 08:00.0, host c5:00.0)
logged `MES(1) failed to respond to msg=INVALIDATE_TLBS` three times two seconds apart, then failed REMOVE_QUEUE and
SUSPEND, amdgpu began a GPU reset, eight seconds later the card read back as all ones ("device lost from bus"), and
two minutes after that the host hard-reset: firmware reason "an uncorrected error caused a data fabric sync flood",
a fatal machine check on one core (SMCA EX, code 0: a core watchdog on a stalled transaction). The BMC log has no
power event. The first suspects (the fifth card, the new host kernel 7.0.14-20 and QEMU 11.0.3-4 that the same
maintenance window installed, five-way load) were all cleared the same night:
- The MES timeout is **chronic**: ~130 occurrences since 2026-09-17 in every guest boot, on every card, on host
  kernel -19 and -20, with four cards and five, clustering at model load and graph capture (page-table churn), and
  self-recovering every time but this one. Two more single timeouts on 10-03 (04:06, 04:07) on two different
  cards during a Flash-Next TP4 launch, at 32-75 W and 32-38 C: not power, not heat.
- The exact 02:17 configuration, relaunched deliberately (three servers at once on five cards, then probes and
  stress on all three together): clean, 0 bad of 1,380.
- The guest already runs the TLB-fence rework (Linux 7.0, backported to 6.19.10) that fixed MES deadlocks on Strix
  Point, and the newest Debian backports firmware (MES 0x91). No fix specific to this message on gfx12 was found.
So the crash is: a chronic firmware hang escalated once into a GPU reset, and a GPU reset under passthrough on these
cards takes the link and then the host down. Containment proposed (Brian's call, a guest kernel parameter):
`amdgpu.gpu_recovery=0`, so a hung MES wedges one GPU and its container instead of resetting the card; a VM reboot
recovers it through the vfio function reset that every VM start already does. This matters for the eight-card plan
too: the hang rate scales with load/capture churn, not with cards per VM, and a reset in any VM takes the host.
Operational notes from the same night: the guest's power-cap service had failed at boot (an inline comment in
`/etc/r9700-powercap.conf` was read as the value; fixed); LACT 0.10.1 (headless) now runs in the guest with
`amdgpu.ppfeaturemask=0xffffffff`, applying -42 mV on every card (Brian's instruction) and the caps (225 W, the
passively cooled fifth card 210 W), then 210 W on every card (Brian, 05:25 UTC): at -42 mV the 27B two-card probe lost
0.7% decode, 1.4% at 8 concurrent and 2.2% on an 8k prefill going from 225 to 210 W, and strict sanity under
overload was 0 bad of 2,040 at 210 W. **Numbers from here on are at 210 W with -42 mV; everything published so far
was 225 W without the undervolt** (the undervolt alone gained ~5% decode at 225 W).

**Found and fixed the same night: the conv window's candidate width.** At temperature 0 the stock-page server gives
the same text for a request alone or in a batch of identical copies; one-page gave the same text within a batch but,
with >= 4 identical copies, two of four prompts differed from the lone run (coherent alternatives, a near-tie flipped),
and under `--enforce-eager` one-page was garbage even alone while stock was coherent. What cleared on the way: the
kernel (new lockstep unit cases, 4 and 8 identical sequences vs lone, one-page and slot bit-identical), page
distinctness and the code path (debug trace), the attention backend (vLLM's own: same divergence) and the larger
attention block (stock kernels on a one-page-sized page: invariant). A per-step checksum trace of layer 0 in eager
mode (`R9K_GDN_TRACE=1`) then showed it: step 1 after the prefill identical to stock in every input and the output;
at step 2 b, a and z still identical but the conv output different -- the conv window written by step 1 was wrong.
Cause: stock's `causal_conv1d_update` takes `max_query_len`, the number of candidate columns the window rolls by, and
the plugin passed `idx.size(1)`. With one page per request the mamba KV group's block table has one column (no
speculative blocks), so on every eagerly built step -- mixed prefill+spec steps, non-uniform batches, enforce-eager
-- `idx` is `[N, 1]` and the window rolled as if one candidate per step. FULL-cudagraph steps were right only by
accident: vLLM copies the `[N, 1]` slice into its `[N, S]` cudagraph buffer with `copy_`, which broadcasts the
column. The fix takes S from the record (`_spec_width`) in the fused path, the mixed-step path and the debug check.
The unit test could not see this (it feeds the conv output directly); the serving check that does is
`~/batchid.py`: identical copies in one batch must equal the lone run, in graph and eager mode.

After the fix (27B two cards, 210 W, -42 mV, prefix caching off): eager one-page bit-identical to stock per step;
strict sanity under overload 0 bad of 990; the six single-request texts identical to stock; sixteen fixed prompts
at 16 concurrent, six runs: every run completes all 4,096 tokens (before the fix one-page runs ended early),
one-page runs agree with each other on 7-16 of 16 texts (stock: 5-16), one-page vs stock 5-9 -- the same spread
stock shows against itself. The remaining run-to-run variation comes from vLLM's mixed prefill+decode steps, which
send a running request's rows through the prefill attention path: the stock-page server shows the same thing
(staggered arrivals: 3 distinct texts of 4), and in the traced mixed step one-page and stock produce identical
conv, input and output checksums for the same request state. Probes after the fix: decode 212, c8 554-579,
c16 673, 8k prefill 3,940 (210 W). Flash-Next two cards after the fix: sanity 8/8, 0 bad of 990, probes unchanged;
four cards: sanity 8/8, 0 bad of 1,870, decode 193-197, c16 898.

**What is left of batch dependence, and what it is.** Two sources, neither in the GDN path:
- On two cards, the compressed all-reduce (`R9K_AR_QUANT=1`, messages >= 128 KB) is lossy, and its rounding depends on
  the message, so a request's logits depend on who shares its batch: four identical copies prefilled together differ
  from the lone run from the fourth character on, in stock mode as much as one-page (the layer-1 prefill input
  already differs by 1.4% while layer 0's output matches). This is the documented trade of that knob, not new.
- On one card there is no all-reduce, and the step trace of four identical copies prefilled together against the
  lone run reads: prefill identical, verify steps 1-3 identical in every input and output, step 4 GDN state still
  identical but the step's rows differ by 1-2% -- the drafter proposed different tokens for the batch than for the
  lone request; acceptance then differs (4 vs 5) and the text flips at char 506. The target's GDN state never
  deviated before the drafts did. Whether stock pages behave the same at four requests on one card cannot be
  measured: stock pins 85 blocks per request there and admits at most two at a time (the thing one-page fixes).
  Four *different* prompts in one batch match their lone runs, and so do two or three identical copies; the effect
  needs three or more identical prompts at temperature 0 in one step, which production traffic does not produce.
Diagnosis knobs kept, off by default: `R9K_GDN_TRACE=1` (per-step checksums of layers 0 and 1 on eagerly built
steps), `R9K_GDN_PAGE_PAD=1` (stock pages with a one-page-sized page). Helper scripts on VM100: `~/batchid.py`
(identical copies vs lone), `~/stepcmp.py` (step trace comparison), `~/c16diag2.py`, `~/lockstep_prefill.py`.

**Remaining**: Flash-Next TP2 on the split pair and TP4 with one-page; whether `R9K_GDN_STATE=onepage` becomes the
default (it needs `PREFIX_CACHE=0`, a serving-behaviour change on the 27B, Brian's call); align-mode (prefix
caching) support via boundary checkpoints from the replay; the 200k single-card run with the drafter still needs
the 25% attention-page padding addressed upstream.

### 2026-10-03, evening: the validation debt paid, the MES firmware tested, and what determinism looks like

**Validation debt (v0.2.3 and v0.2.4 had been tagged on unit gates and strict sanity only).** Run on the production
copy at v0.2.4, 210 W, -42 mV, one server at a time: GSM8K full set (1,319, conc 1) and HumanEval (164) on all three
configurations; the paired thinking-mode GSM8K (first 800, conc 1) on the 27B with one-page and with stock pages;
full BetterBench on all three; 25-minute mixed-length soaks (16 clients, 200-30k tokens) with VRAM sampled; strict
sanity under overload at NSEQ+1 and 2xNSEQ. Everything green; the tables are in `notes/release-v0.2.4.md`. Headlines:
one-page and stock pages score identically with thinking (96.88%, 0 discordant); against the v0.2.0-era baseline
(97.50%) and the reference stack (97.62%) the paired test finds no difference (p = 0.36 / 0.21); BetterBench decode
is level with or above the 225 W references (27B 200.3 vs 197.5, four cards 160.9 vs 159.2, two-card offload 97.3 vs
97.4), concurrency up 6-12% on the 27B, prefill 0-11% lower (cap + attention block); soaks 1,283 requests, 0 errors.
A GSM8K caveat worth remembering: the September baselines are thinking-on, first 800 questions; a no-think full-set
run (94.69% on the 27B) is a different test and `eval.py compare` across the two is meaningless (100% of outputs
differ by construction).

**Determinism, settled.** 18 of 800 thinking answers differed between one-page and stock at conc 1. Replayed alone
with every step traced: no draft-less or short step anywhere (so not the `_nonspec_decodes` path); one-page gave
identical text and identical step counts in every run and on both card pairs; stock pages changed one answer's
length between two runs on the same server (514 vs 536 steps) and its step counts moved again when given the
one-page attention block size. So the slot design (`r9k_gdn_decode_mtp`, stock pages) has a rare run-to-run
nondeterminism and one-page does not; accuracy is unaffected. Low-priority follow-up now that one-page replaces it;
the tools are `~/replay18.py` / `~/replay18b.py` on VM100 and `R9K_GDN_TRACE=1` (now also on the non-spec path).

**MES firmware.** linux-firmware's 2026-09-11 GC 12.0.1 drop (20260916 release: `gc_12_0_1_uni_mes.bin` 0x93, ME
0xc12, PFP 0xc76, MEC 0xd7a; the 20260810 package had MES 0x91 with `uni_mes` a symlink to the 12.0.0 blob) was
installed in the guest and tested with nine launches: 16 `INVALIDATE_TLBS` timeouts in six four-card Flash-Next
launches (four of six affected, 0-6 each, four different cards), none in the two-card and 27B launches, no
escalation, throughput identical. On 0x91 the same day: 5 in four such launches. Not a fix. Rollback to the packaged
set (backup in `~/fw-20260810-backup`) recommended, Brian's call. The report for AMD is a comment on drm/amd #5759
(`notes/bug-mes-invalidate-tlbs.md`), which already describes this stack's problem on another box. Lesson from the
day: VM100 has no cron; `@reboot` hooks silently do nothing -- use systemd or run by hand.

### 2026-10-04: the MES timeouts have a workaround -- `amdgpu.mes_log_enable=1`

Method: six launches of the four-card Flash-Next server per condition (the launch is where every timeout we have
ever seen occurred; serving never produced one), one change per reboot, same box, same day, the kernel log watched
for `MES(1) failed to respond to msg=INVALIDATE_TLBS`.

| condition | launches | timeouts |
|---|--:|--:|
| MES 0x91 (linux-firmware 20260810, the Debian package), defaults | 4 | 5 |
| MES 0x93 (20260916), defaults | 6 | 16 |
| MES 0x8b (20260622, what production runs), defaults | 6 | 6 |
| MES 0x8b, `vm_update_mode=3` + `mes_log_enable=1` | 6 | 0 |
| MES 0x8b, `ras_enable=0` + `mes_log_enable=1` | 6 | 0 |
| MES 0x8b, `mes_log_enable=1` alone | 6 | 0 |
| MES 0x8b, defaults again (negative control) | 6 | 6 |

So: not the firmware version (three generations, same rate), not power or heat, not page-table update mode or RAS,
and not the box healing -- the one variable that moves the count is the MES event log, 0 in 18 launches with it
against 6, 6, 5-in-4 and 16 without. In the driver the flag only adds `enable_mes_event_int_logging = 1` and a
buffer address to the SET_HW_RESOURCES packet; timeout (2,100 ms) and polling are unchanged, so the difference is in
the firmware's behaviour when it is logging. Launch time, decode and prefill were identical in every condition.
PyTorch's expandable-segments allocator was also tried as a way to reduce mapping churn and does not start on this
stack (HSA memory fault at engine init; the IPC handles the all-reduce uses do not cover VMM-backed allocations).

Production (two cards, MES 0x8b, kernel 7.1.10, ROCm 7.14, idle for weeks) has 3 timeouts in its whole history,
all in one burst during its setup period; two-card launches on the test box run at a similar low rate, so the
production comparison mostly says "four-card launches are the trigger", not "production's stack is immune".

Applied (Brian, 2026-10-04 03:30 UTC): both VMs boot with `amdgpu.mes_log_enable=1 amdgpu.gpu_recovery=0`. The
test box is back on the Debian-packaged firmware (20260810, MES 0x91) and ran six more four-card launches clean
with the flag -- 24 of 24 across three firmware versions now, against ~1 per launch without it. Production rebooted
into the same parameters and its stack came back on its own (container restarted, endpoint answering in three
minutes). The drm/amd #5759 comment is written (`notes/bug-mes-invalidate-tlbs.md`); the tracker's spam filter
rejects API posts from the day-old account, so it goes in from the browser. Keep counting: the meswatch stays on
and every launch from here is a data point.

### Next
State at v0.2.0 (Flash-Next TP4, full BetterBench, vs Rob's image on the same box): decode +19%, step -18%, TTFT
1.5x, prefill +18..+29%, concurrency +12..+20%. What is left, in the order it looks worth doing:
1. ~~Per-fusion quality evals against a bit-reproducible baseline~~ -- done 2026-09-29, section above: no fusion
   costs accuracy on the full test set.
2. **Decode node count**: the stock MoE align / top-k / sum glue (~250 graph nodes per step at ~1.5 us of dispatch
   each), then the remaining GDN and spec-decode glue.
3. **Split-V GDN core** for high concurrency (at 16 sequences the fused kernel is state-bandwidth bound).
4. Compressed all-reduce: pipelining message halves across the local link and the uplink (~5% long prefill).
5. Prefill tail, each ~1% of a chunk: `hc_gate_mix`, a WMMA fp8-block GEMM for prefill widths, the `copyBuffer`
   device copies and the stock per-token quant of the block GEMM.
6. **First step, what is left** (2026-10-02): take the gated norm out of the token loop of `r9k_gdn_seq` (three
   block syncs per token; a second pass over the tokens needs none) -- a faster token would move the threshold
   past 256 / 400 and reach the 300-1,000-token prompts, where the 27B still trails the reference by 5-10%; the
   GDN metadata built once per KV group (stock, ~14 ms of a profiled step); the conv update in mixed steps.
7. **`UTIL=0.96` on two cards** is the default since 2026-10-02; the restarted launch has only a 5.7-minute
   soak behind it. Run the full mixed soak on a restarted launch before the next tag.

### Open questions for Brian
- v0.2.3 (the short-prefill GDN core, UTIL=0.96 on two cards) was tagged on 2026-10-02 on a smoke test, by
  Brian's decision ("no need to run full validation suite", "no need for a full GSM8K set"). Owed with the next
  release: a quality eval (short prompts take a numerically different, more exact path), a full BetterBench on
  the three configurations, the mixed soak on a restarted 0.96 launch.
- The published v0.2.0 and v0.2.1 notes say nothing of either bug. Edit them on GitHub, or let v0.2.2's notes
  carry it?
- Prefix caching is off by default for Flash-Next (`PREFIX_CACHE=1` restores cross-request prefix reuse): a
  serving-behaviour change, right for benchmarks and one-shot prompts, wrong for long multi-turn sessions.

## 2026-10-04 (evening): the MES workaround under a soak; radiance test run

**MES workaround, more evidence.** Twelve cycles of {Flash-Next four cards, 27B two cards, Flash-Next two cards}
on the test box (`~/mes-soak.sh`): launch, strict sanity (conc NS+1 x 10), a 5-8 minute mixed-length soak with 16
clients, teardown. 36 launches, 0 MES or GPU kernel events, 0 bad answers, 0 soak errors; with the 24 launches of
the reboot experiments that is 60 launches on `amdgpu.mes_log_enable=1` against the ~1 timeout per four-card launch
without it. One blemish unrelated to MES: in cycle 9 the 27B server answered `/v1/models` after 234 s (it normally
takes ~580 s) and then refused every connection; its log had 7 error lines, which the soak script deleted with the
container before anyone read them; no kernel event. The other 35 launches were normal. A container-log snapshotter
(`~/logsnap.sh`) now keeps the last 400 lines of the running server on disk for the next time.

**radiance.** Brian asked for a test run of https://codeberg.org/StillDeadcode/radiance, an LLM server with its own
C++/HIP engine and kernels (not vLLM). Built VM 102 on the production host to the test box's kernel and firmware,
handed it the two production cards (production VM stopped for five hours, idle anyway), ran the published image (the 1.0.4 build) and
containers through our release checklist with our clients, then gave the cards back. Write-up with the tables:
`notes/radiance-test-run.md`. In short: the 27B (FP8) is correct and stable, GSM8K 95.60% to our 94.69% (p = 0.06),
HumanEval equal, KV pool 549k tokens to our 367k, 38 s to serve; decode 112 vs our 200 tok/s combined (FP8 bytes vs
our 4-bit), prefill equal, concurrency 422 vs 558 at eight. Flash-Next is not usable under load by our bar: 2-6% wrong
or off-format answers in batches of 9-16 (clean alone; worse with speculation off; better but not clean with the exact
all-reduce wire), and prefill falls from 5.6k tok/s alone to ~400 tok/s aggregate with four long prompts in flight
(16-client soak: 72 requests served to our 239). Quality at conc 1 equal to ours (95.83% / 98.17%), time to first
token 141 ms to our 463, eight-client decode 143 to our 114.

## 2026-10-05: radiance 1.0.8 re-test, two cards

At Brian's request, the 1.0.8 image on the same VM and cards, raw numbers only (27B BetterBench and soak cut short,
Flash-Next without evals). Nothing changed on two cards: 27B GSM8K 95.60% again (1,261), probe 133 / 412 / 609 /
3,884, strict sanity 0 of 7,140; Flash-Next probe 56 / 191 / 258 / 5,670 fresh, strict sanity still failing in a
batch (6 of 270 at conc 9, 2 of 320 at conc 16, one clean run of 960; 0 of 60 alone), concurrent long-prompt prefill
still ~420 tok/s, the 16-client soak 26 requests in 759 s. The 1.0.3 -> 1.0.8 diff is four-card Flash-Next support,
a bs4 paged-attention kernel, tests and docs; two-card flags unchanged. Section added to notes/radiance-test-run.md.
Cards returned to production afterwards. Also corrected: the first run's image was the 1.0.4 build, not 1.0.3.

## 2026-10-06: radiance 1.1.1 on two, three and four cards

Brian: pull the latest (1.1.1, published that day), test for accuracy, and can TP=3 run. Two cards on VM 102 as
before; three and four on the five-card test box, since the production host has two R9700s and 1.1.0 added
three-rank serving for Flash-Next only (`docs/TP3.md`). 1.1.1 is the release that fixed Flash-Next on two cards:
strict sanity 0 bad of 2,260 (1.0.x failed ~2% of every batch), the 4-client long-prompt prefill 3,836 prompt tok/s
(was ~420), fresh decode 193 tok/s (was 56), GSM8K 95.83% with output identical to the 1.0.4 run on 99.8% of
questions, HumanEval 160. 27B: GSM8K 95.60% and identical output to 1.0.8/1.0.4 on 100% of questions, HumanEval 160,
probe 156 / 466 / 725 / 3,933 (+10-20%), sanity 0 of 8,540. Three ranks: GSM8K 95.91%, HumanEval 158, probe 205 /
568 / 600 / 5,053, long prompts 3,153 tok/s -- about 6% single-stream for the third card and nothing else. Four
ranks: every expert resident, probe 263 / 710 / 722 / 5,191, long prompts 3,365 tok/s, GSM8K 95.83%, HumanEval 159;
every paired comparison p = 1.00. `--expert-vs-cache-ratio` was removed in 1.1.x (compose files updated; my sed
broke their indentation once, relaunched). Section in notes/radiance-test-run.md. VM 102 left up with the cards;
the Flash-Next container and the TP3/TP4 compose files stay on the test box.

## 2026-10-06: sixth R9700 on the test host, chain C switch-local

Brian added a card to the third PLX and put a fan on the passively cooled one. Chain C = 05:00.0 + 08:00.0 at
0x380/0x390 with switch-local P2P; VM 100 gets a third emulated switch, X-PciMmio64Mb 384G and pci-hole64-size 3072G
(2048G stopped at 0x33f and broke placement for every card); guest 0b/0c, HIP 4/5. Verified 25.25 GB/s both ways on the
new pair, all 30 pairs ~13 GB/s one-way, 0 faults. radiance 1.1.1 refuses --tp 6 on the Flash-Next container (an
expert's 5 blocks of 128 do not split six ways); 2/3/4 ranks are what it serves here. Details host/README.md.

## 2026-10-06: radiance 1.2.0, Flash-Next TP4 + 27B TP2 side by side on the six-card box

1.2.0 adds a Quark MXFP4 27B container with vision and a --p2p flag, removes nothing we use. Flash-Next TP4 on HIP 0-3
and the 27B TP2 on the new chain C pair, together: probes alone vs together agree within noise (FN 264/740/786 vs
263/742/725; 27B 147/430/657 vs 152/432/656), strict sanity 0 of 1,900 and 0 of 7,180 with both loaded, GSM8K 95.83%
and 95.60% with output identical to the 1.1.1 runs on 100% of questions, HumanEval 158 and 160, mixed load 106 long
prompts at 3,579 tok/s beside a 16-client soak of 47 requests. BMC: 783-1,804 W at the wall under load. Chain C pair
runs 15 C hotter (max 67 C) with fans at twice the speed. 0 MES / errors / faults. Section in notes/radiance-test-run.md.


## 2026-10-07: decode node count -- three fusions, ~1% from launches, 2.6% with the P2P all-gather

Brian: "Is there any more tuning we can do on our kernel?" -> "Go ahead and work on it." Took item 2 of the list,
the decode node count on Flash-Next TP4. Built, unit-tested and A/B'd three things (notes/decode-nodes.md has the
method, the bugs and the numbers): one-launch MoE routing (softmax top-k + align tables, `r9k_moe_route`), a fused
top-k sum with the shared expert folded in plus the modular kernel's output alias (`r9k_moe_sum`, moe/fold.py),
and a one-shot P2P all-gather for the MTP head's decode-sized gathers (`r9k_ag_oneshot_nrank`). Step time (512-token
essay, MTP-3, 210 W): stock 16.70 / 16.70 / 16.67 ms; routing 16.56; +fold 16.56 x3 (control 16.68); +all-gather
16.28 / 16.27 / 16.25 (control 16.58); the all-gather ALONE 16.40 with decode 196 -> 200 tok/s, c16 897 -> 904. Quality gate on everything on:
GSM8K 95.60% vs the v0.2.4 record 95.68% (paired 14 vs 13, p = 1.00), HumanEval 160/164, strict sanity 0 of 1,020
after. The lesson is the number: in graph replay a tiny node's dispatch overlaps the previous node, so the 240
launches removed were worth ~0.14 ms, not the ~1.6 us each the 2026-09-28 profile implied; only the all-gather,
which removes GPU time (RCCL 47-127 us -> 6-13 us), moved the step. Four bugs on the way, each invisible to the
unit tests: NaN rows in vLLM's warm-up steps faulting an index-deriving kernel, a module cycle from storing a parent
on its child, `SharedExperts.output` consuming its slot, and a per-call flag read in a torch.compile-traced forward
(baked in at trace time -> double add). Defaults: all-gather ON (R9K_AG=1), routing and fold OFF (R9K_MOE_ROUTE=r9k / R9K_MOE_FOLD=r9k turn them on):
0.15 ms does not buy a replaced router and two vLLM-runner patches on the production path. Nothing tagged,
production copy untouched; the next tag owes the full validation on the all-gather (soak, BetterBench).


## 2026-10-09/10: decode step 16.42 -> 15.01 ms (-8.6%), single-stream 199 -> 217 tok/s

Brian: "ok, lets see if we can get our step time down then" (radiance 1.3.0 at 12.1 ms). notes/decode-step.md has
the method and every run. Two things moved it, both exact in unit tests: (1) the hyper-connection down + up GEMMs
at decode widths on the fp8 fragment-order copies the prefill mix path already holds (kernels/r9k_hc_f8.hip: bf16
WMMA against the bf16 activations, W8A16, loads software-pipelined; the bf16 pair had been the step's largest
stream at 1.29 GB) -- 16.42 -> 15.33; (2) the per-token fp8 quant fused into the dense block GEMM's A-load
(r9k_gemm_fp8_block_qa, operands bit-identical, 96 launches fewer) -- 16.06 alone, 15.01 with (1). Measured and
rejected: a 32 KB one-shot all-reduce cap (16.77), split-K across blocks for the hc down kernel (slower at every
setting), a scalar fp8 kernel before the WMMA one (15.64). The base repeated five times at 16.42-16.44. Gate at
conc 1 on the same tree: GSM8K 95.83% vs 95.75% (paired 8 vs 7, p = 1.00, outputs identical on 25.5%), HumanEval
161 vs 160, strict sanity 0 of 1,040 each. Both are the fn4 defaults now (R9K_HC_FP8_DECODE=0 / R9K_FP8_QA=0 off).
Also measured: a tiny node in a replayed graph costs ~3 us and forked branches do overlap, but the earlier finding
stands -- route + fold reran at -0.13 ms. rocprofv3 under vLLM writes nothing (workers SIGKILLed before the tool
finalizes); the torch profiler GPU-only (PROFACT=CUDA) still stretches the step to 23 ms, so the unprofiled split
of the remaining 2.9 ms to radiance stays an estimate. Nothing tagged; production copy untouched.


## 2026-11-01: post-load window 5 s clean / 12-16 s elevated -- the device, the GPU clocks and the queue knobs are all cleared; it is confined to the two host-loop-bound repack classes

Brian: "先发一版再继续" -- final47 went up first, then the hunt continued. notes/native-4xr9700.md C22 has the full
data. Three instruments went in, each answering one question the census could not:

**(1) the window probe** (`prof._probe_tick`, `R9K_PWAL_PROBE=0` off, `R9K_PWAL_PROBE_MS` spacing): from the load
thread, between `named_modules` yields, hand a 1-element kernel to the driver, stamp, whole-device sync. No second
CUDA context (0.97 util leaves no room) and no GIL-starved helper thread. CUDA event pairs are **unusable** at this
resolution (empty pair 6.7 ms, event-sync 11.9 ms against a 0.27 ms host call), so it stays on the host clock.
Reading: **identical in both regimes** -- submit p50 0.031-0.085 ms, submit+sync p50 0.09-0.34 / p95 3.1-4.5 ms, in a
5.0 s launch and in 11.5-14.1 s launches alike. Only 10-13 of ~65 possible samples ever get taken: the window lives
inside a few long calls, not in the loop, and the probe is (honestly) blind inside them. One tail98 rank paid a single
**8506 ms** submit+sync while submitting cost 0.031 ms -- real multi-second device waits, but that is shape (a); the
other three ranks of that launch never waited over 4.5 ms and still took 13.4-13.9 s.

**(2) per-class user CPU** at identical visit counts (79,253 / 79,374 / 77,694) and identical sys CPU (29-33 s):
`MergedColumnParallelLinear` **14.07x** (14.2 -> 199.3 s) and `RoutedExperts` **8.47x** (22.2 -> 187.8 s), while
`GatedResidual` 1.01x, `RowParallelLinear` 1.04x, `ReplicatedLinear`/QKV/RMSNorm ~1x. Shape (b) is confined to the
requant/repack classes, which also kills uniform explanations (frequency scaling, SMT contention would inflate
everything). Watchdog repeats `stuck in moe.py:255/257 _fold_hist state=R syscall=running` in slow launches only.

**(3) per-card SM clock paired with per-card busy** (`pp_dpm_sclk`, the `*` level; min/max-over-cards was useless --
tail103 had one rank at 11.7 s and three at 5.0 s in one launch, so the census now reports each card separately):
**no throttling** -- every busy sample is 2839-2887 MHz in the 15.5-15.8 s launch (tail107) exactly as in the 5.1-5.2 s
ones (tail104/106), and busy% is *lower* when slow (4-13% vs 4-35%). GPU clock and "the device is the constraint" are out.

What is left is the host: **`bios_limit=2400000` vs `cpuinfo_max_freq=3709357`** -- these 192 CPUs are platform-capped
at 2.4 GHz against a 3.71 GHz peak (schedutil, floor 1.5 GHz, not writable from this account), and `time_in_state`
cumulative is 785.8 M ms at 1500 MHz against 91.4 M ms at 2400 MHz. Inside a clean window the box spends ~93-100
CPU-s at the floor and ~6.5-7.5 CPU-s at 2.4 GHz. So the probe now also times a fixed interpreter loop on the load
thread (`prof._host_speed`): clean baseline **42.0-45.8 ns/iter p50** over nine rank-windows (tail108-110). That is
the reference for the one measurement still missing -- the same loop inside an elevated window. If it rises, the host
itself was slower (P-state/steal/contention) and the repack classes merely pay for it in cycles; if it holds at ~43 ns
while the window triples, the path is doing *more work* on the same tensors and the question becomes a code-count
question.

Also fixed on the way: the probe's disabled path left stale sample lists (now clears all four), and the census line
that mixed cards was replaced. `tests/test_window_probe.py` 19 checks (sampling, spacing, budget, host loop per
sample, off-switch reset, a raising probe cannot break the pass, `named_modules` restored).

Left running: **tail110** -- load 74.58 s, KV **483,024 tokens**, all four post-load windows **5.1-5.6 s**, health
200, GSM8K no-think **98/100**, needles **3/3**. Serve defaults unchanged (warm on, prestage on, `GPU_MAX_HW_QUEUES=1`,
`HSA_ENABLE_MWAITX=1`); no default behaviour changed this round, all new instruments are read-only observers.

**And the split answers on its first run (notes C23), which retires the per-class reading too.** tail118 with
`prof.splitcall` active: window 5.3 s wall, and the three repack helpers show issuing **0.00-0.28 s** against waiting
**25.3 s**, max single wait **5.69 s**, with `gpu_busy_percent` at **4-6%**. Timestamps put those waits at
16:53:26 -- the instant the window opens -- and at 16:53:44, inside the draft pass. So the post-load window is mostly
**one drain-wait at its head**, paid by the first call that asks the device for an answer, while the SMs sit idle;
the 14x/8.5x per-class user CPU from earlier this session is attribution, not slowness (`MergedColumnParallelLinear`
and `RoutedExperts` are just the first synchronising calls in a phase). The syncs cost nothing measurable: wall 5.3 s
with the instrument as against 5.1-5.6 s without. The question is now what is already in flight on the device at the
head of each phase -- DMA/copy backlog from pinned host tables, pin registrations, or another thread's work -- and
the instrument for it is the copy accounting that exists but is not switched on in these launches
(`R9K_LOAD_COPIES=1`), or `prof.split` extended with "copies +N (+X GiB) since the last note". Left running instead:
**tail118** -- weights 61.36 s + draft requant 6.51 s, KV **495,160 tokens**, all four windows 5.3 s, health 200,
GSM8K **98/100**, needles **3/3**.

## 2026-11-02: what `iteration_tokens_total` counts, and the Monitor's cards put on paper

`vllm:iteration_tokens_total` exports HELP "Histogram of number of tokens per engine_step". One request at a time,
delta of `_sum` and `_count` across it, plus which buckets moved (`serve/webui.py`'s own counters, 0.32.0 native):

| request | `_sum` delta | `_count` delta | records, read off the buckets |
|---|---|---|---|
| 4,893-token prompt, 4 generated | 4,897 | 2 | **4,893** and 4 |
| 29,969-token prompt, 4 generated | 29,973 | 3 | **29,969**, 4, 1 |

against `--max-num-batched-tokens 8192`. A chunked prefill is credited in **one** record holding the entire prompt
while each decode step gets a record of its own, so the `sum/count` ratio tracks prompt size, not the step budget:
9,621 tokens since launch and 14,999 in a 2 s window on the 262k launch, both above NBT. `iteration_tokens_total_count
/ requests` is records per request, not steps per request. Consequences: the Monitor's Generation card says "tokens
per iteration record (prompt whole, decode per step)" with this table in its tooltip (`serve/web/monitor.js`), and
the 2026-09-29 "ms per iteration / requests per iteration" table above -- measured on short prompts, where the
records really were decode steps -- should not be reused on a prefill-heavy load. Nothing in `/metrics` gives a
real step count; getting one means counting them in the plugin.

**The Monitor's five headline cards are drawn, not screenshotted** (`docs/monitor-cards.py`, README figure). The
screenshot is 1688 x 1953, so github.com's ~980 px column scales it to 0.58 and its text is unreadable, and
cropping it into a wide strip divides the same column five ways -- so the figure is drawn to the column instead
(1000 px, 3 + 2 cards), text measured against its own font size at 0.6 em a glyph and wrapped, with anything that
still does not fit reported rather than drawn off-canvas. `bench/capture-cards.py` supplies the sample: eight
concurrent prompts sized by a calibration request (this filler costs 24.3 prompt_tokens a repeat, so a word count
lands 3x off -- 46,165 repeats came back as 93,544 tokens), `/api/status` polled at 1 Hz, and the whole payload of
the poll with the longest waiting queue kept and committed as `docs/r9700-stack-monitor-cards.json`. Kept poll,
2026-11-02 06:47:35 UTC: `running 2 / waiting 6 / capacity 6 / deferred 0` at KV **10.1%** of 540,644 tokens,
prompt_tokens 29,984-29,994. Re-drawing the figure from that committed poll is a test
(`tests/test_readme.py`, 21 checks), so the picture, the poll and the caption cannot drift apart.

## 2026-11-03: what `./setup.sh` does when it is used for real

Six of its behaviours were wrong, and all six showed up in one bring-up on this box rather than in review. The
trigger is the kind of failure that reads as "slow boot" from outside: a `--launch native-flashnext.sh` started
by `setup.py` stopped writing mid-load at 05:45 UTC — log cut at **exactly 12288 bytes** on the
`multiproc_executor ... world_size=4` line, no error, no traceback, **no shutdown record in
`~/logs/requests-8080.jsonl`**, four cards back to 0.056 GiB. `setup.py` then polled `/health` for the full 420 s
*after the process was gone* and printed one line about it. **What killed it was the machine, not the launch:**
`journalctl --list-boots` ends that boot (`6a51a2dc`) with kernel lines at 05:45:19 — 36 s after the launch — and
the boot after it replays a machine-check record timed **05:47:12**; that boot died too at 05:57:43, and this one
opened at 06:00:23 replaying an MCE timed **06:00:06**. Three resets in 18 minutes. A new session was never shown
to be the problem, and saying so here at first was wrong; the reasoning and the corrected instrument are in
[notes/launch-methodology.md](notes/launch-methodology.md) §3, the note the CHANGELOG had been citing without
existing.

What changed, and what it measured:

| | before | now |
|---|---|---|
| the container path | `--yes` answered Yes: a multi-GB image pull as the **default** deployment | asked, **default n**; `--docker` / `--no-docker`; `--yes` does not answer it |
| a launch that died | noticed by the 7-minute `/health` timeout, log unread | noticed by the process (child exit code / tmux session gone) in seconds, log's last 14 lines printed with `\r` folded |
| what supervises it | nothing (`start_new_session` only) | tmux session `r9700-native-flashnext`, env pushed with `tmux new-session -e` (tmux would otherwise run the serve in the **tmux server's** environment: `VENV=`, `GPUS=`, `R9K_*` gone, silently) |
| what a later run knows | existence ("libr9k.so, 5164 KiB") | drift: `.r9700-state/native.json` (venv versions, `libr9k.so` hash **of both copies** — build.sh's and the one loaded via `R9K_LIB`, README's build line separates them with a `cp` — newest kernel-source mtime) and `launch.json` (recipe/log/session/`up_at`/`died_at`) |
| what it refuses | nothing, then fails two minutes in | ranks still holding the GPUs (**5 processes** here, `refusing to launch… STOP=1`), and `OFFLOAD_GB` over `RLIMIT_MEMLOCK` (**12,010 MiB** cap against the two-card stream's **34 GB a rank** — the origin of PROGRESS.md's "11.7 GiB per process") |
| `WEBUI=same` | Monitor printed at `PORT+1` — a wrong URL | the serve's own port, where `serve/plugin` actually puts it |

Live result of the same command that failed: `up after 145 s` (cold, 120.3 GB checkpoint, TP4 MTP-2 32k), `/health`
200, `/monitor` 200, a chat completion answering, a second `--launch` refused with the `STOP=1` line, and a later
`--check` printing `the launch recorded 3 min ago (serve/native-flashnext.sh) is up: 5 vLLM process(es); tmux attach
-t r9700-native-flashnext`. `tests/test_setup.py` (57 checks, no GPU/card/port) runs in step 6 beside
`tests/test_webui.py`; it caught one real bug — the dead-launch warning never named the log it had just read — and
then caught itself: two of its checks passed vacuously on a box that happened to have a live serve, so `left_vllm`
is now stubbed in both directions.

Still not done, deliberately: nothing **restarts** a dead four-card serve. tmux supervises in the sense of keeping
it alive under a dying terminal, not of resurrecting it, and until the PCIe/reseat story
(notes/incident-2026-11-03, `tools/reseat-check.py`: BadTLP growing at slot 8 with nothing running) is settled, an
auto-restart would turn a degrading link into a restart loop. The container path is still unexercised on this box
— no `docker` binary on it — so the `--docker` branch is tested by `tests/test_setup.py`'s decision logic, not by a
build.

## 2026-11-03: the Monitor and About headers say which build is answering

Asked from the box: "is what is live the version I just merged?" That answer used to be an ssh, the tmux session's
launch log, and a hash against `git log`. `stack_info()` (`serve/webui.py`) reads the plugin's `__version__` plus the
branch, short commit and dirtiness of the tree that plugin was imported from, once at Hub start, and `/api/status`
ships it as `stack` — `{"plugin": "0.4.0", "branch": "master", "commit": "953a3901", "dirty": false}` — so both
headers read `vLLM 0.32.0 · plugin 0.4.0 · 953a3901` (`*` on a dirty tree) and About carries the same as a fact row.

What it took was the second draft, not the first: the Monitor repaints the header every poll, and the first version
appended, so the line would have read `vLLM 0.32.0 · vLLM 0.32.0 · …` a minute after anyone looked at the page. The
helper now keeps the element's base text and rebuilds from it, and `tests/render-monitor.mjs` — which already runs the
page's JS in a stub DOM — waits past the 2 s poll and fails if the line grew at all. Details (why the fields are
absent rather than guessed, why it rides the engine-unreachable payload, and what the line deliberately does not
track) are in notes/webui.md.

## 2026-11-04: what the box is doing ON AVERAGE -- the Monitor gains 1 min / 15 min / 1 h

Asked from the box with six streams in flight: "decode and prefill are how fast? the Monitor's display is realtime,
it gives too little feedback on the actual average level." Three instruments, the same afternoon:

- **One 1 s poll:** 9.9 tok/s of generation, no prefill at all, 890 ms a step with six running. A poll a minute
  later, caught in a gap: 0 tok/s, running 1, KV 3%. That is the instrument the headline numbers were built on.
- **The ten minutes around it** (the request journal, 37-46 finished requests): mean prompt 90-110 k tokens,
  prefill **7,306-7,876 tok/s per request** and **6,967 tok/s across the span**, generation **41-43 tok/s across the
  span**, decode **20.2 tok/s per stream** (median), ~4 req/min, mean queue 3.3 s.
- **The whole launch** (523 min, 1,189 requests): 70,015,813 prompt tokens against 678,007 generated, 2.2 req/min --
  so 2,232 tok/s of prefill and 21.6 tok/s of generation averaged over everything, and a box that is prefill
  dominated rather than decode dominated.

`/api/status` now also carries `avg`: `1m`, `15m`, `1h`, each computed as a **delta of the engine's cumulative
counters over the delta of time**, and a span latency as `delta(sum)/delta(count)` of the engine's own histogram
families. The Hub keeps a second ring, sampled every 5 s for an hour plus slack, for exactly this. Means of the 1 s
sparkline samples would have been easier and wrong: they weight an idle second like a second with six streams
running. Two fields are new rather than re-cuts of old ones -- `prefill_tok_s` (prompt tokens over the engine's own
prefill seconds) and `decode_tok_s` (generated tokens over decode seconds) -- and `stream_tok_s` divides by the
**mean** running across the span, not by whatever this poll saw. Each rate card prints the line under its live
number; the KV card gains a mean pool over each span.

The refusals are the part worth keeping. A span the page has not had yet keeps its real label (`90s`, `2m`) instead
of the one it asked for; a ring younger than 15 s reports nothing; a span whose counters went down says
`reset: true` and prints "engine restarted" rather than a negative rate; a histogram family the base snapshot did
not have is a dash rather than a division by the whole cumulative counter (that is how a missing
`request_prefill_time_seconds_count` was caught). And the histogram-backed speeds are credited when a request
finishes, so a span shorter than the typical wall time of a request is a mean over the requests that happened to
finish in it -- measured, a 47 s span said 4,840 tok/s of prefill where ten-minute spans say 7,000-8,200.

`tests/test_webui.py` gained 12 checks against a synthetic ring with exactly known counters; `render-monitor.mjs`
prints the five average lines and fails if a card claims an average the payload does not have, or prints `1h` over
40 s of data. Verified live by running the standalone sidecar on :8099 beside the production serve -- the merged
page takes the same code at the next restart, and the serve was not restarted for this. Details in notes/webui.md.

## 2026-11-04 (later): the headline rate becomes an average, and the waveforms gain an hour

The trailing averages went in beside the live number -- and the live number stayed the big one, which is the number a
reader looks at. So the fix went the rest of the way: the four rate cards answer to a **window picker** (this poll /
1 min / 15 min / 1 h, default **15 min**), and the four waveforms pick between the 90-point ring (one point per poll)
and a new hour ring (240 points, one per 15 s).

Why 15 min and not 1 min: a request on this launch runs 17-50 s of wall time, so a 1 min window is still a sample of
one or two requests. In the same minute, `1m` said 30 tok/s, `15m` said 52, and the poll said 13 -- a 4x spread across
the width of a page glance. The poll is not removed, it is demoted: every card's second line carries it (`poll 13`),
and the poll is the only thing that shows a step of 905 ms sitting inside a long prefill while the span says 103 ms.

The rules: a window is only offered once it has **filled** (80% of its width), and if the chosen one has not, the page
shows the widest that has and says so ("1 h has not filled yet - showing 1 min (65 s of counters)"); young spans are
labelled with what they averaged (`90s`, `2m`) everywhere, including the second line; and caption spans come from the
samples' own timestamps, because `points x poll` was true of one ring and a lie about the other. The hour ring takes
its rate over the interval (deltas of cumulative counters, mean running across the interval for the per-stream
divide), gives None when a counter went backwards inside it -- a restart is a gap in the waveform, not a spike of
-100 million tok/s -- and does NOT guard the KV-growth series that way, because a negative there is the pool draining,
which is what that series is for.

`render-monitor.mjs` now derives which window the page should have settled on, pulls the first number out of each
headline and fails if it is not that window's value, fails if a headline is printing this poll under a 15 min label
when the two differ by more than 25%, fails if an averaged headline does not carry `poll <value>`, and presses both
trend buttons to make sure the source ring actually changed. Handlers are attached by id, not by a `querySelectorAll`
sweep, so the stub DOM can press them. Details in notes/webui.md; the production serve was left running and picks the
whole thing up at its next restart -- the standalone sidecar on :8099 is where it is live today.

## 2026-11-04 (later still): a number changing width must not move the page

**The report:** 数字长度变化时可能造成换行变化 - when a number's length changes, the wrapping changes. Real, and
worse than it looks: the note under a stream headline went from one line to two when its number went from `2.0` to
`95.7`; the card grew; the rate row grew; and with `grid-auto-flow: dense` on that row the cards have room to change
columns. The theme already kept the *headline* digits fixed-width (`.value` carries the mono stack and
`tabular-nums`), but it had no rule at all for the note lines - and those 12 px lines carried no font-family, so they
rendered in Inter while the headline above them rendered in mono.

**The fix: fixed slots.** `.slot{white-space:nowrap; text-overflow:ellipsis; font-variant-numeric:tabular-nums}` and
every number-bearing note line in the rate row, the trend row and the pool card became one: rate card = headline +
three slots (why/which, the other spans, this poll), trend card = wave + two (level, spread), pool = headline + two.
The always-wrapping classes were deleted rather than tuned. Overflow ellipsizes and the sentence moves to the
element's `title`. `knum()` caps the width of every number that changes inside a sentence - `999.9`, `16.2k`,
`1.01M`, five characters - so digit counts cannot change the length of the text around them.

**The budget is arithmetic, not taste.** Same grid maths the release page figures use: 0.60 em per glyph, inner card
177 px at the narrowest viewport where a 4-up row still exists (981 px) and 354 px at the 1560 px cap, 12 px text -
so **24 characters** is the tight case, 44 the loose one. `tests/render-monitor.mjs` reads the slot list out of
`monitor.html` (it cannot drift from the markup), prints what the live page writes into each, fails above 44 and
flags above 24. Everything renders at 24 or under.

**What it cost, stated.** Shortening to fit is not free: `x 1.30 tokens a step` - the bridge between the decode rate
and the step rate - left the visible line and came back as `poll 890 ms · x2.33` with its explanation, and the
since-launch fallback, in the title. `pts`, the axis range and the sampling interval moved the same way. And
`· 90s back` was dropped from the wave caption: it reads as "90 seconds ago" when it means "this window spans 90 s",
which is the window bar's job.

**What the check is not.** No browser on this box, so nothing lays the page out and measures pixels. The guarantee
rests on the arithmetic above plus the fact that a slot cannot wrap by construction - the worst a wrong budget can
produce is a clipped word, never a moved card. `tests/test_webui.py`: 349 checks (six new). Clean against the :8099
sidecar (which has `avg` and the hour ring) and against the merged `:8080` page (whose payload has neither, and says
so in the slots rather than throwing).

## 2026-11-04: upstream v0.4.1 merged - and the honest list of what this box cannot gate

`git merge upstream/master` brought the tag **v0.4.1** (25 commits: one-kernel eight-rank all-gather / all-reduce
that never cross a root complex, the DeepSeek decode glue, an EXL3 GEMV for 1-48 rows, the fused MoE combine). The
`v041` and `v042` branches are pointers, 0 commits from master. Upstream touched none of the QSA / fp8-KV / webui
files, so the conflicts were the same two as last time: `kernels/build.sh` and `README.md`.

**A mistake worth keeping.** The first attempt at the build.sh resolution **destroyed the file**: the tail was
rebuilt from "the first line containing `hidden_hostcall_buffer`", and that string is also in the file's header
comment, so everything between line 4 and the conflict vanished (23 sources, the compile line, the guard).
`bash -n` did not object -- a 22-line truncated script is valid bash. What caught it was running the build
(`line 12: D: command not found`) and then a structural count: sources 23, symbols 115, guard present. So the
resolution check is written as counts now, and recovery was `git checkout -m -- kernels/build.sh`, which restores
the conflicted version mid-merge (note it relabels the markers `ours`/`theirs`, which broke the marker strings the
script keyed on).

**Composition, not choice, for the README.** Upstream rewrote it into a shorter conventional outline. The merged
file takes upstream's tagline + badges, What it does, Features, Speed (DeepSeek now 87-97 prose / 202 code single
stream, 179-212 / 368 at four), How it fits together, Quick start, Build on it, Documentation, Development,
Feedback and License; keeps this fork's lockup opening, generated release page, four-card table, the three
native-box speed subsections, the Monitor section and the one-command deployment blocks; and adds fork rows to the
Documentation table for `notes/native-4xr9700.md`, `context-262k`, `kv-fp8-qsa` and `webui`. Upstream's NOTE box
("every number here was measured on one machine: eight R9700s at 210 W") is false for this README, whose first
table is a 4-card uncapped box, so it names both machines and says they do not compare.

**The figure follows the numbers.** `docs/summary-page.py`: DeepSeek 158 -> 202 on code (with "v0.4.0: 158 code, 73
prose" beside it), the card line becomes "EXL3 GEMV + expert decode", the GLM note "one-kernel 8-rank collectives",
and the footer gains the held-back tier sentence -- v0.4.1's 48-token expert tier was 1.5-3x faster on one card and
left the drafter accepting nothing in the server, the same trap this fork's fp8 KV measured (accepted per draft
1.60 -> 1.55). Canvas 1000x1302, PNG 2000x2604. `test_readme.py` now measures every `docs/r9700-stack-v*.svg` the
tags ship (9.8 / 13.7 / 9.0 / 9.0 px in github's column) instead of one hard-coded file, and requires the caption
to name the newest.

**Gates on this box**, with the merged tree and a rebuilt `libr9k.so` (13,052,128 bytes -- the new kernels roughly
double it; 115 exports including `r9k_qsa_attn_f8`; no `hidden_hostcall_buffer`):

- `tests/test_qsa_fp8.py` PASS -- fp8 kernel vs its own reference 1.7e-3..3.9e-3, indistinguishable from bf16.
- `tests/test_qsa_r9k.py` PASS; `tests/test_integration.py` **ALL OK** against this box's stock vLLM
  0.32.0+rocm723 -- the gate that catches a plugin expecting entry points this pin does not have.
- `test_webui.py` (349 checks), `test_webui_plugin.py`, `test_readme.py`, `test_setup.py` PASS;
  `node tests/render-monitor.mjs http://127.0.0.1:8099` clean.
- Library replaced by rename + copy, backups `libr9k.so.pre-v041` in both locations (`.pre-f8` still under both
  from the fp8 work). **The live serve was not restarted**: it still answers `plugin 0.4.0 / 953a3901`, which is
  what the header feature is for.

**What this box cannot gate, said out loud:**

- `tests/test_ar8.py` wants `torchrun --nproc-per-node=8`. Four cards here, all four held by the live serve
  (~1.4 GiB free each).
- `tests/test_exl3_gemv.py` (and the other EXL3 tests) `import exllamav3_ext` at module level with no guard;
  exllamav3 is built inside upstream's image and is not in this venv.
- `tests/test_dsv4_glue.py` runs and reports **95 ok, 10 fail**. Every failure is a bit-exactness gate against the
  installed vLLM's ops: ids identical, fused-router weights within 3e-8 at M<=12 and 6e-8 at M>=24, and one
  attention-input check off by 3 ulp on 2 of 192 values. Upstream's numbers were taken against their 0.32.1rc1
  nightly; this box is 0.32.0+rocm723, and 3e-8 is a different op's rounding, not a broken kernel. Not chased --
  this box does not serve DeepSeek -- but the consequence is recorded: **the v0.4.1 DeepSeek path is unvalidated on
  this pin**, and serving DeepSeek here needs the image (or exllamav3 in the venv), not this tree alone.

Rollback for the merge is `git revert -m 1 <merge>`; for the library alone, the `.pre-v041` copies.

## 2026-11-04: the README carries the speed the box actually runs at, not only the speed it is benched at

**Ask:** 把实际使用的速度测量数据也放到 README 中 — the README had benchmark ladders and one real-traffic
subsection from the bf16 launch; the fp8 launch that has been serving since 15:29 had nothing.

Ten hours of real traffic on the fp8 launch (journal boot `1793719607-24497`, 1,568 completed rows, 29% of the
span with traffic in it) now sits in the README, computed twice from two sources and cross-checked:

- journal: 1,568 rows, prompts p50 52,083 (p90 149,572, p95 163,513, max 198,073), generated p50 322.
- counters: `prompt_tokens_total` 106,686,448 against the journal's 106,214,313; `generation_tokens_total` 877,747
  against 874,128 -- **under 0.5% apart**, which is what makes the per-request numbers trustworthy.
- own decode by max concurrency during its own decode window: one **162.0** tok/s (n=585), two 114.5 (294),
  three-four **41.0** (423), five-six 23.9 (127), all 1,525 requests generating >=40 tokens median **115.4**
  (mean 100.9, p10 17.1, p90 171.1). Per-second concurrency: 1 -> 47.5% of busy seconds, 2 -> 25.1, 3 -> 17.1,
  4 -> 7.7, 5 -> 2.2, 6+ -> 0.4; p50 2, p90 4, max 8.
- aggregates, four denominators: 23.7 over the span, 80.8 over busy seconds, 42.1 token-weighted, 115.4 median of
  rates. From the counters alone: 23.6 and 42.0.
- prefill by prompt size: 8,158 (<20k, n=179), **8,937** (20-80k, n=882), 7,970 (80-160k, n=376), **7,542**
  (>160k, n=101) tok/s medians; and `prompt_tokens_total / request_prefill_time_seconds_sum` = **7,949 tok/s while
  prefilling**, within 1% of the journal's buckets. TTFT p50 **6.50 s**, p95 **24.52 s**, max 56.1 s; the longest
  prompt served, 198,073 tokens, first token at 27.0 s at 7,330 tok/s.
- drafter over the whole launch: 376,433 drafts x 2 tokens, 752,866 draft tokens, 501,271 accepted -> 1.33 per
  draft, **2.33 tokens a step**, 66.6%.
- the Monitor's own three spans (1 min / 15 min / 1 h) put in as a table, including the 1 min prefill figure of
  4,214 tok/s kept **labelled as the metric artefact it is** (`prompt_tokens_total` credits at request end).

The top four-card row changed from `158 alone / 61 as a request finishes` to **`162 alone / 115 as a request
actually finishes`**, because the launch running today is the fp8 one.

**What this is not, said in the README itself.** The bf16 window (1,400 requests, prompts p50 74,381, p90
concurrency 6, median 60.8) and this one differ in traffic mix as well as KV dtype, and the mix is the dominant
term -- so the two are printed side by side as *windows*, not as an fp8-vs-bf16 result. The controlled fp8 numbers
stay the paired arms (pool 540,202 -> 1,009,899, 32-stream aggregate 1,823 -> 1,792, acceptance 1.60 -> 1.55).
The comparison table even says "should not be read as one".

`docs/summary-page.py` gained the fourth fact ("115 tok/s ... median over 1,525 of them"), so the generated release
page carries the in-use number next to the benchmark ones; canvas 1000x1310, PNG 2000x2620, and the README caption
follows. Full method, the concurrency histogram and the thin-bucket caveats: notes/context-262k.md.
