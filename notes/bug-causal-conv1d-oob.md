# `_causal_conv1d_fwd_kernel` faults at KV 99.8 %: the stock Triton conv the device names, and the 1 h 48 m nobody noticed

The standing 262k launch (`~/logs/setup-005730.log`, up since 2026-11-09 00:57:48, five days twenty hours of
traffic) died on 2026-11-14. The device named a kernel this repository does not own: vLLM's own Triton prefill
conv, which the plugin's GDN path deliberately does not cover. The second finding is worse than the fault: it
stayed dead for **1 h 48 m** and nothing on the box said so — `launch.json` still carried `died_at: null`, and
`notes/launch-methodology.md` §5 was describing a watcher that did not exist.

## 1. The fault, both sides of it

Device side (`rocdevice.cpp:3948`, 15 identical blocks, one address, one read):

```
Warning: Queue error - HSA_STATUS_ERROR_MEMORY_FAULT
[host: jiulu-9700, GPU index: 0, faulting addr: 0x738f7a313000, kernel: _causal_conv1d_fwd_kernel]
grid=[9856, 10, 1]  workgroup=[128, 1, 1]  completion_signal=0x0  rptr=16208123330 wptr=16208125469
```

Kernel side, same second — and the kernel names the **physical card**, which the HSA line does not:

```
Nov 14 20:59:09 amdgpu 0000:c3:00.0: [gfxhub] page fault (src_id:0 ring:24 vmid:8 pasid:10925)
Nov 14 20:59:09 amdgpu 0000:c3:00.0:   Process python3 pid 251454        <- Worker_TP0
Nov 14 20:59:09 amdgpu 0000:c3:00.0:   in page starting at address 0x0000738f7a313000 from client 10
Nov 14 20:59:09 amdgpu 0000:c3:00.0: GCVM_L2_PROTECTION_FAULT_STATUS:0x00801430
Nov 14 20:59:09 amdgpu 0000:c3:00.0:   Faulty UTCL2 client ID: SQC (data) (0xa)
Nov 14 20:59:09 amdgpu 0000:c3:00.0:   MORE_FAULTS: 0x0  WALKER_ERROR: 0x0  MAPPING_ERROR: 0x0
Nov 14 20:59:09 amdgpu 0000:c3:00.0:   PERMISSION_FAULTS: 0x3  RW: 0x0
```

`RW: 0x0` is a read; `MAPPING_ERROR: 0x0`/`WALKER_ERROR: 0x0` with `PERMISSION_FAULTS: 0x3` is the shape of a
read inside a mapping the process owns but outside the object it was handed — the same shape the page-aligned
fault in [bug-router-gemm-oob.md](bug-router-gemm-oob.md) had. `MORE_FAULTS: 0x0` and one kernel-side page:
one bad read, not a walk off a corrupt base.

The Python traceback does **not** name the faulting kernel, exactly as in the router case: the first stack frame
is `torch.ops.r9700.hc_down_silu` → `r9700_vllm/hc.py:216 down_silu` → `_fp8_gemm` →
`r9700_vllm/kernels/moe.py:396 quant_rows_fp8` → `torch.empty((M,), ...)`. That is the next reader that tripped
over the poisoned page; the device already said which kernel made it poisoned. Card: `0000:c3:00.0` = slot 3
(`tools/reseat-check.py`: `card1 GPU c3:00.0 slot 3`). `tools/gpu-identity.py`'s warning is live here — that
card is **amd idx 2**, while the HSA line calls it "GPU index 0" (the worker's own rocdevice index). Trust the
kernel's BDF, not either index.

## 2. Which conv this is: the step that reached vLLM's kernel, not ours

`r9700_vllm/models/gdn.py` runs the short-prefill core (our `r9k_gdn_conv` + one recurrence launch) only when
`PREFILL_MAX` (default 256) covers the step's prefill tokens **and** `md.num_decodes == 0` (`gdn.py:378`). The
last step the engine scheduled, dumped at 21:04:09 by `dump_input.py:79`:

| | |
|---|---|
| new request | `chatcmpl-b1a7b2061d7c84c9`, `prompt_token_ids_len=528`, `prefill_token_ids_len=610`, `num_computed_tokens=0`, **610 tokens scheduled** |
| cached requests | 30, each scheduled **3** tokens (`num_scheduled_tokens` 3 = 1 + 2 MTP drafts), `preempted_req_ids=[]` |
| batch total | `total_num_scheduled_tokens=700`, `num_spec_tokens_to_schedule=2` |
| engine | `num_running_reqs=31` (`--max-num-seqs 32`), `num_waiting_reqs=13`, `kv_cache_usage=0.99825` |
| deepest sequence | `num_computed_tokens=236,153`; then 178,053 / 107,637 / 40,587 |

610 prefill tokens **> 256**, and the step is mixed (30 decodes), so on every GDN layer this step went through
stock `causal_conv1d_fn` → Triton `_causal_conv1d_fwd_kernel`. That is the kernel the device named. Nothing
about it is ours: `R9K_GDN_PREFILL_CONV` only chooses who does the conv *inside* the short-prefill path, and
this step was never in that path. The boot log even shows the shape was not warmed:
`01:00:50 jit_monitor.py:140 Triton kernel JIT compilation during inference: _causal_conv1d_fwd_kernel` — i.e.
first use of that configuration came from live traffic.

Boot proof that our paths were on (the fault is not a fallback-to-stock story):
`r9700: GDN short-prefill core on r9k_gdn_seq (steps with <= 256 prefill tokens ...)` and
`r9700: GDN MTP decode core on r9k_gdn_decode_mtp`, plus four ranks x
`r9k router GEMM installed on 48 gates (M <= 4, split 2)` — the cap from [bug-router-gemm-oob.md](bug-router-gemm-oob.md)
holding, and this launch logged zero illegal-access lines in the router kernel in five days.

## 3. The teardown, to the second

| time (11-14) | what the log says |
|---|---|
| 20:59:07 | last three journal records complete (`queue_s` 514 / 1,104 / 754 s) |
| 20:59:09 | gfxhub page fault + `HSA_STATUS_ERROR_MEMORY_FAULT`; `Worker_TP0 (pid 251454)` `torch.AcceleratorError: an illegal memory access` (`multiproc_executor.py:1063`, 218 ERROR lines) |
| 21:00:10, 21:01:10, 21:02:10, 21:03:10 | `shm_broadcast.py:871 No available shared memory broadcast block found in 60 seconds` — the engine core is waiting on a rank that is already gone |
| 21:04:09 | EngineCore `mq.dequeue(timeout=...)` gives up (`core.py:1512`), `dump_input.py:79/81` writes the scheduler output above, `async_llm.py:906` `EngineDeadError`; then `serving.py:929` logs **8 tracebacks** carrying **8,412** error-prefixed lines (vLLM repeats the tag on every line of a multi-line traceback — count the `Traceback` lines, not the tag, or you get 1000x the truth, as this note did on its first pass). What actually died is in the dump: **31 running + 13 waiting** requests, none of them journalled (the journal holds only completed records); `[shutdown] MPClient: start timeout=0s` |
| 21:09:09 | ranks 1/2/3: `WorkNCCL(SeqNum=9000991, OpType=_ALLGATHER_BASE, NumelIn=5649280, NumelOut=22597120, Timeout(ms)=600000) ran for 600009 / 600064 / 600076 ms` |
| 21:10:09 | `To avoid data inconsistency, we are taking the entire process down` → `terminate called after throwing an instance of 'c10::DistBackendError'`; `resource_tracker: 4 leaked semaphore / 6 leaked shared_memory` |
| 21:10:09 | the tmux session's pane closes, so the tmux server (pid 251036) exits — `last -x` shows `tmux(251036).%0 ... - 21:10 (5+20:12)` |

The ten minutes between 20:59 and 21:10 are PyTorch's NCCL watchdog: rank 0's death leaves ranks 1-3 blocked in
`_ALLGATHER_BASE` until `Timeout(ms)=600000` fires, and their abort is what finally frees the launch. Nothing
was killed from outside; the ranks ignore SIGTERM anyway (`serve/native-flashnext.sh`'s own pre-flight, and
`left_vllm()`).

## 4. The load it was under

`~/logs/requests-8080.jsonl` begins at 11-14 17:42:39 with the dead launch's boot id
(`1794185869-251038`), so the 3 h 17 m before the crash are fully covered: **6,503 requests, 9,659,404 prompt
tokens, 5,257,671 completion tokens**, 5,338 of those requests arriving inside the 17:xx hour alone.

What made it a near-full pool:

* **twelve requests generated the whole context**: `completion_tokens` 261,571–261,633, `finish_reason=1`
  (length), started 17:57–19:48, wallclock 2,682–4,846 s each. A thirteenth ran to 261,628 too. `max_tokens`
  arrives from clients as ~261,6xx — nothing between the endpoint and the scheduler bounds it.
* **423 preemptions over 68 requests**, max 63 on one request — the KV pool is over-subscribed, so the scheduler
  is evicting and recomputing.
* **queue up to 1,958 s (33 min)**: median queue over the whole journal is 0 s; 1,958 s is the all-time max, and
  the last three admitted requests queued 514–1,958 s.
* the pool was 66 % full by *tokens* but 99.8 % full by *blocks*: the 30 cached requests in the dumped step hold
  595,562 of 902,153 tokens (four of them 562,430), and with an 832-token block and 1,146 blocks
  (`--kv-cache-dtype fp8_e4m3`, mamba cache mode `align`) the remainder is not grantable in 832-token units.

So: 31 sequences, twelve of them long enough that each one alone wants a quarter of the pool, one 610-token
prefill arriving into that, and the Triton conv reading past its object.

## 5. What is NOT the cause

Checked with this repo's own tools, on the crash window rather than by impression:

* **not the link**: `tools/reseat-check.py` — the faulting card's whole chain is clean
  (`c3:00.0 / c2:00.0 / c1:00.0 / c0:01.1`, `AER cor=0 unc=0`, 32.0 GT/s x16, D0). The chain that does carry
  errors (`04:00.0`, 678 correctables, card4/`06:00.0`) belongs to a different card.
* **not a hardware event at the crash minutes**: `journalctl -k` 20:30–21:30 holds **zero** AER / CPER / MCE /
  reset / OOM lines; the only kernel lines in the window are the page-fault decode above. `tools/journal-pcie.py`
  for this boot (since 2026-11-05): 1,342 AER events, 612 CPER blocks, all `severity corrected`,
  `fru_text PcieError`, concentrated on `04:00.0` (BadTLP 574) — the usual riser-chain noise, none of it at
  20:59.
* **not the sick L2 core**: `r9700-offline-core.service` enabled, cpu17/113 still offline
  (`/sys/devices/system/cpu/online = 0-16,18-112,114-191`), no MCA/MCACOD today (this boot's only mce line is
  the 2026-11-05 HEST threshold line).
* **not the SVM/mm-notifier family**: `journalctl -k` today holds 0 `userptr` lines — the signature that sent
  the router-gemm investigation looking there is absent.
* **not OOM-killer, not a reboot**: uptime was 9 d at the time of writing, and the journal for the day holds no
  `killed process` line.
* **not the router cap**: it is still at 4 and this launch ran five days without a router-kernel fault.

## 6. The supervision gap, and what the fix is

`launch.json` after the crash read `"died_at": null, "pid": null`. `launch()` writes the record at the start of
a launch, waits for `/health`, prints its report **and exits** — so `died_at`/`reason` only ever get written for
a launch that died *before* it first answered `/health`. A launch that came up and died five days later kept a
record that still said `up_at`, and the next run's `launch_status()` clause
("recorded dead HH:MM:SS") had nothing to print. `notes/launch-methodology.md` §5 claimed those fields come
"when the watcher saw it die"; there was no watcher.

And the one thing that was supposed to outlive the launch did not: this box runs `WEBUI=same`, which puts the
Monitor/About pages **inside** the serve (`notes/webui.md`), so the page went down at 21:04:09 with the engine.
"Seen, not resurrected" is only true if something outlives the death; here nothing did, and the death was found
at 22:58 because an operator asked.

What is in the tree now (`setup.py`, `tests/test_setup.py`):

* `setup.py --watch`: a detached watcher armed by `launch()` once `/health` answers (`arm_watch`, pid recorded
  in `launch.json` as `watcher`/`watcher_at`). On death it writes `died_at`/`reason` and appends one line to the
  launch's own log, so a `tail` shows the death too.
* **It records; it does not restart.** §4's rule of 2026-11-08 stands — a four-card serve that restarts into a
  degrading link turns one fault into a loop, and this box does have a degrading link (card4's chain, 678
  correctables). Anyone who wants the loop is one flag away, but it has to be said out loud.
* `watch_tick` is pure and has eleven checks in `tests/test_setup.py` (88 total, all pass, no GPU): a watcher
  must not stamp a record naming someone else's session; it must not re-stamp a death already on the record
  (the first timestamp is the one worth having, which is also what makes two watchers safe); and a launch still
  loading weights — no `/health` yet, ranks present — must never be called dead.
* `launch()` now says, in the same breath as the context check, that the Monitor page lives inside a `WEBUI=same`
  launch and so is not what will report this death.

## 7. What the kernel's own indexing allows, and the step replayed on a real card

`tests/test_conv1d_mixed_step.py` (27 checks: 11 pure geometry -- no GPU, no vLLM; 12 for the fallback bounds
guard of §8.2; 4 replayed on one card). It pins
four things that change the search:

**The dispatch arithmetic is exactly the recorded one.** `grid = (num_program, cdiv(dim, BLOCK_N))` with
`BLOCK_M = 8` (fixed) and `BLOCK_N = 256` (passed at the launch), one program per 8-token chunk of one
sequence. For the dumped step: `ceil(610/8) = 77` programs and `cdiv(2560, 256) = 10` — and the HSA record was
`grid=[9856, 10, 1] workgroup=[128, 1, 1]`: 77 x 128 = 9856. So the call was **one non-spec (prefill) row of
610 tokens, 77 programs, conv dim 2560 per rank** (the whole model's conv is 10,240 wide, TP4). vLLM's own
`compute_causal_conv1d_metadata` gives the same `tot = 77`. There is no program/token/line index outside its
buffer for that shape — nor for the other shapes this launcher can put on the wire (8192-token chunk = 1024
programs, prefill + 30 reclassified decodes, zero-length padded rows, a state line at the pool's last block,
a dim that is not a multiple of BLOCK_N).

**The plugin was not in that step at all.** `gdn.py:589` takes the short-prefill path only when `_prefill_ok`
(`num_prefill_tokens <= R9K_GDN_PREFILL_MAX` = 256, and `num_decodes == 0`); `gdn.py:592`'s mixed
long-prefill+spec handling needs one-page GDN state, and this launch logged at boot
(`gdn.py:325`): *"one-page GDN state not applied at the model level (Qwen4ExpForConditionalGeneration): stock
mamba pages"*. So the step falls through to `gdn.py:613` → **stock** `_forward_core` →
`qwen_gdn_linear_attn.py:1451`, with **stock's own** `non_spec_query_start_loc` and
`non_spec_state_indices_tensor`. Our conv (`r9k_gdn_conv`) never saw that step.

**The usual suspect is off for this model.** The kernel's three-level block indirection (the shape that
normally produces a garbage cache-line index) only runs when `IS_APC_ENABLED`, i.e. when
`block_idx_last_scheduled_token` is not None — and the Qwen GDN call site passes none of
`block_idx_first_scheduled_token / block_idx_last_scheduled_token / initial_state_idx / num_computed_tokens`.
The only indirection in play is `cache_indices[idx_seq]`, 1-D, = `block_table_tensor[non_spec_masks, 0]` —
column 0 of the mamba group's block table (`align` mode, `mamba_get_block_table_tensor`). A bad *value* there
lands on a wrong-but-legal line (silent corruption); a bad value large enough to leave the conv-state buffer
is exactly one page-aligned read fault, and the kernel-side fault names the scalar path
(`Faulty UTCL2 client ID: SQC (data)`), which is where that index load happens. Ruled in as the surviving
suspect, not established.

**And the frozen-graph family does not apply to this step** (an inference from the launch's own config, not an
observation of the step): 700 scheduled tokens is above `max_cudagraph_capture_size = 256` (in the live argv,
confirmed in `setup-230248.log`), and `cudagraph_capture_sizes` is `[1,2,4,...,256]`, so there is no captured
graph for a 700-token step: `grid` and `batch_ptr` were computed for it live. (At ≤ 256 tokens the
frozen-grid/persistent-`batch_ptr` story stays a live suspect — worth remembering for the next fault, not for
this one.)

**The replay itself is clean.** Arm B runs stock's `causal_conv1d_fn` at that shape on a real card
(~50 MB, `validate_data=True`, the state line index set to the largest legal block 1145 of 1146):
no fault after `torch.cuda.synchronize()`, `out` finite with the expected `(2560, 610)`, and the conv state
written at that one line and nowhere else. So the shape alone does not reproduce it — a repro needs the real
pool and the real allocation map, not the geometry. That is the same conclusion
[bug-router-gemm-oob.md](bug-router-gemm-oob.md) reached from the other direction: *indices in range as the
kernel describes them, pointing at memory that no longer describes the object*.

One asymmetry the replay makes visible: the plugin's own `_debug_check` already asserts exactly the invariant
that would catch the surviving suspect (`idx max >= state blocks`), and it runs on **our** paths. The stock
fallback path — the one that ran here — never runs it.

## 8. What would settle the fault, in order of cost

1. ~~Shape/geometry~~ — done, §7: in range, and the step replays clean. **Now also for the second fault**
   (`tests/test_conv1d_mixed_step.py` arm E, 2026-11-15): the 05:11 step read out of the engine's own dump — 23
   spec rows of 3 tokens (drafts still `-1, -1`) plus two new prefills of 160 and 156 tokens landing on GDN pages
   the step itself zeroed (`new_block_ids_to_zero=[246, 464]`, which the dump's sixth block group agrees with) —
   is **also** in range by arithmetic, and its conv update → conv → `torch.cat` at the step's own q/k/v widths
   (512/512/1536) replays clean on a real card with syncs between the stages. Two deaths, two named kernels, two
   step shapes, both in range: the arithmetic is not it, which is what leaves an index that is in range and no
   longer names live memory.
2. **Instrument the fallback path** — *done, 2026-11-14* (`gdn.py:_bounds_problems` / `_fallback_check`, and
   12 checks in `tests/test_conv1d_mixed_step.py` arm C). With `R9K_GDN_DEBUG=1` the fallback now bounds, before
   stock's kernels run, the two walks that kernel can make: every conv-state line index in
   `non_spec_state_indices_tensor` / `spec_state_indices_tensor` against `conv_state.size(0)`, and the non-spec
   token span against `num_actual_tokens`. It costs host syncs, so it stays behind the existing `DEBUG` gate
   (default off) and it runs on **layer 0 only** — the block table and the token counts are per-step, so layer 0
   sees what all 36 GDN layers of that step will see (config.json's `layer_types`: 36 `linear_attention`, 12
   `full_attention`). Off, it is one `if` per layer. This is the check that would
   have named a bad line index in the log instead of leaving the device to name it: our own paths already had
   that assertion in `_debug_check` (`idx max >= state blocks`); the path this fault ran never did.
   A detail the check forced out: the conv kernel skips a slot whose index equals `NULL_BLOCK_ID = 0`, so it
   cannot tell padding from a real block 0 — harmless, because the block pool reserves block 0 as its null
   block (`kv_cache_utils.py:1606`), and now said in the code rather than remembered.
   **Fixed 2026-11-15, before anyone could run it in anger:** that layer-0 restriction lived inside
   `_fallback_check` alone, so the rest of the `DEBUG` block still paid itself on every layer — 3
   `torch.cuda.synchronize()` calls per layer per step on each of the two paths, **108 per step** across 36
   layers, which drains the launch queue and moves the very prefill/decode overlap a reproduction is chasing.
   Both call sites now ask `_dbg_here(self)`: layer 0 unless `R9K_GDN_DEBUG_ALL=1` says otherwise. Nothing is
   lost, because the metadata is per step, and a fault that lives in a later layer still faults and still gets
   its kernel named by the driver. Checked in `tests/test_conv1d_mixed_step.py` arm D: no bare `dbg = DEBUG` is
   left in the file, every sync inside `gdn_core` sits under an `if dbg:`, and `_dbg_here` is exercised as a
   function (layer 0 yes, layer 7 no, both with `DEBUG_ALL`, neither with `DEBUG` off).
3. **Reproduce the step, not the pool** — *restated 2026-11-15, because the second fault refuted the plan as
   written.* It used to say: drive `kv_cache_usage` to ~0.998 with ~12 requests at ≥200k tokens, then a steady
   arrival of 500–700-token prompts. The 2026-11-15 05:11:18 fault took the same class of device fault on a step
   at **`kv_cache_usage=0.591`, `running=25`, `waiting=0`**, so pool pressure is not a condition. What the two
   deaths have in common, and nothing else measured does: a step mixing **new prefills with spec-decode rows**,
   on the same card, with the same fault shape — a **read** (`PERMISSION_FAULTS: 0x3`, `RW: 0x0`,
   `MAPPING_ERROR: 0x0`) at a page-aligned address, `[gfxhub] page fault` on `0000:c3:00.0`. The kernels named
   differ: 11-14 `_causal_conv1d_fwd_kernel` (grid `[9856, 10, 1]`, addr `0x738f7a313000`); 11-15
   `at::native::CatArrayBatchedCopy_contig` (grid `[13312, 3, 1]`, addr `0x766adbac9000`). Two different kernels
   reading past the end of the same allocation is exactly why the guard is written as a **bounds check on the
   arguments** rather than as a fix to one kernel: the named kernel may be the first innocent reader rather than
   the one that produced the bad index. So: a spare launch, `R9K_GDN_DEBUG=1` (cheap enough to leave on for a
   soak now that it is layer 0), mixed steps of 500–700-token prefills arriving into spec-decoding requests, run
   at **low** pool pressure as well as high — two arms, since the pool is no longer believed to matter. Against a
   *spare* launch, never the box behind traffic: a hit costs the ~10-minute NCCL teardown (§6), and
   `tools/recipe-check.py` will call whatever runs afterwards non-PRODUCTION. Since 2026-11-15 05:58 the watcher
   can be told to bring such a soak back after a hit (`WATCH_RESTART=1`, four gates:
   launch-methodology.md §4) — which is the difference between an overnight soak and a scheduled outage.
4. `AMD_SERIALIZE_KERNEL=3` if (3) does not hit: names the faulting kernel rather than the first innocent
   reader, at the cost of every timing the protocol produces.
5. If it returns and stays unreproducible, the mitigation is a config change: `R9K_GDN_PREFILL_MAX=1024` puts
   steps like this one (610 prefill tokens) on our conv and out of the faulting kernel — but that has to be
   *measured*, because PREFILL_MAX=256 is where the one-page conv work was validated, and
   `tools/recipe-check.py` will hold the live engine to whatever value it becomes.

## 9. Box state while this is written

Relaunched at 23:02:48 from the stored configuration (`./setup.sh --config 262k`,
`~/logs/setup-230248.log`): up after 190 s, `KV 902,153 tokens / Maximum concurrency 3.44x at 262,144`,
`health 200`, and `tools/recipe-check.py` reports every measured knob, every `R9K_`/`R9700_` name and every
launch flag of the live serve matching this tree. The dead launch booted at the same 902,153 / 3.44x
(`setup-005730.log`, 01:00:09), so nothing drifted between the two: the fingerprint is the recipe's, not the
incident's. The watcher is armed on the new launch; the next run of `./setup.sh` will answer "is it still up"
from `.r9700-state/launch.json` rather than from an operator's memory. `tests/test_conv1d_mixed_step.py` was
run against this box (one card, ~50 MB, GPU[0] still at 32.2 GiB used afterwards) with the serve in traffic
throughout: health 200 before, during and after.

Re-checked after writing, because two numbers in this note were wrong the first time round:

- the engine-error count (§3): 8 tracebacks / 8,412 tag-prefixed lines, not 8,412 requests;
- the check counts: `tests/test_setup.py` is 85 checks (the methodology doc said 57, then 88 — both wrong; the
  count of *executed* checks is what the run prints), and `tests/test_conv1d_mixed_step.py` is 27 when vLLM and
  a GPU are importable (11 geometry + 12 bounds + 4 replay) and 11 + 2 named skips on the host python.
  22 of those are `ck(` calls in the source — arm A's shape table runs one check over six shapes.
- the watcher was proved end-to-end on a throwaway tmux session inside a temporary state directory (nothing
  written while it was alive; `died_at` + `reason` + a log line when it was killed; the live record untouched).
  One stray `--watch` process from an aborted verification was found and killed; a stray is inert — it can only
  stamp a record naming its own session — but it does poll until the launch dies, so `pgrep -af "setup.py
  --watch"` is worth a look after hand-testing.
- `tests/test_webui.py` failed **once in eleven** runs during this session and passed the other ten, including
  six back-to-back afterwards; it is full of timing windows (0.2 s polls, 0.12 s streams, sleeps) and the failure
  was not reproduced while hunting it. Left as a known flake rather than papered over: if it comes back, the
  run prints which check.

## 10. It reproduces: 431 s of synthetic mixed-step load, and layer 0 was clean on every step

`tools/soak-mixed-step.py` (added the same day) drives the shape §8 item 3 asks for: a standing population of
requests decoding with MTP, a **pair** of ~210-token arrivals every 1.0-1.5 s (a pair, because one short prefill
is under `R9K_GDN_PREFILL_MAX=256` and goes to our kernel while the pair is over it and goes to stock's -- the door
both faults went through), a 610-token arrival every 6 s (the 11-14 shape), fresh unique prompts so prefix caching
cannot dissolve the mixed step, and `--max-load` as a valve so the queue matches the dumps rather than growing
without bound (the first unmetered 2-minute smoke reached 37 waiting; the dumps sat at 25/0 and 31/13). It aims at
the engine on `:8180`, not the gateway: the admission gate is built to suppress exactly this regime, so testing
the fault past the gate would be testing the gate.

Run: TEST launch at 06:11 (`~/logs/TEST-gdnmixed-1115-0612.log`, `R9K_GDN_DEBUG=1` layer 0, `WATCH_RESTART=1`),
soak started 06:21:22. **Hit at 06:28:33, 431 s in** -- the driver's first failing request was a short arrival
answered `HTTP 500`, and `/health`, `/metrics` and the page all went silent inside the same second.

The kernel journal, verbatim (this is the third fault of the family and the same card as the other two):

```
Nov 15 06:28:33 kernel: amdgpu 0000:c3:00.0: [gfxhub] page fault (src_id:0 ring:24 vmid:8 pasid:14055)
Nov 15 06:28:33 kernel: amdgpu 0000:c3:00.0:  Process python3 pid 499003 thread python3 pid 499003
Nov 15 06:28:33 kernel: amdgpu 0000:c3:00.0:   in page starting at address 0x0000748241794000 from client 10
Nov 15 06:28:33 kernel: amdgpu 0000:c3:00.0: GCVM_L2_PROTECTION_FAULT_STATUS:0x00801430
Nov 15 06:28:33 kernel: amdgpu 0000:c3:00.0:          Faulty UTCL2 client ID: SQC (data) (0xa)
Nov 15 06:28:33 kernel: amdgpu 0000:c3:00.0:          PERMISSION_FAULTS: 0x3   MAPPING_ERROR: 0x0   RW: 0x0
```

pid 499003 is `Worker_TP0`. Same card as 11-14 20:59 and 11-15 05:11; same read (`RW: 0x0`,
`PERMISSION_FAULTS: 0x3`, `MAPPING_ERROR: 0x0` -- a mapping that is there but not readable, not a walk failure).
`Faulty UTCL2 client ID: SQC (data)` is new information: the read came in on the **scalar** data path, which is
how a kernel reaches its own arguments and buffer descriptors, so "an index that is in range but no longer names
live memory" is consistent with what the hardware says. Unlike the two earlier faults, the driver did not print a
`kernel: <symbol>` line this time (ring 24 rather than the trap ring), so the faulting kernel is unnamed.

The step it died on is the step we were asking for. The last instrumented steps on that worker:

```
06:28:30 gdn debug ... fallback T=505 md=prefills=2 decodes=0 spec=29 act=505
06:28:31 gdn debug ... fallback T=501 md=prefills=2 decodes=0 spec=28 act=501
06:28:32 gdn debug ... fallback T=481 md=prefills=1 decodes=0 spec=29 act=481   (then core ok, then norm ok)
```

`running=30`, `kv=37.5 %`, `waiting=0`. So §8 item 3's other conclusion is confirmed on the box, twice now:
**pool pressure is not a condition of this fault.**

What the guard did, counted over the whole launch for `Worker_TP0`: **1150 `fallback T=` lines, 1150 `fallback
core ok`, 1150 `fallback norm ok`, and 0 `BAD`.** No step is missing a stage. That is the important negative of
the run, and it says the stock `_forward_core` (its conv, its `torch.cat`, its norm) of **layer 0** completed for
every step the instrumentation saw, including the last one it saw. Combined with the worker's own reported stack
-- which vLLM itself warns is where the error surfaced, not where it was caused --

```
gpu/model_runner.py:1735 execute_model -> add_requests -> sampler.apply_staged_writes
  -> states.py:75 apply_staged_writes -> temperature.copy_to_uva
  -> buffer_utils.py:38  return self._uva[:n]     torch.AcceleratorError: illegal memory access
```

-- the picture is: an already-launched kernel faulted, and the first place the driver could tell the process was
a host->device copy of the sampler's per-request temperatures for the requests being added.

So the two live hypotheses are now separated by a knob rather than by reasoning. `R9K_GDN_DEBUG_ALL=1` makes all
36 GDN layers print `T=` / `core ok` / `norm ok`; if a fault lands with a `T=` and no `core ok`, that layer's stock
`_forward_core` is the faulting kernel, named at last. If every layer prints `norm ok` right up to the fault, GDN
is out of it entirely and the hunt moves to the kernels that step also runs (the sampler's staged writes, the MTP
draft, attention). That run is in progress as this is written (`~/logs/TEST-gdnall-1115-0700.log`,
`tools/soak-mixed-step.py --minutes 25 --pop 12 --max-load 24 --period 1.5`): 22k debug lines/minute, ~11 MB
per 2.5 min, which is affordable and did not stop the fault the first time round.

Two things this run also established about the harness itself:

- **The layer-0 instrumentation does not mask the fault.** Stage syncs and host reads of the index tensors were
  live on every step and it still faulted in 7 minutes. The soak-plus-guard pair is therefore a usable standing
  reproduction, not a fragile one.
- **The guard's span rules were added before the hit, from the `torch.cat` clue, and are now in the file**:
  `_bounds_problems` also checks `non_spec_query_start_loc[-1] + spec_query_start_loc[-1] <= num_actual_tokens`
  and `num_actual_tokens <= rows` (the rows of the buffer actually handed to stock, `qkvz.shape[0]`, which
  `_fallback_check` now receives). A cat reads a *range*, and the 05:11 grid `[13312, 3, 1]` has three in the
  second dimension, which is how many tensors `torch.cat((query, key, value), dim=-1)` -- `qwen_gdn_linear_attn.py`
  lines 1026 and 1116 -- passes. On the 05:11 step as dumped, the new rules are clean too (316 + 69 = 385 rows,
  385 present): `tests/test_conv1d_mixed_step.py` arm C fires them on purpose-built shapes and arm E holds them
  clean on the dumped one.

## 11. What a hit costs, measured: the ranks stay, the pane stays, and the box stays down

The 431-second soak did to the box what the two field faults did, and the failure of *recovery* is now measured
rather than assumed:

- The fault killed `Worker_TP0` and left the other three ranks alive: `pgrep -cf "VLLM[::]"` returned 3, and
  rocm-smi showed **32,371,838,976 / 32,371,949,568 / 32,296,308,736 B used** on the other three cards (card 0
  back down to 374,820,864). Those ranks ignore SIGTERM. (The process titles are `VLLM::Worker_TPn`, which is why
  `left_vllm()` greps `VLLM[::]` and a `pgrep -f "vllm serve"` finds nothing in a crash.)
- The tmux session survived: `pane_pid=498588 cmd=bash dead=0`, 26 minutes after its engine stopped.
- With `WATCH_RESTART=1` armed, `RESTART_MAX=3 / 3600 s`, and a watcher polling every 10 s, **the record never got
  a death at all**: `died_at: null`, no `restarts.json`. `watch_dead()` required either the session to be gone or
  no ranks left, and it had neither. The §4 rule ("seen, not resurrected") failed in *both* directions at once:
  it was never seen, so nothing could resurrect it, and an operator who did not look found a refusing port.
- Worse, once the ranks were cleared by hand (`STOP=1`), the *empty* session still refused the next launch:
  `FAIL a tmux session named r9700-native-flashnext is already up -- not starting a second launch into it`. That
  corpse pane was the reason the box was still down at 06:45.

Fixed in `setup.py` the same morning, all of it host-testable and all of it gated:

1. `watch_dead(port, sess, prev, quiet)` can now say **"crashed and wedged"**: no `/health`, ranks present, a
   record that had already answered `/health` once (`up_at`), at least `WATCH_WEDGE_TICKS` (2) silent probes in a
   row, and a known crash signature in the log tail. A launch that is merely loading has ranks and no `/health`
   too, and is still not called dead -- that is the same test as before, kept.
2. `RESTART_STOP=1` (default on once `WATCH_RESTART=1` has been said out loud; `RESTART_STOP=0` restores the old
   passivity) lets the watcher clear those ranks with **the recipe's own stop** --
   `STOP=1 bash serve/<recipe>` -- rather than a hand-rolled `pkill`, because the launcher knows its ranks and an
   ad-hoc kill would be a second, worse opinion about the same processes. The predicate `stop_wedge_would()` is
   pure and needs all five: switch on, restart armed, ranks really there, a signature, and a record that names a
   recipe. A purely recording watcher never touches anybody's processes.
3. `corpse_session(sess, left, rec)` turns the launcher's refusal into a decision: close the pane when the record
   says this launch is over and nothing is running in it; refuse when there are ranks, when the record does not
   vouch for the session, or when the launch might still be loading (a load with no ranks and no `/health` yet is
   not a corpse).

10 checks in `tests/test_setup.py` hold those three to their rules, including the two that must stay `None`/`False`
(loading, and silent-but-no-fault-line). And the *other* direction got proved on the box without being planned:
at 06:42:46 the armed watcher logged `this launch is gone: the tmux session r9700-native-flashnext holding it is
gone -- nothing restarted it (no known crash signature in the log tail)` for a deliberate `STOP=1`. A stop is not
a crash, and the box now knows that in production rather than only in a test.

## 12. `R9K_GDN_DEBUG_ALL=1` named the dying layer: `layers.1`, inside stock's `_forward_core`

Second soak, same shape, one knob different: `R9K_GDN_DEBUG=1 R9K_GDN_DEBUG_ALL=1` (`~/logs/TEST-gdnall-1115-0700.log`,
soak from 06:47). **Hit at 06:55:41, ~536 s in**: `[gfxhub] page fault` again on `0000:c3:00.0`, `pid 513343` =
`Worker_TP0`, page `0x000076f7e1208000`, `PERMISSION_FAULTS: 0x3`, `RW: 0x0`, `Faulty UTCL2 client ID: SQC (data)`.

Then the instrument paid for itself. Counting, for the dead worker only, how many times each of the three stages
printed per GDN layer (36 of them, `layers.N.linear_attn: fallback T=` / `fallback core ok` / `fallback norm ok`):

```
layers instrumented: 36 | layers whose three counts differ: {1: [615, 614, 614]}
```

**One layer of 36 is short by one: `layers.1` printed its step 615 times and its `core ok` 614 times.** The last
line that worker ever wrote about a step is the `T=` of layer 1 (`fallback T=469 md=prefills=1 ...`), while TP1,
TP2 and TP3 all printed `core ok` and `norm ok` for the same step. And the raise site in the worker's own traceback
is the stage sync that brackets exactly that call:

```
File "/home/jiulu/r9700-stack/r9700_vllm/models/gdn.py", line 633, in gdn_core
    torch.cuda.synchronize()
  ... frame above: <string> __vllm_inlined_submods__5   (the compiled model's inlined GDN submodule)
```

That is the localization §8 was after. It also retires run 1's red herring: the sampler's `copy_to_uva` stack was
where the driver *told* the process, not where the damage was -- vLLM says so in the same traceback, and here the
sync placed moments after `_forward_core` caught it instead. So: a kernel launched by **stock's `_forward_core` for
GDN layer 1**, on **one rank of four**, on a step whose prefill+spec mix three other ranks survived.

The guard did not catch it, and the reason is a bug in the guard rather than in the theory: `_fallback_check` kept
its **own** hardcoded `.layers.0.` test even after the call site learned to ask `_dbg_here(self)`, so with
`DEBUG_ALL` every layer paid for the syncs while only layer 0 ever had its indices read. The dying layer's
`non_spec_state_indices_tensor` / `spec_state_indices_tensor` / spans were never looked at. Fixed the same day --
the helper now checks whichever layer it is handed and the **caller** decides who pays -- with the check in
`tests/test_conv1d_mixed_step.py` rewritten to say so, plus a source check that no `.layers.0.` test comes back
inside the helper.

Third soak, 07:04 (`~/logs/TEST-gdnall2-1115-0720.log`), same shape (`--pop 16 --max-load 30 --period 1.0`,
`running=30`, `kv≈38 %`) with **all 36 layers bounds-checked**: **720 s with no fault and 0 `BAD` lines**. The rate
across the three runs is therefore 431 s, 536 s, and >720 s -- stochastic at about one hit per ten minutes of that
shape, which is the same order as two faults in ~12 h of real traffic. What run 3 bought is coverage: the next hit
gets its offending index printed by the layer that died, in the log, before the device names it in a page fault.

Recovery, this time proved in the field rather than in `tests/`: `.r9700-state/restarts.json` carries

```
"at": [06:36:37, 06:55:57], "sig": "illegal memory access", "session": "r9700-native-flashnext"
```

with the watcher's own line `this launch is gone: the tmux session ... is gone -- RESTARTING: illegal memory access;
restart 2 of 3 in 60 min`. The 06:55 fault was answered 16 s later by a fresh launch through `setup.py` -- which is
the whole point of the mechanism, and it is now the fourth time this box has faulted and the first time it came back
without a human.

Two honest caveats from the same records:

- **The wedge path never fired.** Both restarts have `cleared_ranks: null`: in each case the tmux session had
  already gone, so the ordinary death predicate applied and `RESTART_STOP` was not needed. It stays a tested path
  without field service, and §11's measurement of the wedge still stands.
- **The signature gate reads the log tail, and a tail is long.** The 06:36:37 restart happened after an operator ran
  `STOP=1` on the wedged launch: the fault line from 06:28 was still inside the 256 KiB tail, so the watcher read it
  as a crash and relaunched. That is the right answer for "the engine died and I cleared it" and the wrong one for
  "I want it down" -- so stopping a launch on purpose while a watcher is armed means taking the watcher down first
  (`.r9700-state/launch.json` names its pid). A narrower predicate (signature only after the last shutdown marker)
  is a candidate follow-up; it is not written yet, and saying so here is cheaper than being surprised by it.

## 13. Upstream has already fixed a bug whose mechanism this is: vllm#61174, and our wheel does not have it

Asked whether there is a better way than a hand-written shadow path, the answer turned out to be a search of
upstream rather than more instrumentation. Method, since `api.github.com` is unauthenticated and out of quota on
this box: `git clone --filter=blob:none --depth=4000 https://github.com/vllm-project/vllm.git` (4 000 commits,
2026-09-15 … 2026-11-15 — the whole window since our pinned 0.32.0 wheel) and read the log for the code the fault
window touches: `v1/worker/gpu/block_table.py`, `v1/worker/mamba_utils.py`, `v1/core/single_type_kv_cache_manager.py`,
`model_executor/layers/mamba/**`, `models/qwen4_exp/amd/**`, and messages matching `conv1d`, `mamba`, `GDN`,
`piecewise`, `stale`, `illegal memory`, `num_speculative_blocks`.

**Found: `61bb5908` — "[Bugfix][Spec Decode] Clear stale shared block-table rows" (#61174, 2026-10-17), in
`vllm/v1/worker/gpu/block_table.py`.** `BlockTables` keeps one **shared** `input_block_tables` buffer that the
step's kernels -- including everything replayed out of a captured piecewise graph at a padded batch size -- read
by row. Before the fix, `gather_block_tables` only zeroed rows up to `num_reqs_padded` *of this step*, and
`get_dummy_block_tables` only zeroed up to its own count; rows left behind by an **earlier, larger batch** stay
behind. The fix adds `_num_dirty_rows` and clears `max(needed, dirty)` rows in both places. Its own test docstring:
*"A smaller target batch must leave no stale rows for larger draft graphs."*

That is the mechanism §10-§12 left standing, told from the other side:

- every index in the dumps is **in range** (§7, arm E, and the span rules), because a stale block id is a *valid*
  id -- it just names a block that has since been freed and reallocated, which is how an in-range read lands on an
  unmapped page;
- the fault is a **read** on the **scalar** path (`SQC (data)`), consistent with a kernel following a block id from
  a table row rather than with arithmetic on its own bounds;
- it needs a step where batch size **moves** (this box pads to cudagraph sizes 1,2,4,8,16,24,32,...; `running`
  swings 23→31 constantly, so a 32-row step is followed by a 24-row one within seconds) and both faults were on
  such a step (23-30 spec rows plus 1-2 new prefills);
- and the mamba/GDN state path is where it lands: the pre-fix code already carries the comment *"dummy runs write
  mamba state to the reserved null block rather than through the previous real step's (stale) block ids, which may
  point at blocks since freed and reallocated"* -- §5.7 of the same file, one function away.

**Verified on this box rather than assumed.** Upstream's test file (`tests/v1/worker/test_gpu_block_table.py` at
that commit) copied into `tests/test_vllm_stale_block_table_rows.py`, run against the installed wheel:

```
before the patch:  7 failed     # rows 5-8 of the shared table still hold [5,5,5] [6,6,6] [7,7,7] [8,8,8] / 99s
after the patch:   7 passed
```

(the copy drops upstream's `skipif(not current_platform.is_cuda())` marker -- on a ROCm build that is `False`, so
upstream's own regression test for a mamba-state block-table bug **never runs on the platform where it bites**.)
The patch is `patches/vllm-0.32.0-61174-clear-stale-block-table-rows.patch`; the `_gather_block_tables_kernel`
in our 0.32 is byte-identical to upstream's pre-fix copy, so it applies cleanly (`patch -p1`, 60 lines, python
bookkeeping only). It is **applied to `~/venv-r9700` at 08:49:31** (`stat` on both files agrees) and reported by
`setup.sh`'s box walk
(`VPATCHES` / `venv_patches()` in `setup.py`, plus `patches/README.md`), because the venv is a frozen artifact and
a patched 0.32.0 must not be mistaken for a stock one. **The running engine -- launched 07:18:15, well before 08:49:31 -- predates it and does not carry it** --
that is what the walk's "a launch that started BEFORE the patch does not carry it" line is for.

What the same search ruled OUT, recorded so nobody spends an hour on it twice:

- `6fa67afe` *"[Core] Retire stale Mamba spec blocks in prefill checkpoint steps"* (#59759) — **already in our
  build**, line for line, including the post-fix comment. Not our missing piece.
- `c6b2d6ec` *"[ROCm] Pad Qwen4Exp PLE n-gram inputs for piecewise CUDA graph replays"* (#62675) — our build
  **lacks** it, and the mechanism (a PIECEWISE replay reading a buffer arranged for a different request count) is
  the same family, **but** our checkpoint has `ple_layer_ids = None` / `ngram_size = None`, so the PLE path it
  fixes never runs here. Not applicable.
- `2f6f7861` *int32 token-offset overflows in GDN prefill / checkpoint kernels* (#63761) — the hunks land in
  `mamba/checkpoint.py::store_cache_checkpoints_kernel` and the FLA chunk kernels; our 0.32 `checkpoint.py` is a
  different 169-line file without that kernel (it is the KDA path), and the arithmetic at our shapes
  (state lines ≤ 1 146, step ≤ 8 192 tokens) is ~2 orders of magnitude from 2³¹. Not this fault.

Still to earn, before this is called a fix instead of a candidate: a soak arm at the shape that hit at 431 s and
536 s, run on a launch that actually carries the patch, **twice clean** (≈2 × 20 min). A single clean arm is a
coin flip at this rate.

## 14. §13's candidate is disproved: the fault came back 250 s into a launch that carried #61174

Arm 1 of the plan at the end of §13 — a launch **with the patch in it**, same shape as the run that hit at 431 s
(`R9K_GDN_DEBUG=1` layer 0, `--pop 16 --max-load 30 --period 1.0`), named
`~/logs/TEST-patch61174-a1-1115.log`, soak from 09:21:33:

- **Fault at 09:25:56, ~250 s in** — the *fastest* of the four, against 431 s (unpatched, layer 0) and 536 s
  (unpatched, every layer). `[gfxhub] page fault` on `0000:c3:00.0`, `pid 565881` = `Worker_TP0`, page
  `0x000072c85aa48000`, `PERMISSION_FAULTS: 0x3`, `MAPPING_ERROR: 0x0`, `RW: 0x0`,
  `Faulty UTCL2 client ID: SQC (data)` — indistinguishable from the other three.
- Instrumentation ran and said nothing: layer-0 stage counts **348 `fallback T=` / 348 `core ok` / 348 `norm ok`
  on all four workers**, 0 `BAD`.
- That the patch was really in the process, proved rather than assumed: the installed file's mtime is 08:49:31,
  that launch's first vLLM-timestamped line is `INFO 11-15 09:18:06` (its own clock, ~29 min after the
  patch; the fault line itself is `09:25:56`, ~7.5 min of serving), and
  `r9700_webui.patch_text()` in that same venv prints `#61174 present`. (`/proc/<pid>` mtimes are NOT process
  start times — that trap was walked into once while writing this; `ps -o lstart=` is the honest source.)

So **clearing the stale shared block-table rows is not this fault** — the upstream bug is real (7 failing tests
became 7 passing) and the box keeps the patch for that, but the device fault family is untouched by it. One
hypothesis retired by experiment, which is the point of having a reproduction that takes four minutes.

What the four faults now say when lined up together:

| when | kv_usage | running/waiting | seconds into a fresh launch | rank | card |
| --- | --- | --- | --- | --- | --- |
| 11-14 20:59 | 99.8 % | 31 / 13 | ~6.8 days | (not recorded) | `c3:00.0` |
| 11-15 05:11 | 59.1 % | 25 / 0 | ~28 min | TP0 | `c3:00.0` |
| 11-15 06:28 | 37.5 % | 30 / 0 | 431 s | TP0 | `c3:00.0` |
| 11-15 06:55 | 30.2 % | 24 / 0 | 536 s | TP0 | `c3:00.0` |
| 11-15 09:25 | 35 % | 30 / 0 | 250 s | TP0 (patched) | `c3:00.0` |

**Every one of them is the same physical card, and every one where the rank is known is `Worker_TP0`** (which is
what `HIP_VISIBLE_DEVICES=0,1,2,3` puts on `c3:00.0`). Nothing else in the table is a condition: pool pressure
spans 30 % to 99.8 %, queue spans 0 to 13, uptime spans 4 minutes to 6.8 days, and the patch is on or off.

That leaves two live hypotheses, and one cheap experiment tells them apart:

1. **A rank-0 code path.** Rank 0 does things the other three do not (it owns the sampler's staging, the draft
   head's output path, and it is the rank whose `input_batch` staging the scheduler touches first). The fault
   following the *rank* would mean software.
2. **That card.** `c3:00.0` is one specific 0x1002 display device; a page-table or VRAM fault on it would follow
   the *card* no matter which rank sits on it.

**The discriminator: permute the card order and run the same soak.** `GPUS=3,0,1,2` in the launch (a named TEST
launch, `HIP_VISIBLE_DEVICES` follows `GPUS` in `serve/native-flashnext.sh`) moves a different physical card under
rank 0. If the fault moves to the new rank-0 card, it is the rank's code and `AMD_SERIALIZE_KERNEL=3` (or the
per-layer shadow path) is the next step. If the fault stays on `c3:00.0` with rank 0 sitting elsewhere, the hunt
becomes that card: swap it out, or run the soak with `GPUS=1,2,3` on three cards and see whether the fourth is the
variable. Either way the answer costs one launch and about five minutes of soak, because the shape now reproduces
in minutes rather than days.

## 15. The card-order discriminator: the fault follows the **card**, not the rank

`GPUS=3,0,1,2` (a named TEST launch, `~/logs/TEST-gpuswap-1115-0950.log`, `R9K_GDN_DEBUG=1` layer 0, soak
`--pop 16 --max-load 30 --period 1.0` from 09:51:13). The recipe passes `GPUS` straight into
`HIP_VISIBLE_DEVICES`, so this puts a different physical card under rank 0.

**Ground truth for "which card is under which rank", taken while the engine was alive** — saved to
`~/logs/gpuswap-rankcard-1115.txt`, and now a tool (`tools/gpu-rank-map.py`):

```
KFD enumeration (what HIP_VISIBLE_DEVICES indexes): idx0 c3:00.0  idx1 c6:00.0  idx2 03:00.0  idx3 06:00.0
engine launched with HIP_VISIBLE_DEVICES=3,0,1,2
  Worker_TP0  pid 591141  holds 28.8 GiB on 06:00.0
  Worker_TP1  pid 591142  holds 28.8 GiB on c3:00.0     <-- the old fault card, now under rank 1
  Worker_TP2  pid 591143  holds 28.8 GiB on c6:00.0
  Worker_TP3  pid 591144  holds 28.8 GiB on 03:00.0
```

It reads `/sys/class/kfd/kfd/proc/<pid>/vram_<gpu_id>` (KFD's per-process, per-GPU record: the owning rank shows
~28.8 GiB, the others a few MB of scratch) and maps `gpu_id` → PCI through the topology nodes' `location_id`
(`49920 = 0xc300`). `/proc/<pid>/fd` is **not** a way to do this: every rank holds all four `renderD*` open for
P2P, so the fd list identifies nothing — tried, and it printed all four for all four.

**The answer came back in 45 seconds — the fastest hit of all six:**

```
Nov 15 09:52:00 kernel: amdgpu 0000:c3:00.0: [gfxhub] page fault ... pid 591142
(EngineCore pid=591055) ERROR 09:52:02 Worker proc VllmWorker-1 died unexpectedly
```

`pid 591142` = `Worker_TP1` = the rank sitting on `c3:00.0` under the new order. Same read shape
(`PERMISSION_FAULTS: 0x3`, `MAPPING_ERROR: 0x0`, `RW: 0x0`, `SQC (data)`), 0 `BAD`, and the four ranks printed
identical stage counts (63 `fallback T=` / 63 `core ok` / 63 `norm ok` each) — no rank ahead of the others, which
is what a software race would usually leave behind.

So under a configuration where rank 0 and card `c3:00.0` are **different things**, the fault chose the card.
Rank-local code (the sampler staging on rank 0, the draft head's output path, "TP0 does X first") is ruled out as
the carrier. Six faults, one card.

What this does **not** separate, and it cannot be separated by any config change:

- **the silicon vs the slot.** A permutation moves ranks between cards; it cannot move a card to another slot or
  another root complex. `c3:00.0` sits on its own chain (`c3 → c2/c1 → root`) — so "this GPU" and "this slot/chain"
  are still one hypothesis. Telling them apart means moving the card (or swapping two of them) by hand.
- **whether the link is implicated.** It probably is not: `c3:00.0` and its functions carry **zero** AER counters
  (`TOTAL_ERR_COR 0`, no BadTLP/BadDLLP/Timeout). The noisy device on this box is `0000:04:00.0` — BadTLP 579,
  BadDLLP 3, Timeout 20 — which is a downstream port on the **`06:00.0`** chain, a different card and a different
  failure mode. Recording both here so the two never get merged into one story. (Note the near-identical device
  names `04:00.0` and `c4:00.0`: they are different ports on different chains.)

Honest limits: one hit under the swapped configuration (at 45 s, so it is cheap to repeat); the hit rate is not a
trend across these runs (45 s, 250 s, 431 s, 536 s under different instrumentation and topologies — whatever the
distribution is, it is minutes-wide and it is not the interesting part).

Where that leaves the hunt, in order of what one launch buys:

1. **Run without that card**: `GPUS=1,2,3` with `TP=3` (the recipe's own knobs). If the shape keeps hitting in
   minutes with `c3:00.0` in the group and stays clean for two arms without it, that is both a mitigation the box
   can serve on today and a seventh confirmation. Needs a boot check first — three cards means ~33 % more weights
   per card, and whether the KV pool survives that is a measurement, not a guess.
2. **Move the card** (or swap two) and re-run the same soak: separates silicon from slot. Hands and downtime.
3. **`AMD_SERIALIZE_KERNEL=3`** — now less interesting than it was: naming the faulting kernel of a card-local
   page fault tells us which kernel tripped over the card, not why the card trips.

## 16. "Run without that card" at TP3 is closed: the checkpoint will not shard three ways, and it fails before any memory question is asked

§15 ranked the cheapest mitigation as `GPUS=1,2,3 TP=3` — four-card TP with the faulting card left out, which
would have been both a dodge the box could serve on today and a seventh confirmation. Tried (named TEST launch,
`GPUS=1,2,3 TP=3 NSEQ=32 NBT=8192 R9K_GDN_DEBUG=1`, `~/logs/TEST-3card-1115-1015.log`; the `NSEQ`/`NBT` were
pinned so the soak would drive the same step shape as the arms that hit at 250-536 s, rather than the recipe's
unmeasured TP2/TP3 defaults of 8/4096):

```
(Worker_TP2) ERROR [multiproc_executor.py:953] WorkerProc failed to start.
  ... vllm/model_executor/models/... initialize_model ...
  File "vllm/distributed/utils.py", line 61, in divide
    ensure_divisibility(numerator, denominator)
AssertionError: 16 is not divisible by 3
```

Same assert on all three workers. The 16 is the GDN head count: this checkpoint declares
`linear_num_key_heads = 16`, `linear_num_value_heads = 48`, `num_attention_heads = 24`,
`num_key_value_heads = 2`, `vision_config.num_heads = 16` — so **TP must divide 16**: 1, 2, 4, 8. TP3 is not a
configuration this model can take.

What that also costs is the *other* half of the experiment. The point of a TP3 attempt was partly to learn what
the KV pool looks like when weights are ~33 % heavier per card; the assert fires inside `initialize_model`,
**before vLLM profiles memory at all**, so no capacity number is available from a TP3 attempt at any cost. (The
recipe lowered `gpu-memory-utilization` to 0.94 on its own for non-TP4 shapes and still took the failure —
evidence it is a sharding wall, not a memory one.)

So the remaining ways to keep serving while the card question stay open are:

1. **TP2 on two healthy cards** (`GPUS=1,2 TP=2`): shards fine arithmetically, but capacity halves, and at the
   recipe's TP2 defaults (`NSEQ=8`) the running-sequence counts that every fault arrived at (23-31) may never be
   reached — a weaker test *and* a much worse box. Not the cheap option it looks like.
2. **Dodge the trigger rather than the card, at TP4.** `R9K_GDN_PREFILL_MAX=1024` keeps mixed short-prefill steps
   off the stock `_forward_core` — the path §10-§12 localised the fault to — and is still untested. §15 says the
   card is implicated; it does not say software cannot avoid tripping it. A trigger-side mitigation is a
   mitigation, not a fix, and has to be reported that way.
3. **Replace or move the card** (§15's silicon-vs-slot question, plus a real fix).

Recipe note added next to the TP defaults, so nobody spends another boot cycle rediscovering the wall.

## 17. The trigger-side dodge, measured: `R9K_GDN_PREFILL_MAX=1024` turned four instrumented hits into three clean arms, and costs ~45 ms of TTFT

The knob decides, per step, whether the GDN core is ours or stock's `_forward_core` — the call `gdn.py:378`
`_prefill_ok` makes from the step's prefill-token count (and the absence of plain decode rows). It had always been
256, by code default. The recipe now **names** it (`: ${R9K_GDN_PREFILL_MAX:=256}`) so the live value is visible,
held by `tools/recipe-check.py`, and overridable for a test launch; the engine says which value it loaded in its
own log line (`r9700: GDN short-prefill core on r9k_gdn_seq (steps with <= N prefill tokens)`), which matters
because a process's `environ` cannot be read from outside this box (`yama ptrace_scope=1`).

Same soak as every other arm (`--pop 16 --max-load 30 --period 1.0`):

| arm | `PREFILL_MAX` | instrumentation | outcome |
| --- | --- | --- | --- |
| baseline (many) | 256 | layer 0 | hit at 431 s, 250 s; 45 s with the card order permuted |
| baseline | 256 | every layer | hit at 536 s |
| 1 | 1024 | layer 0 | **clean 20 min** (2 329 requests, 0 failed, 0 faults) |
| 2 | 1024 | layer 0 | **clean 20 min** |
| 3 | 1024 | none | **clean 20 min** |
| control | 256 | none | **clean 20 min** ← read this row |

The routing claim is proven by the counters, not assumed: in arm 1 layer 0 logged 36→64 `fallback` steps over the
arm (against 348 in the matched 256 arm), **and every one of those carried more than 1 024 prefill tokens**
(`prefills=5 … act=2047`, `prefills=4 … act=1648`, `prefills=15 … act=5913`). So the knob is doing exactly what it
says: short-and-mixed steps stop entering stock's core; only genuinely large prefill steps still do.

**The nuance the table forces:** the uninstrumented 256 control also ran 20 clean minutes. So today's
"minutes-wide" hit rate is partly associated with the diagnosis itself — the host syncs pace the step, and this
fault lives in the overlap of launches, so the probe perturbs the thing it measures. That weakens arm 3 as
evidence for the knob, and it sets a different, honest floor: **uninstrumented, this soak does not reproduce the
fault within 20 minutes**, which is the same reason the field took a day. What holds up is the matched comparison:
under layer-0 diagnosis, 256 hit four times in 45–536 s and 1024 came through two 20-minute arms.

What the knob costs, from `tuning/prefill_ttft.py` (every probe `max_tokens=1`, and the 1300-token row is stock's
path under both values — it moved 0.2 %, so the two runs are comparable):

| prompt | conc 1 at 256 | conc 1 at 1024 | conc 8 at 256 | conc 8 at 1024 |
| --- | --- | --- | --- | --- |
| 208 tok | 99.1 ms | 91.8 ms | 316.8 ms | 318.7 ms |
| 610 tok | 161.0 ms | 157.7 ms | 593.7 ms | 591.2 ms |
| 900 tok | **160.7 ms** | **205.7 ms** | **774.6 ms** | **822.4 ms** |
| 1300 tok (control) | 194.3 ms | 195.1 ms | 947.0 ms | 948.7 ms |

+45 ms at conc 1 and +48 ms at conc 8 for a 900-token prefill — against the header's warning that the
token-by-token core costs ~1.8 µs per token per layer (900 × 36 × 1.8 µs ≈ 58 ms), which is the same order, so the
model and the measurement agree. At 610 tokens our path is already level with or slightly ahead of stock's, which
puts this box's crossover nearer 600–700 tokens of prefill than the ~240 quoted when the knob was introduced (that
figure is the GPU cost of the step, not TTFT). Decode is untouched: the soak's per-request rate was 46.8 and
46.9 tok/s with the knob at 1024, instrumented and not.

Side finding worth keeping: **layer-0 diagnosis is almost free** — every probe row matched within 1 % between the
instrumented and uninstrumented 1024 launches. So the guard and stage syncs can stay armed on a field launch,
which is exactly where a future hit needs to be read from.

Decision left open on purpose: production stays at 256. Raising it is a mitigation that would make this fault stop
happening — and thereby stop being observable, which is the same trade already made about automatic restart. If
the box is moved to 1024, the follow-up that keeps the diagnosis alive is a soak whose long arrivals exceed 1 024
tokens, so stock's core still runs regularly: if the fault comes back there, the dodge is not protective and the
trigger hypothesis is wrong; if it does not, the trigger is confirmed as stock's core on that card.

### Decided, 12:39 the same day

Production moved to `R9K_GDN_PREFILL_MAX=1024` (recipe default, so `tools/recipe-check.py` holds it and the engine
prints `steps with <= 1024 prefill tokens` as proof it crossed). Uptime was chosen over the deciding arm: the box
has been down ~2 h across this hunt, and the mitigation is measured as cheap (+45 ms of TTFT at 900 tokens, decode
unaffected) while the diagnosis has no cheaper next step than moving a card by hand.

What that trades, stated plainly: with 1024 armed the fault should stop arriving, so the fault also stops being
observable -- the same objection that settled automatic restart, and accepted here because the mitigation is
priced and the remaining unknown (does removing the trigger remove the fault?) is a question about a card that has
already been identified as the constant across six faults. If a fault does return, it is informative in a new way:
the surviving stock-core steps are only prefills over 1 024 tokens (`prefills=5 … act=2047`, `prefills=15 …
act=5913` in the 1024 arms), so a hit under 1024 means the trigger is broader than the stock core and §10-§12's
localisation was too narrow.

## 18. What the Monitor now measures because this hunt could not answer it at the moment it mattered

Three of this hunt's answers arrived late because the page in front of the operator did not carry the fact: which
**physical card** each rank sits on (§15 needed a permuted launch, a live `tools/gpu-rank-map.py` and a kernel
journal read by hand), whether the driver had **page-faulted at all** since boot (eight events, eleven driver
lines — and the engine's own log never says which card), and whether the step shape that faults is even being
**asked for** any more (§17's
whole interpretability problem: "no faults since 1024" is worth nothing unless someone knows that big prefills are
still arriving). So `serve/webui.py` now reads all three, and the Monitor has a row for them.

* **Which rank owns which card** — `rank_holds()`, from KFD's per-process records: `topology/nodes/*/location_id`
  gives the PCI function, `proc/<pid>/vram_<gpu_id>` gives the holding, and the owning rank is the one showing
  ~28.8 GiB where the others show scratch. Not `/proc/<pid>/fd`: every rank keeps all four `renderD*` open for
  P2P, so the fd list identifies nothing (measured §15). The count rides on the card, so the question is asked
  where it is asked: `rank TP1 holds 28.8 GiB · pid 591142`.
* **Driver page faults, this boot** — `kernel_faults()`, counting `[gfxhub] page fault` lines out of
  `journalctl -k --boot -g …`, per PCI function, with the following `Process … pid N` line paired to its fault.
  The number was ambiguous the moment it landed on the page: **11 lines across 8 events** since this boot, three of
  them in the same second on 11-06 07:45:03. So the headline is events (distinct stamp+card) and the per-card slot
  carries lines — "11 faults" overstates how many times the machine died, and "8 faults" hides that the driver
  logged more than one fault for some of them, which is itself a fact about the fault. Two honesty rules, both
  earned today: journalctl's `-A` is
  `--after-boot` and not "lines after the match", so a mistyped flag prints a usage line and **nothing** on
  stdout — which is how this read first reported "0 faults" on a box that had had eight that boot; and an unreadable journal
  reports `unknown`, never zero, because "no faults" and "cannot see the kernel log" are different claims and only
  one of them is evidence about a card.
* **Engine state, including `wedged`** — `engine_state()`: health plus how many ranks KFD still reports as
  holding memory. §11's aftermath was a state neither the record nor the page could name: /health silent, three
  ranks still holding ~32 GiB each, 26 minutes of "healthy". The note says what to do about it (`STOP=1 bash
  serve/<recipe>`), and a launch that has never answered is `loading`, not dead.
* **Exposure: prompts over the routing bound** — the `/metrics` prompt-size histogram is no longer thrown away;
  buckets are kept as `family;le=<bound>` keys (never folded into the family, which is what makes means lie, and
  the `+Inf` bucket dropped because it is the count twice). The row prints the share of finished requests whose
  prompt clears the nearest published bucket, and the bound itself is printed rather than implied: the build
  publishes 1 000, the recipe's knob is 1 024, and a page that conflates them is reporting precision it does not
  have. This is the card that keeps the other three interpretable — a fault-free stretch with zero big prefills
  is not evidence about §17's dodge.

`tests/render-monitor.mjs` earned two more rules from this and both are worth keeping: the row is **hidden** —
not left as four em-dash cards — when the sidecar is too old to have made the reads (an em-dash row would read as
"no faults here"), and the timestamp/pid live in the `title` because the slot's 24-character budget is what keeps
the row from re-wrapping; a raw ISO timestamp in a slot also trips the harness's bare-negative rule, which is the
harness telling the truth about how a reader would misread it.

First seen on the page at 13:20 the same day, after a restart: the Monitor runs **inside** the serve
(`serve/plugin`, mode `merged`), so these fields can only arrive with a launch. Before that restart the row correctly
hid itself against the pre-change sidecar — the hiding rule proven by the thing it was written for, not by a test.
What the first look showed: `Answering · 4 rank(s) own the GPUs`, the four ranks on the four cards in the order the
default enumeration gives (TP0 on c3:00.0 — the card under every fault), and `faults 11 (c3:00.0 ×11)`. The 11 is
lines, not events: the events split shipped minutes after this launch started, so it arrives with the next one —
which is the same rule as everything else here, stated because it would otherwise read as a discrepancy.
