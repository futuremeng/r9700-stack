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

1. ~~Shape/geometry~~ — done, §7: in range, and the step replays clean.
2. **Instrument the fallback path** — *done, 2026-11-14* (`gdn.py:_bounds_problems` / `_fallback_check`, and
   12 checks in `tests/test_conv1d_mixed_step.py` arm C). With `R9K_GDN_DEBUG=1` the fallback now bounds, before
   stock's kernels run, the two walks that kernel can make: every conv-state line index in
   `non_spec_state_indices_tensor` / `spec_state_indices_tensor` against `conv_state.size(0)`, and the non-spec
   token span against `num_actual_tokens`. It costs host syncs, so it stays behind the existing `DEBUG` gate
   (default off) and it runs on **layer 0 only** — the block table and the token counts are per-step, so layer 0
   sees what all 48 layers of that step will see. Off, it is one `if` per layer. This is the check that would
   have named a bad line index in the log instead of leaving the device to name it: our own paths already had
   that assertion in `_debug_check` (`idx max >= state blocks`); the path this fault ran never did.
   A detail the check forced out: the conv kernel skips a slot whose index equals `NULL_BLOCK_ID = 0`, so it
   cannot tell padding from a real block 0 — harmless, because the block pool reserves block 0 as its null
   block (`kv_cache_utils.py:1606`), and now said in the code rather than remembered.
3. **Reproduce with the pool, not the shape**: ~12 requests at ≥ 200k tokens to drive `kv_cache_usage` to
   ~0.998, then a steady arrival of 500–700-token prompts (the dumped step). Against a *spare* launch, not the
   box behind traffic — a hit costs the 10-minute NCCL teardown.
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
