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
| 21:04:09 | EngineCore `mq.dequeue(timeout=...)` gives up (`core.py:1512`), `dump_input.py:79/81` writes the scheduler output above, `async_llm.py:906` `EngineDeadError`, **8,412** `serving.py:929` EngineDeadError lines — every request in flight, none of them journalled; `[shutdown] MPClient: start timeout=0s` |
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

## 7. What would settle the fault, in order of cost

1. `EAGER=1` plus a 610-token prefill arriving into a batch of long spec-decoding requests. If the fault goes
   away under eager, the piecewise/full cudagraph capture is implicated, the same discriminator the router note
   proposed first.
2. A replay loop: N long requests (each ≥ 200k tokens) to push the pool to 0.998, then a steady arrival of
   ~500–700-token prompts. That is the shape the dump shows, and `bench/` already has both halves
   (`sanity_stress`'s long prefills, `bench/quality.py`'s concurrency).
3. The step shape as a unit check: `tests/test_gdn_prefill_r9k.py` covers our conv; the stock Triton conv is
   covered by nothing at this shape (mixed batch, 610-token prefill, 30 spec rows, `align` mamba mode, fp8 KV).
   Feeding that exact `SchedulerOutput` through `causal_conv1d_fn` with `metadata=md` is the fault on demand.
4. Whether the plugin should take the conv on steps above 256 prefill tokens anyway (it already does the
   spec-row conv in `_spec_rows`): `R9K_GDN_PREFILL_MAX` exists, so a launch at, say, 1024 is a config change
   rather than a code change — but that would be a *measured* change: PREFILL_MAX=256 is where the one-page
   conv work was validated, and `tools/recipe-check.py` will hold the live engine to whatever it becomes.
5. `AMD_SERIALIZE_KERNEL=3` if (1)-(3) fail to reproduce: it names the faulting kernel instead of the first
   innocent reader, at the cost of the whole protocol's timing.

## 8. Box state while this is written

Relaunched at 23:02:48 from the stored configuration (`./setup.sh --config 262k`,
`~/logs/setup-230248.log`): up after 190 s, `KV 902,153 tokens / Maximum concurrency 3.44x at 262,144`,
`health 200`, and `tools/recipe-check.py` reports every measured knob, every `R9K_`/`R9700_` name and every
launch flag of the live serve matching this tree. The dead launch booted at the same 902,153 / 3.44x
(`setup-005730.log`, 01:00:09), so nothing drifted between the two: the fingerprint is the recipe's, not the
incident's. The watcher is armed on the new launch; the next run of `./setup.sh` will answer "is it still up"
from `.r9700-state/launch.json` rather than from an operator's memory.
