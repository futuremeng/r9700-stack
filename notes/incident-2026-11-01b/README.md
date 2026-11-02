# notes/incident-2026-11-01b -- two hard resets 32 minutes apart: a PCIe `BadTLP` storm on the card chains, ending in a firmware "data fabric sync flood"

Box `jiulu-9700`, kernel `7.0.0-42-generic`, amdgpu `7.1.9.31600000`, 4x Radeon AI PRO R9700 (`1002:7551`, rev c0)
behind their own onboard upstream/downstream switch pairs, 93.8 GiB RAM / 8.0 GiB swap. Bare metal (no vfio).
Kernel cmdline carries only `amdgpu.ppfeaturemask=0xffffffff` (overdrive on; `notes/release-v0.2.4.md` records the
undervolt that goes with it).

Captured 2026-11-01 23:40-23:44 UTC from the boot *after* the second crash (boot 0, up 21:03:37). **Nothing on the
box was changed for this incident** -- no sysctl, no modprobe, no BIOS. The bundle is observation only; the service
was brought back with the sanctioned path (`bash bench/relaunch.sh final51`) and is serving.

Two resets, both silent, both attributed by firmware to the same cause:

| boot | lived | died | how |
|---|---|---|---|
| `-2` | 03:27:55 -> 20:30:41 (17 h 03 m) | ~20:30 | no `shutdown` record in wtmp; last journal line is a benign `fwupd-refresh` finish |
| `-1` | 20:36:38 -> 21:02:15 (25 m 37 s) | 21:02:15 | no `shutdown` record; journal ends mid AER dump |

`last -x` still shows three boots as `still running` (03:27, 20:36, 21:03) -- i.e. boots `-2` and `-1` were never
closed by a shutdown. Both were resets.

## What the agent saw

The previous session (`2026-11-01T09-41-40-555Z_01a1967c-...jsonl`) ends with an assistant message, id `f6c6269a`,
at **21:00:25.276 UTC** whose only tool call is `sleep 200` (timeout 300 s) polling `~/logs/relaunch-final49.log`.
**The tool result was never written.** There is no error entry, no abort, no provider failure in the session -- pi
died with the box. The agent's model provider is remote (`http://192.168.1.130:4000/v1`), so pi itself never had a
reason to stop; the box it runs on stopped. `notes/load-oom-svm-wedge.md` rule 4 ("poll with short tool calls, never
`sleep 600`: when the box or the session goes, a long blocking call loses context") is what this is: the 200 s poll
swallowed the 21:00:25 -> 21:02:15 window and the whole context with it.

## The crash that took the session (boot `-1`), in order

```
20:36:38  (boot) x86/amd: Previous system reset reason [0x88000800]: an uncorrected error caused a data fabric
                   sync flood event  /  a software sync flood event occurred
                   BERT: fatal, ProcessorError, IA32/X64 processor error, APIC 0x83, cache error Level 0,
                   Processor Context Corrupt: true, Uncorrected: true     <-- the death of boot -2
20:36:48  pcieport 0000:c0:03.1: [12] Timeout                <-- AER already 10 s into the new boot
20:54:41  amdgpu 0000:06:00.0 / snd_hda_intel 06:00.1 / pcieport 04:00.0 / 05:00.0: [13] NonFatalErr
          pcieport 0000:04:00.0: [ 6] BadTLP                 <-- ~6 min before our launch, no load running
21:00:21  bench/gate.sh: procs=0 inD=0 MemAvail=87.2 GiB SwapFree=8.0 GiB -> gate: PASS
21:01:2x  the four ranks start streaming the 110.6 GiB checkpoint
21:02:01  workqueue: svm_range_restore_work [amdgpu] hogged CPU (4x)
21:02:14  {18..19}[Hardware Error] APEI GHES source 512, severity corrected, fru_text: PcieError,
          device_id 0000:04:00.0, vendor 0x1002 device 0x1478, aer_cor_status 0x40 (BadTLP),
          pcieport 0000:00:03.1: [12] Timeout, aer_layer=Data Link Layer aer_agent=Transmitter ID,
          then a BadTLP flood on 04:00.0 ("aer_ratelimit: 4 callbacks suppressed")
21:02:15  svm_range_restore_work hogged CPU 19 times   ->  journal ends. Reset.
```

The load itself was further along than a first reading of the log tail suggested (progress lines are `^M`
separated, so a `tail` shows the *head* of the progress line): the last recorded state is `23/23 [00:46]`, and the
log never prints `Model loading took`. So the reset landed **as the shard loop ended**, exactly where the HMM/SVM
remapping of the big mappings begins -- `svm_range_restore_work`'s own count climbing 4 -> 5 -> 7 -> 11 -> 19 across
21:02:01-21:02:15 is the driver-side lead-up. See `final49-how-far.txt`, `crash-timeline-boot-1.txt`.

## The firmware says the same thing about both deaths

Both following boots read an identical pair (`mce-firmware-records.txt`):

```
x86/amd: Previous system reset reason [0x88000800]: an uncorrected error caused a data fabric sync flood event
BERT: Error records from previous boot: event severity: fatal; Error 0, type: fatal; fru_text: ProcessorError;
      IA32/X64 processor error; cache error, Level 0, Processor Context Corrupt: true, Uncorrected: true
```

APIC `0x83` for the boot `-2` death, `0x82` for the boot `-1` death -- **different cores**, one per crash; boot 0
additionally logs `mce: CPU 17: Machine Check: 0 Bank 5: aea0000000000108` with `TIME 1793567019` = 21:03:39. This
is the platform-level signature already documented in `notes/bug-mes-invalidate-tlbs.md` and
`notes/mes-timeouts.md` ("an uncorrected error caused a data fabric sync flood event", SMCA/fatal MCE on a core
stalled on a device transaction): on this platform a device-side stall escalates into a fabric sync flood, i.e. a
hang is a reset is an outage. What is *new* here is what precedes it -- see the next section.

## AER counts: per boot, per port, per type (`aer-by-boot-port.txt`)

| boot | window | BadTLP `c4:00.0` | BadTLP `04:00.0` | Timeout `c0:03.1` | Timeout `00:03.1` | Timeout `c4:00.0` | ratelimit notes |
|---|---|---|---|---|---|---|---|
| `-2` | 17 h 03 m | **4042** | 131 | **564** | 8 | 53 | 42 |
| `-1` | 25 m 37 s | 5 | **71** | 1 | 4 | 0 | 1 |
| `0` | 2 h 37 m (two loads) | **53** | 27 | 16 | 2 | 2 | 2 |

Journal counts are **lower bounds** -- the kernel rate-limits AER printing. sysfs (`aer_dev_correctable`, cumulative
for boot 0, read at 23:42) is higher: `c4:00.0 BadTLP=61`, `04:00.0 BadTLP=27`, `c0:03.1 Timeout=16`.

Attribution: `BadTLP` is reported by the **card's own upstream switch port** (`04:00.0` feeds `05:00.0 -> 06:00.0`;
`c4:00.0` feeds `c5:00.0 -> c6:00.0`; `aer_agent=Receiver ID`), and the **CPU root ports** (`00:03.1`, `c0:03.1`)
report `Timeout` -- a request that never got answered. So the malformed TLP is generated on the card side of the
onboard switch and the visible consequence one level up is a dropped completion. Which card leads moves between
boots (`c6:00.0` dominates boot `-2`, `06:00.0` dominates boot `-1`): this is not one dead card, it is the pair of
Gen5 chains under load. Link state is unaffected -- all six bridges/GPUs still train **32.0 GT/s x16 at max**
(`topology.txt`), no retraining, no width drop.

## Correlation with the load path (boot `0`, `aer-timeline-boot0.txt`)

Every AER event in the post-crash boot lands inside a weight load, and the idle box is quiet:

```
23:29:38 gate PASS -> final50 loading -> Model loading took 67.82 s at 23:31:34 -> health 200 at 23:32:0x
   AER: 6 in 23:30, 25 in 23:31
23:33:09 gate PASS -> final51 loading -> Model loading took 70.88 s at 23:35:09 -> health 200 at 23:36:2x
   AER: 52 in 23:34, 16 in 23:35, 1 at 23:36:48
23:36:48 -> 23:44 (idle, serving): 1 event
```

A clean boot with zero loads had zero events (boot 0, 21:03:56 -> 23:29). So: **the trigger is the 4-rank weight
stream / HMM mapping path**, not time, not temperature at idle, not the serving pass itself. That is also the same
moment the load-path work of `notes/load-oom-svm-wedge.md` and the C22/C23 drain-wait of
`notes/native-4xr9700.md` live in.

## What this is NOT (four readings to stop repeating)

1. **Not pi, not the agent, not the model provider.** The session file ends mid tool call because the box reset;
   `dmesg`/journal/wtmp carry the whole story and no OOM kill exists.
2. **Not the memory-overcommit wedge of `notes/load-oom-svm-wedge.md`.** That precondition (previous ranks' ~50 GiB
   still mapped, swap already half gone) was absent: gate PASS with `procs=0 inD=0 MemAvail=87.2 GiB SwapFree=8.0 GiB`
   4 s before the load, on a box 24 min old. The `svm_range_restore_work` workqueue is involved again, but the
   kswapd/notifier cycle is not what killed it -- the kill came 1 s after a PCIe BadTLP flood, not after a hung-task
   block. Note `kernel.hung_task_panic = 0` still (`system-state.txt`), so a wedge would not have self-rebooted;
   the reset here was the firmware's, not ours.
3. **Not MES timeouts.** `grep 'failed to respond to msg'` = **0 in boots -2, -1 and 0**. The
   `INVALIDATE_TLBS`/REMOVE_QUEUE escalation of `notes/bug-mes-invalidate-tlbs.md` is absent on this bare-metal box
   (where `mes_log_enable` is `0`, unlike the `.100` host where it is deliberately `1`), and there is nothing here
   for that flag to paper over. Recorded, not acted on.
4. **`expandable_segments: memory mapping failed with OOM ... (free: 0, total: 34208743424)` is not a fault
   marker** -- I read it as one in the first pass and it is wrong. It appears in 11-20 lines in *completed, healthy*
   loads as well (tail111 17, tail112 17, tail114 15, tail117 11, tail118 16; final50 12, final51 20 -- both
   health 200). It is ROCm's caching allocator failing one 20 MiB expandable-segments map and retrying without it.
   Do not use it as a crash predictor; use AER counts instead.

## Where that leaves it

Ruled in: a data-link-layer fault on both card chains under load (`BadTLP` at the card-side upstream port, root-port
`Timeout` above it), escalating in the fatal case to an uncorrected fabric error and sync flood. Two boots, same
firmware verdict, different cores, different leading chain.

Still unmeasured (the A/B list I did not run, per Brian): whether the rate falls with the chain at Gen4, with
`pcie_aspm` off, with the undervolt/overdrive (`ppfeaturemask=0xffffffff`, 300 W envelope per
`notes/native-4xr9700.md`) relaxed, or per-slot -- and whether either crash is preceded by a countable BadTLP burst
in the minutes before (boot `-1`: yes, 71 in 25 min, burst at -1 s; boot `-2`: **no**, its last AER was 37 min
before it stopped logging).

## Operating rules this incident adds

1. **`bench/gate.sh` is a STOP, not a status probe.** It begins with `STOP=1 bash serve/native-flashnext.sh`. In
   this incident I ran it in the background to "count processes" and it stopped `final50` -- an instance that had
   already finished loading (67.8 s) and answered health 200 twice. Its log's last write, 23:32:31, is that STOP.
   `final51` is the redo. Status is `curl -s -o /dev/null -w '%{http_code}'
   http://127.0.0.1:8080/health`; process counts go through a pattern that cannot match the caller
   (`pgrep -cf 'vll[m]'`), never by invoking the gate.
2. **Short polls** (`notes/load-oom-svm-wedge.md` rule 4) is now confirmed twice, both times at the same place: at
   the end of a weight load. Poll in <=60 s slices and keep the fingerprints recoverable from the log file.
3. **A reset can arrive with no lead-up in the journal at all** (boot `-2`), so post-mortem work has to start from
   firmware records (`Previous system reset reason`, BERT) and sysfs AER counters, which means reading them *while*
   the box is up, after each load, not only after a crash.

## Files in this bundle

| file | what it is |
|---|---|
| `README.md` | this narrative |
| `boots.txt` | `journalctl --list-boots` + `last -x reboot shutdown` (the three `still running` boot rows) |
| `system-state.txt` | kernel / amdgpu version, module parameters (`mes_log_enable:0`, `gpu_recovery:-1`), taint, cmdline, hung-task and panic sysctls, memory now, per-port sysfs AER counters |
| `topology.txt` | `lspci -d 1002:` (four cards, each behind its own upstream/downstream pair), drm card -> PCI function, `rocm-smi --showbusid` |
| `aer-by-boot-port.txt` | journal AER counts per boot / port / type + ratelimit notes, sysfs per-port totals, and the link speed/width of both chains (Gen5 x16 at max) |
| `aer-timeline-boot0.txt` | the post-crash boot's AER events bucketed against the two load windows; MES-timeout count per boot (0, 0, 0) |
| `crash-timeline-boot-1.txt` | the 21:00:21 -> 21:02:15 sequence: gate, load progress, GHES/AER dump lines, `svm_range_restore_work` counter, and the pi session's last entry |
| `crash-timeline-boot-2.txt` | the 20:30:41 death: last journal lines (benign), last AER 37 min earlier, the boot's `svm_range_restore_work` history up to 32771 hogged samples |
| `mce-firmware-records.txt` | `Previous system reset reason` + BERT fatal records read at boots `-1` and `0` (APIC 0x83 / 0x82) |
| `final49-how-far.txt` | the crashed load's real progress: `23/23 [00:46]`, no `Model loading took` line |
| `expandable-segments-not-a-marker.txt` | warning counts across 11 relaunch logs, healthy and crashed |
| `service-restored-final51.txt` | the instance left serving: load 70.88 s, KV 484,642 tokens, health 200, gate PASS times, AER counters at capture |
