# Incident 2026-11-01: load-path memory overcommit -> amdgpu SVM/GPU-fence deadlock -> box unrecoverable

Box: `jiulu-9700`, kernel `7.0.0-42-generic #42~24.04.1-Ubuntu`, amdgpu `7.1.9.31600000` (tainted `G S OE`),
93.8 GiB RAM, 8.0 GiB swap, 4 x 32 GiB GPU. Uptime at capture 14:07.
Trigger: `bench/prefetch_ab.sh` (a prefetch A/B, two launches per variant, launched back to back).
Status: **reboot required** -- nothing userspace can do is left. Evidence bundle: `notes/incident-2026-11-01/`.

## The deadlock (this is the root cause, and it is in the kernel, not in vLLM)

The mutex holder, from the hung-task dump (`kern-blocked-tasks-0140.txt`, `owner-trace-kworker185.txt`):

```
INFO: task kworker/185:3:64157 blocked for more than 122 seconds.
Workqueue: events_freezable svm_range_restore_work [amdgpu]
  dma_fence_default_wait+0x215
  dma_fence_wait_timeout+0x12d
  svm_range_validate_and_map+0x1086 [amdgpu]
  svm_range_restore_work+0xe5 [amdgpu]
```

Everyone else is behind it -- `kswapd0:1423`, `sshd:3290`, `bash:94761`, `kworker/106:1`, `kworker/57:0`,
`kworker/u793:4/6/13`, each reported as `blocked on a mutex likely owned by task kworker/185:3:64157`:

```
INFO: task kswapd0:1423 blocked for more than 122 seconds.
  __mutex_lock / __mutex_lock_slowpath
  svm_range_cpu_invalidate_pagetables+0x7b [amdgpu]
  __mmu_notifier_invalidate_range_start+0x19a
  ... try_to_unmap_one / kswapd
```

Cycle:

1. `svm_range_restore_work` runs with the amdgpu per-process SVM lock held and blocks in
   `svm_range_validate_and_map` waiting on a **GPU fence** (`dma_fence_wait_timeout`) -- validating/mapping HMM
   ranges needs host pages populated.
2. Page reclaim for that same address space must call back into the driver
   (`__mmu_notifier_invalidate_range_start` -> `svm_range_cpu_invalidate_pagetables`), which needs the same lock.
3. kswapd therefore stops. With kswapd stopped, swap/anon reclaim stops, `MemFree` goes to ~1.1 GiB with
   `Committed_AS` at **122.6 GiB against 93.8 GiB of RAM**, and the GPU/HMM population that the fence is waiting
   for can never complete -> the fence never signals -> the lock is never released.
4. Every process that touches its own mm, or allocates, queues behind the notifier: `sshd`, the driver `bash`,
   a dozen kworkers. `load average 396.74` (`wedged-pids.txt`), 2331 tasks.
5. `kill -9` cannot break it: exit needs `exit_mm` -> the same notifier. The four workers are parked there
   (`wchan=exit_mm`, RSS 10.7 / 12.3 / 18.8 / 13.1 GiB still mapped), the APIServer went `Z` because its
   reaper (the driver bash, 94761) is itself in the SVM path. A GPU reset does not release a kernel mutex.

## State captured at 03:05:44 (`wedged-pids.txt`)

```
1423 kswapd0  D
3290 sshd: jiulu@pts/0            D     <- the login session that hosted the agent
94761 bash bench/prefetch_ab.sh   D  wchan=svm_range_cpu_invalidate_pagetables
97500 APIServer                   Z
97851-97854 VLLM::Worker_TP       D  wchan=exit_mm  rss 10.7/12.3/18.8/13.1 GiB
loadavg 396.74 ; MemFree ~1.1 GiB ; MemAvailable 47.7 GiB ; SwapFree 2.03 GiB of 8.0 ; Committed_AS 122.6 GiB
```

VRAM still held 19.4 GiB per GPU; the load never passed `1/23` shards (frozen at 01:36:48,
`frozen-load-tail.txt`).

## How the agent session actually went away (correcting the first reading)

* `who`: `pts/0` since 10-31 12:50 and `pts/1` since 12:57 are **still listed**; `auth.log` has **zero**
  `session closed for user jiulu` lines (`agent-death-evidence.txt`) -- the old connections were never closed
  cleanly. A new `sshd[98440]` login from 192.168.1.200 (session 96) at **01:45:05**, agent restarted 01:45:46.
* The per-connection process of that old session, `sshd: jiulu@pts/0` (3290), is **in D state in the dump above**,
  and it appears in the 01:40:49 hung-task block as blocked behind the amdgpu mutex.
* Conclusion: the agent's terminal was carried by a connection whose kernel worker wedged; the agent process
  stopped writing to its session file (`2026-10-31T15-09-19...jsonl`, last entry 01:32:50.972, a `sleep 600`
  tool call with no tool result -- nothing after it) and the client was reconnected at 01:45. The agent was taken
  down with the wedged session, not by a clean crash: there is no apport/node crash report (`/var/crash` has only
  an old rocm_smi one).
* **Not proven, and now flagged as unsupported:** my first reading said the OOM killer killed the agent. Two
  corrections: (a) `grep -iE "oom|out of memory|killed process|invoked oom|Mem-Info" /var/log/kern.log
  /var/log/syslog` returns **no matches at all** -- no kernel OOM message was logged this boot, while the
  hung-task INFO lines are there, so the kernel OOM path is not evidenced at 01:32-01:45 (`oom_kill 8` in
  `/proc/vmstat` is a cumulative since-boot counter, unattributable); (b) the two `Killed` lines in
  `driver-log.txt` are the driver's **own `STOP` step** SIGKILLing the run-1 and run-2 wrappers (`setsid` wrappers
  exec into `vllm serve`, which is why bash's job table still names them) -- they are expected teardown, not
  OOM evidence. Either way the machine was on `Committed_AS 122.6 GiB / 93.8 GiB RAM`, which is the mistake to
  fix; which signal finished off the agent is secondary and recorded here as unknown.

  Updated with `cgroup-oom-evidence.txt`, which sharpens but does not settle it:

  ```
  user.slice/user-1000.slice/memory.events : oom 0  max 0  high 0  oom_kill 8  sock_throttled 51123
  memory.max                               : max (no limit)
  session-96.scope (the new login)         : oom_kill 0
  ```

  So OOM kills did happen, 8 of them, inside our own user slice, and they were **global** OOM kills (no cgroup
  limit exists, `oom`/`max`/`high` all zero) -- the in-kernel killer, not systemd-oomd (not running) and not
  earlyoom (not running). The old session's scope is already gone, so its own counter cannot be read, and nothing
  names the 8 victims. The agent may well be one of them; it cannot be established post-hoc. Two side findings
  worth keeping: `sock_throttled 51123` says socket memory was being throttled through reclaim for a long time,
  which is a second, independent reason the SSH session carrying the agent froze; and the absence of OOM lines in
  kern.log/syslog is most likely journald/rsyslog **rate limiting** swallowing the multi-thousand-line `Mem-Info`
  dump -- if we want this attributable next time, raise `RateLimitIntervalSec`/`RateLimitBurst` and set
  `kernel.hung_task_panic=1` so the box reboots itself at minute 2 instead of sitting in `D` for 90 minutes.

## The overcommit mistake

`STOP=1 bash serve/native-flashnext.sh` returns as soon as the signal is sent -- it does **not** verify that the
previous 4-rank server is gone and its memory unmapped. So run 3 started streaming a 110.6 GiB checkpoint with
~54 GiB of the previous attempt's worker RSS still mapped, and vLLM's own loader warning says the situation out
loud at every rank:

```
Filesystem type for checkpoints: EXT4. Checkpoint size: 110.60 GiB. Available RAM: 40.74 GiB.
Auto-prefetch is disabled because the filesystem (EXT4) is not a recognized network FS (NFS/Lustre)
and the checkpoint size (110.60 GiB) exceeds 90% of available RAM (40.74 GiB).
```

Only two launches ever produced data, same config, page cache the only difference:
`load_s=208.4 / experts_place_s=137.6` (cold) vs `load_s=103.2 / experts_place_s=34.5` (warm) -- 2.02x
(`driver-results.txt`). The prefetch arms were identical code anyway: `weight_utils.py:1115`
`should_prefetch = safetensors_load_strategy == "prefetch"`; see `bench/prefetch_ab.sh` header.

## Rules going forward (encoded in `bench/coldstart_ab.sh`)

1. `wait_down()` before **every** launch: no `vllm serve|VLLM::` process alive (180 s max, else abort and treat
   the box as wedged) **and** `MemAvailable >= 55 GiB`. Never relax the gate.
2. Watch `SwapFree`, not just `MemAvailable`: the deadlock becomes reachable once anon reclaim starts moving
   pages of GPU-attached processes. `Committed_AS > MemTotal` is the red flag to stop on.
3. One 4-rank load at a time, and "STOP" means verified teardown, not "signal sent".
4. Poll with short tool calls, never `sleep 600`: when the box or the session goes, a long blocking call loses
   context; short calls cost one call.
5. Numbers taken under memory pressure are junk anyway -- reboot before measuring, not after.

## Worth reporting upstream (ROCm / amdgpu)

`svm_range_restore_work` holds the process SVM lock across `dma_fence_wait_timeout`, while the mmu-notifier
(`svm_range_cpu_invalidate_pagetables`) needs that lock from inside page reclaim: under host-RAM exhaustion the
fence cannot complete and reclaim deadlocks against it. Repro shape: 4-way TP vLLM, 110 GiB checkpoint, 94 GiB
RAM, relaunch before the previous server's pages are unmapped -> `Committed_AS` 1.3x RAM. Kernel 7.0.0-42-generic
+ amdgpu 7.1.9.31600000, `hung_task_timeout_secs=120`, `hung_task_panic=0` (setting panic=1 would at least turn
the next occurrence into an automatic reboot instead of a 90-minute zombie box). Traces in this directory.
