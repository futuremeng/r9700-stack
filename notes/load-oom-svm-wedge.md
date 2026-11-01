# Load-path OOM -> amdgpu SVM wedge: the box became unrecoverable (2026-11-01 01:28-02:55)

A prefetch A/B (`bench/prefetch_ab.sh`, two launches per variant, back to back) killed the agent session and
then wedged the machine. Root cause is not vLLM and not the agent: it is a lock inversion between host page
reclaim and the amdgpu KFD SVM/HMM mmu-notifier, which only appears once RAM is exhausted. Kernel 7.0.0-42-generic,
amdgpu 7.1.9.31600000, 93.8 GiB RAM, 8 GiB swap, 4 x RX (9700-class) 32 GiB.

## Timeline

| time | what |
|---|---|
| 01:28:28 | driver started: 3 variants x 2 launches; each launch preceded by `STOP=1 bash serve/native-flashnext.sh` |
| 01:32:54 | variant "default" run 1: `load_s=208.4`, experts place `137.6 s` |
| 01:35:39 | variant "default" run 2 (same config, warm page cache): `load_s=103.2`, experts place `34.5 s` |
| 01:35:46 | driver reports `94900 Killed`, then `96330 Killed` (SIGKILL of the two launcher wrappers) |
| 01:36:48 | run 3 (threads=1 variant) load frozen at `1/23` shards -- never progressed again |
| 01:32:50 | agent's last recorded action is a `sleep 600` tool call with **no toolResult**: session file ends |
| 01:40:49 | kernel: `INFO: task kswapd0:1423 blocked for more than 122 seconds` + 9 more blocked tasks |
| 01:45:46 | agent restarts in a new session (the old process is gone, no error recorded anywhere) |
| 02:49-02:55 | still frozen: 1/23 shards, 19.4 GiB VRAM per GPU held, `kswapd0` still in `D`, load average 62 |

## Why the agent died

`grep oom_kill /proc/vmstat` -> **`oom_kill 8`** for this boot. The driver log shows the two wrappers SIGKILLed,
and the agent session file stops mid-tool-call with no error block -- that is the signature of an OOM-kill, not a
crash (a JS exception leaves a toolResult/error entry). At the time of the kill: `MemFree` ~1.0 GiB of 93.8,
`SwapFree` 1.5 GiB of 8.0, buff/cache 47 GiB (the checkpoint's pages). The agent is a normal-size Node heap and
sat right at the edge.

## Why the box is stuck instead of just slow

kern.log 01:40:49, the block that matters:

```
INFO: task kswapd0:1423 blocked for more than 122 seconds.
  __schedule / schedule / __mutex_lock / __mutex_lock_slowpath
  svm_range_cpu_invalidate_pagetables+0x7b [amdgpu]
  __mmu_notifier_invalidate_range_start+0x19a
  ... kswapd / try_to_unmap_one
INFO: task kswapd0:1423 is blocked on a mutex likely owned by task kworker/185:3:64157.
... amdgpu_hmm_range_get_pages+0xf3/0x1c0 [amdgpu]     (in another task's trace)
```

The cycle: a vLLM worker holds the amdgpu process mutex while validating HMM ranges
(`amdgpu_hmm_range_get_pages`); that path needs pages, so it waits on reclaim; kswapd, unmapping a page of that
same process, must call back into the driver (`svm_range_cpu_invalidate_pagetables`) and blocks on the mutex the
worker holds. Reclaim stops -> RAM pressure rises -> OOM killer fires -> the killed workers' `exit_mm` needs the
same notifier, so they join the queue in `D`. Everything else piles up behind it: `sshd:3290`,
`bash:94761` (the driver), a dozen `kworker`s.

Consequences: SIGKILL cannot clear it. After killing APIServer/EngineCore/workers by hand the state was
`97500 Z`, `97851-97854 D (wchan exit_mm)`, `94761 D (wchan svm_range_cpu_invalidate_pagetables)`, VRAM still
19.4 GiB/GPU, `kswapd0` still `D`. A GPU reset does not release a kernel mutex; the only way out is a reboot.

## The load arithmetic that made it inevitable

vLLM says it out loud at every rank on this stack and the number was ignored:

```
Filesystem type for checkpoints: EXT4. Checkpoint size: 110.60 GiB. Available RAM: 40.74 GiB.
```

* every rank streams the **whole** 110.6 GiB (4 x the reads), and 4 workers measured 10.6 / 11.8 / 18.4 / 12.9 GiB
  of RSS = 53.7 GiB at once;
* the kernel is simultaneously holding ~47 GiB of page cache for those reads, which is exactly the tension noted
  in `notes/native-4xr9700.md` ("the page cache the kernel wants to keep for a 110.6 GiB read is more than
  93.8 GiB wants to hold");
* the driver's `STOP` step returns as soon as the signal is sent, so the next 4-rank load started while the
  previous ~54 GiB was still mapped (its processes were already heading into `exit_mm`).

Swap is the trip wire: the moment `SwapFree` goes near zero and reclaim starts moving anon pages of GPU-attached
processes, the SVM notifier cycle above is reachable.

## Rules for every load-path experiment from here

1. Gate every launch: `MemAvailable >= 55 GiB` and no `vllm serve|VLLM::` process alive (`wait_down()` in
   `bench/coldstart_ab.sh`). If a process will not die in 180 s, stop and treat the box as wedged.
2. Never overlap two 4-rank loads. "STOP then immediately launch" is the bug; teardown must be *verified*.
3. Watch swap, not just free RAM: `grep -E 'MemAvailable|SwapFree' /proc/meminfo` before each launch. If swap is
   meaningfully used, reboot before measuring -- numbers taken in that state are junk anyway.
4. Keep the run unattended and outside the agent's lifetime (`setsid`), and poll with short tool calls rather
   than `sleep 600`, so an OOM kill of the agent costs one call and not the session.
5. Two launches per config still stands: the only real numbers from this run are 208.4 s cold vs 103.2 s warm
   for an identical config (2.02x), with experts place time 137.6 s vs 34.5 s.

## What the prefetch question turned out to be

Moot. `--safetensors-prefetch-num-threads` / `--safetensors-prefetch-block-size` only apply when
`safetensors_load_strategy == "prefetch"`; on EXT4 the strategy is auto-disabled (and would be anyway, since
110.6 GiB > 90% of available RAM). Forcing the strategy stages future shards into the same RAM that is already
short -- vLLM warns "This may cause out-of-memory errors". The cold/warm page-cache arm in
`bench/coldstart_ab.sh` is the only prefetch-shaped lever on local NVMe/EXT4.
