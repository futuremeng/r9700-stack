# notes/incident-2026-11-01 -- pre-reboot evidence bundle

Captured 2026-11-01 03:03-03:06 UTC, immediately before the reboot, from a box that was already unrecoverable
(`load average 396`, kswapd0 in `D`). Narrative + root cause + rules: `../load-oom-svm-wedge.md`.

| file | what it is |
|---|---|
| `kern-blocked-tasks-0140.txt` | the full 01:40:49 hung-task block from `/var/log/kern.log` (10 tasks, 515 lines) |
| `owner-trace-kworker185.txt` | the trace of `kworker/185:3:64157`, the mutex **owner**: `svm_range_restore_work` -> `svm_range_validate_and_map` -> `dma_fence_wait_timeout` |
| `cgroup-oom-evidence.txt` | cgroup v2 layout (`0::/user.slice/user-1000.slice/session-96.scope`), memory.events / memory.max where readable, and whether systemd-oomd/earlyoom are running -- the capture was cut short by the box wedging (the trailing `pgrep` timed out) |
| `oom-lines.txt` | proof that neither `/var/log/kern.log` nor `/var/log/syslog` contains any kernel OOM message this boot (so the agent's death is not evidenced as an OOM kill) |
| `cgroup-oom-evidence.txt` | cgroup v2 `memory.events` per slice: `user.slice` and `user-1000.slice` both `oom 0 max 0 high 0 oom_kill 8 sock_throttled 51123`, `memory.max = max` (no limit), new `session-96.scope` `oom_kill 0`; systemd-oomd / earlyoom not running -- i.e. 8 **global**-OOM kills inside our user slice, victims unnamed |
| `wedged-pids.txt` | live state at 03:05:44: pid, state, wchan, RSS for kswapd0 / the old session's sshd / the driver bash / APIServer / the 4 workers, plus loadavg and memory counters |
| `system-state.txt` | kernel + amdgpu version, cmdline, cgroup, `/proc/meminfo`, swap, hung-task sysctls, loadavg |
| `agent-death-evidence.txt` | `who` (old pts/0, pts/1 never closed), `auth.log` session-close count = 0, new login at 01:45:05, `/var/crash` (no node crash), agent session-file mtimes |
| `agent-death-evidence2.txt` | (partial -- command timed out on the wedged box) old-session sshd state |
| `driver-results.txt` | the only two data points the A/B produced (`load_s=208.4` cold vs `103.2` warm, same config) |
| `driver-log.txt` | the two `Killed` job-control lines -- these are the driver's own `STOP` step, not OOM |
| `frozen-load-tail.txt` | where the load died: `1/23` shards, and the loader's own `Available RAM: 40.74 GiB` / `Auto-prefetch is disabled` warnings |

Not capturable from userspace (no root, box wedged): `/proc/<pid>/stack`, `dmesg` ring dump, GPU `amdgpu_fw` dump.
What survives a reboot: this directory, plus `kernel.hung_task_panic` if we choose to set it.
