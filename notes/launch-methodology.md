# How a launch is started, watched and stopped on this box — 2026-11-03

The CHANGELOG points here twice and the file did not exist: the SIGTERM behaviour had been recorded in
[native-4xr9700.md](native-4xr9700.md) (line 300, "hit twice while probing") and nowhere else, while the *method*
— what supervises a launch, what refuses one, what notices a death — had never been written down. It is worth
writing down because two of those mechanisms were found broken by using `./setup.sh` for real on 2026-11-03,
and the failure is invisible from the outside unless you go looking.

Everything below is about the **native** path (`serve/native-flashnext.sh`: stock vLLM wheels in
`~/venv-r9700` plus a host-built `libr9k.so`, no docker, no root). The container path (`serve/serve.sh`,
`sudo docker run -d`) has its own supervisor and is not what the box behind the README's numbers runs.

## 1. The chain, and where the log is

`serve/native-flashnext.sh` pre-flights, exports the `R9K_*` defaults, and **`exec`s** `vllm serve` — the launcher
process *becomes* the API server, so one pid is the whole serve (its children are `VLLM::EngineCore` and the four
`VLLM::Worker_TP*`). `LOG=` names the file the Monitor's About page points at; since 2026-11-03 `setup.py` runs the
launcher inside a tmux session whose pane pipes `2>&1 | tee -a "$LOG"`, so the file the page names and the file
holding the crash are the same file. Cold start measured on this box (120.3 GB checkpoint, TP4, MTP-2, 32k):
**145 s** from the tmux start to the first `/health` 200, and `pgrep -cf 'VLLM[::]'` answers **5** once it is up
(API server + EngineCore + four ranks).

## 2. SIGTERM does not stop the ranks

Restating native-4xr9700.md's measurement, because every later mechanism depends on it: `pkill -f 'vllm serve'`,
and even SIGTERM to the ranks, leaves `VLLM::Worker_TP*` alive holding ~27-31 GiB a card; the next launch dies two
minutes in with `Free memory on device cuda:0 (0.0/31.86 GiB)`. Consequences built into the scripts:

- the launcher refuses outright (`LEFT=$(pgrep -cf 'VLLM[::]')`), with `STOP=1` doing the `kill -9` and exiting;
- `setup.py` runs the **same count before starting**, so a `--launch` on a box whose GPUs are still held is refused
  in one line with the `STOP=1` command, rather than costing a fresh two-minute failure;
- a launch is never "stopped" by a signal anyone sends by accident: nothing here sends one.

Measured on 2026-11-03: with a live serve, `./setup.sh --yes --launch native-flashnext.sh` stopped at
`refusing to launch: 5 vLLM process(es) still holding the GPUs`.

## 3. How a launch got lost anyway (the reason supervision changed)

At 05:45 UTC on 2026-11-03 a `--launch native-flashnext.sh` started by `setup.py` with
`subprocess.Popen(..., start_new_session=True)` died mid-load:

- the log stops at **exactly 12288 bytes**, on the `multiproc_executor ... world_size=4` line — a whole number of
  blocks, i.e. the writer was cut off, not an error path;
- no error line, no traceback, **no shutdown record in the request journal** (`~/logs/requests-8080.jsonl` stayed at
  its previous row count), nothing in `dmesg`;
- four cards back to 0.056 GiB, no `VLLM` process left, port 8080 answering nothing.

That is what a death looks like from the outside. **It was not a process-tree kill, and the first write-up of it
here blamed the wrong thing.** `journalctl --list-boots` puts the boot that ran that launch — `6a51a2dc` — ending
with kernel lines at **05:45:19**, 36 s after the launch started and inside the weight load, and the boot that
followed it opens by **replaying a machine-check record timed 05:47:12** (`mce: [Hardware Error]: … SOCKET 0
APIC 82`). That boot died too, at 05:57:43, and the one running now opened at 06:00:23 replaying an MCE timed
**06:00:06**: three resets in 18 minutes, one landing exactly on the moment the launch stopped writing.
`tools/journal-pcie.py` is the instrument, and it separates `live` from `replayed` records for precisely this
reason — so a crash is attributed to the boot that ended, not to whatever job was running in the boot that began.

Two corrections that fall out of it, both worth keeping because each is a trap:

- **A new session was never shown to be the problem.** Nothing here has demonstrated that a `start_new_session`
  child dies with its caller's tree. tmux is still the right shape, for the reasons that do hold: the serve must
  outlive the ssh session that started it by four minutes of loading, there is a pane to attach to while it loads,
  and `tmux has-session` is a liveness probe. `--no-tmux` loses those three things; it does not add a hazard.
- **The AER counters do not explain these deaths.** The boot in which that launch stopped logging had **one** AER
  event. The deaths are CPU-side machine checks (HEST/GHES: boot -2 even logged `HEST corrected error threshold
  limit: 10`), so "0 uncorrectable on the link" is no reassurance about them at all — the right question for a
  death is the replayed-record timeline, not the AER totals.

And what the link counters *do* correlate with, measured the same morning (boot `5ec7e479`): **all 53 AER event
lines fall in 06:02–06:05**, which is the window in which `vllm serve` read 120.3 GB of weights out of host RAM
and pushed them across slot 6's riser (`0000:c4:00.0` upstream port counting BadTLP as its receiver, the root port
`0000:c0:03.1` counting the Timeout on the same link seen as its transmitter), while the idle rate after the load
measured **+1 event in 5.7 min**. So this is sustained host→device DMA on one riser chain, not random PHY noise.
Totals are per boot and zeroed at enumeration — and this box reboots on its own — so only an in-boot rate means
anything: `tools/reseat-check.py --save before.json … --diff before.json`.

The second failure was the reporting: `setup.py` polled `/health` for the full 420 s **after the process was gone**
and then printed one line, `not answering /health within 7 min`, without showing the log or saying the process had
exited. A box with a dead serve looked like a box with a slow serve.

## 4. What supervises it now

`setup.py` starts a native launch in a tmux session named `r9700-<recipe>` (tmux is what this box has;
`systemd-run --user` is the other shape and is available). Three parts of this are load-bearing and easy to get
wrong:

- **Liveness is the session.** The pane runs `bash serve/<x> 2>&1 | tee -a <log>`, so the session exists exactly as
  long as the serve does; `tmux has-session` is the liveness probe `setup.py` polls alongside `/health`. A
  `Popen` child is polled with `poll()` instead, and both produce a reason (`the launcher exited (code 3)`,
  `the tmux session X holding it is gone`) rather than a timeout in seven minutes' time.
- **The environment must be pushed through `tmux new-session -e`.** tmux starts its session in the *tmux server's*
  environment, not the caller's: `VENV=`, `GPUS=`, `TP=`, `NSEQ=`, `OFFLOAD_GB=` and every `R9K_*` would otherwise
  vanish at the boundary and the launch would silently run on defaults. The names are scraped from the launch
  script's own `${NAME:-...}` uses, so a knob added to a script is a knob `setup.py` cannot forget.
- **`--no-tmux` is the fallback, not the default.** It keeps the bare detached start and prints what it gives up:
  no pane to watch while the engine loads four minutes of weights, and no session to ask afterwards whether the
  launch is still up.

A tmux session still dies if the tmux server is killed. What the box has for that is (a) a watcher `setup.py`
arms once the launch answers `/health` (`--watch`, detached; it writes `died_at`/`reason` into `launch.json` and
appends one line to the launch's log), and (b) the Monitor sidecar (notes/webui.md) — either outlives the launch,
which is the honest version of "supervised": **seen**. Resurrecting it used to be refused outright, and the reason
it gave was the right reason: *a serve that restarts into a degrading link turns a fault into a loop*. That is now
available **under that condition**, as a watcher option rather than a unit, and the default is still off.

`WATCH_RESTART=1` lets the watcher re-run the launch, and four gates have to agree first
(`restart_ok()`, pure, so `tests/test_setup.py` can hold it to its rules without killing a box):

1. **Someone asked.** Default off; the answer the watcher prints names which case it is in, and the launch record
   carries `restart_on_crash` / `restart_max` / `restart_window_s` so "would it have come back?" is answerable
   after the fact without remembering which shell armed the watcher. A hand-armed watcher takes its own
   environment and writes that onto the record at claim time.
2. **The log says it is the crash we know.** `CRASH_SIGS` is this box's own words — `illegal memory access`,
   `EngineDeadError`, `died unexpectedly`, `HSA_STATUS_ERROR`, each carrying the timestamp it was copied from.
   A `STOP=1`, a killed session and a reboot produce none of them and get no restart.
3. **No ranks left behind.** The 2026-11-14 fault left its ranks alive for 11 more minutes; restarting into that
   is two launches fighting for four cards, which is a worse outage than the one on screen. So this gate makes
   the recovery *wait* for the wedge to drain — honest limitation, stated: for that death the serve would have
   come back around 21:10, not 20:59.
4. **The breaker, and the link.** At most `RESTART_MAX` (3) inside `RESTART_WINDOW_S` (3600 s), and only while no
   AMD display device is running below its rated link (`gpu_links()` reads `current_link_speed/_width` against
   `max_link_speed/_width` — free to read, no root, no `/dev/kfd`). It is deliberately about the **rating** and
   not the AER counters, because this box carries dozens of correctables a minute at a full 32.0 GT/s ×16 during a
   weight load: a counter moving is not the signal, a link that negotiated short is. An unreadable link does not
   block the recovery, and the reason line says nobody measured it.

What it re-runs is the recorded launch through `setup.py` itself (`--yes --launch <recipe> --webui <mode>
--config <name>` plus the `rest` flags and the `--no-tmux` choice, now recorded), not a bespoke restart path: the
restart then does the `/health` wait, the context check, the gateway bring-up and the watcher arming like any
other launch. The death survives the relaunch in `.r9700-state/restarts.json`, because the new launch rewrites
`launch.json`.

Armed on this box since 2026-11-15 05:58 (`WATCH_RESTART=1`, 3 per 60 min), and dry-run against real inputs
rather than assumed: the live launch's own log tail carries no signature → no restart; the tail of the launch that
really died at 05:11 (`setup-044236.log`) yields `EngineDeadError` → `(True, 'EngineDeadError; restart 1 of 3 in
60 min')`; the four cards read 32.0 GT/s ×16 against the same rating, so the link gate is open.

**A crash can hide from every one of those gates at once, and on 2026-11-15 06:28 it did.** A soak hit the device
fault (notes/bug-causal-conv1d-oob.md §10) and left: `Worker_TP0` dead, `VLLM::Worker_TP1/2/3` alive and holding
32.37/32.37/32.30 GiB (they ignore SIGTERM), `/health` refusing, and the **tmux session still open**
(`pane_current_command=bash`, `dead=0`). The watcher was armed and polling every 10 s, and the record went on
saying `died_at: null` for 26 minutes: `watch_dead()` needed the session gone **or** no ranks left, and it had
neither. So the §4 rule failed in both directions at once — it was never *seen*, so it could never be resurrected.
Then the empty pane took over: with the ranks cleared by hand, `setup.py --launch native-flashnext.sh` answered
`a tmux session named r9700-native-flashnext is already up` and the box stayed down another 17 minutes.

Three changes, all host-tested (`tests/test_setup.py`), all behind the same opt-in:

- `watch_dead(port, sess, prev, quiet)` can now return **crashed and wedged**: no `/health`, ranks present, a
  record that had already answered `/health` once (`up_at`), `WATCH_WEDGE_TICKS` (2) silent probes in a row, and a
  known signature in the log tail. The loading case is still protected — a launch loading weights has ranks and no
  `/health` too, and with no `up_at` it is not called dead.
- `RESTART_STOP=1` (default once restarts are armed; `=0` keeps the old passivity) runs the **recipe's own**
  `STOP=1 bash serve/<recipe>` to clear ranks that outlived their engine — the operator's command, not an ad-hoc
  `pkill`, because the launcher knows its ranks. `stop_wedge_would()` is the pure predicate: switch, armed
  restart, ranks really present, signature, and a record naming a recipe. Gate 3 above stops being "wait for the
  wedge to drain" and becomes "drain it", so the 11-14 gap between 20:59 and 21:10 shrinks to ~2 probes plus the
  stop.
- `corpse_session(sess, left, rec)` decides the launcher's refusal instead of holding it: close a pane the record
  says is over with nothing running in it; refuse for one with ranks, one the record does not vouch for, or one
  that might still be loading.

And the gate that must never fire on a stop was proved on the box rather than in a test: at 06:42:46, after a
deliberate `STOP=1`, the armed watcher logged `this launch is gone: the tmux session … is gone — nothing restarted
it (no known crash signature in the log tail)`.

**Current state of that switch on this box (2026-11-15, 07:40): off.** The launch is watched in **record-only**
mode — `launch.json` says `restart_on_crash: false`, `restart_max: 0`, and the watcher's own environment carries
no `WATCH_RESTART`. The reason is the operator's, and it is the right call: **restarting a faulted four-card serve
is availability, not a fix**, and a box that quietly comes back can also quietly stop producing the evidence the
fault needs. Recording stays on because 2026-11-14's 1 h 48 m of nobody knowing is a worse failure than a manual
relaunch. The mechanism above stays implemented, tested, and used for **soak** launches, where a hit is the point
and the thing worth protecting is the run rather than the traffic (both of today's field restarts were on TEST
launches, `restarts.json` `at: [06:36:37, 06:55:57]`).

The watcher exists because of 2026-11-14: the launch answered `/health`, so `setup.py` exited, and the engine
faulted five days later. `launch.json` went on saying `up_at` with `died_at: null` and the death sat unnoticed
for 1 h 48 m, because the box runs `WEBUI=same` and the Monitor page lives **inside** the serve — it went down
with it. So the rule about which webui mode a production launch runs is a supervision choice, not a cosmetic
one: `same` is one less process and no page that outlives the death
(notes/bug-causal-conv1d-oob.md §6).

## 5. What the next run can answer

`.r9700-state/` (gitignored) holds what the last run measured, so a run reports drift instead of existence:

- `native.json` — the venv's `vllm/torch/hip`, the `libr9k.so` hash for **both** copies (the one
  `kernels/build.sh` writes and the one the launcher loads through `R9K_LIB`, which README's own build line
  separates with a `cp`), the newest kernel-source mtime, and the `docker/Dockerfile` pin.
- `launch.json` — recipe, port, log, session, `started`, `up_at`, `watcher`/`watcher_at` (the detached `--watch`
  process armed when the launch came up), and `died_at`/`reason` when that watcher saw it die. Step 8 then
  answers the two questions that matter after a death: *is the launch I started still up* (5 processes, and
  `tmux attach -t r9700-native-flashnext`), or *here are its log's last lines and the STOP=1 that clears the
  ranks*. Before 2026-11-14 the `died_at` half was a promise with no code behind it: only a launch that died
  **before** first answering `/health` got stamped.

  Two properties of the watcher worth knowing when a box looks like it has too many of them: a watcher only ever
  stamps the record whose `session` matches the one it was armed for — a stale watcher (one armed by an earlier
  launch, or one started by hand) writes **nothing** and exits at the first death it does not own; and two
  watchers on one death are safe, because the second sees `died_at` already set and stops. Both are asserted
  (`tests/test_setup.py`), and the recording path itself was proved end-to-end against a throwaway tmux session
  in a temporary state directory: nothing written while the session was up, `died_at` + `reason` + one appended
  log line the moment it was killed, and the live record untouched.

`--check` writes no state and rewrites none: that is asserted, not assumed
(`tests/test_setup.py`, 85 checks, no GPU and no card touched).

## 6. Refusals, and the numbers behind them

`setup.py` refuses a native launch before starting it when:

| condition | this box | where the number comes from |
|---|---|---|
| ranks still hold the GPUs | 5 processes, ~27-31 GiB a card | native-4xr9700.md; the launcher's own `LEFT=` check |
| `OFFLOAD_GB` wants more pinned memory than `RLIMIT_MEMLOCK` | cap **12,010 MiB**; the two-card expert stream wants **34 GB a rank** | PROGRESS.md's "11.7 GiB per process"; `serve.sh` buys the container's `--ulimit memlock=-1`, which is what the native path does not have |

The pinned-memory one is the interesting pair: on four cards nothing is pinned at all (`Mlocked 0`, notes/native-
4xr9700.md), so a finite cap is invisible until someone sets `OFFLOAD_GB` — two minutes and a dead engine later.

## 7. The commands, pasteable

```
tmux attach -t r9700-native-flashnext            # watch the serve load (detach: Ctrl-b d)
tail -f ~/logs/setup-<HHMMSS>.log                # the same bytes in a file
./setup.sh --check                               # is the recorded launch up? what drifted?
STOP=1 bash serve/native-flashnext.sh            # stop (kill -9: the ranks ignore SIGTERM) and exit
WEBUI=same bash serve/native-flashnext.sh        # relaunch by hand
./setup.sh --yes --launch native-flashnext.sh --webui same    # or the whole check, then a supervised launch
```

## 8. A test config, and getting back to production the same day

Measuring something on this box usually means changing something: today it was `PREFIX_CACHE=1`, to price what the
prefix cache would buy. A launch that carries a test config is not the machine the recipe describes, and every
number the Monitor then shows silently belongs to the other machine -- which is worse than not measuring, because
the page looks the same. Three rules, and one command that checks them:

- **Name the test session and its log as tests.** `tmux new-session -d -s r9700-TEST-prefixcache "PREFIX_CACHE=1
  WEBUI=same LOG=~/logs/test-prefixcache-$(date +%H%M).log bash serve/native-flashnext.sh 2>&1 | tee -a ..."`:
  `tmux ls` then answers "is a test running right now" without reading a log, and the test's bytes never land in
  the production log the About page points at.
- **Revert by relaunching the untouched command**, not by re-typing the test one without the flag. The default is
  the recipe; the flag was the exception.
- **Prove it with `tools/recipe-check.py`.** It asks `serve/native-flashnext.sh` for its own `DRY=1` launch line
  (so it can never disagree with the tree about what production means), reads the live serve's `config` block and
  the recipe meta file, and prints either `PRODUCTION recipe` or every measured knob that differs, with the
  restore command. Exit status 1 when it differs, so it goes in a gate.

**What the 2026-11-04 test bought, and what it does not mean.** Six concurrent requests, each the *same* 14,530-token
filler with a unique nonce at the end, 250 output tokens, started from an idle engine:

| | wall | summed tok/s | median TTFT | median own rate |
|---|---:|---:|---:|---:|
| production recipe (`PREFIX_CACHE=0`) | 32.3 s | 46.0 | 25.3 s | 37.6 tok/s |
| test config, cold prefix (`PREFIX_CACHE=1`) | 13.7 s | 103.9 | 10.0 s | 71.5 tok/s |
| test config, same prefix re-sent | **3.6 s** | **389.9** | **0.66 s** | 88.6 tok/s |

The 8.6x on wall and 38x on TTFT are the **best case**: the corpus is six copies of one prompt, so 133,120 of the
346,754 prompt tokens were cache hits (block-quantized to the 880-token hybrid block, which the engine's own
`prompt_tokens_cached_total` reported and the Monitor subtracted). Real traffic pays in proportion to how much of
its prompt is actually shared. That A/B ran while `PREFIX_CACHE=0` was still the recipe, and the whole 262k
measurement series (notes/context-256k.md, notes/kv-fp8-qsa.md) was measured with the cache off -- those tables now
carry that label. Extending the series means measuring on the recipe that runs, not keeping the old default so the
tables stay comparable. The number worth remembering is the shape: for identical-prefix traffic the cache
does not redistribute prompt work, it **removes** it -- which is the one lever the step census in
notes/context-256k.md says the scheduler's fairness knobs cannot reach.

That filler corpus is fine for pricing a mechanism and useless for deciding whether to run the cache always on.
For the decision, replay the box's own conversations instead: `tools/prefix-replay.py`, and
notes/prefix-cache-decision.md for the method, the numbers, and the verdict (98.1% of prompt tokens back, no extra
queueing, 2% of pool as the price).

## 9. A death during measurement: 2026-11-04 07:53

The production serve died on its own, mid-traffic, and the Monitor noticed before anyone did (the sidecar's
`upstream not answering` notice is what sent me to the log).

- **Signature:** `torch.AcceleratorError: CUDA error: an illegal memory access was encountered`
  (`hipErrorIllegalAddress`) in `Worker_TP0`, then `EngineDeadError`, then a 500 to the client whose stream was
  open. Not an OOM: the scheduler dump says 2 running requests, `kv_cache_usage=0.156`, 0 preempted, and the step
  was six scheduled tokens (two requests, MTP `[-1, -1]`) at 36,260 and 111,044 computed tokens. A 111k-deep
  decode step, so the same deep-context decode region §6 of these notes and notes/bug-mes-invalidate-tlbs.md live
  in. Nothing about the config had changed: `tools/recipe-check.py` re-run against the relaunch exits 0.
- **What the guard bought:** the API server was gone but the four ranks were still holding 32 GB of VRAM each
  (`rocm-smi --showpids`, `pgrep -cf 'VLLM[::]'` = 4), which is exactly why the launcher refuses instead of
  trying. `STOP=1` cleared them; the same command came back healthy in 168 s.
- **What was NOT the cause:** the prefix-cache test config had already been reverted and the test session killed
  (§8) 20 minutes earlier; the traffic in flight was real client traffic from 192.168.1.x, and the journal shows no
  test wave at 07:53. Recorded so the next reader does not blame the experiment that happened to be nearby.

## 10. A box fact is not a repo fact: LOCAL_ENV

`serve/native-flashnext.sh` sources `~/.config/r9700-stack/native.env` (path overridable with `LOCAL_ENV=`) before
its own defaults, and prints `local env: <path>` at every launch. The dividing line for what may live in it:

**Precedence, stated rather than assumed (2026-11-15).** The file used to be sourced with a bare
`set -a; . "$LOCAL_ENV"; set +a`, which made a box fact an OVERRIDE: `GATEWAY_PORT=8090 bash serve/gateway.sh` got
8080 back because the file says 8080, and `serve/webui.sh`'s own documented usage line
(`PORT=8100 bash serve/webui.sh # a second launch on 8100 -> monitor on 8101`) silently watched 8180. Worse, it
disagreed with how `setup.py` reads a launcher — `launcher_var` asks the environment first, then the file, then the
script's default — which is this section's own failure class in a new costume: two readers, two orders, two meanings
for one launch line. The sourcing now lives in one place, `serve/lib/localenv.sh`, sourced by all five launchers: the
file fills only names nobody exported, `set -a` is kept so those facts still reach the child processes, and the
launcher prints which names it left alone.

- **a knob that changes what a measured number means stays in the launcher**, next to the measurement that justifies
  it, so `tools/recipe-check.py` can hold the running engine to it: `PREFIX_CACHE`, `NBT`, `NSEQ`, `UTIL`, `KVDTYPE`,
  `MTP`, each with its numbers in the comment above it.
- **a fact about this machine that changes no measurement goes in the local file.** Today it holds exactly one:
  `HOST=0.0.0.0`. The repo's default is `127.0.0.1`, which is also upstream's (`args.host or "127.0.0.1"`), and that
  is what a fresh checkout on a new box should bind: a committed `0.0.0.0` with no `--api-key` is an exposure that
  follows the code to wherever the code goes next. On this box a loopback bind refuses the traffic it carries -- `ss`
  shows 192.168.1.130 and 192.168.1.200 holding :8080 open -- so the address lives with the box, survives a
  `git checkout` or `git reset`, and is named in the launch log instead of remembered by whoever typed the command.

Provable rather than remembered: with `LOCAL_ENV=/nonexistent` the `DRY=1` line must read `--host 127.0.0.1`; with a
local env naming another address it must print that address and the `local env:` line. The recipe file records
`host` and the About page shows it, while `launch_hint()` deliberately leaves it out of the restore command because
the launcher picks it up from the local file again -- a page that outlives the box must not hand out a hard-coded
address as if it were the recipe.

Found while adding this, and worth keeping as a rule: **a preview must not touch state.** `DRY=1` used to write the
recipe file and could start the sidecar. A `DRY=1 WEBUI=same` run meant to read the launch line back overwrote the
recipe of the serve that was actually running, so a cache-off engine's About page described a cache-on launch and
`recipe-check.py`'s meta-against-engine agreement became a lie until a restart. `DRY` now writes nothing and starts
nothing; `METAONLY=1` is the deliberate version of the same write, for refreshing the recipe file of a launch that
is already up or was started by hand.

### The same rule covers the Monitor's sidecar, and sourcing has a trap in it (2026-11-04)

`0f64838` made the sidecar bind loopback by default -- `/api/*` carries the engine's config, the request records and
the kernel-log summary, so a default of `0.0.0.0` was an exposure no caller asked for. This box reads the Monitor
from other machines for the same reason it binds `:8080` openly, so the choice goes in the same machine-local file
as `R9K_WEB_BIND=0.0.0.0`; `serve/webui.sh` now sources that file too and passes the resolved value explicitly
(`WEBUI_BIND` -> `R9K_WEB_BIND` -> `127.0.0.1`), which keeps the running process's bind address readable in its own
command line instead of inferred from which env happened to be set.

The trap: **sourcing a file makes shell variables, not environment variables.** `serve/webui.py` reads `R9K_WEB_BIND`
out of the *environment* as its `--bind` default, so a launcher that sources the file and does not pass `--bind`
inherits `127.0.0.1` anyway. Measured, not guessed: a sidecar restarted from a hand-typed command that sourced the
file came up on `127.0.0.1:8099` while the file said `0.0.0.0`. Both launchers now source with `set -a; . file;
set +a`, and `serve/webui.sh` passes `--bind "$BIND"` explicitly on top of that, so the fact works whether the
process is started by the launcher, by the engine launcher's `WEBUI=same` path, or by hand.

`DRY=1 bash serve/webui.sh` was added with it, for the same reason the engine launcher has one: `webui.sh` used to
run its `pkill` before doing anything else, so a command meant only to print the sidecar line would take down the
sidecar it was printing -- and its 1 m / 15 m / 1 h windows restart from empty, which is a state change with
visible consequences on the page. The checks in `tests/test_webui.py` run the preview for real: with this box's file
it must print `--bind 0.0.0.0`, with `LOCAL_ENV=/nonexistent-box.env` it must print `--bind 127.0.0.1`, and the
sidecar that is actually running must be listening on the address the file claims (`ss -ltn`).

### The file moved into a helper, and the reader kept looking for the old line (2026-11-16)

`launcher_local_env` found a launcher's machine-local file by matching the inline line
`LOCAL_ENV=${LOCAL_ENV:-…}`. Section 10's own refactor moved that default into **one** file —
`serve/lib/localenv.sh`, which says `: "${LOCAL_ENV:=$HOME/.config/r9700-stack/native.env}"` — and the reader was
not moved with it. It returned `None`, so `launcher_var` fell through to the recipe's `${PORT:-8080}` and the launch
record wrote **port 8080 on a box whose engine binds 8180 with the front door on 8080**. Two consequences, both found
only because something else was being checked:

* `tools/recipe-check.py` follows the record's port to reach *the engine's own* page and read the launch argv. Sent
to 8080, it read the front door's payload instead — a sidecar's argv and a sidecar's env — and printed "NOT the
production recipe: 1 measured knob and 19 environment knobs differ" against an engine that `#61174`-verified and
`R9K_GDN_PREFILL_MAX=1024`-loaded was configured exactly as the recipe says. A gate that reports noise on a good
state is a gate that gets ignored.
* `setup.py --watch` probed `os.environ.get("PORT", "8080")`, i.e. the **door**. It worked by luck — the gateway
forwards `/health` to the engine — but it makes a dead door indistinguishable from a dead engine, which is exactly
the confusion an automatic restart must not act on, and it left the record naming a port nothing served.

So the resolution order is now explicit in one place: an exported `LOCAL_ENV`, then the launcher's own inline
default, then the helper's — and the watchdog's probe port comes from `watch_port()`: the record's port first (a
death should be judged against the port that launch bound), the launcher's second (the record may be missing after a
load that died), `8080` last. `watch_loop` prints the answer once at the claim — `probing :8180 for this launch
(session …); restart on a known crash signature: yes (up to 3 per 60 min)` — because "the watchdog called it dead"
is only checkable if the record says what was probed. `tests/test_setup.py` pins all of it, including a live claim
that the record's port answers `/api/status` with the **serve's** argv and is not the gateway's port.

## 11. `STOP=1` kills anything whose command line contains the pattern -- including your own command

The stop path is `pkill -9 -f 'VLLM[::]'; pkill -9 -f 'vll[m] serve'`. The brackets are there so the launcher does
not match its own command line. They are also the reason a routine check can come back dead: I ran

    STOP=1 bash serve/native-flashnext.sh ... pgrep -cf 'VLLM::'; pgrep -cf 'vllm serve'

in one command, and the whole shell exited **137** -- `pkill -9 -f 'VLLM[::]'` matched the literal `VLLM::` sitting in
my own wrapper's command line and SIGKILLed it mid-flight. Nothing was wrong with the stop path; the operator's
command line was inside its blast radius. Two rules: **run `STOP=1` by itself**, and when checking afterwards use the
bracket forms (`'vll[m] serve'`, `'VLLM[::]'`) so your probe cannot match its own pattern. The same applies to any
`pgrep -f` in a one-liner that also mentions the thing being killed.

A second, quieter effect of a hard stop: the engine pane dies by SIGKILL, so the tmux **server** exits with it
(`no server running on /tmp/tmux-1000/default`) -- the session does not linger in a dead state, and `tmux ls` is not
a way to tell "stopped" from "crashed". What answers that question is the log's last lines plus
`rocm-smi --showpids` (and, on a crash, the four ranks that held ~32 GB each after the API server left -- §9).

## 12. A crash discriminator that lives in an env var: the test config, named and revertible

**Why this one exists.** Two `hipErrorIllegalAddress` deaths on 2026-11-04 (07:53 with the prefix cache off, 14:01
with it on) share one signature: the driver names `r9k_router_gemm_kernel<8, 2, true>` as the faulting kernel, and
the crashed step is a decode step of **6 rows** (2 streams x (1 + 2 MTP drafts)), deep context (up to 203,257 computed
tokens), 0 preempted, KV usage 16-27%. `MT=8` is exactly the instantiation the C++ dispatcher picks for M in 5..8, so
the shape and the faulting instantiation agree. The discriminator: `R9K_ROUTER_MAX_M` (r9700_vllm/router.py, default
8) caps the rows our gate kernel takes; at 4, an M=6 step falls back to hipBLASLt while M<=4 stays on ours. If the
crashes stop or move to another kernel, the fault is in that instantiation's path; if they persist in another kernel,
ours is the reporter and not the corrupter -- which is what vLLM's own error text suggests ("CUDA kernel errors might
be asynchronously reported at some other API call"). Cost, measured and written in router.py's own docstring: our gate
is ~5 us, hipBLASLt's tile ~19.6 us, 51 calls per decode step, so only steps wider than 4 rows pay ~0.75 ms.

**It is a test, and it is named as one.** tmux session `r9700-native-TEST-routermaxm4`, log
`~/logs/TEST-routermaxm4-1104-1412.log`. The proof that the box is actually running it -- not that a variable was
typed -- is two commands, and both must answer:

    tr '\0' \n < /proc/$(pgrep -f 'vll[m] serve' | head -1)/environ | grep '^R9K_ROUTER_MAX_M='
    grep "r9k router GEMM installed" ~/logs/TEST-routermaxm4-*.log | tail -1

which on this launch printed `R9K_ROUTER_MAX_M=4` and `... installed on 1 gates (M <= 4, split 2)` -- against the
crashed launch's `(M <= 8, split 2)`. The engine's own config is unchanged by it (`enable_prefix_caching True`,
`block_size 832`, `num_gpu_blocks 1242`, pool 977,726), so the experiment moves one knob and nothing else.

**Revert is a relaunch, not an edit.**

    STOP=1 bash serve/native-flashnext.sh
    tmux new-session -d -s r9700-native-flashnext \
      'MAXLEN=262144 NSEQ=32 NBT=8192 WEBUI=same LOG=/home/jiulu/logs/kvf8b-152626.log bash /home/jiulu/r9700-stack/serve/native-flashnext.sh 2>&1 | tee -a /home/jiulu/logs/kvf8b-152626.log'

and it is proved the same way: the new log must say `(M <= 8, split 2)` and `/proc/<pid>/environ` must have no
`R9K_ROUTER_MAX_M`.

**The gap this exposed in the gate -- and closing it.** `tools/recipe-check.py` compares knobs the *engine
measures* -- `max_model_len`, `max_num_seqs`, `max_num_batched_tokens`, `kv_cache_dtype`, `enable_prefix_caching`,
`num_speculative_tokens`. It answered "PRODUCTION recipe" for this test launch, correctly and uselessly: an env var
that changes which kernel runs is invisible to it, and env is exactly where most of this stack's experiments live
(`R9K_*`, and the two crashes happened with `R9K_ROUTER_MAX_M` unset, i.e. the default 8). So §8's rule (a test
config named, logged, and reverted by relaunching the untouched command) was what protected us that evening.

It is closed now, in three places that have to agree:

- the launcher prints its own environment (`ENVRECIPE=1` -> `env recipe: {json}`) and records it in the recipe file
  as an `env` object. The snapshot is taken **after the last default the script sets** -- taken earlier it missed
  `R9K_QSA_KVF8`, and the gate then blamed the live serve for a name the launcher itself exports -- and reading it
  no longer requires `FORCE=1`, because reading a recipe is not a launch and the gate is asked precisely while a
  serve is up;
- the gate reads the live environment from `/proc/<pid>/environ` when it can (the payload names the pid) and falls
  back to the environment the serve publishes about itself, printing which source it used -- a whitelist can hide
  the very name being looked for, `/proc` cannot;
- `R9K_`/`R9700_` names are compared by value, with two deliberate exceptions: the Monitor's own plumbing
  (`R9K_WEB_*`, `R9700_WEBUI_*`) configures a page and not a measurement, and a name that looks like a credential
  is compared for presence only -- from either side, so the report cannot leak a key the launcher happens to export.

And the *production* side of the comparison is asked with a stripped environment (`OPERATOR_KNOBS` plus the
namespace): run from a shell that exports `MAXLEN=999 R9K_ROUTER_MAX_M=7`, the launcher still answers with this
tree's line and no router knob, because otherwise "production" would depend on who typed the command.

The proof it works is the box itself, run minutes after the experiment started:

    tools/recipe-check.py
      1 environment knob(s) differ -- these change which kernel runs, and the engine's config does not show them
      (env read from /proc/92955/environ):
        env:R9K_ROUTER_MAX_M: this tree says None, the live serve says engine=4 meta=None
      exit=1

which is the correct verdict: **the box is running a test config**, and now the gate says so unprompted. Note the
`meta=None`: the recipe file on disk predates this change and has no `env` block, and it is deliberately not
refreshed with `METAONLY=1` here -- `METAONLY` records the environment of the shell that runs it, and re-recording
a launch I did not start would write a recipe that never happened.

### §12 result: 9 h 37 min clean at the cap, then the revert -- proved the same way it was set

The test launch (`R9K_ROUTER_MAX_M=4`, session `r9700-native-TEST-routermaxm4`, log
`TEST-routermaxm4-1104-1412.log`) ended at **9 h 37 m** of uptime carrying real traffic:

- `illegal memory access` lines in its log: **0**. `EngineDeadError`: **0**. Preemptions: **0**.
- 329,983 drafts, **1.317 accepted per draft** on live traffic, 49,419,968 prompt tokens arriving as cache hits.
- boot lines: each rank logs the install **twice** -- `installed on 48 gates (M <= 4, split 2)` at 23:48 and
  `installed on 1 gates` about a minute later. That is not a regression: the first pass is the target model's 48
  MoE gates, the second is the MTP draft model, which has one. `tail -1` on that log says "1 gates" and reads as
  though almost nothing was replaced -- grep for all of them.

Against the two deaths at the default cap -- one ~6 h into its launch, one 29 min into its launch -- 9 h 37 min of
real traffic without a fault is **suggestive and not proof**: the crash was rare, traffic differed, and moving the
cap also moves which kernel touches the router logits. What it does buy is the next step, should the fault return
at `M <= 8`: `AMD_SERIALIZE_KERNEL=3`, which the driver itself suggests, names the faulting kernel instead of the
first innocent reader that trips over the poisoned page.

The revert is a relaunch of the untouched production command, exactly as §12 prescribes, and its proof is the same
two commands that proved the test (which is the point of writing them down):

    (23:47) boot log:  ... installed on 48 gates (M <= 8, split 2)
           tr '\0' \n < /proc/<pid>/environ | grep '^R9K_ROUTER_MAX_M='   -> no match
           tools/recipe-check.py                                         -> PRODUCTION recipe, exit 0

and it is the first launch whose recipe file carries the environment: `env` is a 15-name object, with no
`R9K_ROUTER_MAX_M` in it. So the gate's answer no longer depends on remembering which experiment was running --
the recipe and the process disagree or they do not.

Restating the obvious because it is the part experiments erode: reverting **re-exposes** the default `M <= 8` path.
If the fault comes back in `r9k_router_gemm_kernel<8, 2, true>`, the discriminator did not discriminate, and the
next move is serialization, not another cap.

## 13. A recipe has three faces: measured knobs, environment, and launch flags

`tools/recipe-check.py` used to compare two things: the engine's measured config labels, and the `R9K_*` /
`R9700_*` environment. Both are blind to **flags**. Whether the serve came up `--language-model-only` or
`--limit-mm-per-prompt {"image": 2, "video": 0}` appears in no `vllm:*` label, and it decides what the box can be
asked and what the KV pool has to spare for the vision tower. A gate that cannot see flags cannot see that choice,
so the launcher's `DRY=1` line is now compared against the live process's own argv (published by the merged plugin
as `/api/status -> argv`) token by token.

Three things that fell out of writing it:

- **Presence is part of the comparison.** A bare flag parses to value `None`; a flag that is absent also reads as
  `None` if you are careless. The first draft did exactly that, and it reported *one* deviation for the mm flip
  where there are two: the added `--limit-mm-per-prompt` and the removed `--language-model-only`. Removing the
  text-only flag is the change; losing it from the report is how a gate starts lying.
- **Unprovable is not matching.** A payload with no argv (an in-process page older than this comparison) now prints
  `the LAUNCH FLAGS ARE UNPROVABLE` instead of a clean line that quietly counted them as equal.
- **`MM` is an operator knob, so it is stripped when the gate asks the launcher.** An ambient `MM=1` in the shell
  that happens to run `recipe-check` must not move the production answer, same as an ambient `NBT`.

The procedure for a test that changes a **flag** is the §8/§12 procedure plus two extra proofs, because a flag's
effect is not visible in the config block:

```
DRY=1 bash serve/native-flashnext.sh | grep -m1 'launch:'          # what production would be (images are ON)
DRY=1 MM=0 bash serve/native-flashnext.sh | grep -m1 'launch:'     # the way back, and a test config of it
# ... launch the test under a TEST-named session and log, measure, then revert with the untouched command:
STOP=1 bash serve/native-flashnext.sh
tmux new-session -d -s r9700-native-flashnext 'MAXLEN=262144 NSEQ=32 NBT=8192 WEBUI=same LOG=<log> ' \
  'bash $PWD/serve/native-flashnext.sh 2>&1 | tee -a <log>'
# and prove the revert four ways, not one:
/home/jiulu/venv-r9700/bin/python3 - <<'PY'
import json, urllib.request
a = json.load(urllib.request.urlopen("http://127.0.0.1:8080/api/status", timeout=30))["argv"]
print("--language-model-only" in a, "--limit-mm-per-prompt" in a)
PY
# 2) behaviour: POST one image. With images ON it answers; with --language-model-only it must be 400
#    "At most 0 image(s) may be provided in one prompt". The behaviour is the proof, not the flag string.
# 3) geometry: the KV pool follows the choice -- measured 2026-11-05: 1,188 blocks / 935,216 tokens with the vision
#    tower against 1,242 / 977,726 without, and `Maximum concurrency for 262,144 tokens per request` 3.57x vs 3.73x
# 4) tools/recipe-check.py -> "PRODUCTION recipe ... every launch flag in the live argv matches", exit 0
```

One test-writing trap worth keeping: the fixture launch line in `tests/test_webui.py` predates the mm flag, so
`fixture.replace("--language-model-only", ...)` was a **no-op on a string that never contained it**, and the
"caught in both directions" check would have compared a line against itself and passed. Any test that builds its
input by mutating a string has to assert the mutation happened -- `ck("the replace actually changed the line", ...)`
is now in the file for that reason.

The trap then recurred in its mirror image, and this is why the file no longer mutates strings at all: when the
**default flipped to images-on** (2026-11-05), every check that built a deviation with
`.replace("--language-model-only", ...)` silently matched nothing again -- same bug, other direction. The tests now
call one function, `_mm_set(line, on)`, which strips whichever multimodal form is present and appends the one being
asked for, so the checks say what they mean whichever way the default is set, and a guard check asserts the flip
actually changed the line.

### §11 again, in the middle of documenting §11

Restarting the Monitor sidecar, one shell line did: `kill <pid>`, then `pkill -f 'tee -a .../sidecar-8099[.]log'`,
then the relaunch. The bracketed pattern did not match itself — but the **relaunch string later in the same command
line** contained the unbracketed target text, so `pkill` matched the shell running the whole thing and the line died
with status 143 halfway through: the sidecar was killed and never came back until the next command.

The rule is not "use brackets", it is: **a pattern must not appear anywhere in the command line that carries it, and
a process you know the pid of should be killed by pid.** Half-finished operational lines are worse than failed ones,
because the state they leave (a dead Monitor, an apparent `0` in the listening-socket count) is a state someone has
to discover.
