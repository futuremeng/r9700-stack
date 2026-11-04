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

A tmux session still dies if the tmux server is killed, and nothing here restarts a serve that died. What the
box has for that is the Monitor sidecar (notes/webui.md) — it outlives the launch and reports it dead, which is
the honest version of "supervised": seen, not resurrected. A unit that restarts a four-card serve automatically
is not in place and is not recommended before the PCIe/reseat story
(notes/incident-2026-11-03, `tools/reseat-check.py`) is settled: a serve that restarts into a degrading link
turns a fault into a loop.

## 5. What the next run can answer

`.r9700-state/` (gitignored) holds what the last run measured, so a run reports drift instead of existence:

- `native.json` — the venv's `vllm/torch/hip`, the `libr9k.so` hash for **both** copies (the one
  `kernels/build.sh` writes and the one the launcher loads through `R9K_LIB`, which README's own build line
  separates with a `cp`), the newest kernel-source mtime, and the `docker/Dockerfile` pin.
- `launch.json` — recipe, port, log, session, `started`, `up_at`, and `died_at`/`reason` when the watcher saw it
  die. Step 8 then answers the two questions that matter after a death: *is the launch I started still up* (5
  processes, and `tmux attach -t r9700-native-flashnext`), or *here are its log's last lines and the STOP=1 that
  clears the ranks*.

`--check` writes no state and rewrites none: that is asserted, not assumed
(`tests/test_setup.py`, 57 checks, no GPU and no card touched).

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
its prompt is actually shared, and a `PREFIX_CACHE=0` recipe stays the right default for the 262k measurement
series (notes/kv-fp8-qsa.md). The number worth remembering is the shape: for identical-prefix traffic the cache
does not redistribute prompt work, it **removes** it -- which is the one lever the step census in
notes/context-262k.md says the scheduler's fairness knobs cannot reach.

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
