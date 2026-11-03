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

That is an external kill of the process tree, and it reached a child that had its **own session**. A new session
detaches from the caller's terminal and process *group*; it is not the same as being outside the caller's tree, and
the distinction only matters when something kills that tree — here, the non-interactive call that had run setup.

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
- **`--no-tmux` is the fallback, not the default.** It keeps the old detached `start_new_session` start and says so
  on the line, because that is the shape that lost the launch above.

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
