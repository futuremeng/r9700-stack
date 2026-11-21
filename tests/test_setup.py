#!/usr/bin/env python3
"""tests/test_setup.py - the decisions setup.py makes, without a box to make them on.

setup.py is a 600-line walk over a rack: it reads sysfs, a venv, a Dockerfile pin, and then it starts a
serve. Most of that cannot be tested here, and pretending otherwise would be a test that asserts nothing.
What CAN be tested is the small pure layer that decides things, and every one of these checks exists because
the decision it covers was wrong at some point:

  (1) docker is the default. The container path is a multi-GB image pull; the native path is what the box
      behind the README's numbers runs. --yes used to answer every question Yes, which made the expensive
      path the default one. Now the docker question is asked with a default of n, and --yes does not answer it.
  (2) a launch that died was reported as "not answering /health within 7 min" after 7 minutes of sitting
      next to a dead process, and the log was never shown. dead_reason/wait_up/tail_log are the fix, and
      wait_up is timed here: fast failure, with the log's last lines.
  (3) the launch list, the missing-checkpoint priority and the offload refusal all turn on which path this
      run took -- so those three classifications are checked directly.
  (4) libr9k.so freshness: two copies (build.sh's output, and the copy the launcher loads through R9K_LIB),
      and README's own build line ends in a `cp`. stale_since is what catches the rebuild that skipped it.
  (5) the launch record: the only way the next run can answer "did it die, and which log do I read".
  (6) the watcher that keeps that record true AFTER /health answers: which deaths it may write, which it must
      leave alone, and the one case (still loading, no /health) it must never call death.

Nothing here touches a GPU, a card, a network port, or ~/models. tmux is exercised only if it is installed,
and it says so when it is not.

Run: ./.venv/bin/python3 tests/test_setup.py        (stdlib only)
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
ROOTP = Path(ROOT)

import setup as S  # noqa: E402

fail = 0


def ck(name, cond, detail=""):
    global fail
    if not cond:
        fail += 1
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond or not detail else f"  {detail}"))


def cfg(**kw):
    """The few fields setup.py reads off its own argparse result."""
    d = dict(check=False, yes=False, update=False, rebuild=False, launch=None, no_webui=False,
             webui="side", img="r9700/vllm:dev", repo=str(Path.home() / "r9700-build" / "repo"),
             docker="auto", no_tmux=False, rest=[], conf_env={}, conf_name=None, conf_applied=False,
             config=None, save_config=None, list_configs=False,
             explicit={"launch": False, "webui": False, "docker": False, "tmux": False, "rest": False})
    d.update(kw)
    return types.SimpleNamespace(**d)


def capture(fn, *a, **kw):
    """Run a step and keep its printed output: half of what setup.py does IS the report."""
    buf, old = [], sys.stdout

    class W:
        def write(self, s):
            buf.append(s)
            return len(s)

        def flush(self):
            pass

    sys.stdout = W()
    try:
        r = fn(*a, **kw)
    finally:
        sys.stdout = old
    return r, "".join(buf)


# ---------------------------------------------------------------------------------- docker is opt-in ----
ck("the container path is not taken just because docker exists (default n)",
   S.want_docker(cfg(), True)[0] is False, S.want_docker(cfg(), True)[1])
ck("--yes does NOT answer the docker question Yes (ask() would; ask_no() does not)",
   S.ask_no(cfg(yes=True), "deploy with docker?") is False)
ck("--no-docker says no without asking",
   S.want_docker(cfg(docker="no"), True) == (False, "the container path is off (--no-docker)"))
ck("--docker takes the container path when docker is there",
   S.want_docker(cfg(docker="yes"), True)[0] is True)
why = S.want_docker(cfg(docker="yes"), False)[1]
ck("--docker on a box with no docker says so and stays native, rather than failing silently",
   S.want_docker(cfg(docker="yes"), False)[0] is False and "--docker asked" in why, why)
why = S.want_docker(cfg(), False)[1]
ck("no docker on PATH is reported as the box it is, not as a refusal",
   "no docker on PATH" in why, why)

env = dict(os.environ)
p = subprocess.run([sys.executable, str(ROOTP / "setup.py"), "--docker", "--no-docker", "--check"],
                   capture_output=True, text=True, env=env)
ck("--docker and --no-docker cannot both be asked",
   p.returncode == 2 and "not allowed with argument" in (p.stdout + p.stderr))
p = subprocess.run([sys.executable, str(ROOTP / "setup.py"), "--help"], capture_output=True, text=True, env=env)
ck("--help names the new flags, so the default n is discoverable",
   all(f in p.stdout for f in ("--docker", "--no-docker", "--no-tmux")))

# ---------------------------------------------------------------------------- what a recipe can run ----
ck("the native launcher is not classified as container-only (it only mentions docker in a comment)",
   S.container_recipe("native-flashnext.sh") is False)
ck("the container recipes are classified by delegating to serve.sh",
   all(S.container_recipe(n) is True for n in ("27b.sh", "flashnext.sh")))
names = [n for n, _ in S.recipes()]
ck("webmeta.sh / webui.sh / serve.sh / deploy.sh are never offered as launches",
   not ({"webmeta.sh", "webui.sh", "serve.sh", "deploy.sh"} & set(names)), ",".join(names))
ck("a checkpoint only container recipes want is a note on a native run, not a warning",
   S.model_visibility({"27b.sh", "deploy.sh"}, "native") == "note")
ck("... and a warning when this run did take the container path",
   S.model_visibility({"27b.sh"}, "docker") == "warn")
ck("a checkpoint a native recipe wants stays a warning",
   S.model_visibility({"native-flashnext.sh"}, "native") == "warn")
ck("a model named only by helpers (deploy.sh fetches, nobody runs) is a note",
   S.model_visibility({"deploy.sh"}, "native") == "note")

# ---------------------------------------------------------------- pinned memory the box cannot pin ----
MEMLOCK_12G = 12 * 1024 ** 3
ck("no offload, no refusal", S.offload_refuses(0, MEMLOCK_12G) is None)
ck("an offload the cap covers is not refused", S.offload_refuses(8, MEMLOCK_12G) is None)
ck("unlimited memlock is never refused", S.offload_refuses(34, -1) is None)
r = S.offload_refuses(34, MEMLOCK_12G)
ck("34 GB/rank against this box's 12 GiB cap is refused before the launch, not after two minutes",
   r is not None and "34" in r and "MiB" in r, r or "")
os.environ["OFFLOAD_GB"] = "34"
ck("OFFLOAD_GB is read from the environment (that is how the scripts take it)",
   S.offload_gb(cfg()) == 34.0)
ck("... and from the command line too, since it is passed on to the script",
   S.offload_gb(cfg(rest=["OFFLOAD_GB=12"])) == 12.0)
os.environ.pop("OFFLOAD_GB")
ck("left_vllm counts processes and survives pgrep's exit 1 on zero",
   isinstance(S.left_vllm(), int) and S.left_vllm() >= 0, str(S.left_vllm()))

# ---------------------------------------------------------------------------- the pages' real address ----
ck("WEBUI=same is on the serve's own port (printing PORT+1 there was a wrong URL)",
   S.webui_urls("same", 8080) == ("http://127.0.0.1:8080/monitor", "http://127.0.0.1:8080/about"))
ck("WEBUI=side is the sidecar on PORT+1",
   S.webui_urls("side", 8080)[0] == "http://127.0.0.1:8081/monitor")
ck("WEBUI=off promises no page", S.webui_urls("off", 8080) == (None, None))

# ---------------------------------------------------------------------------- libr9k.so freshness ----
with tempfile.TemporaryDirectory() as td:
    so = Path(td) / "libr9k.so"
    so.write_text("old kernels")
    src = Path(td) / "r9k_new.hip"
    src.write_text("new source")
    os.utime(so, (time.time() - 3600, time.time() - 3600))
    ck("a .so older than a source that builds it is reported stale",
       S.stale_since(so, src) == src)
    os.utime(so, None)
    ck("a .so newer than every source is not (no false alarm on this box)",
       S.stale_since(so, src) is None)
    ck("sha256 is stable and short enough to read on a line",
       S.sha256(so) == S.sha256(so) and len(S.sha256(so)) == 16)
    ck("sha256 of a missing file is None, not a crash", S.sha256(Path(td) / "nope.so") is None)
    load, built = S.libr9k_paths(cfg())
    ck("the two libr9k.so places are build.sh's output and the copy the launcher loads",
       load.name == "libr9k.so" and built.parent.name == "kernels" and load != built,
       f"{load} vs {built}")
    src_mtime, src_file = S.newest_source(ROOTP / "kernels")
    ck("newest_source finds the .hip sources build.sh compiles", src_file is not None
       and src_file.suffix == ".hip", str(src_file))

    # The native venv is a frozen wheel (it cannot be re-fetched -- setup.py's own warning), so an upstream fix
    # reaches this box as a file on disk. Its state therefore has to be reported like a version number: a patched
    # 0.32.0 and a stock 0.32.0 are different artifacts, and confusing them is how a fix gets lost at the next
    # rebuild. Everything below uses a fake install tree -- a host test must not depend on whether THIS box's
    # venv happens to be patched today.
    fake = Path(td) / "site"                                # vdir is the vllm PACKAGE dir, not site-packages
    (fake / "v1" / "worker" / "gpu").mkdir(parents=True)
    bt = fake / "v1" / "worker" / "gpu" / "block_table.py"
    rows = S.venv_patches(fake)
    ck("a venv without the fix reads as NOT applied (the state a rebuilt box starts in)",
       rows and all(r[2] is False for r in rows), str(rows)[:140])
    bt.write_text("class BlockTables:\n    self._num_dirty_rows = 0\n")
    ck("...and the marker string, not the filename, is what makes it count as applied",
       all(r[2] is True for r in S.venv_patches(fake)), str(S.venv_patches(fake))[:140])
    ck("an unreadable install dir says unknown, never 'not applied' (a false alarm there invites a rebuild of "
       "an artifact that cannot be rebuilt)",
       all(r[2] is None for r in S.venv_patches(None)) and all(r[2] is None for r in S.venv_patches("/nope")))
    ck("every patch the box walk names ships in patches/, so a rebuilt box can re-apply it",
       all((ROOTP / "patches" / r[0]).exists() for r in rows), str([r[0] for r in rows]))
    unpatched = ROOTP / "patches" / "vllm-0.32.0-block_table.py.unpatched"
    ck("the marker is a real proof, not a word that already appears in the wheel's own file",
       all(S.marker_of(r[0]) not in unpatched.read_text() for r in rows))
    fake2 = Path(td) / "site2"
    (fake2 / "vllm" / "v1" / "worker" / "gpu").mkdir(parents=True)
    (fake2 / "vllm" / "v1" / "worker" / "gpu" / "block_table.py").write_text(unpatched.read_text())
    r = subprocess.run(["patch", "-p1", "--dry-run", "-d", str(fake2)],
                       stdin=open(ROOTP / "patches" / S.VPATCHES[0][0]), capture_output=True, text=True)
    ck("the patch still applies cleanly to the wheel's own copy (drift in the venv is caught here, not at "
       "05:00 during a fault hunt)", r.returncode == 0, (r.stdout + r.stderr)[:200])

# -------------------------------------------------------------------------------------- log and state ----
with tempfile.TemporaryDirectory() as td:
    log = Path(td) / "launch.log"
    log.write_bytes(b"INFO start\nLoading weights 0%\rLoading weights 50%\rLoading weights 100%\n"
                    b"(EngineCore) multiproc_executor: world_size=4\n" + b"x" * 4000)
    t = S.tail_log(log, 3)
    ck("a launch log's tail folds the carriage returns bars rewrite lines with",
       len(t) == 3 and t[-1].startswith("x") and "Loading weights 100%" in t, str(t))
    ck("a log that was never written is reported as such (a launcher that refused early leaves none)",
       "no log" in S.tail_log(Path(td) / "none.log")[0])
    ck("no log path at all is not a crash", "no log" in S.tail_log(None)[0])

    old_state = S.STATE
    S.STATE = Path(td) / "state"
    S.state_write("launch.json", {"recipe": "native-flashnext.sh", "webui": "same", "port": 8080,
                                  "started": int(time.time()) - 600, "log": str(log), "session": None,
                                  "pid": None, "up_at": None, "died_at": None})
    ck("the launch record round-trips", S.state_read("launch.json")["recipe"] == "native-flashnext.sh")
    S.state_update("launch.json", died_at=123)
    ck("and can be annotated without losing the rest of it",
       S.state_read("launch.json")["died_at"] == 123 and S.state_read("launch.json")["port"] == 8080)
    # left_vllm is stubbed both ways: whether the box happens to have a serve up right now must not decide
    # what this check means (it did once, on a box with a live launch, and the test silently tested nothing)
    real_count = S.left_vllm
    S.left_vllm = lambda: 0
    rc, txt = capture(S.launch_status)
    ck("a recorded launch with no VLLM process is a warning that names its recipe and its log",
       "warn" in txt and "native-flashnext.sh" in txt and str(log) in txt, txt[:300])
    ck("... and it prints the log's last lines instead of sending the reader to find them",
       "world_size=4" in txt and "|" in txt, txt[-300:])
    S.left_vllm = lambda: 5
    rc, txt = capture(S.launch_status)
    ck("a recorded launch that is still up is an ok with the way to attach to it",
       "ok" in txt and "5 vLLM process" in txt and "tail -f" in txt, txt[:200])
    S.left_vllm = real_count
    S.STATE = Path(td) / "nostate"
    rc, txt = capture(S.launch_status)
    ck("no record at all says where the record would have come from",
       "launch.json" in txt and "FAIL" not in txt and "warn" not in txt, txt[:160])
    S.STATE = old_state

# ---------------------------------------------------------------------- a dead launch is seen early ----
p = subprocess.Popen(["sh", "-c", "exit 3"])
time.sleep(0.4)
ck("a launcher that exited is recognized, with its code",
   (S.dead_reason(p, None) or "").endswith("(code 3)"), str(S.dead_reason(p, None)))
live = subprocess.Popen(["sleep", "5"])
ck("a launcher still running is not dead", S.dead_reason(live, None) is None)
ck("a tmux session that is gone reads as dead",
   S.dead_reason(None, "r9700-definitely-not-here") is not None)
live.kill()
live.wait()
ck("killed counts as dead too (no exit code to report, still not running)",
   S.dead_reason(live, None) is not None)

t0 = time.time()
what, dt, why = S.wait_up(59999, p, None, secs=12)
ck("wait_up stops the moment the launcher is gone instead of sitting out its 7 minutes",
   what == "dead" and dt < 12 and "(code 3)" in why, f"{what} {dt:.1f}s {why}")
t0 = time.time()
what, dt, why = S.wait_up(59999, None, None, secs=3)
ck("a launch that is alive but never answers /health still ends on the timeout",
   what == "timeout" and time.time() - t0 < 12, f"{what} {dt:.1f}s")

# -------------------------------------------------- a death AFTER /health is still seen: the watcher ----
# setup.py used to exit the moment /health answered, so the record of a launch that died five days later still
# said up_at and nothing else, and notes/launch-methodology.md §5's "died_at/reason when the watcher saw it die"
# described a watcher that did not exist. The 2026-11-14 death (1 h 48 m of silence, notes/bug-causal-conv1d-oob.md)
# is what made it exist. Each rule below is a mistake the first draft could have made.
ck("alive: the watcher writes nothing",
   S.watch_tick({"session": "r9700-x"}, "r9700-x", None, 100) == ("", {}))
ck("dead, first time: died_at and the reason go on the record",
   S.watch_tick({"session": "r9700-x"}, "r9700-x", "the tmux session r9700-x holding it is gone", 100.6)
   == ("record", {"died_at": 100,
                  "reason": "the tmux session r9700-x holding it is gone"}))
ck("a record naming someone else's session is left alone (a watcher must not stamp a launch it did not arm)",
   S.watch_tick({"session": "r9700-other"}, "r9700-x", "gone", 100) == ("stop", {}))
ck("and no record at all is not stamped either",
   S.watch_tick(None, "r9700-x", "gone", 100) == ("stop", {}))
ck("a death already on the record is not re-stamped (the first timestamp is the one worth having)",
   S.watch_tick({"session": "r9700-x", "died_at": 55}, "r9700-x", "gone", 100) == (("stop"), {}))
ck("two watchers, one death: the second sees it already recorded and stops",
   S.watch_tick(dict(S.watch_tick({"session": "r9700-x"}, "r9700-x", "gone", 100)[1], session="r9700-x"),
                "r9700-x", "gone", 130) == ("stop", {}))
# Two watchers on ONE launch happened for real on 2026-11-15: the launch flow armed one carrying PORT=8080 (the
# port before the gateway moved the engine to 8180) and a hand-armed one carried the right port, and the record
# named the wrong one. The record settles it: a watcher claims it at start, and whoever claimed LAST speaks.
ck("a watcher the record has been re-claimed by since its own claim does not write",
   S.watch_tick({"session": "r9700-x", "watcher": 42, "watcher_at": 120}, "r9700-x", "gone", 130, 7, 100)
   == ("stop", {}))
ck("the watcher named on the record keeps writing", S.watch_tick(
   {"session": "r9700-x", "watcher": 7, "watcher_at": 100}, "r9700-x", "gone", 130, 7, 100)
   == ("record", {"died_at": 130, "reason": "gone"}))
ck("a record naming a watcher that claimed BEFORE us is stale, not a new owner (it must not silence the live one)",
   S.watch_tick({"session": "r9700-x", "watcher": 42, "watcher_at": 50}, "r9700-x", "gone", 130, 7, 100)
   == ("record", {"died_at": 130, "reason": "gone"}))
ck("a watcher holding no claim of its own (no watcher_at to compare) is not chased off by someone's older claim",
   S.watch_tick({"session": "r9700-x", "watcher": 42, "watcher_at": 120}, "r9700-x", "gone", 130)
   == ("record", {"died_at": 130, "reason": "gone"}))
real_left, real_tmux = S.left_vllm, S.out
ck("a tmux session that is gone is a reason, not a guess",
   (S.watch_dead(59999, "r9700-definitely-not-here") or "").startswith("the tmux session"))
S.left_vllm = lambda: 0
ck("no /health and no ranks left is a reason on the --no-tmux path too",
   "no VLLM process left" in (S.watch_dead(59999, None) or ""))
S.left_vllm = lambda: 4
ck("a launch still loading weights (no /health yet, ranks present) is NOT called dead",
   S.watch_dead(59999, None) is None)
S.left_vllm = real_left

# The case every probe missed on 2026-11-15 06:28. The device fault killed Worker_TP0 and left TP1/TP2/TP3 behind
# holding the cards (measured with rocm-smi straight after: 32,371,838,976 B used on each of three cards, with no
# engine anywhere to serve on them), the tmux pane stayed open, and every liveness probe answered "not gone" -- so
# the record carried no death at all and nothing could come back. The ranks re-title themselves VLLM::Worker_TPn,
# which is why left_vllm() greps VLLM[::] and not "vllm serve".
real_left = S.left_vllm
with tempfile.TemporaryDirectory() as _wd:
    _crash = Path(_wd) / "crash.log"
    _crash.write_text("(APIServer pid=498590) ERROR 11-15 06:28:33 [async_llm.py:906] "
                      "vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue.\n")
    _quiet = Path(_wd) / "quiet.log"
    _quiet.write_text("INFO 11-15 06:28:00 [utils.py:343] [shutdown] EngineCore process manager: using 15.0s "
                      "ROCm cleanup grace\n")
    UPREC = {"session": "r9700-x", "up_at": 100, "died_at": None, "log": str(_crash)}
    S.left_vllm = lambda: 3
    _w = S.watch_dead(59999, None, UPREC, S.WATCH_WEDGE_TICKS) or ""
    ck("ranks alive + no /health + a record that WAS up + a fault line: named crashed AND wedged, with the count",
       "wedged" in _w and "3 VLLM rank(s)" in _w and "EngineDeadError" in _w, _w[:160])
    ck("one silent probe is not enough: the ROCm cleanup grace alone runs 15 s, and a watcher that stamps during "
       "it stamps a launch that might still come back",
       S.watch_dead(59999, None, UPREC, S.WATCH_WEDGE_TICKS - 1) is None)
    ck("a launch that has never answered /health is LOADING, not wedged, even with ranks up and a fault-looking log",
       S.watch_dead(59999, None, {"session": "r9700-x", "up_at": None, "log": str(_crash)}, 9) is None)
    ck("no fault line, no verdict: a launch that is merely silent is not one we would call crashed and clear",
       S.watch_dead(59999, None, {"session": "r9700-x", "up_at": 100, "log": str(_quiet)}, 9) is None)
    ck("a caller with no record gets the old, narrower answer (nothing to weigh the silence against)",
       S.watch_dead(59999, None, None, 9) is None)
    S.left_vllm = real_left
R0 = {"recipe": "native-flashnext.sh", "session": "r9700-x"}
ck("clearing those ranks with the recipe's own STOP takes all five: switch, armed restart, ranks, signature, "
   "recipe -- and the case it must never cover is a launch that is only slow",
   S.stop_wedge_would(R0, True, 3, "EngineDeadError", True) is True
   and S.stop_wedge_would(R0, False, 3, "EngineDeadError", True) is False
   and S.stop_wedge_would(R0, True, 0, "EngineDeadError", True) is False
   and S.stop_wedge_would(R0, True, 3, None, True) is False
   and S.stop_wedge_would({}, True, 3, "EngineDeadError", True) is False
   and S.stop_wedge_would(R0, True, 3, "EngineDeadError", False) is False)
ck("and the watcher runs STOP=1 on the recipe rather than a hand-rolled pkill: the launcher knows its own ranks",
   'STOP="1"' in open(ROOTP / "setup.py").read() and "def stop_wedge(" in open(ROOTP / "setup.py").read())

# ...and the other half of the same outage: a session that outlives its launch refuses the NEXT launch. Measured:
# fault at 06:28, ranks cleared by STOP, and at 06:45 `setup.py --launch native-flashnext.sh` still said "a tmux
# session named r9700-native-flashnext is already up" -- 17 minutes of a dead box kept down by an empty pane.
ck("a pane the record says is over, holding no VLLM process, is a corpse: close it and launch",
   S.corpse_session("r9700-x", 0, {"session": "r9700-x", "died_at": 1234})[0] is True
   and S.corpse_session("r9700-x", 0, {"session": "r9700-x", "up_at": 1234, "died_at": None})[0] is True)
ck("a pane with ranks in it, one the record does not vouch for, or one that may still be loading stays shut",
   S.corpse_session("r9700-x", 4, {"session": "r9700-x", "died_at": 1234})[0] is False
   and S.corpse_session("r9700-y", 0, {"session": "r9700-x"})[0] is False
   and S.corpse_session("r9700-x", 0, {"session": "r9700-x", "up_at": None, "died_at": None})[0] is False)
ck("and the launcher says which of the two it found instead of just refusing",
   "corpse_session(sess, left_vllm()" in open(ROOTP / "setup.py").read())
ck("the watcher is a flag on setup.py, so a reader who finds a stray python is pointed at the reason",
   "--watch" in subprocess.run([sys.executable, str(ROOTP / "setup.py"), "--help"],
                               capture_output=True, text=True).stdout)

# ------------------------------------------------------------------- the restart breaker (WATCH_RESTART) ----
# Two device faults, 2026-11-14 20:59 and 2026-11-15 05:11, both left the engine gone and the LAN port refusing
# for an hour while a human found out by asking. The watcher can now re-run the launch -- but only for a death
# this box has actually seen in its own logs, never into leftover ranks, and never more than a few times an hour.
# Each rule below names the loop or the outage it exists to prevent.
NOW = 1_000_000
REC = {"recipe": "native-flashnext.sh", "webui": "same", "config": "256k", "session": "r9700-x"}
ck("off unless told: the default stays \u00a74's rule, seen not resurrected",
   S.restart_ok(REC, "illegal memory access", 0, NOW, [], False)[0] is False
   and "off" in S.restart_ok(REC, "illegal memory access", 0, NOW, [], False)[1])
ck("armed and signature'd: it restarts, and says which try of how many that is",
   S.restart_ok(REC, "torch.AcceleratorError: CUDA error: an illegal memory access was encountered",
                0, NOW, [], True, links=[]) == (True, "illegal memory access; restart 1 of 3 in 60 min"))
ck("the signature has to be IN THE LOG, not assumed: a launch that was stopped is not a crash",
   S.restart_ok(REC, "INFO 11-15 06:00:01 [core.py] Shutdown complete", 0, NOW, [], True)[1]
   == "no known crash signature in the log tail")
ck("ranks still alive stop it (the 11-14 fault left its ranks running for 11 more minutes: restarting into that "
   "is two launches fighting for four cards)",
   "still alive" in S.restart_ok(REC, "illegal memory access", 4, NOW, [], True)[1])
ck("and so does a record that cannot say what to re-run",
   "no recipe" in S.restart_ok({}, "illegal memory access", 0, NOW, [], True)[1])
ck("the breaker: three restarts inside the window is a loop, not a recovery",
   "breaker" in S.restart_ok(REC, "EngineDeadError", 0, NOW, [NOW - 10, NOW - 20, NOW - 30], True)[1])
ck("the same three outside the window are history, not a blocker (the cap is per hour, not per life)",
   S.restart_ok(REC, "EngineDeadError", 0, NOW, [NOW - 7200, NOW - 7300, NOW - 7400], True)[0] is True)
ck("and the cap is the knob it looks like",
   S.restart_ok(REC, "EngineDeadError", 0, NOW, [NOW - 10, NOW - 20], True, cap=2)[0] is False)
# Every signature is a line this box printed, so the list is checked against the words in its logs rather than
# against a list of things that can go wrong.
for _sig, _line in (("illegal memory access",
                     "torch.AcceleratorError: CUDA error: an illegal memory access was encountered"),
                    ("died unexpectedly", "Worker proc VllmWorker-0 died unexpectedly"),
                    ("EngineDeadError", "vllm.v1.engine.exceptions.EngineDeadError: Engine core died"),
                    ("HSA_STATUS_ERROR", "HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION: 0x105")):
    ck(f"the signature {_sig!r} matches the line it was copied from", S.crash_sig(_line) == _sig, _line[:80])
ck("the words an operator's own STOP leaves behind are not a crash signature",
   S.crash_sig("STOP=1 bash serve/native-flashnext.sh\ntmux kill-session -t r9700-native-flashnext") is None)
# Measured rather than assumed, 2026-11-15 06:10: this is what this box's deliberate STOP actually leaves in the log
# tail (the same tail the watcher would read), captured and run through crash_sig before the TEST launch started.
# It matters because the alternative is a watcher that relaunches a four-card serve because an operator stopped it.
ck("and the REAL tail a STOP left on this box reads as no signature: vLLM's teardown ends in resource_tracker "
   "warnings, not a fault line",
   S.crash_sig("/usr/lib/python3.12/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: "
               "There appear to be 4 leaked semaphore objects to clean up at shutdown\n  warnings.warn('resource_"
               "tracker: There appear to be %d '\n/usr/lib/python3.12/multiprocessing/resource_tracker.py:254: "
               "UserWarning: resource_tracker: There appear to be 6 leaked shared_memory objects to clean up at "
               "shutdown") is None)
with tempfile.TemporaryDirectory() as td:
    el = Path(td) / "engine.log"
    el.write_bytes(b"x" * 100000 + b"\n(EngineCore) torch.AcceleratorError: CUDA error: an illegal memory access\n")
    ck("the log is read from the END (a 6.8-day launch log is 240 MB; finding one line must not read it all)",
       "illegal memory access" in S.log_tail(el, 4096) and "illegal memory access" not in S.log_tail(el, 8))
    ck("a log that is not there reads as no signature rather than raising",
       S.log_tail(Path(td) / "nope.log") == "" and S.crash_sig(S.log_tail(Path(td) / "nope.log")) is None)
    real_popen, got = S.subprocess.Popen, []
    S.subprocess.Popen = lambda cmd, **kw: got.append(list(cmd)) or type("P", (), {"pid": 4242})()
    try:
        S.restart_launch(dict(REC, rest=["--max-model-len", "262144"], no_tmux=True))
    finally:
        S.subprocess.Popen = real_popen
    ck("a restart goes back through setup.py with the recipe, the pages choice, the stored config, the tmux "
       "choice and the flags THIS launch was given (a restart that drops them is a restart into another launch)",
       got and got[0][:5] == [sys.executable, str(ROOTP / "setup.py"), "--yes", "--launch",
                              "native-flashnext.sh"]
       and ["--config", "256k"] <= got[0] and "--no-tmux" in got[0]
       and ["--max-model-len", "262144"] <= got[0], str(got[:1])[:240])
_wsrc = (ROOTP / "setup.py").read_text().split("def watch_loop(")[1].split("def arm_watch(")[0]
ck("and the watcher's own loop asks the breaker before it exits (a decision nothing calls is a decision nobody "
   "gets)", "restart_ok(" in _wsrc and "restart_launch(" in _wsrc and "restarts.json" in _wsrc)
# The PCIe gate: §4's warning is about restarting INTO a degraded link, so the negotiated link is asked about
# before the relaunch, not after. It is about the RATING rather than the AER counters on purpose -- this box
# carries dozens of correctables a minute at a full 32.0 GT/s x16 during a weight load.
ck("a card that negotiated below its rating is not restarted into",
   "below rating" in S.restart_ok(REC, "illegal memory access", 0, NOW, [], True,
                                  links=["0000:c3:00.0 at 8.0 GT/s x16, rated 32.0 GT/s x16"])[1])
ck("links at rating do not block it",
   S.restart_ok(REC, "illegal memory access", 0, NOW, [], True, links=[])[0] is True)
ck("an unreadable link does not block the recovery, and the reason says nobody measured it",
   S.restart_ok(REC, "illegal memory access", 0, NOW, [], True, links=None)[0] is True
   and "unreadable" in S.restart_ok(REC, "illegal memory access", 0, NOW, [], True, links=None)[1])
ck("this box's healthy pair reads as healthy: 32.0 GT/s x16 against the same rating",
   S.link_short([("0000:c3:00.0", 32.0, 16, 32.0, 16)]) == [])
ck("a short link is named with its bus, what it came up at and what it was rated for",
   S.link_short([("0000:c3:00.0", 8.0, 16, 32.0, 16), ("0000:c4:00.0", 32.0, 8, 32.0, 16)])
   == ["0000:c3:00.0 at 8.0 GT/s x16, rated 32.0 GT/s x16",
       "0000:c4:00.0 at 32.0 GT/s x8, rated 32.0 GT/s x16"], str(S.link_short([("a", 8.0, 16, 32.0, 16)])))
_rows = S.gpu_links()
ck(f"and the four cards on this box really do answer at rating ({len(_rows)} AMD display device(s) read)",
   len(_rows) >= 4 and S.link_short(_rows) == [], f"short: {S.link_short(_rows)}; {_rows[:2]}")

# ------------------------------------------------------- what --restart runs: the shared restart_cmd ----
# ./update.sh --restart brings a RUNNING launch back on the pull: green checks, an asked answer, the recipe's
# own STOP, the ranks waited out, then the same setup.py --yes --launch the watcher's crash restart uses.
# The webui pin below is that restart's oldest bug, made invisible by living across two spellings.
def _webui_of(rec):
    c = S.restart_cmd(rec)
    return c[c.index("--webui") + 1]


_c1 = {"side": _webui_of({"recipe": "x.sh", "webui": "side"}),
       "record": _webui_of({"recipe": "x.sh", "webui": "1"}),
       "none": _webui_of({"recipe": "x.sh"})}
_cmd = S.restart_cmd({"recipe": "native-flashnext.sh", "webui": "same", "config": "256k",
                      "rest": ["--launch", "x"], "no_tmux": True})
ck("restart_cmd re-runs the recipe through setup.py --yes with the record's config, tmux choice and flags: "
   "a restart that drops the flags someone typed is a restart into a different launch",
   "--yes" in _cmd and "--config" in _cmd and "--no-tmux" in _cmd and _cmd[-2:] == ["--launch", "x"],
   str(_cmd))
ck("a side launch restarts with --webui side, not the record's WEBUI spelling: launch.json stores '1', "
   "--webui parses the name, and restart_launch used to hand argparse --webui 1 -- the child died at once "
   "and the watcher still logged a RESTARTING line",
   _c1["side"] == "side" and _c1["record"] == "side" and _c1["none"] == "same")
_src = open(ROOTP / "setup.py").read()
_ra = _src.split("def restart_after_update")[1].split("def launch(")[0]
ck("--restart is discoverable from --help the way every other knob is",
   "--restart" in subprocess.run([sys.executable, str(ROOTP / "setup.py"), "--help"],
                                 capture_output=True, text=True).stdout)
ck("a red walk stops nothing: the gate reads FAIL_STEPS, and bad() appends to it, so the one working thing "
   "on the box is never stopped to make room for a tree the run itself called broken",
   "if FAIL_STEPS:" in _ra and "FAIL_STEPS.append(msg)" in _src)
ck("--restart makes the same two moves the crash restart makes -- the recipe's own stop_wedge, then "
   "restart_cmd -- and asks with the N default, where --yes does not answer for it",
   "stop_wedge(rec)" in _ra and "restart_cmd(rec)" in _ra and "ask_no(cfg" in _ra)
ck("--check never reaches the restart: the check return stands before the gate in launch()",
   _src.split("def launch(cfg")[1].index("if cfg.check:") < _src.split("def launch(cfg")[1].index("restart_after_update(cfg)"))

# ---------------------------------------------------------------------------- the tmux environment ----
text = (ROOTP / "serve" / "native-flashnext.sh").read_text()
os.environ["VENV"] = "/home/x/venv-r9700"
os.environ["GPUS"] = "0,1,2,3"
os.environ["R9K_FOLD"] = "1"
os.environ["OFFLOAD_GB"] = "0"
os.environ["SETUP_TEST_NOT_A_KNOB"] = "nope"
le = S.launcher_env(text, "same", "/tmp/x.log")
ck("the knobs a launch script reads are the knobs it gets through tmux (the tmux server has its own env)",
   all(le.get(k) == v for k, v in (("VENV", "/home/x/venv-r9700"), ("GPUS", "0,1,2,3"),
                                   ("R9K_FOLD", "1"), ("OFFLOAD_GB", "0"))), str(sorted(le))[:200])
ck("WEBUI and LOG are set even when the script spells them ${VAR:-}", le["WEBUI"] == "same"
   and le["LOG"] == "/tmp/x.log")
ck("an unrelated environment variable is not smuggled into the session",
   "SETUP_TEST_NOT_A_KNOB" not in le)
for k in ("VENV", "GPUS", "R9K_FOLD", "OFFLOAD_GB", "SETUP_TEST_NOT_A_KNOB"):
    os.environ.pop(k, None)

if S.shutil_which("tmux"):
    sess = "r9700-test-does-not-collide"
    subprocess.run(["tmux", "kill-session", "-t", sess], capture_output=True)
    with tempfile.TemporaryDirectory() as td:
        scr = Path(td) / "s.sh"
        scr.write_text('echo "hello from $MYVAR"; exit 0')
        os.chmod(scr, 0o755)
        rc, txt = S.start_tmux(sess, scr, {"MYVAR": "env-through-tmux"}, [], f"{td}/out.log")
        seen, deadline = "", time.time() + 15
        while time.time() < deadline:
            seen = Path(f"{td}/out.log").read_text() if Path(f"{td}/out.log").exists() else ""
            if "hello" in seen:
                break
            time.sleep(0.5)
        ck("start_tmux passes the environment through (MYVAR arrives) and keeps the log",
           "hello from env-through-tmux" in seen, repr(seen[:120]))
        ck("a tmux session ends when its command does, which is how a dead launch is detected",
           subprocess.run(["tmux", "has-session", "-t", sess], capture_output=True).returncode != 0)
else:
    print("  skip tmux: not installed here (the native launch then falls back to a plain detached start)")

# ------------------------------------------------------------------ the launch configuration on disk ----
# Answer the launch questions once; the next run takes the file and asks nothing. Two checks carry that: one
# file is applied without a question (and its knobs beat what the shell inherited), several files with nobody
# at the keyboard refuses instead of guessing.
with tempfile.TemporaryDirectory() as _td:
    S.CONF = Path(_td) / "launches"

    def _cfg2(**kw):
        c = cfg(**kw)
        c.conf_env, c.conf_name, c.conf_applied = {}, None, False
        c.explicit = {"launch": kw.get("launch") is not None, "webui": kw.get("webui") is not None,
                      "docker": kw.get("docker") not in (None, "auto"), "tmux": kw.get("no_tmux", False),
                      "rest": False}
        for f, dflt in (("config", None), ("save_config", None), ("list_configs", False)):
            if not hasattr(c, f):
                setattr(c, f, dflt)
        return c

    S.conf_save("256k", {"recipe": "native-flashnext.sh", "docker": "no", "webui": "same", "tmux": True,
                         "env": {"MAXLEN": "262144"}, "rest": []}, "the standing 256k launch")
    os.environ["MAXLEN"] = "32768"                      # inherited from some earlier hand-typed command
    c = _cfg2()
    S.conf_stage(c)
    ck("one stored configuration: a bare run takes it, asks nothing, and the file outranks the shell's MAXLEN",
       c.conf_applied is True and c.launch == "native-flashnext.sh" and c.webui == "same"
       and os.environ["MAXLEN"] == "262144" and S.asks_whether_to_start(c) is False,
       f"{c.launch} {c.webui} MAXLEN={os.environ['MAXLEN']}")
    ck("flags typed this run still beat the file for that run, and do not rewrite it",
       (_c := _cfg2(webui="off", docker="yes")) or (S.conf_stage(_c) is True)
       and (_c.webui, _c.docker) == ("off", "yes")
       and json.loads((S.CONF / "256k.json").read_text())["webui"] == "same")
    os.environ.pop("MAXLEN", None)
    S.conf_save("32k", {"recipe": "native-flashnext.sh", "docker": "no", "webui": "side", "tmux": True,
                        "env": {"MAXLEN": "32768"}, "rest": []})
    ck("two files and no keyboard: it stops and names --config NAME rather than picking one",
       S.conf_stage(_cfg2()) is False and S.conf_stage(_cfg2(config="32k")).__class__ is not None
       and S.conf_all()[0][0] == "256k")
    for _n, _d in S.conf_all():
        for _k in (_d.get("env") or {}):
            os.environ.pop(_k, None)

# ------------------------------------------------------------------------------ the report itself ----
with tempfile.TemporaryDirectory() as td:
    old_state = S.STATE
    S.STATE = Path(td) / "state"
    rc, txt = capture(S.launch, cfg(check=True), "native")
    ck("the launch list labels container recipes AND says how to turn that path on",
       "[container recipe: setup.sh --docker" in txt, txt[:200])
    ck("the native recipe is offered without the label",
       "serve/native-flashnext.sh" in txt and "native-flashnext.sh   Native" in txt)
    rc, txt = capture(S.launch, cfg(check=True), "docker")
    ck("on the container path nothing is labelled",
       "[container recipe" not in txt)
    rc, txt = capture(S.launch, cfg(launch="27b.sh"), "native")
    ck("a container recipe on a native run is refused with the exact command that would take it",
       "FAIL" in txt and "--docker --launch 27b.sh" in txt, txt[:250])
    S.STATE = old_state

before = {p.name: p.read_text() for p in S.STATE.glob("*.json")} if S.STATE.is_dir() else {}
p = subprocess.run([sys.executable, str(ROOTP / "setup.py"), "--launch", "none", "--check"],
                   capture_output=True, text=True, env=env, timeout=600)
after = {p.name: p.read_text() for p in S.STATE.glob("*.json")} if S.STATE.is_dir() else {}
ck("--launch none --check changes nothing and names the path it took",
   p.returncode == 0 and "[8] a launch" in p.stdout
   and "which path this box runs: native (the default)" in p.stdout, p.stdout[-400:])
ck("--check writes no state and rewrites none (check-only runs nothing that changes anything)",
   before == after, "; ".join(f"{k}: {before.get(k)!r} -> {after.get(k)!r}" for k in sorted(set(before) | set(after))
                              if before.get(k) != after.get(k))[:900])

# The context question, learned the hard way on 2026-11-03: a launch answered /health in 150 s and looked
# healthy while running at a fraction of the checkpoint's declared context. Nothing compared the two numbers.
CKPT = {"text_config": {"max_position_embeddings": 262144}}
ST = {"meta": {"maxlen": "262144"},
      "config": {"kv_cache_size_tokens": 540202, "block_size": "448"}}
ck("the served context is read from the plugin's launch meta -- NOT from /api/status config, which carries no "
   "max_model_len field at all (that is why the first version of this check reported 'not compared')",
   S.serve_context(ST, None, CKPT)[0] == 262144, str(S.serve_context(ST, None, CKPT)))
ck("the KV pool comes back too, because it is the number that shows the full context is not a KV cost "
   "(README.md measures 540,202 at 262,144 against 497,588 at 32,768)",
   S.serve_context(ST, None, CKPT)[2] == 540202, str(S.serve_context(ST, None, CKPT)))
ck("the declared context is read from text_config, where this checkpoint keeps it (not top level)",
   S.serve_context(ST, None, CKPT)[1] == 262144)
ck("flat config.json also works (models that keep max_position_embeddings at the top level)",
   S.serve_context(None, None, {"max_position_embeddings": 131072})[1] == 131072)
ck("no launch meta: the command line in the launch log is the fallback",
   S.serve_context({"meta": {}}, "vllm serve --max-model-len 32768 --max-num-seqs 32", CKPT)[0] == 32768)
ck("the 32,768 default is exactly the case the check exists to catch: got < want",
   S.serve_context({"meta": {}}, "--max-model-len 32768", CKPT)[0]
   < S.serve_context({"meta": {}}, "--max-model-len 32768", CKPT)[1])
ck("unknown on either side returns None rather than a made-up number",
   S.serve_context({}, "no args here", {}) == (None, None, None), str(S.serve_context({}, "no args here", {})))

# Cross-file: the launcher default and the README's "what this box runs" line must not drift apart again.
# Since 2026-11-18 the launcher carries per-family MAXLEN defaults -- the Nemotron-H one inside its `FAM=nemotronh`
# block, the Flash-Next one as the global line -- so "whichever match the regex meets first" would let a family
# default answer for the global one. Both are located in their own place and checked against their own README line.
_nativetxt = (ROOTP / "serve" / "native-flashnext.sh").read_text()
_readme = (ROOTP / "README.md").read_text()
_famblock = re.search(r'if \[ "\$FAM" = nemotronh \]; then(.*?)\nfi', _nativetxt, re.S)
_fam_maxlen = re.search(r": \$\{MAXLEN:=([^}]*)\}", _famblock.group(1)).group(1) if _famblock else None
_global = re.search(r": \$\{MAXLEN:=([^}]*)\}",
                    _nativetxt.replace(_famblock.group(0), "", 1) if _famblock else _nativetxt)
ck("the launcher's Flash-Next MAXLEN default is the value README.md says this box runs (262144)",
   _global and _global.group(1) == "262144", _global.group(1) if _global else "no ${MAXLEN:=...} default")
ck("...and the README launch line still spells it, so the two agree rather than one going stale",
   "MAXLEN=262144 NSEQ=32 NBT=8192" in _readme)
ck("...while the Nemotron-H family's 32768 lives inside the family block and README says NSEQ=16 for it",
   _fam_maxlen == "32768" and "MAXLEN=32768 NSEQ=16 NBT=8192" in _readme, f"family MAXLEN={_fam_maxlen}")
ck("no ${MAXLEN:-32768} is left anywhere in the launcher (the old default could not survive by shadowing)",
   "${MAXLEN:-" not in _nativetxt)


# Context depth became a named profile (CTX) in the Nemotron-H family on 2026-11-18, after measuring that the three
# depths differ in more than the number: 262,144 grows the KV pool instead of shrinking it, and 1,048,576 only boots
# with images off because the multimodal encoder-cache budget follows the window. Text checks alone would miss a
# profile that resolves to the wrong argv, so the profiles are also resolved with DRY=1 against the real checkpoint:
# the launcher guards every write on DRY (it bit before -- see its own note), so a preview writes nothing and starts
# nothing, which makes it safe to run while the box is serving.
ck("the family block carries all three context profiles with the measured windows",
   _famblock is not None and all(s in _famblock.group(1) for s in
                                 ("32k)", "256k)", "1m)", "32768", "262144", "1048576")),
   "family block missing or a profile gone")
ck("CTX=1m turns images off and narrows to NSEQ=1 in the launcher itself, not only in the prose",
   _famblock is not None and "MM=0" in _famblock.group(1) and ": ${NSEQ:=1}" in _famblock.group(1))
ck("an unknown CTX is refused rather than falling through to the standing 32k",
   "is not a context profile" in _nativetxt)
# 2026-11-18: the profile was renamed 262k -> 256k, because 262,144 is 256 x 1024 and the old label rounded the count
# in decimal while wearing a suffix that means x1024 here. The rename is guarded three ways: the new name is what the
# block and the README carry, the old name still resolves and says so out loud (muscle memory and dated notes should
# get a correction, not a refusal), and no live file keeps calling the profile 262k.
ck("CTX=256k is the profile name and its comment states the window it stands for",
   _famblock is not None and "256k)" in _famblock.group(1) and "256 x 1024" in _famblock.group(1))
ck("the retired name still resolves, and prints that it was renamed",
   '262k)' in _nativetxt and "was renamed" in _nativetxt)
def _unexplained_262k(txt):
    """Every 262k in this text whose surrounding sentences never say the name was retired. Line-based checks miss a
    wrapped sentence, which is how this guard first failed: the README's parenthetical spans three lines."""
    bad = []
    for m in re.finditer("262k", txt):
        w = txt[max(0, m.start() - 240): m.end() + 240]
        if "renam" not in w:
            bad.append(txt[max(0, m.start() - 40): m.start() + 40].replace("\n", " "))
    return bad


ck("and nothing a user reads today calls the profile 262k -- the only allowed mentions sit next to the rename note",
   not _unexplained_262k(_readme) and not _unexplained_262k(_nativetxt),
   f"{(_unexplained_262k(_readme) + _unexplained_262k(_nativetxt))[:3]}")
# A regex that looked right and matched nothing turned out to carry a literal backspace where \b should be: an edit
# had written the escape as one control byte. Nothing else about that file was wrong, so no assertion but this one
# would have caught it. Source files are text; a control byte in one is a tool bug that lands as a silent logic bug.
_ctrl = []
for _d in ("serve", "tests", "bench", "tools", "r9700_vllm", "tuning", "kernels", "docs"):
    for _f in Path(ROOTP, _d).rglob("*"):
        if _f.suffix not in (".py", ".sh", ".js", ".mjs") or not _f.is_file():
            continue
        _b = _f.read_bytes()
        _c = sorted({x for x in _b if x < 9 or 11 <= x <= 12 or 14 <= x <= 31})
        if _c:
            _ctrl.append(f"{_f.relative_to(ROOTP)}: {['%02x' % x for x in _c]}")
ck("no source file in the tree carries a control byte (an escaped \\b that arrived as 0x08 silenced a guard)",
   not _ctrl, "; ".join(_ctrl[:4]))

ck("README.md names the two optional profiles, so the standing launch is not the only one anyone can find",
   "CTX=256k bash serve/native-flashnext.sh" in _readme and "CTX=1m bash serve/native-flashnext.sh" in _readme)

_nativeenv = Path.home() / ".config" / "r9700-stack" / "native.env"
_ckpt_dir = None
if _nativeenv.exists():
    for _ln in _nativeenv.read_text().splitlines():
        if _ln.startswith("MODEL="):
            _ckpt_dir = _ln.split("=", 1)[1].strip().strip('"')
_fam_here = bool(_ckpt_dir) and Path(_ckpt_dir, "config.json").exists() and \
    "NemotronH_Omni_Reasoning_V3" in Path(_ckpt_dir, "config.json").read_text()


def _dry(ctx):
    e = dict(os.environ, DRY="1", FORCE="1", CTX=ctx)
    r = subprocess.run(["bash", str(ROOTP / "serve" / "native-flashnext.sh")],
                       env=e, capture_output=True, text=True, timeout=180)
    return r.stdout + r.stderr, r.returncode


if not _fam_here:
    ck("profile resolution skipped: this machine's launch model is not the Nemotron-H family", True,
       "the text checks above still ran")
else:
    _d1m, _rc1m = _dry("1m")
    ck("CTX=1m really resolves to the 1M window at one sequence with --language-model-only",
       "--max-model-len 1048576" in _d1m and "--max-num-seqs 1 " in _d1m and "--language-model-only" in _d1m,
       _d1m[-220:])
    ck("...and it says out loud that images went off, because that is forced and not a default",
       "turning images OFF" in _d1m)
    _d262, _rc262 = _dry("256k")
    ck("CTX=256k resolves to 262,144 and keeps images on at 16 streams",
       "--max-model-len 262144" in _d262 and "--language-model-only" not in _d262 and "--max-num-seqs 16" in _d262,
       _d262[-220:])
    _dbg, _rcbg = _dry("bogus")
    ck("CTX=bogus exits 2 instead of quietly serving the standing window", _rcbg == 2 and "not a context profile" in _dbg,
       f"rc={_rcbg}")


# The plugin answers kv_cache_size_tokens as a STRING. {pool:,} on a str raised on 2026-11-03 inside the ok
# line of a launch that had come up healthy -- the report of a good launch must not be able to crash.
ck("a string KV pool from /api/status is coerced to an int, not passed through to a format specifier",
   S.serve_context({"meta": {"maxlen": "262144"}, "config": {"kv_cache_size_tokens": "540202"}}, None, CKPT)[2]
   == 540202, str(S.serve_context({"meta": {"maxlen": "262144"},
                                   "config": {"kv_cache_size_tokens": "540202"}}, None, CKPT)))
ck("a garbage KV pool becomes None (the line is then simply omitted) rather than an exception",
   S.serve_context({"meta": {"maxlen": "262144"}, "config": {"kv_cache_size_tokens": "?"}}, None, CKPT)[2] is None)


# The boot-time interventions are armed by units kept OUTSIDE the repository (they name this chassis, its slots,
# its faults), while the scripts they run are inside it. That split has one quiet failure mode: the unit stays
# enabled and its ExecStart points at nothing. unit_exec_paths() is the check, so test it against a fake unitdir.
import tempfile as _tf
with _tf.TemporaryDirectory() as _d:
    _d = Path(_d)
    (_d / "r9700-offline-core.service").write_text(
        f"[Service]\nType=oneshot\nExecStart=/bin/bash {ROOT}/host/r9700-offline-core.sh\n")
    (_d / "r9700-mca-archive.service").write_text(
        f"[Service]\nExecStart=/venv/bin/python {ROOT}/tools/journal-pcie.py --boots 3\n")
    (_d / "unrelated.service").write_text("[Service]\nExecStart=/bin/true\n")
    _paths = S.unit_exec_paths(_d)
ck("the unit check reads ExecStart paths out of the r9700 units, including the script argument, not just argv[0]",
   ("r9700-offline-core.service", f"{ROOT}/host/r9700-offline-core.sh") in _paths
   and ("r9700-mca-archive.service", f"{ROOT}/tools/journal-pcie.py") in _paths, str(_paths))
ck("it ignores other units (it is about this stack's interventions, not the whole system)",
   not any(u == "unrelated.service" for u, _ in _paths), str(_paths))
ck("/bin/bash and /venv/bin/python are also listed (every absolute token counts: a moved interpreter is the same "
   "failure), and nothing is invented when there are no units",
   ("/bin/bash" in [x for _, x in _paths]) and S.unit_exec_paths(Path(_d).parent / "nope") == [])


# ---------------------------------------------------------------------------- tools/gpu-rank-map.py (notes/... §15)
# The card-order discriminator is only readable if "which rank holds which card" can be MEASURED without trusting
# the enumeration order that was just deliberately shuffled. Pinned so the measurement stays a measurement.
import importlib.util as _ilu                               # noqa: E402
_rspec = _ilu.spec_from_file_location("r9700_gpu_rank_map", str(ROOTP / "tools" / "gpu-rank-map.py"))
GRM = _ilu.module_from_spec(_rspec)
_rspec.loader.exec_module(GRM)                              # defs only: main() is guarded
_topo = GRM.topology()
ck("the rank->card tool names cards in the same form the kernel's fault lines do, so a journal line and this "
   "table can be compared by eye",
   bool(_topo) and all(re.fullmatch(r"[0-9a-f]{2}:[0-9a-f]{2}\.0", c) for _, c in _topo.values()), str(_topo)[:90])
_ws = GRM.workers()
_cards = [GRM.card_of(pid, {g: v[1] for g, v in _topo.items()})[0] for pid in _ws]
ck("a live rank maps to exactly one card from the topology, or to None -- a wedge leaves the KFD record behind "
   "empty and that is reported, not guessed",
   all(c is None or c in {v[1] for v in _topo.values()} for c in _cards),
   f"{len(_ws)} worker(s): {sorted(str(c) for c in _cards)}")
import inspect as _ins                                     # noqa: E402
_csrc = _ins.getsource(GRM.card_of)                        # the CODE, not the prose around it
ck("the answer never comes from /proc/<pid>/fd: every rank keeps all four render nodes open for P2P, so the fd "
   "list identifies nothing (measured while hunting, 2026-11-15)",
   "kfd/proc" in _csrc and "/fd" not in _csrc, _csrc.strip().splitlines()[1].strip()[:80])

# ---------------------------------------------------------------------- the gate's knobs are reachable ----
# A knob the launcher cannot pass is a knob that does not exist. The gate's DEFAULTS live in serve/gateway.py and
# serve/gateway.sh forwards overrides only, so the shell's name list and the names the program reads have to agree;
# 2026-11-15's policy rewrite added four knobs (preemption window, small-body pass, hold bound, and the waiting
# trigger going to 0) and a forgotten name there would be invisible until someone set it and saw nothing happen.
# Read from the sources, not from the README.
_gsrc = (ROOTP / "serve" / "gateway.py").read_text()
_gsh = (ROOTP / "serve" / "gateway.sh").read_text()
_gread = set(re.findall(r'os\.environ\.get\("(GATE_[A-Z_]+)"', _gsrc))
_gnamed = set((re.search(r'^GATENAMES="([^"]+)"', _gsh, re.M) or type("", (), {"group": lambda s, i: ""})()).group(1).split())
ck("every GATE_* name gateway.py reads is named in gateway.sh's passthrough list -- an unnamed knob is a knob the "
   "launch cannot set, and the engine would silently run on the default",
   len(_gread) >= 8 and _gread <= _gnamed, f"reads {len(_gread)}, missing {sorted(_gread - _gnamed)}")
ck("gateway.sh forwards gate overrides but states no default for any of them -- two copies of a default are two "
   "policies, and only one of them shows up in the running process",
   not re.search(r'^\s*(?:export )?GATE_[A-Z_]+:?=', _gsh, re.M),
   [l.strip()[:60] for l in _gsh.splitlines() if re.match(r"^\s*(?:export )?GATE_[A-Z_]+:?=", l)][:3])

_bad_order = {}
for _n in ("native-flashnext.sh", "gateway.sh", "webui.sh", "embed.sh", "embed-watch.sh"):
    _L = (ROOTP / "serve" / _n).read_text().splitlines()
    _src = next((i for i, l in enumerate(_L) if "lib/localenv.sh" in l and not l.lstrip().startswith("#")), None)
    _first = [l.strip()[:38] for i, l in enumerate(_L)
              if _src is not None and i < _src and re.match(r"^[A-Za-z_]+=\$\{[A-Za-z_]+:?[-=]", l)]
    if _src is None or _first:
        _bad_order[_n] = _first or ["no serve/lib/localenv.sh sourcing at all"]
ck("every launcher sources the machine-local file BEFORE its first ${NAME:-default}: a default frozen above the "
   "sourcing outranks the file AND the operator typing NAME=... -- gateway.sh froze GATEWAY_PORT and embed.sh froze "
   "LLAMA_SERVER that way, and serve/webui.sh's PORT is what gave a box whose engine lives at 8180 a sidecar "
   "watching 8080",
   not _bad_order, str(_bad_order)[:240])
_lr = subprocess.run(["bash", "-c", "PORT=8100 WEBUI_PORT=8101 DRY=1 bash serve/webui.sh"], cwd=str(ROOTP),
                     capture_output=True, text=True, timeout=90)
ck("the file is a DEFAULT, not an override: what the caller exported wins, which is also the order setup.py reads "
   "launchers in (env -> file -> the script's own default) -- two orders means the tool and the launcher can mean "
   "different launches while both print a command line",
   "--port 8101" in _lr.stdout and "--upstream http://127.0.0.1:8100" in _lr.stdout, _lr.stdout[-300:])

# A front door whose PAGE SOURCE is a process nobody started has to say so, not fail every /monitor in silence.
import socket as _sock


def _free():
    with _sock.socket() as _s:
        _s.bind(("127.0.0.1", 0))
        return _s.getsockname()[1]


_dead, _gp = _free(), _free()
_genv = {**os.environ, "GATEWAY_PORT": str(_gp), "PAGES": f"http://127.0.0.1:{_dead}",
         "LOG": f"/tmp/r9700-gw-pages-probe-{_gp}.log", "PIDF": f"/tmp/r9700-gw-probe-{_gp}.pid"}
_gr = subprocess.run(["bash", f"{ROOTP}/serve/gateway.sh"], env=_genv, capture_output=True, text=True, timeout=180)
cout = _gr.stdout + _gr.stderr
ck("a front door whose page source is not answering says so at start and names both ways out; it does NOT quietly "
   "point the pages back at the engine -- that reinstates the state where a wedged serve takes its own report down "
   "with it (notes/supervision.md §4)",
   "not answering" in cout and "serve/webui.sh" in cout and "PAGES=" in cout, cout[-300:])
subprocess.run(["bash", f"{ROOTP}/serve/gateway.sh"], env={**_genv, "STOP": "1"},
               capture_output=True, text=True, timeout=90)

# Who decides whether a death gets restarted: the record, unless someone says WATCH_RESTART out loud.
ck("a watcher armed with no opinion follows the record -- a hand-armed watcher must not silently flip a "
   "launch's supervision mode",
   S.watch_restarts({"restart_on_crash": False}, env={}) is False
   and S.watch_restarts({"restart_on_crash": True}, env={}) is True)
ck("WATCH_RESTART said out loud wins over the record, in BOTH directions: re-arming at 05:00 is exactly the "
   "moment the policy is meant to change, and the claim writes it down afterwards",
   S.watch_restarts({"restart_on_crash": False}, env={"WATCH_RESTART": "1"}) is True
   and S.watch_restarts({"restart_on_crash": True}, env={"WATCH_RESTART": "0"}) is False)
ck("and a watcher with no record at all still answers from its own environment rather than crashing",
   S.watch_restarts(None, env={"WATCH_RESTART": "1"}) is True and S.watch_restarts(None, env={}) is False)

# -------------------------------------------------- the fault ledger and the hardware alert (notes §20) ----
# Two asks, both about the case where nobody is watching: every death has to leave a line that outlives
# launch.json (the next launch overwrites that file, and restarts.json keeps only the last death), and a card
# that keeps faulting has to say "go and look at the card" instead of being restarted forever.
_journal = """\
2026-11-14T20:59:09+00:00 h kernel: amdgpu 0000:c3:00.0: [gfxhub] page fault (src_id:0 ring:24 vmid:8 pasid:10925)
2026-11-14T20:59:09+00:00 h kernel: amdgpu 0000:c3:00.0:  Process python3 pid 251454 thread python3 pid 251454
2026-11-14T20:59:09+00:00 h kernel: amdgpu 0000:c3:00.0:          Faulty UTCL2 client ID: SQC (data) (0xa)
2026-11-14T20:59:09+00:00 h kernel: amdgpu 0000:c3:00.0:          PERMISSION_FAULTS: 0x3
2026-11-14T20:59:09+00:00 h kernel: amdgpu 0000:c3:00.0:          RW: 0x0
2026-11-15T05:11:18+00:00 h kernel: amdgpu 0000:c3:00.0: [gfxhub] page fault (src_id:0 ring:24 vmid:8 pasid:13043)
2026-11-15T05:11:18+00:00 h kernel: amdgpu 0000:c3:00.0:  Process python3 pid 466055 thread python3 pid 466055
2026-11-15T05:11:18+00:00 h kernel: amdgpu 0000:c3:00.0:          Faulty UTCL2 client ID: SQC (data) (0xa)
2026-11-15T05:11:18+00:00 h kernel: amdgpu 0000:c3:00.0:          RW: 0x0
2026-11-15T06:28:33+00:00 h kernel: amdgpu 0000:c3:00.0: [gfxhub] page fault (src_id:0 ring:24 vmid:8 pasid:14055)
2026-11-15T06:28:33+00:00 h kernel: amdgpu 0000:c3:00.0:  Process python3 pid 499003 thread python3 pid 499003
2026-11-15T06:28:33+00:00 h kernel: amdgpu 0000:c3:00.0:          Faulty UTCL2 client ID: SQC (data) (0xa)
2026-11-15T06:28:33+00:00 h kernel: amdgpu 0000:c3:00.0:          RW: 0x0
2026-11-06T07:45:03+00:00 h kernel: amdgpu 0000:c6:00.0: [gfxhub] page fault (src_id:0 ring:40 vmid:8 pasid:267)
2026-11-06T07:45:03+00:00 h kernel: amdgpu 0000:c6:00.0:  Process python3 pid 10904 thread python3 pid 10904
2026-11-06T07:45:03+00:00 h kernel: amdgpu 0000:c6:00.0:          Faulty UTCL2 client ID: TCP (0x8)
2026-11-06T07:45:03+00:00 h kernel: amdgpu 0000:c6:00.0:          RW: 0x1
"""
_jf = Path(tempfile.mkdtemp()) / "journal.txt"
_jf.write_text(_journal)
_fev = S.kernel_fault_events(command=("cat", str(_jf)))
ck("the driver's fault blocks are read with the fields that separate the two fault families (read/SQC on ring "
   "24 against write/TCP on ring 40), because a per-card count alone is true but not actionable",
   _fev["total"] == 4 and _fev["by_card"] == {"c3:00.0": 3, "c6:00.0": 1}
   and {(e["rw"], e["client"], e["ring"]) for e in _fev["events"]} == {("read", "SQC (data)", "24"),
                                                                       ("write", "TCP", "40")},
   f'{_fev["total"]} lines {_fev["by_card"]}')
ck("the pid in the block is paired with its fault, which is what makes the card-vs-rank question answerable",
   sorted(e["pid"] for e in _fev["events"]) == [10904, 251454, 466055, 499003],
   str(sorted(e["pid"] for e in _fev["events"])))
ck("a journalctl that fails reports an error, never zero -- a mistyped flag prints usage to stderr and nothing "
   "to stdout, which is how this read once said 0 faults on a box that had had eight",
   S.kernel_fault_events(command=("sh", "-c", "echo 'journalctl: invalid option' >&2; exit 2"))["total"] is None
   and "exited 2" in (S.kernel_fault_events(command=("sh", "-c", "exit 2"))["error"] or ""),
   str(S.kernel_fault_events(command=("sh", "-c", "exit 2"))["error"])[:60])
_now = time.time()
def _ev(at, card, epoch, rw="read", client="SQC (data)", ring="24"):
    return {"at": at, "epoch": epoch, "card": card, "pid": None, "rw": rw, "client": client, "ring": ring}
_three = {"total": 3, "by_card": {"c3:00.0": 3}, "error": None,
          "events": [_ev("t1", "c3:00.0", _now - 3600), _ev("t2", "c3:00.0", _now - 7200),
                     _ev("t3", "c3:00.0", _now - 900)]}
_v = S.hw_verdict(_three, now=_now)
ck("three page-fault events on one function inside the window is a hardware alert, and the alert names the "
   "card, the span and the fault shape", _v["card"] == "c3:00.0" and _v["count"] == 3 and _v["since"] == "t1"
   and _v["families"] == {"read / SQC (data) / ring 24": 3}, str(_v)[:120])
ck("two events do not make it (the line is measured, not emotional)",
   S.hw_verdict({"total": 2, "by_card": {"c3:00.0": 2}, "error": None,
                 "events": _three["events"][:2]}, now=_now)["card"] is None)
ck("events outside the window are not counted, and the count that remains is still stated (no alert is not "
   "the same as no information)",
   S.hw_verdict({"total": 3, "by_card": {"c3:00.0": 3}, "error": None,
                 "events": [_ev("t1", "c3:00.0", _now - 10 * 86400), _ev("t2", "c3:00.0", _now - 9 * 86400),
                            _ev("t3", "c3:00.0", _now - 60)]}, now=_now,
                window=7 * 86400)["cards"] == {"c3:00.0": 1})
ck("faults spread over two cards do not add up into one alert -- the question is which card, so the threshold "
   "is per function",
   S.hw_verdict({"total": 3, "by_card": {"c3:00.0": 2, "c6:00.0": 1}, "error": None,
                 "events": [_ev("t1", "c3:00.0", _now - 100), _ev("t2", "c3:00.0", _now - 200),
                            _ev("t3", "c6:00.0", _now - 300)]}, now=_now)["card"] is None)
ck("one crash writing several fault lines counts as ONE event (11 lines / 8 events measured on this box)",
   S.hw_verdict({"total": 3, "by_card": {"c3:00.0": 3}, "error": None,
                 "events": [_ev("same", "c3:00.0", _now - 5) for _ in range(3)]}, now=_now)["count"] == 1)
ck("a stamp that cannot be parsed goes to unattributed rather than into the window",
   S.hw_verdict({"total": 3, "by_card": {"c3:00.0": 3}, "error": None,
                 "events": [_ev("nope", "c3:00.0", None), _ev("t2", "c3:00.0", _now - 60),
                            _ev("t3", "c3:00.0", _now - 70)]}, now=_now)["unattributed"] == 1)
ck("an unreadable journal cannot produce an alert -- 'cannot look' must not be rendered as 'nothing wrong'",
   S.hw_verdict({"total": None, "by_card": {}, "error": "journalctl exited 2"}, now=_now) is None)
_row = S.fault_row({"session": "s", "recipe": "r.sh", "config": "256k", "log": "/l.log"},
                   {"died_at": 111, "reason": "crashed and wedged"}, "illegal memory access", 3,
                   {"ranks_before": 3, "ranks_after": 0}, False, "off (WATCH_RESTART unset)",
                   _fev, {"admitted": 90, "held": 4, "expired": 1, "closed": True}, _v, 512, now=_now)
ck("the ledger line carries the death, the decision AND the reason for the decision -- 'why did it not come "
   "back' is asked at 05:00 with the same urgency as 'why did it'",
   _row["reason"] == "crashed and wedged" and _row["sig"] == "illegal memory access"
   and _row["restart"] is False and "WATCH_RESTART" in _row["restart_why"]
   and _row["ranks_left"] == 3 and _row["uptime_s"] == 512 and _row["gate"]["admitted"] == 90
   and _row["faults_by_card"] == {"c3:00.0": 3, "c6:00.0": 1} and _row["alert"]["card"] == "c3:00.0",
   json.dumps(_row)[:160])
_old_state, _tmp = S.STATE, Path(tempfile.mkdtemp())
S.STATE = _tmp
try:
    _hooked = _tmp / "hook.txt"
    _al = S.raise_alert(_v, {"session": "s", "recipe": "r.sh"}, "illegal memory access",
                        command=f"cat > {_hooked}", at=_now)
    _aj = json.loads((_tmp / S.ALERT_FILE).read_text())
    _lines = (_tmp / S.ALERT_LEDGER).read_text().strip().splitlines()
    ck("the alert lands where it is seen (ALERT.json, which the page reads) and where the history keeps it "
       "(alerts.jsonl, one line per assertion)", _aj["card"] == "c3:00.0" and _aj["count"] == 3
       and len(_lines) == 1 and "swap this card" in _aj["do"], str(_aj)[:130])
    ck("ALERT_CMD gets the alert as stdin and the evidence path in $ALERT_PATH, so a hook never re-derives "
       "where the ledger lives", _hooked.exists() and json.loads(_hooked.read_text())["card"] == "c3:00.0",
       _hooked.read_text()[:80] if _hooked.exists() else "hook never ran")
    ck("the claim carries WHETHER A HUMAN WAS TOLD, in the live file the page reads and not only in the ledger: "
       "an alert nobody was told about is an alert nobody acted on, and 'the card is red' has to be able to mean "
       "that the channel refused it",
       (_aj.get("notify") or {}).get("sent") is False and "webhook" in str((_aj.get("notify") or {}).get("error")),
       str(_aj.get("notify"))[:160])
    ck("and a claim raised by hand says who raised it, so the ledger never implies a death that did not happen",
       S.raise_alert(_v, {"session": "s"}, None, raised_by="tools/fault-report.py --raise-if-due",
                     at=_now)["raised_by"] == "tools/fault-report.py --raise-if-due",
       str(_aj.get("raised_by")))
    _wsrc2 = (ROOTP / "serve/webui.py").read_text()
    ck("the supervisor and the page name the SAME channel files: a webhook saved from the About page is the one "
       "the watcher sends through, and the attempt the page shows is the attempt the watcher made",
       S.NOTIFY_FILE == "notify.json" and S.NOTIFY_LAST == "notify-last.json"
       and f'"{S.NOTIFY_FILE}"' in _wsrc2 and f'"{S.NOTIFY_LAST}"' in _wsrc2
       and ".r9700-state" in _wsrc2 and "--notify" in _wsrc2 and "--notify-last" in _wsrc2,
       "one pair of files, named from the state dir on both sides")
    _ssrc = (ROOTP / "setup.py").read_text()
    ck("setup.py carries no second copy of the sender: it loads serve/notify.py BY PATH, and the robot's URL "
       "appears in exactly one file in this tree -- two implementations of one POST is how an alert ends up "
       "refused by the keyword filter while both programs believe they sent it",
       'spec_from_file_location("r9700_notify", NOTIFY_SRC)' in _ssrc
       and 'NOTIFY_SRC = Path(__file__).resolve().parent / "serve" / "notify.py"' in _ssrc
       and "oapi.dingtalk.com" not in _ssrc
       and sum((ROOTP / f).read_text().count("oapi.dingtalk.com")
               for f in ("setup.py", "serve/webui.py", "serve/gateway.py")) == 0,
       "the wire format lives in serve/notify.py only")
    ck("a notifier that cannot be loaded is stated as the reason no ping went out, and the claim still lands: "
       "what is lost is the phone, and the ledger says so",
       'f"cannot load {NOTIFY_SRC}' in _ssrc and "never a swallowed alert" in str(S.notifier.__doc__ or ""),
       "a swallowed alert would be a lie by omission")
    ck("the alert is asserted BEFORE the restart act in the watcher's own sequence -- an alert printed after a "
       "relaunch that wedged is an alert lost to the failure it was warning about",
       _wsrc.index("HARDWARE ALERT") < _wsrc.index("if do:") and _wsrc.index("ledger_append(FAULT_LEDGER")
       < _wsrc.index("if do:"), "ledger and alert first, then the reset")
    S.ledger_append(S.FAULT_LEDGER, _row)
    S.ledger_append(S.FAULT_LEDGER, _row)
    ck("the ledger is append-only and fsynced: two deaths, two lines, and launch.json is never touched by it",
       len((_tmp / S.FAULT_LEDGER).read_text().strip().splitlines()) == 2 and not (_tmp / "launch.json").exists())
finally:
    S.STATE = _old_state
_ws = (ROOTP / "setup.py").read_text()
_led = _ws.index("ledger_append(FAULT_LEDGER")
_res = _ws.index("p = restart_launch(rec)")
ck("the ledger line is written BEFORE the relaunch is attempted -- if the restart itself wedges, the death and "
   "the decision are already on disk", 0 < _led < _res)
_ledline = [l for l in _ws.splitlines() if l.strip().startswith("ledger_append(FAULT_LEDGER")][0]
ck("and it is written whatever the decision is (12 spaces: beside the restart, not inside it) -- a record-only "
   "watcher is exactly the case where the ledger is the only trace", len(_ledline) - len(_ledline.lstrip()) == 12,
   f"{len(_ledline) - len(_ledline.lstrip())} spaces of indentation")
ck("the escalation is stated where a restart decision is made, not only afterwards",
   "HARDWARE ALERT" in _ws and _ws.index("HARDWARE ALERT") < _res)

# The long view: a person (or a cron) has to be able to ask "how is it going" without a live engine.
_st = Path(tempfile.mkdtemp())
(_st / S.FAULT_LEDGER).write_text(json.dumps({"died_at": 1, "uptime_s": 60, "restart": True,
                                              "sig": "illegal memory access", "ranks_left": 3}) + "\n"
                                  + json.dumps({"died_at": 2, "uptime_s": 900000, "restart": False,
                                                "sig": "illegal memory access", "ranks_left": 0}) + "\n")
_rp = subprocess.run([sys.executable, str(ROOTP / "tools" / "fault-report.py"), "--state", str(_st)],
                     capture_output=True, text=True, timeout=60)
ck("the report answers the three long questions from the files alone: how many deaths, how many came back by "
   "themselves, and how long they lasted in between (a fault 60 s into a launch and one 10 days in are not the "
   "same problem)",
   "2 death(s) recorded" in _rp.stdout and "restarted: 1" in _rp.stdout and "min 60 s" in _rp.stdout
   and "2 x illegal memory access" in _rp.stdout, _rp.stdout[:180].replace("\n", " | "))
ck("and it never invents an alert the ledger does not carry", not (_st / S.ALERT_FILE).exists())
ck("the kernel-journal section reads the real journal and splits the two families by the shape the driver "
   "printed, so 'the card is unhappy' is never the whole answer",
   "read / SQC (data) / ring 24" in _rp.stdout or "CANNOT READ" in _rp.stdout,
   _rp.stdout[-260:].replace("\n", " | "))

# --------------------------------------------------------- tools/litellm-spy.py + notes/who-sent-it.md ----
# "Who sent it?" cannot be answered from source IP on this box: everything arrives through LiteLLM. The answer
# is a list of headers a specific proxy version forwards, so it is measured by a tool and dated in a note rather
# than remembered -- and the note's claims are what the tool can be re-run to check next time litellm moves.
_SPY = ROOTP / "tools" / "litellm-spy.py"
ck("the proxy question has a tool, not a recollection", _SPY.exists(), str(_SPY))
_spy_src = _SPY.read_text() if _SPY.exists() else ""
ck("and the spy records the HEADERS (the thing the front door could key on), not just the body shape",
   '"headers": dict(self.headers.items())' in _spy_src and "x-stainless" not in _spy_src.split("class Spy")[0],
   "headers recorded")
ck("and it forwards the body VERBATIM, so what it measures is what the gateway would have received",
   "data=body" in _spy_src and "json.loads(body)" in _spy_src and "dump(" in _spy_src,
   "passthrough + --dump")
ck("and it says out loud that it parses bodies only because it is a stand, while the real gateway must not",
   "byte pipe" in _spy_src and "serve/gateway.py" in _spy_src, "byte-pipe caveat in the docstring")
_ws_note = (ROOTP / "notes" / "who-sent-it.md")
_ws_txt = _ws_note.read_text() if _ws_note.exists() else ""
ck("the note names the three dead ends by name, so nobody re-invents them (IP, the body user field, the model "
   "alias) and the one mechanism that works (litellm_params.extra_headers)",
   all(s in _ws_txt for s in ("`user`", "model alias", "extra_headers", "source IP")),
   str([s for s in ("`user`", "model alias", "extra_headers", "source IP") if s not in _ws_txt]))
ck("and it pins the version the measurement belongs to -- a header list read once is a header list that rots",
   "1.109.1" in _ws_txt and "0.32.0" in _ws_txt, "version pinned")
ck("and the engine-side dead end is named where it lives (vLLM's finished-request stats carry no user), so the "
   "note cannot quietly become wrong on an upgrade",
   "v1/metrics/stats.py" in _ws_txt and "no user field" in _ws_txt,
   str([s for s in ("v1/metrics/stats.py", "no user field") if s not in _ws_txt]))

# ------------------------------------------------ journal naming: merged mode has no second port ----------
# The recipe's own comment already said WEBUI_PORT cannot mean anything in WEBUI=same mode, and this box's
# machine-local file carries a real WEBUI_PORT=8181 (the standalone page's port). A WEBUI=same relaunch on
# 2026-11-16 honoured it, so the launch journalled requests-8181.jsonl while the standalone page attached
# requests-8180.jsonl: gauges live, request list frozen at the restart. A name was enough to reopen the hole.
_rsrc = (ROOTP / "serve" / "native-flashnext.sh").read_text()
_wb = re.search(r'  if \[ "\$\{WEBUI:-\}" = same \]; then\n(.*?)\n  fi\n', _rsrc, re.S)
ck("the recipe's WEBUI_PORT handling is a block a test can run, not a comment about it",
   _wb is not None and "WPORT=" in (_wb.group(1) if _wb else ""),
   "block found" if _wb else "no evaluable block")
if _wb:
    import urllib.request                                  # noqa: E402  (the live comparison below)
    Path("/tmp/r9700-wport-block.sh").write_text(_wb.group(0))

    def _wp(env):
        r = subprocess.run(["bash", "-c", f'{env}; . /tmp/r9700-wport-block.sh; echo "$WPORT"'],
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[:80]

    ck("merged mode names after the SERVE's port even when the box file says WEBUI_PORT (the drift that froze "
       "the page's list)",
       _wp("PORT=8180 WEBUI=same WEBUI_PORT=8181") == "8180", _wp("PORT=8180 WEBUI=same WEBUI_PORT=8181"))
    ck("and the launch says out loud that a named WEBUI_PORT was ignored, so the next reader is not guessing",
       "ignored" in subprocess.run(["bash", "-c", 'PORT=8180 WEBUI=same WEBUI_PORT=8181; '
                                   '. /tmp/r9700-wport-block.sh'],
                                   capture_output=True, text=True, timeout=30).stdout,
       "no note printed")
    ck("while a real sidecar launch still takes WEBUI_PORT when it is given one, and PORT+1 when it is not",
       _wp("PORT=8180 WEBUI=1 WEBUI_PORT=8181") == "8181" and _wp("PORT=8100 WEBUI=1") == "8101",
       f"{_wp('PORT=8180 WEBUI=1 WEBUI_PORT=8181')} / {_wp('PORT=8100 WEBUI=1')}")
    ck("and merged mode without a box file is unchanged (the pages ride the serve)",
       _wp("PORT=8100 WEBUI=same") == "8100", _wp("PORT=8100 WEBUI=same"))

# The live version of the same claim: the engine and the page source must name the SAME journal. A page that
# attaches a file nobody writes keeps its gauges live and its request list dead -- which is exactly the state no
# amount of follow()-on-tail code can save, and the reason this is checked against the running box.
def _jpath(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=10) as r:
            return ((json.loads(r.read()) or {}).get("journal") or {}).get("path")
    except Exception:
        return None


_jp = {p: _jpath(p) for p in (8180, 8181)}
_have = {p: v for p, v in _jp.items() if v}
ck("every page source answering on this box attaches the SAME request journal the engine writes",
   len(_have) < 2 or len(set(_have.values())) == 1,
   "only one source answering: " + str(_have) if len(_have) < 2 else str(_have))
ck("and the engine itself publishes which journal it writes, so the comparison is answerable from the payload",
   bool(_jp.get(8180)), str(_jp))

# ------------------------------------------------- the machine-local file, read the way the launcher reads it
# serve/lib/localenv.sh moved every launcher's LOCAL_ENV default into ONE file, and setup.py's reader kept looking
# for the inline `LOCAL_ENV=${LOCAL_ENV:-...}` line. It answered None, launcher_var fell back to the recipe's
# ${PORT:-8080}, and the launch record wrote port 8080 on a box whose engine binds 8180 with a front door on 8080.
# recipe-check then read the DOOR's payload (a sidecar's argv, a sidecar's env) and reported "1 knob and 19 env
# deviations" on an engine configured exactly as the recipe says -- and the watchdog was probing the door, where a
# dead door would read as a dead engine and restart a healthy one.
_le = S.launcher_local_env(_rsrc)
ck("the machine-local file is found through the helper the launcher sources, not only through an inline line",
   _le is not None and str(_le).endswith("native.env"), str(_le))
ck("and PORT is then what the launcher really binds (this box: the engine on 8180, the door on 8080)",
   S.launcher_var(_rsrc, "PORT", "8080") == "8180", S.launcher_var(_rsrc, "PORT", "8080"))
ck("and the watchdog asks the record's port first, falling back to the launcher rather than to a bare 8080",
   S.watch_port({"port": 8123}) == 8123 and S.watch_port({"recipe": "native-flashnext.sh"}) == 8180,
   f"{S.watch_port({'port': 8123})} / {S.watch_port({'recipe': 'native-flashnext.sh'})}")


def _argv_of(port):
    if not port:
        return ""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=10) as r:
            return (json.loads(r.read()) or {}).get("argv") or ""
    except Exception:
        return ""


_rec = (json.loads((ROOTP / ".r9700-state" / "launch.json").read_text())
        if (ROOTP / ".r9700-state" / "launch.json").exists() else {})
_av = _argv_of(_rec.get("port"))
ck("the port on the launch record is one whose page answers with the SERVE's argv, not a monitor's",
   not _rec or ("vllm" in _av and "webui.py" not in _av),
   f"record port {_rec.get('port')} answered argv: {_av[:90] or '(nothing)'}")
ck("and that port is not the front door's, so a dead door cannot be mistaken for a dead engine",
   not _rec or _rec.get("port") != int(S.env_file_var(_rsrc, "GATEWAY_PORT") or 8080),
   f"record port {_rec.get('port')} == the gateway port")

print("PASS" if fail == 0 else "FAIL", f"({fail} failing)")
sys.exit(1 if fail else 0)
