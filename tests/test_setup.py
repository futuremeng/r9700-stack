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

Nothing here touches a GPU, a card, a network port, or ~/models. tmux is exercised only if it is installed,
and it says so when it is not.

Run: ./.venv/bin/python3 tests/test_setup.py        (stdlib only)
"""
import json
import os
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
             docker="auto", no_tmux=False, rest=[])
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
   before == after, f"{sorted(before)} -> {sorted(after)}")

print("PASS" if fail == 0 else "FAIL", f"({fail} failing)")
sys.exit(1 if fail else 0)
