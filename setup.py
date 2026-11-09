#!/usr/bin/env python3
"""Bring this repository up on a box, or re-check a box that already runs it.

The box this repo lives on is a rack: R9700s passed through to a VM with their full 32 GB BAR0, the serving
stack in a pinned stock-vLLM image, the plugin bind-mounted from a SECOND checkout ($HOME/r9700-build/repo),
weights in ~/models. So "install" here is not a pip install: docker/Dockerfile pip-installs the plugin inside
the image, and the venv this script makes holds no torch at all -- the host Python is for the tests that need
no GPU, for bench/, for serve/webui.py (stdlib only) and for this file.

The default path is the native one (no docker, no root: stock vLLM wheels in a venv plus a host-built
libr9k.so), because that is what the box behind this repo's numbers runs. The container path is the one thing
that is ASKED for, with a default of no, and every later step follows that answer: with it off the image is
not inspected or built, the second checkout serve.sh mounts is not cloned, the container recipes are labelled
in the launch list rather than offered, and a checkpoint only they want is a note instead of a warning.

  ./setup.sh                    check every step, make what is ours to make, then offer a launch
  ./setup.sh --check            report only: nothing runs that changes anything, no launch offered
  ./setup.sh --yes              no prompts (builds the image and clones the build repo when the container
                                path was asked for; still no launch, and still no docker unless --docker)
  ./setup.sh --update           what update.sh runs after a git pull: pin, build repo, tests
  ./setup.sh --rebuild          build the image again even though the pin says it is current
  ./setup.sh --launch 27b       start that recipe instead of asking (none = stop after the checks)
  ./setup.sh --no-webui         launch without the Monitor/About sidecar
  ./setup.sh --docker           deploy with the pinned image. This is the ONE question asked with a default of
                                no: the native path is the default, and every later step follows that answer.
                                --yes does not answer it; --docker (or typing y) does.
  ./setup.sh --no-docker        keep the container path off without asking
  ./setup.sh --no-tmux          start a native launch detached the old way, without tmux (see launch())
  --img NAME --repo PATH        the image tag (r9700/vllm:dev) and the build repo serve.sh mounts

Every number and requirement below is read from this repository or measured on the box, with the file it came
from named. Nothing here invents a driver version: the host-side BAR/ACS/switch work is host/README.md, and
this script only checks what a running container would need from the guest.
"""
import argparse
import hashlib
import json
import os
import re
import resource
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE = ROOT / ".r9700-state"
DRM = Path("/sys/class/drm")
NATIVE_VENV = Path(os.environ.get("VENV", Path.home() / "venv-r9700"))
# WEBUI is what the launch scripts read: 1 = the sidecar on PORT+1, same = the pages inside the serve
# (serve/plugin), 0 = neither. Names kept as the scripts spell them, so setup.py cannot invent a value.
WEBUI_ENV = {"side": "1", "same": "same", "off": "0"}


# -------------------------------------------------------------------------------------------- reporting ---
def say(msg=""):
    print(msg, flush=True)


def step(n, title):
    say(f"\n[{n}] {title}")


def ok(msg):
    say(f"    ok   {msg}")


def warn(msg, fix=None):
    say(f"    warn {msg}")
    if fix:
        say(f"         -> {fix}")


def bad(msg, fix=None):
    say(f"    FAIL {msg}")
    if fix:
        say(f"         -> {fix}")


def note(msg):
    say(f"    note {msg}")


def out(cmd, timeout=120, env=None):
    """Run and capture; never raises. Returns (rc, stripped text)."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return p.returncode, (p.stdout + p.stderr).strip()
    except FileNotFoundError:
        return 127, f"{cmd[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, f"{' '.join(cmd)}: timed out"
    except Exception as e:                                              # noqa: BLE001
        return 1, f"{cmd}: {e}"


def run(cmd, cfg, timeout=None):
    """Run visibly (nothing on a rack box should happen silently). --check never runs."""
    say(f"    $ {' '.join(cmd)}")
    if cfg.check:
        return 0
    try:
        return subprocess.call(cmd, timeout=timeout)
    except Exception as e:                                              # noqa: BLE001
        say(f"    ({e})")
        return 1


def ask(cfg, question):
    if cfg.yes:
        return True
    if not sys.stdin.isatty():
        return False
    return input(f"    {question} [y/N] ").strip().lower() in ("y", "yes")


def ask_no(cfg, question):
    """A question whose default is NO, and which --yes does NOT answer.

    `ask` defaults to Yes under --yes, which is right for building an image somebody asked for and wrong for
    choosing a deployment nobody asked for: the image is a multi-GB pull whose kernels compile inside, while
    the native path is what notes/native-4xr9700.md measures and what the box behind the README's numbers
    runs. So the container path is only taken when it is asked for out loud (--docker) or typed y."""
    if cfg.yes or not sys.stdin.isatty():
        return False
    return input(f"    {question} [y/N] ").strip().lower() in ("y", "yes")


# ------------------------------------------------------------------------------------ state on this box ---
def state_read(name):
    try:
        return json.loads((STATE / name).read_text())
    except (OSError, ValueError):
        return None


def state_update(name, **kw):
    rec = state_read(name) or {}
    rec.update(kw)
    state_write(name, rec)


def state_write(name, data):
    """Record what was measured, so the NEXT run can report drift instead of existence. Never called under
    --check: a check-only run changes nothing (the directory itself is in .gitignore)."""
    STATE.mkdir(exist_ok=True)
    (STATE / name).write_text(json.dumps(data, sort_keys=True) + "\n")


# -------------------------------------------------------------------- the launch choices, on disk ---
# Why a file rather than a question every run: each invocation of setup.py re-asked the same handful of
# questions (which recipe, the pages or not, container or native, tmux or not), and on a given box those
# answers stopped changing the day the box was built. Re-asking is worse than tedious: what gets launched then
# depends on whatever the calling shell happened to carry, so a MAXLEN= exported for an experiment three days
# ago silently changes what a launch MEANS -- the same failure mode ~/.config/r9700-stack/native.env exists to
# stop ("written here instead of being whatever the last hand-typed command happened to inherit"). A stored
# configuration is that same idea for the launch itself: the out-loud choice made once, then loaded. One file
# means a bare ./setup.sh asks nothing and launches; several means it lists them and asks which; --config
# re-asks and writes a new file when the answers differ. It lives next to native.env for the same reason that
# file does: it is a fact about this box and this operator, not about the repository.
CONF = Path(os.environ.get("R9K_CONF_DIR", Path.home() / ".config" / "r9700-stack" / "launches"))
# The keys that make a configuration. Everything else in the file (its name, a note, when it was saved or last
# used) is metadata and is deliberately NOT in the fingerprint: renaming or annotating must not fork a config.
CONF_KEYS = ("recipe", "docker", "webui", "tmux", "env", "rest")


def conf_all():
    """The stored configurations as [(name, dict)] in name order. A file that does not parse, or that names no
    recipe, is skipped rather than half-applied -- a broken file must not start a half-known launch."""
    got = []
    for path in sorted(CONF.glob("*.json")):
        try:
            d = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and d.get("recipe"):
            got.append((path.stem, d))
    return got


def conf_shape(cfg):
    """This run's shape, in the form the file stores it."""
    return {"recipe": cfg.launch, "docker": cfg.docker, "webui": cfg.webui, "tmux": not cfg.no_tmux,
            "env": dict(getattr(cfg, "conf_env", None) or {}), "rest": list(getattr(cfg, "rest", None) or [])}


def conf_fingerprint(d):
    return hashlib.sha256(json.dumps({k: d.get(k) for k in CONF_KEYS}, sort_keys=True).encode()).hexdigest()[:10]


def conf_auto_name(d):
    """A name for a shape nobody named: the recipe plus the knobs that are the reason it is a new shape. The
    store is meant to be READ, so `native-flashnext-262144` and not `native-flashnext-3f9a1c2b04`."""
    env = [str(v) for k, v in sorted((d.get("env") or {}).items())]
    base = (d.get("recipe") or "launch")[:-3] if (d.get("recipe") or "").endswith(".sh") else (d.get("recipe") or "launch")
    tail = "-".join(re.sub(r"[^A-Za-z0-9.]", "", x) for x in env[:3])
    return f"{base}-{tail}" if tail else base


def conf_save(name, shape, note_txt=None):
    """Write one configuration. Never called under --check: a check-only run changes nothing, and
    tests/test_setup.py holds that line true."""
    CONF.mkdir(parents=True, exist_ok=True)
    rec = {k: shape.get(k) for k in CONF_KEYS}
    rec.update(name=name, fingerprint=conf_fingerprint(shape), saved_at=int(time.time()))
    if note_txt:
        rec["note"] = note_txt
    path = CONF / f"{name}.json"
    path.write_text(json.dumps(rec, sort_keys=True, indent=1) + "\n")
    return path


def conf_used(name):
    """Record that a configuration was the one this launch came from: the store then answers 'which one is this
    box on, and when' without anybody remembering."""
    path = CONF / f"{name}.json"
    try:
        rec = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    rec["used_at"] = int(time.time())
    path.write_text(json.dumps(rec, sort_keys=True, indent=1) + "\n")


def conf_line(name, d):
    env = " ".join(f"{k}={v}" for k, v in sorted((d.get("env") or {}).items()))
    return (f"    {name:<26} serve/{d.get('recipe')}  webui={d.get('webui')}  "
            f"{'container' if d.get('docker') == 'yes' else 'native'}"
            f"{'' if d.get('tmux', True) else '  no-tmux'}" + (f"  {env}" if env else ""))


def conf_apply(cfg, name, d):
    """Apply a stored configuration, then put this run's out-loud flags back on top of it: the file is the
    default for a bare ./setup.sh, and a flag typed this run still beats it for that run (it does not rewrite
    the file -- that is --config / --save-config, so an experiment cannot quietly become the box's config)."""
    if not cfg.explicit.get("launch"):
        cfg.launch = d.get("recipe") or cfg.launch
    if not cfg.explicit.get("docker"):
        cfg.docker = d.get("docker") or cfg.docker
    if not cfg.explicit.get("webui"):
        cfg.webui = d.get("webui") or cfg.webui
    if not cfg.explicit.get("tmux"):
        cfg.no_tmux = not d.get("tmux", True)
    if not cfg.explicit.get("rest"):
        cfg.rest = list(d.get("rest") or [])
    env = {k: str(v) for k, v in (d.get("env") or {}).items()}
    cfg.conf_env = env
    cfg.conf_name = name
    cfg.conf_applied = True
    ok(f"launch configuration from {CONF / (name + '.json')}: " + conf_line(name, d).strip()[len(name) + 1:])
    for k, v in sorted(env.items()):
        if os.environ.get(k, v) != v:
            note(f"{k}={os.environ[k]} came in with this shell; the configuration pins {k}={v} and wins "
                 f"(edit the file with --config {name}, or unset it in the shell)")
        os.environ[k] = v


def asks_whether_to_start(cfg):
    """Does this run stop and ask before the GPUs go busy? No when a stored configuration chose the launch: the
    file is that question already answered, and answering it again is what made every run a ceremony."""
    return not (cfg.yes or getattr(cfg, "conf_applied", False))


def conf_tell(cfg):
    """What a bare ./setup.sh would do, printed where the reader can see it before a launch starts."""
    store = conf_all()
    if not store:
        note("no launch configuration stored under " + str(CONF) + ": every run re-asks the launch questions "
             "until ./setup.sh --config writes one (or --save-config NAME writes it from the flags)")
        return
    say(f"    {len(store)} launch configuration(s) under {CONF}:")
    for n, d in store:
        say(conf_line(n, d))


def conf_ask(cfg, store):
    """--config with no name: ask the questions once more and write the answers. Defaults shown are the shape
    this run already resolved (stored file plus any flags typed), so reconfiguring means changing ONE answer."""
    base = conf_shape(cfg)
    names = [n for n, _ in recipes()]
    say("    reconfiguring the launch -- Enter keeps what is shown")
    cur = base["recipe"] if base["recipe"] in names else (names[0] if names else None)
    got = input(f"    launch which ({', '.join(names)}) [{cur}]? ").strip()
    recipe = got if got in names else cur
    cur = base["webui"]
    got = input(f"    the Monitor/About pages: side | same | off [{cur}]? ").strip().lower()
    webui = got if got in ("side", "same", "off") else cur
    cur = "y" if base["docker"] == "yes" else "n"
    got = input("    container path (the pinned image)? [y/N] " if cur == "n" else "    container path (the pinned image)? [Y/n] ").strip().lower()
    docker = "yes" if (got == "y" if cur == "y" else got in ("y", "yes")) else "no"
    cur = "Y" if base["tmux"] else "n"
    got = input("    start it inside a tmux session (a pane to attach to, and the session is the liveness probe)? [Y/n] ").strip().lower()
    tmux = False if (got == "n" if cur == "n" else got in ("no", "n")) else True
    cur = " ".join(f"{k}={v}" for k, v in sorted(base["env"].items()))
    got = input(f"    environment for the launcher (NAME=VALUE, space separated) [{cur or 'none'}]? ").strip()
    env = {}
    if got:
        for tok in got.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                env[k] = v
    note_txt = input("    a note saying what this one is for (Enter for none)? ").strip() or None
    shape = dict(base, recipe=recipe, docker=docker, webui=webui, tmux=tmux, env=env)
    fp = conf_fingerprint(shape)
    for n, d in store:
        if conf_fingerprint(d) == fp:
            note(f"that is already stored as {n}: nothing written")
            conf_apply(cfg, n, d)
            return
    name = input(f"    save it as [{conf_auto_name(shape)}]? ").strip() or conf_auto_name(shape)
    path = conf_save(name, shape, note_txt)
    say(f"    wrote {path}")
    others = [n for n, _ in store if conf_fingerprint(_) != fp]
    if others:
        say(f"    alongside {', '.join(others)}: a bare ./setup.sh will ask which one to take "
            f"(or take --config NAME)")
    conf_apply(cfg, name, shape)


def conf_stage(cfg):
    """Resolve the launch configuration BEFORE the checks, because a native run and a container run check
    different things -- the choice cannot wait for step 8. Returns False when the run must stop (a name that is
    not there, or a choice nobody can make without a keyboard)."""
    store = conf_all()
    if cfg.config == "*" and cfg.check:
        note("--check: --config with no name asks questions and writes a file, and a check-only run does "
             "neither -- the stored ones are reported as they are")
        conf_tell(cfg)
        return True
    if cfg.config not in (None, "*"):
        for n, d in store:
            if n == cfg.config:
                conf_apply(cfg, n, d)
                return True
        bad(f"no stored configuration named {cfg.config}",
            "the names under " + str(CONF) + ": " + (", ".join(n for n, _ in store) if store else "(none yet)"))
        return False
    if cfg.config == "*":
        if not sys.stdin.isatty():
            bad("--config asks its questions, and this is not a terminal",
                "--config NAME applies a stored one; or type the flags and record them with "
                "--save-config NAME (see --help)")
            return False
        conf_ask(cfg, store)
        return True
    if len(store) == 1:
        # applied under --check too: a native box and a container box get checked for different things, so the
        # report has to know which one this is. Nothing is written by any of it.
        conf_apply(cfg, *store[0])
    elif len(store) > 1:
        conf_tell(cfg)
        if cfg.explicit.get("launch"):
            note("--launch names the recipe out loud, so no stored configuration is applied to this run")
        elif not sys.stdin.isatty():
            bad("several configurations stored and no keyboard to choose one with",
                "./setup.sh --config NAME   (the names above)")
            return False
        else:
            got = input("    which configuration (name, or Enter to answer the questions as they come)? ").strip()
            hit = [d for n, d in store if n == got]
            if not got:
                note("nothing applied: this run asks the questions as it goes")
            elif not hit:
                bad(f"no stored configuration named {got}", "the names above")
                return False
            else:
                conf_apply(cfg, got, hit[0])
    if cfg.save_config and cfg.check:
        note(f"--check: --save-config {cfg.save_config} would write a file, so nothing is written")
    elif cfg.save_config:
        if cfg.launch in (None, "none"):
            note(f"--save-config {cfg.save_config} not written: the shape to record has no recipe in it "
                 f"(name it with --launch, or use --config)")
        else:
            shape = conf_shape(cfg)
            for n, d in store:
                if conf_fingerprint(d) == conf_fingerprint(shape):
                    note(f"what the flags say is already stored as {n}: nothing written "
                         f"(a different shape becomes a new file)")
                    break
            else:
                say(f"    wrote {conf_save(cfg.save_config, shape)}")
    return True


def sha256(p):
    """A short hash of a file, for 'did this change since the last run'. 5 MB libr9k.so, so it is cheap."""
    try:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except OSError:
        return None


def newest_source(d):
    """(mtime, path) of the newest source libr9k.so is built from -- the .hip files build.sh lists in SRCS
    plus the headers under kernels/ and its third_party tree."""
    best = (0, None)
    if not d.is_dir():
        return best
    pats = ["*.hip", "*.h", "*.hpp", "*.cc", "third_party/**/*.h", "third_party/**/*.hpp"]
    for pat in pats:
        for p in d.glob(pat):
            try:
                m = p.stat().st_mtime
            except OSError:
                continue
            if m > best[0]:
                best = (m, p)
    return best


def stale_since(so, src):
    """The source newer than the .so, or None: a .so older than its own input is a .so that keeps serving the
    old kernels, which is the failure a rebuild-and-forgot-the-copy leaves behind."""
    if src is None:
        return None
    try:
        return src if so.stat().st_mtime < src.stat().st_mtime else None
    except OSError:
        return None


def tail_log(path, n=14):
    """The last lines of a launch log with \\r folded: vLLM's progress bars rewrite a line with carriage
    returns, so a raw tail of a dead launch is one enormous line of bar. Never raises -- a launcher that
    refused early leaves no log at all, and that is worth saying."""
    if not path:
        return ["(no log was written)"]
    try:
        data = Path(path).read_text(errors="replace").replace("\r", "\n")
    except OSError:
        return [f"(no log at {path})"]
    lines = [ln.strip() for ln in data.splitlines() if ln.strip()]
    return lines[-n:] or [f"(empty log: {path})"]


def launcher_env(text, webui=None, log=None):
    """The environment a launch script actually reads, taken from its own ${NAME:-...} uses.

    tmux starts its new session in the tmux SERVER's environment, not ours: without this, VENV=, GPUS=, TP=,
    OFFLOAD_GB= and every R9K_* would silently vanish at the tmux boundary. Reading the names out of the
    script is the point -- a knob added to the script is a knob this file cannot forget."""
    names = set(re.findall(r"\$\{([A-Z][A-Z0-9_]+)", text)) | set(re.findall(r"\$([A-Z][A-Z0-9_]{2,})", text))
    env = {k: v for k, v in os.environ.items() if k in names}
    if webui is not None:
        env["WEBUI"] = webui
    if log is not None:
        env["LOG"] = log
    return env


def container_recipe(name):
    """Container-only is a property of the script. Grepping for the word "docker" got this backwards: the
    NATIVE launcher explains in a comment that it exists because "sudo docker run is not available", so it
    matched, and the container recipes -- which end by exec'ing serve.sh -- did not. Match the delegation."""
    try:
        return bool(re.search(r'/serve\.sh"', (ROOT / "serve" / name).read_text(errors="replace")))
    except OSError:
        return False


def left_vllm():
    """How many vLLM processes hold the GPUs. SIGTERM does not stop the ranks, so a launch can be dead while
    27-31 GiB a card stays allocated (serve/native-flashnext.sh's own pre-flight, measured 2026-10-31).
    pgrep -c prints 0 and exits 1 when there are none, so the number is what is parsed, not the return code."""
    txt = out(["pgrep", "-cf", "VLLM[::]"])[1].strip()
    return int(txt) if txt.isdigit() else 0


def offload_gb(cfg):
    """The --cpu-offload-gb the launch will get. The scripts take it as OFFLOAD_GB= in the environment rather
    than as an argument, so both spellings are read."""
    v = os.environ.get("OFFLOAD_GB", "0")          # conf_apply puts a stored OFFLOAD_GB into the environment
    for a in getattr(cfg, "rest", []):
        m = re.match(r"^OFFLOAD_GB=(\S+)$", a)
        if m:
            v = m.group(1)
    try:
        return float(v)
    except ValueError:
        return 0.0


def offload_refuses(gb, memlock_bytes):
    """A reason not to start an offload launch on a box that cannot pin it, or None. This box's RLIMIT_MEMLOCK
    is where PROGRESS.md's '11.7 GiB per process' comes from, which is why the 34 GB/rank expert stream there
    is a container --ulimit question: starting it anyway costs two minutes and a dead engine to find out."""
    if gb <= 0 or memlock_bytes == resource.RLIM_INFINITY or gb * 1e9 <= memlock_bytes:
        return None
    return (f"OFFLOAD_GB={gb:g} pins about {gb:g} GB of host memory per rank, and this box's "
            f"RLIMIT_MEMLOCK is {memlock_bytes // 2 ** 20} MiB")


def webui_urls(mode, port):
    """Where the Monitor/About pages are, in the spelling the launcher itself prints. Printing PORT+1
    unconditionally was wrong for WEBUI=same, whose pages sit on the serve's own port (serve/plugin)."""
    if mode == "same":
        return f"http://127.0.0.1:{port}/monitor", f"http://127.0.0.1:{port}/about"
    if mode == "side":
        return f"http://127.0.0.1:{port + 1}/monitor", f"http://127.0.0.1:{port + 1}/about"
    return None, None


def dead_reason(proc, sess):
    """None while the launch is still running, otherwise why it stopped. With tmux the session IS the launch:
    the pane closes when the pipeline ends, so has-session going false means the engine exited."""
    if proc is not None:
        rc = proc.poll()
        return None if rc is None else f"the launcher exited (code {rc})"
    if sess:
        if out(["tmux", "has-session", "-t", sess])[0] == 0:
            return None
        return f"the tmux session {sess} holding it is gone"
    return None


def wait_up(port, proc=None, sess=None, secs=420):
    """Poll /health, watching the process at the same time.

    The old loop sat for the full 7 minutes after the launch had already died and then printed one line about
    it; the box then looked healthy because nothing said otherwise. Returns (state, seconds, reason)."""
    t0 = time.time()
    while time.time() - t0 < secs:
        d = dead_reason(proc, sess)
        if d:
            return "dead", time.time() - t0, d
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return "up", time.time() - t0, None
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(5)
    return "timeout", time.time() - t0, None


def serve_context_read(port, launcher_text, log=None):
    """I/O wrapper: the running serve's own answers, plus the launcher text and its log. See serve_context."""
    st = None
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=5) as r:
            st = json.load(r)
    except (urllib.error.URLError, OSError, ValueError):
        pass
    logtxt = None
    if log:
        try:
            logtxt = Path(log).read_text(errors="replace")
        except OSError:
            pass
    ckpt = None
    m = re.search(r"MODEL:-([^}]*)\}", launcher_text)
    if m:
        try:
            ckpt = json.loads((Path(os.path.expandvars(m.group(1))) / "config.json").read_text())
        except (OSError, ValueError):
            pass
    return serve_context(st, logtxt, ckpt)


def serve_context(status, log_text, ckpt_config):
    """(what the launch actually runs at, what the checkpoint declares, what the KV pool measured) -- no I/O.

    Both sides have to be read live, because the failure mode is silent: on 2026-11-03 a launch answered
    /health in 150 s, looked healthy, and was running a fraction of the context the checkpoint declares (the
    launcher's MAXLEN default and config.json disagreed, and nothing compared them). Not every field is in every
    mode, hence three sources for the served number: the plugin's launch meta, then the command line in the log.
    "/api/status config" is NOT one of them -- it carries no max_model_len at all.
    """
    got = None
    if status:
        for v in ((status.get("meta") or {}).get("maxlen"), (status.get("config") or {}).get("max_model_len")):
            if v and str(v).isdigit():
                got = int(v)
                break
    if got is None and log_text:
        m = re.search(r"--max-model-len (\d+)", log_text)
        got = int(m.group(1)) if m else None
    want = None
    if ckpt_config:
        want = (ckpt_config.get("text_config") or ckpt_config).get("max_position_embeddings")
    pool = (status.get("config") or {}).get("kv_cache_size_tokens") if status else None
    # the plugin reports this as a STRING ("540202") -- {pool:,} on a str is a TypeError-ish ValueError, and a
    # crash here would kill the report of a launch that actually came up healthy (seen 2026-11-03)
    if pool is not None:
        try:
            pool = int(pool)
        except (TypeError, ValueError):
            pool = None
    return got, want, pool


# --------------------------------------------------------------------- facts read from this repo ------
def dockerfile_base():
    """The nightly this image is pinned to (docker/Dockerfile), so a pin change is visible."""
    try:
        m = re.search(r"^ARG BASE=(\S+)", (ROOT / "docker" / "Dockerfile").read_text(), re.M)
        return m.group(1) if m else "?"
    except OSError:
        return "?"


def recipes():
    """serve/*.sh with the first line of their own header: the measured claim is in the file, not here.

    The skip list is helpers, not launches: serve.sh is the container runner, webui.sh and webmeta.sh are the
    monitor and its recipe file, deploy.sh fetches weights. Listing webmeta.sh as a recipe (it was) offers the
    user a script that starts nothing."""
    out_ = []
    for p in sorted((ROOT / "serve").glob("*.sh")):
        if p.name in ("serve.sh", "webui.sh", "webmeta.sh", "deploy.sh"):
            continue
        head = ""
        for line in p.read_text(errors="replace").splitlines()[1:6]:
            if line.startswith("#") and len(line) > 12:
                head = line.lstrip("# ").strip()
                break
        out_.append((p.name, head))
    return out_


def wanted_models():
    """Every /models/<name> the serve scripts name -> which scripts want it, plus the HF repo id that
    serve/deploy.sh fetches it from (so a missing one says who needs it and where it comes from)."""
    names, repos, by = {}, {}, {}
    for p in sorted((ROOT / "serve").glob("*.sh")):
        text = p.read_text(errors="replace")
        for m in re.finditer(r"/models/([A-Za-z0-9._-]+)", text):
            if m.group(1) not in ("...", "x"):
                names.setdefault(m.group(1), set()).add(p.name)
        for m in re.finditer(r"hf_get\s+(\S+)/(\S+)\s+/models/(\S+)", text):
            repos[m.group(3)] = f"{m.group(1)}/{m.group(2)}"
    by.update(names)
    return by, repos


# ------------------------------------------------------------------------------------------- the steps ---
def cards():
    """amdgpu cards the guest can see: (drm name, PCI bus, driver). The BMC's AST card is not one of them."""
    found = []
    if not DRM.is_dir():
        return found
    for d in sorted(DRM.glob("card[0-9]*")):
        ue = d / "device" / "uevent"
        try:
            txt = ue.read_text()
        except OSError:
            continue
        drv = re.search(r"^DRIVER=(\S+)", txt, re.M)
        bus = re.search(r"^PCI_SLOT_NAME=(\S+)", txt, re.M)
        if drv and drv.group(1) == "amdgpu" and bus:
            found.append((d.name, bus.group(1).lower(), drv.group(1)))
    return found


def bar0_bytes(bus):
    """Size of BAR0 as the kernel actually placed it -- host/README.md: the ReBAR capability lying while
    Region 0 is absent is the whole failure mode of the second card on a PLX switch."""
    try:
        first = Path(f"/sys/bus/pci/devices/{bus}/resource").read_text().splitlines()[0].split()
        return int(first[1], 16) - int(first[0], 16) + 1
    except (OSError, ValueError, IndexError):
        return None


def switches():
    """Cards grouped by the bridge they hang off: TP wants a pair on one switch, expert streaming from host
    RAM wants one card per switch (serve/flashnext.sh measures 81/54/1379 against 114/132/2464 for that)."""
    grp = {}
    for _, bus, _ in cards():
        p = Path(f"/sys/bus/pci/devices/{bus}")
        try:
            parent = p.resolve().parent.name
        except OSError:
            parent = "?"
        grp.setdefault(parent, []).append(bus)
    return grp


def unit_exec_paths(unitdir=Path("/etc/systemd/system")):
    """Installed r9700-*.service units -> the file paths their ExecStart lines name. Pure; no printing.

    Why this is checked at all: the boot-time interventions on this box (take the failing core out of service,
    archive the replayed machine-check record) are units whose ExecStart names scripts in this tree, while the
    units themselves and the per-machine configuration deliberately live outside the repository. The failure mode
    is quiet and expensive -- someone moves a script, the unit still "succeeds", and the symptom is that the
    sick core is back in service after the next reset, which is exactly the thing those units exist to prevent.
    """
    out = []
    try:
        units = sorted(unitdir.glob("r9700-*.service"))
    except OSError:
        return out
    for u in units:
        try:
            txt = u.read_text(errors="replace")
        except OSError:
            continue
        # `\s*`, not `\s+`: systemd's syntax is `ExecStart=/path args` with NO space after the equals
        for m in re.finditer(r"^ExecStart=-?\s*(.*)$", txt, re.M):
            for tok in m.group(1).split():
                if tok.endswith((".sh", ".py")) or tok.startswith("/"):
                    out.append((u.name, tok))
    return out


def ops_units():
    """Report whether the installed boot-time units still point at files that exist."""
    named = [(u, p) for u, p in unit_exec_paths() if p.endswith((".sh", ".py"))]
    if not named:
        note("no r9700-*.service units with a script ExecStart are installed under /etc/systemd/system -- if "
             "this box is supposed to keep a core offline or archive machine-check records at boot, that "
             "intervention is not currently armed")
        return
    for unit, path in named:
        if Path(path).exists():
            ok(f"{unit}: ExecStart -> {path} exists")
        else:
            bad(f"{unit}: ExecStart -> {path} is MISSING -- the unit will run and do nothing, so the "
                f"intervention is silently disarmed (a core can come back on the next reset)",
                "re-copy the unit and its script from wherever this machine's operational files are kept, "
                "then: sudo systemctl daemon-reload && systemctl start <unit>")


def dead_core():
    """The core this box keeps offline, and whether it is actually offline right now.

    A reset puts every core back, and this box resets by itself (notes/incident-2026-11-03: 20 identical fatal
    machine checks in the 33 boots the journal holds, all reported by the two SMT threads of one core), so
    'someone offlined it once' is not a state worth trusting. host/offline-core.cpus is the pairing; the unit
    host/r9700-offline-core.service is what re-applies it at boot."""
    conf = ROOT / "host" / "offline-core.cpus"
    if not conf.exists():
        return
    cpus = []
    for ln in conf.read_text().splitlines():
        ln = ln.split("#")[0].strip()
        if ln and "=" not in ln:
            cpus = ln.split()
    if not cpus:
        return
    st = {}
    for c in cpus:
        f = Path(f"/sys/devices/system/cpu/cpu{c}/online")
        try:
            st[c] = f.read_text().strip()
        except OSError:
            st[c] = "?"
    off = [c for c in cpus if st[c] == "0"]
    if len(off) == len(cpus):
        ok(f"the failing core is out of service: cpu {' '.join(cpus)} offline "
           f"(host/offline-core.cpus, notes/incident-2026-11-03)")
    else:
        bad(f"cpu {' '.join(c for c in cpus if st.get(c) != '0')} is ONLINE again -- that is the core whose L2 "
            f"produced the fatal machine checks (state: " + ", ".join(f"cpu{c}={st[c]}" for c in cpus) + ")",
            f"sudo bash {conf.parent / 'r9700-offline-core.sh'}   and enable the unit next to it, or every "
            f"reset re-arms it (tools/journal-pcie.py shows the MCA record of each boot)")


def host(cfg):
    step(1, "the guest the launch will run in (host-side BAR/ACS work is host/README.md, not this file)")
    cs = cards()
    if not cs:
        bad("no amdgpu card visible in /sys/class/drm", "pass the cards through and load amdgpu; "
            "the 32 GB BAR0 setup and the switch chains are host/README.md")
    else:
        ok(f"{len(cs)} amdgpu card(s): " + ", ".join(f"{n} {b}" for n, b, _ in cs))
        for name, bus, _ in cs:
            sz = bar0_bytes(bus)
            if sz is None:
                warn(f"{name}: BAR0 size unreadable")
            elif sz < 32 * 1024 ** 3:
                bad(f"{name}: BAR0 placed at {sz // 1024 ** 3} GB, not 32 GB",
                    "host/r9700-barfix.sh (and r9700-chainfix for the second card on a switch): full VRAM "
                    "mapping and KFD/RCCL peer-to-peer both need the whole BAR")
            else:
                ok(f"{name}: BAR0 {sz // 1024 ** 3} GB placed")
        for dev in ("/dev/kfd",):
            (ok if os.path.exists(dev) else bad)(f"{dev} present" if os.path.exists(dev)
                                                 else f"{dev} missing (amdgpu KFD; without it there is no "
                                                 "peer-to-peer and no ROCm runtime)")
        renders = list(Path("/dev/dri").glob("renderD*")) if Path("/dev/dri").exists() else []
        if renders:
            ok(f"/dev/dri render nodes: {len(renders)}")
        else:
            bad("/dev/dri has no render node (amdgpu DRM; serve.sh passes --device=/dev/dri and group 991)")
        grp = switches()
        for br, bs in sorted(grp.items()):
            if len(bs) > 1:
                note(f"same bridge {br}: {' '.join(bs)} -- good for TP (P2P stays in the switch), and "
                     f"serve/flashnext.sh wants the opposite for host-RAM experts (GPUS=0,2: 114/132/2464 "
                     f"on the split pair against 81/54/1379 on the same-switch pair)")
    cmd = Path("/proc/cmdline").read_text()
    for want, why in (("amdgpu.ppfeaturemask", "full feature mask: the MES/GDC paths the plugin uses want it"),
                      ("iommu=pt", "HOST kernel cmdline (notes/bug-mes-invalidate-tlbs.md): with passthrough the "
                                  "guest's GPU addresses match the host's, which is what keeps P2P working")):
        if want in cmd:
            ok(f"{want} on the kernel cmdline ({why})")
        elif want == "iommu=pt":
            note(f"{want} not on THIS cmdline -- it belongs to the host, whose cmdline this is not")
        else:
            bad(f"{want} NOT on the kernel cmdline ({why})")
    hp = int(re.search(r"^HugePages_Total:\s+(\d+)", Path("/proc/meminfo").read_text(), re.M).group(1))
    note(f"HugePages_Total {hp} (the host inventory of PROGRESS.md runs vm.nr_hugepages=20000; only the host "
         f"needs it for the vfio pinning -- set it there, not here)")
    if "amdgpu.mes_log_enable=1" not in cmd:
        note("amdgpu.mes_log_enable=1 is not on this guest's cmdline; notes/mes-timeouts.md and notes/native-"
             "4xr9700.md both had to work around its absence when chasing the tail timeouts")
    _, hard = resource_limits()
    (ok if hard == resource.RLIM_INFINITY else warn)(
        f"RLIMIT_MEMLOCK here: {'unlimited' if hard == resource.RLIM_INFINITY else f'{hard // 2 ** 20} MiB'}"
        + ("" if hard == resource.RLIM_INFINITY else
           " -- pinned host memory wants unlimited (notes/tcclaviger-fork-analysis.md); serve.sh passes "
           "--ulimit memlock=-1 into the container, so a finite cap here only bites the native path "
           "(serve/native-flashnext.sh) and host-side pinning"))
    note(f"this box's cap is where PROGRESS.md's '11.7 GiB per process' comes from: offloading 34 GB/rank "
         f"of experts is a container --ulimit and vm.nr_hugepages question, not a ulimit question")
    dead_core()
    ops_units()


def resource_limits():
    return resource.getrlimit(resource.RLIMIT_MEMLOCK)

def want_docker(cfg, have_docker):
    """Step 2's decision, kept apart from its printing so it can be tested: does this box take the container
    path? Default no. --no-docker says no without asking; --docker says yes and fails loudly if there is no
    docker to say yes to."""
    if cfg.docker == "no":
        return False, "the container path is off (--no-docker)"
    if cfg.docker == "yes":
        if not have_docker:
            return False, ("--docker asked for the container path, but docker is not on PATH: the native path "
                           "is the only one this box has")
        return True, "asked for with --docker"
    if not have_docker:
        return False, ("no docker on PATH: this box is a native-path box (serve/native-flashnext.sh runs vLLM "
                       "out of a venv instead of the pinned image)")
    return (ask_no(cfg, "deploy with docker (the pinned stock-vLLM image; serve/serve.sh and the container "
                        "recipes)?"),
            "asked for at this prompt")


def deployment(cfg):
    """Step 2: which path this box takes -- native (the default) or the pinned image (docker) -- and the
    checks belonging to the one taken. serve.sh calls `sudo docker run`, so the container path also needs the
    daemon reachable: that is checked only when the container path was actually chosen."""
    step(2, "which path this box runs: native (the default) or the pinned image (docker)")
    want, why = want_docker(cfg, bool(shutil_which("docker")))
    if not want:
        note(why)
        note("native path: serve/native-flashnext.sh (stock vLLM wheels in a venv plus a host-built "
             "libr9k.so). --docker turns the container path on, --no-docker keeps it off without asking")
        native(cfg)
        return "native"
    ok(f"container path, {why}")
    rc, txt = out(["docker", "info", "--format", "{{.ServerVersion}}"])
    if rc != 0:
        rc2, txt2 = out(["sudo", "-n", "docker", "info", "--format", "{{.ServerVersion}}"])
        if rc2 == 0:
            ok(f"daemon reachable through `sudo -n docker` (server {txt2})")
        else:
            bad("docker is installed but the daemon is not reachable without a password",
                f"start it, or add yourself to the docker group (sudo usermod -aG docker $USER) so serve.sh's "
                f"`sudo docker` stops mattering: {txt.splitlines()[-1] if txt else ''}")
            return None
    else:
        ok(f"daemon reachable (server {txt})")
    return "docker"


def shutil_which(x):
    return shutil.which(x)


def native(cfg):
    """The no-docker, no-root path: serve/native-flashnext.sh. Its wheels are ~20 GB, so they are reported,
    never fetched here."""
    venv = NATIVE_VENV
    py = venv / "bin" / "python3"
    if not py.exists():
        warn(f"native path needs a venv with stock ROCm vLLM at {venv} (missing)",
             "python3 -m venv ~/venv-r9700 && ~/venv-r9700/bin/pip install vllm==0.33.0+rocm72 --extra-index-url "
             "https://wheels.vllm.ai/rocm/4ba787ec9959e566d3b587a796dff07673324da5/rocm72/ && pip install orjson "
             "&& pip install -e . (that index is per-commit and expires upstream: none of the six 0.32-era pins "
             "checked on 2026-11-02 is live -- the 0.32 wheel cannot be re-fetched at all, so the venv you have IS "
             "the artifact of record -- keep it. 0.33 renames the flavor to rocm72 and builds a rocm100 one beside "
             "it (both verified live at that pin, 2026-11-08); the docker pin is the rocm100 image, the native venvs "
             "measured here are rocm72. notes/vllm-0.33.md carries both 0.33.0 gates: the container line moved to "
             "the v0.33.0-rocm100 image, and this venv owes a paired re-measurement before it follows. The wheel "
             "build matters -- the box's own is what notes/native-4xr9700.md measures)")
        return
    # the ROCm vLLM wheels link OpenMPI, which this box has unpacked locally, not system-wide: exactly what
    # serve/native-flashnext.sh's OPENMPI= line puts on LD_LIBRARY_PATH, so check the import the same way
    mpi = Path(os.environ.get("OPENMPI", Path.home() / "opt" / "openmpi" / "usr" / "lib" / "x86_64-linux-gnu"))
    env = dict(os.environ)
    if mpi.exists():
        env["LD_LIBRARY_PATH"] = f"{mpi}{':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else ''}"
        ok(f"OpenMPI runtime for the vLLM wheels: {mpi}")
    else:
        warn(f"no OpenMPI runtime at {mpi}: the ROCm vLLM wheels cannot load (libmpi_cxx.so.40)",
             "unpack the runtime locally (no root): serve/native-flashnext.sh's OPENMPI= points at it")
    rc, txt = out([str(py), "-c", "import vllm, torch; print(vllm.__version__, torch.__version__, "
                                   "torch.version.hip)"], timeout=240, env=env)
    if rc == 0:
        ok(f"native venv: vllm/torch/hip {txt}")
    else:
        tail = [ln for ln in txt.splitlines() if ln.startswith(("OSError", "ImportError", "ModuleNotFoundError"))]
        bad(f"native venv cannot import vllm: {(tail or txt.splitlines())[-1]}",
            "the LD_LIBRARY_PATH and OPENMPI= lines of serve/native-flashnext.sh are the answer for "
            "libmpi/libhipblas errors; a GPU is not needed for this import")
    rec = libr9k(cfg)
    versions = txt if rc == 0 else None
    prev = state_read("native.json")
    if prev:
        if prev.get("versions") and versions and prev["versions"] != versions:
            note(f"the native venv changed since the last run: vllm/torch/hip {prev['versions']} -> {versions}")
        was = (prev.get("libr9k") or {}).get("loaded", {}).get("sha")
        now = rec.get("loaded", {}).get("sha")
        if was and now and was != now:
            note(f"libr9k.so changed since the last run ({was} -> {now}): the kernels a serve loaded are "
                 f"recorded in .r9700-state/native.json, which is how a regression gets attributed to a build")
    if not cfg.check:
        state_write("native.json", {"venv": str(venv), "versions": versions, "libr9k": rec,
                                    "pin": dockerfile_base(), "at": int(time.time())})


def libr9k_paths(cfg):
    """The two places libr9k.so can be. kernels/build.sh writes kernels/libr9k.so; the launchers load
    r9700_vllm/kernels/libr9k.so through R9K_LIB, and README's own line is `./build.sh && cp libr9k.so
    ../r9700_vllm/kernels/` -- so a rebuild that skipped the cp is a rebuild that never reached the serve,
    and both copies have to be reported, not just the one that happens to exist."""
    repo = Path(os.environ.get("REPO", str(ROOT)))
    return repo / "r9700_vllm" / "kernels" / "libr9k.so", ROOT / "kernels" / "libr9k.so"


def libr9k(cfg):
    """Report libr9k.so as freshness, not existence: which of the two copies is where, whether they agree,
    whether the one being loaded is older than the newest source that builds it. Returns what was measured,
    for the state file."""
    load, built = libr9k_paths(cfg)
    rec = {}
    for key, p in (("loaded", load), ("built", built)):
        if p.exists():
            rec[key] = {"path": str(p), "sha": sha256(p), "size": p.stat().st_size,
                        "mtime": int(p.stat().st_mtime)}
    if not load.exists():
        warn(f"{load} missing -- serve/native-flashnext.sh refuses without it",
             "cd kernels && HIPCFLAGS=-I$HOME/opt/rocm-dev/opt/rocm/core-10.1/include ./build.sh && "
             "cp libr9k.so ../r9700_vllm/kernels/ ($HOME/opt/rocm-dev is where this box's hipcc headers live)")
        return rec
    ok(f"libr9k.so built for the native path: {load} ({load.stat().st_size // 1024} KiB, sha "
       f"{rec['loaded']['sha']})")
    if "built" in rec and rec["built"]["sha"] != rec["loaded"]["sha"]:
        warn("the two copies of libr9k.so disagree: kernels/build.sh wrote one, the launcher loads the other",
             f"cp {built} {load} -- the serve runs the copy it loads, not the one build.sh just wrote")
    elif "built" not in rec:
        note(f"{built} is not here (that is build.sh's own output); only the copy the launchers load exists")
    src_mtime, src_file = newest_source(ROOT / "kernels")
    st = stale_since(load, src_file)
    if st:
        warn(f"libr9k.so is older than {st.relative_to(ROOT)} -- a stale .so keeps serving the old kernels "
             f"no matter what was edited since",
             "cd kernels && HIPCFLAGS=-I$HOME/opt/rocm-dev/opt/rocm/core-10.1/include ./build.sh && "
             "cp libr9k.so ../r9700_vllm/kernels/ (build.sh checks all 68 exports before it claims success)")
    return rec


def image(cfg, have="docker"):
    step(3, f"the image {cfg.img} (stock vLLM ROCm 10 nightly + the plugin; the kernels compile inside)")
    pin = dockerfile_base()
    note(f"docker/Dockerfile is pinned to {pin}")
    if have == "native":
        note("not checked and not built: this run takes the native path, which needs no image. The container "
             "recipes (serve/serve.sh, serve/27b.sh, serve/flashnext.sh) do: setup.sh --docker takes that path "
             "and this step becomes a build/inspect of the pin")
        return
    if have is None:
        warn("image not checked: docker is not usable (step 2)")
        return
    STATE.mkdir(exist_ok=True)
    marker = STATE / "image.json"
    was = json.loads(marker.read_text()) if marker.exists() else None
    rc, txt = out(["docker", "image", "inspect", "--format", "{{.Id}} {{.Created}}", cfg.img])
    if rc != 0:
        if ask(cfg, f"build {cfg.img} from docker/ (pulls {pin.split(':')[-1][:24]}..., compiles gfx1201 kernels)?"):
            rc = run(["docker", "build", "-t", cfg.img, str(ROOT / "docker")], cfg, timeout=7200)
            if rc == 0 and not cfg.check:
                marker.write_text(json.dumps({"base": pin, "built_at": int(time.time())}) + "\n")
                ok("image built; its pin recorded")
            else:
                bad("the image build failed (its output is above)")
        else:
            warn("no image and none built: serve.sh will fail until one exists")
        return
    ok(f"image present: {txt.split()[1]}")
    if was and was.get("base") != pin:
        warn("the pinned nightly has moved since this image was built "
             f"({was.get('base','?').split(':')[-1][:24]} -> {pin.split(':')[-1][:24]})",
             "./setup.sh --rebuild (the kernel build inside can change with it: gfx1201 codegen, aiter, torch)")
    elif not was and not cfg.check:
        marker.write_text(json.dumps({"base": pin, "built_at": None}) + "\n")
    # libr9k.so: serve.sh rebuilds it inside the container when it is missing or older than any .hip
    so = Path(os.environ.get("REPO", Path.home() / "r9700-build" / "repo")) / "r9700_vllm" / "kernels" / "libr9k.so"
    if so.exists():
        ok(f"libr9k.so in the build repo: {so} ({so.stat().st_size // 1024} KiB)")
    else:
        note("libr9k.so not built yet: serve.sh does it inside the image on the first launch "
             "(cd kernels && ./build.sh); the native path needs hipcc (see serve/native-flashnext.sh)")


def buildrepo(cfg, have="docker"):
    step(4, f"the build repo serve.sh mounts at /opt/r9700 ({cfg.repo})")
    p = Path(cfg.repo)
    if not p.exists():
        rc, url = out(["git", "-C", str(ROOT), "remote", "get-url", "origin"])
        if rc != 0:
            bad("this copy has no git origin to clone from", "clone the repository into the path yourself")
            return
        if have != "docker":
            note(f"{p} missing: only the container path needs it (serve.sh mounts it). The native path runs the "
                 f"plugin from this checkout (REPO={ROOT}); setup.sh --docker is what would need this one")
            return
        if cfg.check:
            note(f"would clone {url} into {p}")
            return
        if ask(cfg, f"clone {url} into {p}"):
            p.parent.mkdir(parents=True, exist_ok=True)
            if run(["git", "clone", url, str(p)], cfg) == 0:
                ok("cloned")
                note("the clone holds what has been PUSHED: work you have committed here but not pushed is not "
                     "in it. To run this checkout instead, point serve.sh at it: REPO=$PWD bash serve/flashnext.sh")
            else:
                bad("the clone failed")
                return
        else:
            warn("no build repo: serve.sh mounts it, so every launch fails without it")
            return
    rc, txt = out(["git", "-C", str(p), "rev-parse", "--short", "HEAD"])
    here = out(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"])[1]
    if rc != 0:
        bad(f"{p} is not a git checkout")
        return
    ok(f"build repo at {txt} (this checkout is {here})")
    if txt != here:
        note(f"they differ: serve.sh runs the plugin from {p}, not from this folder. To bring it here: "
             f"git -C {p} fetch && git -C {p} checkout {out(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'])[1]}"
             " (refused if it carries its own edits -- check `git -C " + str(p) + " status` first)")
    dirty = out(["git", "-C", str(p), "status", "--porcelain", "--untracked-files=no"])[1]
    if dirty:
        note(f"{p} has its own edits ({len(dirty.splitlines())} file(s)); untouched by this script")


def model_visibility(who, have):
    """warn or note for a missing checkpoint. A model that only the container-only recipes want is not a hole
    on a box that took the native path -- it is noise, and noise in a 40-line report trains people to skip
    the report. The weights are still listed, just at their real priority.

    `who` names every serve script that mentions the path, including helpers: deploy.sh FETCHES weights and
    27b.sh would run them, and only the second is a hole on this box. So the set is intersected with the
    launch recipes recipes() offers, exactly the list step 8 prints."""
    if have == "docker":
        return "warn"
    launchable = {n for n, _ in recipes()}
    return "warn" if any(not container_recipe(s) for s in who if s in launchable) else "note"


def models(cfg, have="docker"):
    step(5, f"checkpoints in {Path.home() / 'models'} (mounted at /models; third-party weights, not fetched here)")
    names, repos = wanted_models()
    d = Path.home() / "models"
    if not d.is_dir():
        warn(f"{d} does not exist", "mkdir it and put the checkpoints in it, or change the MODEL= paths")
        return
    total = 0
    for n in sorted(names):
        p = d / n
        who = ", ".join(sorted(f"serve/{s}" for s in names[n]))
        if not p.is_dir():
            src = repos.get(n)
            line = (f"/models/{n} missing (wanted by {who})"
                    + (f"; serve/deploy.sh fetches it from {src}" if src else ""))
            if model_visibility(names[n], have) == "warn":
                warn(line, "bash serve/deploy.sh (~25 GB through the image)" if src
                     else "see CREDITS.md for where the "
                     "weights come from; they are third-party and are not part of this repository")
            else:
                note(line + " -- no recipe this box can run wants it: container-only recipes, and this run "
                            "takes the native path")
            continue
        rc, txt = out(["du", "-s", "-b", str(p)])
        size = int(txt.split()[0]) if rc == 0 else 0
        total += size
        cfgjson = "config.json" if (p / "config.json").exists() else "NO config.json"
        ok(f"/models/{n}: {size / 1e9:.1f} GB, {cfgjson}, {len(list(p.glob('*.safetensors')))} safetensors "
           f"(used by {who})")
    if total:
        note(f"{total / 1e9:.1f} GB of weights present in total")
    tpl = ROOT / "serve" / "templates" / "qwen-fixed-v22.3.jinja"
    (ok if tpl.exists() else bad)(f"chat template {'present' if tpl.exists() else 'missing'}: "
                                  f"{tpl.name} (worth ~+14% speculative acceptance; CREDITS.md)")


def tests(cfg):
    step(6, "host tests that need no GPU")
    py = str(ROOT / ".venv" / "bin" / "python") if (ROOT / ".venv" / "bin" / "python").exists() else sys.executable
    if not (ROOT / ".venv" / "bin" / "python").exists():
        note("no .venv here (setup.sh makes one); falling back to " + py)
    # test_setup.py checks setup.py's own decisions (which path, what to offer, when to refuse, how fast a
    # dead launch is noticed), so it runs before the launch it would otherwise gate
    for t in ("tests/test_setup.py", "tests/test_webui.py"):
        p = ROOT / t
        if cfg.check:
            note(f"would run: {py} {t}")
        elif p.exists():
            if ask(cfg, f"run {t}?"):
                rc = run([py, str(p)], cfg, timeout=900)
                (ok if rc == 0 else bad)(f"{t} {'passed' if rc == 0 else 'FAILED'}")
            else:
                note(f"skipped: {py} {t}")
        else:
            warn(f"{t} missing")
    note("the kernel and plugin unit tests need free GPUs and run inside the image: bench/bringup-tests.sh")


def sidecar(cfg):
    step(7, "the Monitor / About pages (notes/webui.md)")
    ok("stdlib only for the sidecar; the merged plugin needs only FastAPI, which the serve already has")
    note("watch a launch that is already up:  bash serve/webui.sh          (http://127.0.0.1:8081/monitor)")
    note("start it with the launch, with this launch's recipe:  WEBUI=1 bash serve/serve.sh")
    note("it records per-request detail only in --proxy mode, which costs 2-3 ms per request, flat in prompt "
         "length (bench/webui_ab.py: +5.3% of a 76-token request, +0.5% of a 4.4k-token one); observe mode, the "
         "default, changes nothing in the inference path")
    note("OR put the pages on the serve's OWN port -- WEBUI=same, serve/plugin: /metrics from the process's own "
         "registry, and per-request queue/prefill/decode as the ENGINE measured them, at the cost of a page that "
         "dies with the launch and never sees a request body")
    py = str(NATIVE_VENV / "bin" / "python") if (NATIVE_VENV / "bin" / "python").exists() else sys.executable
    code = ("import sys; from importlib.metadata import entry_points as ep; "
            "sys.exit(0 if any(e.name == 'r9700_webui' for g in ('vllm.endpoint_plugins', "
            "'vllm.stat_logger_plugins') for e in ep(group=g)) else 1)")
    if subprocess.run([py, "-c", code], capture_output=True).returncode == 0:
        ok(f"monitor plugin installed in {py.rsplit('/python', 1)[0]}: the merged routes will load")
    else:
        warn(f"monitor plugin not installed: WEBUI=same would start the launch WITHOUT the pages. "
             f"Install with:  {py} -m pip install --no-build-isolation -e {ROOT / 'serve' / 'plugin'}")


def start_tmux(sess, script, env, rest, log):
    """The native launch inside a tmux session.

    Why a session rather than a bare detached child: the serve has to outlive the ssh session that started it by
    four minutes of loading, there is a pane to attach to while it loads, and after it is up the session IS the
    liveness probe (`tmux has-session`) -- `--no-tmux` loses all three. Not because a death was ever traced to
    the caller's tree: the 2026-11-03 launch whose log stops mid multiproc_executor at exactly 12288 bytes was
    the machine itself going down (that boot's kernel lines end 36 s after the launch, and the boot after it
    replays a machine-check record timed 05:47:12) -- see notes/launch-methodology.md §3 and tools/journal-pcie.py.
    The environment is passed with -e because the tmux server has its own (see launcher_env)."""
    cmd = f'bash {shlex.quote(str(script))} {" ".join(shlex.quote(a) for a in rest)} 2>&1 | tee -a {shlex.quote(log)}'
    args = ["tmux", "new-session", "-d", "-s", sess]
    for k in sorted(env):
        args += ["-e", f"{k}={env[k]}"]
    return out(args + [cmd], timeout=60)


def launch_status():
    """Report the launch this script last started and whether it is still up.

    A launch that dies during the load leaves no shutdown record, no /metrics and no process anyone told us
    about, so without this record the only way to find out is reading a log by hand -- which is how the 2026-11-03
    launch was found stopped mid-load (that one turned out to be the machine going down, not the launcher: see
    notes/launch-methodology.md §3). The record is written at the start of a launch, so a death is visible even
    when setup.py itself never saw it."""
    prev = state_read("launch.json")
    if not prev:
        note("no launch recorded here yet (.r9700-state/launch.json): the record is what makes 'did it die, "
             "and which log do I read' answerable on the next run")
        return None
    n = left_vllm()
    age = (time.time() - prev.get("started", 0)) / 60
    recipe, log = prev.get("recipe"), prev.get("log")
    if n:
        where = f"tmux attach -t {prev['session']}" if prev.get("session") else f"tail -f {log}"
        ok(f"the launch recorded {age:.0f} min ago (serve/{recipe}) is up: {n} vLLM process(es); {where}")
    else:
        warn(f"the launch this script started {age:.0f} min ago (serve/{recipe}, log {log}) is not running: "
             f"no VLLM process left"
             f"{'' if prev.get('died_at') is None else ', recorded dead ' + time.strftime('%H:%M:%S', time.localtime(prev['died_at']))}",
             f"its log's last lines are below; relaunch: "
             + (f"./setup.sh --config {prev['config']}" if prev.get("config")
                else f"WEBUI={prev.get('webui')} bash serve/{recipe}"))
        for ln in tail_log(log, 8):
            say(f"         | {ln}")
    return prev


def launch(cfg, have="docker"):
    step(8, "a launch")
    rs = recipes()
    for name, head in rs:
        container = container_recipe(name)
        needs = "" if not container or have == "docker" else "   [container recipe: setup.sh --docker turns that path on]"
        say(f"    serve/{name}   {head[:90]}{needs}")
    launch_status()
    if cfg.check:
        return
    pick = cfg.launch
    if pick == "none":
        return
    if pick is not None:
        # --launch names a recipe the way the list above prints it. Accept the short form but resolve it here,
        # because further down the name becomes a path (serve/<pick>) and a missing ".sh" used to arrive as a
        # FileNotFoundError traceback out of pathlib -- after the line saying what it was about to run (2026-11-08).
        names = [n for n, _ in rs]
        if pick.endswith(".sh") and pick in names:
            pass
        elif pick + ".sh" in names:
            pick += ".sh"
        else:
            bad(f"no serve/{pick}", "--launch takes a name from the list above, e.g. --launch "
                + (names[0] if names else "native-flashnext.sh"))
            return
        cfg.launch = pick
    if pick is None:
        if not sys.stdin.isatty():
            note("not a terminal: not starting anything")
            return
        names = [n for n, _ in rs]
        pick = input(f"    start which ({', '.join(names)} or Enter for none)? ").strip() or None
        if pick is None:
            return
        if pick not in names:
            bad(f"no serve/{pick}")
            return
    if container_recipe(pick) and have != "docker":
        bad(f"serve/{pick} runs through serve.sh, and this run takes the native path",
            "./setup.sh --docker --launch " + pick + "   (or run a native recipe: serve/native-flashnext.sh)")
        return
    env = dict(os.environ, WEBUI=WEBUI_ENV[cfg.webui])
    port = int(os.environ.get("PORT", 8080))
    log = os.environ.get("LOG") or str(Path.home() / "logs" / f"setup-{time.strftime('%H%M%S')}.log")
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    # pre-flight in the launcher's own shape: refuse here rather than start a launch that refuses two minutes
    # in (its LEFT check and the pinned-memory ceiling are the two ways a native launch fails outright)
    if pick.startswith("native"):
        n = left_vllm()
        if n and os.environ.get("FORCE", "0") != "1":
            bad(f"refusing to launch: {n} vLLM process(es) still holding the GPUs -- the ranks ignore SIGTERM, "
                f"so a stopped launch still owns 27-31 GiB a card",
                f"STOP=1 bash serve/{pick}   (stops them and exits); FORCE=1 launches anyway")
            return
        why = offload_refuses(offload_gb(cfg), resource_limits()[1])
        if why:
            bad(why, "this is what a container's --ulimit memlock=-1 buys (./setup.sh --docker), or raise the "
                     "cap for the launching shell, or lower OFFLOAD_GB "
                     "(notes/tcclaviger-fork-analysis.md, PROGRESS.md's '11.7 GiB per process')")
            return
    say(f"    $ WEBUI={WEBUI_ENV[cfg.webui]} LOG={log} bash serve/{pick}")
    # A stored configuration IS the out-loud choice, made once, so it does not get asked again: that is the
    # point of the file. --yes keeps its meaning for a run that answered the questions live.
    if asks_whether_to_start(cfg) and not ask(cfg, f"start serve/{pick} now (GPUs go busy for ~75 s)?"):
        return
    proc, sess = None, None
    if pick.startswith("native"):
        script = ROOT / "serve" / pick
        if not cfg.no_tmux and shutil_which("tmux"):
            sess = "r9700-" + pick[:-3]
            if out(["tmux", "has-session", "-t", sess])[0] == 0:
                bad(f"a tmux session named {sess} is already up -- not starting a second launch into it",
                    f"tmux attach -t {sess}   (or ./setup.sh --no-tmux for a plain detached start)")
                return
            rc, txt = start_tmux(sess, script, launcher_env(script.read_text(errors="replace"),
                                                            WEBUI_ENV[cfg.webui], log), cfg.rest, log)
            if rc != 0:
                bad(f"tmux new-session failed: {txt}", "./setup.sh --no-tmux falls back to a plain detached start")
                return
            say(f"    tmux session {sess} -- watch it with: tmux attach -t {sess}   (detach: Ctrl-b d)")
        else:
            with open(log, "ab") as fh:
                proc = subprocess.Popen(["bash", str(script), *cfg.rest], stdout=fh, stderr=subprocess.STDOUT,
                                        env=env, start_new_session=True)
            note("no tmux here (or --no-tmux): detached only -- no pane to attach to while the engine loads, and "
                 "no session left to ask whether the launch is still up (notes/launch-methodology.md)")
    else:
        subprocess.call(["bash", str(ROOT / "serve" / pick), *cfg.rest], env=env)
    # the configuration the launch came from is part of the record: a death should be re-launchable by name
    if not cfg.check:
        shape, fp = conf_shape(cfg), conf_fingerprint(conf_shape(cfg))
        named = getattr(cfg, "conf_name", None)
        if named:
            # The file this launch came from gets a timestamp, so the store answers "which configuration is this
            # box on, and when was it last taken" without anybody remembering. Flags typed on top of it are NOT
            # written back: an experiment must not become tomorrow's default.
            conf_used(named)
        elif not any(conf_fingerprint(d) == fp for _, d in conf_all()):
            # First launch of a shape nobody wrote down: that is the choice worth keeping -- rule 1 of the file,
            # the parameters chosen the first time get saved so the next run does not have to be told again.
            nm = conf_auto_name(shape)
            say(f"    recorded as a new configuration: {conf_save(nm, shape)} -- the next run takes it without "
                f"asking (./setup.sh --config {nm} names it, ./setup.sh --config re-asks)")
            cfg.conf_name = nm
    state_write("launch.json", {"recipe": pick, "webui": WEBUI_ENV[cfg.webui], "port": port, "path": have,
                                "started": int(time.time()), "log": log, "session": sess,
                                "pid": proc.pid if proc else None, "up_at": None, "died_at": None,
                                "config": getattr(cfg, "conf_name", None)})
    say(f"    waiting for http://127.0.0.1:{port}/health ...")
    what, dt, why = wait_up(port, proc, sess)
    stopper = f"STOP=1 bash serve/{pick}"
    if what == "up":
        ok(f"up after {dt:.0f} s")
        state_update("launch.json", up_at=int(time.time()))
        got, want, pool = serve_context_read(port, (ROOT / "serve" / pick).read_text(), log)
        if got and want:
            if got >= want:
                ok(f"context: the launch runs at {got:,} tokens, the checkpoint declares {want:,}"
                   + (f", KV pool {pool:,} tokens" if pool else ""))
            else:
                bad(f"the launch runs at {got:,} tokens of context but {pick}'s checkpoint declares {want:,}: "
                    f"requests longer than {got:,} are refused -- and this box's ledger is not a short-prompt "
                    f"ledger (README.md, notes/native-4xr9700.md)",
                    f"{stopper}, then MAXLEN={want} bash serve/{pick}")
        else:
            note("context not compared: no /api/status answer, or no MODEL= default in the launcher to read "
                 "config.json from")
        mon, about = webui_urls(cfg.webui, port)
        if mon:
            say(f"    monitor: {mon}    about: {about}"
                + ("   (inside the serve itself: serve/plugin)" if cfg.webui == "same" else ""))
    elif what == "dead":
        bad(f"{why} after {dt:.0f} s without ever answering /health. {log}, last lines:")
        for ln in tail_log(log, 14):
            say(f"         | {ln}")
        state_update("launch.json", died_at=int(time.time()), reason=why)
        say(f"    the ranks ignore SIGTERM: clear anything left with  {stopper}")
        say("    then relaunch:  " + (f"./setup.sh --config {cfg.conf_name}" if getattr(cfg, "conf_name", None)
                                      else f"WEBUI={WEBUI_ENV[cfg.webui]} bash serve/{pick}"))
        return
    else:
        warn(f"not answering /health within {dt / 60:.0f} min: {log}, last lines:")
        for ln in tail_log(log, 10):
            say(f"         | {ln}")
        say(f"    it may still be coming up: the log is the place to look, and {stopper} stops it")
        return
    note("the first launch of a NEW configuration compiles and keeps ~0.45 GiB that the next one hands to the "
         "KV pool (serve/flashnext.sh): restart once, then measure")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report only; run nothing that changes anything")
    ap.add_argument("--yes", action="store_true", help="no prompts -- except the docker question, whose "
                                                      "default is no and which only --docker answers")
    ap.add_argument("--update", action="store_true", help="what update.sh runs after a git pull")
    ap.add_argument("--rebuild", action="store_true", help="build the image even if the pin matches")
    ap.add_argument("--launch", default=None, help="serve/<name>.sh to start, or none")
    ap.add_argument("--no-webui", action="store_true", help="launch without the Monitor/About sidecar")
    ap.add_argument("--webui", choices=("side", "same", "off"), default=None,
                    help="side: the sidecar on PORT+1 (default). same: the pages INSIDE the serve, on its "
                         "own port, via serve/plugin (needs: pip install -e serve/plugin). off: neither.")
    ap.add_argument("--img", default=os.environ.get("IMG", "r9700/vllm:dev"))
    ap.add_argument("--repo", default=os.environ.get("REPO", str(Path.home() / "r9700-build" / "repo")))
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--docker", dest="docker", action="store_const", const="yes", default="auto",
                     help="deploy with the pinned image (the container path). Asked for, never assumed: the "
                          "native path is the default and --yes does not answer this one.")
    grp.add_argument("--no-docker", dest="docker", action="store_const", const="no",
                     help="keep the container path off without asking (the default answer, said out loud)")
    ap.add_argument("--config", nargs="?", const="*", default=None,
                    help="with a name: launch that stored configuration. with no name: re-ask the launch "
                         "questions and write the answers (a shape that differs from what is stored becomes a "
                         "new file). The files live under ~/.config/r9700-stack/launches, and a bare ./setup.sh "
                         "applies the one it finds -- no questions when there is exactly one, a list when "
                         "there are several.")
    ap.add_argument("--save-config", default=None, metavar="NAME",
                    help="record what the flags say as a configuration named NAME, without asking anything: the "
                         "way to set the box's config from a script")
    ap.add_argument("--list-configs", action="store_true", help="print the stored launch configurations, stop")
    ap.add_argument("--no-tmux", action="store_true",
                    help="start a native launch detached without tmux: no pane to watch while it loads and no "
                         "session to ask afterwards whether the launch is still up")
    cfg, extra = ap.parse_known_args()
    cfg.rest = extra
    cfg.conf_env, cfg.conf_name, cfg.conf_applied = {}, None, False
    # Which answers this run typed out loud. A stored configuration supplies the defaults of a bare run; a flag
    # said this run beats it for that run, and beats it WITHOUT rewriting the file, so an experiment with
    # hand-typed flags cannot quietly become what the box boots into tomorrow.
    cfg.explicit = {"launch": cfg.launch is not None, "webui": cfg.webui is not None or cfg.no_webui,
                    "docker": cfg.docker != "auto", "tmux": cfg.no_tmux, "rest": bool(extra)}
    cfg.webui = cfg.webui or ("off" if cfg.no_webui else "side")
    if cfg.list_configs:                     # the store alone: no box walk, no questions
        conf_tell(cfg)
        return
    if not conf_stage(cfg):
        sys.exit(2)
    say(f"r9700-stack bring-up -- {ROOT}")
    say(f"check-only: {cfg.check}   (every requirement below names the file it came from)")
    host(cfg)
    have = deployment(cfg)
    if cfg.rebuild and have == "docker" and not cfg.check:
        out(["docker", "rmi", cfg.img])
        note(f"removed {cfg.img}: the image step will build it again")
    elif cfg.rebuild and have != "docker":
        note("--rebuild asked for, but this run takes the native path: nothing was removed or built "
             "(./setup.sh --docker --rebuild is the image)")
    image(cfg, have)
    buildrepo(cfg, have)
    models(cfg, have)
    tests(cfg)
    sidecar(cfg)
    if cfg.update:
        step("u", "what an update does NOT touch: the weights in ~/models, the caches "
                  "(~/vllmstock-cache-*, ~/vllm-native-*), and any launch that is running")
    launch(cfg, have)
    say("\nnotes: README.md (build/serve/check), PROGRESS.md (what was measured and rejected), "
        "host/README.md (BAR0/ACS/switch chains), notes/webui.md (the pages).")


if __name__ == "__main__":
    main()
