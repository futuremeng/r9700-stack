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
  ./setup.sh --embedding        the CPU embedding sidecar: serve/embed.sh on :8090, llama.cpp built in
                                $HOME when missing (mmap GGUF only -- see the step and serve/embed.sh)
  (no flag)                     if this box's machine-local file sets GATEWAY_PORT, the launch also brings up
                                serve/gateway.sh on that port once /health answers: one LAN port routing to the
                                engine plus the CPU sidecars, with the engine itself on loopback. The launcher
                                cannot do this itself -- it ends in `exec vllm serve` -- so setup.py does.
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
import importlib.util
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
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE = ROOT / ".r9700-state"
DRM = Path("/sys/class/drm")
NATIVE_VENV = Path(os.environ.get("VENV", Path.home() / "venv-r9700"))
# WEBUI is what the launch scripts read: 1 = the sidecar on PORT+1, same = the pages inside the serve
# (serve/plugin), 0 = neither. Names kept as the scripts spell them, so setup.py cannot invent a value.
WEBUI_ENV = {"side": "1", "same": "same", "off": "0"}
# and back: what launch.json RECORDS is the WEBUI spelling and what --webui PARSES is the name, so the
# record's "1" must become "side" before it goes into a restart command -- side launches used to die at
# argparse with invalid choice '1'
WEBUI_NAME = {value: name for name, value in WEBUI_ENV.items()}


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


FAIL_STEPS = []      # what said FAIL this run: --restart refuses to stop a working box on their word


def bad(msg, fix=None):
    say(f"    FAIL {msg}")
    FAIL_STEPS.append(msg)
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


LOCAL_ENV_HELPER = ROOT / "serve" / "lib" / "localenv.sh"      # where every launcher's LOCAL_ENV default lives


def launcher_local_env(text):
    """The machine-local file a launcher sources.

    It used to be found by one regex for the inline line `LOCAL_ENV=${LOCAL_ENV:-...}`. On 2026-11-15 that line
    moved out of every launcher into ONE file (serve/lib/localenv.sh, which defaults it with
    `: "${LOCAL_ENV:=$HOME/.config/r9700-stack/native.env}"`), and this reader kept looking for the inline form:
    it answered None, `launcher_var` fell back to the launcher's `${PORT:-8080}`, and the launch record wrote
    port 8080 on a box whose engine binds 8180 with the front door on 8080 — which sent recipe-check to the
    door's payload (a sidecar's argv, a sidecar's env) and made it report 1 knob and 19 environment
    "deviations" on an engine that was configured exactly as the recipe says. So: an exported LOCAL_ENV wins,
    then the launcher's own inline default, then the helper's — the same order the shell reads them in.
    """
    cands = []
    if os.environ.get("LOCAL_ENV"):
        cands.append(os.environ["LOCAL_ENV"])
    cands += [m.group(1).strip().strip('"') for m in re.finditer(r"LOCAL_ENV=\$\{LOCAL_ENV:-([^}]+)\}", text)]
    if "lib/localenv.sh" in text and LOCAL_ENV_HELPER.is_file():
        _h = LOCAL_ENV_HELPER.read_text(errors="replace")
        cands += [m.group(1).strip().strip('"')
                  for m in re.finditer(r':\s*"\$\{LOCAL_ENV:=([^}]+)\}"', _h)]
    for c in cands:
        p = Path(os.path.expanduser(os.path.expandvars(c)))
        if p.is_file():
            return p
    return None


def env_file_var(text, name):
    """A NAME set in the launcher's machine-local file only -- no environment, no ${NAME:-default} fallback.

    launcher_var is the right question for "what port will the engine bind"; this is the right question for
    "does this box run a front door at all". Asking the wrong one starts a gateway on a box that never asked
    for one, because serve/gateway.sh defaults GATEWAY_PORT to 8080 and would then fight the engine for it."""
    p = launcher_local_env(text)
    if not p:
        return None
    for ln in p.read_text(errors="replace").splitlines():
        ln = ln.strip()
        if ln.startswith(f"{name}="):
            return ln.split("=", 1)[1].split("#")[0].strip().strip('"')
    return None


def launcher_var(text, name, default=None):
    """What serve/<script> will really use for NAME, in the order the script reads it: an exported environment
    value, then the machine-local file it sources, then its own ${NAME:-default}.

    setup.py used to ask only os.environ (and 8080) for PORT. On a box that sets PORT in native.env -- this one
    moved the engine to 8180 on 2026-11-15 so a gateway could own 8080 -- the tmux'd launcher obeyed the file
    while setup.py waited for /health on, and armed its death-watcher against, a port nothing was listening on.
    The launcher is the authority; read it the way the launcher does."""
    if os.environ.get(name):
        return os.environ[name]
    p = launcher_local_env(text)
    if p:
        for ln in p.read_text(errors="replace").splitlines():
            ln = ln.strip()
            if ln.startswith(f"{name}="):
                return ln.split("=", 1)[1].split("#")[0].strip().strip('"')
    m = re.search(r"\$\{" + name + r":-([^}]*)\}", text)
    return m.group(1) if m else default


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


# ------------------------------------------------------------------ watching a launch that is already up ---
# Why this exists at all: notes/launch-methodology.md §5 promises `died_at`/`reason` "when the watcher saw it
# die", and there was no watcher. launch() writes the record, waits for /health, prints its report and exits --
# so a launch that came up and died later kept a record that still said `up_at` and nothing else, and the next
# run's "recorded dead HH:MM:SS" clause had nothing to print. On 2026-11-14 that meant 1 h 48 m of nothing: the
# engine faulted at 20:59:09, the last rank aborted at 21:10:09, and neither launch.json nor the Monitor said a
# word (WEBUI=same puts the page INSIDE the serve, so it went down with it) until an operator asked at 22:58 --
# notes/bug-causal-conv1d-oob.md.
#
# It records; it does not restart. §4's rule stands: seen, not resurrected -- a four-card serve that restarts
# into a degrading link turns one fault into a loop.
WATCH_TICK = int(os.environ.get("WATCH_TICK", "10"))        # s between liveness probes

# Restart on a death, or keep recording it? §4's rule ("seen, not resurrected") is still the default, and this
# block does not quietly weaken it: WATCH_RESTART=1 has to be said out loud, and even then the watcher restarts
# only a death whose log carries a signature this box has actually seen, only when no ranks are left behind, and
# at most RESTART_MAX times inside RESTART_WINDOW_S. The reason to have it at all is measured: the 11-14 fault
# and the 11-15 fault were both device faults (same card, same page-fault shape), both left the engine gone for
# good, and both were followed by an hour or more of a LAN port answering "connection refused" while a human
# found out by asking. A restart in that case is the correct first response -- provided it stops after a few
# tries, because a serve that restarts into a degrading link turns one fault into a loop.
RESTART_MAX = int(os.environ.get("RESTART_MAX", "3"))                    # restarts allowed inside the window
RESTART_WINDOW_S = int(os.environ.get("RESTART_WINDOW_S", "3600"))       # the breaker's window

# A crash can leave its ranks behind, and then nothing can come back: on 2026-11-15 06:28 the device fault killed
# Worker_TP0 and left Worker_TP1/2/3 sitting on 32.3/32.3/32.2 GiB (rocm-smi: 32,371,838,976 B on each of three
# cards) with the engine gone. Those ranks ignore SIGTERM, so the record read "alive", the watcher waited, and
# the only way back was for a human to run STOP -- which is exactly the hour-plus of "connection refused" this
# whole block exists to avoid. RESTART_STOP=1 (the default once WATCH_RESTART=1 has been said out loud) lets the
# watcher run the RECIPE'S OWN stop -- `STOP=1 bash serve/<recipe>`, the same command the operator would type --
# to clear ranks that outlived their engine, and only for a death that already passed the signature gate. It is
# still not a free pass to kill somebody else's launch: it needs this launch's record, a known crash signature,
# and the ranks still there WATCH_WEDGE_TICKS probes in a row (a launch that is merely loading has ranks and no
# /health too, and stays untouched). RESTART_STOP=0 keeps the older, purely passive behaviour.
RESTART_STOP = os.environ.get("RESTART_STOP", "1") == "1"
RESTART_STOP_WAIT = int(os.environ.get("RESTART_STOP_WAIT", "120"))      # s to let the stop clear the ranks
WATCH_WEDGE_TICKS = int(os.environ.get("WATCH_WEDGE_TICKS", "2"))        # silent probes before "wedged"

# The signatures, each one from this box's own logs rather than from a list on the internet.
CRASH_SIGS = (
    ("illegal memory access",
     "2026-11-15 05:11:18, ProcessGroupNCCL's report of the device fault (Worker VllmWorker-0 died unexpectedly)"),
    ("EngineDeadError", "vLLM's own: the engine core is gone behind a live front door -- both 11-14 and 11-15"),
    ("HSA_STATUS_ERROR", "the ROCr runtime's status for a memory/aperture fault on a kernel"),
    ("died unexpectedly", "2026-11-15 05:11: 'Worker proc VllmWorker-0 died unexpectedly'"),
)


def crash_sig(tail):
    """The first known crash signature in the log tail, or None. Deliberately narrow: `STOP=1`, a killed tmux
    session and a machine reboot all end a launch with no signature here, and none of them should be answered
    with a relaunch."""
    for s, _ in CRASH_SIGS:
        if s in tail:
            return s
    return None


def log_tail(path, nbytes=262144):
    """The end of the launch log, which is where the fault line is. A big log is read backwards because these
    files run to hundreds of MB on a long launch (setup-005730.log was 240 MB with 6.8 days in it)."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            end = fh.tell()
            fh.seek(max(0, end - nbytes))
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def gpu_links(root="/sys/bus/pci/devices"):
    """[(bdf, speed, width, max_speed, max_width)] for the AMD display-class devices, straight from sysfs.

    Read before a restart, because §4's warning is specifically about restarting INTO a degraded link, and the
    negotiated link is the one fact in this box's failure family that can be measured before the relaunch rather
    than after it. Free to read (ordinary user, no root, no /dev/kfd), and it is the same pair the Monitor's card
    badge shows (Gen5 x16, or the generation it came up short at).
    """
    rows = []
    for p in Path(root).glob("*"):
        try:
            if (p / "vendor").read_text().strip() != "0x1002":          # not AMD
                continue
            if not (p / "class").read_text().strip().startswith("0x0300"):   # not a display controller
                continue
            sp = float((p / "current_link_speed").read_text().split()[0])
            ms = float((p / "max_link_speed").read_text().split()[0])
            w, mw = int((p / "current_link_width").read_text()), int((p / "max_link_width").read_text())
            rows.append((p.name, sp, w, ms, mw))
        except (OSError, ValueError, IndexError):
            continue                                                    # a card busy enough to refuse is not a rating
    return rows


def link_short(rows):
    """One phrase per card running below what it rated -- the shape §4 says must not be restarted into.

    Deliberately about the RATING rather than about error counters: this box carries dozens of correctable AERs a
    minute through a weight load at a full 32.0 GT/s x16 (notes/webui.md), so a counter moving is not the signal;
    a link that negotiated below its rating is."""
    return [f"{bdf} at {sp} GT/s x{w}, rated {ms} GT/s x{mw}"
            for (bdf, sp, w, ms, mw) in rows if sp < ms or w < mw]


def stop_wedge_would(rec, restart, left, sig, enabled):
    """Pure half of the wedge question, so tests/test_setup.py can hold it to its rules without a wedged box in
    front of it: may this watcher run the recipe's OWN stop to clear ranks that outlived their engine?

    All five have to be true: the switch is on (RESTART_STOP, default on once restarts are armed), a restart was
    actually asked for (WATCH_RESTART -- a purely recording watcher never touches anybody's processes), ranks are
    really there, the death carries a known crash signature (an operator's STOP or a killed session clears with no
    signature and is never answered this way), and the record names a recipe whose stop we can run. Without the
    signature gate, a watcher would be killing a launch that is merely slow to answer."""
    return bool(enabled and restart and left and sig and rec and rec.get("recipe"))


def stop_wedge(rec):
    """Run the recipe's own stop: `STOP=1 bash serve/<recipe>` -- the exact command the operator would type, not a
    hand-rolled kill. The launcher knows which processes are its ranks and how to stop ranks that ignore SIGTERM;
    an ad-hoc `pkill` here would be a second, worse opinion about the same processes.

    Returns (rc, last output line) for the record."""
    script = ROOT / "serve" / rec["recipe"]
    try:
        p = subprocess.run(["bash", str(script)], env=dict(os.environ, STOP="1"),
                           capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as e:
        return 1, f"{type(e).__name__}: {e}"
    lines = [x for x in (p.stdout + "\n" + p.stderr).splitlines() if x.strip()]
    return p.returncode, (lines[-1] if lines else "no output")


def restart_ok(rec, tail, left, now, hist, enabled, cap=RESTART_MAX, window=RESTART_WINDOW_S, links=None):
    """(restart?, why). Pure, so tests/test_setup.py can hold the breaker to its rules without killing a box.

    Four gates, in the order that costs nothing: switched on; a known crash signature in the log; no VLLM ranks
    still alive (the 11-14 fault left its ranks running for 11 more minutes -- restarting into that makes two
    launches fighting for four cards, which is a different and worse outage); and the breaker. `hist` is the
    restart timestamps on the record, so the cap survives the watcher that set them.

    `links` is the PCIe gate: a list of cards measured BELOW their rating (empty is healthy), or None when sysfs
    would not say. A short link stops the restart -- that is the loop §4 warns about, and a four-card serve
    loading 40 GB of weights through a link that came up at Gen3 is how one fault becomes an afternoon. An
    unknown link does not stop it, and says so in the reason: refusing to recover because one sysfs file was
    unreadable is its own outage, and the reader is told which case they are looking at.
    """
    if not enabled:
        return False, "off (WATCH_RESTART unset): \u00a74's rule stands, seen not resurrected"
    sig = crash_sig(tail)
    if not sig:
        return False, "no known crash signature in the log tail"
    if left:
        return False, f"{left} VLLM process(es) still alive: a wedge is not a crash, and a second launch into " \
                      f"the same four cards is a worse outage than the one on screen"
    if not rec or not rec.get("recipe"):
        return False, "the record names no recipe to re-run"
    if links:
        return False, "PCIe link(s) below rating, not restarting into that (\u00a74): " + "; ".join(links)
    recent = [t for t in (hist or []) if now - float(t) <= window]
    if len(recent) >= cap:
        return False, (f"breaker: {len(recent)} restart(s) in the last {window // 60} min "
                       f"(RESTART_MAX={cap}) -- a serve restarting into a degrading link turns one fault into "
                       f"a loop (notes/launch-methodology.md \u00a74)")
    return True, f"{sig}; restart {len(recent) + 1} of {cap} in {window // 60} min" \
        + ("" if links is not None else " (link state unreadable: not a block, but nobody measured it)")


def superseded(prev, me, claimed_at):
    """Has another watcher taken this record over from us?

    A launch can end up with two watchers: the launch flow arms one and someone arms another by hand. That is
    what happened on 2026-11-15 -- the armed one carried PORT=8080 (read before the launcher-port fix, when 8080
    still was the engine) and the hand-armed one carried the right 8180, both polling one launch.json, and the
    record named the wrong one. Two watchers means two opinions about which port is the engine's, and the record
    can only carry one death.

    So watch_loop CLAIMS the record (watcher=own pid, watcher_at=now) when it starts, and the record is the
    arbiter: whoever claimed last speaks. The claim TIME decides, so a record still naming a watcher that
    stopped long ago does not chase a live one away."""
    return bool(prev) and prev.get("watcher") not in (None, me) \
        and prev.get("watcher_at") is not None and claimed_at is not None \
        and prev["watcher_at"] >= claimed_at


def watch_tick(prev, sess, dead, now, me=None, claimed_at=None):
    """One tick of the watcher as a decision: ("", {}) keep waiting, ("record", fields) write the death,
    ("stop", {}) leave the record alone and get out.

    Pure, so tests/test_setup.py can hold it to its rules without a box to kill: the record must name the
    session this watcher was armed for (someone else's launch -- a relaunch under another session, a hand-started
    serve -- is stamped by its own launcher, not by a watcher left awake for the previous one); a watcher the
    record has since been re-claimed by is not the one that writes (see superseded()); a death already
    on the record is not re-stamped, because the first timestamp is the one worth having; and while the launch
    answers, nothing is written at all."""
    if prev is None or (prev.get("session") or None) != (sess or None):
        return "stop", {}
    if superseded(prev, me, claimed_at):
        return "stop", {}
    if dead is None:
        return "", {}
    if prev.get("died_at") is not None:
        return "stop", {}
    return "record", {"died_at": int(now), "reason": dead}


def watch_dead(port, sess, prev=None, quiet=0):
    """Why the launch is gone, or None while it is still there.

    With tmux the session IS the probe (the pane closes when the pipeline ends, per start_tmux). Without one it
    is /health AND the process count: a launch still loading weights has no /health yet, and calling that dead
    is how a watcher writes a lie into the record.

    The third case is the one that bit on 2026-11-15 06:28: the session stayed up (the pane's pipeline had not
    closed), the ranks stayed alive holding 32 GiB a card, and /health went silent -- so every probe said "not
    gone" and the record never got a death at all. A launch that had ALREADY answered /health once (the record's
    up_at) and then goes quiet, twice in a row, with a known fault line at the end of its log, is not loading:
    it crashed and its ranks are wedged. `prev=None` (a caller with no record) keeps the old, narrower answer."""
    if sess and out(["tmux", "has-session", "-t", sess])[0] != 0:
        return f"the tmux session {sess} holding it is gone"
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            if r.status == 200:
                return None
    except (urllib.error.URLError, OSError):
        pass
    n = left_vllm()
    if not n:
        return f"no /health answer on {port} and no VLLM process left"
    if prev is not None and prev.get("up_at") and quiet >= WATCH_WEDGE_TICKS:
        sig = crash_sig(log_tail(prev.get("log") or ""))
        if sig:
            return (f"no /health answer on {port}, {n} VLLM rank(s) still holding the GPUs, and '{sig}' in its "
                    f"log: crashed and wedged (the ranks ignore SIGTERM)")
    return None                           # ranks present, no answer: still loading -- not "gone"


def restart_cmd(rec):
    """The setup.py command that re-runs this launch -- pure, so tests/test_setup.py can hold it without a
    box. The recipe, the pages choice, the stored configuration name and the extra flags this launch was
    given all come off the record: a restart that silently drops the flags someone typed is a restart into a
    different launch. The record's webui is the WEBUI spelling ("1"); --webui parses the name, so both
    spellings are taken and mapped here."""
    webui = str(rec.get("webui"))
    cmd = [sys.executable, str(ROOT / "setup.py"), "--yes", "--launch", rec["recipe"],
           "--webui", webui if webui in WEBUI_ENV else WEBUI_NAME.get(webui, "same")]
    if rec.get("config"):
        cmd += ["--config", rec["config"]]
    if rec.get("no_tmux"):
        cmd += ["--no-tmux"]
    cmd += list(rec.get("rest") or [])
    return cmd


def restart_launch(rec):
    """Re-run THIS launch through setup.py, not through a bespoke restart path.

    That choice is the whole safety of the thing: the restart goes down the same code a human would use, so it
    re-does the /health wait, the context check, the gateway bring-up, the recipe check and the watcher arming
    instead of inventing a second, slightly different launch that nothing else in this repo tests. The recipe,
    the pages choice, the stored configuration name and the extra flags this launch was given all come off the
    the record, because a restart that silently drops the flags someone typed is a restart into a different
    launch.
    """
    return subprocess.Popen(restart_cmd(rec), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                            start_new_session=True)


# ---- the ledger: what the box keeps after the record it was written into is overwritten ------------------
# `launch.json` describes the CURRENT launch and the next launch rewrites it; `restarts.json` keeps the last
# death, and only the one that led to a decision. Neither is a history, and the questions that get asked weeks
# later are history questions: how many times has THIS card taken a page fault, did a restart follow each one,
# how long was the box online in between, and how much traffic was in flight when it happened. So every death
# the watcher sees appends one JSONL line here -- including the deaths a record-only watcher deliberately left
# alone, which is exactly the case where this file is the only trace. fsynced like the request journal, because
# a record that only survives when the box stays up is useless for the case it exists for.
FAULT_LEDGER = "faults.jsonl"
ALERT_FILE = "ALERT.json"          # what is asserted right now (the Monitor reads this one)
ALERT_LEDGER = "alerts.jsonl"     # every time it was asserted, for the long view

# Hardware escalation, counted per CARD rather than per launch: §15's card-order run showed the fault follows
# the physical function, and what turns "a software fault I can restart away" into "a card I have to go and
# look at" is repetition. Three page-fault events on one PCI function inside a week is where the honest action
# stops being `./setup.sh` and starts being opening the case and swapping slots -- so the watcher says so out
# loud, in the log and on the page, at the same moment it is bringing the engine back. Both numbers are knobs
# because what counts as repetition depends on how hard the box is driven.
HW_ALERT_N = int(os.environ.get("HW_ALERT_N", "3"))
HW_ALERT_WINDOW_S = int(os.environ.get("HW_ALERT_WINDOW_S", str(7 * 24 * 3600)))
ALERT_CMD = (os.environ.get("ALERT_CMD") or "").strip()      # your channel: ntfy, mail, desktop, whatever

# A webhook an operator pasted into the About page, written 0600. It lives in the state directory and not in the
# env because it is a credential, and a credential in a launcher is a credential in a log line (the launch line is
# printed at every start, and `setup.py` prints it on purpose).
NOTIFY_FILE = "notify.json"
NOTIFY_LAST = "notify-last.json"      # the last send attempt, so the page can answer "does the channel work?"
NOTIFY_SRC = Path(__file__).resolve().parent / "serve" / "notify.py"
_notifier_cache = None


def notifier():
    """`serve/notify.py`, loaded once. (module, None) or (None, why).

    The wire format lives in ONE file because the alert is RAISED here (by the watcher, seconds before a relaunch)
    and CONFIGURED and TESTED in the page process (`serve/webui.py`, where the webhook is pasted). Two
    implementations of one POST is how an alert ends up silently refused by the robot's keyword filter while both
    programs believe they sent it.

    A missing notifier is a warning, never a swallowed alert: the claim is still in ALERT.json, on the Monitor, and
    in alerts.jsonl. What is lost is the ping to a phone, and that gets said out loud in the ledger line.
    """
    global _notifier_cache
    if _notifier_cache is None:
        try:
            spec = importlib.util.spec_from_file_location("r9700_notify", NOTIFY_SRC)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _notifier_cache = (mod, None)
        except (OSError, ValueError, ImportError, AttributeError) as exc:
            _notifier_cache = (None, f"cannot load {NOTIFY_SRC}: {type(exc).__name__}: {exc}")
    return _notifier_cache

JOURNAL_CMD = ("journalctl", "-k", "--boot", "--no-pager", "-o", "short-iso", "-g",
               "gfxhub] page fault|Faulty UTCL2|RW: 0x|Process .* pid")
J_FAULT = re.compile(r"amdgpu (?:[0-9a-f]{4}:)?([0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]):\s*\[gfxhub\] page fault")
J_RING = re.compile(r"\(src_id:\d+ ring:(\d+)")
J_PID = re.compile(r"Process \S+ pid (\d+)")
J_CLIENT = re.compile(r"Faulty UTCL2 client ID: (.+?) \(0x[0-9a-f]+\)")
J_RW = re.compile(r"RW: (0x[0-9a-f]+)")


def _epoch(stamp):
    """`journalctl -o short-iso` -> epoch seconds, or None. An unreadable stamp is carried as None rather than
    guessed at: an event that cannot be placed in time cannot be put inside a window."""
    try:
        return datetime.fromisoformat(stamp).timestamp()
    except (ValueError, TypeError):
        return None


def kernel_fault_events(command=None, timeout=10.0):
    """The driver's page-fault blocks for this boot, one dict per fault LINE, with the fields under each one.

    Richer than the Monitor row needs, on purpose: the fields the driver prints beneath a fault -- ring, UTCL2
    client, read/write -- are what separate the two fault families this box has (notes §20: six READ faults by
    `SQC (data)` on ring 24 from the vLLM launches, two WRITE faults by `TCP` on ring 40 from two experiments
    that are long over). A per-card count alone would call both "c3:00.0 is unhappy", which is true and not
    actionable.

    Always a dict. `error` non-None with `total` None when the journal cannot be read -- never an empty list,
    because "no faults" and "cannot see the kernel log" are different claims and only one of them is evidence
    about a card. This is the trap that already happened once here: `journalctl -A` is `--after-boot`, not
    "lines after the match", so a mistyped flag prints a usage line to stderr and NOTHING to stdout, which read
    as zero faults on a box that had had eight that boot.
    """
    command = command or JOURNAL_CMD
    try:
        r = subprocess.run(list(command), capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"events": [], "by_card": {}, "total": None, "error": f"{type(exc).__name__}: {exc}"}
    if r.returncode != 0:
        first = ((r.stderr or r.stdout or "").strip().splitlines() or [""])[0]
        return {"events": [], "by_card": {}, "total": None, "error": f"journalctl exited {r.returncode}: {first}"}
    out, cur = [], None
    for line in r.stdout.splitlines():
        m = J_FAULT.search(line)
        if m:
            stamp = (line.split(" kernel:")[0].strip().split() or [""])[0]
            rg = J_RING.search(line)
            cur = {"at": stamp, "epoch": _epoch(stamp), "card": m.group(1).lower(), "ring": rg.group(1),
                   "pid": None, "client": None, "rw": None}
            out.append(cur)
            continue
        if cur is None:
            continue
        pm = J_PID.search(line)
        if pm and cur["pid"] is None:
            cur["pid"] = int(pm.group(1))
            continue
        cm = J_CLIENT.search(line)
        if cm:
            cur["client"] = cm.group(1).strip()
        rm = J_RW.search(line)
        if rm:
            cur["rw"] = "write" if int(rm.group(1), 16) & 1 else "read"
            cur = None                                       # the status block ends at RW
    per = {}
    for e in out:
        per[e["card"]] = per.get(e["card"], 0) + 1
    return {"events": out, "by_card": per, "total": sum(per.values()), "error": None}


def hw_verdict(faults, now=None, n=HW_ALERT_N, window=HW_ALERT_WINDOW_S):
    """The card with the most page-fault EVENTS inside the window, or None.

    Pure, so a test can hand it a journal it made. Counted per distinct (stamp, card) rather than per line,
    because one crash can write several fault lines (measured: 11 lines across 8 events). An event whose stamp
    cannot be parsed goes to `unattributed` instead of being guessed into the window; a failed journal read
    returns None, and the ledger then carries the error, so "no alert" is explainable as "could not look"
    rather than reading as "nothing wrong".
    """
    if not faults or faults.get("total") is None:
        return None
    now = time.time() if now is None else now
    evs, unattr = [], 0
    for e in faults.get("events") or []:
        if e.get("epoch") is None:
            unattr += 1
        elif e["epoch"] >= now - window:
            evs.append(e)
    seen, uniq = set(), []
    for e in evs:
        k = (e["at"], e["card"])
        if k not in seen:
            seen.add(k)
            uniq.append(e)
    by, fam = {}, {}
    for e in uniq:
        by[e["card"]] = by.get(e["card"], 0) + 1
        key = f'{e.get("rw") or "?"} / {e.get("client") or "?"} / ring {e.get("ring") or "?"}'
        fam.setdefault(e["card"], {}).setdefault(key, 0)
        fam[e["card"]][key] += 1
    if not by:
        return {"card": None, "count": 0, "cards": {}, "families": fam, "unattributed": unattr,
                "window_s": window, "since": None, "until": None}
    card = max(by, key=lambda c: by[c])
    if by[card] < n:
        return {"card": None, "count": by[card], "cards": by, "families": fam, "unattributed": unattr,
                "window_s": window, "since": None, "until": None}
    hits = [e for e in uniq if e["card"] == card]
    return {"card": card, "count": by[card], "cards": by, "families": fam.get(card, {}),
            "unattributed": unattr, "window_s": window,
            "since": min(e["at"] for e in hits), "until": max(e["at"] for e in hits)}


def alert_words(v):
    """The alert as two sentences a person can act on: what was counted, and what to do about it."""
    fam = ", ".join(f"{c} {k}" for k, c in sorted(v["families"].items())) or "shape not printed by the driver"
    return (f"{v['count']} driver page faults on 0000:{v['card']} within "
            f"{v['window_s'] // 86400} d ({v['since']} .. {v['until']}) -- {fam}",
            "the fault follows the CARD, not the rank (launch-methodology §15 / bug-causal-conv1d-oob §15,§20): "
            "swap this card with a neighbour (or move it to another slot) and see whether the faults move with "
            "it; while it is unresolved, do not restart INTO it repeatedly -- keep the ledger, "
            ".r9700-state/faults.jsonl, as the evidence.")


def fault_row(rec, fields, sig, left, cleared, do_restart, why, faults, gate, verdict, uptime_s, now=None):
    """The ledger line, built without touching anything so a test can hold it to its shape.

    It carries the death (reason, signature, ranks left holding GPUs), the decision (restart yes/no and WHY --
    the breaker's reason is as much history as the crash), the card evidence (per-card fault counts, or the
    error that meant there were none), and the traffic context from the one witness that survives an engine
    death: the front door."""
    return {"at": int(time.time() if now is None else now),
            "session": (rec or {}).get("session"), "recipe": (rec or {}).get("recipe"),
            "config": (rec or {}).get("config"), "log": (rec or {}).get("log"),
            "died_at": (fields or {}).get("died_at"), "reason": (fields or {}).get("reason"),
            "sig": sig, "ranks_left": left, "cleared_ranks": cleared,
            "restart": bool(do_restart), "restart_why": why,
            "uptime_s": uptime_s,
            "faults_total": (faults or {}).get("total"), "faults_by_card": (faults or {}).get("by_card"),
            "faults_error": (faults or {}).get("error"),
            "alert": ({"card": verdict.get("card"), "count": verdict.get("count"),
                       "families": verdict.get("families")} if verdict and verdict.get("card") else None),
            "gate": gate}


def ledger_append(name, row):
    STATE.mkdir(exist_ok=True)
    with (STATE / name).open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def raise_alert(v, rec, sig=None, command=None, at=None, raised_by=None):
    """Assert the hardware alert where the operator will see it and where the history keeps it.

    ALERT.json is the live claim (the Monitor reads it and the About page can), alerts.jsonl is every time it
    was claimed, and the message out to a human is either the webhook in `.r9700-state/notify.json` (configured
    from the About page, sent by `serve/notify.py`) or `ALERT_CMD` if the box wires one. The command gets the
    path of ALERT.json as argv[1] and in $ALERT_PATH, so a hook never has to re-derive where the evidence is.

    The claim is written BEFORE anything is sent: a POST that hangs while the machine is being restarted must not
    cost the record. The delivery result goes into the ledger line, because "it was alerted" and "a robot refused
    it because the keyword was wrong" look identical in a file that does not say which happened.
    """
    what, do = alert_words(v)
    row = {"at": int(time.time() if at is None else at), "card": v["card"], "count": v["count"],
           "window_s": v["window_s"], "since": v["since"], "until": v["until"], "cards": v["cards"],
           "families": v["families"], "what": what, "do": do, "sig": sig,
           "session": (rec or {}).get("session"), "recipe": (rec or {}).get("recipe"),
           "config": (rec or {}).get("config"), "log": (rec or {}).get("log"),
           # which program asserted it. A claim raised by hand next to an outage is a normal thing to do, and the
           # ledger has to say that a human did it rather than imply a death triggered it.
           "raised_by": raised_by}
    sent = notify_alert(row)                               # the POST happens after the claim is on disk
    row["notify"] = sent                                   # so the PAGE can say "and nobody was told"
    state_write(ALERT_FILE, row)
    ledger_append(ALERT_LEDGER, row)
    cmd = ALERT_CMD if command is None else command
    if cmd:
        try:
            subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30,
                           env=dict(os.environ, ALERT_PATH=str(STATE / ALERT_FILE)), input=json.dumps(row))
        except (OSError, subprocess.SubprocessError) as exc:
            warn(f"ALERT_CMD failed ({type(exc).__name__}: {exc}) -- the alert is still in {ALERT_FILE}")
    return row


def notify_alert(row, now=None):
    """Send the standing alert to the configured webhook. Returns the delivery result, safe to put in a ledger line:
    it never carries the webhook, and it always says whether a human was actually told.

    Three states are kept apart on purpose, because collapsing them is how a broken channel reads as a working
    one: no channel configured, a channel file that cannot be read or has no webhook in it, and a robot that took
    the POST and answered with a non-zero errcode (a keyword mismatch, or a dead token).
    """
    mod, err = notifier()
    if mod is None:
        warn(f"{err} -- the alert is on the page and in the ledger, but nobody was pinged")
        return {"sent": False, "at": int(time.time() if now is None else now), "error": err}
    cfg, cerr = mod.load(STATE / NOTIFY_FILE)
    if cerr:
        warn(f"alert channel: {cerr}")
        return {"sent": False, "at": int(time.time() if now is None else now), "error": cerr}
    if cfg is None:
        return {"sent": False, "at": int(time.time() if now is None else now),
                "error": f"no webhook configured (the About page, or {STATE / NOTIFY_FILE})"}
    res = mod.post(cfg, row)
    mod.remember(STATE / NOTIFY_LAST, res, mod.mask(cfg))
    if res.get("sent"):
        say(f"alert sent to the webhook ({res.get('errmsg') or 'ok'})")
    else:
        warn(f"alert NOT delivered: {res.get('error')}")
    return res


def gateway_counters(port=None):
    """The front door's counters, asked at the moment of death.

    The gateway is a separate process: it survives the engine, so it is the only witness that can still say how
    much traffic was in flight, how many requests it had admitted and how many it had held at the door when the
    rank died. None if nothing is listening on 8080: a box without a gateway is normal, and an answer-less
    question is not a zero.

    `policy` goes in with them, because since 2026-11-16 the door usually holds NOTHING on purpose: a ledger line
    reading `held 0` has to be distinguishable from a door that held nobody because it was watching, from one that
    was armed and idle, and from one that was not there at all. `arrived_shut` is the count that mode keeps -- the
    arrivals its harm signal found.
    """
    port = port or os.environ.get("GATEWAY_PORT", "8080")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/gate", timeout=3) as r:
            d = json.loads(r.read())
        c = d.get("counters") or {}
        return {"admitted": c.get("admitted"), "held": c.get("held"), "expired": c.get("expired"),
                "arrived_shut": c.get("arrived_shut"), "policy": d.get("policy"), "closed": d.get("closed"),
                "shut": (d.get("shut") or {}).get("frac")}
    except (urllib.error.URLError, OSError, ValueError):
        return None


def watch_restarts(rec, env=None):
    """Should THIS watcher restart a death?

    Two sources, in this order, because they mean different things. `WATCH_RESTART` in the environment is the
    operator saying it out loud right now -- for the watcher armed by hand at 05:00, that is exactly the moment
    the policy is supposed to change, and the claim below writes the change onto the record so the answer is
    still knowable afterwards. Without that env, the record rules: a watcher that carries no opinion must not
    silently flip the launch's supervision mode, and `watch_loop` printing `restart_on_crash` afterwards is what
    makes the decision auditable rather than a memory question.
    """
    said = (os.environ if env is None else env).get("WATCH_RESTART")
    if said is not None:
        return said == "1"
    return bool((rec or {}).get("restart_on_crash"))


def watch_port(rec=None):
    """Which port the watchdog should probe: the launch record's if it names one, else what the launcher will bind.

    `--watch` used to read `os.environ.get("PORT", "8080")`. On this box that is 8080, which is the FRONT DOOR: the
    door forwards /health to the engine, so the probe still answered, but it also means a dead door reads as a dead
    engine (and would restart a healthy one), and it is the same half-read that let a launch record name a port
    nothing served. The record first, because a death should be judged against the port that launch actually bound;
    the launcher second, because the record may be missing (a launch that died during its load).
    """
    rec = rec if rec is not None else state_read("launch.json")
    if rec and rec.get("port"):
        return int(rec["port"])
    pick = (rec or {}).get("recipe")
    if pick and (ROOT / "serve" / pick).is_file():
        return int(launcher_var((ROOT / "serve" / pick).read_text(errors="replace"), "PORT", "8080"))
    return int(os.environ.get("PORT", "8080"))


def watch_loop(port, sess, log):
    """Poll a launch that is already up until it stops, then put the moment on the record and exit.

    It claims the record first. That is what makes arming a watcher by hand safe rather than a duplicate: the
    claim settles who speaks, and a superseded watcher exits saying so instead of quietly stamping the record
    from a port it picked up in an older run."""
    me, claimed = os.getpid(), int(time.time())
    quiet = 0                                            # consecutive probes with no /health answer
    rec = state_read("launch.json")
    # Would this watcher restart, and is that answer knowable afterwards? arm_watch puts it on the record; a
    # watcher armed BY HAND carries no such field, so its own environment decides and the claim writes the
    # decision down -- "would it have come back?" must not depend on remembering which shell started the watcher.
    restart = watch_restarts(rec)
    if rec is not None and (rec.get("session") or None) == (sess or None):
        # `restart_max` travels with the decision, not with the default: a record that says `restart_on_crash:
        # false, restart_max: 3` reads as "it would come back, three times an hour" to whoever finds it at 05:00.
        state_update("launch.json", watcher=me, watcher_at=claimed, restart_on_crash=restart,
                     restart_max=(RESTART_MAX if restart else 0), restart_window_s=RESTART_WINDOW_S)
    else:
        warn("no launch.json for this session: the watcher reports on stdout only")
    # Which port am I asking? Said once, at the claim, because the question gets asked the moment it matters:
    # "the watchdog called it dead" is only checkable if the record says what it probed. A door port answered
    # here for a day on this box (it forwards /health, so it appeared to work) while the engine was on 8180.
    say(f"[watcher] probing :{port} for this launch"
        + (f" (session {sess})" if sess else " (session: whatever the record names)")
        + f"; restart on a known crash signature: {'yes' if restart else 'no'}"
        + (f" (up to {RESTART_MAX} per {RESTART_WINDOW_S // 60} min)" if restart else ""))
    while True:
        rec = state_read("launch.json")
        d = watch_dead(port, sess, rec, quiet)
        quiet = 0 if d is None else quiet + 1
        act, fields = watch_tick(rec, sess, d, time.time(), me, claimed)
        if act == "":
            time.sleep(WATCH_TICK)
            continue
        line = f"[watcher] {time.strftime('%Y-%m-%d %H:%M:%S')} "
        if act == "stop" and superseded(rec, me, claimed):
            say(line + f"this watcher is superseded: the record names pid {rec.get('watcher')}, claimed "
                       f"{rec.get('watcher_at')} -- that one speaks for this launch. Nothing written.")
            return 0
        line += f"this launch is gone: {d}"
        restarted = None
        if act == "record":
            state_update("launch.json", **fields)
            # The breaker's decision is printed either way, with the reason, because "why is the box back up"
            # and "why is it still down" are the same question and both get asked at 05:00.
            rst = state_read("restarts.json") or {}
            hist = [t for t in (rst.get("at") or []) if isinstance(t, (int, float))]
            tail = log_tail(rec.get("log") or log)
            sig = crash_sig(tail)
            left = left_vllm()
            cleared = None
            faults = kernel_fault_events()      # the driver's own record; None-ish on an unreadable journal
            if stop_wedge_would(rec, restart, left, sig, RESTART_STOP):
                say(line + f" -- {left} rank(s) are still holding the GPUs behind the fault; running this "
                           f"launch's own stop (STOP=1 bash serve/{rec['recipe']}) so a relaunch has cards to "
                           f"load onto. Ranks ignore SIGTERM; this is the command an operator would type.")
                before, rc, out_txt = left, 1, ""
                rc, out_txt = stop_wedge(rec)
                deadline = time.time() + RESTART_STOP_WAIT
                while left_vllm() and time.time() < deadline:
                    time.sleep(3)
                left = left_vllm()
                cleared = {"at": int(time.time()), "rc": rc, "ranks_before": before, "ranks_after": left,
                           "stop": f"STOP=1 bash serve/{rec['recipe']}", "last_line": out_txt}
                say(line + f" -- after the stop: {left} rank(s) left ({out_txt})")
            # RESTART_LINK=0 skips the sysfs gate (cards too busy to read, or an operator who has already looked
            # at the links and just wants the serve back).
            rows = gpu_links() if os.environ.get("RESTART_LINK", "1") != "0" else []
            do, why = restart_ok(rec, tail, left, time.time(), hist, restart,
                                 links=link_short(rows) if rows else None)
            # The ledger and the alert come BEFORE the act. If the restart itself wedges, the death and the
            # decision are already on disk, and the hardware claim is in the log while there is still a machine
            # to read it from -- an alert printed after a successful relaunch is an alert that can be lost to the
            # very failure it is warning about.
            verdict = hw_verdict(faults)
            if verdict and verdict.get("card"):
                what, do2 = alert_words(verdict)
                say(line + f" -- HARDWARE ALERT: {what}. {do2}")
                raise_alert(verdict, rec, sig)
            ledger_append(FAULT_LEDGER, fault_row(rec, fields, sig, left, cleared, do, why, faults,
                                                 gateway_counters(), verdict,
                                                 int(time.time() - rec["up_at"]) if rec.get("up_at") else None))
            nfault = faults.get("total")
            say(line + f" -- this death appended to {STATE / FAULT_LEDGER} "
                       f"({'no journal reading' if nfault is None else nfault} fault line(s) this boot"
                       f"{'; ' + faults['error'] if faults.get('error') else ''}).")
            if do:
                state_write("restarts.json", {
                    "session": rec.get("session"), "at": hist + [int(time.time())],
                    # the death outlives the record it was written into: the new launch rewrites launch.json,
                    # so the reason and the signature live here as well
                    "last": {"died_at": fields["died_at"], "reason": fields["reason"], "sig": sig,
                             "recipe": rec.get("recipe"), "config": rec.get("config"),
                             "cleared_ranks": cleared}})
                p = restart_launch(rec)
                restarted = f"{why} (setup.py pid {p.pid})"
                line += f" -- RESTARTING: {restarted}. The new launch arms its own watcher and rewrites " \
                        f"launch.json; the death stays in .r9700-state/restarts.json."
            else:
                line += f" -- nothing restarted it ({why}): see notes/launch-methodology.md \u00a74."
        else:
            line += " -- nothing written: that record is not this launch's, or its death is already on it"
        say(line)
        if log:
            try:
                with open(log, "a") as fh:
                    fh.write(line + "\n")
            except OSError:
                pass
        return 0


def arm_watch(port, sess, log):
    """Put the watcher on a launch that came up: detached, so it outlives setup.py the way the serve does."""
    try:
        p = subprocess.Popen([sys.executable, str(ROOT / "setup.py"), "--watch"],
                             env=dict(os.environ, PORT=str(port), LOG=log or "",
                                      WATCH_SESSION=sess or ""),
                             stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, start_new_session=True)
        armed = os.environ.get("WATCH_RESTART", "0") == "1"
        state_update("launch.json", watcher=p.pid, watcher_at=int(time.time()),
                     # what the watcher would do with a death, on the record, so the answer to "would it have
                     # come back?" does not depend on remembering the shell that launched it -- and the cap is
                     # written as 0 when it is off, because `restart_max: 3` beside `restart_on_crash: false`
                     # reads like a crash would be answered
                     restart_on_crash=armed,
                     restart_max=(RESTART_MAX if armed else 0), restart_window_s=RESTART_WINDOW_S)
        return p.pid
    except OSError as e:
        warn(f"no watcher on this launch ({e}): died_at/reason will stay unset, so a death after /health is "
             f"only visible on the next run as 'no VLLM process left'")
        return None


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
        if p.name in ("serve.sh", "webui.sh", "webmeta.sh", "deploy.sh", "embed.sh", "embed-watch.sh",
                      "gateway.sh"):      # gateway.sh starts no vLLM and owns no GPU: it is a front door, not a launch
            # embed.sh is not an engine recipe: it starts no vLLM and owns no GPU, so it is offered by its own
            # flag (--embedding) and its own step, not in the launch list a GPU launch is picked from.
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


def launch_model_names(text):
    """The checkpoints ONE launcher will actually touch: every /models/<name> its script names, plus whatever
    its MODEL=/DRAFT=/DRAFTER= lines resolve to (exported environment first, then the machine-local file, then
    the script's own default -- launcher_var's order).

    Why both halves: serve/27b.sh writes MODEL=/models/... (the regex sees it), serve/native-flashnext.sh
    writes MODEL=${MODEL:-$HOME/models/...} (only launcher_var sees that, and an operator's MODEL= outranks
    the default the way the shell ranks it). A name only ANOTHER recipe's script mentions is not this
    launch's business -- that is what made setup step 5 demand checkpoints the selected launch never
    loads.

    A comment mentioning /models/x counts: better a false "wanted" (fall back to the old behaviour) than a
    false "not wanted" on the one line the operator needed to see."""
    want = {m.group(1) for m in re.finditer(r"/models/([A-Za-z0-9._-]+)", text)
            if m.group(1) not in ("...", "x")}
    for k in ("MODEL", "DRAFT", "DRAFTER"):
        v = launcher_var(text, k)
        if v:
            want.add(Path(v).name)
    return want


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


# Upstream fixes this box carries in its native venv, named here rather than remembered. The venv IS the artifact
# of record (the 0.32 wheel cannot be re-fetched -- see native()'s own warning), so "we run stock vLLM 0.32.0"
# stops being true the day a patch goes in, and a fix nobody can point at is a fix that gets lost at the next
# rebuild. Each entry: the patch file under patches/, the installed file it touches, a string that only exists
# once applied, and why this box wants it.
VPATCHES = (
    ("vllm-0.32.0-61174-clear-stale-block-table-rows.patch",
     "v1/worker/gpu/block_table.py", "_num_dirty_rows",     # path inside the vllm package dir, which is what
                                                            # venv_patches() is handed (`vllm.__file__`'s dirname)
     "upstream #61174 (2026-10-17) 'Clear stale shared block-table rows': after a smaller batch, the rows a larger "
     "one left in the SHARED input block-table buffer must be cleared -- otherwise the next step reads block ids "
     "that are perfectly in range and name blocks since freed and reallocated. Measured on this box: upstream's "
     "own test file fails 7/7 against the wheel as installed and passes 7/7 after the patch "
     "(tests/test_vllm_stale_block_table_rows.py). Carried because it is a real bug in a buffer this box reads "
     "block ids out of -- NOT because it fixes the mixed-step device faults: a soak hit the identical fault 250 s "
     "into a launch that carried it (notes/bug-causal-conv1d-oob.md \u00a714)."),
)


def marker_of(patch):
    """The marker string for a named venv patch (for the report line)."""
    return next(m for p, _f, m, _w in VPATCHES if p == patch)


def venv_patches(vdir):
    """[(patch, file, present, why)] for a vllm install dir: is each fix actually in the code on disk?

    Pure-ish (it only reads files), so tests/test_setup.py can hold it to its rules with a temporary tree -- the
    real venv is not needed and its state is not something a host test should depend on."""
    rows = []
    for patch, rel, marker, why in VPATCHES:
        root = None if vdir is None else Path(vdir)
        if root is None or not root.is_dir():
            present = None                         # no install directory to ask: unknown, not "missing the fix"
        elif not (root / rel).exists():
            present = False                        # the file it patches is not there: the fix is not in
        else:
            try:
                present = marker in (root / rel).read_text(errors="replace")
            except OSError:
                present = None                     # a directory busy enough to refuse is not a verdict
        rows.append((patch, rel, present, why))
    return rows


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
    # the venv's patch state, reported like a version number: a patched wheel and a stock one are different
    # artifacts, and the difference only bites at 05:00 when a fault is being attributed to something
    rc2, vdir = out([str(py), "-c", "import vllm, os; print(os.path.dirname(vllm.__file__))"],
                    timeout=240, env=env)
    for patch, rel, present, why in venv_patches(vdir.strip() if rc2 == 0 else None):
        if present:
            ok(f"venv patch applied: {patch} ({rel} carries {marker_of(patch)})")
            note(f"    why: {why}")
            note("    a launch that started BEFORE the patch was applied does not carry it: the running process is "
                 "the artifact that matters, not the file -- compare its start time with the patch file's mtime")
        elif present is None:
            note(f"venv patch state unknown ({patch}): cannot read {vdir or 'the vllm install directory'}")
        else:
            warn(f"venv patch NOT applied: {patch} -- {rel} has no {marker_of(patch)}",
                 f"patch -p1 -d <site-packages> < patches/{patch}   then restart the launch "
                 f"(a running engine keeps the old code)")
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
    # Which launch this run actually made decides which checkpoint is a hole and which is scenery: a box
    # launching 27b does not need the 120B fetched, and saying so at warn level is how a 40-line report
    # trains people to skip the report. With no recipe selected (a plain check, --launch none) every
    # recipe's models stay on the list, exactly as before.
    sel = cfg.launch if cfg.launch and cfg.launch != "none" else None
    want = None
    if sel:
        try:
            want = launch_model_names((ROOT / "serve" / sel).read_text(errors="replace"))
        except OSError:
            note(f"serve/{sel} is not readable: this check covers every recipe's models, not only this launch")
    d = Path.home() / "models"
    if not d.is_dir():
        warn(f"{d} does not exist", "mkdir it and put the checkpoints in it, or change the MODEL= paths")
        return
    total = 0
    for n in sorted(names):
        p = d / n
        who = ", ".join(sorted(f"serve/{s}" for s in names[n]))
        in_launch = want is None or n in want
        if not p.is_dir():
            if not in_launch:
                note(f"/models/{n} missing: not wanted by the selected launch ({sel}), so this run does not "
                     f"ask for it (wanted by {who})")
                continue
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
        if not in_launch:                                    # present, but not this launch's model: no du,
            note(f"/models/{n} is here; only {who} want it -- not this launch's model")   # not in the total
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


def _up(url):
    """A GET that answers 200 -- the whole health question, the same one wait_up asks one level up."""
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def llama_build(cfg, want=False):
    """The llama.cpp build in $HOME the embedding sidecar runs from (no root, no venv: cmake --build, native
    dispatch -- on the EPYC Genoa that routes q8_0 through the AVX-512/VNNI kernels by itself). Returns the
    binary path or None. An update only checks it; the clone + compile happens only when asked -- the one
    build update.sh's contract says an update does not do by itself."""
    p = Path(os.environ.get("LLAMA_SERVER", Path.home() / "llama.cpp/build/bin/llama-server"))
    if p.exists():
        ok(f"llama-server: {p}")
        return str(p)
    if want and not cfg.check:
        if shutil_which("cmake") is None:
            bad("cmake is not installed -- llama.cpp needs it", "sudo apt install cmake g++ git")
            return None
        src = Path.home() / "llama.cpp"
        if not src.is_dir():
            if not ask(cfg, f"clone llama.cpp into {src} (~2 min, ~1 GB)?"):
                return None
            if run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", str(src)],
                   cfg, timeout=900) != 0:
                bad("git clone llama.cpp failed (the reason is above)")
                return None
        # -S names the source tree: run() has no cwd, so a bare `cmake -B <build>` takes THIS folder as the
        # source and configure dies with "the source directory .../r9700-stack does not appear to contain
        # CMakeLists.txt" (seen on the box 2026-11-15 -- the first run of this step ever).
        if run(["cmake", "-S", str(src), "-B", f"{src}/build", "-DGGML_NATIVE=ON",
                "-DCMAKE_BUILD_TYPE=Release"], cfg) != 0:
            bad("cmake configure failed (the reason is above)")
            return None
        if run(["cmake", "--build", f"{src}/build", "-j16"], cfg, timeout=1800) != 0:
            bad("cmake --build failed (the reason is above)")
            return None
        if p.exists():
            ok(f"built {p}")
            return str(p)
        bad("the build finished but the binary is not where it should be", f"ls {src}/build/bin")
        return None
    note(f"llama-server not built yet: {p} -- ./setup.sh --embedding clones and compiles it in $HOME "
         "(or LLAMA_SERVER=<path> to use one that lives elsewhere)")
    return None


def embedding(cfg, have, build=False):
    step("e", "the embedding sidecar (serve/embed.sh: llama-server + Qwen3-Embedding-0.6B q8_0 on :8090, CPU)")
    note("why CPU: ~1.35 GiB spare a card at util 0.96 (README 'VRAM per card') -- a second process does not "
         "fit; and ~11 of 192 EPYC cores busy with the engine (notes/native-4xr9700.md), so the rest are free")
    so = llama_build(cfg, want=build)
    gguf = Path.home() / "models/qwen3-embedding-0.6b-q8_0.gguf"
    if gguf.exists():
        ok(f"{gguf}: {gguf.stat().st_size / 1e9:.2f} GB -- mmap page cache, not anon: the WEIGHTS never enter "
           "the swap or the Committed_AS that wedged the box on 2026-11-01 (notes/load-oom-svm-wedge.md). The "
           "KV pool does: measured +2.64 GiB of Committed_AS at -c 8192 (977 MB of it anon KV, the 618 MB of "
           "weights clean file-backed) -- notes/embed-sidecar-box-test.md")
    else:
        note(f"{gguf} missing -- serve/embed.sh fetches it (~0.65 GB) on its first run, or:")
        say("      hf download unsloth/Qwen3-Embedding-0.6B-GGUF qwen3-embedding-0.6b-q8_0.gguf "
            "--local-dir ~/models && mv ~/models/qwen3-embedding-0.6b-q8_0.gguf " + str(gguf))
    try:
        with urllib.request.urlopen("http://127.0.0.1:8090/health", timeout=3) as r:
            up = r.status == 200
    except (urllib.error.URLError, OSError):
        up = False
    if up:
        ok("embedding up on :8090 -- /v1/embeddings is OpenAI-compatible, point RAG clients at http://<box>:8090/v1")
        say("      monitor: http://127.0.0.1:8091/ is watching" if _up("http://127.0.0.1:8091/health") else
            "      monitor: bash serve/embed-watch.sh   (http://127.0.0.1:8091/ -- webui.py is vLLM-shaped, "
            "this one speaks llama.cpp)")
    else:
        note("not running: bash serve/embed.sh starts it (STOP=1 stops it, DRY=1 previews, LLAMA_SERVER=... "
             "names the binary)"
             + ("" if so else " -- build it first: ./setup.sh --embedding"))
    say("      and the reranker beside it: TASK=rerank CTX=4096 bash serve/embed.sh (:8092, cores 32-47 of its "
        "own) -- measured next to the standing launch: 1,014 tok/s, and the engine's ITL did not move "
        "(notes/embed-sidecar-box-test.md)")


def embedding_launch(cfg):
    """serve/embed.sh as --launch's target: no GPUs to be free, no tmux pane worth having, no engine context
    to compare -- the whole contract is /health on 8090, so none of launch()'s GPU machinery applies."""
    script = ROOT / "serve/embed.sh"
    log = os.environ.get("LOG") or str(Path.home() / "logs" / f"embed-{time.strftime('%H%M%S')}.log")
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    so = llama_build(cfg, want=True)
    if so is None:
        return
    if asks_whether_to_start(cfg) and not ask(cfg, "start the embedding sidecar on :8090 (no GPU, ~0.65 GB mmap)?"):
        return
    say(f"    $ LLAMA_SERVER={so} LOG={log} bash serve/embed.sh")
    with open(log, "ab") as fh:
        proc = subprocess.Popen(["bash", str(script)], stdout=fh, stderr=subprocess.STDOUT,
                                env=dict(os.environ, LLAMA_SERVER=so, LOG=log), start_new_session=True)
    say("    waiting for http://127.0.0.1:8090/health ...")
    what, dt, why = wait_up(8090, proc, None, secs=120)
    if what == "up":
        ok(f"up after {dt:.0f} s -- POST http://127.0.0.1:8090/v1/embeddings (OpenAI shape; the query side "
           "carries the Instruct/Query prompt, the document side plain text)")
    elif what == "dead":
        bad(f"{why} without answering /health. {log}, last lines:")
        for ln in tail_log(log, 10):
            say(f"         | {ln}")
        say("    stop and relaunch:  STOP=1 bash serve/embed.sh")
    else:
        warn(f"not answering /health within {dt / 60:.0f} min: {log}, last lines:")
        for ln in tail_log(log, 10):
            say(f"         | {ln}")


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


def corpse_session(sess, left, rec):
    """Is a live tmux session a corpse? (close it?, why)

    start_tmux makes the session the liveness probe, and it is a good probe while anything is running -- but
    neither a crash nor STOP closes it, and that turns the next launch away. Measured 2026-11-15: the launch
    faulted at 06:28, `STOP=1` cleared its ranks, and at 06:45 `setup.py --launch native-flashnext.sh` still
    refused with "a tmux session named r9700-native-flashnext is already up" -- a session holding nothing, 17
    minutes after the engine stopped, was keeping the box down.

    Refusing stays the default: a session with ranks in it, or one the record does not vouch for, or one whose
    launch might still be loading, is not ours to close. Only the record's own session, with no VLLM process in
    it and a record that says the launch is over (stamped dead, or it answered /health once and now holds
    nothing), gets closed -- because that is the state in which the session IS the launch, and the launch is gone.
    """
    if left:
        return False, f"{left} VLLM process(es) are running in it"
    if not rec or (rec.get("session") or None) != (sess or None):
        return False, "the record does not name this session, so it is somebody else's pane"
    if rec.get("died_at") is not None:
        return True, ("the record says this launch died "
                      f"{time.strftime('%H:%M:%S', time.localtime(rec['died_at']))} and nothing is running in it")
    if rec.get("up_at") is not None:
        return True, "it answered /health once and now holds no VLLM process: the pane outlived the engine"
    return False, ("the record does not say this launch is over -- a load that has not answered /health yet has "
                   "no ranks either, and that is not a corpse")


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


def restart_after_update(cfg):
    """--restart: bring the launch this box runs back up on what was just pulled. Three rules, each one
    someone's incident: the checks above must be GREEN -- a red check says the updated tree is known-broken,
    and stopping the one working thing for it turns an update into an outage (which is also why every run
    WITHOUT this flag leaves a running launch alone, update.sh's header); the stop is the recipe's OWN
    STOP=1 -- the command stop_wedge proved against ranks that ignore SIGTERM, where an ad-hoc pkill would
    be a second, worse opinion about the same processes; and the restart goes back through setup.py --yes
    --launch, the tested restart decision, so the /health wait, the context check, the gateway bring-up and
    the watcher arming are the ones this file already runs rather than a second launch nothing tests.
    Returns True when this run ends here; False only when there was nothing to restart, and the ordinary
    offer below runs."""
    rec = state_read("launch.json")
    if not rec or not rec.get("recipe"):
        note("--restart: no launch recorded here (.r9700-state/launch.json): this flag restarts what runs; "
             "starting a fresh one is what the offer below is for")
        return False
    if not (ROOT / "serve" / rec["recipe"]).exists():
        bad(f"--restart: the launch on the record runs serve/{rec['recipe']}, which this checkout no longer "
            f"has (renamed in what was pulled?): the stack is left running it",
            f"./stop.sh if you want it down; ./setup.sh --launch names the recipes this pull has")
        return True
    n = left_vllm()
    if not n:
        note(f"--restart: serve/{rec['recipe']} is not running -- why is above, in launch_status: nothing "
             f"to stop, and the offer below is what brings it back")
        return False
    if FAIL_STEPS:
        say(f"--restart: NOT stopping anything: {len(FAIL_STEPS)} check(s) above said FAIL, and the running "
            f"launch is the one thing on this box that demonstrably works")
        for m in FAIL_STEPS[:4]:
            say(f"         FAIL {m}")
        say(f"         the stack keeps running what it was started with: fix those, then ./stop.sh and "
            f"./setup.sh --launch {rec['recipe'][:-3]}")
        return True
    if not ask_no(cfg, f"the checks above are green: stop serve/{rec['recipe']} ({n} vLLM process(es) up) "
                       f"and bring it back up on what was just pulled (GPUs go busy for ~75 s)?"):
        say("--restart declined: the stack keeps running what it was started with, not what was just pulled")
        return True
    say(f"    $ STOP=1 bash serve/{rec['recipe']}")
    rc, last = stop_wedge(rec)
    deadline = time.time() + RESTART_STOP_WAIT
    while left_vllm() and time.time() < deadline:
        time.sleep(3)
    left = left_vllm()
    say(f"    the stop returned {rc}: {last}; ranks after it: {left}")
    if left:
        bad(f"--restart: {left} rank(s) still hold the GPUs {RESTART_STOP_WAIT} s after the recipe's own "
            f"stop -- not restarting into that: two launches on four cards is the 11-14 fault again",
            f"./stop.sh walks everything and counts what is left, then ./setup.sh --launch "
            f"{rec['recipe'][:-3]}")
        return True
    cmd = restart_cmd(rec)
    say("    ranks cleared; bringing it back through the tested restart path:")
    say("    $ " + " ".join(cmd[1:] if cmd[0] == sys.executable else cmd))
    subprocess.call(cmd)
    return True


def launch(cfg, have="docker"):
    step(8, "a launch")
    rs = recipes()
    for name, head in rs:
        container = container_recipe(name)
        needs = "" if not container or have == "docker" else "   [container recipe: setup.sh --docker turns that path on]"
        say(f"    serve/{name}   {head[:90]}{needs}")
    launch_status()
    if cfg.check:
        if getattr(cfg, "restart", False):
            note("--check restarts nothing: a full run with --restart asks first, then stops a running "
                 "launch with its own STOP=1 and re-runs it through setup.py --yes --launch, only once "
                 "every check above it is green")
        return
    if getattr(cfg, "restart", False) and restart_after_update(cfg):
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
    if pick == "embed.sh":
        embedding_launch(cfg)   # the CPU sidecar: no GPU pre-flight, no tmux, no context comparison
        return
    if container_recipe(pick) and have != "docker":
        bad(f"serve/{pick} runs through serve.sh, and this run takes the native path",
            "./setup.sh --docker --launch " + pick + "   (or run a native recipe: serve/native-flashnext.sh)")
        return
    env = dict(os.environ, WEBUI=WEBUI_ENV[cfg.webui])
    # the port THIS launcher will bind, not the one setup.py's environment happens to carry -- see launcher_var
    port = int(launcher_var((ROOT / "serve" / pick).read_text(errors="replace"), "PORT", "8080"))
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
                kill, why = corpse_session(sess, left_vllm(), state_read("launch.json"))
                if not kill:
                    bad(f"a tmux session named {sess} is already up -- not starting a second launch into it",
                        f"tmux attach -t {sess}   ({why}; or ./setup.sh --no-tmux for a plain detached start)")
                    return
                say(f"    the session {sess} is a corpse: {why} -- closing it so the launch has a pane to "
                    f"start in (that is what kept this box down for 17 min on 2026-11-15)")
                out(["tmux", "kill-session", "-t", sess])
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
                                # what to re-run, if a watcher is ever told to: the recipe alone is not this launch,
                                # because the flags typed on top of it and the tmux choice are part of the shape
                                "rest": list(cfg.rest), "no_tmux": bool(cfg.no_tmux),
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
        pid = arm_watch(port, sess, log)
        if pid:
            say(f"    watching it (pid {pid}): the moment it stops goes into .r9700-state/launch.json as "
                f"died_at/reason, and the same line is appended to {log}. "
                + ("It also restarts the launch on a KNOWN crash signature, up to "
                   f"{RESTART_MAX} time(s) per {RESTART_WINDOW_S // 60} min (WATCH_RESTART=1)."
                   if os.environ.get("WATCH_RESTART", "0") == "1" else
                   "It records and does not restart (notes/launch-methodology.md \u00a74); WATCH_RESTART=1 arms "
                   "the signature-plus-breaker restart.")
                + "")
            if cfg.webui == "same":
                note("this launch carries the Monitor page inside itself (WEBUI=same), so the page goes down "
                     "with it: the record, not a page, is what outlives the death. A sidecar "
                     "(./setup.sh --webui side, or bash serve/webui.sh against a dead launch) is what reports "
                     "it live (notes/webui.md)")
        # The front door, if this box asked for one. serve/native-flashnext.sh ends in `exec vllm serve`, so the
        # launcher cannot start anything after itself; setup.py is the thing that waits for /health, so it is the
        # thing that brings the gateway up. Without this the box comes back from a reboot with a working engine
        # on loopback and a dead LAN port -- the one place clients actually knock.
        gwport = env_file_var((ROOT / "serve" / pick).read_text(errors="replace"), "GATEWAY_PORT")
        if gwport and str(gwport) != str(port):
            g = subprocess.run(["bash", str(ROOT / "serve" / "gateway.sh")], capture_output=True, text=True)
            line = [l for l in (g.stdout + g.stderr).splitlines() if l.strip()][-1:] or ["no output"]
            (ok if g.returncode == 0 else say)(f"front door on :{gwport} -- {line[0]}")
            note("the gateway is not stopped by STOP=1 here: it answers /monitor and the sidecar routes, so it "
                 "stays up to report an engine that went down. STOP=1 bash serve/gateway.sh takes it down.")
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
    ap.add_argument("--embedding", action="store_true",
                    help="the embedding sidecar: check llama.cpp and the GGUF, and build llama.cpp in $HOME "
                         "when missing (the only step that compiles third-party source; an update only "
                         "reports it). Start the sidecar itself with --launch embed.sh or bash serve/embed.sh.")
    ap.add_argument("--restart", action="store_true",
                    help="what ./update.sh --restart runs: after a green walk, ASK, then stop the launch "
                         "this box is running with its own STOP=1, wait for the ranks to clear, and re-run "
                         "it through setup.py --yes --launch. A FAIL above it stops nothing (the running "
                         "launch is the working thing), --check restarts nothing, and --yes does NOT answer "
                         "for you -- what goes down is a working box")
    ap.add_argument("--watch", action="store_true",
                    help="internal (setup.py arms this itself, detached, once a launch answers /health): poll "
                         "that launch and write died_at/reason into .r9700-state/launch.json when it stops. "
                         "Records by default; with WATCH_RESTART=1 it also re-runs the recorded launch through "
                         "setup.py, but only for a death whose log carries a known crash signature, only with no "
                         "VLLM ranks left, and only inside RESTART_MAX per RESTART_WINDOW_S "
                         "(notes/launch-methodology.md \u00a74). PORT, LOG and WATCH_SESSION come from the "
                         "environment.")
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
    if cfg.watch:                            # no box walk: the watcher has one job and no questions
        sys.exit(watch_loop(watch_port(),
                            os.environ.get("WATCH_SESSION") or None, os.environ.get("LOG") or None))
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
    embedding(cfg, have, build=cfg.embedding)
    if cfg.update:
        step("u", "what an update does NOT touch: the weights in ~/models, the caches "
                  "(~/vllmstock-cache-*, ~/vllm-native-*), and any launch that is running")
    launch(cfg, have)
    say("\nnotes: README.md (build/serve/check), PROGRESS.md (what was measured and rejected), "
        "host/README.md (BAR0/ACS/switch chains), notes/webui.md (the pages).")


if __name__ == "__main__":
    main()
