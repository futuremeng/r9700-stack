#!/usr/bin/env python3
"""Bring this repository up on a box, or re-check a box that already runs it.

The box this repo lives on is a rack: R9700s passed through to a VM with their full 32 GB BAR0, the serving
stack in a pinned stock-vLLM image, the plugin bind-mounted from a SECOND checkout ($HOME/r9700-build/repo),
weights in ~/models. So "install" here is not a pip install: docker/Dockerfile pip-installs the plugin inside
the image, and the venv this script makes holds no torch at all -- the host Python is for the tests that need
no GPU, for bench/, for serve/webui.py (stdlib only) and for this file.

  ./setup.sh                    check every step, make what is ours to make, then offer a launch
  ./setup.sh --check            report only: nothing runs that changes anything, no launch offered
  ./setup.sh --yes              no prompts (build the image, clone the build repo, no launch)
  ./setup.sh --update           what update.sh runs after a git pull: pin, build repo, tests
  ./setup.sh --rebuild          build the image again even though the pin says it is current
  ./setup.sh --launch 27b       start that recipe instead of asking (none = stop after the checks)
  ./setup.sh --no-webui         launch without the Monitor/About sidecar
  --img NAME --repo PATH        the image tag (r9700/vllm:dev) and the build repo serve.sh mounts

Every number and requirement below is read from this repository or measured on the box, with the file it came
from named. Nothing here invents a driver version: the host-side BAR/ACS/switch work is host/README.md, and
this script only checks what a running container would need from the guest.
"""
import argparse
import json
import os
import re
import resource
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


# --------------------------------------------------------------------- facts read from this repo ------
def dockerfile_base():
    """The nightly this image is pinned to (docker/Dockerfile), so a pin change is visible."""
    try:
        m = re.search(r"^ARG BASE=(\S+)", (ROOT / "docker" / "Dockerfile").read_text(), re.M)
        return m.group(1) if m else "?"
    except OSError:
        return "?"


def recipes():
    """serve/*.sh with the first line of their own header: the measured claim is in the file, not here."""
    out_ = []
    for p in sorted((ROOT / "serve").glob("*.sh")):
        if p.name in ("serve.sh", "webui.sh", "deploy.sh"):
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


def host(cfg):
    step(1, "the guest the container will run in (host-side BAR/ACS work is host/README.md, not this file)")
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


def resource_limits():
    return resource.getrlimit(resource.RLIMIT_MEMLOCK)

def docker(cfg):
    step(2, "docker (serve.sh calls `sudo docker run`: daemon up, and the device/group flags it needs)")
    if not shutil_which("docker"):
        note("no docker on PATH: this box is a native-path box (serve/native-flashnext.sh runs vLLM out of a "
             "venv instead of the pinned image). Its own pre-flight checks are the thing to read.")
        native(cfg)
        return "native"
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
    venv = Path(os.environ.get("VENV", Path.home() / "venv-r9700"))
    py = venv / "bin" / "python3"
    if not py.exists():
        warn(f"native path needs a venv with stock ROCm vLLM at {venv} (missing)",
             "python3 -m venv ~/venv-r9700 && ~/venv-r9700/bin/pip install vllm --extra-index-url "
             "https://download.pytorch.org/whl/rocm6.4 && pip install -e . (the ROCm wheel build matters: the "
             "box's own is what notes/native-4xr9700.md measures)")
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
    so = Path(os.environ.get("REPO", str(ROOT))) / "r9700_vllm" / "kernels" / "libr9k.so"
    if so.exists():
        ok(f"libr9k.so built for the native path: {so} ({so.stat().st_size // 1024} KiB)")
    else:
        warn(f"{so} missing -- serve/native-flashnext.sh refuses without it",
             "cd kernels && HIPCFLAGS=-I$HOME/opt/rocm-dev/opt/rocm/core-10.1/include ./build.sh "
             "($HOME/opt/rocm-dev is where this box's hipcc headers live)")


def image(cfg, have="docker"):
    step(3, f"the image {cfg.img} (stock vLLM ROCm 10 nightly + the plugin; the kernels compile inside)")
    pin = dockerfile_base()
    note(f"docker/Dockerfile is pinned to {pin}")
    if have == "native":
        note("nothing to check or build here without docker: the container path (serve/serve.sh, serve/27b.sh, "
             "serve/flashnext.sh) needs it, the native path (serve/native-flashnext.sh) does not")
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
                 f"plugin from this checkout (REPO={ROOT})")
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


def models(cfg):
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
            warn(f"/models/{n} missing (needed by {who})"
                 + (f"; serve/deploy.sh fetches it from {src}" if src else ""),
                 "bash serve/deploy.sh (~25 GB through the image)" if src else "see CREDITS.md for where the "
                 "weights come from; they are third-party and are not part of this repository")
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
    t = ROOT / "tests" / "test_webui.py"
    if cfg.check:
        note(f"would run: {py} tests/test_webui.py (and bench/bringup-tests.sh inside the image, on free GPUs)")
    elif t.exists():
        if ask(cfg, "run tests/test_webui.py (the Monitor/About sidecar, fake vLLM, fake sysfs)?"):
            rc = run([py, str(t)], cfg, timeout=600)
            (ok if rc == 0 else bad)(f"tests/test_webui.py {'passed' if rc == 0 else 'FAILED'}")
        else:
            note(f"skipped: {py} tests/test_webui.py")
    else:
        warn("tests/test_webui.py missing")
    note("the kernel and plugin unit tests need free GPUs and run inside the image: bench/bringup-tests.sh")


def sidecar(cfg):
    step(7, "the Monitor / About pages (notes/webui.md)")
    ok("stdlib only, nothing to install")
    note("watch a launch that is already up:  bash serve/webui.sh          (http://127.0.0.1:8081/monitor)")
    note("start it with the launch, with this launch's recipe:  WEBUI=1 bash serve/serve.sh")
    note("it records per-request detail only in --proxy mode, which costs ~1 ms per request "
         "(bench/webui_ab.py); observe mode changes nothing in the inference path")


def launch(cfg, have="docker"):
    step(8, "a launch")
    rs = recipes()
    for name, head in rs:
        needs = "" if name.startswith("native") or have == "docker" else "   [needs docker: this box is a native-path box]"
        say(f"    serve/{name}   {head[:90]}{needs}")
    if cfg.check:
        return
    pick = cfg.launch
    if pick == "none":
        return
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
    env = dict(os.environ, WEBUI="0" if cfg.no_webui else "1")
    say(f"    $ WEBUI={'0' if cfg.no_webui else '1'} bash serve/{pick}")
    if not cfg.yes and not ask(cfg, f"start serve/{pick} now (GPUs go busy for ~75 s)?"):
        return
    subprocess.call(["bash", str(ROOT / "serve" / pick), *cfg.rest], env=env)
    port = int(os.environ.get("PORT", 8080))
    say(f"    waiting for http://127.0.0.1:{port}/health ...")
    t0 = time.time()
    while time.time() - t0 < 420:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    ok(f"up after {time.time() - t0:.0f} s")
                    break
        except (urllib.error.URLError, OSError):
            time.sleep(5)
    else:
        warn("not answering /health within 7 min: the launch log is the place to look")
    if not cfg.no_webui:
        say(f"    monitor: http://127.0.0.1:{port + 1}/monitor    about: http://127.0.0.1:{port + 1}/about")
    note("the first launch of a NEW configuration compiles and keeps ~0.45 GiB that the next one hands to the "
         "KV pool (serve/flashnext.sh): restart once, then measure")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report only; run nothing that changes anything")
    ap.add_argument("--yes", action="store_true", help="no prompts")
    ap.add_argument("--update", action="store_true", help="what update.sh runs after a git pull")
    ap.add_argument("--rebuild", action="store_true", help="build the image even if the pin matches")
    ap.add_argument("--launch", default=None, help="serve/<name>.sh to start, or none")
    ap.add_argument("--no-webui", action="store_true", help="launch without the Monitor/About sidecar")
    ap.add_argument("--img", default=os.environ.get("IMG", "r9700/vllm:dev"))
    ap.add_argument("--repo", default=os.environ.get("REPO", str(Path.home() / "r9700-build" / "repo")))
    cfg, extra = ap.parse_known_args()
    cfg.rest = extra
    say(f"r9700-stack bring-up -- {ROOT}")
    say(f"check-only: {cfg.check}   (every requirement below names the file it came from)")
    host(cfg)
    have = docker(cfg)
    if cfg.rebuild and have == "docker" and not cfg.check:
        out(["docker", "rmi", cfg.img])
        note(f"removed {cfg.img}: the image step will build it again")
    image(cfg, have)
    buildrepo(cfg, have)
    models(cfg)
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
