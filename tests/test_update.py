#!/usr/bin/env python3
"""update.sh's own checks -- host-safe: the one run it makes is in a copy with no .git, where the script
must stop before its exec of setup.sh --update could ever start.

What this guards is the exit itself: the branch that tells a non-git copy to re-download has to END the
script, not fall through into the build-repo print and the exec, and the guard has to recognize a linked
worktree (where .git is a file, not a directory) as a checkout that can pull itself.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
fails = 0


def ck(name, cond, extra=""):
    global fails
    print(("ok   " if cond else "FAIL ") + name + (f"  [{extra}]" if not cond and extra else ""))
    if not cond:
        fails += 1


UPDATE = ROOT / "update.sh"
SRC = UPDATE.read_text()

rc = subprocess.run(["sh", "-n", str(UPDATE)], capture_output=True, text=True)
ck("sh -n update.sh", rc.returncode == 0, rc.stderr.strip())

# The guard tests for a .git ENTRY, not a directory: a linked worktree keeps .git as a file, and telling it
# "not made with git" would silently skip the pull it could have done.
ck("both .git probes look at the entry, not the directory type",
   '[ ! -e .git ]' in SRC and '[ -e "$repo/.git" ]' in SRC,
   [l for l in SRC.splitlines() if ".git" in l and "if [" in l])

# A copy that was not made with git must be told so and STOPPED there. In the copy the guard is the pure
# [ ! -e .git ] test, so this run touches no git and reaches no exec -- and if the exit were ever dropped
# again, the fall-through would print the build-repo line and fail on the missing ./setup.sh next door.
with tempfile.TemporaryDirectory() as td:
    copy = Path(td) / "r9700-stack"
    copy.mkdir()
    shutil.copy2(UPDATE, copy / "update.sh")
    rc = subprocess.run(["sh", str(copy / "update.sh")], capture_output=True, text=True, timeout=60)
    ck("a copy without .git exits 1", rc.returncode == 1,
       f"rc={rc.returncode} out={rc.stdout[:200]!r} err={rc.stderr[:200]!r}")
    ck("it names the problem and the fix in the same breath",
       "not made with git" in rc.stdout and "git clone" in rc.stdout, rc.stdout[:200])
    ck("it stopped before the build-repo print and the setup exec",
       "The build repo serve.sh mounts" not in rc.stdout, rc.stdout[-200:])

print(f"{fails} failed")
raise SystemExit(1 if fails else 0)
