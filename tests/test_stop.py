#!/usr/bin/env python3
"""stop.sh's own checks -- host-safe: nothing here kills anything, and the one run it makes is DRY --check.

What this guards is the coupling between the stopper and the launchers: the patterns it kills with must stay
readable out of the launchers' own files (a pattern in two files is a pattern that drifts), the sidecar tasks
it walks must stay readable out of serve/embed.sh's own case block, and the flags it documents must stay the
flags it implements.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
fails = 0


def ck(name, cond, extra=""):
    global fails
    print(("ok   " if cond else "FAIL ") + name + (f"  [{extra}]" if not cond and extra else ""))
    if not cond:
        fails += 1


STOP = ROOT / "stop.sh"
SRC = STOP.read_text()

rc = subprocess.run(["bash", "-n", str(STOP)], capture_output=True, text=True)
ck("bash -n stop.sh", rc.returncode == 0, rc.stderr.strip())
rc = subprocess.run(["bash", "-n", str(ROOT / "serve/gateway.sh")], capture_output=True, text=True)
ck("bash -n serve/gateway.sh", rc.returncode == 0, rc.stderr.strip())

# Usage: an unknown flag is a refusal, not a silent full stop.
rc = subprocess.run(["bash", str(STOP), "--bogus"], capture_output=True, text=True)
ck("an unknown flag exits 2 and names itself", rc.returncode == 2 and "--bogus" in rc.stderr,
     f"rc={rc.returncode} err={rc.stderr[:120]!r}")

# The quiet path: --check forces DRY, and a DRY run must only ever print commands.
rc = subprocess.run(["bash", str(STOP), "--check"], capture_output=True, text=True, timeout=180)
ck("./stop.sh --check exits 0 and says so in the first line",
   rc.returncode == 0 and "check only, nothing is stopped" in rc.stdout,
     f"rc={rc.returncode} out={rc.stdout[:200]!r}")
ck("the check names every component it counts: vLLM, sidecars, watcher, Monitor page, and the door",
   all(s in rc.stdout for s in ("vLLM", "llama-server", "embedding monitor", "Monitor page", "door")),
   rc.stdout[-300:])
gate_lines = [l for l in rc.stdout.splitlines() if "gateway" in l]
ck("the door is only named in a full stop (no flags): --engine/--embedding must leave it up for the other half",
   "--engine) ENGINE=1; EMB=0; GATE=0" in SRC and "--embedding) EMB=1; ENGINE=0; GATE=0" in SRC,
   str(gate_lines[:2]))

# The kill patterns stay in the launchers: pats() reads pkill -9 lines, pat_any() reads PAT= lines, and the
# alternation stop.sh counts vLLM with must cover exactly the patterns the engine launcher's STOP kills with.
gated = (ROOT / "serve/gateway.sh").read_text()
pat_g = re.search(r'^PAT="([^"]+)"', gated, re.M)
ck("serve/gateway.sh grew the PAT= line stop.sh reads -- before it, the door was the one process the honest "
   "tail could not count", pat_g is not None, gated[:0] or "no PAT=")
nf = (ROOT / "serve/native-flashnext.sh").read_text()
engine_pats = set(re.findall(r"pkill -9 -f '([^']*)'", nf))
left = re.search(r"left\(\) \{ count '([^']*)'; \}", SRC).group(1)
ck("stop.sh's vLLM counter covers both pkill patterns the engine launcher kills with",
   engine_pats and all(p in left.split("|") for p in engine_pats), f"{engine_pats} vs {left}")
for s in ("embed-watch.sh", "webui.sh", "embed.sh"):
    p = re.search(r'^PAT="([^"]+)"', (ROOT / "serve" / s).read_text(), re.M)
    ck(f"serve/{s} keeps a PAT= line for the stopper to read", p is not None)

# The sidecar list is read out of the launcher's case block, so a third task there is stopped without an edit here.
emb = (ROOT / "serve/embed.sh").read_text()
block = re.search(r'case "\$TASK" in(.*?)^esac', emb, re.S | re.M)
tasks = set(re.findall(r'^  ([a-z][a-z0-9]*)\)', block.group(1), re.M)) if block else set()
ck("serve/embed.sh's TASK block still names the two sidecars stop.sh walks (embed on :8090, the reranker on :8092)",
   {"embed", "rerank"} <= tasks, str(tasks))
ck("and stop.sh reads THAT block, not a list it keeps", 'case "\\$TASK"' in SRC, SRC[:0] or "tasks() rewritten?")

print(("PASS" if fails == 0 else f"FAIL ({fails})") + f" ({fails} failing)")
sys.exit(1 if fails else 0)
