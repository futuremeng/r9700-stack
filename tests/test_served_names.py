#!/usr/bin/env python3
"""No tool in this tree names a model in its own source -- it asks the serve.

On 2026-11-18 this box switched from Qwen3.8 to SuperVL, and the switch was quiet everywhere except in the tools:
sixteen files under bench/ and tools/ still posted `"model": "Qwen3.8"`. The serve answered every one of them with
`The model `Qwen3.8` does not exist.` -- a 404 with no list of what is served, so it reads like a server bug rather
than a stale string in a file, and the failure showed up at request time in the middle of a measurement run instead
of at launch.

The fix is `bench/served_model.py`: read the name from `/v1/models`, with `MODEL=` as the explicit override for A/B
runs. The first id is the canonical one by construction -- vLLM builds `base_model_paths` in `--served-model-name`
order and its serving layer returns `base_model_paths[0].name` -- and this box serves exactly one name, with no
`Qwen3.8` alias on purpose: a client that still sends that string believes it is talking to a model this box does not
run.

What is guarded here: the drift class itself (a remembered name in a request), the resolver agreeing with what the
launcher actually serves, and needle_long counting its haystacks against the checkpoint the serve has loaded -- an id
from one model and a tokenizer from another mis-counts every tier and errors nowhere.
"""

import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = os.environ.get("R9700_ROOT") or str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, os.path.join(ROOT, "bench"))

fail = 0


def ck(name, cond, detail=""):
    global fail
    if not cond:
        fail += 1
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond or not detail else f"  {detail}"))


def _files(*dirs):
    for d in dirs:
        for p in sorted(Path(ROOT, d).glob("*.py")):
            yield p


# 1. The drift class: a model name remembered in a request or a default. Not a general style rule -- the patterns are
#    the exact shapes that carried the stale name: a literal in a body, a literal default, a literal argv fallback.
_BAD = [
    re.compile(r"""["']model["']\s*:\s*["'][^"'{}]+["']"""),                     # {"model": "Qwen3.8", ...}
    re.compile(r"""\b(?:MODEL|SERVED|SERVED_NAME|MODEL_NAME|EVAL_MODEL)\s*=\s*["'][A-Za-z0-9._-]+["']"""),
    re.compile(r"""(?:os\.environ\.get\([^)]*,|sys\.argv\[[^\]]+\][^\n]*else|default=)\s*["'][A-Za-z0-9._-]*(?:Qwen|VL|Nemotron|DeepSeek)[A-Za-z0-9._-]*["']"""),
    re.compile(r"""TOKENIZER\s*=\s*[^\n]*else\s*["']/"""),                      # a hardcoded tokenizer path
]
_offenders = []
for p in _files("bench", "tools"):
    if p.name == "served_model.py":
        continue                                    # its docstring quotes the old name to explain this file
    text = p.read_text(errors="replace")
    for rx in _BAD:
        for m in rx.finditer(text):
            _offenders.append(f"{p.relative_to(ROOT)}: {m.group(0)[:70]}")
ck("no bench/ or tools/ file carries a model name in its own source",
   not _offenders, "; ".join(_offenders[:4]))

# 2. The resolver and the launcher must agree about the box's one name, or a green bench run is measuring something
#    the launcher did not start.
_ltxt = Path(ROOT, "serve", "native-flashnext.sh").read_text()
# The launcher names the served model per family: the Nemotron-H block sets one, everything else falls back to the
# ${SERVED_NAME:-...} in the argv. Both are read, so this guard follows the box across a checkpoint swap instead of
# pinning whichever family happened to be standing the day it was written (it was written the day Super VL took the
# cards, and failed the evening Qwen came back -- a true failure, but not the one the guard is for).
_svl = re.search('FAM"\\s*=\\s*nemotronh[^A-Za-z0-9_].*?:\\s*\\$\\{SERVED_NAME:=([^}]*)\\}', _ltxt, re.S)
_oth = re.search(r'\$\{SERVED_NAME:-([A-Za-z0-9._-]+)\}', _ltxt)
_name_svl = _svl.group(1).split()[0] if _svl else ""
_name_oth = _oth.group(1) if _oth else ""
ck("the launcher names a served model for each family it knows",
   bool(_name_svl) and bool(_name_oth), f"nemotronh={_name_svl!r} other={_name_oth!r}")

_base = os.environ.get("BASE", "http://127.0.0.1:8080")
try:
    with urllib.request.urlopen(_base + "/v1/models", timeout=10) as r:
        _ids = [d.get("id") for d in json.loads(r.read()).get("data", []) if d.get("id")]
except (urllib.error.URLError, OSError, ValueError) as e:
    _ids = []
    ck("live-serve name checks skipped: nothing answering on " + _base, True, str(e))
if _ids:
    import served_model as SM
    _card = SM.served_card(_base)
    _fam = "nemotronh" if bool(_card[1]) and Path(_card[1], "config.json").exists() and \
        "NemotronH_Omni_Reasoning_V3" in Path(_card[1], "config.json").read_text() else "other"
    _want, _stale = (_name_svl, _name_oth) if _fam == "nemotronh" else (_name_oth, _name_svl)
    ck("/v1/models answers exactly ONE name for this box -- no alias was quietly added",
       len(_ids) == 1, f"names: {_ids}")
    ck(f"the name the bench resolves is the launcher's default for the family it runs ({_fam})",
       SM.resolve(_base) == _want == _ids[0], f"serve {_ids[0]!r}, launcher {_want!r}, resolver {SM.resolve(_base)!r}")
    ck("...and the other family's name is refused rather than served under the rug",
       b"does not exist" in subprocess.run(["curl", "-s", "-X", "POST", _base + "/v1/chat/completions",
                                            "-H", "Content-Type: application/json",
                                            "-d", json.dumps({"model": _stale,
                                                              "messages": [{"role": "user", "content": "hi"}],
                                                              "max_tokens": 1})],
                                           capture_output=True, timeout=60).stdout, f"posted {_stale!r}")
    ck("needle_long counts haystacks against the checkpoint the serve itself reports",
       bool(_card[1]) and Path(_card[1], "config.json").exists(), f"card root: {_card[1]!r}")

print("PASS (0 failing)" if not fail else f"FAIL ({fail} failing)")
sys.exit(1 if fail else 0)
