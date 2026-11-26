#!/usr/bin/env python3
"""tests/test_convertcache.py - the disk cache for the load-time requant, without a GPU or torch.

The requant is the 30-minute cold start (Super VL 120B: 40,960 expert matrices, every launch until
2026-11-26 paid it again), and its cache decides whether a launch loads the weights the checkpoint has or
the ones the cache claims - which makes the key the whole test surface:

  (1) the fingerprint must be stable across reads, must change when a shard's size changes and when a
      fingerprinted config file is edited, and must stay silent (no cache, no crash) when the model is not
      a local directory at all;
  (2) put -> load round-trips, and a file written under other shapes, a corrupt file, and R9K_CONVERT_CACHE=0
      are all plain misses - a cache fault may never become a failed load;
  (3) the files land directly under <dir>/<fingerprint>/ with a sanitized name, so a TP rank cannot read
      another rank's entry by accident and a layer named like a path cannot write outside the fingerprint
      directory.

Run: ./.venv/bin/python3 tests/test_convertcache.py    (stdlib only; stubs torch when the host has no torch)
"""
import os
import pickle
import re
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

fail = 0


def ck(name, cond, detail=""):
    global fail
    if not cond:
        fail += 1
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond or not detail else f"  {detail}"))


try:                                                    # the host venv holds no torch; the module must not need it
    import torch                                        # noqa: F401
except Exception:
    class _TorchStub:
        """Only the surface convertcache uses: save(obj, path), load(path, map_location, weights_only)."""

        @staticmethod
        def save(obj, path):
            with open(path, "wb") as f:
                pickle.dump(obj, f)

        @staticmethod
        def load(path, map_location=None, weights_only=True):
            with open(path, "rb") as f:
                return pickle.load(f)

    sys.modules["torch"] = _TorchStub()

import r9700_vllm.quant.convertcache as CC             # noqa: E402


class FakeT:
    """A Parameter stand-in: the cache must detach() and cpu() whatever it is handed, then store plain data."""

    def __init__(self, v):
        self.v = v

    def detach(self):
        return self

    def cpu(self):
        return self.v


tmp = tempfile.mkdtemp(prefix="r9700-cc-")
model = os.path.join(tmp, "model")
cache = os.path.join(tmp, "vllm-cache", "r9700-convert")
os.makedirs(model)
Path(model, "config.json").write_bytes(b'{"architectures": ["X"]}\n')
Path(model, "hf_quant_config.json").write_bytes(b'{"quant_method": "modelopt_mixed"}\n')
Path(model, "model-00001-of-00002.safetensors").write_bytes(b"a" * 100)
Path(model, "model-00002-of-00002.safetensors").write_bytes(b"b" * 50)
CC._model_dir = lambda: model

_saved = {k: os.environ.pop(k, None) for k in ("R9K_CONVERT_CACHE", "VLLM_CACHE_ROOT")}
os.environ["VLLM_CACHE_ROOT"] = os.path.join(tmp, "vllm-cache")

# ------------------------------------------------------------------ knob: where the cache lives at all
CC._reset()
ck("R9K_CONVERT_CACHE unset resolves under VLLM_CACHE_ROOT", CC._resolve() == cache, str(CC._resolve()))

os.environ["R9K_CONVERT_CACHE"] = os.path.join(tmp, "elsewhere")
CC._reset()
ck("a path relocates it", CC._resolve() == os.path.join(tmp, "elsewhere"), str(CC._resolve()))

os.environ["R9K_CONVERT_CACHE"] = "0"
CC._reset()
ck("0 turns it off", CC._resolve() is None, str(CC._resolve()))
CC.put("layers.0.mlp/w13", {"packed": FakeT([1]), "scale": FakeT([2])}, [(2, 4, 8), (2, 4, 2)])
ck("and off means no file and a miss", CC.load("layers.0.mlp/w13", [(2, 4, 8), (2, 4, 2)]) is None
   and not any(p.suffix == ".pt" for p in Path(tmp).rglob("*.pt")))

os.environ.pop("R9K_CONVERT_CACHE", None)

# --------------------------------------------------------------------------- fingerprint of the checkpoint
CC._reset()
fp1 = CC._fingerprint()
ck("fingerprint is a sha256 hex", fp1 is not None and re.fullmatch(r"[0-9a-f]{64}", fp1) is not None, str(fp1))
ck("and stable across reads", CC._fingerprint() == fp1)

Path(model, "model-00002-of-00002.safetensors").write_bytes(b"b" * 60)          # a shard grew
CC._reset()
fp2 = CC._fingerprint()
ck("a shard changing size changes the fingerprint", fp2 != fp1, f"{fp2[:12]} vs {fp1[:12]}")

Path(model, "model-00002-of-00002.safetensors").write_bytes(b"b" * 50)          # back
Path(model, "config.json").write_bytes(b'{"architectures": ["X"], "r9700": true}\n')
CC._reset()
ck("and editing config.json changes it (the checkpoint is not what it was)", CC._fingerprint() != fp1)
Path(model, "config.json").write_bytes(b'{"architectures": ["X"]}\n')
CC._reset()
ck("putting the file back puts the fingerprint back", CC._fingerprint() == fp1)

# ------------------------------------------------------------------------ round-trip and the honest miss
CC._reset()
tag = "model.layers.3.mlp.experts/w13"
shapes = [(2, 4, 8), (2, 4, 2)]
CC.put(tag, {"packed": FakeT([1, 2, 3]), "scale": FakeT([7])}, shapes)
files = list(Path(cache).rglob("*.pt"))
ck("put wrote exactly one file", len(files) == 1, str(files))
rel = files[0].relative_to(cache) if files else Path("x/y")
ck("one directory deep - the fingerprint dir - and no path separators left in the name",
   len(rel.parts) == 2 and "/" not in rel.name and re.fullmatch(r"[A-Za-z0-9._-]+\.pt", rel.name) is not None,
   str(rel))
got = CC.load(tag, shapes)
ck("load returns what put stored, tensors detached to plain data",
   got == {"packed": [1, 2, 3], "scale": [7]}, str(got))
ck("a different shape is a different file: miss, not someone else's tensor",
   CC.load(tag, [(2, 4, 8), (2, 4, 4)]) is None)
CC.put(tag + "x", {"packed": FakeT([1]), "weight": FakeT([2])}, shapes)          # right shape, wrong names:
ck("tensor names that do not match this layer are a miss, not a KeyError at the caller's index",      # same dir
   CC.load(tag + "x", shapes, keys=("packed", "scale")) is None)

with open(files[0], "wb") as f:
    f.write(b"garbage" * 10)                                                   # torn by hand; the tmp/rename
ck("a corrupt file is a miss and no raise (a cache fault must never fail a load)",                # race did
   CC.load(tag, shapes) is None)                                              # the same thing naturally

CC.put(tag, {"packed": FakeT([1, 2, 3]), "scale": FakeT([7])}, shapes)         # repaired: hit again
ck("and after the miss the next put lands and hits again", CC.load(tag, shapes) is not None)

# ------------------------------------------------------------------------- the model is not here at all
saved_dir = CC._model_dir
CC._model_dir = lambda: "nvidia/Some-NVFP4"                                   # an HF id: nothing local to hash
CC._reset()
ck("a non-directory model keeps the cache inert: no files, no fingerprint",
   CC.load(tag, shapes) is None and CC.put(tag, {"packed": FakeT([1])}, [(2, 4, 8)]) is None
   and CC._fingerprint() is None)
CC._model_dir = saved_dir

for k, v in _saved.items():                                                    # leave the environment as found
    if v is None:
        os.environ.pop(k, None)
    else:
        os.environ[k] = v

print("PASS" if fail == 0 else "FAIL", f"({fail} failing)")
sys.exit(1 if fail else 0)
