"""Disk cache for the MXFP4 requantization done at load (NVFP4 -> MXFP4, and R9K_FP8_TO_MXFP4).

Why: the requant at load is the whole 30-minute cold start. Super VL 120B carries 40,960 expert matrices
and quant/nvfp4.py dequantizes and re-searches every one of them on EVERY launch -- the first cut's own words
were "no offline repack, no extra disk". The conversion is deterministic given the checkpoint bits, so its
output is worth a file: the bill gets paid once per checkpoint instead of once per start.

Every input that could change the answer and is cheap to read goes into the key:

  * a fingerprint of the checkpoint: sha256 over config.json + hf_quant_config.json in full, plus the sorted
    (name, size) of every safetensors shard -- the same level of trust as vLLM's own caches, and the same
    caveat: a shard edited in place under an unchanged name and size is invisible to it;
  * the layer tag and the tensor shapes, so a different TP/EP split of the same checkpoint (which changes the
    shard shapes) cannot read another split's files;
  * the TP/EP rank and sizes. Under moe-TP the experts' N dim is split, so two ranks hold DIFFERENT bytes of
    the same layer behind the SAME shape and name -- with no rank in the key the last rank to boot would
    poison the cache for every other rank. The rank is enough for determinism: rank r of a given topology
    always loads the same bytes of a given checkpoint, and the shapes carry the topology's size.

R9K_CONVERT_CACHE: unset or 1 = on, at $VLLM_CACHE_ROOT/r9700-convert (default ~/.cache/vllm/r9700-convert);
0 = off; any other value = that directory. A model that is not a local directory (an HF id) has nothing local
to fingerprint and the cache stays inert for the process. EVERY failure -- unreadable model dir, corrupt
file, disk full, torch too old for weights_only -- degrades to converting and logs once: a plugin must never
break a load, and this is a plugin's cache.
"""
from __future__ import annotations

import hashlib
import os
import re

try:
    import torch                                            # tests/test_convertcache.py runs this without it
except Exception:                                           # a cache module must not be the reason an import fails
    torch = None

try:
    from vllm.logger import init_logger
    logger = init_logger("vllm." + __name__)
except Exception:
    import logging
    logger = logging.getLogger("vllm." + __name__)

_UNSET = object()
_RESOLVED = _UNSET          # the cache dir once known; None when off or inert
_FP = _UNSET                # checkpoint fingerprint; None once it is known there is no fingerprint to have
_WARNED = False
_STATS = {"hit": 0, "miss": 0, "put": 0}


def _once(msg, *a):
    fn = getattr(logger, "info_once", None) or logger.info
    fn(msg, *a) if a else fn(msg)


def _fail(msg: str) -> None:
    global _WARNED
    if not _WARNED:
        _WARNED = True
        logger.warning("r9700: convert cache: %s (further cache faults stay silent; the load converts "
                       "instead)", msg)


def _reset() -> None:
    """Forget every resolution: for tests, and for a process that read the environment too early."""
    global _RESOLVED, _FP, _WARNED
    _RESOLVED = _UNSET
    _FP = _UNSET
    _WARNED = False
    for k in _STATS:
        _STATS[k] = 0


def _resolve() -> "str | None":
    """The cache directory, or None when off. Read lazily: the plugin is imported before the engine has
    settled VLLM_CACHE_ROOT and the model, so nothing here may be decided at import time."""
    global _RESOLVED
    if _RESOLVED is not _UNSET:
        return _RESOLVED
    val = os.environ.get("R9K_CONVERT_CACHE")
    if val == "0":
        _RESOLVED = None
    elif val and val != "1":
        _RESOLVED = os.path.abspath(os.path.expanduser(val))
    else:
        root = os.environ.get("VLLM_CACHE_ROOT") or os.path.join(os.path.expanduser("~"), ".cache", "vllm")
        _RESOLVED = os.path.join(root, "r9700-convert")
    return _RESOLVED


def _model_dir():
    try:
        from vllm.config import get_current_vllm_config
        return get_current_vllm_config().model_config.model
    except Exception:                                       # not inside a model context yet: ask again next time
        return None


def _fingerprint() -> "str | None":
    """sha256 of the checkpoint's identity: the two json files in full + every safetensors (name, size).

    Resolved once and kept for the process: the checkpoint does not change underneath a running engine, and
    a failure to read it once will not get better per layer, so a model dir that errors degrades to no cache."""
    global _FP
    if _FP is not _UNSET:
        return _FP
    model = _model_dir()
    if not model:
        return None                                         # context not up yet; stays _UNSET, retries
    if not os.path.isdir(model):
        _FP = None                                          # an HF id: nothing local to fingerprint
        return None
    try:
        h = hashlib.sha256()
        for name in ("config.json", "hf_quant_config.json"):
            p = os.path.join(model, name)
            if os.path.isfile(p):
                with open(p, "rb") as f:
                    h.update(f.read())
        shards = []
        for dirpath, dirnames, filenames in os.walk(model):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for fn in filenames:
                if fn.endswith(".safetensors"):
                    fp = os.path.join(dirpath, fn)
                    shards.append((os.path.relpath(fp, model), os.path.getsize(fp)))
        for name, size in sorted(shards):
            h.update(name.encode() + b"\0" + str(size).encode() + b"\0")
        _FP = h.hexdigest()
    except OSError as e:
        _fail(f"checkpoint fingerprint failed ({e}); this load will not use the convert cache")
        _FP = None
    return _FP


def _shard() -> str:
    """The parallel position this tensor's bytes came from -- see the moe-TP note in the docstring."""
    try:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
        tp, tps = get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
    except Exception:
        tp, tps = 0, 1
    try:
        from vllm.distributed.parallel_state import get_expert_model_parallel_rank, get_expert_model_parallel_world_size
        ep, eps = get_expert_model_parallel_rank(), get_expert_model_parallel_world_size()
    except Exception:
        ep, eps = 0, 1
    return f"t{tp}of{tps}e{ep}of{eps}"


def _path(tag, shapes):
    d = _resolve()
    if not d or not tag:
        return None
    fp = _fingerprint()
    if not fp:
        return None
    dims = "-".join("x".join(str(x) for x in s) for s in shapes)
    name = re.sub(r"[^A-Za-z0-9._-]", "_", tag)
    if len(name) > 96:                                      # 255-byte filename budget
        name = name[:80] + "-" + hashlib.sha256(tag.encode()).hexdigest()[:12]
    return os.path.join(d, fp, f"{name}__{dims}__{_shard()}.pt")


def _count(kind: str) -> None:
    _STATS[kind] += 1
    if kind == "hit" and _STATS["hit"] == 1:
        _once("r9700: convert cache HIT under %s -- the requant this launch does not pay", _RESOLVED)
    elif kind == "put" and _STATS["hit"] == 0 and _STATS["put"] == 1:
        _once("r9700: caching converted weights under %s (R9K_CONVERT_CACHE=0 to turn this off)", _RESOLVED)


def load(tag, shapes, keys=("packed", "scale")):
    """The cached tensors for (this checkpoint, this layer, this shard, these shapes) as CPU tensors, or
    None for miss. The tensor names are validated too: the caller indexes them unguarded, so a name that
    does not match this layer's put is reported as a miss and never reaches that index. Never raises: a
    bad file is a miss, logged once."""
    if torch is None or not tag:
        return None
    try:
        p = _path(tag, shapes)
        if not p or not os.path.exists(p):
            _count("miss")
            return None
        obj = torch.load(p, map_location="cpu", weights_only=True)
        tensors = obj.get("tensors") if isinstance(obj, dict) else None
        if (not isinstance(tensors, dict) or set(tensors) != set(keys)
                or obj.get("shapes") != [tuple(s) for s in shapes]):
            _fail(f"{p} does not match its key; converting instead")
            return None
        _count("hit")
        return tensors
    except Exception as e:                                  # corrupt file, torch without weights_only, ...
        _fail(f"read of {tag} failed ({type(e).__name__}: {e}); converting instead")
        return None


def put(tag, tensors: dict, shapes) -> None:
    """Store the conversion. Atomic (.tmp + os.replace) because TP ranks share the directory; a torn file
    would otherwise be the next launch's corrupt cache entry. Never raises."""
    if torch is None or not tag:
        return
    try:
        p = _path(tag, shapes)
        if not p:
            return
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = f"{p}.{os.getpid()}.tmp"
        torch.save({"shapes": [tuple(s) for s in shapes],
                    "tensors": {k: (v.detach().cpu() if hasattr(v, "detach") else v)
                               for k, v in tensors.items()}}, tmp)
        os.replace(tmp, p)
        _count("put")
    except Exception as e:                                  # disk full is the whole story here
        _fail(f"write of {tag} failed ({type(e).__name__}: {e}); converting without caching")
