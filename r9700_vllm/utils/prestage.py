"""Pre-read the checkpoint byte ranges that the MTP draft pass is going to consume.
On unless ``R9K_PRESTAGE_MTP=0`` (that is also the control switch for the A/B).

Why this exists, measured rather than assumed (five gated relaunches, ``notes/native-4xr9700.md``): the draft
model's loader iterates the whole checkpoint index -- 2182 tensors / 110.6 GiB a rank -- to consume a few GiB of
bf16 MTP MLP weights, which it then requantizes to MXFP4. By the time it runs, four ranks have taken ~50 GiB of
anon plus 9.7 GiB a rank of pinned int6 PLE table, and that has evicted the very pages the target pass had just
faulted in. So those few GiB come off the block device again: the draft pass costs **5.5-8 s when its bytes are
still resident and 60-76 s when they are not**, and inside one instance one rank paid 5.2 s while another paid
47.7 s, because they race each other for the cache.

``posix_fadvise(POSIX_FADV_WILLNEED)`` over exactly those ranges, once per rank at the end of the target pass,
turns four serialized post-eviction re-reads into one shared readahead. The ranges come from the safetensors
header (tensor ``data_offsets``), so this reads nothing the draft pass will not touch.
"""
from __future__ import annotations

import json
import os
import re
import struct
import time

from vllm.logger import init_logger

from . import prof

logger = init_logger("vllm." + __name__)

_MTP_NAME = re.compile(r"(^|\.)mtp\.")


def _keep(name: str) -> bool:
    """A tensor the draft model will touch: any MTP weight (the two drops in models/qwen4_exp.py are fork-private
    draft-head and unused bf16 KV scales, both tiny, so leaving them in the readahead costs nothing)."""
    return bool(_MTP_NAME.search(name))


def _model_dir() -> str | None:
    try:
        from vllm.config import get_current_vllm_config
        return get_current_vllm_config().model_config.model
    except Exception as e:                                   # noqa: BLE001 - plugin must never break a load
        logger.info("r9700 prestage: no model dir (%s)", e)
        return None


def _shard_files(model: str) -> list[str]:
    idx = os.path.join(model, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            files = set(json.load(f)["weight_map"].values())
        return [os.path.join(model, name) for name in sorted(files)]
    return sorted(os.path.join(model, n) for n in os.listdir(model) if n.endswith(".safetensors"))


def _ranges(path: str) -> list[tuple[int, int]]:
    """Merged (offset, length) spans of the MTP tensors in one shard. safetensors layout: 8-byte little-endian
    header length N, N bytes of JSON, tensor data from 8+N."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))
    base = 8 + n
    spans = []
    for name, m in meta.items():
        if name == "__metadata__" or not _keep(name):
            continue
        a, b = m["data_offsets"]
        if b > a:
            spans.append((base + a, base + b))
    out: list[list[int]] = []
    for s, e in sorted(spans):
        if out and s - out[-1][1] <= 1 << 20:               # merge near-neighbours: one fadvise, not 300
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e - s) for s, e in out]


def prestage(tag: str = "prestage mtp") -> None:
    """Issue readahead for the draft pass's byte ranges. Best effort: any failure is logged, never raised."""
    if os.environ.get("R9K_PRESTAGE_MTP", "1") != "1":
        logger.info("r9700 %s: disabled (R9K_PRESTAGE_MTP=0)", tag)
        return
    t0 = time.perf_counter()
    model = _model_dir()
    if not model:
        return
    total = nranges = 0
    try:
        for path in _shard_files(model):
            rs = _ranges(path)
            if not rs:
                continue
            fd = os.open(path, os.O_RDONLY)
            try:
                for off, ln in rs:
                    os.posix_fadvise(fd, off, ln, os.POSIX_FADV_WILLNEED)
                    total += ln
                    nranges += 1
            finally:
                os.close(fd)
    except OSError as e:
        logger.info("r9700 %s: failed: %s", tag, e)
        return
    logger.info("r9700 %s: %.1f GiB in %s ranges under %s, %.2f s to queue",
                tag, total / 2**30, nranges, model, time.perf_counter() - t0)
    prof.snap(f"{tag} end")
