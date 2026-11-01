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


# ---------------------------------------------------------------- stream-ahead readahead
#
# ``bench/mmapread_probe.py`` says the target pass is fault-bound, not device-bound: reading a cold shard with
# streaming pread runs at 3.49 GiB/s, touching the same cold shard through mmap one byte per 4 KiB page runs at
# **0.57 GiB/s** -- which is exactly the ~0.5-0.6 GiB/s a rank's expert placement measures -- and WILLNEED followed
# by the same mmap pass runs at **4.12 GiB/s**. mmap page faults get no readahead of their own; queueing it does.
# So: while the consumer places tensor i, ask for the tensors that come after it. Cost is a few fadvise calls per
# tensor run; the depth (bytes) is ``R9K_PRESTAGE_AHEAD``, 0 = off until measured.

_AHEAD_DEPTH = int(os.environ.get("R9K_PRESTAGE_AHEAD", "0"))
_CACHE_WINDOW = int(os.environ.get("R9K_CACHE_WINDOW", "0"))   # MiB of consumed shard left mapped; 0 = off
_EVD: dict[str, int] = {}         # per shard: offset handed back so far (header end at the start)
_ORDER: list | None = None            # (name, path, start, end) in per-shard header order = the order the loader walks
_POS: dict[str, int] = {}
_FDS: dict[str, int] = {}
_QED: dict[str, int] = {}         # per shard: highest offset already queued, so overlapping windows queue once
_AHEAD_TOT = {"tensors": 0, "ranges": 0, "bytes": 0, "queue_s": 0.0, "indexed": 0}


def _index(model: str) -> None:
    global _ORDER, _POS
    order: list = []
    for path in _shard_files(model):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            meta = json.loads(f.read(n))
        base = 8 + n
        for name, m in meta.items():
            if name == "__metadata__":
                continue
            a, b = m["data_offsets"]
            if b > a:
                order.append((name, path, base + a, base + b))
    _ORDER = order
    _POS = {name: i for i, (name, _, _, _) in enumerate(order)}
    _AHEAD_TOT["indexed"] = len(order)
    logger.info("r9700 stream ahead: %s tensors indexed under %s, depth %d MiB", len(order), model, _AHEAD_DEPTH >> 20)


def ahead(name: str) -> None:
    """Queue readahead for the tensors after ``name`` (``R9K_PRESTAGE_AHEAD`` bytes deep) and, with
    ``R9K_CACHE_WINDOW`` set, hand the shard bytes behind ``name`` back. Best effort."""
    if _AHEAD_DEPTH <= 0 and _CACHE_WINDOW <= 0:
        return
    if _ORDER is None:
        model = _model_dir()
        if not model:
            return
        try:
            _index(model)
        except (OSError, ValueError, KeyError) as e:
            logger.info("r9700 stream ahead: disabled, indexing failed: %s", e)
            globals()["_AHEAD_DEPTH"] = 0
            return
    p = _POS.get(name)
    if p is None:
        return
    if _CACHE_WINDOW > 0:
        _, _path, _a, _end = _ORDER[p]
        _evict_behind(_path, _a)
    if _AHEAD_DEPTH <= 0:
        return
    i, got = p + 1, 0
    run_start = run_end = None
    run_path = None
    t0 = time.perf_counter()
    while i < len(_ORDER) and got < _AHEAD_DEPTH:
        _, path, a, b = _ORDER[i]
        if run_path is not None and (path != run_path or a - run_end > (1 << 20)):
            _queue(run_path, run_start, run_end)
            got += run_end - run_start
            run_start = run_end = run_path = None
        if run_path is None:
            run_path, run_start, run_end = path, a, b
        else:
            run_end = max(run_end, b)
        _AHEAD_TOT["tensors"] += 1
        i += 1
    if run_path is not None:
        _queue(run_path, run_start, run_end)
        got += run_end - run_start
    _AHEAD_TOT["queue_s"] += time.perf_counter() - t0


def _evict_behind(path: str, at: int) -> None:
    """Give the consumed part of a shard back to the kernel.

    The checkpoint (110.6 GiB) never fits in the cache this load can hold, so consumed pages are not free: keeping
    them is what makes the kernel take anon instead (the not-yet-pinned PLE tables), and that debt is repaid at
    ~1 ms/page in the draft pass. ``POSIX_FADV_DONTNEED`` on clean pages loses nothing; the risk is a slower rank
    that still needs those bytes, and that would show up directly as extra ``read_bytes`` per rank.
    """
    start = _EVD.get(path)
    if start is None:
        with open(path, "rb") as f:                        # data starts after the 8-byte length + JSON header
            start = 8 + struct.unpack("<Q", f.read(8))[0]
    cut = at - (_CACHE_WINDOW << 20)
    if cut <= start:
        _EVD[path] = start
        return
    fd = _FDS.get(path)
    if fd is None:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            return
        _FDS[path] = fd
    try:
        os.posix_fadvise(fd, start, cut - start, os.POSIX_FADV_DONTNEED)
    except OSError:
        return
    _EVD[path] = cut


def _queue(path: str, a: int, b: int) -> None:
    a = max(a, _QED.get(path, 0))                      # never queue the same bytes twice: sliding windows overlap
    if b <= a:
        return
    _QED[path] = b
    fd = _FDS.get(path)
    if fd is None:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError as e:
            logger.info("r9700 stream ahead: %s: %s", path, e)
            globals()["_AHEAD_DEPTH"] = 0
            return
        _FDS[path] = fd
    try:
        os.posix_fadvise(fd, a, b - a, os.POSIX_FADV_WILLNEED)
    except OSError as e:
        logger.info("r9700 stream ahead: fadvise failed: %s", e)
        globals()["_AHEAD_DEPTH"] = 0
        return
    _AHEAD_TOT["ranges"] += 1
    _AHEAD_TOT["bytes"] += b - a


def ahead_done(tag: str = "stream ahead") -> None:
    """Report and release. Called once per rank at the end of the weight stream."""
    if _CACHE_WINDOW > 0:
        logger.info("r9700 %s: cache window %d MiB, %s shards handed back", tag, _CACHE_WINDOW, len(_EVD))
    for path, fd in _FDS.items():
        try:
            os.close(fd)
        except OSError:
            pass
    _FDS.clear()
    t = _AHEAD_TOT
    if not t["tensors"]:
        return
    logger.info("r9700 %s: queued %.1f GiB in %s ranges covering %s tensors (%.2f s of fadvise wall)",
                tag, t["bytes"] / 2**30, t["ranges"], t["tensors"], t["queue_s"])
