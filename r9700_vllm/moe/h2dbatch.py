"""Batch the per-expert host->device copies that the stock MoE loader issues.

Stock's ``RoutedExperts._load_w13`` / ``_load_w2`` place one expert at a time. For this checkpoint (512 experts,
MXFP4-packed uint8, 192 fused tensors a rank) that is **147,912 pageable ``copy_`` calls per rank** of 106 KiB and
51 KiB: 15.0 GiB in 34.5 s, i.e. 0.33-0.52 GB/s (``R9K_LOAD_COPIES=1`` copy classes, notes/native-4xr9700.md). The
same bytes moved as a few large copies run at 9-47 GB/s (``bench/expertcopy_probe.py``: 512 slices of 106 KiB per
expert 75 ms, one copy 10.7 ms, ``tensor.to(device)`` 0.3 ms) -- which looked like per-transfer overhead. It is not;
see the measurement at the end.

This wrapper accumulates *runs of adjacent destination slices* into one pinned host buffer and issues a single
``copy_`` per run. The bytes written are exactly the same; only the number of transfers changes. Deliberately narrow:

* uint8 only (the packed MXFP4 weights -- the two hot classes), contiguous on both sides, slice <= 1 MiB,
  CPU source, CUDA destination;
* one active run; a call that is not adjacent to it, or does not match the pattern, first flushes the run and then
  does the plain copy;
* any call that touches the *same destination storage* but is not batched (a scale, a padded path, ...) flushes the
  run first, so nothing we staged can be clobbered by something we did not stage;
* the weight stream calls :func:`flush_all` at its end, which reports what is still pending and writes it.

Off unless ``R9K_MOE_BATCH_H2D=1``.

**Measured: it does not pay, and the reason is the interesting part** (``h2db36`` against the ``final31`` baseline).
With this patch on, a rank staged 98,496 slices into ... 98,496 copies -- the runs never joined, because stock loads
the w1 and w3 halves of one expert alternately (two different storages), so every call broke the active run. More
fundamentally: the *host-side staging read alone* cost **18.35 s for 10.0 GiB (0.56 GB/s)** while the stock
``copy_``s it replaces cost **19.56 s for the same 10.0 GiB (0.51 GB/s)** -- the same bytes at the same rate. The
cost is therefore on the **source side** (touching the mmap'd checkpoint pages), not per-transfer launch overhead:
the H2D piece is only ~5 s (10.0 GiB at 2.0 GB/s with four ranks sharing the device). Total load with the patch on:
**80.6 s vs 73.6 s**, so it stays off. The microbench gain (``bench/expertcopy_probe.py``, 512 slices of an already
resident tensor: 46.8 ms -> 16.8 ms) was measuring launch overhead in RAM, which is not this bottleneck.
"""
from __future__ import annotations

import os
import time

import torch
from vllm.logger import init_logger

logger = init_logger("vllm." + __name__)

_MAX_SLICE = 1 << 20          # bytes: only small per-expert slices are worth batching
_MAX_RUN = 1 << 26            # 64 MiB staging buffer (a power of two, so torch's pinned allocator does not round up)

# one active run: (storage, storage_offset_in_elements, length_in_elements, dtype, device, staged_bytes)
_RUN: list | None = None
_PIN: torch.Tensor | None = None
_FLUSHING = False           # reentrancy guard: flush()'s own copy_ must not be batched again
_TOT = {"calls": 0, "batches": 0, "bytes": 0, "stage_s": 0.0, "copy_s": 0.0}


def enabled() -> bool:
    return os.environ.get("R9K_MOE_BATCH_H2D", "0") == "1"


def _pin() -> torch.Tensor:
    global _PIN
    if _PIN is None:
        t0 = time.perf_counter()
        _PIN = torch.empty(_MAX_RUN, dtype=torch.uint8, device="cpu", pin_memory=True)
        logger.info("r9700 moe h2d batch: %d MiB pinned staging buffer in %.2f s", _MAX_RUN >> 20,
                    time.perf_counter() - t0)
    return _PIN


def _eligible(d: torch.Tensor, s: torch.Tensor) -> bool:
    return (d.dtype == torch.uint8 and s.dtype == torch.uint8
            and d.is_contiguous() and s.is_contiguous()
            and 0 < d.numel() <= _MAX_SLICE and d.numel() == s.numel()
            and d.device.type == "cuda" and s.device.type == "cpu")

def flush() -> None:
    """Issue the single copy for the staged run, if any."""
    global _RUN
    if _RUN is None:
        return
    storage, off, n, dtype, dev, used = _RUN
    dst = torch.empty(0, dtype=dtype, device=dev)
    dst.set_(storage, off, (used,), (1,))          # exactly the bytes the individual calls would have written
    src = _pin()[:used]
    global _FLUSHING
    _FLUSHING = True
    t0 = time.perf_counter()
    dst.copy_(src)
    dt = time.perf_counter() - t0
    _FLUSHING = False
    _TOT["batches"] += 1
    _TOT["bytes"] += used
    _TOT["copy_s"] += dt
    _RUN = None


def _stage(d: torch.Tensor, s: torch.Tensor) -> None:
    global _RUN
    nb = d.numel()
    buf = _pin()
    if _RUN is not None:
        storage, off, n, dtype, dev, used = _RUN
        same = (storage is d.untyped_storage() and dtype == d.dtype and dev == d.device)
        if same and used + nb <= _MAX_RUN and d.data_ptr() == storage.data_ptr() + (off + used) * d.element_size():
            t0 = time.perf_counter()
            buf[used:used + nb].copy_(s.view(-1))          # host->host, same page-cache read as stock's copy
            _TOT["stage_s"] += time.perf_counter() - t0
            _RUN = (storage, off, n, dtype, dev, used + nb)
            _TOT["calls"] += 1
            return
        flush()                                            # run broken (or full): write it, start over
    _RUN = (d.untyped_storage(), d.storage_offset(), d.numel(), d.dtype, d.device, 0)
    t0 = time.perf_counter()
    buf[0:nb].copy_(s.view(-1))
    _TOT["stage_s"] += time.perf_counter() - t0
    _RUN = (_RUN[0], _RUN[1], _RUN[2], _RUN[3], _RUN[4], nb)
    _TOT["calls"] += 1


def flush_all(what: str = "end of the weight stream") -> None:
    """Write whatever is still staged and report. Called once per rank by the loader."""
    flush()
    t = _TOT
    if not t["calls"]:
        return
    logger.info("r9700 moe h2d batch: %s -- %s expert slices staged into %s large copies (%.1f GiB): "
                "%.2f s host staging + %.2f s device writes", what, t["calls"], t["batches"],
                t["bytes"] / 2**30, t["stage_s"], t["copy_s"])


def _wrap(fn, name: str):
    """Unused: kept out of the call path -- see install() for why the copy itself is intercepted."""
    raise NotImplementedError


def install() -> bool:
    """Wrap ``Tensor.copy_``. Idempotent; installs only its own predicate, everything else passes through.

    Intercepting the copy rather than ``RoutedExperts._load_w13`` / ``_load_w2`` is deliberate: at those methods'
    entry the checkpoint tensor is still the whole fused [experts, ...] weight and the per-expert slicing happens
    inside them (measured: a wrapper there sees ``loaded_weight.numel()`` 512x the destination slice, so it matches
    nothing). At ``copy_`` the arguments are the slices themselves.
    """
    if not enabled():
        return False
    from ..compat import gate
    gate.check("moe_h2d_batch")
    orig = torch.Tensor.copy_
    if getattr(orig, "_r9k_h2d_batch", False):
        return True

    def wrapper(self, src, *a, **kw):
        if not a and not kw and not _FLUSHING and isinstance(src, torch.Tensor) and _eligible(self, src):
            try:
                return _stage(self, src)
            except Exception as exc:                      # never let an optimization break a load
                logger.warning("r9700 moe h2d batch: falling back to a plain copy (%s)", exc)
                flush()
                return orig(self, src)
        if _RUN is not None and not _FLUSHING and _RUN[0] is self.untyped_storage():
            flush()                                       # something else touches this storage: write first
        return orig(self, src, *a, **kw)

    wrapper.__name__ = "copy_"
    wrapper._r9k_h2d_batch = True
    wrapper._r9k_orig = orig
    torch.Tensor.copy_ = wrapper
    logger.info("r9700 moe h2d batch: installed on Tensor.copy_ (staging %d MiB pinned)", _MAX_RUN >> 20)
    return True
