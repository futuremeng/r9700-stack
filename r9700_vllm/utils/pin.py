"""Fault the worker's own anonymous memory back in before the post-load pass, instead of letting the pass fault it.

Why: on this box the target post-load pass takes ~2.2k major faults per worker when it runs in 4.5 s and ~40k when it
runs 11-75 s -- it reads back ~380 MiB of its own swapped-out memory per worker (1.5 GiB across four ranks) while the
device sits idle (``notes/native-4xr9700.md`` C16-C17). Whether that paging causes the slow window or only marks it
is answerable: pay for the pages in one measured step before the window and see whether the window falls with them.

Two modes, deliberately separate because they cost different things:

* ``populate`` -- ``madvise(MADV_POPULATE_WRITE)`` over the largest private anonymous ranges. Faults in (and, for
  swapped pages, reads back) whatever is not resident. Adds no unswappable memory, so it is safe on a box already
  carrying ~50 GiB of pinned host tables (the PLE int6 n-gram table and the routed-expert UVA tables are pinned per
  rank -- see ``ple/int6.py`` and ``moe/cache.py``). Pages may be evicted again during the pass, so a populate run is
  the conservative version of the test.
* ``lock`` -- ``mlock`` the same ranges: faults in *and* keeps out of swap, which is the decisive version, but it
  counts against ``RLIMIT_MEMLOCK`` (11.7 GiB per process here, which the pinned PLE table already leans on) and it
  makes that memory unreclaimable for every other rank. Budgeted for exactly that reason; a failure to lock is
  reported, not fatal.

Both are measured: the time they take is real and lands on the load timeline, so a saved window that costs more than
it saves is visible as such.
"""
from __future__ import annotations

import ctypes
import errno
import logging
import os
import re
import resource
import time

try:  # the repo's convention: our log lines must land in the server log, which needs vllm's logger
    from vllm.logger import init_logger
    logger = init_logger("vllm." + __name__)
except Exception:  # noqa: BLE001 - unit tests import this module without vllm configured
    logger = logging.getLogger(__name__)

_MAPS = re.compile(
    r"^([0-9a-f]+)-([0-9a-f]+)\s+([rwxps-]{4})\s+[0-9a-f]+\s+[0-9a-f]+:[0-9a-f]+\s+(\d+)\s*(.*)$"
)
_MADV_POPULATE_WRITE = 23      # Linux >= 5.14; this guest runs a 7.x kernel
_MIN_REGION = 8 << 20          # sub-8 MiB mappings are not where a 380 MiB fault storm lives


def _anon_private_ranges() -> list[tuple[int, int]]:
    """Private, writable, nameless mappings -- the heap and the allocator's segments, biggest first."""
    out: list[tuple[int, int]] = []
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                m = _MAPS.match(line.rstrip("\n"))
                if not m:
                    continue
                start, end, perms, inode, name = m.groups()
                if inode != "0" or name.strip():          # anything file-backed
                    continue
                if "w" not in perms or "p" not in perms:  # writable, private (not shared)
                    continue
                s, e = int(start, 16), int(end, 16)
                if e - s >= _MIN_REGION:
                    out.append((s, e))
    except OSError:
        return []
    return sorted(out, key=lambda r: r[0] - r[1])


def _status_kib(key: str) -> int:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(key + ":"):
                    return int(line.split()[1])
    except OSError:
        pass
    return -1


def _majflt() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_majflt


def prestage_anon(tag: str = "prestage anon") -> dict:
    """Run the mode named by ``R9K_PWAL_PRESTAGE`` ("" = off, ``populate`` or ``lock``) over ``R9K_PWAL_GIB`` of memory.

    Returns stats. Nothing here raises: a mitigation that can sink a launch is worse than no mitigation.
    """
    mode = os.environ.get("R9K_PWAL_PRESTAGE", "").strip().lower()
    if mode not in ("populate", "lock"):
        return {"mode": "off"}
    try:
        gib = float(os.environ.get("R9K_PWAL_GIB", "8"))
    except ValueError:
        gib = 8.0
    stats = {"mode": mode, "ranges": 0, "bytes": 0, "seconds": 0.0, "errors": 0, "first_error": None,
             "lck_before_kib": _status_kib("VmLck"), "lck_after_kib": -1,
             "swap_before_kib": _status_kib("VmSwap"), "swap_after_kib": -1,
             "majflt_before": _majflt(), "majflt_after": -1, "soft_limit_kib": 0}
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    limit_bytes = max(0, int(gib * (1 << 30)))
    if mode == "lock":
        soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        stats["soft_limit_kib"] = soft if soft != resource.RLIM_INFINITY else -1
        if 0 <= soft < limit_bytes:
            try:  # worth trying; this host's soft limit is 11.7 GiB and the pinned PLE table already spends it
                resource.setrlimit(resource.RLIMIT_MEMLOCK, (limit_bytes, hard))
                stats["soft_limit_kib"] = limit_bytes // 1024
            except (OSError, ValueError):
                pass  # cannot raise it: mlock will report ENOMEM and we carry on
        mlock = libc.mlock
        mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        mlock.restype = ctypes.c_int
        call = lambda addr, n: mlock(addr, n)          # noqa: E731
    else:
        madvise = libc.madvise
        madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        madvise.restype = ctypes.c_int
        call = lambda addr, n: madvise(addr, n, _MADV_POPULATE_WRITE)  # noqa: E731
    want_total = 0
    t0 = time.perf_counter()
    for start, end in _anon_private_ranges():
        if want_total >= limit_bytes:
            break
        n = min(end - start, limit_bytes - want_total)
        rc = call(start, n)
        if rc != 0:
            e = ctypes.get_errno()
            stats["errors"] += 1
            if stats["first_error"] is None:
                stats["first_error"] = errno.errorcode.get(e, str(e))
            if e in (errno.ENOMEM, errno.EPERM):
                break  # out of locked-memory budget; more ranges would fail the same way
            continue
        stats["ranges"] += 1
        stats["bytes"] += n
        want_total += n
    stats["seconds"] = time.perf_counter() - t0
    stats["lck_after_kib"] = _status_kib("VmLck")
    stats["swap_after_kib"] = _status_kib("VmSwap")
    stats["majflt_after"] = _majflt()
    logger.info(
        "r9700 %s: %s %.2f GiB over %d ranges in %.2f s (VmSwap %.2f -> %.2f GiB, VmLck %.2f -> %.2f GiB, "
        "majflt +%d, %d errors%s)",
        tag, mode, stats["bytes"] / 2**30, stats["ranges"], stats["seconds"],
        max(0, stats["swap_before_kib"]) / 2**20, max(0, stats["swap_after_kib"]) / 2**20,
        max(0, stats["lck_before_kib"]) / 2**20, max(0, stats["lck_after_kib"]) / 2**20,
        stats["majflt_after"] - stats["majflt_before"], stats["errors"],
        f", first {stats['first_error']}" if stats["first_error"] else "",
    )
    try:
        from r9700_vllm.utils import prof
        if stats["bytes"]:
            prof.record(f"prestage anon ({mode}): {stats['ranges']} ranges, {stats['bytes'] / 2**30:.2f} GiB",
                        stats["seconds"])
    except Exception:  # noqa: BLE001
        pass
    return stats
