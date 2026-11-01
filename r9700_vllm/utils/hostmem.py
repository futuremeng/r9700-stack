"""Exact-size pinned host memory + UVA views.

torch's pinned allocator (tensor.pin_memory(), torch.empty(pin_memory=True)) rounds every block up to a power of
two: a 19.4 GiB PLE table costs 32 GiB of locked RAM (the first stock bring-up was OOM-killed at 49.9 GB shmem per
worker). Here we allocate ordinary host memory and lock it with hipHostRegister(Portable|Mapped) at its exact size,
then take vLLM's UVA device view of it -- the same approach tcclaviger's expert offloader uses.
"""
from __future__ import annotations

import contextlib

import torch

_KEEP: list[torch.Tensor] = []
_REGISTER_PORTABLE_MAPPED = 3


def _register(t: torch.Tensor) -> torch.Tensor:
    import time
    from vllm.logger import init_logger
    _lg = init_logger("vllm.r9700.hostmem")
    n = t.numel() * t.element_size()
    t1 = time.perf_counter()
    if n:
        rc = torch.cuda.cudart().cudaHostRegister(t.data_ptr(), n, _REGISTER_PORTABLE_MAPPED)
        if int(rc) != 0:  # fall back to torch's (rounding) allocator rather than fail
            import warnings
            warnings.warn(f"r9700: hipHostRegister failed ({int(rc)}); using torch pinned memory (rounds to 2^k)")
            t = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=True)
    if n and not t.is_pinned():
        raise RuntimeError("r9700: host buffer is not recognised as pinned; a UVA view would silently be a copy")
    _lg.info("r9700 hostmem: hipHostRegister of %.2f GiB took %.2f s (ends %s)",
             n / 2**30, time.perf_counter() - t1, time.strftime("%H:%M:%S"))
    _KEEP.append(t)       # registered memory must outlive every view; these are process-lifetime buffers
    return t


def pinned_empty(shape, dtype, pin: bool = True) -> torch.Tensor:
    import time
    from vllm.logger import init_logger
    _lg = init_logger("vllm.r9700.hostmem")
    t0 = time.perf_counter()
    t = torch.empty(shape, dtype=dtype, device="cpu")
    n = t.numel() * t.element_size()
    _lg.info("r9700 hostmem: %s %s alloc'd (%.2f GiB) in %.2f s at %s, registering%s...",
             tuple(shape), dtype, n / 2**30, time.perf_counter() - t0, time.strftime("%H:%M:%S"),
             "" if pin else " LATER (R9K_PLE_PIN_LATE=1)")
    if not pin:
        _KEEP.append(t)
        return t
    return _register(t)


# ---------------- late pinning ----------------
# The PLE int6 table is 9.69 GiB of host memory per rank, and while it is *registered* it is unswappable and outside
# the kernel's reclaim options: with ~15 GiB of anon and ~46 GiB of page cache on a 93.8 GiB box, the kernel responds
# to the pressure by writing ~5.9 GiB of swap at ~1.4 GB/s on the device that holds the mmap'd checkpoint, and copies
# that touch a non-resident page then wait 5-60 s with the GPU idle (notes/native-4xr9700.md). Registration is not
# needed until the first forward, so late mode allocates ordinary (reclaimable) memory, fills it during the stream,
# and registers it once the stream is over: 4.7 s a rank, once, after the load.

_LATE: list[tuple[torch.Tensor, object]] = []


def late_enabled() -> bool:
    import os
    return os.environ.get("R9K_PLE_PIN_LATE", "0") == "1"


def register_late(host: torch.Tensor, then) -> None:
    """Remember an unregistered buffer and the callback that rebuilds its view once it is registered."""
    _LATE.append((host, then))


def finalize_late_pins(what: str = "after the weight stream") -> float:
    """Register every late buffer and let its owner rebuild its view. Returns GiB registered (0 if none)."""
    if not _LATE:
        return 0.0
    gib = 0.0
    for host, then in _LATE:
        gib += host.numel() * host.element_size() / 2.0**30
        _register(host)
        then(host)
    _LATE.clear()
    from vllm.logger import init_logger
    init_logger("vllm.r9700.hostmem").info("r9700 hostmem: late-pinned %.1f GiB %s", gib, what)
    return gib


def uva_empty(shape, dtype) -> torch.Tensor:
    """Device-typed (UVA) view of a fresh exact-size pinned host buffer."""
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    host = pinned_empty(shape, dtype)
    v = get_accelerator_view_from_cpu_tensor(host)
    v._r9k_host = host
    return v


@contextlib.contextmanager
def exact_pinning():
    """Within the block, CPU ``Tensor.pin_memory()`` pins at exact size (hipHostRegister) instead of torch's 2^k
    rounding. Used around model construction, where vLLM's UVA offloader pins every offloaded parameter
    (+~22% host RAM for Flash-Next's experts otherwise). Restored on exit; not thread-scoped (construction is
    single-threaded per worker)."""
    stock = torch.Tensor.pin_memory

    def exact(self, *a, **k):
        if self.device.type != "cpu":
            return stock(self, *a, **k)
        out = pinned_empty(self.shape, self.dtype)
        out.copy_(self)
        return out

    torch.Tensor.pin_memory = exact
    try:
        yield
    finally:
        torch.Tensor.pin_memory = stock
