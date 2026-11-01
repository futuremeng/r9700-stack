"""Load-time timers, enabled with R9K_LOAD_PROF=1 (the same switch models/qwen4_exp.py's stream profile uses).

`R9K_LOAD_PROF=1` on the checkpoint stream says *which group* of tensors the load spends its seconds on but not
*what work* -- the consumer side (vLLM's weight_loader plus our quant/MoE/PLE hooks) is one number. These timers
open that box: wrap a load-path function with @timed("name") and it reports total seconds and call count at
process exit. Zero cost when the switch is off."""
import atexit
import functools
import os
import resource
import time

from vllm.logger import init_logger

logger = init_logger("vllm." + __name__)
ON = os.environ.get("R9K_LOAD_PROF", "") == "1"
_ACC: dict[str, list] = {}
_LAST = [time.perf_counter()]


def record(name: str, seconds: float) -> None:
    if not ON:
        return
    a = _ACC.setdefault(name, [0.0, 0])
    a[0] += seconds
    a[1] += 1
    # a server process does not exit, so an atexit report would never print: also dump every 15 s while loading
    now = time.perf_counter()
    if now - _LAST[0] > 15.0:
        _LAST[0] = now
        report()


def timed(name: str):
    """Decorator: time calls of a load-path function. Off unless R9K_LOAD_PROF=1."""
    def deco(fn):
        if not ON:
            return fn

        @functools.wraps(fn)
        def wrap(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                record(name, time.perf_counter() - t0)
        return wrap
    return deco


def snap(tag: str) -> None:
    """Log page-cache state and this process's page faults at a load phase boundary.

    The load streams safetensors through mmap, so the cost of a tensor is paid when its bytes are first touched:
    a warm run shows up as minor faults, a run whose pages were evicted by the ranks' own anon (4 x ~13 GiB plus a
    38.8 GiB pinned PLE table) shows up as **major faults** -- i.e. a disk re-read. That is the difference between
    a 6 s and a 73 s draft requant on this box, and this is the counter that proves it. R9K_LOAD_PROF=1 only."""
    if not ON:
        return
    cached = swapfree = -1
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("Cached:"):
                    cached = int(line.split()[1]) // 2**20
                elif line.startswith("SwapFree:"):
                    swapfree = int(line.split()[1]) // 2**20
        ru = resource.getrusage(resource.RUSAGE_SELF)
        rchar = rd_bytes = -1
        try:
            io = dict()
            with open("/proc/self/io") as f:
                for line in f:
                    k, v = line.split(":")
                    io[k] = int(v)
            rchar, rd_bytes = io.get("rchar", -1), io.get("read_bytes", -1)
        except OSError:
            pass
        psi = ""
        try:
            with open("/proc/pressure/memory") as f:
                rows = dict((l.split()[0], l.split()[1:]) for l in f if l.split())
            if "some" in rows and "full" in rows:
                # avg10 is the percent of the last 10 s with tasks stalled; total is microseconds of stall.
                psi = (f" psi some={rows['some'][0].split('=')[1]:>5}%/{int(rows['some'][3].split('=')[1]) / 1e6:6.1f}s"
                       f" full={rows['full'][0].split('=')[1]:>5}%/{int(rows['full'][3].split('=')[1]) / 1e6:6.1f}s")
        except OSError:
            pass
        logger.info("r9700 load snap %-12s Cached=%4d GiB  SwapFree=%3d GiB  majflt=%9d  minflt=%9d  "
                    "rchar=%7.1f GiB  disk reads=%6.1f GiB%s",
                    tag, cached, swapfree, ru.ru_majflt, ru.ru_minflt, rchar / 2**30, rd_bytes / 2**30, psi)
    except OSError as e:
        logger.info("r9700 load snap %-12s unavailable: %s", tag, e)


def report() -> None:
    if not ON or not _ACC:
        return
    for name, (s, n) in sorted(_ACC.items(), key=lambda kv: -kv[1][0])[:12]:
        logger.info("r9700 load timer: %8.1f s on %-38s (%s calls)", s, name, n)


if ON:
    atexit.register(report)
