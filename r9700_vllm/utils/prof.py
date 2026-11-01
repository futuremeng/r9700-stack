"""Load-time timers, enabled with R9K_LOAD_PROF=1 (the same switch models/qwen4_exp.py's stream profile uses).

`R9K_LOAD_PROF=1` on the checkpoint stream says *which group* of tensors the load spends its seconds on but not
*what work* -- the consumer side (vLLM's weight_loader plus our quant/MoE/PLE hooks) is one number. These timers
open that box: wrap a load-path function with @timed("name") and it reports total seconds and call count at
process exit. Zero cost when the switch is off."""
import atexit
import functools
import os
import resource
import signal
import sys
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

    The load streams safetensors through mmap, so the cost of a tensor is paid when its bytes are first touched.
    Which counter shows that depends on the path: safetensors reads with read() only ~0.5 GiB of this checkpoint
    (rchar), so page-cache misses land in **read_bytes**, not majflt -- measured, and the reason an earlier version
    of this docstring blamed majflt (127-153k a rank is ~0.5 GiB of anon swap-in, nothing to do with the file).
    read_bytes says 22-29 GiB a rank is re-read from the block device during the target pass even with Cached at
    38-46 GiB. R9K_LOAD_PROF=1 only."""
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
        rss = swap = pin = -1.0
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss = int(line.split()[1]) / 2**20
                    elif line.startswith("VmSwap:"):
                        swap = int(line.split()[1]) / 2**20
                    elif line.startswith("VmPin:"):
                        pin = int(line.split()[1]) / 2**20
        except OSError:
            pass
        logger.info("r9700 load snap %-12s Cached=%4d GiB  SwapFree=%3d GiB  majflt=%9d  minflt=%9d  "
                    "rchar=%7.1f GiB  disk reads=%6.1f GiB  self rss/swap/pin=%5.1f/%5.1f/%5.1f%s",
                    tag, cached, swapfree, ru.ru_majflt, ru.ru_minflt, rchar / 2**30, rd_bytes / 2**30, rss, swap,
                    pin, psi)
    except OSError as e:
        logger.info("r9700 load snap %-12s unavailable: %s", tag, e)


def report() -> None:
    if not ON or not _ACC:
        return
    for name, (s, n) in sorted(_ACC.items(), key=lambda kv: -kv[1][0])[:12]:
        logger.info("r9700 load timer: %8.1f s on %-38s (%s calls)", s, name, n)


if ON:
    atexit.register(report)


# --------------------------------------------------------------------------------------
# Sampling stack profiler (R9K_LOAD_STACKS=1). Pure stdlib, no stock code touched: SIGPROF fires on user+sys CPU
# of this process, and the handler reads every Python thread's stack. This is how "which group of tensors" (the
# stream profile above) turns into "which function": the experts term is 92-121 s a rank for 59.8 GiB, which is
# 0.5-0.65 GB/s, and disk bandwidth (5.8 GB/s raw), H2D (19.65 GiB a rank) and reclaim stall (PSI +2.7 s) are all
# too small to explain it -- so the seconds are inside some copy/repack on the consumer side, and the histogram
# says whose. Caveats worth knowing: a thread parked inside C without touching the interpreter is charged to
# whatever Python frame it returns to, and threads with no Python frame (ROCm async copies) are simply absent.
STACKS = os.environ.get("R9K_LOAD_STACKS", "") == "1"
_INCL: dict[str, int] = {}
_LEAF: dict[str, int] = {}
_TICKS = [0]


def _where(fr) -> str:
    f = fr.f_code.co_filename.rsplit("/", 1)[-1]
    return f"{f}:{fr.f_lineno} {fr.f_code.co_name}"


def _sample(signum, frame) -> None:
    # Use the signal handler's `frame` -- the frame this thread was actually interrupted in. Do NOT use
    # sys._current_frames() for the current thread: while the handler runs, that entry is the HANDLER's own
    # frame, and the first version of this sampler accordingly reported `prof.py _sample` as the leaf of 100% of
    # samples. It also charged every idle `threading.wait` thread as a sample, diluting the interesting frames by
    # the thread count; the load runs in this thread (its stack carries _load_target), so this thread alone is the
    # population. Frames inside C (torch copy_, hipMemcpy) are charged to the Python line that called them, which
    # is what we want; threads that never enter Python are invisible.
    try:
        fr, depth, path = frame, 0, []
        while fr is not None and depth < 48:
            w = _where(fr)
            if w not in path:
                _INCL[w] = _INCL.get(w, 0) + 1
                path.append(w)
            if depth == 0:
                _LEAF[w] = _LEAF.get(w, 0) + 1
            fr, depth = fr.f_back, depth + 1
        _TICKS[0] += 1
    except Exception:                               # a profiler must never take down a load
        pass


def reset_stacks() -> None:
    """Drop the accumulated samples so the next window is measured on its own."""
    _TICKS[0] = 0
    _LEAF.clear()
    _INCL.clear()


def start_stacks() -> None:
    if not STACKS:
        return
    signal.signal(signal.SIGPROF, _sample)
    signal.setitimer(signal.ITIMER_PROF, 0.002, 0.002)      # 500 Hz of user+sys CPU
    logger.info("r9700 stack sampler on (SIGPROF 500 Hz, this thread only)")


def stop_stacks() -> None:
    if STACKS:
        signal.setitimer(signal.ITIMER_PROF, 0.0, 0.0)


def dump_stacks(tag: str, top: int = 14) -> None:
    if not STACKS:
        return
    n = max(_TICKS[0], 1)
    logger.info("r9700 stacks (%s): %d samples of this thread, %d distinct frames", tag, _TICKS[0], len(_INCL))
    for title, acc in (("leaf", _LEAF), ("inclusive", _INCL)):
        for w, c in sorted(acc.items(), key=lambda kv: -kv[1])[:top]:
            if c * 100.0 / n < 0.5:
                break
            logger.info("r9700 stacks %-10s %5.1f%%  %6d  %s", title, c * 100.0 / n, c, w)


# --------------------------------------------------------------------------------------
# torch.Tensor.copy_ accounting (R9K_LOAD_COPIES=1). The stack sampler says WHERE (94% of the target pass sits in
# two lines of vllm/model_executor/layers/fused_moe/routed_experts.py, both `expert_data.copy_(loaded_weight)`);
# this says HOW MUCH of it those copies are, per (dst device, src device, dtype, contiguity) class, so a fix can
# be aimed at the right copy. Caveat: for CPU->CUDA the wall time is the submission and any strided packing, not
# the DMA completion (no sync here by design -- syncing would change the thing being measured).
COPIES = os.environ.get("R9K_LOAD_COPIES", "") == "1"
_CPY: dict[tuple, list] = {}
_SLOW: list[tuple] = []          # (seconds, nbytes, key, t_start) -- the outliers are the whole story here
_BUCK = [0, 0, 0, 0, 0]          # <1 ms, 1-10 ms, 10-100 ms, 0.1-1 s, >1 s
_BUCKT = [0.0, 0.0, 0.0, 0.0, 0.0]
_T0 = [0.0]                     # perf_counter at probe install, to place slow calls on the load timeline
_W0 = [0.0]                     # wall clock at the same instant, to line the slow calls up with loadsample.sh
_CURRENT = [""]                 # tensor name the load stream is currently placing, set by models/qwen4_exp.py


def set_current(name: str) -> None:
    if COPIES:
        _CURRENT[0] = name


def _bucket(dt: float) -> int:
    return 0 if dt < 1e-3 else 1 if dt < 1e-2 else 2 if dt < 0.1 else 3 if dt < 1.0 else 4


def install_copy_probe() -> None:
    if not COPIES:
        return
    import torch
    if getattr(torch.Tensor, "_r9k_copy_probe", False):
        return
    orig = torch.Tensor.copy_

    def copy_(self, src, *a, **k):
        t0 = time.perf_counter()
        f0 = resource.getrusage(resource.RUSAGE_SELF)      # one getrusage a call, ~0.6 us; 148k calls cost 0.1 s
        r = orig(self, src, *a, **k)
        dt = time.perf_counter() - t0
        fl = (0, 0)
        if dt > 0.05:                                      # only the interesting ones pay for the second sample
            f1 = resource.getrusage(resource.RUSAGE_SELF)
            fl = (f1.ru_minflt - f0.ru_minflt, f1.ru_majflt - f0.ru_majflt)
        key = (str(self.device), str(getattr(getattr(src, "device", None), "type", "?")), str(self.dtype),
               "dstcontig" if self.is_contiguous() else "dststrided",
               "srccontig" if getattr(src, "is_contiguous", _true)() else "srcstrided")
        b = _CPY.setdefault(key, [0, 0, 0.0, 0.0])
        b[0] += 1
        b[1] += self.numel() * self.element_size()
        b[2] += dt
        if dt > b[3]:
            b[3] = dt
        i = _bucket(dt)
        _BUCK[i] += 1
        _BUCKT[i] += dt
        if dt > 0.2:
            _SLOW.append((dt, self.numel() * self.element_size(), key, t0, _CURRENT[0], fl))
            _SLOW.sort(reverse=True)
            del _SLOW[8:]
        return r

    torch.Tensor.copy_ = copy_
    torch.Tensor._r9k_copy_probe = True
    _T0[0] = time.perf_counter()
    _W0[0] = time.time()
    start_watchdog()
    logger.info("r9700 copy probe on (Tensor.copy_)")


def _true() -> bool:
    return True


def report_copies(tag: str) -> None:
    if not COPIES:
        return
    stop_watchdog()
    tot = sum(v[2] for v in _CPY.values())
    n = sum(v[0] for v in _CPY.values())
    logger.info("r9700 copies (%s): %d copy_ calls, %.1f GiB written, %.1f s total copy wall; probe at %s",
                tag, n, sum(v[1] for v in _CPY.values()) / 2**30, tot,
                time.strftime("%H:%M:%S", time.localtime(_W0[0])))
    for k, v in sorted(_CPY.items(), key=lambda kv: -kv[1][2])[:8]:
        logger.info("r9700 copies %6d calls %7.2f GiB %8.2f s (max %6.3f s, %5.2f GB/s)  %s",
                    v[0], v[1] / 2**30, v[2], v[3], (v[1] / 2**30) / max(v[2], 1e-9), " ".join(k))
    logger.info("r9700 copies by size of call: n<1ms=%d (%.1f s) 1-10ms=%d (%.1f s) 10-100ms=%d (%.1f s) "
                "0.1-1s=%d (%.1f s) >1s=%d (%.1f s)",
                _BUCK[0], _BUCKT[0], _BUCK[1], _BUCKT[1], _BUCK[2], _BUCKT[2], _BUCK[3], _BUCKT[3],
                _BUCK[4], _BUCKT[4])
    for dt, nb, key, t0, name, fl in _SLOW[:6]:
        logger.info("r9700 copies slow call %7.2f s  %8.1f MiB  +%6.1f s into the load  minflt+%-6d majflt+%-5d "
                    "on [%s] %s", dt, nb / 2**20, t0 - _T0[0], fl[0], fl[1], name, " ".join(key))


# --------------------------------------------------------------------------------------
# Stuck-thread watchdog. The copy probe says a 0.2 MiB copy_ blocked 55 s with minflt+4 / majflt+2 (inst15) -- so
# it is neither bandwidth, nor disk, nor reclaim (GPU busy 3 %, cached/swap flat). The next question is what it is
# waiting ON, and only the kernel's own wait channel answers that. A helper thread watches the main thread: when
# its Python frame has not moved for `after` seconds, it logs the frame, the thread state and /proc/self/task/TID/
# wchan. futex_wait says a userspace runtime lock; anything amdgpu/kfd-ish says the driver.
_WD: list = []


def start_watchdog(interval: float = 0.5, after: float = 2.0, lifespan: float = 900.0) -> None:
    import threading

    # Two different ids are needed and they are not interchangeable: sys._current_frames() is keyed by CPython
    # thread id, which for the main thread is the pthread_self handle (observed 127864336392320), while
    # /proc/self/task/<tid>/ wants the kernel TID, which for the main thread equals the PID. Using one for both
    # fails in two different ways: PID-as-frame-key never matches (silent, no output), pthread-handle-as-/proc-tid
    # gives ENOENT on every read (observed as state=? wchan=?).
    frame_key = threading.main_thread().ident
    main_tid = os.getpid()

    def run() -> None:
        last, same, t_end = None, 0.0, time.perf_counter() + lifespan
        while _WD and time.perf_counter() < t_end:
            time.sleep(interval)
            fr = sys._current_frames().get(frame_key)
            key = (fr.f_code.co_filename.rsplit("/", 1)[-1], fr.f_lineno, fr.f_code.co_name) if fr else None
            if key is not None and key == last:
                same += interval
                if same >= after:
                    state = wchan = syscall = "?"
                    try:
                        base = f"/proc/self/task/{main_tid}"
                        wchan = open(base + "/wchan").read().strip() or "0"
                        state = open(base + "/stat").read().split()[2]
                        syscall = (open(base + "/syscall").read().split() or ["?"])[0]
                    except OSError:
                        pass
                    logger.info("r9700 watchdog: main thread stuck in %s:%d %s for %.1f s  state=%s wchan=%s "
                                "syscall=%s", key[0], key[1], key[2], same, state, wchan, syscall)
                    same = 0.0                                  # re-arm: catch the next episode too
            else:
                last, same = key, 0.0

    t = threading.Thread(target=run, name="r9700-watchdog", daemon=True)
    _WD.append(t)
    t.start()
    logger.info("r9700 watchdog on (main thread, >%.0f s in one frame)", after)


def stop_watchdog() -> None:
    _WD.clear()


# --------------------------------------------------------------------------- post-load pass, per module class

_PWA = {"on": False, "acc": {}, "orig": None}


def start_pwal() -> None:
    """Attribute the loader's post-load pass (``process_weights_after_loading``) to module classes.

    The stock pass is one synchronous ``for name, module in model.named_modules()`` loop, so the time between two
    consecutive yields belongs to the module yielded before it. Wrapping ``named_modules`` measures that without
    duplicating -- and therefore version-pinning -- the body of the loop, and it counts blocked time too, which is
    what the window is made of (6 s in final38 against 31 s in final39, notes/native-4xr9700.md).
    """
    if _PWA["on"]:
        return
    import torch
    _PWA["acc"].clear()
    _PWA["orig"] = torch.nn.Module.named_modules

    def timed(self, *a, **kw):
        orig = _PWA["orig"]
        if not _PWA["on"]:
            yield from orig(self, *a, **kw)
            return
        prev, tprev = None, time.perf_counter()
        for m in orig(self, *a, **kw):
            now = time.perf_counter()
            if prev is not None:
                # some callers iterate (name, module) pairs rather than modules; normalize both forms
                nm, mod = (prev[0], prev[1]) if isinstance(prev, tuple) and len(prev) == 2 else (None, prev)
                cls = type(mod).__name__
                e = _PWA["acc"].setdefault(cls, [0, 0.0])
                e[0] += 1
                e[1] += now - tprev
                if nm and now - tprev > 0.2:
                    d = _PWA["acc"].setdefault("@" + nm, [0, 0.0])
                    d[1] += now - tprev
            prev, tprev = m, now
            yield m

    _PWA["on"] = True
    torch.nn.Module.named_modules = timed
    if not _WD:
        start_watchdog()                 # the slow version of this window is blocked, not computed: name the line
        _PWA["wd"] = True
    logger.info("r9700 post-load timer on (gaps between named_modules yields)")


def stop_pwal(tag: str = "post-load") -> None:
    """Restore ``named_modules`` and report the accumulated per-class totals."""
    if not _PWA["on"]:
        return
    import torch
    _PWA["on"] = False
    if _PWA.pop("wd", False):
        stop_watchdog()
    if _PWA["orig"] is not None:
        torch.nn.Module.named_modules = _PWA["orig"]
        _PWA["orig"] = None
    acc = _PWA["acc"]
    if not acc:
        logger.info("r9700 %s: the timer saw no named_modules loop", tag)
        return
    logger.info("r9700 %s: %.1f s over %s module visits", tag, sum(v[1] for k, v in acc.items()
                if not k.startswith("@")), sum(v[0] for k, v in acc.items() if not k.startswith("@")))
    for k, (c, s) in sorted(((k, v) for k, v in acc.items() if not k.startswith("@")), key=lambda kv: -kv[1][1])[:8]:
        if s < 0.05:
            break
        logger.info("r9700 %s   %7.2f s  %5d  %s", tag, s, c, k)
    for k, (c, s) in sorted(((k, v) for k, v in acc.items() if k.startswith("@")), key=lambda kv: -kv[1][1])[:8]:
        logger.info("r9700 %s   %7.2f s  %s", tag, s, k[1:])
    # the post-load pass is where the fold guard runs for dense/shared-expert tensors, so print the @timed section
    # totals here too -- by now they cover both the stream and this window, which is the only complete picture a
    # server process (which never exits, so atexit never fires) gives
    report()
