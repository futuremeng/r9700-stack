"""What the checkpoint read path can deliver, and why four ranks do not get there.

The experts term reads the whole 110.6 GiB checkpoint once (per-instance `read_bytes` totals sum to ~1.07x of the
checkpoint, so there is no read amplification to reclaim) and spends ~46 s doing it: **2.4 GB/s aggregate**. The raw
stream floor measured earlier is 5.8 GB/s. This probe separates the two candidate reasons -- the device cannot do
better with four streams, or four processes faulting into the *same* mmap'd file serialize against each other -- by
measuring cold-cache reads of real shards with 1 vs 4 processes, same file vs distinct files, mmap-fault vs pread.

Every config starts from the same cold state with `posix_fadvise(POSIX_FADV_DONTNEED)` on the files it is about to
read (clean pages, so this is safe without root and does not need a cache drop).

    F=model-00017.safetensors NPROC=1 MODE=mmap ~/venv-r9700/bin/python3 bench/mmapread_probe.py
    F=model-00017.safetensors NPROC=4 MODE=mmap ~/venv-r9700/bin/python3 bench/mmapread_probe.py
    FILES="a.safetensors b.safetensors c.safetensors d.safetensors" NPROC=4 MODE=mmap ...
    MODE=pread CHUNK=16777216 ...
"""
import mmap
import multiprocessing as mp
import os
import sys
import time

D = os.environ.get("D", os.path.expanduser("~/models/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ"))
MODE = os.environ.get("MODE", "mmap")
NPROC = int(os.environ.get("NPROC", "1"))
STRIDE = int(os.environ.get("STRIDE", "4096"))
CHUNK = int(os.environ.get("CHUNK", str(1 << 20)))
FILES = (os.environ.get("FILES") or "").split() or [os.environ.get("F", "model-00017.safetensors")]
FILES = [f if f.startswith("/") else os.path.join(D, f) for f in FILES]


def evict(path: str) -> int:
    fd = os.open(path, os.O_RDONLY)
    size = os.fstat(fd).st_size
    os.posix_fadvise(fd, 0, size, os.POSIX_FADV_DONTNEED)
    os.close(fd)
    return size


def cached_gib() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("Cached:"):
            return int(line.split()[1]) / 2**20
    return -1.0


def worker(path: str, start, q) -> None:
    fd = os.open(path, os.O_RDONLY)
    size = os.fstat(fd).st_size
    start.wait()
    marks = []
    if MODE in ("willneed", "willneed_then_mmap"):
        t0 = time.perf_counter()
        os.posix_fadvise(fd, 0, size, os.POSIX_FADV_WILLNEED)
        marks.append(("willneed", size, time.perf_counter() - t0))
        if MODE == "willneed":
            q.put((path, marks))
            os.close(fd)
            return
    t0 = time.perf_counter()
    if MODE == "mmap":
        mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
        acc = 0
        for off in range(0, size, STRIDE):
            acc += mm[off]                     # one byte per page: the loader touches every page of what it places
        mm.close()
    else:
        acc = 0
        off = 0
        while off < size:
            b = os.pread(fd, CHUNK, off)
            if not b:
                break
            acc += b[0]
            off += len(b)
    marks.append((MODE, size, time.perf_counter() - t0))
    q.put((path, marks))
    os.close(fd)


def main() -> None:
    mp.set_start_method("fork")
    for f in FILES:
        if not os.path.exists(f):
            print(f"missing {f}")
            sys.exit(1)
    sizes = [evict(f) for f in FILES]
    time.sleep(0.5)
    print(f"mode={MODE} nproc={NPROC} stride={STRIDE} chunk={CHUNK} cache before={cached_gib():.2f} GiB "
          f"(evicted {sum(sizes) / 2**30:.2f} GiB)")
    start = mp.Manager().Event()
    q = mp.Queue()
    t0 = time.perf_counter()
    ps = [mp.Process(target=worker, args=(FILES[i % len(FILES)], start, q)) for i in range(NPROC)]
    for p in ps:
        p.start()
    start.set()
    got = [q.get() for _ in ps]
    for p in ps:
        p.join()
    wall = time.perf_counter() - t0
    for path, marks in got:
        for name, n, dt in marks:
            print(f"  [{os.path.basename(path)}] {name}: {n / 2**30:.2f} GiB in {dt:.2f} s = {n / 2**30 / dt:.2f} GiB/s")
    consumed = sum(n for _, m in got for _, n, _ in m if m[0][0] != "willneed")
    print(f"  streams={NPROC} distinct_files={len(set(FILES))}: wall {wall:.2f} s, "
          f"consumed aggregate {consumed / 2**30 / wall:.2f} GiB/s; cache after={cached_gib():.2f} GiB")
    for f in FILES:
        evict(f)


if __name__ == "__main__":
    main()
