#!/usr/bin/env python3
"""Send ONE tiny chat request and watch the PCIe link states around it.

Why this exists: since the 02:50 boot on 2026-11-03 the switch uplink `c4:00.0 <-> c0:03.1` has sat at
16.0 GT/s x16 while every other link on the box, including card4's structurally identical upstream port,
holds 32.0. The open question is whether that hop renegotiates when the cards actually work. This script
answers exactly that, at 10 Hz, and it is honest about the limit of the test: inference streams weights
from HBM, so the PCIe load from one short request is small. What it can show is a driver-initiated
renegotiation in response to activity; what it cannot show is whether the hop is marginal under a heavy
transfer (that needs a long-prompt prefill, which is a bigger ask on a serving box).

Reads only sysfs (ordinary user) plus one POST of its own. Nothing else on the box is touched.
"""
import json
import sys
import threading
import time
import urllib.request

BASE = "/sys/bus/pci/devices"
# The four chains as the Monitor walks them: endpoint, switch downstream, switch upstream, root port.
CHAIN = {
    "card3": ["0000:03:00.0", "0000:02:00.0", "0000:01:00.0", "0000:00:01.1"],
    "card4": ["0000:06:00.0", "0000:05:00.0", "0000:04:00.0", "0000:00:03.1"],
    "card1": ["0000:c3:00.0", "0000:c2:00.0", "0000:c1:00.0", "0000:c0:01.1"],
    "card2": ["0000:c6:00.0", "0000:c5:00.0", "0000:c4:00.0", "0000:c0:03.1"],
}
NODES = sorted({b for ids in CHAIN.values() for b in ids})
CARDS = ["card1", "card2", "card3", "card4"]
URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "Qwen3.8"
HZ = 10.0


def rd(path):
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def speed(bdf):
    raw = rd(f"{BASE}/{bdf}/current_link_speed")
    if not raw:
        return None
    try:
        return float(raw.split()[0])
    except ValueError:
        return None


def sample():
    out = {"t": time.time()}
    for b in NODES:
        out[b] = [speed(b), rd(f"{BASE}/{b}/current_link_width")]
        try:
            out[b + ":cor"] = sum(int(l.split(":")[-1]) for l in
                                  open(f"{BASE}/{b}/aer_dev_correctable") if ":" in l)
        except OSError:
            out[b + ":cor"] = None
    out["busy"] = {c: rd(f"/sys/class/drm/{c}/device/gpu_busy_percent") for c in CARDS}
    return out


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    samples = []
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            samples.append(sample())
            time.sleep(1.0 / HZ)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    t0 = time.time()
    time.sleep(1.0)                                    # a quiet second first
    mark_start = time.time()
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
        "max_tokens": 1,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    err, txt, usage, wall = None, None, None, None
    try:
        w0 = time.time()
        with urllib.request.urlopen(req, timeout=180) as r:
            got = json.load(r)
        wall = time.time() - w0
        txt = (got.get("choices") or [{}])[0].get("message", {}).get("content")
        usage = got.get("usage")
    except Exception as exc:                                   # noqa: BLE001 - report whatever it is
        err = repr(exc)
    mark_end = time.time()
    # Keep sampling for the rest of the requested window: whether a link renegotiates a few seconds AFTER the
    # activity is exactly the part a short request would otherwise miss.
    time.sleep(max(1.0, seconds - (time.time() - t0)))
    stop.set()
    th.join(timeout=5)

    def rel(ts):
        return ts - t0

    print(f"samples {len(samples)} over {time.time() - t0:.1f}s at {HZ:.0f} Hz")
    print(f"request window: {rel(mark_start):.2f}s .. {rel(mark_end):.2f}s  wall={wall if wall is None else round(wall, 2)}s")
    if err:
        print("request error:", err)
    else:
        print("reply:", repr(txt), "usage:", usage)

    print("\nlink speed by node (min / distinct values / samples at its maximum):")
    for b in NODES:
        vals = [s[b][0] for s in samples if s[b][0] is not None]
        mx = rd(f"{BASE}/{b}/max_link_speed")
        mx = float(mx.split()[0]) if mx and mx.split() else None
        distinct = sorted(set(vals))
        role = "endpoint" if b.endswith(":00.0") and not b.split(":")[1][:1].isalpha() else ""
        print(f"  {b.split(':', 1)[1]:<10} max {mx} GT/s   distinct {distinct}   "
              f"at-max {sum(1 for v in vals if mx and v >= mx)}/{len(vals)}")
    print("\nAER correctable by node (first -> last, delta over the window):")
    for b in NODES:
        first = samples[0][b + ":cor"]
        last = samples[-1][b + ":cor"]
        if first != last:
            print(f"  {b.split(':', 1)[1]:<10} {first} -> {last}  +{last - first}")
    moved = [b for b in NODES if samples[0][b + ":cor"] != samples[-1][b + ":cor"]]
    print("  (no change anywhere)" if not moved else "  moved: " + ", ".join(b.split(':', 1)[1] for b in moved))
    busy = [s["busy"]["card1"] for s in samples]
    print("\ncard1 busy % seen:", sorted(set(busy)))


if __name__ == "__main__":
    main()
