#!/usr/bin/env python3
"""Which physical card each vLLM TP rank actually holds, read out of the kernel's own records.

Why this exists (2026-11-15, notes/bug-causal-conv1d-oob.md §15): every `[gfxhub] page fault` this box has taken
named card `0000:c3:00.0`, and every one whose rank was known named `Worker_TP0` -- which under
`HIP_VISIBLE_DEVICES=0,1,2,3` is the SAME fact stated twice, so it proves nothing. Permuting `GPUS=3,0,1,2` moves
a different card under rank 0, and to read the result you have to know which card each rank is on WITHOUT
re-deriving it from the enumeration you just deliberately shuffled.

Method, and the trap it avoids:

* **Not `/proc/<pid>/fd`.** Every rank has all four `/dev/dri/renderD*` open (P2P/IPC), so the fd list identifies
  nothing -- printed once during the hunt and it showed all four for all four ranks.
* **`/sys/class/kfd/kfd/proc/<pid>/vram_<gpu_id>` instead.** KFD keeps a per-process, per-GPU record, and the rank
  that owns a card has ~28.8 GiB in it while the others sit at a few MB of scratch. That is ground truth about
  the running process, independent of any enumeration order.
* **`gpu_id` -> PCI** comes from `/sys/class/kfd/kfd/topology/nodes/*/properties`'s `location_id`, which is the
  PCI devfn in hex (`49920 = 0xc300 -> c3:00.0`). The node order is what `HIP_VISIBLE_DEVICES` indexes, so the
  enumeration is printed too -- and can then be seen to be shuffled, rather than assumed.

Read-only. Exits 3 when no engine is up, so it can be dropped into a check without lying.
"""

import argparse
import glob
import json
import os
import sys


def topology():
    """gpu_id -> (kfd node number, 'c3:00.0'). GPU nodes are the ones with no cpu cores."""
    out = {}
    for n in glob.glob("/sys/class/kfd/kfd/topology/nodes/*"):
        try:
            props = open(n + "/properties").read()
        except OSError:
            continue
        if not props.startswith("cpu_cores_count 0"):
            continue
        gid = int(open(n + "/gpu_id").read())
        loc = int([l.split()[1] for l in props.splitlines() if l.startswith("location_id")][0])
        out[gid] = (int(os.path.basename(n)), "%02x:%02x.0" % (loc >> 8, loc & 0xFF))
    return out


def workers():
    """Every live vLLM worker rank: pid -> rank name, from its own argv (the name is in the process title)."""
    out = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            title = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")[0].decode("utf-8", "replace")
        except OSError:
            continue
        if title.startswith("VLLM::Worker"):
            out[int(pid)] = title
    return out


def card_of(pid, gpu2card):
    """(card, GiB) for the GPU whose per-process KFD record is loaded -- i.e. the card this rank owns."""
    rows = []
    for d in glob.glob(f"/sys/class/kfd/kfd/proc/{pid}/vram_*"):
        gid = int(d.rsplit("_", 1)[1])
        try:
            vals = [int(x) for x in open(d).read().split()]
        except (OSError, ValueError):
            continue
        if vals:
            rows.append((max(vals), gid))
    if not rows:
        return None, None
    big, gid = max(rows)
    return gpu2card.get(gid, f"gpu_id {gid}"), big / 2**30


def main():
    ap = argparse.ArgumentParser(description="rank -> physical card, from KFD, for the running engine")
    ap.add_argument("--json", action="store_true", help="machine-readable, one object per rank")
    a = ap.parse_args()

    gpu2card = {g: v[1] for g, v in topology().items()}
    if not gpu2card:
        print("no KFD GPU nodes: this box has no amdgpu driver loaded", file=sys.stderr)
        return 3
    order = sorted(gpu2card, key=lambda g: topology()[g][0])
    vis = os.environ.get("HIP_VISIBLE_DEVICES", "(unset)")

    ws = workers()
    if not ws:
        print("no VLLM::Worker process is up -- nothing to map", file=sys.stderr)
        return 3

    # the engine's own environment, not this shell's: the whole point is what the launch asked for, and a shell
    # that ran `GPUS=3,0,1,2 ... setup.py --launch` does not carry that variable into the tool's own process.
    vis = "(not readable)"
    for any_pid in ws:                                      # try every rank until one answers: /proc reads race
        try:                                                # against worker restarts, and a rank that is going
            for kv in open(f"/proc/{any_pid}/environ", "rb").read().split(b"\0"):  # away has an unreadable environ
                if kv.startswith(b"HIP_VISIBLE_DEVICES="):
                    vis = kv.split(b"=", 1)[1].decode("utf-8", "replace")
                    break
        except OSError:
            continue
        if vis != "(not readable)":
            break
    if vis == "(not readable)":                             # on this box it never is: kernel.yama.ptrace_scope=1
        try:                                                # means a process that is not a descendant cannot read
            scope = open("/proc/sys/kernel/yama/ptrace_scope").read().strip()  # anyone else's environ, and the
        except OSError:                                     # engine's processes are all non-dumpable besides (-r--
            scope = "?"                                     # ---- and size 0). Say where to look instead of lying.
        vis = f"(not readable from here: yama ptrace_scope={scope} hides other processes' environ; "
        vis += "the order the launch asked for is in its own command, `GPUS=... bash serve/<recipe>`"
        vis += ", and the order that resulted is the rank table below)"

    rows = []
    for pid, rank in sorted(ws.items(), key=lambda kv: kv[1]):
        card, gib = card_of(pid, gpu2card)
        rows.append({"pid": pid, "rank": rank, "card": card, "gib": round(gib, 1) if gib else None})

    if a.json:
        for r in rows:
            print(json.dumps(r))
        return 0

    print("KFD enumeration order (what HIP_VISIBLE_DEVICES indexes; the engine was launched with %s):" % vis)
    for i, g in enumerate(order):
        print(f"  index {i}: {gpu2card[g]}")
    print("rank -> the card it actually holds (largest per-process KFD vram record):")
    for r in rows:
        print(f"  {r['rank']:<20} pid {r['pid']:<7} {'holds %5.1f GiB on %s' % (r['gib'], r['card']) if r['gib'] else 'no KFD vram record'}")
    print("\ncompare with the driver's own words when it faults: `journalctl -k | grep -E 'gfxhub|Process python3'`")
    print("names a PCI function AND a pid; the pid is one of these, and the card it names is the answer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
