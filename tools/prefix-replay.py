#!/usr/bin/env python3
"""Would leaving the prefix cache ON pay for itself on THIS box's real conversations?

    tools/prefix-replay.py                     # replays ~/.pi/agent/sessions/*/*.jsonl
    tools/prefix-replay.py --pool-blocks 1175  # override the pool (default: read it from a live serve)
    tools/prefix-replay.py --speed 4           # what happens at 4x the arrival rate

Why a replay and not a benchmark. Every number already measured on this box about the prefix cache came from a
corpus built for the question: six copies of one 14,530-token filler. It says what a cache CAN do (133,120 of
346,754 prompt tokens skipped, wall 32.3 s -> 3.6 s) and nothing about what it DOES to the traffic this box
actually carries. That question is not about the cache at all -- it is about arrival order and capacity:

  * an agent harness resends the whole conversation every step, so consecutive requests from one session are
    prefix-nested, and the hit looks enormous (~99% of the prompt);
  * but a hit only exists if those blocks are STILL in the pool when the next request arrives, and the pool is
    1,175 blocks of 880 tokens (1,009,899 tokens) while a single deep session occupies hundreds of them;
  * and requests from different sessions interleave, so the interesting question is what survives that.

So this tool takes the box's own conversation records (`~/.pi/agent/sessions`: every model call with its
timestamp, its latency, and its full message content -- 6,534 calls over four days, one side of them against a
proxy in front of this machine), rebuilds the prompt of every call the way the harness sends it, cuts it into the
engine's own 880-token blocks, and simulates the engine's own rule: a request reuses the CONTIGUOUS prefix of
whole blocks that are resident, finished requests leave their blocks in a free pool that is LRU-evicted when
something else needs them, and a partial tail block is never shared.

Two runs, same events, same capacity: cache OFF (blocks return to the free list at once, nothing is shared) and
cache ON. The difference is the answer -- prompt tokens that would not be computed, the prefill seconds that buys
back at this box's measured read rate, and the cost side that a hit-rate number hides: how many cached blocks had
to be evicted, and how often the running set plus the retained set did not fit the pool (that is what turns into
queueing, and at the limit into preemption and re-prefill).
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

DEFAULT_SESSIONS = os.path.expanduser("~/.pi/agent/sessions/*/*.jsonl")
# The order pi concatenates its system sections in. The content is whatever the record says; the order only has
# to be stable, because the whole point is which blocks are BYTE-identical between two calls.
SECTIONS = ("preamble", "tools", "rules", "docs", "cwd")


def blocks_of(ids, size):
    """Whole-block hashes of a token sequence. The partial tail is not hashed: vLLM never shares a partial
    block, so counting it as a hit would be counting a token the engine would still compute."""
    out = []
    for i in range(0, len(ids) - size + 1, size):
        out.append(hashlib.blake2b(bytes(str(ids[i:i + size]), "utf-8"), digest_size=16).digest())
    return out


def render(msg):
    """The message as it goes back into the next request's prompt."""
    role = msg.get("role")
    parts = []
    c = msg.get("content")
    if isinstance(c, str):
        parts.append(c)
    elif isinstance(c, list):
        for b in c:
            t = b.get("type")
            if t == "text":
                parts.append(b.get("text") or "")
            elif t == "thinking":
                parts.append(b.get("thinking") or b.get("text") or "")
            elif t == "toolCall":
                parts.append(f"{b.get('name')}({json.dumps(b.get('arguments'), ensure_ascii=False)})")
    if role == "toolResult":
        parts.append(f"[tool {msg.get('toolName')} result]" + "".join(parts))
        parts = [parts[-1]]
    return f"<|{role}|>\n" + "\n".join(parts) + "\n"


def load_sessions(pattern, tokenizer):
    """Every model call in every session, as (t_start, dur_s, session, prompt_token_ids, client_input_tokens).

    The prompt of call k is the system sections plus everything the harness had put in the transcript by then --
    which is exactly what an agent resends. It is built incrementally (encode the delta, append the ids) because
    these transcripts run to 220,000 tokens and re-encoding the whole prefix per call would be quadratic. Delta
    encoding is not byte-identical to whole-string encoding at the seam (BPE merges across a boundary); it IS
    identical for both simulation runs, so it cannot move the comparison, and `fidelity()` below reports how far
    the reconstruction is from the token count the client itself recorded.

    Some session files have zero-filled regions (lines of NUL where a write died mid-flight), so a bad line is
    skipped and counted rather than taking the whole file with it: 1,050 of 6,540 calls live behind those lines,
    and a corpus that silently drops a sixth of itself would not be a test set.
    """
    calls, skipped = [], 0
    for path in sorted(glob.glob(pattern)):
        sid = Path(path).name
        sys_ids, ids = [], []
        try:
            fh = open(path)
        except OSError:
            continue
        for line in fh:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                skipped += 1                      # zero-filled or truncated record
                continue
            if r.get("type") == "compaction":
                # The harness rewrote its own context: from here the prefix is a different string. The record does
                # not keep the pre-compaction bytes, so the rebuild restarts from what it does have.
                ids = list(sys_ids)
            if r.get("type") != "message":
                continue
            m = r.get("message") or {}
            if m.get("role") == "system" and not sys_ids:
                sys_ids = tokenizer.encode("\n".join(
                    (m.get("sections") or {}).get(k, "") for k in SECTIONS)).ids
                ids = list(sys_ids)
                continue
            if m.get("role") == "assistant":
                u = m.get("usage") or {}
                calls.append((r.get("timestamp"), (m.get("durationMs") or 0) / 1000.0, sid, list(ids),
                              u.get("input"), u.get("cacheRead")))
            ids = ids + tokenizer.encode(render(m)).ids
        fh.close()
    return [c for c in calls if c[0]], skipped


def fidelity(calls):
    """How faithful the rebuilt prompts are, against the client's own accounting.

    The record stores `usage.input` AND `usage.cacheRead`. On the endpoint these sessions actually went to,
    `input` is only the NEW tokens of a call and `cacheRead` is the part it served from cache -- so the prompt of
    a call is `input + cacheRead`, and the fraction `cacheRead / (input + cacheRead)` is that endpoint's real
    measured hit rate on the same conversations this tool is replaying. Worth keeping side by side: it is one of
    the two independent checks on the simulated number.
    """
    pairs = [(len(c[3]), c[4] + (c[5] or 0)) for c in calls if c[4]]
    if not pairs:
        return None
    ratios = sorted(got / want for got, want in pairs)
    med = ratios[len(ratios) // 2]
    return {"calls": len(pairs), "median_ratio": med,
            "p10": ratios[len(ratios) // 10], "p90": ratios[-max(1, len(ratios) // 10)]}


def parse_ts(s):
    from datetime import datetime
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def simulate(calls, block, pool_blocks, speed=1.0, cache_on=True, read_rate=7428.0):
    """One pass over the replay, gated the way the engine gates: a request is admitted when its blocks are there.

    Arrival order and overlap come from the corpus -- each call starts at the timestamp the client recorded and
    holds its blocks for the latency that client measured. `speed` compresses the timeline, which is how the same
    four days of conversations look at N times the concurrency.

    Blocks are counted as INSTANCES, because that is what the pool counts. One occurrence of a block is one block:
    a transcript that contains the same 880 tokens twice pays for two of them, since only a CONTIGUOUS prefix can
    be shared. A prefix hit borrows the instance that is already in the pool (refcounted, so nobody evicts it from
    under a running request); a miss allocates a fresh one. When a request finishes, OFF hands its instances back
    to the pool and ON keeps them -- and a kept instance still occupies the pool, which is exactly the cost a hit
    rate leaves out.

    Returns the hit side (prompt tokens the engine would not compute, and the prefill seconds that is at this box's
    measured read rate) and the cost side: instances evicted to admit something, requests that had to wait for room,
    and the peak occupancy -- the number that decides whether a cache is affordable on this pool at all.

    One asymmetry to keep honest, because it points the conservative way: both runs hold their blocks for the
    latency the CLIENT recorded, and those latencies were measured in a world where a cache was serving 99% of the
    prompt. In the OFF run the same requests would also have spent the prefill the cache skips -- about 21 hours of
    it across this corpus -- before they started decoding, so they would hold their blocks longer and queue more.
    The queueing difference below is therefore a FLOOR on what OFF costs, not a ceiling on ON. A live A/B is what
    would put a number on that gap; this tool is what decides whether it is worth loading the box to find out.
    """
    ev = sorted(calls, key=lambda c: parse_ts(c[0]))
    t0 = parse_ts(ev[0][0])
    active = []                    # (finish_abs, [(instance_id, hash_or_None)])
    inst_hash = {}                 # instance_id -> content hash (None for a partial tail block)
    inst_ref = {}                  # instance_id -> requests currently using it
    cached_inst = {}               # content hash -> instance ids whose content is still in the pool
    live_retained = set()          # instances holding content that nothing is currently using
    lru = []                       # instance ids holding content nothing is using, oldest first
    used = 0                       # live instances: in use OR retained
    prompt_tokens = hit_tokens = evictions = waited = 0
    seq = 0                          # instance ids: every allocated block gets one, so counting is unambiguous
    over_cap = peak_used = low_water = 0
    prefill_s = 0.0
    for ts, dur, sid, ids, _inp, _cr in ev:
        now = t0 + (parse_ts(ts) - t0) / speed

        def retire(upto):
            """Requests that have finished. OFF returns their blocks; ON keeps them, and a kept block still
            occupies the pool -- which is precisely the cost a hit-rate number leaves out."""
            nonlocal used
            keep = []
            for finish, alloc in active:
                if finish > upto:
                    keep.append((finish, alloc))
                    continue
                for iid, h in alloc:
                    inst_ref[iid] -= 1
                    if inst_ref[iid] > 0:
                        continue                       # still borrowed by another request
                    del inst_ref[iid]
                    if h is None or not cache_on:      # a tail block is never retained; OFF never retains
                        del inst_hash[iid]
                        cached_inst.get(h, []).remove(iid) if h in cached_inst else None
                        used -= 1
                    else:
                        lru.append(iid)                # stays in `used`: retained, evictable, still holding it
                        live_retained.add(iid)
            active[:] = keep

        retire(now)
        whole = blocks_of(ids, block)
        tail = 1 if len(ids) % block else 0
        hit, hit_of = 0, []            # only the CONTIGUOUS prefix: the engine stops at the first miss
        for h in whole:
            iid = next((k for k in cached_inst.get(h, ()) if k in inst_ref or k in live_retained), None)
            if cache_on and iid is not None:
                hit += 1
                hit_of.append(iid)
            else:
                break
        need = len(whole) + tail - hit
        # admit: evict the oldest retained instances if that is what it takes, then wait for a running request
        i = 0
        while used + need > pool_blocks and i < len(lru):
            iid = lru[i]
            if iid in inst_ref or iid not in live_retained:
                i += 1                         # borrowed, or already reused/evicted: a stale entry, skip it
                continue
            live_retained.discard(iid)
            h = inst_hash.pop(iid)
            if h is not None:
                lst = cached_inst.get(h) or []
                if iid in lst:
                    lst.remove(iid)
            used -= 1
            evictions += 1
            lru[i] = None
            i += 1
        if evictions:
            lru[:] = [x for x in lru if x is not None]
        while used + need > pool_blocks and active:
            nxt = min(f for f, _ in active)    # nothing evictable: the request waits for the next finisher
            waited += (nxt - now) * speed
            now = nxt
            retire(now)
        if used + need > pool_blocks:
            over_cap += 1                      # does not fit even an empty pool: queueing cannot help
        alloc = []
        for n, h in enumerate(whole):
            if n < hit:                        # borrow the instance that is already in the pool
                iid = hit_of[n]
                live_retained.discard(iid)     # it stops being evictable the moment a request is using it
            else:
                seq += 1
                iid = seq
                inst_hash[iid] = h
                cached_inst.setdefault(h, []).append(iid)
                used += 1
            inst_ref[iid] = inst_ref.get(iid, 0) + 1
            alloc.append((iid, h))
        if tail:                               # the partial block is allocated and never shared
            seq += 1
            iid = seq
            inst_hash[iid] = None      # tail blocks hold no reusable content
            inst_ref[iid] = 1
            alloc.append((iid, None))
            used += 1
        peak_used = max(peak_used, used)
        low_water = min(low_water, pool_blocks - used)
        active.append((now + max(dur, 0.001), alloc))
        prompt_tokens += len(ids)
        hit_tokens += hit * block
        prefill_s += (len(ids) - hit * block) / read_rate
    return {"requests": len(calls), "prompt_tokens": prompt_tokens, "hit_tokens": hit_tokens,
            "hit_pct": 100.0 * hit_tokens / prompt_tokens if prompt_tokens else 0.0,
            "prefill_s": prefill_s, "admission_waits_s": waited, "evictions": evictions,
            "over_capacity_requests": over_cap, "peak_blocks": peak_used,
            "low_water_free_blocks": pool_blocks - peak_used,
            # live_retained, not len(lru): the LRU list keeps stale entries for instances that were borrowed
            # again, and reporting its length would say the cache holds more than it does
            "retained_blocks_at_end": len(live_retained)}


def pool_from_serve(url):
    try:
        with urllib.request.urlopen(f"{url}/api/status", timeout=8) as f:
            c = json.load(f).get("config") or {}
        return int(c.get("num_gpu_blocks")), int(c.get("effective_attention_block_size") or c.get("block_size"))
    except Exception:
        return None, None


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default=DEFAULT_SESSIONS)
    ap.add_argument("--pool-blocks", type=int, default=None)
    ap.add_argument("--block", type=int, default=None)
    ap.add_argument("--serve", default="http://127.0.0.1:8080")
    ap.add_argument("--read-rate", type=float, default=7428.0,
                    help="prompt tok/s this box reads (measured 2026-11-04 on the prefill card)")
    ap.add_argument("--speed", type=float, default=1.0, help="load multiplier: 4 = four times as many sessions per hour")
    a = ap.parse_args(argv)
    pool, block = a.pool_blocks, a.block
    if pool is None or block is None:
        p, b = pool_from_serve(a.serve)
        pool = pool or p or 1175
        block = block or b or 880
    tok = None
    for cand in (os.path.expanduser("~/models"),):
        for d in Path(cand).iterdir():
            if (d / "tokenizer.json").exists():
                from tokenizers import Tokenizer
                tok = Tokenizer.from_file(str(d / "tokenizer.json"))
                break
        if tok:
            break
    if tok is None:
        print("no tokenizer.json under ~/models -- cannot count tokens", file=sys.stderr)
        return 2
    calls, skipped = load_sessions(a.sessions, tok)
    if not calls:
        print("no model calls found in the session records", file=sys.stderr)
        return 2
    fid = fidelity(calls)
    cre = sum((c[5] or 0) for c in calls)
    cin = sum((c[4] or 0) for c in calls)
    off = simulate(calls, block, pool, a.speed, False, a.read_rate)
    on = simulate(calls, block, pool, a.speed, True, a.read_rate)
    saved = on["hit_tokens"]
    print(f"corpus: {len(calls)} model calls"
          + (f" ({skipped} records unreadable in the session files)" if skipped else "")
          + f", {off['prompt_tokens']:,} prompt tokens")
    if fid:
        print(f"  fidelity vs the client's own prompt counts (input + cacheRead): median ratio {fid['median_ratio']:.3f} "
              f"(p10 {fid['p10']:.3f}, p90 {fid['p90']:.3f})")
    if cre + cin:
        print(f"  that endpoint's OWN measured hit rate on these calls: cacheRead {cre:,} of {cre+cin:,} prompt "
              f"tokens = {100.0*cre/(cre+cin):.1f}% ({len(calls)} calls) -- the upstream it talks to runs its cache ON")
    print(f"  model: block {block} tokens, pool {pool} blocks = {pool*block:,} tokens; "
          f"arrivals compressed x{a.speed}; prefill read rate {a.read_rate:,.0f} tok/s")
    hdr = (f"  {'':22s}{'prompt tok':>12s}{'hit':>8s}{'prefill s':>11s}{'waits s':>9s}"
           f"{'evict':>8s}{'over cap':>9s}{'peak blk':>10s}{'free low':>9s}")
    print(hdr)
    for name, r in (("prefix cache OFF", off), ("prefix cache ON", on)):
        print(f"  {name:22s}{r['prompt_tokens']:>12,d}{r['hit_pct']:>7.1f}%{r['prefill_s']:>11,.0f}"
              f"{r['admission_waits_s']:>9,.0f}{r['evictions']:>8,d}{r['over_capacity_requests']:>9,d}"
              f"{r['peak_blocks']:>10,d}{r['low_water_free_blocks']:>9,d}")
    print(f"\n  ON buys back {saved:,} prompt tokens = {saved / a.read_rate:,.0f} s of prefill "
          f"({100.0*(saved/a.read_rate)/off['prefill_s']:.1f}% of the OFF run's prefill), at a cost of {on['evictions']:,} "
          f"retained blocks evicted and {on['retained_blocks_at_end']:,} still held when the replay ended.")
    print(f"  queueing: OFF waited {off['admission_waits_s']:,.0f} s across the window, "
          f"ON {on['admission_waits_s']:,.0f} s"
          + (" -- ON queues MORE: the retained blocks are holding room the next request needed."
             if on["admission_waits_s"] > off["admission_waits_s"] else
             " -- ON queues no more than OFF, so the retention is not costing admission."))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
