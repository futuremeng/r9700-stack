#!/usr/bin/env python3
"""serve/gateway.py - one port, one front door: path routing to the engine and the two CPU sidecars.

Why this file exists, and why it is a PROPOSAL rather than a default: :8080 belongs to the running vLLM
process, so anything that wants to answer on 8080 is either the engine itself (an endpoint plugin -- a
restart, and proxying inside the process that produces the numbers) or something standing in front of it
(this: the engine moves to a loopback port, the gateway owns 8080). Both need the launch restarted, so
nothing here is wired into a launch until that is a decision and not an experiment.

What it is NOT: it is not serve/webui.py's proxy mode. That one parses and re-emits request bodies, which is
why it costs 2-3 ms per request (bench/webui_ab.py: +5.3 % of a 76-token request, +0.5 % of a 4.4k one). This
reads the request LINE, picks an upstream from the path, and then pipes bytes: no body parsing, no
re-encoding, nothing held in memory. Response framing (Content-Length / chunked) is honoured, so HTTP
keep-alive and SSE streaming pass through untouched.

Routing (longest first segment wins; unmapped goes to the engine, because the engine is the thing that must
not notice this exists):

    /v1/embeddings                         -> EMBED   (llama-server, :8090)
    /v1/rerank /rerank /reranking          -> RERANK  (llama-server, :8092)
    /monitor /about /web/* /api/* /favicon -> PAGES   (default the engine's own plugin pages)
    /api/sidecars                          -> this file: the embed-watch pages' state, for one Monitor page
    /api/gate                              -> this file: the admission gate's own state (see "the gate" below)
    everything else (/v1/chat/completions, /health, /metrics, /tokenize, ...) -> ENGINE

/Metrics is the ENGINE's: the Monitor's parser speaks vLLM's Prometheus, and llama.cpp's dialect has its own
watcher (serve/embed-watch.py). Each sidecar's /health stays internal; the gateway's /health answers for the
engine, which is what setup.py's watcher and every client already poll.

Knobs (all env): GATEWAY_PORT (18080), GATEWAY_BIND (127.0.0.1), ENGINE, EMBED, RERANK,
PAGES (default ENGINE; point it at the webui sidecar http://127.0.0.1:8081 for a page that survives the
engine dying), SIDECAR_STATE (comma-separated embed-watch state URLs, e.g.
http://127.0.0.1:8091/state.json,http://127.0.0.1:8093/state.json), and the gate's own GATE_* set below.

## the gate: two budgets, and the live one is NOT the sequence count

The box's own request journal and the engine's 10 s stats say a decode step costs

    step_ms ~= 13.6  +  1.79 x N(sequence)  +  14.9 ns x Sum(resident tokens)

(fitted on 28 clean pure-decode windows of the 6.8-day launch, R^2 0.89; checked against 103 windows of the
11-15 launch -- e.g. N=6 / 251,701 tokens: 77 tok/s/req measured against 79 predicted). That decomposes into
two budgets with different exchange rates, and `--max-num-seqs` holds the wrong one:

  * per-step time -- one 80k-token row costs 1.6x a 1.5k row, one 236k row 2.9x. Concurrency is cheap here:
    the 13.6 ms fixed part gets amortised, which is why total throughput climbs 144 -> ~1000 tok/s.
  * the KV pool -- 902,153 tokens (6.65 GiB, ~7.7 KB/token), and a sequence costs nothing there while its
    tokens cost 1:1. One 80k row is 8.9% of the pool: the same footprint as 53 short rows. **50x per row.**

Both crashes this box has taken (2026-11-14 20:59, 2026-11-15 05:11) were steps carrying rows at 236k and 80k
own-context; the slow windows of the long launch (11-12, 11-14) sat at kv_cache_usage 84-100% with a waiting
queue, 423 preemptions in the last 3.3 h and TTFT p99 376 s / max 1 960 s. So the gate's decision variable is
the pool -- the engine's own gauges -- and never a sequence count: `--max-num-seqs 32` was in the argv of every
launch that was slow AND of the launch that was fine. So the gate's decision variable is what the ENGINE
reports about itself, never a sequence count: `--max-num-seqs 32` was in the argv of every launch that was slow
AND of the launch that was fine.

WHAT THE TRIGGER IS NOT, learned the hard way on 2026-11-15. The first version closed on pool level (>= 0.85,
re-open below 0.78) or on any engine queue (waiting >= 1). A day later a batch of ~100k-token data-processing
prompts parked the pool at 0.87-0.90 for its whole duration with 17 requests waiting on scheduling capacity --
and num_preemptions_total moved ONCE. v1 read those numbers as an emergency, and its own history said so: 180 of
180 polls shut, 14 requests held, 9 of them admitted anyway after waiting the entire 25 s ceiling. A door shut on
a full-but-not-thrashing pool protects nothing: the KV is already held, the engine is queueing its own work in
the order it chose, and the only measured effect was +25 s on every new arrival -- paid mostly by the small
requests, which are the cheapest thing this box does. A long batch must not be able to shut the door for the
whole of its own duration.

So the trigger is HARM, and each axis closes on its own terms:
  * preemption activity -- vllm:num_preemptions_total MOVING between polls is the pool taking blocks back from a
    running request, which is what the 11-12 and 11-14 slow windows' victims paid a whole re-prefill for. The
    door stays shut GATE_PREEMPT_S after the last event: a window, which ends, unlike a pool threshold that a
    resident batch makes unreachable.
  * a pool near full -- GATE_KV_HIGH, with hysteresis down to GATE_KV_LOW on the POOL AXIS ONLY (a door shut by a
    preemption must not be latched shut by the pool's release line).
  * GATE_WAIT remains as an operator backstop on the engine queue. Its shipped default is 0: a queue is what
    saturation looks like, not what thrash looks like.
And three escapes, because "held" must never quietly become "queued twice":
  * a request whose DECLARED body is <= GATE_SMALL_BYTES passes a shut door. Header only -- the pipe still never
    reads a body to decide anything, but it already holds that line to frame the upstream request, and it is
    enough to tell a 200-token question from a 100k-token document.
  * at most GATE_MAX_HELD requests are ever held at once; past that the gate admits, because a gate with an
    unbounded hold queue has just become a second queue, and the engine already has one.
  * the hold ceiling GATE_HOLD_S is the client-visible bound, shipped at 5 s -- 9 of v1's 14 holds ended by
    hitting the 25 s ceiling, and a bound that is usually reached is a delay, not a bound. GATE_HOLD_S=0 turns
    the hold into a 503 that names the reason.

What the gate does while closed: it opens no upstream connection. The client simply waits -- HTTP backpressure,
no body read, nothing buffered, the same refusal to parse that the whole file exists for. It is a hold, not a
rejection.

What it deliberately does NOT do: estimate prompt tokens. A body-reading proxy costs 2-3 ms a request
(bench/webui_ab.py) and a Content-Length/N token estimate is a number no test could hold, so the declared length
is used only as a small/big switch and every other number comes from what the engine already measured about
itself: vllm:kv_cache_usage_perc, vllm:num_requests_running, vllm:num_requests_waiting and
vllm:num_preemptions_total, one GET /metrics per GATE_METRICS_S (12 ms and 68 KB on this box, in a coroutine, off
the request path).

Fail-open is the default on purpose: a gauge that cannot be read, or a vLLM that renamed it, opens the gate
and says so in `err`. A front door that stops serving because its telemetry broke is worse than no gate.

Knobs: GATE_DISABLE=1 (off), GATE_KV_HIGH (0.92 -- close), GATE_KV_LOW (0.85 -- re-open the pool axis: one
8 192-token prefill moves this pool ~0.9%, so a single threshold would chatter), GATE_PREEMPT_S (20 -- stay shut
this long after the last observed preemption), GATE_WAIT (0 shipped: an engine queue alone never closes the
door; set it to make the queue a trigger), GATE_SMALL_BYTES (8192 -- a declared body this size or smaller passes
a shut door; 0 disables), GATE_MAX_HELD (8 -- never hold more than this many at once), GATE_MAX_INFLIGHT
(0 = off; the sequence axis, kept as a knob because the two budgets are independent), GATE_HOLD_S (5 s ceiling;
0 = refuse instead of hold), GATE_METRICS_S (1.0), GATE_STALE_S (5). Only /v1/chat/completions and
/v1/completions are gated -- /health is what setup.py's
watcher polls, /metrics is what the gate itself reads, /api/* is the page, and the two sidecar routes are CPU
serves that cannot take a KV hit.
"""
import asyncio
import json
import os
import re
import sys
import time
import urllib.request

ENGINE = os.environ.get("ENGINE", "http://127.0.0.1:8080")
EMBED = os.environ.get("EMBED", "http://127.0.0.1:8090")
RERANK = os.environ.get("RERANK", "http://127.0.0.1:8092")
PAGES = os.environ.get("PAGES", ENGINE)
SIDECAR_STATE = [s.strip() for s in os.environ.get("SIDECAR_STATE", "").split(",") if s.strip()]
BIND = os.environ.get("GATEWAY_BIND", "127.0.0.1")
PORT = int(os.environ.get("GATEWAY_PORT", "18080"))
CHUNK = 64 * 1024
# one upstream connection per request, closed after the response body: the gateway keeps no state, which is
# what makes it safe to stand in front of a five-day-old serve
# Longest prefix wins, and the match is on the whole path prefix, not on one segment: the first version split
# the path into "first segment" and compared, so /v1/embeddings arrived as "v1" and went to the ENGINE, which
# answered its own {"detail":"Not Found"} -- a 404 from the wrong server (2026-11-15, first test of this file).
#
# But prefix matching on an API path has the mirror-image bug, which tests/test_gateway.py caught the same day:
# /v1/embeddings-x went to the EMBEDDER and came back as llama.cpp's 404 -- again the wrong server answering,
# this time on a path the gateway should never have claimed. So API paths match EXACTLY (query string
# stripped); only the two namespaces the pages own match as prefixes.
EXACT = {"/v1/embeddings": EMBED, "/v1/embeddings/": EMBED, "/v1/rerank": RERANK, "/rerank": RERANK,
         "/reranking": RERANK, "/monitor": PAGES, "/about": PAGES, "/favicon.svg": PAGES}
PREFIX = (("/web/", PAGES), ("/api/", PAGES))

# ---- the gate -------------------------------------------------------------------------------------------
# Policy v2, rewritten on 2026-11-15 after a day of production data said v1 was wrong in a specific way: with a
# batch of ~100k-token data-processing requests resident, the pool parks at 0.87-0.90 for the whole batch, the
# engine queues 17 of them on its own (all reason="capacity"), and the preemption counter went up ONCE. v1 closed
# the door on `kv >= 0.85` or `waiting >= 1` and re-opened only below 0.78 -- a level five resident 100k-token
# contexts keep the pool above by construction. Result, read off the gate's own history: 180 of 180 polls closed,
# 14 requests held, 9 of them admitted anyway after waiting the full 25 s. That is a permanent 25-second tax on
# every new request, paid for a harm that was not happening (notes/admission-gate.md).
#
# So the trigger moved from LEVEL to HARM, and the door got two escapes:
#   * preemption activity is the harm -- the pool taking blocks back from a running request, which is what made
#     the 11-14 slow window (68 preempted requests). A rising preemption counter closes the door for
#     GATE_PREEMPT_S after the last event, which is a window that actually ends.
#   * a pool that is merely full is not: `waiting` alone no longer closes anything. The engine queueing its own
#     work is the engine's job, and a second queue in front of it adds latency, hides the engine's ordering, and
#     changes nothing about the KV already held.
#   * requests declaring a small body pass even while closed (header only -- the pipe never reads a body to make
#     a decision). Small requests are the ones v1 starved, and they are not what fills the pool.
#   * the hold ceiling got short (25 s -> 5 s) because the data said the ceiling, not the release, is how most
#     held requests left; and GATE_MAX_HELD bounds the number held at all, because a gate with an unbounded hold
#     queue has just become a queue with extra steps.
GATE_ON = os.environ.get("GATE_DISABLE", "0") != "1"
GATE_KV_HIGH = float(os.environ.get("GATE_KV_HIGH", "0.92"))     # near-full backstop; preempt usually starts here
GATE_KV_LOW = float(os.environ.get("GATE_KV_LOW", "0.85"))       # re-open line (hysteresis)
GATE_PREEMPT_S = float(os.environ.get("GATE_PREEMPT_S", "20"))   # close this long after the last preemption
GATE_WAIT = int(os.environ.get("GATE_WAIT", "0"))                # 0 = waiting alone never closes (v1's 1 did)
GATE_MAX_INFLIGHT = int(os.environ.get("GATE_MAX_INFLIGHT", "0"))
GATE_SMALL_BYTES = int(os.environ.get("GATE_SMALL_BYTES", "8192"))   # a declared body at most this big passes
GATE_MAX_HELD = int(os.environ.get("GATE_MAX_HELD", "8"))        # never a second queue: beyond this, admit
GATE_HOLD_S = float(os.environ.get("GATE_HOLD_S", "5"))          # v1's 25 s was the usual exit, not the exception
GATE_POLL_S = float(os.environ.get("GATE_METRICS_S", "1.0"))
GATE_STALE_S = float(os.environ.get("GATE_STALE_S", "5"))
GATED = ("/v1/chat/completions", "/v1/completions")
G = {"kv": None, "run": None, "wait": None, "at": 0.0, "err": None, "closed": False, "why": "",
     # preemption watch: the counter's last value, when it last moved, and until when the door stays shut.
     "prem": None, "prem_seen": 0, "thrash_until": 0.0, "pool_shut": False,
     # display only: turns the usage FRACTION into tokens in the reason string and in /api/gate, so a reader
     # sees "770,000 of 902,153 tokens" rather than 0.854. The decision itself never reads it -- the fraction
     # is what the engine measures, and this box's pool is 902,153 tokens (6.65 GiB, from its own boot log).
     "pool_tokens": int(os.environ.get("GATE_POOL_TOKENS", "902153")),
     "inflight": 0, "held": 0, "held_now": 0, "released": 0, "expired": 0, "refused": 0, "admitted": 0,
     "small_passed": 0, "held_over": 0, "polls": 0,
     # One entry per poll: 1 shut, 0 open. A poll whose gauges were unusable draws 0 on purpose: fail-open is
     # shown, not hidden, because a strip that goes blank when the reading dies would blame the traffic for a
     # telemetry fault. 180 polls ~= 3 min at the default GATE_METRICS_S -- long enough to watch a pool climb
     # into the cliff and stay there, short enough that one bar is one decision a reader can line up against a
     # request row. The span is tunable already (hist_n x GATE_METRICS_S), so hist_n stays a constant.
     "hist": [], "hist_n": 180,
     "task": None}

# vLLM 0.32 spells them this way (verified against the live /metrics, 2026-11-15). `num_requests_waiting{` is
# written with its brace on purpose: without it the prefix also matches num_requests_waiting_by_reason, and a
# request waiting for a reason the scheduler chose would look like a queue the engine chose to grow.
GAUGES = ((b"vllm:kv_cache_usage_perc", "kv"), (b"vllm:num_requests_running{", "run"),
          (b"vllm:num_requests_waiting{", "wait"), (b"vllm:num_preemptions_total{", "prem"))


async def read_gauges():
    """One /metrics read. Aggregate: max over models/engines for a usage fraction, sum for counts."""
    try:
        host, port = split_url(ENGINE)
        r, w = await asyncio.open_connection(host, port)
        w.write(b"GET /metrics HTTP/1.1\r\nhost: gate\r\naccept: text/plain\r\nconnection: close\r\n\r\n")
        await w.drain()
        body = b""
        while len(body) < 512 * 1024:
            d = await r.read(CHUNK)
            if not d:
                break
            body += d
        try:
            w.close()
        except OSError:
            pass
    except (OSError, ValueError) as e:                       # a dead engine is the gate's data, not its policy
        G["err"] = f"{type(e).__name__}: {e}"
        return
    got = {"kv": [], "run": [], "wait": [], "prem": []}
    for line in body.split(b"\n"):
        for pre, key in GAUGES:
            if line.startswith(pre):
                try:
                    got[key].append(float(line.rsplit(b" ", 1)[1]))
                except (ValueError, IndexError):
                    pass
                break
    if not got["kv"]:
        G["err"] = "no vllm:kv_cache_usage_perc in the answer"    # a rename must not close the gate
        return
    G["kv"] = max(got["kv"])
    G["run"] = sum(got["run"]) if got["run"] else None
    G["wait"] = sum(got["wait"]) if got["wait"] else None
    # The preemption counter is cumulative since the engine started, so what matters is whether it MOVED: one
    # event is the pool taking blocks back from a running request, and a moving one is the thrash the 11-14 slow
    # window was made of. A jump since the last poll also covers a poll we missed.
    prem = sum(got["prem"]) if got["prem"] else None
    if prem is not None:
        if G["prem"] is not None and prem > G["prem"]:
            G["prem_seen"] += int(prem - G["prem"])
            G["thrash_until"] = time.time() + GATE_PREEMPT_S
        G["prem"] = prem
    G["at"], G["polls"], G["err"] = time.time(), G["polls"] + 1, None


def gate_state():
    """(closed, why) -- HARM first: preemption activity, then a pool near full. Never "the engine has a queue".

    The order is the reflection of 2026-11-15's data. A batch of ~100k-token prompts parks the pool at 0.87-0.90
    and leaves 17 requests waiting on scheduling capacity, and that is the engine doing its job: the preemption
    counter moved once all day, and v1's thresholds kept the door shut for 180 of 180 polls while it happened.
    A door that is shut on a full-but-not-thrashing pool does not protect the pool -- the KV is already held --
    it only delays new arrivals, and it delays the small ones most, which are the cheapest things the box can do.

    Hysteresis stays for the pool backstop (re-open at LOW, not HIGH: one 8 192-token prefill moves this pool
    ~0.9 %, so one threshold would chatter).
    """
    if not GATE_ON:
        return False, "off"
    if G["at"] == 0.0 or time.time() - G["at"] > GATE_STALE_S:
        return False, "no gauge"                            # unknown OPENS the gate
    now = time.time()
    # Hysteresis belongs to the pool axis alone. It used to key off the door's own state, which meant a door shut
    # by a preemption stayed shut until the pool fell below LOW -- under a long batch that never happens, which is
    # half of how v1 ended up permanently closed. The pool axis keeps its own latch; the preemption axis has a
    # timer instead.
    hi = GATE_KV_LOW if G["pool_shut"] else GATE_KV_HIGH
    G["pool_shut"] = G["kv"] is not None and G["kv"] >= hi
    why = []
    left = G["thrash_until"] - now
    if left > 0:
        # Named first, because it is the reason with a mechanism behind it: blocks taken back from a running
        # request, which is what a victim pays a whole re-prefill for.
        why.append(f"{G['prem_seen']} preemption(s) observed, door shut {left:.0f} s more "
                   f"(window {GATE_PREEMPT_S:.0f} s)")
    if G["pool_shut"]:
        why.append(f"kv {G['kv']:.3f} >= {hi} ({G['kv'] * G['pool_tokens']:,.0f} of {G['pool_tokens']:,} tokens)")
    if GATE_WAIT > 0 and (G["wait"] or 0) >= GATE_WAIT:
        why.append(f"engine waiting {G['wait']}")
    if GATE_MAX_INFLIGHT > 0 and G["inflight"] >= GATE_MAX_INFLIGHT:
        why.append(f"in flight {G['inflight']} >= {GATE_MAX_INFLIGHT}")
    return (bool(why), "; ".join(why))


def declared_length(headers):
    """How big the request DECLARES its body is, from the header bytes already in hand. None if chunked/absent.

    Header only, on purpose. The pipe still never reads a body to make a decision -- but it already holds this
    line to frame the upstream request, and a declared size is enough to tell a 200-token question from a
    100k-token document, which is the difference the door was starving.
    """
    m = re.search(rb"\r?\ncontent-length: *(\d+)", headers.lower())
    return int(m.group(1)) if m else None


async def admit(path, declared=None):
    """Hold a new engine request while the gate is closed. (ok, held_s, why).

    Holding means: no upstream connection, no body read. Three escapes, all of them measured needs:
      * a declared-small body passes a shut door (v1 starved exactly these); GATE_SMALL_BYTES=0 turns that off;
      * at most GATE_MAX_HELD requests are ever held -- past that, admit, because a gate with an unbounded hold
        queue has just become a queue with extra steps and the engine already has one;
      * the ceiling GATE_HOLD_S is the client-visible bound. It is 5 s, not v1's 25, because the counter said the
        ceiling (not the re-open) was how most held requests left: 14 held, 9 expired. A bound that is usually
        hit is not a bound, it is a delay.
    """
    if path not in GATED:
        return True, 0.0, "not gated"
    if GATE_SMALL_BYTES > 0 and declared is not None and declared <= GATE_SMALL_BYTES:
        G["admitted"], G["small_passed"] = G["admitted"] + 1, G["small_passed"] + 1
        return True, 0.0, f"declared {declared} B <= {GATE_SMALL_BYTES} B: small requests pass a shut door"
    G["inflight"] += 1
    t0, was_held = time.time(), False
    try:
        while True:
            closed, why = gate_state()
            if not closed:
                if was_held:
                    G["released"] += 1
                else:
                    G["admitted"] += 1
                return True, time.time() - t0, why
            if not was_held:
                if GATE_MAX_HELD > 0 and G["held_now"] >= GATE_MAX_HELD:
                    G["admitted"], G["held_over"] = G["admitted"] + 1, G["held_over"] + 1
                    return True, 0.0, f"{G['held_now']} already held: admitting rather than queueing behind them"
                was_held = True
                G["held"], G["held_now"] = G["held"] + 1, G["held_now"] + 1
            if GATE_HOLD_S <= 0:
                G["refused"] += 1
                return False, 0.0, why
            if time.time() - t0 >= GATE_HOLD_S:
                G["expired"] += 1                           # the ceiling is the promise; keep it
                return True, time.time() - t0, f"held past the {GATE_HOLD_S:.0f} s ceiling: {why}"
            await asyncio.sleep(min(0.25, GATE_POLL_S or 0.25))
    finally:
        G["inflight"] -= 1
        if was_held:
            G["held_now"] -= 1


def gate_blob():
    closed, why = gate_state()
    G["closed"], G["why"] = closed, why
    return json.dumps({"closed": closed, "why": why, "gauges": {k: G[k] for k in ("kv", "run", "wait", "prem",
                                                                                   "at")},
                       "err": G["err"], "inflight": G["inflight"], "polls": G["polls"],
                       "thrash_left": max(0.0, G["thrash_until"] - time.time()),
                       "prem_seen": G["prem_seen"], "pool_shut": G["pool_shut"],
                       "hist": G["hist"], "hist_n": G["hist_n"],
                       "gated": list(GATED),
                       "counters": {k: G[k] for k in ("held", "held_now", "released", "expired", "refused",
                                                      "admitted", "small_passed", "held_over")},
                       "config": {"GATE_ON": GATE_ON, "GATE_KV_HIGH": GATE_KV_HIGH, "GATE_KV_LOW": GATE_KV_LOW,
                                  "GATE_PREEMPT_S": GATE_PREEMPT_S, "GATE_WAIT": GATE_WAIT,
                                  "GATE_MAX_INFLIGHT": GATE_MAX_INFLIGHT, "GATE_SMALL_BYTES": GATE_SMALL_BYTES,
                                  "GATE_MAX_HELD": GATE_MAX_HELD,
                                  "GATE_HOLD_S": GATE_HOLD_S, "GATE_METRICS_S": GATE_POLL_S,
                                  "GATE_STALE_S": GATE_STALE_S, "pool_tokens": G["pool_tokens"]},
                       "at": time.time()}).encode()


async def gate_page(writer):
    """GET /api/gate -- the gate as data, so the Monitor can show why a request is late. Keep-alive survives it,
    exactly as with /api/sidecars."""
    body = gate_blob()
    writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nCache-Control: no-store\r\n"
                 b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
    await writer.drain()


async def gate_poll():
    while True:
        await read_gauges()
        # the strip is the poller's own record, not something /api/gate reconstructs on read: a page that asked
        # the question during a shut window and a page that asked after it would otherwise disagree.
        G["hist"].append(1 if gate_state()[0] else 0)
        while len(G["hist"]) > G["hist_n"]:
            G["hist"].pop(0)
        await asyncio.sleep(GATE_POLL_S)


def upstream(path):
    p = path.split("?", 1)[0].split("#", 1)[0]
    if p in EXACT:
        return EXACT[p]
    for pref, up in PREFIX:
        if p.startswith(pref):
            return up
    return ENGINE


def split_url(u):
    m = re.match(r"http://([^:/]+):(\d+)", u)
    if not m:
        raise ValueError(f"not an http URL: {u}")
    return m.group(1), int(m.group(2))


async def read_headers(reader, first=b""):
    """A header block: lines up to the first blank line (size-capped: this is a LAN box).

    Reading until the blob *ends* in "\r\n\r\n" was wrong and hung a 96 KB chat request through the gateway:
    a 100 Continue response is a status line plus ONE blank line, so the loop did not see its end and ate the
    real 200's status line out of the stream afterwards (2026-11-15, first big-body test through the gateway)."""
    data = first
    while True:
        line = await reader.readline()
        if line == b"":
            raise ValueError("peer closed inside a header block")
        data += line
        if line in (b"\r\n", b"\n"):
            if len(data) > 128 * 1024:
                raise ValueError("header block over 128 KiB")
            return data


async def copy_exact(reader, writer, n):
    while n:
        b = await reader.read(min(CHUNK, n))
        if not b:
            return False                      # peer ended early: the keep-alive loop must stop
        n -= len(b)
        writer.write(b)
        await writer.drain()
    return True


async def copy_chunked(reader, writer):
    """Forward the chunk framing verbatim -- including the 0-size terminator and any trailers."""
    while True:
        line = await reader.readline()
        if not line:
            return False
        writer.write(line)
        try:
            n = int(line.split(b";")[0], 16)
        except ValueError:
            return False
        if n == 0:
            while True:                       # trailers, then the blank line
                t = await reader.readline()
                writer.write(t)
                await writer.drain()
                if t in (b"\r\n", b"\n", b""):
                    return True
        else:
            if not await copy_exact(reader, writer, n):
                return False
            writer.write(await reader.readexactly(2))     # the CRLF after each chunk's data
            await writer.drain()


async def relay_body(uread, writer, headers):
    low = headers.lower()
    if b"transfer-encoding:" in low and b"chunked" in low:
        return await copy_chunked(uread, writer)
    m = re.search(rb"\r?\ncontent-length: *(\d+)", low)
    if m:
        return await copy_exact(uread, writer, int(m.group(1)))
    while True:                               # no framing: answer ends when the upstream closes
        b = await uread.read(CHUNK)
        if not b:
            return False
        writer.write(b)
        await writer.drain()


def _state_fetch(url):
    """One watcher's state.json. Blocking on purpose, and always called through a thread: this runs inside an
    async handler, and a blocking read here parks the whole event loop. Measured on 2026-11-15 with three
    concurrent /api/sidecars polls running (a Monitor tab polls every 2 s): streamed chat chunks arrived in
    bursts -- median gap between chunks 0.0 ms instead of ~13 -- while the engine itself was decoding at its
    normal rate. The hop was not slow; the page's own poll was holding the pipe shut."""
    with urllib.request.urlopen(url, timeout=4) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _state_pairs():
    out = []
    for i, e in enumerate(SIDECAR_STATE, 1):
        if "://" in e and "=" in e and e.index("=") < e.index("://"):
            name, url = e.split("=", 1)
        else:
            name, url = f"sidecar{i}", e
        out.append((name, url))
    return out


async def _fetch_states():
    """Fetch every watcher's state.json CONCURRENTLY and off the loop, and encode the blob once.

    The concurrency matters: each watcher answers in ~0.2 s, so serially this was the slowest thing the gateway
    answers (0.42 s for two). Off the loop matters more -- it used to call urllib straight from an async
    handler, which parked the whole front door for the duration."""
    pairs = _state_pairs()
    got = await asyncio.gather(*[asyncio.to_thread(_state_fetch, u) for _, u in pairs],
                               return_exceptions=True)
    out = []
    for (name, url), d in zip(pairs, got):
        if isinstance(d, Exception):                                # a dead sidecar is data, not a 500
            d = {"error": f"{type(d).__name__}: {d}", "health": None, "rates": {}, "host": {}, "serve": {}}
        d["name"], d["url"] = name, url
        out.append(d)
    return json.dumps({"sidecars": out, "at": time.time()}).encode()


SC = {"body": None, "t": 0.0, "task": None}
SC_TTL = float(os.environ.get("SIDECAR_TTL", "1.5"))


async def sidecars(writer):
    """GET /api/sidecars -- the embed-watch pages' state in one blob, so the Monitor has ONE page for the
    engine and the CPU serves. With no SIDECAR_STATE configured it answers [] and the card does not appear.

    An entry may be `name=url` (the Monitor card labels itself from `name`); a bare url gets sidecar1, 2... --
    the split is on the first = BEFORE the ://, so a url carrying a query string still works.

    Cached for SC_TTL seconds behind a SINGLE FLIGHT. The watchers refresh their own numbers every poll (2 s),
    so a front door that re-fans-out on every browser poll is amplifying its own traffic: each extra tab, and
    every retry of a poll the page aborted at 4 s, bought two more watcher fetches. On 2026-11-15 that was the
    multiplier behind a fork storm inside the watcher (its own per-request pgrep, fixed there too) that left 6
    of 12 polls unanswered. The shield keeps one client aborting from cancelling the fetch the other clients
    are waiting on."""
    now = time.time()
    if SC["body"] is not None and now - SC["t"] < SC_TTL:
        body = SC["body"]
    else:
        if SC["task"] is None or SC["task"].done():
            SC["task"] = asyncio.create_task(_fetch_states())
        try:
            body = await asyncio.shield(SC["task"])
            SC["t"], SC["body"] = time.time(), body
        except Exception:                                          # noqa: BLE001 -- stale beats absent
            body = SC["body"] or b'{"sidecars": [], "at": 0}'
    writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nCache-Control: no-store\r\n"
                 b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
    await writer.drain()


async def handle(reader, writer):
    while True:
        line = await reader.readline()
        if not line or line in (b"\r\n", b"\n"):
            return
        parts = line.split()
        if len(parts) != 3:
            writer.write(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")
            return
        _method, target, version = parts
        path = target.split(b"?")[0].decode("latin-1")
        try:
            headers = await read_headers(reader)
        except (asyncio.IncompleteReadError, ValueError):
            return
        keep = version != b"HTTP/1.0" and b"connection: close" not in headers.lower()
        if path == "/api/gate":
            await gate_page(writer)
            if not keep:
                return
            continue
        ok, held, why = await admit(path, declared_length(headers))
        if not ok:
            # 503 only exists with GATE_HOLD_S=0. A held request that the gate refused never reached the
            # engine, so the body says what the pressure was -- that is the whole difference between "the
            # front door is slow" and "the front door is protecting something".
            msg = json.dumps({"error": {"message": f"gateway refused {path!r} while the KV pool is at "
                                                   f"{why}; retry later", "type": "gateway_admission",
                                                   "gate": json.loads(gate_blob())}}).encode()
            writer.write(b"HTTP/1.1 503 Service Unavailable\r\nRetry-After: 2\r\nContent-Type: application/json"
                         b"\r\nContent-Length: " + str(len(msg)).encode() + b"\r\n\r\n" + msg)
            await writer.drain()
            if not keep:
                return
            continue
        if path == "/api/sidecars":
            # Answered here, and the CONNECTION stays open. The first version returned, ending the keep-alive
            # connection under a browser that still had polls to send down it: the next request on that socket
            # gets an EOF or a BrokenPipe, and Chrome reports it as "canceled" -- which is what "the Monitor
            # keeps dropping" looked like while every endpoint answered in well under 200 ms. A page that polls
            # three endpoints every 2 s reuses one connection; the front door may not close it.
            await sidecars(writer)
            if not keep:
                return
            continue
        try:
            host, port = split_url(upstream(path))
            uread, uwrite = await asyncio.open_connection(host, port)
        except (OSError, ValueError) as e:
            msg = json.dumps({"error": {"message": f"gateway cannot reach {path!r}: {e}"}}).encode()
            writer.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Type: application/json\r\nContent-Length: "
                         + str(len(msg)).encode() + b"\r\n\r\n" + msg)
            await writer.drain()
            return
        uwrite.write(line + headers)                          # body streamed, never buffered
        await uwrite.drain()
        # The request body and the response are copied CONCURRENTLY, and 1xx is forwarded without being treated
        # as the response: a client that sends `Expect: 100-continue` (curl does, over ~1 KiB) waits for the
        # Continue before it sends the body, so a sequential "body first, response second" stalls a second per
        # request -- and uvicorn's 100 line would otherwise be parsed as the response status.
        m = re.search(rb"\r?\ncontent-length: *(\d+)", headers.lower())

        async def send_body():
            if m:
                await copy_exact(reader, uwrite, int(m.group(1)))

        body_task = asyncio.create_task(send_body())
        try:
            while True:
                status = await uread.readline()
                if not status:
                    return
                rheaders = await read_headers(uread)
                writer.write(status + rheaders)
                await writer.drain()
                if re.match(rb"HTTP/1\.[01] 1\d\d", status):
                    continue                                   # 100 Continue and friends: not the response
                break
            if not await relay_body(uread, writer, rheaders):
                return
        except (asyncio.IncompleteReadError, ConnectionResetError, ConnectionRefusedError, ValueError, OSError):
            return
        finally:
            body_task.cancel()
            try:
                uwrite.close()
            except OSError:
                pass
        if not keep:
            return


async def client(reader, writer):
    """One client connection. A browser that aborts a poll -- the pages abort at 4 s, by design -- resets the
    socket mid-request, and that is ordinary HTTP, not an incident. Letting it escape into asyncio's default
    handler printed a traceback per abort and left the upstream connection to the GC (the first version did:
    "Unhandled exception in client_connected_cb", ConnectionResetError, twice in the log of a quiet hour)."""
    try:
        await handle(reader, writer)
    except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError, asyncio.CancelledError,
            OSError):
        pass
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except (OSError, ConnectionResetError):
            pass


async def main():
    srv = await asyncio.start_server(client, BIND, PORT)
    # flush=True: redirected to a log file, python block-buffers stdout, and the one line that names what this
    # front door is actually configured to do stayed invisible in the log while its tracebacks (stderr, line
    # buffered) filled the file. The gate's thresholds belong in the log for the same reason the launch line does.
    print(f"gateway: http://{BIND}:{PORT}/  engine={ENGINE} embed={EMBED} rerank={RERANK} pages={PAGES} "
          f"sidecars={len(SIDECAR_STATE)}", flush=True)
    if GATE_ON:
        G["task"] = asyncio.create_task(gate_poll())
        print(f"gate: {', '.join(GATED)} held at kv>={GATE_KV_HIGH} / waiting>={GATE_WAIT}"
              f"{' / inflight>=' + str(GATE_MAX_INFLIGHT) if GATE_MAX_INFLIGHT else ''}"
              f", re-open at kv<{GATE_KV_LOW}, ceiling {GATE_HOLD_S:.0f}s, /metrics every {GATE_POLL_S}s "
              f"(state: /api/gate)", flush=True)
    else:
        print("gate: disabled (GATE_DISABLE=1)", flush=True)
    async with srv:
        await srv.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
