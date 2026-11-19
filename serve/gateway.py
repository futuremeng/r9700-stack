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

What it does NOT forward: the engine's own control plane. --api-key in the pinned vLLM guards FOUR prefixes
(0.32's GUARDED_PREFIX = /v1 /v2 /inference /cohere), and the engine registers sleep / wake_up / pause / resume /
abort_requests / reset_{prefix,mm,encoder}_cache / release_kv_cache_memory / start_profile / stop_profile /
update_weights / weight_* / collective_rpc OUTSIDE them -- enumerated from the installed wheel, not from the
docs. A door whose rule is "everything else -> ENGINE" therefore hands anything that can reach the door the
switch that puts the model to sleep and the switch that wipes the KV cache, with no key in the way. Those paths
answer 404 here, with a body that says so and names the knob; GATEWAY_ADMIN=1 forwards them, and then to a
loopback peer only -- the operator's way in is an ssh tunnel, not an open route.

Who may ask for anything (GATEWAY_KEY): the door groups the paths it forwards, because two groups have to stay
open for the box to work at all.

    OPEN     /health (setup.py's watcher and every client already poll it), /metrics (this file's own gate
             poller reads it raw, see read_gauges), and the Monitor's page plus its data endpoints -- the page
             whose whole job is to survive a wedged engine is the wrong place to put a credential question
    KEYED    everything that reaches a model: /v1*, /v2*, /inference, /cohere, the rerank routes, /tokenize and
             /classify (both OUTSIDE vLLM's four guarded prefixes, so the door is the only thing deciding), and
             /api/notify, which writes the webhook that pages a phone
    DENIED   the engine control plane above, whatever the key

Keyed by DEFAULT, open by exception: a new engine route is protected by being unknown. The key lives here rather
than in the engine's --api-key for three reasons measured on this box: the engine, both CPU sidecars and the page
already bind loopback (from the LAN address, 8180/8181/8090/8092 are connection refused), bench/ and tools/ aim
at THIS port (ten of them say localhost:8080), and an engine --api-key is a command-line argument -- readable by
every local user through ps, where a key read from a mode-0600 file is not.

It ships OBSERVE, exactly like the admission gate: a key that silently 401s the LiteLLM clients in front of this
box is an outage rather than a security improvement, so the door first counts and logs what it WOULD have
refused, and that count is what decides when to set GATEWAY_KEY_ENFORCE=1.

The key asks REMOTE peers. A request that arrived on loopback is exempt unless GATEWAY_KEY_LOCAL=1, for the same
reason the control-plane rule above is loopback-shaped: every bench, tool and the Monitor sidecar already reach
the engine directly at 127.0.0.1:8180 with no credential, so keying a local peer at the door would add ceremony
and no boundary -- it would only decide which loopback door a local process presents a credential to. The
boundary that matters is the one the LiteLLM hosts cross, and that is what this asks.

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

## and then it was turned off: GATE_POLICY=observe is the shipped mode (2026-11-16)

A second afternoon said v2 was wrong the same way v1 had been, and this time the numbers name the mechanism. The
box was running a 100k-token-prompt batch (median prompt 43k, 94.8% of its prompt tokens served from the prefix
cache, wall clock 7.6-37 s) when four requests arrived with **no `max_tokens`** -- prompt 515-532, and vLLM's
default commitment is the rest of the window, so each asked for 261,612-261,629 tokens and each ran 3,196-4,214 s
(finish_reason=length). Four full-context rows are 1,046,516 tokens against a **902,153**-token pool: the engine's
own boot line says `Maximum concurrency for 262,144 tokens per request: 3.44x`. The pool pegged at 92-100%, the
engine queue went to 66-72 (all reason="capacity"), aggregate decode fell to 8.9-51.7 tok/s, TTFT p99 landed in
the 640-2560 s bucket, and num_preemptions_total ran 165 -> 296 in ten minutes.

Three things follow, all of them measured:

  * **the door cannot re-open.** Two axes, both non-terminating. The preemption window re-arms on every event, and
    at 12.9 events/min a 20 s window buys 258 s of shut per 60 s of wall clock -- the door has no way to be open.
    And the pool axis latches below the workload's own steady state: GATE_KV_LOW=0.85 against an hour whose kv p50
    was 0.945, 69% of samples >= 0.92. §6's claim that a window "always ends" is true of one event, not of a
    stream of them.
  * **holding the arrival does not touch the harm.** The blocks are held by requests that are already running, and
    they are being taken back as those rows GROW. Delaying a new arrival by 5 s cannot free one block. Measured on
    the live box with the door shut: a 20 KB-body request took 5.019 s through the gate and 0.002 s direct to the
    engine; 25 of 28 holds ended by hitting the ceiling and 3 by the door opening. A flat delay, not a bound.
  * **the class it taxes and the class that harms are different classes.** The four full-context rows declared
    ~1.2 KB bodies, so the small-request escape waved all four through. The per-request preemption counts in the
    engine's journal say who paid: 19 requests / 79 preemptions, every one of them in the prompt<=1k class (one
    victim 40 times); across the 973 long-prompt requests the maximum was **zero**. The door was closing on harm
    made by traffic it waved in and charging the traffic it could see.

So the shipped policy is `GATE_POLICY=observe`: the poller reads the engine, `gate_state()` computes harm exactly
as before, the ring, `/api/gate` and the Monitor row stay -- and **no request is held, refused or delayed**. What
is counted instead of done: `gated_seen`, `declared_small`/`declared_big`, and `arrived_shut` with the same split,
which is the set a holding policy would have delayed. `shut` states the door in one number -- the fraction of the
ring it was shut, the unbroken span it is inside now, the longest one in the ring -- because "was it effectively
permanently shut?" was the question of the afternoon and 900 dots are not an answer to it.

`GATE_POLICY=hold` re-arms the intervention (every v1/v2 threshold is still a knob and still tested); an unknown
value observes and says so in `policy_err`, because the fail-open direction of this file has always been the
request path. `GATE_DISABLE=1` is the other switch and stays different: it removes the gate, the reading and the
row, while observe keeps all three and touches nothing.

Fail-open is the default on purpose: a gauge that cannot be read, or a vLLM that renamed it, opens the gate
and says so in `err`. A front door that stops serving because its telemetry broke is worse than no gate.

Knobs: GATE_DISABLE=1 (the gate does not exist: no reading, no /api/gate, no Monitor row), GATE_POLICY
(`observe` shipped -- the harm signals are read, counted and drawn, and no request is touched; `hold` arms the
intervention described above), GATE_KV_HIGH (0.92 -- close), GATE_KV_LOW (0.85 -- re-open the pool axis: one
8 192-token prefill moves this pool ~0.9%, so a single threshold would chatter), GATE_PREEMPT_S (20 -- stay shut
this long after the last observed preemption), GATE_WAIT (0 shipped: an engine queue alone never closes the
door; set it to make the queue a trigger), GATE_SMALL_BYTES (8192 -- a declared body this size or smaller passes
a shut door; 0 disables; it is also the size CLASS the observation reports, which on 2026-11-16 turned out to be
a prompt size and not a pool cost), GATE_MAX_HELD (8 -- never hold more than this many at once), GATE_MAX_INFLIGHT
(0 = off; counts the queue at the door, NOT the engine's in-flight count -- see the note at its definition),
GATE_HOLD_S (5 s ceiling;
0 = refuse instead of hold), GATE_METRICS_S (1.0), GATE_STALE_S (5). Only /v1/chat/completions and
/v1/completions are gated -- /health is what setup.py's
watcher polls, /metrics is what the gate itself reads, /api/* is the page, and the two sidecar routes are CPU
serves that cannot take a KV hit.
"""
import asyncio
import json
import os
import re
import hashlib
from collections import OrderedDict, deque
import secrets
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

# The engine's control plane, spelled as the installed wheel spells it. These change engine state or run code
# on it, and vLLM's --api-key does NOT guard any of them (prefix list above), so the door is the one that
# decides.
ADMIN = frozenset("""/sleep /wake_up /pause /resume /abort_requests /reset_prefix_cache /reset_mm_cache
    /reset_encoder_cache /release_kv_cache_memory /start_profile /stop_profile /update_weights
    /start_weight_update /start_draft_weight_update /finish_weight_update /update_weight_version
    /init_weight_transfer_engine /shutdown_weight_transfer_engine /weight_checker /weight_info
    /collective_rpc /get_world_size /is_scaling_elastic_ep""".split())
ADMIN_ON = os.environ.get("GATEWAY_ADMIN", "") == "1"


def is_loopback(peer):
    """The loopback NET, not one address. 127.0.0.1 is the peer a local client usually presents, but this host
    also answers on 127.0.1.1 (its own name, on lo) and a peer address of ::ffff:127.x is the same thing mapped
    into v6 -- all of them came from the box, which is the boundary both rules below are drawn on. Naming one
    address would make the rule about a spelling."""
    return peer.startswith("127.") or peer.startswith("::ffff:127.") or peer in ("::1", "")


def admin_denied(path, peer):
    """True when this path is engine control plane and this door is not allowed to answer it for this peer."""
    return path in ADMIN and not (ADMIN_ON and is_loopback(peer))

# ---- who may ask for anything ---------------------------------------------------------------------------
# Keyed by default, open by exception: an engine route nobody has looked at is protected, not exposed.
OPEN_EXACT = frozenset("""/health /metrics /ready /readyz /ping /version /load /monitor /about /favicon.svg
    /api/status /api/requests /api/host /api/gate /api/sidecars""".split())
OPEN_PREFIX = ("/web/",)                        # the page's own static assets
KEYED_EXACT = frozenset(("/api/notify",))       # writes the webhook that pages a phone
KEYED_EXACT = KEYED_EXACT | {"/api/capture"}   # reads back captured prompt and answer text


def auth_group(path):
    """None for a path the door answers keyless, "keyed" for one that needs the door's key."""
    if path in KEYED_EXACT:
        return "keyed"
    if path in OPEN_EXACT or path.startswith(OPEN_PREFIX):
        return None
    return "keyed"


def _read_key():
    """The door's key, and where it came from. GATEWAY_KEY_FILE (mode 0600) is the way it should be set: a key
    passed as GATEWAY_KEY= lands in this process's own argv, readable by every local user through ps -- which is
    the exact reason an engine --api-key is the weaker answer on this box."""
    f = os.environ.get("GATEWAY_KEY_FILE", "")
    if f:
        try:
            with open(f) as fh:
                return fh.read().strip(), f"file:{f}"
        except OSError as e:
            return "", f"file:{f} unreadable ({e}) -- not enforcing"
    if os.environ.get("GATEWAY_KEY", ""):
        return os.environ["GATEWAY_KEY"], "GATEWAY_KEY (in argv: readable by ps; use GATEWAY_KEY_FILE)"
    return "", "unset"


KEY, KEY_SRC = _read_key()
KEY_ENFORCE = os.environ.get("GATEWAY_KEY_ENFORCE", "") == "1"
KEY_LOCAL = os.environ.get("GATEWAY_KEY_LOCAL", "") == "1"
KEY_HASH = hashlib.sha256(KEY.encode()).digest() if KEY else None


def auth_ok(headers):
    """Bearer check against the door's own key, constant-time, on the headers already in hand: the door is a
    byte pipe, and answering this must not turn it into a frame parser."""
    if not KEY_HASH:
        return False
    m = re.search(rb"(?im)^authorization: *(.+?)\r?\n", headers)
    if not m:
        return False
    scheme, _, param = m.group(1).decode("latin-1").strip().partition(" ")
    return scheme.lower() == "bearer" and secrets.compare_digest(
        hashlib.sha256(param.encode()).digest(), KEY_HASH)

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
# observe = measure harm and never touch a request; hold = the intervention. See "and then it was turned off"
# above: on 2026-11-16 the door could not re-open, a hold could not relieve the harm it keyed on, and 25 of 28
# holds were pure delay. An unknown value observes and records the mistake rather than acting on it.
GATE_POLICY_REQ = os.environ.get("GATE_POLICY", "observe").strip().lower()
GATE_POLICY = GATE_POLICY_REQ if GATE_POLICY_REQ in ("observe", "hold") else "observe"
GATE_ON = os.environ.get("GATE_DISABLE", "0") != "1"
GATE_KV_HIGH = float(os.environ.get("GATE_KV_HIGH", "0.92"))     # near-full backstop; preempt usually starts here
GATE_KV_LOW = float(os.environ.get("GATE_KV_LOW", "0.85"))       # re-open line (hysteresis)
GATE_PREEMPT_S = float(os.environ.get("GATE_PREEMPT_S", "20"))   # close this long after the last preemption
GATE_WAIT = int(os.environ.get("GATE_WAIT", "0"))                # 0 = waiting alone never closes (v1's 1 did)
# NOT the engine's in-flight count: G["inflight"] is incremented inside admit() and released the moment the
# request is forwarded, so it is the DEPTH OF THE QUEUE AT THE DOOR. The sequence axis the engine itself enforces
# is --max-num-active-seqs, not this.
GATE_MAX_INFLIGHT = int(os.environ.get("GATE_MAX_INFLIGHT", "0"))
GATE_SMALL_BYTES = int(os.environ.get("GATE_SMALL_BYTES", "8192"))   # a declared body at most this big passes
GATE_MAX_HELD = int(os.environ.get("GATE_MAX_HELD", "8"))        # never a second queue: beyond this, admit
GATE_HOLD_S = float(os.environ.get("GATE_HOLD_S", "5"))          # v1's 25 s was the usual exit, not the exception
GATE_POLL_S = float(os.environ.get("GATE_METRICS_S", "1.0"))
GATE_STALE_S = float(os.environ.get("GATE_STALE_S", "5"))
# How many polls of shut/open history the gate keeps. It used to be 180 (~3 min), which is enough to watch a pool
# climb into a cliff and not enough to answer the question that actually gets asked afterwards: "was the door shut
# at 15:05?" An afternoon of traffic is the evidence for whether this policy is a tax or a relief, so the ring is
# ~15 min at the default poll rate, and the payload says what span it covers rather than making the reader multiply.
GATE_HIST = int(os.environ.get("GATE_HIST", "900"))
GATED = ("/v1/chat/completions", "/v1/completions")
G = {"kv": None, "run": None, "wait": None, "at": 0.0, "err": None, "closed": False, "why": "",
     # preemption watch: the counter's last value, when it last moved, and until when the door stays shut.
     "prem": None, "prem_seen": 0, "thrash_until": 0.0, "pool_shut": False,
     # display only: turns the usage FRACTION into tokens in the reason string and in /api/gate, so a reader
     # sees "770,000 of 902,153 tokens" rather than 0.854. The decision itself never reads it -- the fraction
     # is what the engine measures, and this box's pool is 902,153 tokens (6.65 GiB, from its own boot log).
     "pool_tokens": int(os.environ.get("GATE_POOL_TOKENS", "902153")),
     "admin_denied": 0,      # control-plane paths this door refused to hand to the engine
     "auth_open": 0, "auth_keyed": 0, "auth_ok": 0, "auth_401": 0, "auth_would401": 0, "auth_local": 0,
     "capture_wanted": 0, "capture_stored": 0, "capture_orphan": 0, "capture_hits": 0, "capture_misses": 0,
     "inflight": 0, "held": 0, "held_now": 0, "released": 0, "expired": 0, "refused": 0, "admitted": 0,
     "small_passed": 0, "held_over": 0, "polls": 0,
     # Counted because observing is now the job: who came, of which declared-size class, and how many arrived while
     # the harm signal was on -- i.e. exactly the set a holding policy would have delayed. "What would this policy
     # have cost?" stays answerable without arming it on a live box to find out.
     "gated_seen": 0, "declared_small": 0, "declared_big": 0,
     "arrived_shut": 0, "arrived_shut_small": 0, "arrived_shut_big": 0,
     # One entry per poll: 1 shut, 0 open. A poll whose gauges were unusable draws 0 on purpose: fail-open is
     # shown, not hidden, because a strip that goes blank when the reading dies would blame the traffic for a
     # telemetry fault. The ring is GATE_HIST polls (~15 min at the default 1 s), and the payload carries the span
     # so a reader does not have to multiply two numbers to know how far back the strip reaches.
     "hist": [], "hist_n": GATE_HIST,
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

    Under GATE_POLICY=observe this function is a REPORT, not a decision: it still says what the harm signals say
    and still drives the ring, and nothing acts on it. Two caveats it now carries, both from 2026-11-16 and both
    relevant if the intervention is ever re-armed: neither way this returns "shut" is guaranteed to end (a
    preemption event every GATE_PREEMPT_S re-arms the window forever, and a pool whose steady state sits above
    GATE_KV_LOW latches the pool axis shut), and when the harm is resident rows GROWING, no delay applied to an
    arrival can relieve it. The fix for either is not a better threshold -- it is a limit on what a request may
    COMMIT (max_tokens) and on the queue the engine will hold, which is token arithmetic the engine can do and a
    byte pipe cannot.
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


def shut_stats():
    """The DOOR as one number rather than 900 dots: the fraction of the ring it was shut, the unbroken span it is
    inside now, and the longest unbroken span inside the ring. On 2026-11-16 the answer was the finding (0.71 of
    the ring shut, a 339 s run and still running) and the strip could not say it: a preemption rate of 12.9/min
    against a 20 s window leaves no arithmetic under which the door is open."""
    h = G["hist"]
    if not h:
        return {"frac": None, "run_s": None, "max_s": None}
    run = 0
    for x in reversed(h):
        if not x:
            break
        run += 1
    best = cur = 0
    for x in h:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return {"frac": round(sum(h) / len(h), 3), "run_s": round(run * GATE_POLL_S), "max_s": round(best * GATE_POLL_S)}


def declared_length(headers):
    """How big the request DECLARES its body is, from the header bytes already in hand. None if chunked/absent.

    Header only, on purpose. The pipe still never reads a body to make a decision -- but it already holds this
    line to frame the upstream request, and a declared size is enough to tell a 200-token question from a
    100k-token document, which is the difference the door was starving.
    """
    m = re.search(rb"\r?\ncontent-length: *(\d+)", headers.lower())
    return int(m.group(1)) if m else None


async def admit(path, declared=None):
    """Ask the gate about a new engine request. (ok, held_s, why).

    AS SHIPPED (GATE_POLICY=observe) this function decides nothing: it counts the arrival -- size class, and
    whether the harm signal was on when it arrived -- and returns immediately. No hold, no refusal, no delay,
    no upstream connection opened early. The counting is the point: it is what lets "was the door shut during my
    batch, and how much would holding it have cost?" be answered from a file instead of from a feeling, which is
    the thing that could not be answered on 2026-11-16 and cost a day of believing the front door was the fix.

    Under GATE_POLICY=hold it holds, and holding means: no upstream connection, no body read. Three escapes, all
    of them measured needs:
      * a declared-small body passes a shut door (v1 starved exactly these); GATE_SMALL_BYTES=0 turns that off;
      * at most GATE_MAX_HELD requests are ever held -- past that, admit, because a gate with an unbounded hold
        queue has just become a queue with extra steps and the engine already has one;
      * the ceiling GATE_HOLD_S is the client-visible bound. It is 5 s, not v1's 25, because the counter said the
        ceiling (not the re-open) was how most held requests left: 14 held, 9 expired. A bound that is usually
        hit is not a bound, it is a delay -- and 2026-11-16's numbers made it 25 expired of 28 holds, which is
        why this path is no longer the shipped one.
    """
    if path not in GATED:
        return True, 0.0, "not gated"
    small = GATE_SMALL_BYTES > 0 and declared is not None and declared <= GATE_SMALL_BYTES
    # Counted in BOTH modes: in observe mode this is the whole job -- who came, of which declared-size class, and
    # how many of them arrived while the harm signal was on.
    G["gated_seen"] += 1
    G["declared_small" if small else "declared_big"] += 1
    closed_now, why_now = gate_state()
    if closed_now:
        G["arrived_shut"] += 1
        G["arrived_shut_small" if small else "arrived_shut_big"] += 1
    if GATE_POLICY != "hold":
        G["admitted"] += 1
        return True, 0.0, why_now
    # Two facts about the hold path, both measured on 2026-11-16 and both worth reading before re-arming it. A
    # client that gives up DURING a hold is not noticed -- nothing is read from the socket while the door is shut,
    # so the request is forwarded anyway and a retrying client buys the engine a new request per give-up. And a
    # declared body is a PROMPT size, not a pool cost: the four rows that filled the pool that day declared ~1.2 KB
    # each and passed this exact line, while the 100k-prompt rows that cost nothing were the ones held.
    if small:
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
    return json.dumps({"closed": closed, "why": why, "policy": GATE_POLICY,
                       "policy_err": (None if GATE_POLICY == GATE_POLICY_REQ
                                      else f"GATE_POLICY={GATE_POLICY_REQ!r} is not a policy: observing"),
                       "gauges": {k: G[k] for k in ("kv", "run", "wait", "prem",
                                                                                   "at")},
                       "err": G["err"], "shut": shut_stats(),
                       "inflight": G["inflight"], "polls": G["polls"],
                       "thrash_left": max(0.0, G["thrash_until"] - time.time()),
                       "prem_seen": G["prem_seen"], "pool_shut": G["pool_shut"],
                       "hist": G["hist"], "hist_n": G["hist_n"],
                       "hist_span_s": int(round(G["hist_n"] * GATE_POLL_S)),
                       "gated": list(GATED),
                       "capture": {"GATEWAY_CAPTURE": CAPTURE_CAP, "keep": CAPTURE_KEEP,
                                   "stored": len(CAPTURES), "orphans": len(CAPTURE_ORPHAN),
                                   "counts": {k: G[k] for k in ("capture_wanted", "capture_stored",
                                                                "capture_orphan", "capture_hits",
                                                                "capture_misses")}},
                       "admin": {"paths": len(ADMIN), "GATEWAY_ADMIN": ADMIN_ON,
                                 "denied": G["admin_denied"]},
                       "auth": {"key": {"set": bool(KEY), "source": KEY_SRC, "chars": len(KEY)},
                                "enforce": KEY_ENFORCE, "open_exact": len(OPEN_EXACT),
                                "keyed_by_default": True,
                                "loopback_exempt": not KEY_LOCAL,
                                "counts": {k: G[k] for k in ("auth_open", "auth_keyed", "auth_ok", "auth_401",
                                                             "auth_would401", "auth_local")},
                                "unkeyed_paths": dict(sorted(G.get("auth_401_paths", {}).items(),
                                                             key=lambda kv: -kv[1])[:8])},
                       "counters": {k: G[k] for k in ("held", "held_now", "released", "expired", "refused",
                                                      "admitted", "small_passed", "held_over", "gated_seen",
                                                      "declared_small", "declared_big", "arrived_shut",
                                                      "arrived_shut_small", "arrived_shut_big")},
                       "config": {"GATE_ON": GATE_ON, "GATE_POLICY": GATE_POLICY,
                                  "GATE_KV_HIGH": GATE_KV_HIGH, "GATE_KV_LOW": GATE_KV_LOW,
                                  "GATE_PREEMPT_S": GATE_PREEMPT_S, "GATE_WAIT": GATE_WAIT,
                                  "GATE_MAX_INFLIGHT": GATE_MAX_INFLIGHT, "GATE_SMALL_BYTES": GATE_SMALL_BYTES,
                                  "GATE_MAX_HELD": GATE_MAX_HELD,
                                  "GATE_HOLD_S": GATE_HOLD_S, "GATE_METRICS_S": GATE_POLL_S,
                                  "GATE_STALE_S": GATE_STALE_S, "GATE_HIST": GATE_HIST,
                                  "pool_tokens": G["pool_tokens"]},
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


async def copy_exact(reader, writer, n, tap=None):
    while n:
        b = await reader.read(min(CHUNK, n))
        if not b:
            return False                      # peer ended early: the keep-alive loop must stop
        n -= len(b)
        writer.write(b)
        if tap:
            tap(b)                            # bytes already in hand; the pipe does not wait on this
        await writer.drain()
    return True


async def copy_chunked(reader, writer, tap=None):
    """Forward the chunk framing verbatim -- including the 0-size terminator and any trailers. The tap sees the
    framing bytes too: it is a byte tap, not a parser, and reassembling chunked bodies here would be the first
    step back toward the proxy that costs 2-3 ms."""
    while True:
        line = await reader.readline()
        if tap:
            tap(line)
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


async def relay_body(uread, writer, headers, tap=None):
    low = headers.lower()
    if b"transfer-encoding:" in low and b"chunked" in low:
        return await copy_chunked(uread, writer, tap)
    m = re.search(rb"\r?\ncontent-length: *(\d+)", low)
    if m:
        return await copy_exact(uread, writer, int(m.group(1)), tap)
    while True:                               # no framing: answer ends when the upstream closes
        b = await uread.read(CHUNK)
        if not b:
            return False
        writer.write(b)
        if tap:
            tap(b)
        await writer.drain()


# ---- an opt-in byte tap ------------------------------------------------------------------------------
# A Monitor row with timings and no text is the honest shape of a pipe that never reads bodies -- but "the
# request body as it arrived" as a heading over an empty pane is not honest, it is a label for a thing that was
# never kept. The only process on this topology that ever holds both the request and the answer is THIS one, so
# the capture lives here, and it is opted into per request by a header (X-R9K-Capture: 1). That keeps the cost
# where it belongs: an ordinary request pays one regex over header bytes already in hand, and only the request
# that asked for its own copy gets buffered. GATEWAY_CAPTURE=0 (the shipped default) turns the feature off
# entirely and the header is then ignored.
CAPTURE_CAP = int(os.environ.get("GATEWAY_CAPTURE", "0") or 0)
CAPTURE_KEEP = max(1, int(os.environ.get("GATEWAY_CAPTURE_KEEP", "50")))
CAPTURE_HARD = 1 << 20                      # per side, whatever the knob says: this process must stay bounded
if CAPTURE_CAP > CAPTURE_HARD:
    CAPTURE_CAP = CAPTURE_HARD
CAPTURE_HEAD = max(int(os.environ.get("GATEWAY_CAPTURE", "0") or 0), 2048)   # index head, see Tap
CAPTURES = OrderedDict()                    # response id -> row, oldest dropped first
CAPTURE_ORPHAN = deque()                    # captures whose response carried no id we could read


class Tap:
    """One opted-in request's bytes. Two separate counters: how many bytes went past (n_in/n_out) and how many
    were kept -- so the row can say "truncated at N of M" instead of quietly showing a prefix."""

    def __init__(self, path):
        self.at, self.path, self.state = time.time(), path, "incomplete"
        self.inp, self.out, self.head_in, self.head_out = [], [], [], []
        self.n_in = self.n_out = 0
        self.stored = False

    def take(self, side):
        def put(b):
            # Two buffers per side. The display copy is what GATEWAY_CAPTURE says it is. The HEAD exists because a
            # capture with no readable id cannot be found again -- the id is in the ANSWER, past the framing, so a
            # 32-byte cap would store rows nothing could ever ask for. The head is bounded too, and by the operator's
            # cap whenever that is the bigger number: it is an index, not a second copy.
            if side == "in":
                self.n_in += len(b)
                for buf, room_n in ((self.inp, CAPTURE_CAP), (self.head_in, CAPTURE_HEAD)):
                    room = room_n - sum(map(len, buf))
                    if room > 0:
                        buf.append(b[:room])
            else:
                self.n_out += len(b)
                for buf, room_n in ((self.out, CAPTURE_CAP), (self.head_out, CAPTURE_HEAD)):
                    room = room_n - sum(map(len, buf))
                    if room > 0:
                        buf.append(b[:room])
        return put


def capture_wanted(headers):
    """The header is the opt-in. Absent, or 0 / false / off, means do not buffer anything for this request."""
    m = re.search(rb"(?im)^x-r9k-capture: *([^\r\n]*)", headers)
    if not m:
        return False
    return m.group(1).strip().lower() not in (b"", b"0", b"false", b"off", b"no")


def store_capture(tap):
    """Index the capture by the id the engine put in its own answer, which is the only key the Monitor rows are
    addressed by. Reading that id is the one place this file looks inside a body, and it looks only in a tap that
    a client asked for, bounded at 8 KiB -- not in the stream on the way through."""
    data_in, data_out = b"".join(tap.inp), b"".join(tap.out)
    head_in, head_out = b"".join(tap.head_in), b"".join(tap.head_out)
    rid = None
    m = re.search(rb'"id"\s*:\s*"(chatcmpl-[0-9a-zA-Z-]{4,80})"', head_out)
    if m:
        rid = m.group(1).decode("latin-1")
    model = None
    m2 = re.search(rb'"model"\s*:\s*"([^"]{1,120})"', head_in)
    if m2:
        model = m2.group(1).decode("latin-1")
    row = {"at": round(tap.at, 3), "path": tap.path, "model": model, "state": tap.state,
           "stream": b'"stream"' in head_in and b"true" in head_in,
           "bytes_in": tap.n_in, "bytes_out": tap.n_out,
           "input_truncated": tap.n_in > len(data_in), "response_truncated": tap.n_out > len(data_out),
           "input": data_in.decode("utf-8", "replace"), "response": data_out.decode("utf-8", "replace")}
    if rid:
        CAPTURES[rid] = row
        while len(CAPTURES) > CAPTURE_KEEP:
            CAPTURES.popitem(last=False)
    else:
        CAPTURE_ORPHAN.appendleft(row)
        while len(CAPTURE_ORPHAN) > CAPTURE_KEEP:
            CAPTURE_ORPHAN.pop()
    tap.stored = True
    G["capture_stored"] += 1
    if not rid:
        G["capture_orphan"] += 1
    return rid


def capture_page(writer, q):
    """GET /api/capture -- the index, or one row by id. Keyed, never open: these rows contain prompt text, and a
    path nobody has looked at is protected by the rule above, not by luck. Loopback reads it keyless, which is how
    the Monitor's own process merges it into the row it is showing."""
    rid = None
    for kv in q.split(b"&"):
        k, _, v = kv.partition(b"=")
        if k == b"id":
            rid = v.decode("latin-1")
    if rid:
        row = CAPTURES.get(rid)
        G["capture_hits" if row else "capture_misses"] += 1
        body = json.dumps({"id": rid, "found": bool(row), "row": row})
    else:
        body = json.dumps({"cap": CAPTURE_CAP, "keep": CAPTURE_KEEP, "stored": len(CAPTURES),
                           "orphans": len(CAPTURE_ORPHAN),
                           "counts": {k: G[k] for k in ("capture_wanted", "capture_stored", "capture_orphan",
                                                        "capture_hits", "capture_misses")},
                           "ids": [{"id": k, "at": v["at"], "path": v["path"], "model": v["model"],
                                    "bytes_in": v["bytes_in"], "bytes_out": v["bytes_out"],
                                    "state": v["state"]}
                                   for k, v in reversed(list(CAPTURES.items()))][:CAPTURE_KEEP]})
    msg = body.encode()
    writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                 + str(len(msg)).encode() + b"\r\nConnection: close\r\n\r\n" + msg)
    return True


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
    # Read once, at connection time: no header can change which socket this came in on, so "loopback" is a fact
    # about the connection and not something a request gets to assert.
    peer = (writer.get_extra_info("peername") or ("", 0))[0]
    while True:
        line = await reader.readline()
        if not line or line in (b"\r\n", b"\n"):
            return
        parts = line.split()
        if len(parts) != 3:
            writer.write(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")
            return
        _method, target, version = parts
        # The query is kept because /api/capture?id= is addressed by it; the routing decision still uses the
        # path alone, exactly as before.
        tq = target.split(b"?")
        path = tq[0].decode("latin-1")
        query = tq[1] if len(tq) > 1 else b""
        try:
            headers = await read_headers(reader)
        except (asyncio.IncompleteReadError, ValueError):
            return
        keep = version != b"HTTP/1.0" and b"connection: close" not in headers.lower()
        if admin_denied(path, peer):
            # 404 rather than 401: to anything on the network that route does not exist, and the body still says
            # what it would have been and which knob answers it -- a door that refuses without naming a reason is
            # indistinguishable from a door that is down. Counted apart from the admission gate on purpose: the
            # gate's holds are a KV-pressure story, and an auth-shaped refusal mixed into them would make the
            # Monitor row lie about both.
            G["admin_denied"] += 1
            msg = json.dumps({"error": {"message": f"gateway does not forward {path!r} to the engine",
                                        "type": "gateway_admin",
                                        "hint": "engine control plane is not exposed through the door; "
                                                "GATEWAY_ADMIN=1 forwards it to a loopback peer only",
                                        "admin_denied": G["admin_denied"]}}).encode()
            writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nContent-Length: "
                         + str(len(msg)).encode() + b"\r\n\r\n" + msg)
            await writer.drain()
            if not keep:
                return
            continue
        grp = auth_group(path)
        # A loopback peer is exempt unless the box says otherwise -- see the docstring for why that is the boundary
        # and not a hole: a local process that wanted the model unkeyed already has 127.0.0.1:8180.
        if grp and is_loopback(peer) and not KEY_LOCAL:
            G["auth_local"] += 1
            grp = None
        if grp:
            allowed = auth_ok(headers)
            G["auth_keyed"] += 1
            if allowed:
                G["auth_ok"] += 1
            else:
                G["auth_401" if KEY_ENFORCE else "auth_would401"] += 1
                seen = G.setdefault("auth_401_paths", {})
                seen[path] = seen.get(path, 0) + 1
            if not allowed and KEY_ENFORCE:
                msg = json.dumps({"error": {"message": f"gateway requires a key for {path!r}",
                                            "type": "gateway_auth",
                                            "hint": "Authorization: Bearer <the door's key>; /health, /metrics "
                                                    "and the Monitor stay open",
                                            "unauthorized": G["auth_401"]}}).encode()
                writer.write(b'HTTP/1.1 401 Unauthorized\r\nWWW-Authenticate: Bearer realm="r9700-stack"\r\n'
                             b"Content-Type: application/json\r\nContent-Length: "
                             + str(len(msg)).encode() + b"\r\n\r\n" + msg)
                await writer.drain()
                if not keep:
                    return
                continue
        else:
            G["auth_open"] += 1
        if path == "/api/gate":
            await gate_page(writer)
            if not keep:
                return
            continue
        if path == "/api/capture":
            # NOT awaited: like admin_page, this is a synchronous write to the transport that returns a bool,
            # and an `await` on it raised "object bool can't be used in 'await' expression" the first time the
            # suite actually asked for the page.
            return capture_page(writer, query)      # always closes: a read of a buffer, not a proxied request
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
        tap = None
        if CAPTURE_CAP and capture_wanted(headers):
            tap = Tap(path)
            G["capture_wanted"] += 1
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
                await copy_exact(reader, uwrite, int(m.group(1)), tap.take("in") if tap else None)
            elif tap:
                # a chunked request body: the tap sees the framing, because it is a byte tap
                await copy_chunked(reader, uwrite, tap.take("in"))

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
            if not await relay_body(uread, writer, rheaders, tap.take("out") if tap else None):
                return
            if tap:
                tap.state = "completed"
                store_capture(tap)
        except (asyncio.IncompleteReadError, ConnectionResetError, ConnectionRefusedError, ValueError, OSError):
            return
        finally:
            # A capture is kept even when the connection died on the way: an aborted or reset request is exactly
            # the row someone opens the detail pane for, and "incomplete" is a fact worth keeping rather than a
            # reason to throw the bytes away.
            if tap and not tap.stored:
                store_capture(tap)
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
          f"sidecars={len(SIDECAR_STATE)}  control-plane={'loopback only' if ADMIN_ON else 'not forwarded'} "
          f"({len(ADMIN)} paths)", flush=True)
    kmode = "loopback keyed too" if KEY_LOCAL else "loopback exempt: bench/, tools/ and the sidecar ride it"
    print(f"auth: keyed by default, open={len(OPEN_EXACT)} exact + {list(OPEN_PREFIX)} prefix; "
          f"asked of remote peers only ({kmode}); "
          f"key={'set from ' + KEY_SRC if KEY else 'NOT SET'}; "
          + ("ENFORCING (401)" if KEY_ENFORCE and KEY else
             ("GATEWAY_KEY_ENFORCE=1 with no key: NOT enforcing" if KEY_ENFORCE else
              "observing -- unkeyed model requests are counted at /api/gate, nothing refused")),
          flush=True)
    print("capture: " + (f"off -- nothing on this box buffers a request body, so a Monitor row shows timings "
                         f"and no text, and says which" if not CAPTURE_CAP else
                         f"OPT-IN (header X-R9K-Capture: 1), {CAPTURE_CAP} bytes per side, the last "
                         f"{CAPTURE_KEEP} rows in RAM, readable at /api/capture (keyed, loopback exempt); "
                         f"a request that does not ask pays one header regex"), flush=True)
    if GATE_ON and GATE_POLICY != "hold":
        G["task"] = asyncio.create_task(gate_poll())
        # The MODE goes in the log next to the thresholds: someone comparing two gateway logs has to be able to tell
        # "this door was measuring" from "this door was deciding" without asking anybody.
        print(f"gate: OBSERVE ONLY (GATE_POLICY=observe) -- watches {', '.join(GATED)} for harm "
              f"(preemption window {GATE_PREEMPT_S:.0f}s / kv>={GATE_KV_HIGH}, re-open kv<{GATE_KV_LOW}, "
              f"waiting>={GATE_WAIT}) and holds, refuses or delays NOTHING; what it would have done is counted "
              f"at /api/gate", flush=True)
        if GATE_POLICY != GATE_POLICY_REQ:
            print(f"gate: GATE_POLICY={GATE_POLICY_REQ!r} is not a policy -- observing, which is the shipped "
                  f"default; set GATE_POLICY=hold in serve/gateway.sh if intervention was intended", flush=True)
    elif GATE_ON:
        G["task"] = asyncio.create_task(gate_poll())
        print(f"gate: HOLDING (GATE_POLICY=hold) {', '.join(GATED)} held at kv>={GATE_KV_HIGH}"
              f" / waiting>={GATE_WAIT}"
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
