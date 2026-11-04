#!/usr/bin/env python3
"""serve/webui.py - the Monitor and About pages for a running serve/ launch; the pages live in serve/web.

Stdlib only, and it runs on the HOST next to the container (the launch is --network=host, so the
upstream is just 127.0.0.1:PORT). Two modes, and the difference matters on a box where every
millisecond is a measurement:

  observe (default)  poll the upstream's /health, /v1/models and /metrics every POLL_S and read the
                     amdgpu sysfs for per-card busy% / VRAM. Nothing of ours sits in the inference
                     path; the request list stays empty, because stock vLLM has no per-request API.
  --proxy            additionally forward API traffic through this process. That is where the
                     per-request records come from - input, output, wall-clock, first token, token
                     counts - the part of Strata's monitor that needs the server's cooperation
                     (Strata's own server keeps those records; here they can only be taken by a
                     hop). Costs: one extra TCP hop and one thread per concurrent request; the SSE
                     bytes are forwarded as they arrive, never buffered. Opt-in for that reason.

What stock vLLM cannot answer per request (queue wait, prefill/decode split) is reported as the
engine-wide mean from /metrics instead, and the page says so. For the per-request breakdown of
those use bench/ttft_breakdown.py.

usage: python3 serve/webui.py [--port 8081] [--upstream http://127.0.0.1:8080] [--proxy] [--key K]
                              [--poll 1.0] [--keep 100] [--meta launch.json] [--gpus /sys/class/drm]
env:   R9K_WEB_PORT R9K_WEB_UPSTREAM R9K_WEB_PROXY R9K_WEB_KEY R9K_WEB_POLL R9K_WEB_KEEP
       R9K_WEB_META R9K_WEB_GPUS  (flags win; serve/webui.sh is the usual entry point)
"""
import argparse
import http.client
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB = Path(__file__).resolve().parent / "web"
KIND = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
        ".js": "text/javascript; charset=utf-8", ".svg": "image/svg+xml", ".json": "application/json",
        ".woff2": "font/woff2", ".txt": "text/plain; charset=utf-8", ".png": "image/png"}
# 256 KiB of retained input/output per request, like Strata's monitor: the API response is never
# shortened, only our copy of it, and the page says so when it happened.
CAP = 256 * 1024
HIST = 90                                   # sparkline points kept (poll_s * HIST of history)
# The same four waveforms over an hour, one point every SLOW_STEP seconds. The rates in it are taken OVER THE
# INTERVAL (deltas of the cumulative counters), not sampled at that instant: sampling a 1 s window once every 15 s
# would draw a heartbeat, not a trend, and it would miss the work that happened between the beats.
SLOW_STEP = 15.0
SLOW_KEEP = 240                              # 240 x 15 s = an hour of trend, ~40 KB of dicts
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer",
       "transfer-encoding", "upgrade", "content-length", "date", "server", "host"}
NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_:]*")
KVLBL = re.compile(r'num_gpu_blocks="(\d+)"')
BSLBL = re.compile(r'block_size="(\d+)"')
# vLLM has no "what am I configured as" endpoint; the geometry it does publish sits in the labels of
# its own counters, and which labels exist moves between releases. Harvest whatever is there.
LBL = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"')
# vLLM 0.32 publishes the waiting queue twice, once whole and once by reason. Its HELP text says "'capacity' =
# waiting for scheduling capacity; 'deferred' = deferred by transient constraints (LoRA budget, KV transfer,
# blocked status). Sum of all reasons equals vllm:num_requests_waiting"; what the code assigns
# (v1/metrics/loggers.py:1150-1159, v1/core/sched/scheduler.py:2947-2948) is capacity = num_waiting_reqs (the
# ordinary queue) and deferred = num_skipped_waiting_reqs. "capacity" is therefore a name for the queue, not a
# claim about KV blocks -- measured on the live 262k launch: 8 x 30k prompts held running 2 / waiting 6 /
# capacity 6 / deferred 0 with the pool 8.8-10.2% full, because long prefills take the 8,192-token step one at
# a time. Showing the split is what makes that readable without reading the source.
REASON = re.compile(r'reason="([^"]*)"')
WANT = ("model_name", "model_compiler", "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
        "max_model_len", "block_size", "num_gpu_blocks", "num_cpu_blocks", "kv_cache_dtype", "cache_format",
        "speculative_config", "enable_prefix_caching", "enable_chunked_prefill", "scheduler_policy",
        "num_lookahead_tokens", "num_speculative_tokens", "method", "version",
        # the labels the native build (vllm 0.32.0+rocm723, measured 2026-11-02) actually carries on
        # cache_config_info -- and kv_cache_size_tokens is the one that matters on a hybrid GDN model:
        # 1230 blocks x 448 = 551,040 is the attention geometry, the pool holds 497,588 tokens
        "kv_cache_size_tokens", "cache_dtype", "effective_attention_block_size", "gpu_memory_utilization",
        "mamba_block_size", "mamba_cache_dtype", "kv_offloading_backend", "num_gpu_blocks_override")
# The metric names move between vLLM releases: 0.32 renamed gpu_cache_usage_perc to kv_cache_usage_perc and
# carries the per-output-token histogram as request_time_per_output_token_seconds (inter_token_latency_seconds
# alongside it). Read whichever of the build publishes -- and say null when it publishes none.
KV_USED = ("kv_cache_usage_perc", "gpu_cache_usage_perc")
# NOT the same quantity, and the difference is speculation:
#   inter_token_latency_seconds            one observation per engine STEP -- what a client feels as one delta
#   request_time_per_output_token_seconds  one observation per REQUEST, that request's own mean per ACCEPTED
#                                          token, so MTP's accepted drafts divide it.
# Which one is "a stream's tok/s" was written backwards here until 2026-11-02: it claimed 1/step, from a pair
# (15.6 ms a step vs 3.19 ms per accepted token) that implies 4.9 tokens a step -- impossible with
# num_speculative_tokens: 2, whose ceiling is 1 + 2 = 3. Measured on one stream on this launch, one window:
# step 12.84 ms, per accepted token 6.27 ms, generation_tokens_total moving 155 tokens a second. 1/6.27 ms =
# 159.5 tok/s matches the counter; 1/12.84 ms = 77.9 is the STEP rate and undercounts by the tokens a step
# produces (12.84 / 6.27 = 2.05, i.e. 1 + ~1.05 accepted of the 2 drafted). So stream_tok_s follows the
# per-output-token family and only falls back to the step family when that is all the build publishes, and the
# page carries tokens_per_step so the two can be reconciled by eye. One more measured wrinkle: at 0.32 the
# per-output-token family carries ONE observation per REQUEST, so while a request is in flight the window has no
# observation of it at all (itl 12.8-13.0 ms with tpot None, then tpot 6.3 ms the moment the request lands). The
# fallback therefore shows the step rate mid-request -- 77 instead of 155, off by exactly the tokens a step
# produces -- which is why the headline rate is stream_tok_s_live (counter-derived, live every window) and not
# this field. stream_tok_s stays in the payload for the finished-request view.
ITL_STEP = ("inter_token_latency_seconds",)
TPOT = ("request_time_per_output_token_seconds", "time_per_output_token_seconds")


# -------------------------------------------------------------------------------- upstream metrics
def parse_metrics(text):
    """Every vllm:* series, collapsed by name: name -> (sum, max, series_count).

    Counters want the sum (the ranks each count their part); gauges want the mean, and the max too
    because "one rank's KV pool is full" is the sentence that matters. Buckets are skipped: _sum and
    _count are the two numbers any mean we want comes out of. Also returned: the KV pool geometry,
    which in vLLM is only readable as metric labels, and the model name from those same labels."""
    out, cap, info = {}, None, {}
    for line in text.splitlines():
        if not line.startswith("vllm:"):
            continue
        m = NAME.match(line)
        if not m:
            continue
        name = m.group(0)[5:]
        if name.endswith("_bucket") or name.endswith("_created"):
            continue
        rest = line[m.end():].strip()
        labels = ""
        if rest.startswith("{"):
            end = rest.index("}")
            labels, rest = rest[:end], rest[end + 1:].strip()
        try:
            value = float(rest.split()[0])
        except (IndexError, ValueError):
            continue
        s = out.get(name)
        if s is None:
            out[name] = [value, value, 1]
        else:
            s[0] += value
            s[1] = max(s[1], value)
            s[2] += 1
        rsn = REASON.search(labels)
        if rsn:                               # a labelled gauge, kept per label value as well
            sub = out.get(f"{name};{rsn.group(1)}")
            if sub is None:
                out[f"{name};{rsn.group(1)}"] = [value, value, 1]
            else:
                sub[0] += value
                sub[1] = max(sub[1], value)
                sub[2] += 1
        if cap is None and "num_gpu_blocks" in labels:
            b, bs = KVLBL.search(labels), BSLBL.search(labels)
            if b and bs:
                cap = int(b.group(1)) * int(bs.group(1))
                kvn = re.search(r'kv_cache_size_tokens="(\d+)"', labels)
                if kvn:                       # the pool's real capacity, which on a hybrid model is not
                    cap = int(kvn.group(1))   # blocks x block_size
        for k, v in LBL.findall(labels):
            if k in WANT and info.get(k) in (None, ""):
                info[k] = v
    return out, cap, info


def since(s, g):
    """The same means over the engine's WHOLE run, from the cumulative counters.

    Every histogram in vLLM is observed when a request FINISHES, so a window mean is undefined for the
    whole time a long request is in flight -- a 2 s poll of a 600-token generation would read "no data"
    for TTFT, queue, prefill and decode the entire time. The cumulative ratio is not a window mean and is
    labelled as such on the page ("since launch"), which is what the panel falls back to. The engine's
    counters are cumulative from its own start, so that is exactly the span they describe."""
    am = lambda h: (g(h + "_sum") / g(h + "_count") * 1e3) if g(h + "_count") > 0 else None
    ac = lambda h: (g(h + "_sum") / g(h + "_count")) if g(h + "_count") > 0 else None
    ma = lambda fam: next((m for m in (am(h) for h in fam) if m is not None), None)
    dgrams = g("spec_decode_num_draft_tokens_total")
    return {"ttft_ms": am("time_to_first_token_seconds"), "itl_ms": ma(ITL_STEP) or ma(TPOT),
            "tpot_ms": ma(TPOT),
            "e2e_ms": am("e2e_request_latency_seconds"), "queue_ms": am("request_queue_time_seconds"),
            "prefill_ms": am("request_prefill_time_seconds"), "decode_ms": am("request_decode_time_seconds"),
            # NOT x1e3: these families count tokens, not seconds. The live run caught it -- the first
            # /api/status of a real engine reported a mean prompt of 2,028,442 tokens because the time
            # scaling was applied to a token count (it is 2,028).
            "prompt_tokens_mean": ac("request_prompt_tokens"),
            "generation_tokens_mean": ac("request_generation_tokens"),
            # Cumulative counts, not means: the Monitor's decode total card shows these, because a count cannot be
            # misread as an average. This launch is at ~124 million prompt tokens against ~1 million generated,
            # which is the single most useful fact about its decode rate (notes/context-262k.md).
            "generation_tokens_total": g("generation_tokens_total"),
            "prompt_tokens_total": g("prompt_tokens_total"),
            "iterations": g("iteration_tokens_total_count"),
            # iteration_tokens_total is labelled "tokens per engine_step" and is not one: at 0.32.0 the observation
            # is computed prompt tokens + generation tokens (v1/metrics/loggers.py:1273), and a chunked prefill is
            # credited in a single record holding the whole prompt -- measured: a 4,893-token prompt produced
            # records 4,893 and 4; a 29,969-token one produced 29,969, 4 and 1, while decode steps record
            # individually. So this ratio tracks prompt size, can exceed max-num-batched-tokens, and
            # iters_per_req below is records per request rather than steps. The page says so on the line.
            "tokens_per_iter": (g("iteration_tokens_total_sum") / g("iteration_tokens_total_count"))
                                if g("iteration_tokens_total_count") > 0 else None,
            "accept_pct": (100.0 * g("spec_decode_num_accepted_tokens_total") / dgrams) if dgrams > 0 else None,
            # accepted tokens per DRAFT, from the two counters that mean exactly that
            # (v1/spec_decode/metrics.py: observe_draft adds one to num_drafts and <= num_spec_tokens to
            # num_accepted_tokens). With MTP=2 it cannot exceed 2.0, which is the sanity check the page has been
            # missing -- and it is a number that belongs nowhere near the iteration-record ratio above, which is
            # four orders of magnitude larger on a prefill-heavy launch and reads as a speculation figure if put
            # under that heading.
            "accepted_per_draft": (g("spec_decode_num_accepted_tokens_total")
                                   / g("spec_decode_num_drafts_total"))
                                  if g("spec_decode_num_drafts_total") > 0 else None,
            "iters_per_req": (g("iteration_tokens_total_count") / g("e2e_request_latency_seconds_count"))
                             if g("e2e_request_latency_seconds_count") > 0 else None,
            "requests": g("e2e_request_latency_seconds_count") if s.get("e2e_request_latency_seconds_count") else 0}


# -------------------------------------------------------------------------------------- trailing averages
# The poll window is 1 s, and on a busy launch one poll is a coin toss. Measured on this box (2026-11-04, six
# streams in flight): one poll read 9.9 tok/s of generation and no prefill at all, while the ten minutes around it,
# from the engine's own counters, were 41 tok/s of generation, 6,900 tok/s of prefill and ~19 tok/s per stream.
# The window is the right instrument for "what is it doing right now" and the wrong one for "what has it been
# doing". The page had the window and the whole-run since() block, and the gap between those two is where the
# question actually gets asked -- a since-launch mean over a day with quiet hours in it is not an average level.
AVG_SPANS = (("1m", 60.0), ("15m", 900.0), ("1h", 3600.0))
AVG_STEP = 5.0                                      # snapshot spacing: this ring is for minutes, not seconds
AVG_KEEP = int(max(s for _, s in AVG_SPANS) / AVG_STEP) + 8
AVG_MIN = 15.0                                      # shorter than this and it is a poll, not an average

# Deliberately a name list rather than the whole parsed dict: the ring would otherwise carry every family vLLM
# happens to publish, for an hour, at 5 s spacing, to average a dozen of them. _sum and _count pairs are listed
# explicitly, because a mean over a span needs both halves of the histogram family.
AVG_COUNTERS = ("generation_tokens_total", "prompt_tokens_total", "e2e_request_latency_seconds_count",
                "e2e_request_latency_seconds_sum",
                "request_prefill_time_seconds_sum", "request_prefill_time_seconds_count",
                "request_decode_time_seconds_sum", "request_decode_time_seconds_count",
                "request_prompt_tokens_sum", "request_prompt_tokens_count",
                "request_generation_tokens_sum", "request_generation_tokens_count",
                "request_time_per_output_token_seconds_sum", "request_time_per_output_token_seconds_count",
                "inter_token_latency_seconds_sum", "inter_token_latency_seconds_count",
                "time_to_first_token_seconds_sum", "time_to_first_token_seconds_count",
                "request_queue_time_seconds_sum", "request_queue_time_seconds_count",
                "e2e_request_latency_seconds_sum",
                "spec_decode_num_drafts_total", "spec_decode_num_accepted_tokens_total",
                "spec_decode_num_draft_tokens_total", "prefix_cache_queries_total", "prefix_cache_hits_total",
                # the record counter: what the decode cards divide by to say "tokens a step" over a span instead
                # of a second. Records, not steps, at 0.32 -- see the note in since() -- which the titles say.
                "iteration_tokens_total_count", "iteration_tokens_total_sum")


def avg_sample(cur, t0):
    """One point of the trailing-average ring: the cumulative counters, plus the gauges whose mean is wanted."""
    kvn = next((n for n in KV_USED if cur.get(n)), None)
    s = {"at": t0,
         "run": (cur.get("num_requests_running") or [0, 0.0, 0])[1],
         "wait": (cur.get("num_requests_waiting") or [0, 0.0, 0])[1],
         "kv": (cur[kvn] or [0, 0.0, 0])[1] if kvn else None}
    for n in AVG_COUNTERS:
        v = cur.get(n)
        if v:
            s[n] = v[0]
    return s


def trailing(snaps, cur, t0, spans=AVG_SPANS):
    """The window's means, recomputed over the last minute, quarter hour and hour.

    A rate over a span is the delta of a cumulative counter over the delta of time, and a mean latency is
    delta(sum)/delta(count) of the engine's own histogram family: both exact over the span, because the counters
    are cumulative. What would be wrong here is averaging the 1 s samples the sparkline keeps -- that weights a
    second with nothing running the same as a second with six streams running, and on a launch like this one most
    seconds have a gap in them.

    Two conditions make a span weaker than it asks for, and both are reported rather than papered over. The ring
    is younger than the span, so `span_s` carries what was actually averaged. And a counter that went DOWN means
    the engine restarted inside the span, which would otherwise come out as a large negative rate -- the whole
    span is then None, with `reset` saying why. A span younger than AVG_MIN reports None fields too: a dash costs
    the reader nothing, while a 12 s window wearing an hour's label costs the argument built on it.
    """
    out = {}
    for label, span in spans:
        base = None
        for s in snaps:                              # the newest snapshot at or before the far edge
            if s["at"] <= t0 - span and (base is None or s["at"] > base["at"]):
                base = s
        if base is None and snaps:
            base = min(snaps, key=lambda s: s["at"])  # the ring is younger than the span asked for
        if base is None or (t0 - base["at"]) < AVG_MIN:
            out[label] = {"span_s": (t0 - base["at"]) if base else None, "generation_tok_s": None}
            continue
        span_s = t0 - base["at"]
        miss = lambda n: n not in base                # a family that appeared after the base snapshot
        dn = lambda n: None if miss(n) else (cur.get(n) or [0.0, 0.0, 0])[0] - base[n]
        msk = lambda fam: (1e3 * dn(fam + "_sum") / dn(fam + "_count")) if dn(fam + "_count") else None
        mtk = lambda fam: (dn(fam + "_sum") / dn(fam + "_count")) if dn(fam + "_count") else None
        first = lambda fams: next((msk(h) for h in fams if dn(h + "_count")), None)
        gen, pre = dn("generation_tokens_total"), dn("prompt_tokens_total")
        pt, dtm = dn("request_prefill_time_seconds_sum"), dn("request_decode_time_seconds_sum")
        # Decode time, from the family that grows EVERY decode step rather than at request end. Measured on this
        # launch (2026-11-04): inter_token_latency_seconds carries 430,954 observations summing to 24,236.14 s
        # and request_decode_time_seconds carries 1,812 observations summing to 24,216.27 s -- 0.08% apart, which
        # is what licenses using either as "seconds the box spent in the decode phase". The step family is
        # preferred because a short span in which no request finished still has a denominator; at request-end
        # granularity a 60 s span of pure decode would report none. Both are reported so a reader can reconcile.
        dtl = dn("inter_token_latency_seconds_sum")
        run = [x["run"] for x in snaps if x["at"] > base["at"]]
        wait = [x["wait"] for x in snaps if x["at"] > base["at"]]
        kvv = [x["kv"] for x in snaps if x["at"] > base["at"] and x["kv"] is not None]
        drafts, acc = dn("spec_decode_num_drafts_total"), dn("spec_decode_num_accepted_tokens_total")
        iters = dn("iteration_tokens_total_count")
        dgrams = dn("spec_decode_num_draft_tokens_total")
        hitq, hits = dn("prefix_cache_queries_total"), dn("prefix_cache_hits_total")
        # What the engine READ, not what the prompt was called. vLLM credits cached prefix tokens to
        # prompt_tokens_total, so with the prefix cache on, prompt_tokens / prefill_seconds is not a read speed:
        # it counts tokens that were never read and overstates it by the hit rate. Strata's run average uses
        # `prompt_tokens - reused` for exactly this reason (serve/web/app.js), and says so on the card. On this
        # recipe the cache is off on purpose (PREFIX_CACHE=0), so read == credited today; the distinction is here
        # so that turning the cache back on cannot quietly inflate the number.
        read = None if pre is None else pre - (hits or 0.0)
        reset = any(v is not None and v < 0 for v in (gen, pre, drafts, iters, dtm, dtl, hits))
        itl, tpot = first(ITL_STEP), first(TPOT)
        genps = gen / span_s if gen is not None else None
        runmean = (sum(run) / len(run)) if run else None
        out[label] = {
            "span_s": span_s, "reset": reset, "samples": len(run),
            "generation_tok_s": None if reset else genps,
            # The same span answered with COUNTS instead of rates: what actually went in and out of the box in
            # it, and how many of its records were decode work. A count cannot be mistaken for an average.
            "generation_tokens": None if reset else gen,
            "prompt_tokens": None if reset else pre,
            "records": None if reset else iters,
            "gen_per_step": (gen / iters) if (gen is not None and iters) else None,
            "drafts_per_step": (drafts / iters) if (drafts is not None and iters) else None,
            "prompt_tok_s": None if reset or pre is None else pre / span_s,
            "requests_per_min": None if reset or dn("e2e_request_latency_seconds_count") is None
                                 else 60.0 * dn("e2e_request_latency_seconds_count") / span_s,
            # aggregate over the MEAN concurrency of the span, which is the honest long-window version of the
            # window's aggregate-over-running figure: running moves, and dividing by this poll's count would be
            # wrong by whatever it moved by.
            "stream_tok_s": (genps / runmean) if (genps is not None and runmean) else None,
            # The two speeds the box is actually judged on, token-weighted and exact over the span: prompt tokens
            # per second of prefill time, and generated tokens per second of decode time. The engine's own
            # prefill/decode seconds, not wall time -- which is why they stay readable while the queue moves.
            # READ tokens per second of prefill time. The credited version stays alongside it so a reader (or a
            # future us) can see what the cache took out, the same way the decode rate reports both families.
            "prefill_tok_s": (read / pt) if (read is not None and pt) else None,
            "prefill_tok_s_credited": (pre / pt) if (pre is not None and pt) else None,
            "prompt_read_tokens": None if reset else read,
            # Generated tokens per second of DECODE time: prompt chewing is not in the denominator, so this is
            # "how fast it decodes when it is decoding". Measured on the live launch, 68.8 tok/s against 20.1
            # summed over the span's wall seconds -- the gap is that only 29% of those seconds held a stream in
            # its decode phase while 83% held somebody's prefill. The wall figure stays visible (generation_tok_s
            # above) because that is what the traffic produces per wall second; the two answer different
            # questions and the page shows both.
            "decode_tok_s": (gen / dtl) if (gen is not None and dtl) else None,
            "decode_time_s": None if reset else dtl,
            "decode_time_req_s": (gen / dtm) if (gen is not None and dtm) else None,
            "decode_time_pct": (100.0 * dtl / span_s) if (dtl is not None and span_s) else None,
            "itl_ms": itl, "tpot_ms": tpot,
            "stream_tok_s_hist": (1e3 / tpot) if tpot else ((1e3 / itl) if itl else None),
            "ttft_ms": msk("time_to_first_token_seconds"), "queue_ms": msk("request_queue_time_seconds"),
            "e2e_ms": msk("e2e_request_latency_seconds"),
            "prefill_ms": msk("request_prefill_time_seconds"), "decode_ms": msk("request_decode_time_seconds"),
            "prompt_tokens_mean": mtk("request_prompt_tokens"), "generation_tokens_mean": mtk("request_generation_tokens"),
            "tokens_per_step": (itl / tpot) if (itl and tpot) else None,
            "accept_pct": (100.0 * acc / dgrams) if (dgrams and acc is not None) else None,
            "accepted_per_draft": (acc / drafts) if (drafts and acc is not None) else None,
            "prefix_hit_pct": (100.0 * hits / hitq) if (hitq and hits is not None) else None,
            "running_mean": runmean, "waiting_mean": (sum(wait) / len(wait)) if wait else None,
            "kv_used_pct_mean": (100.0 * sum(kvv) / len(kvv)) if kvv else None}
        if reset:
            for k in list(out[label]):
                if k not in ("span_s", "reset", "samples"):
                    out[label][k] = None
    return out


def derive(cur, prev, dt, cap, info, mode, kept, hist):
    """One /api/status payload out of two metric snapshots dt seconds apart.

    dt is None on the very first read, before any window exists: the gauges and the configuration are
    reportable then, a rate is not. Reporting 600 tok/s because we divided the whole run's counter by
    0.001 s is the kind of number this box generates enough of already."""
    have = bool(prev) and dt is not None and dt > 0
    g = lambda n, i=0: (cur.get(n) or [0.0, 0.0, 0])[i]                    # summed value of a series
    d = lambda n: (g(n) - ((prev.get(n) or [0.0, 0.0, 0])[0])) if have else 0.0
    per = lambda v: v / dt if have else None
    nrec = max(d("e2e_request_latency_seconds_count"), 0.0)
    mean = lambda h: (d(h + "_sum") / d(h + "_count") * 1e3) if d(h + "_count") > 0 else None
    meanc = lambda h: (d(h + "_sum") / d(h + "_count")) if d(h + "_count") > 0 else None   # token families
    drafts, accepted = d("spec_decode_num_drafts_total"), d("spec_decode_num_accepted_tokens_total")
    dgrams = d("spec_decode_num_draft_tokens_total")          # tokens drafted (0.31+ name)
    hitq, hits = (d("prefix_cache_queries_total"), d("prefix_cache_hits_total")) if have \
        else (g("prefix_cache_queries_total"), g("prefix_cache_hits_total"))
    mean_any = lambda fam: next((m for m in (mean(h) for h in fam) if m is not None), None)
    itl = mean_any(ITL_STEP) or mean_any(TPOT)          # a stream's rate is 1 / step; older builds only have
    tpot = mean_any(TPOT)                               # the per-token family, which then stands in for both
    kv_used = next((n for n in KV_USED if cur.get(n)), None)
    swapped = cur.get("num_requests_swapped")
    wr = lambda r: (cur.get(f"num_requests_waiting_by_reason;{r}") or [None, None, 0])[1]
    prem_tot = cur.get("num_preemptions_total")
    live = {"running": g("num_requests_running", 1), "waiting": g("num_requests_waiting", 1),
            "waiting_capacity": wr("capacity"), "waiting_deferred": wr("deferred"),
            "swapped": (swapped[1] if swapped else None), "preempting": g("num_preemptions_total") > 0,
            # Cumulative since launch, not just "is it happening" -- a preempt costs the victim a whole re-prefill,
            # so the count over a session is the number that predicts the pain (measured: two 160k-token requests
            # at a 100%-full pool, first token 20.9 s / 30.2 s, their own decode 14.8 / 3.4 tok/s). Read the
            # series' presence rather than g(), so "never preempted" is a real 0 and an older build that does not
            # publish the counter stays unknown.
            "preemptions_total": (prem_tot[0] if prem_tot else None),
            "kv_used_pct": 100.0 * g(kv_used, 1) if kv_used else None,
            "kv_used_pct_mean": 100.0 * g(kv_used) if kv_used else None,
            "kv_tokens": cap,
            "prefix_hit_pct": (100.0 * hits / hitq) if hitq > 0 else None}
    # What is live WHILE a request runs, measured on this box against 0.32.0+rocm723 (2026-11-02, the
    # traces are in notes/webui.md): generation_tokens_total advances every engine step (one 900-token
    # stream moved it ~120 tokens per 0.5 s sample), and so does the KV gauge (0.036 -> 0.1007 across a
    # 51,859-token prefill). prompt_tokens_total and iteration_tokens_total_* do NOT: the same prefill
    # held prompt_tokens_total flat for 10 s and then jumped 129 -> 51,988 in one step. Every latency
    # histogram is observed at request end too. So the two rates the page puts in its headline are the two
    # that can be computed from what moves per step; the prefill card shows both its live estimate and the
    # exact figure, and says which is which.
    gen_rate = per(d("generation_tokens_total"))
    run_now = g("num_requests_running", 1)
    kv_now = (cap or 0) * g(kv_used, 1) if (kv_used and cap) else None
    kv_was = (cap or 0) * ((prev.get(kv_used) or [0.0, 0.0, 0])[1]) if (kv_used and cap and have) else None
    rate = {"window_s": dt, "requests": nrec if have else None, "requests_per_min": per(nrec),
            "generation_tokens": d("generation_tokens_total") if have else None,
            "generation_tok_s": gen_rate, "prompt_tok_s": per(d("prompt_tokens_total")),
            # the same correction for one poll: cached tokens were not read (see trailing())
            "prompt_read_tok_s": per(d("prompt_tokens_total") - d("prefix_cache_hits_total")),
            "prompt_read_tokens": (d("prompt_tokens_total") - d("prefix_cache_hits_total")) if have else None,
            # Per poll: generated tokens per second of DECODE time in this window. d(ITL_sum) is 0 when no decode
            # step ran (a window that was all prefill, or idle) -- then there is no denominator, which is a fact
            # about the window rather than a rate of 0.
            "decode_tok_s": ((d("generation_tokens_total") / (d("inter_token_latency_seconds_sum")
                                                              or d("request_decode_time_seconds_sum")))
                             if (d("inter_token_latency_seconds_sum") or d("request_decode_time_seconds_sum")) > 0
                             else None),
            "decode_time_s": (d("inter_token_latency_seconds_sum")
                              or d("request_decode_time_seconds_sum")) or None,
            # aggregate / running: a rate for the stream you are watching, available while it runs (1/itl
            # is not -- see the block above). Exact only when the streams run at the same rate.
            "stream_tok_s_live": (gen_rate / run_now) if (gen_rate is not None and run_now >= 1) else None,
            # tokens of context added per second, from the live gauge. An estimate: it counts blocks, so it
            # tracks prefill work within a few percent (measured: 5,787 tok/s against a true 5,459 over the
            # same 51,859-token prefill) and goes NEGATIVE when requests finish and free blocks.
            "kv_tok_s": per(kv_now - kv_was) if (kv_now is not None and kv_was is not None) else None,
            "ttft_ms": mean("time_to_first_token_seconds"), "itl_ms": itl, "tpot_ms": tpot,
            "stream_tok_s": (1e3 / tpot) if tpot else ((1e3 / itl) if itl else None),
            # tokens a step turns into, from the two families' means: 1.0 without speculation, up to
            # 1 + num_speculative_tokens with it. This is the factor that makes the step rate and the token
            # rate differ, so the page can put both on one line instead of choosing between them.
            "tokens_per_step": (itl / tpot) if (itl and tpot) else None,
            "e2e_ms": mean("e2e_request_latency_seconds"), "queue_ms": mean("request_queue_time_seconds"),
            "prefill_ms": mean("request_prefill_time_seconds"), "decode_ms": mean("request_decode_time_seconds"),
            "prompt_tokens_mean": meanc("request_prompt_tokens"),
            "generation_tokens_mean": meanc("request_generation_tokens"),
            # generated tokens per iteration RECORD in this window -- a total across streams, deliberately not a
            # per-stream mean: on a prefill-heavy launch this is ~2-4 while a lone stream reaches ~2.3 tokens a
            # step of its own, and the difference is fan-out and step time, not a per-stream average (the census
            # is in notes/context-262k.md). Records, not steps, at 0.32: a whole chunked prefill lands in one.
            "gen_per_step": (d("generation_tokens_total") / d("iteration_tokens_total_count"))
                            if d("iteration_tokens_total_count") > 0 else None,
            "iters_per_req": (d("iteration_tokens_total_count") / nrec) if nrec else None,
            "tokens_per_iter": (d("iteration_tokens_total_sum") / d("iteration_tokens_total_count"))
                               if d("iteration_tokens_total_count") > 0 else None,
            "drafts_per_req": (drafts / nrec) if nrec else None,
            "accept_pct": (100.0 * accepted / dgrams) if dgrams > 0 else None,
            "accepted_per_draft": (accepted / drafts) if drafts > 0 else None,
            "preemptions": d("num_preemptions_total") if have else None}
    return {"mode": mode, "at": time.time(), "model_name": info.get("model_name"), "config": info,
            "live": live, "rate": rate, "since": since(cur, g),
            "history": hist, "requests_kept": kept, "metrics_seen": len(cur)}


# -------------------------------------------------------------------------------------- gpu sysfs
def _f(s):
    try:
        return float(str(s).split()[0])
    except (IndexError, ValueError):
        return None


SMI_DEFAULT = ["rocm-smi", "--showbus", "--showuse", "--showmeminfo", "vram", "--showtemp",
               "--showpower", "--showclocks", "--showproductname", "--json"]


def read_smi(command, timeout=8.0):
    """busy% / VRAM / temperature / power / clocks from `rocm-smi --json`, keyed by PCI bus id so the
    numbers land on the card the sysfs describes instead of next to it.

    Opt-in (--smi), and deliberately not the default: rocm-smi opens /dev/kfd on every call, which on a
    box you are measuring is a perturbation, not a read. It is worth having where the sysfs answers
    EBUSY, which on this box is most of it. Key names move between ROCm versions, so this matches on
    substrings and drops "N/A" instead of expecting a schema. Units: degrees C, watts, MHz -- as
    rocm-smi prints them, unlike the sysfs' millidegrees.
    """
    try:
        out = subprocess.run(command, capture_output=True, text=True, timeout=timeout).stdout
        data = json.loads(out[out.index("{"): out.rindex("}") + 1])
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        return None, f"{type(e).__name__}: {str(e)[:200]}"
    keys = (("gpu use", "busy_pct"), ("vram total used", "vram_used"), ("vram total memory", "vram_total"),
            ("total visible vram", "vis_vram_used"), ("edge", "temp_c"), ("junction", "temp_c"),
            ("graphics temperature", "temp_c"), ("package power", "power_w"), ("device name", "name"),
            ("product name", "name"), ("pci bus", "bus"))
    by_bus = {}
    for card, vals in (data or {}).items():
        row = {}
        for k, v in (vals or {}).items():
            v = str(v).strip()
            if not v or v.upper() in ("N/A", "NULL", "0X0"):
                continue
            kl = k.lower()
            for pat, key in keys:
                if pat in kl:
                    if row.get(key) is None:
                        row[key] = v if key in ("name", "bus") else _f(v)
                    break
            else:
                if "frequency" in kl and ("mhz" in kl or "clock" in kl):
                    if "sclk" in kl and row.get("sclk_mhz") is None:
                        row["sclk_mhz"] = _f(v)
                    elif "mclk" in kl and row.get("mclk_mhz") is None:
                        row["mclk_mhz"] = _f(v)
        if not row:
            continue
        row["smi_card"] = card
        row["bus"] = (row.get("bus") or "").lower().replace("0000:", "")
        by_bus[row["bus"] or card] = row
    return by_bus, None


def merge_smi(gpus, by_bus):
    """Fill only what the sysfs refused to answer, and say which number came from where."""
    seen = set()
    for g in gpus:
        row = by_bus.get(g.get("bus") or "")
        if not row:
            continue
        seen.add(g.get("bus"))
        g.setdefault("from_smi", [])
        for k, v in row.items():
            if k in ("bus", "smi_card") or v is None:
                continue
            if g.get(k) is None:
                g[k] = v
                g["from_smi"].append(k)
    for bus, row in by_bus.items():                     # a card the sysfs root does not describe at all
        if bus in seen:
            continue
        g = {"card": row.get("smi_card", bus), "name": row.get("name") or "amdgpu", "bus": bus,
             "unread": [], "from_smi": [], "vram_used": None, "vram_total": None}
        for k in ("busy_pct", "mem_busy_pct", "vram_used", "vram_total", "temp_c", "power_w",
                  "sclk_mhz", "mclk_mhz", "name"):
            if row.get(k) is not None:
                g[k] = row[k]
                g["from_smi"].append(k)
        gpus.append(g)
    return gpus


def read_host():
    """The box the launch runs on: CPU model, thread count, RAM, kernel, uptime - all from procfs."""
    def cat(p):
        try:
            return Path(p).read_text()
        except OSError:
            return ""

    cpu = ""
    for line in cat("/proc/cpuinfo").splitlines():
        if line.lower().startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break
    mem = 0
    for line in cat("/proc/meminfo").splitlines():
        if line.startswith("MemTotal:"):
            mem = int(line.split()[1]) * 1024
            break
    try:
        up = float(Path("/proc/uptime").read_text().split()[0])
    except (OSError, ValueError):
        up = None
    u = os.uname()
    return {"cpu": cpu, "threads": os.cpu_count(), "ram": mem, "kernel": f"{u.sysname} {u.release}",
            "hostname": u.nodename, "uptime_s": up}


def read_gpus(root):
    """Per-card state from the amdgpu sysfs: no ROCm runtime, no container, nothing started on the card.

    Read tolerantly and say what could not be read: which attributes answer depends on the state of the card,
    and on this box it has flipped within a day. With nothing running, all four answered EBUSY and only the VRAM
    files replied (57 MiB); with the serve holding a context on each, gpu_busy_percent (99), temp1_input (61-72
    C), power1_average (232-440 W) and the labelled sclk (2.7-3.3 GHz) all answer. So "refused" and "zero" are
    different states and the page must not confuse them: a card that answers nothing is unreadable, never busy 0%."""
    gpus = []
    for card in sorted(Path(root).glob("card[0-9]*")):
        dev = card / "device"
        if not dev.is_dir():
            continue

        def num(p):
            try:
                return float(p.read_text().strip())
            except (OSError, ValueError):
                return None

        vals, unread = {}, []
        for key, name in (("busy_pct", "gpu_busy_percent"), ("mem_busy_pct", "mem_busy_percent"),
                          ("vram_used", "mem_info_vram_used"), ("vram_total", "mem_info_vram_total"),
                          ("gtt_used", "mem_info_gtt_used")):
            p = dev / name
            if p.exists():                              # exists but unreadable is the interesting case
                v = num(p)
                if v is None:
                    unread.append(name)
                else:
                    vals[key] = v
        if not vals:                                      # not a GPU (or no readable counters at all)
            continue
        # hwmon: names move between kernels -- power1_input / power1_average, and the clock files are
        # labelled sclk and mclk, which is the SM clock the load-time mysteries are always about. The sensor
        # directory itself moves too: on the stack this box runs it is device/hwmon/hwmonN, and it used to be
        # device/hwmonN. Globbing only the second form is how the Monitor showed a blank temperature and power
        # for months on a box whose files read perfectly well (measured 2026-11-02: temp1_input 65000,
        # power1_average 205000000, freq1_label "sclk" -> 3311000000, and `unread` empty the whole time).
        # ...and temperatures are read BY LABEL, because there are three of them and the first is the least
        # interesting: edge / junction / mem. Reading temp1 (edge) is what made one card look 10 C hotter than
        # its neighbours while its die sat at everyone else's temperature and its GDDR7 sat at 96 C (measured
        # 2026-11-02, 180 s at 1 Hz, four cards at 291-293 W each): the anomaly was in the board and memory,
        # which temp1 cannot see. Junction is the headline because it is the sensor with a published limit
        # (temp*_crit = 110 C) and the one the fan map follows; mem has no published threshold on these cards,
        # so it is shown and never coloured.
        temps, crit, crits, fan, pwm = {}, None, {}, None, None
        power = sclk = mclk = cap = None
        for hm in sorted(dev.glob("hwmon*")) + sorted(dev.glob("hwmon*/hwmon*")):
            for f in hm.glob("temp*_label"):
                try:
                    lab = f.read_text().strip().lower()
                except OSError:
                    continue
                stem = f.name[:-len("_label")]
                got = num(hm / (stem + "_input"))
                if got is None:
                    continue
                temps.setdefault(lab, got / 1e3)
                e = num(hm / (stem + "_emergency"))
                if e is not None:
                    crits.setdefault(lab + "_emergency", e / 1e3)
                c = num(hm / (stem + "_crit"))
                if c is not None:
                    crits.setdefault(lab, c / 1e3)      # every sensor carries its OWN limit and its own
                    if crit is None:                    # emergency point: junction 110/115, mem 108/113
                        crit = c / 1e3
            t1 = num(hm / "temp1_input")                # a kernel with no labels still gets what it had
            if t1 is not None:
                temps.setdefault("edge", t1 / 1e3)
            power = power if power is not None else (num(hm / "power1_input") or num(hm / "power1_average"))
            cap = cap if cap is not None else (num(hm / "power1_cap") or num(hm / "power1_cap_max"))
            fan = fan if fan is not None else num(hm / "fan1_input")
            pwm = pwm if pwm is not None else num(hm / "pwm1")
            for f in hm.glob("freq*_label"):
                try:
                    lab = f.read_text().strip()
                except OSError:
                    continue
                got = num(hm / (f.name[: -len("_label")] + "_input"))
                if got is None:
                    continue
                if lab == "sclk" and sclk is None:
                    sclk = got
                elif lab == "mclk" and mclk is None:
                    mclk = got
        try:
            name = (dev / "product_name").read_text().strip()
        except OSError:
            name = "amdgpu"
        bus = None                                        # the join key for rocm-smi, whose card indices
        try:                                              # are its own
            m = re.search(r"PCI_SLOT_NAME=(\S+)", (dev / "uevent").read_text())
            bus = m.group(1).lower().replace("0000:", "") if m else None
        except OSError:
            pass
        vals.update({"card": card.name, "name": name, "bus": bus,
                     # temp_c is the headline sensor: junction where the card publishes it (limit-bearing and
                     # fan-relevant), edge only as the fallback an unlabelled kernel leaves us with.
                     "temp_c": temps.get("junction") or temps.get("edge"),
                     "temp_junction_c": temps.get("junction"), "temp_edge_c": temps.get("edge"),
                     "temp_mem_c": temps.get("mem"), "temp_crit_c": crits.get("junction") or crit,
                     "temp_mem_crit_c": crits.get("mem"),
                     "temp_mem_emerg_c": crits.get("mem_emergency"),
                     # Fan duty belongs in the thermal story: the loop follows the junction, so a card can run a
                     # lower duty than its neighbours while its memory is the hottest thing on the box.
                     "fan_rpm": fan, "pwm": pwm,
                     "power_w": (power / 1e6) if power else None,
                     # The power ceiling is what makes a power waveform mean anything: a card pinned at its cap
                     # and one idling at 205 of 500 W look identical on an autoscaled axis.
                     "power_cap_w": (cap / 1e6) if cap else None,
                     "sclk_mhz": (sclk / 1e6) if sclk else None, "mclk_mhz": (mclk / 1e6) if mclk else None,
                     "unread": unread})
        gpus.append(vals)
    return gpus


# --------------------------------------------------------------------------------------- pcie sysfs
# PCIe: what is actually collectable here, measured rather than assumed.
#
# Not throughput. The counters that would answer "how many bytes crossed the link" -- BANDWIDTH,
# CURRENT_BANDWIDTH_SENT / _RECEIVED, MAX_PACKET_SIZE, REPLAY_COUNT -- all answer N/A on these RDNA4 cards for
# every GPU (one amd-smi probe each), and the kernel exposes no byte counter either. What CAN be watched, for
# free, is the link layer: the speed and width every link negotiated, and the AER error counters the kernel keeps
# for each device and for every bridge above it. Ordinary user, no root, no /dev/kfd, a few dozen file reads.
#
# That is exactly the evidence the reset storm was missing. Measured on this box since the 11:07 boot: upstream
# port 0000:c4:00.0 holding 24 correctable (all BadTLP), 0000:c1:00.0 holding 3, the switch downstream ports
# 0000:c5:00.0 and 0000:05:00.0 one each, the GPU endpoints 0000:c6:00.0 and 0000:06:00.0 one each, and every
# link still at its maximum 32.0 GT/s x16 -- the reassuring half of that picture.

PCI_BDF = re.compile(r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-f]$")
AER_FILES = {"cor": "aer_dev_correctable", "nonfatal": "aer_dev_nonfatal", "fatal": "aer_dev_fatal"}
AER_TOTAL = {"cor": "TOTAL_ERR_COR", "nonfatal": "TOTAL_ERR_NONFATAL", "fatal": "TOTAL_ERR_FATAL"}


def _num(v):
    """A counter that answered is a number; one that did not is None, never 0 -- "the file refused" and
    "the link is at zero" have to stay different all the way to the waveform."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _pci_num(path):
    try:
        return int((path).read_text().strip())
    except (OSError, ValueError):
        return None


def _pci_speed(path):
    try:
        return float(path.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _read_str(path):
    try:
        return path.read_text().strip() or None
    except OSError:
        return None


def _aer(path):
    """An aer_dev_* file: 'BadTLP 24' one per line. Totals split out, the rest kept only if non-zero --
    the full list is 23 counters wide and the ones that matter are the ones that moved.

    Two tokens per line, and the value parsed as a whole number: an earlier version of this tested the value
    string's length instead of the row's, so every count of 10 or more failed to parse -- on this box that
    silently dropped the single most interesting number, 24 correctable BadTLP on 0000:c4:00.0."""
    try:
        rows = [ln.split() for ln in path.read_text().splitlines() if ln.strip()]
    except OSError:
        return None, {}
    vals = {row[0]: int(row[1]) for row in rows
            if len(row) == 2 and re.fullmatch(r"-?\d+", row[1])}
    return vals, {n: v for n, v in vals.items() if v and not n.startswith("TOTAL")}


def read_slots(root):
    """`/sys/bus/pci/slots/`: the machine's own answer to "which physical slot is this card in".

    Why this is worth a read of its own: `card2` is a name the driver assigned this boot, and it says nothing
    about the chassis. The thing a person at the machine can act on is the BDF and, through the BIOS's slot
    table, the slot number. Measured on this box (2026-11-03), the four GPUs sit in slots 1, 3, 6 and 8, and the
    addresses are the CARDS' switch upstream ports, not the GPU endpoints -- so the join has to be against the
    chain, not against the GPU's own bus id:

        slot 1 -> 0000:01:00 (card3)   slot 3 -> 0000:c1:00 (card1)
        slot 6 -> 0000:c4:00 (card2)   slot 8 -> 0000:04:00 (card4)

    Readable without root here. Absent on boards whose firmware does not describe slots, in which case the page
    simply has no slot to show.
    """
    out = {}
    try:
        entries = sorted(Path(root).iterdir())
    except OSError:
        return out
    for d in entries:
        addr = _read_str(d / "address")
        if addr:
            out[addr.lower().strip()] = d.name           # '0000:c4:00' -> '6'
    return out


def slot_of(devs, slots):
    """The first node on a chain that the BIOS names a slot for, and which node it was."""
    for d in devs:
        bdf = (d.get("bdf") or "").lower()
        base = bdf.rsplit(".", 1)[0]                     # 0000:c4:00.0 -> 0000:c4:00
        if base in slots:
            return slots[base], d.get("bdf"), d.get("role")
    return None, None, None


def slot_key(slot, bus):
    """Sort key that means *chassis order*, not string order.

    Firmware slot names are not zero-padded and can carry a sub-slot: this box's table reads
    `1, 1-1, 2, 3, 3-1, 4, 6, 8, 10`. Plain `sorted()` puts slot 10 second, which is exactly the kind of
    quietly-wrong ordering a panel can carry for months. A card whose firmware names no slot sorts last, by
    BDF, so an unnamed card never jumps ahead of the chassis order a person is reading off the machine.
    """
    name = str(slot if slot is not None else "")
    m = re.match(r"(\d+)(?:[-.](\d+))?", name)
    if not m:
        return (1, 10 ** 9, 10 ** 9, name, str(bus or ""))
    return (0, int(m.group(1)), int(m.group(2) or 0), name, str(bus or ""))


def order_by_slot(gpus, chains):
    """Put the cards and their PCIe chains in physical-slot order and number them (1-based `pos`).

    Both identities stay on screen: `cardN` is the driver's name and the key that aligns sensor history with
    the chain rows, so it cannot simply be replaced; `pos` is the position in chassis order, which lets the
    page say "chassis #3" without inventing a third name for the same card. Sorting here rather than in the
    page means the API and every consumer (Cards panel, PCIe rows, About page) agree on one order.
    """
    gpus.sort(key=lambda g: slot_key(g.get("slot"), g.get("bus")))
    for i, g in enumerate(gpus, 1):
        g["pos"] = i
    chains.sort(key=lambda c: slot_key(c.get("slot"), c.get("bus")))
    for i, ch in enumerate(chains, 1):
        ch["pos"] = i
    return gpus, chains


def read_pcie(root):
    """One chain per GPU endpoint: the card, then each bridge above it, with link state and AER counters."""
    # /sys/bus/pci/slots and /sys/bus/pci/devices are siblings, so the cfg's devices root gives us both.
    slots = read_slots(Path(root).parent / "slots")
    root = Path(root)
    chains = []
    for dev in sorted(root.glob("*")):
        if not PCI_BDF.match(dev.name):
            continue
        try:
            if "amdgpu" not in (dev / "uevent").read_text():
                continue
        except OSError:
            continue
        # sysfs resolves a PCI device into its physical topology, so the parent of the resolved path is the
        # bridge the link hangs off -- walking it gives the whole chain without parsing lspci.
        chain = []
        real = Path(os.path.realpath(dev))
        while True:
            chain.append(_pci_node(root / real.name, real.name, len(chain)))
            if not PCI_BDF.match(real.parent.name):
                break
            real = real.parent
        lows = {k: [d[k] for d in chain if _num(d[k])] for k in ("speed", "width")}
        # Which physical slot this card is in, per the firmware's slot table. The name a person reads at the
        # machine, so it belongs on the row: "card2, slot 6" is something an operator can act on, "card2" is not.
        slot, slot_at, slot_role = slot_of(chain, slots)
        chains.append({"bus": dev.name.replace("0000:", ""), "card": None, "slot": slot,
                       "slot_at": slot_at, "slot_role": slot_role,
                       "devices": chain,
                       "speed": min(lows["speed"]) if lows["speed"] else None,
                       "width": min(lows["width"]) if lows["width"] else None,
                       "degraded": any(_num(d["speed"]) and _num(d["max_speed"]) and d["speed"] < d["max_speed"]
                                       or _num(d["width"]) and _num(d["max_width"]) and d["width"] < d["max_width"]
                                       for d in chain),
                       # A parked card explains a slow link without anything being wrong: with the GPUs runtime-
                       # power-suspended (D3hot -- the state this box sits in between launches) the switch
                       # upstream and root-port links drop to Gen1, all four chains read "below maximum", and
                       # the panel would scream danger at an idle box. Measured 2026-11-03 03:00: every chain
                       # 2.5 GT/s x16 at the upstream port, endpoints in D3hot, nothing running.
                       "parked": any((d.get("power_state") or "").upper().startswith("D3")
                                     for d in chain if d["role"] == "endpoint")})
    return chains


def _pci_node(path, bdf, depth):
    role = {0: "endpoint", 1: "downstream port", 2: "upstream port"}.get(depth, "bridge")
    cor, cor_named = _aer(path / AER_FILES["cor"])
    nf, nf_named = _aer(path / AER_FILES["nonfatal"])
    ft, ft_named = _aer(path / AER_FILES["fatal"])
    return {"bdf": bdf, "role": role,
            "speed": _pci_speed(path / "current_link_speed"), "max_speed": _pci_speed(path / "max_link_speed"),
            "width": _pci_num(path / "current_link_width"), "max_width": _pci_num(path / "max_link_width"),
            "power_state": _read_str(path / "power_state"),
            "cor": (cor or {}).get(AER_TOTAL["cor"]), "nonfatal": (nf or {}).get(AER_TOTAL["nonfatal"]),
            "fatal": (ft or {}).get(AER_TOTAL["fatal"]),
            "notable": {**cor_named, **{f"{k}#nf": v for k, v in nf_named.items()},
                        **{f"{k}#ft": v for k, v in ft_named.items()}}}


def aer_deltas(chains, last, named=None, events=None, at=None, last_err=None, recent=None, window_s=300.0):
    """Per-poll AER deltas per chain, from counters that are cumulative since boot.

    A counter that went DOWN is not negative traffic: the kernel zeroes these on a link recovery or a driver
    reload, so the window's delta is unknown (None) and the chain is flagged as reset -- drawing -24 as a bar
    would be worse than drawing a gap.

    When `named` and `events` are given, each individual counter that moved is also recorded as an event
    (`c4:00.0 BadTLP +3`, with the time). These counters are eventual, not continuous -- measured: 0 events in
    90 idle samples and 0 through an eight-stream burst, ~12 clustered around two serve restarts -- so a
    waveform of them is flat 99.9% of the time and an event with a timestamp is the whole point of watching.
    Diffing the non-zero counter names is enough: these counts only grow (apart from the zeroing above), so a
    change always lands in the non-zero set.

    `last_err` gives each chain a `quiet_s`: seconds since this watcher last saw a counter move. It is a watch
    clock, not the fault's clock -- a chain that already had counters when we started watching is seeded at the
    first sighting (the same reason the first sighting does not emit events: those errors happened before we
    were watching, so they are not ours to date). Measured on this box the distinction is the whole point: the
    c6 chain carried 42 correctables from a load window that ended at 09:00:03, and then sat at exactly 42 for
    the next 42 minutes with the link holding 32 GT/s -- a number that never moves should stop being painted as
    if it were happening now.

    `recent` adds `rate_per_min`: counter **events per minute averaged over the trailing `window_s`** (default
    300 s, the same span `quiet_s` uses to decide something just happened). A trailing average and not a
    lifetime one, because a lifetime average never relaxes. It is what lets a chain adding up to one counter a
    minute collapse to an info glyph while one adding tens a minute keeps its number: measured here a
    weight-load window put 34 counters on the c6 chain in 3.0 min = 11.3/min (digits, coloured), while the
    trickle this chain shows during real traffic -- one BadTLP per few minutes of large prefills -- is ~0.2/min
    (glyph). `window_s` travels with it so a page can name the span.
    """
    out, nxt = [], {}
    for ch in chains:
        d = {"bus": ch["bus"], "speed": ch["speed"], "width": ch["width"], "degraded": ch["degraded"],
             "parked": ch.get("parked")}
        reset = False
        for k in ("cor", "nonfatal", "fatal"):
            tot, acc = 0, 0
            for dev in ch["devices"]:
                v, prev = dev.get(k), last.get((dev["bdf"], k))
                nxt[(dev["bdf"], k)] = v
                if named is not None and events is not None:
                    cur = dev.get("notable") or {}
                    if dev["bdf"] in named:            # first sighting only sets a baseline: those events happened
                        was = named[dev["bdf"]]       # before we were watching, so they are not ours to report
                        for nm, nv in cur.items():
                            if nv > was.get(nm, 0):
                                events.append({"at": at, "bus": ch["bus"], "card": ch.get("card"),
                                               "bdf": dev["bdf"], "role": dev["role"],
                                               "kind": nm, "n": nv - was.get(nm, 0)})
                                if recent is not None:
                                    recent.setdefault(ch["bus"], deque()).append((at, nv - was.get(nm, 0)))
                    named[dev["bdf"]] = dict(cur)
                if v is None:
                    continue
                if prev is None:
                    acc = None if acc == 0 else acc          # first sighting: no delta for this window
                    continue
                if v < prev:
                    reset = True
                    acc = None
                    continue
                if acc is not None:
                    acc += v - prev
            d[f"{k}_d"] = acc
        d["reset"] = reset
        if last_err is not None:
            moved = reset or any(isinstance(d.get(f"{k}_d"), int) and d[f"{k}_d"] > 0
                                 for k in ("cor", "nonfatal", "fatal"))
            if moved or ch["bus"] not in last_err:
                last_err[ch["bus"]] = at or time.time()
            d["quiet_s"] = round((at or time.time()) - last_err[ch["bus"]], 1)
        if recent is not None:
            seen = recent.setdefault(ch["bus"], deque())
            now = at or time.time()
            while seen and now - seen[0][0] > window_s:
                seen.popleft()
            d["rate_per_min"] = round(sum(n for _, n in seen) / (window_s / 60.0), 3)
            d["window_s"] = window_s
        out.append(d)
    return out, nxt


# --------------------------------------------------------------------------------- the previous death
# A hard reset leaves almost nothing behind on this box, but not nothing: the firmware keeps one record from
# the crash in ACPI **BERT**, and the kernel replays it into the log at the START of the next boot. Measured
# across every boot in this machine's journal -- 18 deaths from 2026-11-01 20:36 to 2026-11-03 02:50 -- each
# boot carries exactly one `BERT: Total records found: 1`, and each record's own TIME lands 17-18 s before
# that boot, which is the BMC's reset-and-POST latency. So the deaths are not silent and not unrecorded: they
# are CPU machine checks. Every one of the 18 is Bank 5 with IPID 0x500b024029a00 (or ...b00), which the
# kernel's own table resolves to SMCA_EX, `execution_unit`; the CPU number is 17 or 113, and sysfs says
# 17 and 113 are SMT siblings of ONE physical core (core_id 65, package 0). That is the strongest clue this
# box has produced: one core's execution unit, 18 times, at both ends of the machine's idle and load range.
#
# The Monitor shows it because the alternative is worse: after a reboot the page comes back with a healthy
# green look and no memory of what killed the last launch.
SMCA_IPID_HWID = 0xFFF
SMCA_IPID_MCATYPE = 0xFFFF0000

# From arch/x86/kernel/cpu/mce/amd.c, smca_hwid_mcatypes: (hwid, mcatype) -> bank name. Only the blocks a
# machine-check record can plausibly name here are listed, and the point of listing them is that the page can
# say "execution unit" or "memory controller" instead of leaving the reader with a hex number.
SMCA_NAMES = {
    (0xB0, 0x00): "load-store unit", (0xB0, 0x01): "instruction fetch", (0xB0, 0x02): "L2 cache",
    (0xB0, 0x03): "decode", (0xB0, 0x05): "execution unit", (0xB0, 0x06): "floating point",
    (0xB0, 0x07): "L3 cache", (0xB0, 0x10): "load-store unit",
    (0x96, 0x00): "memory controller", (0x96, 0x01): "memory controller",
    (0x46, 0x00): "PCIe", (0x46, 0x01): "PCIe", (0x18, 0x00): "NBIO", (0x6C, 0x00): "NBIF",
    (0x80, 0x00): "system fabric", (0x50, 0x00): "xGMI PCS", (0x259, 0x00): "xGMI PHY",
    (0x01, 0x00): "SMU", (0xFF, 0x00): "PSP", (0xAA, 0x00): "USB", (0xA8, 0x00): "SATA",
}

MC_LINE = re.compile(r"Machine check: (\d+) Bank (\d+): ([0-9a-f]+)", re.I)
MC_CPU = re.compile(r"CPU (\d+):", re.I)
MC_IPID = re.compile(r"IPID ([0-9a-f]+)", re.I)
MC_PPIN = re.compile(r"PPIN ([0-9a-f]+) SYND ([0-9a-f]+)", re.I)
MC_PROC = re.compile(r"PROCESSOR (\S+) TIME (\d+) SOCKET (\d+) APIC (\d+)", re.I)
MC_ADDR = re.compile(r"ADDR ([0-9a-f]+)", re.I)
MC_BERT = re.compile(r"BERT: Total records found: (\d+)", re.I)
# The CPER header lines carry the human-level answer -- this box's records say `event severity: fatal` and
# `fru_text: ProcessorError`, which is a stronger statement than any decoded bit and comes straight from the
# firmware that saw it.
MC_SEV = re.compile(r"event severity:\s*(\w+)", re.I)
MC_FRU = re.compile(r"fru_text:\s*(\S+)", re.I)
MC_STATUS_UC = 1 << 61          # MCi_STATUS bit 61: uncorrectable


def smca_bank(ipid):
    """Name the SMCA bank an MCA_IPID points at, the way the kernel does."""
    hi = (ipid >> 32) & 0xFFFFFFFF
    key = (hi & SMCA_IPID_HWID, (hi & SMCA_IPID_MCATYPE) >> 16)
    return SMCA_NAMES.get(key) or f"bank hwid 0x{key[0]:x} type 0x{key[1]:x}"


def core_of(cpu, root="/sys/devices/system/cpu"):
    """Which physical core a logical CPU belongs to: SMT siblings plus the core id.

    This is what turns "CPU 17" and "CPU 113" into one suspect: sysfs says both are siblings of core_id 65,
    so 18 records over two days point at one physical core rather than at two, or at the platform.
    """
    try:
        sib = _read_str(Path(f"{root}/cpu{cpu}/topology/thread_siblings_list"))
        cid = _read_str(Path(f"{root}/cpu{cpu}/topology/core_id"))
    except OSError:
        sib = cid = None
    return {"cpu": cpu, "siblings": sib, "core_id": cid}


def parse_death(lines):
    """Pull the boot-replayed error record out of kernel log lines (oldest fields win)."""
    out = {}
    for line in lines:
        # The record arrives as a block of separate lines with no common prefix: `mce:` for the CPU line,
        # `[Hardware Error]:` for the CPER fields, bare `PPIN ... IPID ...` for the identity of the failing
        # block. Gating on `mce:` alone silently loses the IPID, which is the whole decode.
        if not any(k in line for k in ("mce:", "Hardware Error", "BERT", "IPID", "PPIN", "fru_text")):
            continue
        m = MC_BERT.search(line)
        if m:
            out["bert_records"] = int(m.group(1))
            continue
        m = MC_SEV.search(line)
        if m and "severity" not in out:
            out["severity"] = m.group(1)
            continue
        m = MC_FRU.search(line)
        if m and "fru" not in out:
            out["fru"] = m.group(1)
            continue
        m = MC_LINE.search(line)
        if m and "status" not in out:
            out["bank"] = int(m.group(2))
            out["status"] = int(m.group(3), 16)
            out["uncorrectable"] = bool(out["status"] & MC_STATUS_UC)
            c = MC_CPU.search(line)
            if c:
                out["cpu"] = int(c.group(1))
            continue
        m = MC_IPID.search(line)
        if m and "ipid" not in out:
            out["ipid"] = int(m.group(1), 16)
            out["bank_name"] = smca_bank(out["ipid"])
            continue
        m = MC_PPIN.search(line)
        if m:
            out["ppin"] = m.group(1)
            out["synd"] = m.group(2)
            a = MC_ADDR.search(line)
            if a:
                out["addr"] = a.group(1)
            continue
        m = MC_PROC.search(line)
        if m:
            out["at"] = int(m.group(2))
            out["socket"] = int(m.group(3))
            out["apic"] = int(m.group(4))
            out["cpuid"] = m.group(1)
            mc = re.search(r"microcode (\S+)", line)
            if mc:
                out["microcode"] = mc.group(1)
            continue
        m = MC_ADDR.search(line)
        if m and "addr" not in out:
            out["addr"] = m.group(1)
    if "at" not in out:
        return None
    if out.get("cpu") is not None:
        out["core"] = core_of(out["cpu"])
    return out


def launch_hint(meta):
    """The launch command for THIS launch, from the meta file the launch itself writes.

    The Monitor is the thing still standing when the serve is not, so it is where the command should be. Built
    from the recorded parameters rather than a hardcoded recipe, because the recipe is what the launch actually
    used: maxlen / nseq / nbt / prefix_cache come from the meta, and the shell script is the entry point people
    are meant to use (it is the one that checks for leftover ranks holding the GPUs).
    """
    if not meta:
        return None
    parts = []
    for key, env in (("maxlen", "MAXLEN"), ("nseq", "NSEQ"), ("nbt", "NBT"),
                     ("prefix_cache", "PREFIX_CACHE")):
        if meta.get(key) not in (None, ""):
            parts.append(f"{env}={meta[key]}")
    return " ".join(parts + ["WEBUI=same", "bash", "serve/native-flashnext.sh"]) or None


def boot_ids(limit=12):
    """The boots the journal still holds, newest last, capped: `-b all` is not an option here -- measured on
    this box it scans 517 MB of journal and takes 55 s, while one boot reads in 0.4 s."""
    try:
        out = subprocess.run(["journalctl", "--list-boots", "--no-pager"],
                             capture_output=True, text=True, timeout=10.0).stdout
    except (OSError, ValueError, subprocess.SubprocessError):
        return []
    ids = [l.split()[0] for l in out.splitlines() if l.split() and l.split()[0].lstrip("-").isdigit()]
    return ids[-limit:]


def boot_log(bid):
    try:
        out = subprocess.run(["journalctl", "-k", "-b", bid, "--no-pager", "-o", "short-iso"],
                             capture_output=True, text=True, timeout=10.0).stdout
    except (OSError, ValueError, subprocess.SubprocessError):
        return []
    return out.splitlines()


def split_records(lines):
    """Cut a kernel log into its boot-error record blocks (one per previous death)."""
    chunks, cur = [], []
    for line in lines:
        if "Error records from previous boot" in line:
            if cur:
                chunks.append(cur)
            cur = [line]
        elif cur:
            cur.append(line)
    if cur:
        chunks.append(cur)
    return chunks


def death_history(lines=None, boots=12):
    """Every previous death the journal can still name, as a count plus who each one blamed.

    Worth one pass at startup (in a thread, ~0.3 s per boot): on this box the answer is the whole argument --
    18 records across the boots the journal holds (2026-11-01 20:36 to 2026-11-03 02:50), and every one of
    them names the same bank and the same physical core. A count with the named block beside it is what
    separates a box that dies of the platform from a box that dies of one bad core.
    """
    logs = [lines] if lines is not None else [boot_log(b) for b in boot_ids(boots)]
    recs = [r for log in logs for r in (parse_death(c) for c in split_records(log)) if r]
    if not recs:
        return None
    named, cores = {}, {}
    for r in recs:
        named[r.get("bank_name") or f"bank {r.get('bank')}"] = named.get(r.get("bank_name") or
                                                                        f"bank {r.get('bank')}", 0) + 1
        c = (r.get("core") or {}).get("core_id")
        if c is not None:
            cores[f"core {c}"] = cores.get(f"core {c}", 0) + 1
    return {"records": len(recs), "first_at": min(r["at"] for r in recs),
            "last_at": max(r["at"] for r in recs), "named": named, "cores": cores,
            "severities": {s: sum(1 for r in recs if r.get("severity") == s)
                           for s in {r.get("severity") for r in recs} if s},
            "boots_scanned": len(logs)}


def read_death(lines=None):
    """The record the firmware kept from the crash that ended the PREVIOUS boot.

    Read from the CURRENT boot's kernel log, because that is where the replay lands. `journalctl -k` is
    readable without root here (verified 2026-11-03); where it is not, this returns None and the page simply
    has no death line.
    """
    if lines is None:
        try:
            out = subprocess.run(["journalctl", "-k", "-b", "0", "--no-pager", "-o", "short-iso"],
                                 capture_output=True, text=True, timeout=6.0).stdout
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
        if not out:
            return None
        lines = out.splitlines()
    try:
        return parse_death(lines)
    except Exception:                                    # noqa: BLE001 - a page must not die on a log
        return None


# --------------------------------------------------------------------------------- request record
# What a row in the request list carries, in one place because two different backends build rows: the
# relay (`Records`, which sees bytes) and the engine's own stat logger (`EngineRecords`, which sees the
# queue/prefill/decode split). If a field the cards read lives only in the detail, the card silently falls
# back to something else while its label still describes the field it lost: measured on the live merged
# page, "Last decode speed ... One stream" showed 27.4 tok/s -- the engine-wide 1/step from an 36.4 ms
# inter-token interval -- where that request's own 87.8 tok/s belonged. Same for prefill: the row and the
# cards had no per-request prefill to show, only the window mean.
ROW_KEYS = ("id", "seq", "path", "model", "state", "started_at", "wallclock_s", "first_token_s",
            "load_s", "queue_s", "stream", "http_status", "response_format", "usage", "timings", "boot")


def shared_decode(rows):
    """Mean number of requests in flight across each row's OWN decode window, itself included.

    Why this rides the payload: measured off 1,400 real requests, a stream's decode rate is 158 tok/s
    alone, 119 at two in flight and 23 at three or four ([notes/context-262k.md](../notes/context-262k.md)). A
    decode number with no concurrency beside it is therefore unreadable -- the reader cannot tell a slow
    request from a busy box. It is derived, so it is computed at read time and never journalled: ROW_KEYS
    decides what reaches disk, and this is not in them.

    Mean over the window, not the value at its midpoint. Midpoint sampling calls a request "alone" whenever
    nothing happened to be in flight at that instant, which on this journal reported 361 alone requests at 137
    tok/s where never-shared gives 158.

    The window is the request's decode span, anchored at its END minus `decode_s` -- see the comment in the
    loop for why the two monitor paths do not share a clock origin.

    A row with no timings yet (a proxy row still in flight) has no span to place and contributes nothing: the
    merged path only ever reports finished requests, and guessing an end for a running one is how a live count
    starts to lie.
    """
    span = []
    for r in rows:
        s, w = r.get("started_at"), r.get("wallclock_s")
        if not isinstance(s, (int, float)) or not isinstance(w, (int, float)):
            span.append(None)
            continue
        d1 = s + w
        # End-anchored on purpose: the two paths do not share a clock origin. A merged (engine) row starts at
        # the moment INFERENCE began and its wallclock_s is inference_time, while its first_token_s counts
        # queued_time + prefill_time -- so `started_at + first_token_s` can land AFTER `started_at +
        # wallclock_s` (a real row: first token 21.8 s, wallclock 10.8 s, because it queued 12.3 s before the
        # engine began). decode_s is a duration, so subtracting it from the end is right on both paths; a
        # proxy row's started_at is the arrival and its wallclock_s is the whole connection.
        dec = (r.get("timings") or {}).get("decode_s")
        d0 = (d1 - dec) if isinstance(dec, (int, float)) and 0 < dec <= w \
            else s + (r.get("first_token_s") or 0)
        span.append((min(d0, d1), d1))
    timed = [x for x in span if x is not None]             # a row still in flight has no timings to place
    ev = sorted([(d0, 1) for d0, _ in timed] + [(d1, -1) for _, d1 in timed])
    out = []
    for pair in span:
        if pair is None:
            out.append(None)
            continue
        d0, d1 = pair
        acc, cur, prev = 0.0, 0, None
        for t, delta in ev:
            if prev is not None and t > prev:                 # `cur` is the count across (prev, t)
                a, b = max(prev, d0), min(t, d1)
                if b > a:
                    acc += cur * (b - a)
            cur += delta
            prev = t
            if t >= d1:
                break
        out.append(round(acc / (d1 - d0), 2) if d1 > d0 else 1.0)
    return out


class Journal:
    """Append-only JSONL of finished request rows, so the list survives what this box does.

    The request list lives in the serve's memory, and this box dies without writing a shutdown record -- 15
    boots on 2026-11-02, several of them mid-request -- so after a hard reset the Monitor came back showing
    nothing, and "what was it doing when it died" had to be answered from side logs written by hand. One line
    per finished request closes that: the page can carry a dead launch's last rows across the reset.

    Three choices worth their reasons:

    - **fsync per row.** A write to the page cache survives a crash but not a hard reset, and surviving hard
      resets is the entire purpose. The cost is one fsync per finished request -- requests end on the scale of
      seconds apart, and it is measured in tests/webui rather than assumed.
    - **No bodies.** The engine records never had text, the relay's are capped and large, and the incident
      question is about timing and tokens, not content. Only ROW_KEYS go to disk.
    - **A boot id per process.** A list that spans a reset is two launches, and it should read that way rather
      than as one afternoon with strange gaps in it.
    """

    MAX_BYTES = 8 << 20                                     # rotate well before a page read gets slow

    def __init__(self, path, keep=100, readonly=False):
        self.path, self.keep, self.readonly = Path(path), keep, readonly
        self.lock = threading.Lock()
        self.boot = f"{int(time.time())}-{os.getpid()}"
        self.f, self.error, self.rows_in = None, None, 0
        self.start()

    def start(self):
        if self.readonly:
            # Attached, not owned: this Monitor reads a journal another process writes. No append handle, no
            # fsync, no rotation -- two writers in one file is how two Monitors end up replaying each other's
            # rows, which is the reason the journal is one-per-port in the first place. What attaching buys is
            # the case that matters: the serve is dead, and its request history is still worth looking at.
            self.f = None
            try:
                self.path.open(encoding="utf-8").close()
            except OSError as e:
                self.error = f"{type(e).__name__}: {e}"
            return self
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.f = self.path.open("a", encoding="utf-8")
        except OSError as e:
            # A journal that cannot be written is a degraded Monitor, never a broken serve.
            self.f, self.error = None, f"{type(e).__name__}: {e}"
        return self

    def append(self, row):
        if self.f is None or self.readonly:
            return False
        try:
            with self.lock:
                row["boot"] = self.boot                     # the row in memory knows its launch too, so a list
                self.f.write(json.dumps({k: row.get(k) for k in ROW_KEYS},
                                        ensure_ascii=False, default=str) + "\n")
                self.f.flush()
                os.fsync(self.f.fileno())
                if self.f.tell() > self.MAX_BYTES:
                    self._compact()
            return True
        except OSError as e:
            self.error = f"{type(e).__name__}: {e}"
            return False

    def _compact(self):
        """Rewrite the file keeping only the last `keep` rows (caller holds the lock)."""
        rows, path = self.replay(), self.path
        self.f.close()
        try:
            with path.open("w", encoding="utf-8") as f:
                for r in reversed(rows):                    # oldest first on disk, newest at the tail
                    f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError as e:
            self.error = f"compact {type(e).__name__}: {e}"
        self.f = path.open("a", encoding="utf-8")

    def replay(self):
        """Rows from earlier launches, newest first, at most `keep` of them.

        Read from the tail backwards: a hard reset can cut the last write in half, and a line that does not
        parse is dropped rather than trusted -- a half-row that renders as a request is worse than no row."""
        try:
            raw = self.path.read_text(errors="replace").splitlines()
        except OSError:
            return []
        rows = []
        for ln in reversed(raw[-(self.keep + 8) * 4:]):
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except ValueError:
                continue                                    # the torn line
            if isinstance(r, dict) and r.get("seq") is not None:
                rows.append(r)
                if len(rows) >= self.keep:
                    break
        return rows

    def info(self):
        return {"path": str(self.path), "ok": self.error is None if self.readonly else self.f is not None,
                "error": self.error, "restored": self.rows_in, "boot": self.boot,
                # Attached rows belong to another process's launch list: the page says so rather than letting
                # them read as this Monitor's own record of a live launch.
                "readonly": self.readonly}


class EngineRecords:
    """Per-request records taken from the engine itself, not from a relay.

    vLLM 0.32 hands a stat logger (`vllm.stat_logger_plugins`) an `IterationStats` whose
    `finished_requests` list carries, per request: prompt / generation tokens, queued, prefill,
    inference and decode seconds, mean time per output token, cached tokens, max_tokens and the finish
    reason (vllm/v1/metrics/stats.py:288-306). That is a BETTER record than the relay can build -- the
    queue/prefill/decode split is what the engine measured, where the proxy can only see bytes and infer
    -- with two gaps it cannot close: it has no request text, and it only sees a request when it FINISHES,
    so the list has no in-flight rows (the counts above it do, from the gauges).

    Keys are the ones the page already reads from `Records`, so one renderer serves both.
    """

    def __init__(self, keep=100, journal=None):
        self.keep, self.items, self.n, self.lock = keep, deque(), 0, threading.Lock()
        # Carried over from the launches before this one, if a journal is configured: on a box that hard resets
        # without a shutdown record, the rows that died are the ones worth still being able to read.
        self.journal = journal
        if journal is not None:
            rows = journal.replay()
            journal.rows_in = len(rows)
            self.items.extend(rows)                       # newest first, which is how replay returns them
            self.n = max([r.get("seq") or 0 for r in rows], default=0)

    def add(self, f, now=None):
        """One FinishedRequestStats (duck-typed: this file must import without vllm installed)."""
        now = now if now is not None else time.time()
        q, pre, inf, dec = (float(getattr(f, "queued_time", 0) or 0), float(getattr(f, "prefill_time", 0) or 0),
                            float(getattr(f, "inference_time", 0) or 0), float(getattr(f, "decode_time", 0) or 0))
        gen = int(getattr(f, "num_generation_tokens", 0) or 0)
        prompt, cached = int(getattr(f, "num_prompt_tokens", 0) or 0), int(getattr(f, "num_cached_tokens", 0) or 0)
        # Prefill speed for THIS request, computed at the moment the record is made. The engine hands over both
        # the tokens and the seconds, so unlike the "prompt tok/s" card (a counter credited at request end, read
        # as a rate) this is a measured per-request mean. Two numerators matter: with a prefix-cache hit the
        # engine computed `prompt - cached`, so prompt/prefill would report a speed it never earned.
        computed = prompt - cached
        pre_all = (prompt / pre) if prompt and pre > 0 else None
        pre_comp = (computed / pre) if computed > 0 and pre > 0 else None
        pre_rate, pre_basis = ((pre_comp, f"{computed} computed tokens / prefill time ({cached} cached, not read)")
                               if cached > 0 and pre_comp else
                               (pre_all, "prompt tokens / prefill time"))
        e2e = inf if inf > 0 else (q + pre + dec)
        with self.lock:
            self.n += 1
            r = {"id": getattr(f, "request_id", None) or f"eng-{self.n}", "seq": self.n, "path": None,
                 "model": None, "state": "completed",
                 "started_at": now - e2e, "finished_at": now,
                 "wallclock_s": e2e, "first_token_s": q + pre, "queue_s": q,
                 "stream": None, "response_format": "engine", "http_status": None,
                 "usage": {"prompt_tokens": prompt,
                           "completion_tokens": gen,
                           "prompt_tokens_details": {"cached_tokens": cached}},
                 "timings": {"predicted_per_second": (gen / dec) if gen and dec > 0 else None,
                             "speed_basis": "per accepted token (the engine's own mean)",
                             "tokens_are_estimate": False,
                             "prefill_tok_s": pre_rate, "prefill_basis": pre_basis,
                             "queue_s": q, "prefill_s": pre, "decode_s": dec,
                             "preemptions": int(getattr(f, "num_preemptions", 0) or 0),
                             "max_tokens_param": getattr(f, "max_tokens_param", None),
                             "finish_reason": getattr(f, "finish_reason", None)},
                 "error": None}
            self.items.appendleft(r)
            while len(self.items) > self.keep:
                self.items.pop()
            if self.journal is not None:
                self.journal.append(r)
            return r

    def listing(self):
        with self.lock:
            rows = [{k: r.get(k) for k in ROW_KEYS} for r in self.items]
        for r, n in zip(rows, shared_decode(rows)):          # derived, so never journalled
            r["shared_decode"] = n
        return rows

    def get(self, rid):
        with self.lock:
            for r in self.items:
                if rid in (r["id"], f"req-{r['seq']}"):
                    return dict(r, source="engine", input=None, output=None, response=None, reasoning=None,
                                input_note="The engine reports timings and token counts per request, not the "
                                           "text: this monitor runs inside the serve (serve/plugin), it never "
                                           "sits between the client and the model. Captured bodies need the "
                                           "separate sidecar: serve/webui.sh with WEBUI_MODE=proxy.")
        return None

    def __len__(self):
        with self.lock:
            return len(self.items)


def web_asset(web, rel):
    """Resolve a /web/<rel> request to a file under `web`, or None. A path COMPONENT check, not a prefix one:
    serve/webui.py starts with "serve/web", so a plain startswith() would happily serve the sidecar's own
    source to anyone who asks. Shared by the sidecar and the in-serve plugin so the guard is one function."""
    p = (Path(web) / rel).resolve()
    root = Path(web).resolve()
    if p != root and not str(p).startswith(str(root) + os.sep):
        return None
    return p if p.is_file() else None


class Records:
    """The last --keep requests, newest first, kept in a lock so the poll thread and the page can
    read them while a request is still filling its own entry in."""

    def __init__(self, keep=100, journal=None):
        self.keep, self.items, self.n, self.lock = keep, deque(), 0, threading.Lock()
        self.journal = journal
        if journal is not None:
            rows = journal.replay()
            journal.rows_in = len(rows)
            self.items.extend(rows)
            self.n = max([r.get("seq") or 0 for r in rows], default=0)

    def start(self, path, model, body, stream):
        with self.lock:
            self.n += 1
            r = {"id": None, "seq": self.n, "path": path, "model": model, "state": "queued",
                 "started_at": time.time(), "wallclock_s": None, "first_token_s": None, "load_s": None,
                 "queue_s": None, "stream": stream, "response_format": "stream" if stream else "json",
                 "http_status": None, "usage": None, "timings": None, "error": None,
                 "_in": body[:CAP], "_in_truncated": len(body) > CAP, "_out": [], "_out_len": 0,
                 "_resp": [], "_reason": [], "tokens_est": 0}
            self.items.appendleft(r)
            while len(self.items) > self.keep:
                self.items.pop()
            return r

    def finish(self, r, state, status=None, error=None):
        if status is not None:
            r["http_status"] = status
        if error and not r["error"]:
            r["error"] = {"message": str(error)[:2000]}
        if r.get("_closed"):                            # the pump closed it; the relay has nothing to add
            return
        r["state"] = state
        if state in ("completed", "error", "disconnected"):
            r["_closed"] = True
            if self.journal is not None:
                self.journal.append(r)
            r["wallclock_s"] = time.time() - r["started_at"]
            gen = (r["usage"] or {}).get("completion_tokens") or r["tokens_est"] or None
            span = max(r["wallclock_s"] - (r["first_token_s"] or 0.0), 0.0) or r["wallclock_s"]
            r["timings"] = {"predicted_per_second": (gen / span) if gen and span > 0 else None,
                            "tokens_are_estimate": not bool((r["usage"] or {}).get("completion_tokens"))}
            # Same per-request prefill mean as the engine-side record, built from what a relay can see: the
            # prompt token count the response reports and the wait for the first token. Labelled as this page's
            # own measurement (it includes anything the engine queued before starting), never presented as the
            # engine's.
            ptok = (r["usage"] or {}).get("prompt_tokens")
            pspan = (r["first_token_s"] - (r.get("queue_s") or 0.0)) if r.get("first_token_s") else None
            r["timings"]["prefill_tok_s"] = (ptok / pspan) if ptok and pspan and pspan > 0 else None
            r["timings"]["prefill_basis"] = ("prompt tokens over first-token latency, timed through this page"
                                             if r["timings"]["prefill_tok_s"] else None)

    def note(self, r, text):
        if r["_out_len"] < CAP:
            r["_out"].append(text[:CAP - r["_out_len"]])
            r["_out_len"] += len(text)

    def note_reason(self, r, text):
        if len(r["_reason"]) < CAP:
            r["_reason"].append(text[:CAP - len(r["_reason"])])

    def note_resp(self, r, chunk):
        if sum(map(len, r["_resp"])) < CAP:
            r["_resp"].append(chunk[:CAP - sum(map(len, r["_resp"]))])

    def listing(self):
        with self.lock:
            rows = [{k: r.get(k) for k in ROW_KEYS}
                    | {"id": r["id"] or f"req-{r['seq']}"} for r in self.items]
        for r, n in zip(rows, shared_decode(rows)):          # derived, so never journalled
            r["shared_decode"] = n
        return rows

    def get(self, rid):
        with self.lock:
            for r in self.items:
                if rid in (r["id"], f"req-{r['seq']}"):
                    if "_in" not in r:                    # a row carried over from an earlier launch: the
                        out = {k: r.get(k) for k in ("id", "seq", "path", "model", "state", "started_at",
                                                     "wallclock_s", "first_token_s", "load_s", "queue_s",
                                                     "stream", "http_status", "response_format", "usage",
                                                     "timings", "error")}
                        out["id"] = r["id"] or f"req-{r['seq']}"
                        out["input"] = out["output"] = out["response"] = out["reasoning"] = None
                        out["input_note"] = ("This row was restored from the request journal "
                                             f"(launch {r.get('boot')}), which keeps timings and token counts "
                                             "but never the text.")
                        return out
                    out = {k: r.get(k) for k in ("id", "seq", "path", "model", "state", "started_at",
                                                 "wallclock_s", "first_token_s", "load_s", "queue_s",
                                                 "stream", "http_status", "response_format", "usage",
                                                 "timings", "error")}
                    out["id"] = r["id"] or f"req-{r['seq']}"
                    out["input"] = r["_in"] + ("…[truncated]" if r["_in_truncated"] else "")
                    out["output"] = "".join(r["_out"])
                    out["reasoning"] = "".join(r["_reason"]) or None
                    out["response"] = b"".join(r["_resp"]).decode("utf-8", "replace")
                    out["output_truncated"] = r["_out_len"] >= CAP
                    return out
        return None


# ---------------------------------------------------------------------------------------- the sidecar
def stack_info():
    """Which code the serve behind these pages is running: the plugin's version and the commit of the tree it was
    loaded from.

    Read once at startup rather than on every poll: it is a fact about this process, and a header that says
    "vLLM 0.32 - plugin 0.4.0 - b85a496" answers "is the live one the new one?" from a screenshot, which is the
    question that otherwise costs an ssh and a look at the launch log. Fields are simply absent when the plugin is
    not importable (a sidecar running outside the serve's venv) or the code was installed rather than checked out
    (site-packages has no .git); the pages print what is there and drop what is not.

    One wrinkle worth naming: the static files are read per request, so a CSS or JS edit lands in the page before
    this line moves. `stack` describes the tree the process IMPORTED when it started -- a watcher of the tree would
    be a different fact, and a cheaper but less useful one.
    """
    out = {}
    try:
        import r9700_vllm
        out["plugin"] = getattr(r9700_vllm, "__version__", "") or ""
        root = Path(getattr(r9700_vllm, "__file__", "")).resolve().parents[1]
    except Exception:
        return out
    git = root / ".git"
    if not git.is_dir():
        return out
    try:
        head = (git / "HEAD").read_text().strip()
        if head.startswith("ref: "):
            out["branch"] = head[5:].split("/")[-1]         # refs/heads/master -> master: the page shows it
            ref = git / head[5:]
            commit = ref.read_text().strip() if ref.exists() else ""
        else:
            commit = head                                   # detached HEAD: the sha is in the file
        if not commit:                                      # packed refs
            commit = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                                    capture_output=True, text=True, timeout=5).stdout.strip()
        if commit:
            out["commit"] = commit[:8]
            out["dirty"] = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                                               capture_output=True, text=True, timeout=10).stdout.strip())
    except Exception as e:
        out["commit_error"] = str(e)[:120]
    return out


class Hub:
    def __init__(self, cfg):
        self.cfg = cfg
        cfg.setdefault("stack", stack_info())
        # The request journal: a JSONL file of finished rows, replayed at startup, so the list survives what
        # this box does to a process (see Journal). Off unless a path is configured.
        # A sidecar may ATTACH the merged monitor's journal read-only instead of owning one: that is what keeps
        # the dead launch's rows on screen while the serve itself is down (the incident case).
        jpath = cfg.get("journal") or cfg.get("journal_attach")
        self.journal = Journal(jpath, cfg["keep"], readonly=not cfg.get("journal")) if jpath else None
        self.rec = Records(cfg["keep"], journal=self.journal)
        # When the monitor runs INSIDE the serve (serve/plugin), the per-request records come from the
        # engine's own stat logger instead of a relay, and /metrics is read from the process's Prometheus
        # registry instead of over a socket it happens to be serving itself.
        self.eng = cfg.get("engine_records")
        self.status = {"mode": cfg["mode"], "at": 0.0, "reachable": False, "upstream": cfg["upstream"],
                       "live": {}, "rate": {}, "history": [], "requests_kept": 0, "models": []}
        self.gpus, self.models, self.meta = [], [], {}
        self.prev, self.prev_t, self.hist = {}, 0.0, deque(maxlen=HIST)
        # The trailing-average ring: cumulative counters sampled every AVG_STEP seconds, so a 1 h mean can be an
        # exact delta of two cumulative reads instead of an average of 3,600 one-second guesses.
        self.avgring = deque(maxlen=AVG_KEEP)
        # The hour-long version of the four waveforms (see SLOW_STEP): a point per interval, each holding the
        # counters it needs to difference against for the next one.
        self.slowhist = deque(maxlen=SLOW_KEEP)
        # Per-card history, sampled by gpu_poll: one entry per gpu poll with one slot per card, aligned by
        # index (read_gpus sorts by card name, and the card set does not change under us). VRAM in particular
        # is a number people ask us to prove by hand -- the soak runs log it to a side file to show a leak is
        # not happening -- and a ring on the page is the same evidence, continuously, with no script.
        self.gpuhist = deque(maxlen=HIST)
        # PCIe chain state and its ring, same beat as the cards. The waves are per-poll deltas of counters that
        # are cumulative since boot, which is why the sidecar keeps the last totals to difference against.
        self.pcie, self.pciehist, self.aer_last = [], deque(maxlen=HIST), {}
        # AER events with timestamps: the counters are eventual, so the interesting object is the event, not
        # the curve. Kept newest-first for the page, and bounded -- a day of a misbehaving link must not grow
        # the payload.
        self.aer_named, self.pcie_events = {}, deque(maxlen=60)
        self.aer_quiet = {}            # chain bus -> when a counter last moved (drives quiet_s)
        self.aer_recent = {}           # chain bus -> (time, n) events in the trailing window (drives rate_per_min)
        self.health, self.engine, self.host = None, {}, read_host()
        # What ended the PREVIOUS boot (ACPI BERT, replayed by the kernel at startup) and how many such
        # records the journal still holds. Read once: neither can change while this process is alive.
        self.slotmap = []
        self.death, self.deaths = read_death(), None
        self.launch = None                                  # set below, once the meta file has been read
        self.key_needed, self.smi_error = False, None
        self.lock = threading.Lock()
        if self.death is not None:
            # The count costs one kernel-log read per boot the journal holds, so it runs in a thread: the page
            # comes up at once and gains "18 deaths, all naming the same block" a second or two later.
            threading.Thread(target=self._read_deaths, name="r9700-webui-deaths", daemon=True).start()
        if cfg["meta"]:
            try:
                self.meta = json.loads(Path(cfg["meta"]).read_text())
            except (OSError, ValueError) as e:
                print(f"webui: --meta {cfg['meta']} not read: {e}", file=sys.stderr)
        if self.meta:
            # The command that would bring THIS launch back, assembled from the parameters the launch recorded
            # for itself rather than retyped by whoever is standing at the machine at 03:00. It has to be built
            # after the meta read: built before it, the dict was empty and the page had no command to show
            # (measured: `launch: None` on the sidecar's first run with --meta passed).
            self.launch = launch_hint(self.meta)

    def _read_deaths(self):
        """The death count, off the startup path (see Hub.__init__)."""
        hist = death_history()
        with self.lock:
            self.deaths = hist

    # ---- upstream reads (observe mode's whole footprint) -----------------------------------------
    def get(self, path, timeout=4.0):
        hdrs = {"Authorization": f"Bearer {self.cfg['key']}"} if self.cfg["key"] else {}
        req = urllib.request.Request(self.cfg["upstream"] + path, headers=hdrs)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()

    def metrics_text(self):
        fn = self.cfg.get("metrics_fn")
        if fn:
            return fn()
        return self.get("/metrics")[1].decode("utf-8", "replace")

    def kept(self):
        return len(self.eng) if self.eng is not None else len(self.rec.items)

    def record_detail(self, rid):
        """One request's detail, engine records first: in merged mode there are no relay records."""
        if self.eng is not None:
            return self.eng.get(rid)
        return self.rec.get(rid)

    def slow_point(self, cur, cap, t0):
        """One point of the hour-long waveforms.

        The rate is the interval's, not the instant's: generated and prompt tokens differenced against the
        previous point, KV growth the same way. A 1 s window sampled once every 15 s would show the gaps between
        the beats instead of the trend -- which is the exact complaint this whole change is about. The per-stream
        rate divides by the mean running across the interval, taken from the fast ring (which has a sample every
        poll), not by whatever this poll happened to see.
        """
        base = self.slowhist[-1] if self.slowhist else None
        dt = (t0 - base["at"]) if base and t0 > base["at"] else None
        kvn = next((n for n in KV_USED if cur.get(n)), None)
        gnow = (cur.get("generation_tokens_total") or [0.0, 0.0, 0])[0]
        pnow = (cur.get("prompt_tokens_total") or [0.0, 0.0, 0])[0]
        kvnow = (cap * cur[kvn][1]) if (cap and kvn) else None
        inow = (cur.get("iteration_tokens_total_count") or [0.0, 0.0, 0])[0]
        runs = [h.get("running") for h in self.hist if h.get("at", 0) > (base["at"] if base else 0)]
        runs = [r for r in runs if isinstance(r, (int, float))]
        runm = (sum(runs) / len(runs)) if runs else (cur.get("num_requests_running") or [0, 0.0, 0])[1]
        gen = (gnow - base["c"]["gen"]) / dt if (base and dt) else None
        prompt = (pnow - base["c"]["prompt"]) / dt if (base and dt) else None
        kvrate = ((kvnow - base["c"]["kv"]) / dt) if (base and dt and kvnow is not None
                                                     and base["c"]["kv"] is not None) else None
        # A counter that went backwards is an engine restart inside the interval: the difference is not a rate,
        # and a rate of -100 million tok/s is a number nobody can act on. None, so the waveform draws a gap.
        # The KV-growth series is NOT guarded that way: a negative there is the pool draining as requests
        # finish, which is what that series is for.
        gen = None if (gen is not None and gen < 0) else gen
        prompt = None if (prompt is not None and prompt < 0) else prompt
        return {"at": t0,
                "gen_tok_s": gen,
                "prompt_tok_s": prompt,
                "stream_tok_s": (gen / runm) if (gen is not None and runm) else None,
                "kv_tok_s": kvrate,
                # generated tokens per iteration record over the interval -- a count ratio, so it survives a
                # restart-free interval the way the rates do not, and it is a total, not a per-stream mean
                "gen_per_step": ((gnow - base["c"]["gen"]) / (inow - base["c"]["iter"]))
                                if (base and base["c"].get("iter") is not None
                                    and (inow - base["c"]["iter"]) > 0) else None,
                "kv_used_pct": (100.0 * cur[kvn][1]) if kvn else None,
                "running": runm,
                "c": {"gen": gnow, "prompt": pnow, "kv": kvnow, "iter": inow}}

    def poll(self):
        while True:
            t0 = time.time()
            try:
                cur, cap, info = parse_metrics(self.metrics_text())
                dt = (t0 - self.prev_t) if self.prev_t else None
                with self.lock:
                    payload = derive(cur, self.prev, dt, cap, info, self.cfg["mode"],
                                     self.kept(), list(self.hist))
                    # Means over 1 min / 15 min / 1 h next to the 1 s window's rate: the window answers "right
                    # now", and on a launch with six streams and 90 k-token prompts one poll is a coin toss.
                    # Sampled on its own beat, not per poll -- these spans are minutes wide, and a 1 s ring for
                    # an hour is 720 points of noise to average.
                    if not self.avgring or t0 - self.avgring[-1]["at"] >= AVG_STEP:
                        self.avgring.append(avg_sample(cur, t0))
                    payload["avg"] = trailing(list(self.avgring), cur, t0)
                    if not self.slowhist or t0 - self.slowhist[-1]["at"] >= SLOW_STEP:
                        self.slowhist.append(self.slow_point(cur, cap, t0))
                    payload["history_slow"] = list(self.slowhist)
                    payload["slow_step_s"] = SLOW_STEP
                    payload["upstream"] = self.cfg["upstream"]
                    payload["reachable"] = True
                    payload["health"] = self.health
                    payload["models"] = self.models
                    payload["engine"] = self.engine
                    payload["gpus"] = self.gpus
                    payload["gpus_history"] = list(self.gpuhist)
                    payload["pcie"] = self.pcie
                    payload["pcie_history"] = list(self.pciehist)
                    payload["pcie_events"] = list(self.pcie_events)
                    payload["smi"] = bool(self.cfg["smi"])
                    payload["smi_error"] = self.smi_error
                    payload["meta"] = self.meta
                    payload["meta_src"] = self.cfg["meta"] or None
                    payload["poll_s"] = self.cfg["poll"]
                    payload["key_needed"] = self.key_needed or bool(self.cfg["key"])
                    payload["merged"] = self.eng is not None     # `is not None`, not bool(): an empty
                                                                  # EngineRecords is falsy by __len__
                    # Whether the list in front of the reader extends back past this process, and where that
                    # came from -- after a hard reset, the rows on screen are evidence from the dead launch.
                    j = getattr(self.eng, "journal", None) or self.journal
                    payload["journal"] = j.info() if j else None
                    # The last thing the firmware recorded about the death that precedes this launch.
                    payload["last_death"] = self.death
                    payload["deaths"] = self.deaths
                    payload["port"] = self.cfg.get("port")
                    payload["slot_map"] = self.slotmap
                    payload["launch"] = self.launch
                    payload["stack"] = self.cfg.get("stack") or {}
                    self.prev, self.prev_t, self.status = cur, t0, payload
                self.hist.append({"at": t0, "gen_tok_s": payload["rate"].get("generation_tok_s"),
                          "gen_per_step": payload["rate"].get("gen_per_step"),
                                  "stream_tok_s": payload["rate"].get("stream_tok_s_live"),
                                  "kv_tok_s": payload["rate"].get("kv_tok_s"),
                                  "kv_used_pct": payload["live"].get("kv_used_pct"),
                                  "running": payload["live"].get("running")})
            except Exception as e:                                # a stopped launch is normal state
                if isinstance(e, urllib.error.HTTPError):
                    self.key_needed = e.code == 401                 # a keyed serve answers 401 to our GETs
                with self.lock:
                    self.status = dict(self.status, reachable=False, at=time.time(),
                                       upstream=self.cfg["upstream"], error=str(e)[:300],
                                       key_needed=self.key_needed or bool(self.cfg["key"]),
                                       gpus=self.gpus, gpus_history=list(self.gpuhist),
                                       pcie=self.pcie, pcie_history=list(self.pciehist),
                                       slot_map=self.slotmap,
                                       pcie_events=list(self.pcie_events),
                                       models=self.models, engine=self.engine,
                                       meta=self.meta, requests_kept=self.kept(),
                                       stack=self.cfg.get("stack") or {})
            time.sleep(self.cfg["poll"])

    def slow_poll(self):
        """/health and /v1/models do not need 2 s freshness; /metrics is the one that carries rates."""
        while True:
            try:
                code, _ = self.get("/health")
                self.health = code
                _, body = self.get("/v1/models")
                self.models = json.loads(body).get("data", [])
                self.engine = {"vllm_version": self.cfg["version"]} if self.cfg["version"] else {}
            except Exception:
                self.health = None
            time.sleep(min(15.0, max(2.0, self.cfg["poll"] * 5)))

    def gpu_poll(self):
        while True:
            self.gpus = read_gpus(self.cfg["gpus"])
            if self.cfg["smi"]:
                by_bus, err = read_smi(self.cfg["smi"])
                self.smi_error = err
                if by_bus:
                    self.gpus = merge_smi(self.gpus, by_bus)
            # What a card refuses to answer (EBUSY while a compute context holds it) is stored as a null, not
            # a 0: the gap is drawn as a gap, and a card that stops answering does not look like a card that
            # went idle.
            self.gpuhist.append({"at": time.time(), "cards": [
                {"card": g.get("card"), "bus": g.get("bus"), "busy_pct": g.get("busy_pct"),
                 "vram_used": g.get("vram_used"), "vram_total": g.get("vram_total"),
                 "sclk_mhz": g.get("sclk_mhz"), "power_w": g.get("power_w"),
                 "power_cap_w": g.get("power_cap_w"), "temp_c": g.get("temp_c"),
                 # mem gets its own slot because on this box it is the sensor that separates the cards at all
                 # (card4 95.7 C against 82.7-86.0 for the rest, with junction equal); edge stays out of the
                 # ring, since three temperature waves per card is where a panel starts being unreadable.
                 "temp_mem_c": g.get("temp_mem_c")}
                for g in self.gpus]})
            self.host = read_host()
            # .get, not []: a cfg built by the other entry point (serve/plugin) is not guaranteed to carry the
            # newest keys, and a KeyError here kills the poll thread silently -- the Cards panel keeps its last
            # frame and looks alive. That is exactly what happened when the PCIe key shipped without the plugin
            # cfg knowing it: the merged page had one card sample and no PCIe, and the serve log held a dead thread.
            if self.cfg.get("pci"):
                chains = read_pcie(self.cfg["pci"])
                # Normalise the domain: read_gpus reports "0000:c3:00.0" and the chain summary strips it to
                # "c3:00.0", and an unnormalised join left every chain row labelled by BDF with no card name
                # (visible on the live page: "06:00.0 32.0 GT/s" instead of "card4").
                by_bus = {(g.get("bus") or "").replace("0000:", ""): g.get("card") for g in self.gpus}
                for ch in chains:
                    ch["card"] = by_bus.get(ch["bus"])
                # And the other direction: the card rows carry the slot their chain sits in. The join is by
                # chain, not by bus id, because the BIOS names the slot at the card's switch upstream port
                # (0000:c4:00), which is not the GPU endpoint's own bus id (c6:00.0).
                slot_by_ep = {ch["bus"]: ch.get("slot") for ch in chains if ch.get("slot")}
                for g in self.gpus:
                    g["slot"] = slot_by_ep.get((g.get("bus") or "").replace("0000:", "").lower())
                # Chassis order, and the one-line answer to "which GPU is which PCIe address in which slot".
                order_by_slot(self.gpus, chains)
                self.slotmap = [{"pos": ch.get("pos"), "slot": ch.get("slot"), "card": ch.get("card"),
                                 "bus": ch["bus"], "speed": ch.get("speed"), "width": ch.get("width"),
                                 # The ceiling this chain could run at, so the line can say "16.0 of 32.0".
                                 # Not `_num(...)`: that helper answers "is this a number" (a bool), and using it
                                 # here put `True` in the field the page prints as a speed.
                                 "max_speed": max([(d.get("max_speed") or 0) for d in ch["devices"]]) or None,
                                 "chain": [d["bdf"] for d in ch["devices"]],
                                 "slot_at": ch.get("slot_at"),
                                 "corr": sum((d.get("cor") or 0) for d in ch["devices"]),
                                 "uncorr": sum((d.get("nonfatal") or 0) + (d.get("fatal") or 0)
                                               for d in ch["devices"])}
                                for ch in chains]
                new_events = []
                deltas, self.aer_last = aer_deltas(chains, self.aer_last, self.aer_named, new_events,
                                                   time.time(), self.aer_quiet, self.aer_recent)
                for e in reversed(new_events):
                    self.pcie_events.appendleft(e)
                self.pcie = chains
                self.pciehist.append({"at": time.time(), "chains": deltas})
            time.sleep(max(1.0, self.cfg["poll"]))

    # ---- the API the pages read -------------------------------------------------------------------
    def api(self, path):
        if path == "/api/status":
            with self.lock:
                return dict(self.status, requests_kept=self.kept())
        if path == "/api/requests":
            return {"requests": (self.eng.listing() if self.eng is not None else self.rec.listing()),
                    "kept": self.cfg["keep"], "mode": self.cfg["mode"],
                    "source": "engine" if self.eng is not None else "relay"}
        if path == "/api/host":
            return self.host
        return None


# -------------------------------------------------------------------------------------- proxy mode
class H(BaseHTTPRequestHandler):
    hub = None
    protocol_version = "HTTP/1.1"
    server_version = "r9700-webui/1"

    def log_message(self, fmt, *args):
        if self.hub.cfg["verbose"]:
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -- helpers
    def json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def page(self, name):
        p = WEB / name
        ctype = KIND.get(p.suffix)
        if ctype is None or not p.is_file():
            return self.json(404, {"error": {"message": f"{name} not found"}})
        body = p.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def want_key(self):
        """Our own key (--key) guards the API and the relay, not the pages: the HTML has no secrets
        in it, and an open relay to a keyed upstream is the thing worth stopping."""
        if not self.hub.cfg["key"]:
            return True
        got = (self.headers.get("Authorization") or "").removeprefix("Bearer ").strip()
        if got and got == self.hub.cfg["key"]:
            return True
        self.json(401, {"error": {"message": "this monitor needs its API key", "type": "auth"}})
        return False

    def read_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            while True:
                size = int(self.rfile.readline().split(b";")[0] or b"0", 16)
                if size == 0:
                    self.rfile.readline()
                    break
                raw += self.rfile.read(size)
                self.rfile.readline()
        return raw

    # -- routing
    def do_GET(self):
        path = urllib.parse.urlsplit(self.path).path
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
        if path in ("/", "/monitor", "/about"):
            return self.page("about.html" if path == "/about" else "monitor.html")
        if path.startswith("/web/"):
            p = web_asset(WEB, urllib.parse.unquote(path[5:]))
            if p is None:
                return self.json(404, {"error": {"message": f"{path[5:]} not found"}})
            return self.page(str(p.relative_to(WEB.resolve())))   # the validated RELATIVE path, not p.name:
                                                                  # the fonts live in serve/web/fonts/
        if path == "/favicon.svg":
            return self.page("favicon.svg")
        if path.startswith("/api/"):
            if not self.want_key():
                return
            r = q.get("id", [None])[0]
            if path == "/api/requests" and r:
                d = self.hub.record_detail(r)
                return self.json(404 if not d else 200, d or {"error": {"message": r + " not kept"}})
            out = self.hub.api(path)
            return self.json(404 if out is None else 200, out if out is not None
                             else {"error": {"message": path + " not an endpoint"}})
        if path == "/healthz":
            with self.hub.lock:
                st = self.hub.status
            return self.json(200, {"ok": True, "upstream_reachable": st.get("reachable"),
                                   "mode": self.hub.cfg["mode"], "records": len(self.hub.rec.items)})
        return self.forward("GET")

    def do_POST(self):
        return self.forward("POST")

    def do_DELETE(self):
        return self.forward("DELETE")

    def do_PUT(self):
        return self.forward("PUT")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Allow", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    # -- the relay. Non-event-stream responses are read whole (so Content-Length is honest); SSE is
    # pumped chunk-for-chunk with our own framing, and the counters are lifted off the deltas as
    # they pass without holding the bytes back.
    def forward(self, method):
        if not self.want_key():
            return
        u = urllib.parse.urlsplit(self.path)
        target = u.path + ("?" + u.query if u.query else "")
        raw = self.read_body() if method in ("POST", "PUT") else b""
        api = target.startswith("/v1/") or target.startswith("/v1beta")
        want = api and self.hub.cfg["mode"] == "proxy" and method == "POST"
        rec = None
        if want:
            try:
                parsed = json.loads(raw or b"{}")
            except ValueError:
                parsed = {}
            rec = self.hub.rec.start(target, str(parsed.get("model") or ""), raw.decode("utf-8", "replace"),
                                     bool(parsed.get("stream")))
        up = urllib.parse.urlsplit(self.hub.cfg["upstream"])
        conn = (http.client.HTTPSConnection if up.scheme == "https" else http.client.HTTPConnection)(
            up.hostname, up.port or (443 if up.scheme == "https" else 80), timeout=self.hub.cfg["timeout"])
        hdrs = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        if self.hub.cfg["key"] and not hdrs.get("Authorization"):
            hdrs["Authorization"] = f"Bearer {self.hub.cfg['key']}"
        try:
            conn.request(method, target, body=raw or None, headers=hdrs)
            res = conn.getresponse()
        except Exception as e:
            if rec:
                self.hub.rec.finish(rec, "error", error=f"upstream unreachable: {e}")
            return self.json(502, {"error": {"message": f"upstream {self.hub.cfg['upstream']} unreachable: {e}"}})
        ctype = res.getheader("Content-Type", "")
        streaming = "text/event-stream" in ctype
        self.send_response(res.status)
        for k, v in res.getheaders():
            if k.lower() not in HOP:
                self.send_header(k, v)
        if streaming:
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()
            self._pump(res, rec, streaming=True)
        else:
            body = res.read()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            if rec:
                self._absorb(rec, body)
        if rec:
            self.hub.rec.finish(rec, "completed" if res.status < 400 else "error", status=res.status,
                                error=None if res.status < 400 else (rec.get("error") or {}).get("message"))

    def _pump(self, res, rec, streaming):
        buf = b""
        try:
            while True:
                # read1, not read: read(n) on a body with no Content-Length blocks until n bytes or EOF,
                # which for a slow SSE stream would deliver the first delta when the generation ends --
                # the one thing a monitor in the path must not do (bench/webui_ab.py caught it).
                chunk = res.read1(65536) if hasattr(res, "read1") else res.read(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
                if rec:
                    self.hub.rec.note_resp(rec, chunk)
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        self._sse(rec, line.decode("utf-8", "replace").strip())
        except (BrokenPipeError, ConnectionResetError):
            if rec:
                return self.hub.rec.finish(rec, "disconnected", status=rec["http_status"],
                                           error="the client hung up")
        except Exception as e:
            if rec:
                return self.hub.rec.finish(rec, "error", error=str(e))
        if rec:
            self.hub.rec.finish(rec, rec["state"] if rec["state"] in ("queued",) else "completed",
                                status=rec["http_status"])

    def _sse(self, rec, line):
        """One SSE line of a vLLM stream: keep the first-token stamp, the text, the reasoning text
        and (when the client asked for it) the final usage block."""
        if not line.startswith("data:"):
            return
        payload = line[5:].strip()
        if payload in ("", "[DONE]"):
            return
        try:
            obj = json.loads(payload)
        except ValueError:
            return
        if obj.get("id") and not rec["id"]:
            rec["id"] = obj["id"]
        if obj.get("usage"):
            u = obj["usage"]
            rec["usage"] = {"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens")}
        for ch in obj.get("choices", []):
            if rec["first_token_s"] is None and (ch.get("text") or (ch.get("delta") or {})):
                rec["first_token_s"] = time.time() - rec["started_at"]
                rec["state"] = "generating"
            txt = ch.get("text")
            d = ch.get("delta") or {}
            rc = d.get("reasoning_content") or d.get("reasoning")
            if rc:
                self.hub.rec.note_reason(rec, rc)
            txt = txt or d.get("content")
            if txt:
                rec["tokens_est"] += 1
                self.hub.rec.note(rec, txt)
            if ch.get("finish_reason"):
                rec["finish_reason"] = ch["finish_reason"]

    def _absorb(self, rec, body):
        try:
            obj = json.loads(body.decode("utf-8", "replace"))
        except ValueError:
            self.hub.rec.note(rec, body.decode("utf-8", "replace")[:CAP])
            return
        rec["id"] = obj.get("id") or rec["id"]
        rec["usage"] = obj.get("usage") or rec["usage"]
        if rec["first_token_s"] is None:
            rec["first_token_s"] = time.time() - rec["started_at"]
        if isinstance(obj.get("error"), dict):
            rec["error"] = {"message": obj["error"].get("message", "")}
        for ch in obj.get("choices", []):
            m = ch.get("message") or {}
            self.hub.rec.note(rec, m.get("content") or ch.get("text") or "")
            self.hub.rec.note_reason(rec, m.get("reasoning_content") or "")


def build(cfg):
    """(hub, server) -- split out of main() so a test can run the whole thing against a fake upstream."""
    hub = Hub(cfg)
    for t in (threading.Thread(target=hub.poll, daemon=True),
              threading.Thread(target=hub.slow_poll, daemon=True),
              threading.Thread(target=hub.gpu_poll, daemon=True)):
        t.start()
    H.hub = hub
    ThreadingHTTPServer.allow_reuse_address = True
    ThreadingHTTPServer.daemon_threads = True
    return hub, ThreadingHTTPServer(("", cfg["port"]), H)


def main(argv=None):
    ap = argparse.ArgumentParser(description="the r9700 serve monitor and about pages")
    ap.add_argument("--port", type=int, default=int(os.environ.get("R9K_WEB_PORT", 8081)))
    ap.add_argument("--upstream", default=os.environ.get("R9K_WEB_UPSTREAM", "http://127.0.0.1:8080"))
    ap.add_argument("--proxy", action="store_true",
                    default=os.environ.get("R9K_WEB_PROXY", "0") == "1",
                    help="relay API traffic through here, which is what fills the request list")
    ap.add_argument("--key", default=os.environ.get("R9K_WEB_KEY", ""))
    ap.add_argument("--poll", type=float, default=float(os.environ.get("R9K_WEB_POLL", 1.0)))
    ap.add_argument("--keep", type=int, default=int(os.environ.get("R9K_WEB_KEEP", 100)))
    ap.add_argument("--meta", default=os.environ.get("R9K_WEB_META", ""))
    ap.add_argument("--gpus", default=os.environ.get("R9K_WEB_GPUS", "/sys/class/drm"))
    ap.add_argument("--pci", default=os.environ.get("R9K_WEB_PCI", "/sys/bus/pci/devices"),
                    help="PCIe sysfs root for the link/AER card (empty = off). Read-only and needs no root "
                         "and no /dev/kfd, unlike the byte counters, which answer N/A on these cards.")
    ap.add_argument("--journal", default=os.environ.get("R9K_WEB_JOURNAL", ""),
                    help="append finished requests to this JSONL file and replay it at startup, so the request "
                         "list survives a restart or a hard reset of the serve (empty = memory only)")
    ap.add_argument("--journal-attach", default=os.environ.get("R9K_WEB_JOURNAL_ATTACH", ""),
                    help="read another monitor's journal READ-ONLY (no writes, no rotation): a standalone "
                         "Monitor keeps showing the dead launch's rows while its serve is down")
    ap.add_argument("--smi", default=os.environ.get("R9K_WEB_SMI", ""),
                    help="fill what the sysfs refuses from rocm-smi (empty = off; pass a command to override "
                         "the default -- opens /dev/kfd on every poll, so keep it off while measuring)")
    ap.add_argument("--version", default=os.environ.get("R9K_WEB_VERSION", ""), help="engine version to show")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    cfg = {"port": a.port, "upstream": a.upstream.rstrip("/"), "mode": "proxy" if a.proxy else "observe",
           "key": a.key, "poll": a.poll, "keep": a.keep, "meta": a.meta,
           "gpus": a.gpus, "pci": a.pci, "journal": a.journal, "journal_attach": a.journal_attach, "smi": (SMI_DEFAULT if a.smi == "1" else a.smi.split()) if a.smi else None,
           "timeout": 600.0, "verbose": a.verbose, "version": a.version}
    _, srv = build(cfg)
    print(f"webui: http://127.0.0.1:{cfg['port']}/monitor  upstream {cfg['upstream']}  "
          f"mode {cfg['mode']}  keep {cfg['keep']}  poll {cfg['poll']}s", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
