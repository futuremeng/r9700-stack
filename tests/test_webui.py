#!/usr/bin/env python3
"""tests/test_webui.py - the Monitor/About sidecar (serve/webui.py).

Four things are worth testing here, and only one of them is "the HTTP works":

  (1) /metrics reading. vLLM's gauges are per rank: summed, "one rank's KV pool is full" disappears.
      Buckets must be skipped or every mean is a lie. The KV pool size exists only inside labels.
  (2) the request ring: eviction, the fields the page reads, the 256 KiB honesty flag.
  (3) the amdgpu sysfs read against a fake tree (this test must not touch a card).
  (4) end to end against a fake vLLM: the SSE bytes reach the client exactly as they left the
      upstream, and the record the Monitor lists was taken from the stream that passed.

Run: ~/venv-r9700/bin/python3 tests/test_webui.py      (also: pytest tests/test_webui.py)
"""
import json
import http.client
import os
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "serve"))

import webui as W  # noqa: E402

fail = 0


def ck(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name} {detail}")
    if not cond:
        fail += 1


METRICS = """# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="Qwen3.8",engine="0"} 1.0
vllm:num_requests_running{model_name="Qwen3.8",engine="1"} 3.0
vllm:num_requests_waiting{model_name="Qwen3.8",engine="0"} 0.0
vllm:num_requests_waiting{model_name="Qwen3.8",engine="1"} 2.0
vllm:gpu_cache_usage_perc{model_name="Qwen3.8",engine="0"} 0.40
vllm:gpu_cache_usage_perc{model_name="Qwen3.8",engine="1"} 0.72
vllm:cache_config_info{block_size="1088",num_gpu_blocks="456",num_cpu_blocks="0",cache_format=""} 1.0
vllm:prompt_tokens_total{model_name="Qwen3.8"} 1000.0
vllm:generation_tokens_total{model_name="Qwen3.8"} 200.0
vllm:num_preemptions_total{model_name="Qwen3.8"} 0.0
vllm:prefix_cache_queries_total{model_name="Qwen3.8"} 900.0
vllm:prefix_cache_hits_total{model_name="Qwen3.8"} 300.0
vllm:spec_decode_num_drafts_total{model_name="Qwen3.8"} 40.0
vllm:spec_decode_num_accepted_tokens_total{model_name="Qwen3.8"} 90.0
vllm:spec_decode_num_draft_tokens_total{model_name="Qwen3.8"} 120.0
vllm:iteration_tokens_total_sum{model_name="Qwen3.8"} 640.0
vllm:iteration_tokens_total_count{model_name="Qwen3.8"} 128.0
vllm:time_to_first_token_seconds_sum{model_name="Qwen3.8"} 4.0
vllm:time_to_first_token_seconds_count{model_name="Qwen3.8"} 8.0
vllm:time_to_first_token_seconds_bucket{le="0.25"} 2.0
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 8.0
vllm:time_per_output_token_seconds_sum{model_name="Qwen3.8"} 1.6
vllm:time_per_output_token_seconds_count{model_name="Qwen3.8"} 80.0
vllm:request_queue_time_seconds_sum{model_name="Qwen3.8"} 0.02
vllm:request_queue_time_seconds_count{model_name="Qwen3.8"} 8.0
vllm:request_prefill_time_seconds_sum{model_name="Qwen3.8"} 1.2
vllm:request_prefill_time_seconds_count{model_name="Qwen3.8"} 8.0
vllm:e2e_request_latency_seconds_sum{model_name="Qwen3.8"} 20.0
vllm:e2e_request_latency_seconds_count{model_name="Qwen3.8"} 8.0
"""

# The same serve a window later: four more requests, 600 more generated tokens, and every histogram
# moved by an amount that makes the derived means below checkable by hand.
METRICS2 = METRICS \
    .replace('vllm:generation_tokens_total{model_name="Qwen3.8"} 200.0',
             'vllm:generation_tokens_total{model_name="Qwen3.8"} 800.0') \
    .replace('vllm:prompt_tokens_total{model_name="Qwen3.8"} 1000.0',
             'vllm:prompt_tokens_total{model_name="Qwen3.8"} 1600.0') \
    .replace('vllm:e2e_request_latency_seconds_count{model_name="Qwen3.8"} 8.0',
             'vllm:e2e_request_latency_seconds_count{model_name="Qwen3.8"} 12.0') \
    .replace('vllm:time_to_first_token_seconds_sum{model_name="Qwen3.8"} 4.0',
             'vllm:time_to_first_token_seconds_sum{model_name="Qwen3.8"} 6.0') \
    .replace('vllm:time_to_first_token_seconds_count{model_name="Qwen3.8"} 8.0',
             'vllm:time_to_first_token_seconds_count{model_name="Qwen3.8"} 12.0') \
    .replace('vllm:time_per_output_token_seconds_sum{model_name="Qwen3.8"} 1.6',
             'vllm:time_per_output_token_seconds_sum{model_name="Qwen3.8"} 2.4') \
    .replace('vllm:time_per_output_token_seconds_count{model_name="Qwen3.8"} 80.0',
             'vllm:time_per_output_token_seconds_count{model_name="Qwen3.8"} 120.0') \
    .replace('vllm:iteration_tokens_total_sum{model_name="Qwen3.8"} 640.0',
             'vllm:iteration_tokens_total_sum{model_name="Qwen3.8"} 660.0') \
    .replace('vllm:iteration_tokens_total_count{model_name="Qwen3.8"} 128.0',
             'vllm:iteration_tokens_total_count{model_name="Qwen3.8"} 132.0') \
    .replace('vllm:prefix_cache_queries_total{model_name="Qwen3.8"} 900.0',
             'vllm:prefix_cache_queries_total{model_name="Qwen3.8"} 1200.0') \
    .replace('vllm:prefix_cache_hits_total{model_name="Qwen3.8"} 300.0',
             'vllm:prefix_cache_hits_total{model_name="Qwen3.8"} 400.0') \
    .replace('vllm:spec_decode_num_drafts_total{model_name="Qwen3.8"} 40.0',
             'vllm:spec_decode_num_drafts_total{model_name="Qwen3.8"} 44.0') \
    .replace('vllm:spec_decode_num_accepted_tokens_total{model_name="Qwen3.8"} 90.0',
             'vllm:spec_decode_num_accepted_tokens_total{model_name="Qwen3.8"} 120.0') \
    .replace('vllm:spec_decode_num_draft_tokens_total{model_name="Qwen3.8"} 120.0',
             'vllm:spec_decode_num_draft_tokens_total{model_name="Qwen3.8"} 160.0')


# ------------------------------------------------------------------ (1) /metrics -------------------
cur, cap, info = W.parse_metrics(METRICS)
ck("series parsed", len(cur) >= 20, f"{len(cur)} series")
ck("buckets skipped", "time_to_first_token_seconds_bucket" not in cur)
ck("counter summed over ranks", cur["prompt_tokens_total"][0] == 1000.0)
ck("gauge max is the worst rank", cur["gpu_cache_usage_perc"][1] == 0.72, str(cur["gpu_cache_usage_perc"]))
ck("KV capacity from labels", cap == 456 * 1088, f"{cap}")
ck("model name harvested", info.get("model_name") == "Qwen3.8", str(info))
ck("block size harvested", info.get("block_size") == "1088")

prev, _, _ = W.parse_metrics(METRICS)
cur2, cap2, info2 = W.parse_metrics(METRICS2)
st = W.derive(cur2, prev, 10.0, cap2, info2, "observe", 0, [])
ck("generation rate over the window", st["rate"]["generation_tok_s"] == 60.0, str(st["rate"]["generation_tok_s"]))
ck("requests per window", st["rate"]["requests"] == 4.0)
ck("mean TTFT over the window", abs(st["rate"]["ttft_ms"] - 500.0) < 1e-9, str(st["rate"]["ttft_ms"]))
ck("mean ITL from per-output-token", abs(st["rate"]["itl_ms"] - 20.0) < 1e-9, str(st["rate"]["itl_ms"]))
ck("stream tok/s is 1/per-output-token", abs(st["rate"]["stream_tok_s"] - 50.0) < 1e-9)
ck("spec acceptance 90/120", abs(st["rate"]["accept_pct"] - 75.0) < 1e-9, str(st["rate"]["accept_pct"]))
ck("tokens per engine step", abs(st["rate"]["tokens_per_iter"] - 5.0) < 1e-9)
ck("KV uses the worst rank", abs(st["live"]["kv_used_pct"] - 72.0) < 1e-9, str(st["live"]["kv_used_pct"]))
ck("running is per-rank max", st["live"]["running"] == 3.0)
ck("prefix hit rate", abs(st["live"]["prefix_hit_pct"] - 100.0 / 3.0) < 1e-6)
ck("a counter that did not move reads None, not 0", W.derive(cur2, cur2, 10.0, cap2, info2, "observe", 0, [])
   ["rate"]["ttft_ms"] is None)

# The two rates the headline row is built from are the two that move WHILE a request runs. Measured on this
# box (0.32.0+rocm723, 2026-11-02): generation_tokens_total advances every engine step (one 900-token stream
# moved it ~120 tokens per 0.5 s sample) and so does the KV gauge (0.036 -> 0.1007 across a 51,859-token
# prefill), while prompt_tokens_total, iteration_tokens_total_* and every latency histogram are credited when
# the request finishes (that same prefill held the prompt counter flat for 10 s, then one jump 129 -> 51,988).
ck("single-stream rate = aggregate over running, live",
   abs(st["rate"]["stream_tok_s_live"] - 20.0) < 1e-9,
   f"{st['rate']['stream_tok_s_live']} for 60 tok/s across 3 running")
METRICS3 = METRICS2.replace('vllm:gpu_cache_usage_perc{model_name="Qwen3.8",engine="1"} 0.72',
                           'vllm:gpu_cache_usage_perc{model_name="Qwen3.8",engine="1"} 0.75')
cur3, cap3, info3 = W.parse_metrics(METRICS3)
stk = W.derive(cur3, cur2, 10.0, cap3, info3, "observe", 0, [])["rate"]
ck("context growth in tokens a second, from the live gauge",
   abs(stk["kv_tok_s"] - 0.03 * 456 * 1088 / 10.0) < 1e-6, str(stk.get("kv_tok_s")))
nojob = METRICS2.replace('vllm:num_requests_running{model_name="Qwen3.8",engine="0"} 1.0',
                         'vllm:num_requests_running{model_name="Qwen3.8",engine="0"} 0.0') \
                .replace('vllm:num_requests_running{model_name="Qwen3.8",engine="1"} 3.0',
                         'vllm:num_requests_running{model_name="Qwen3.8",engine="1"} 0.0')
ncur, _, _ = W.parse_metrics(nojob)
ck("no streams running gives no single-stream rate, not a division by zero",
   W.derive(ncur, prev, 10.0, cap2, info2, "observe", 0, [])["rate"]["stream_tok_s_live"] is None)

# The two latency families are different quantities under speculation and the TOKEN rate is the per-token one.
# Measured on this launch (MTP=2), one stream, one window: step 12.84 ms, per accepted token 6.27 ms, and
# generation_tokens_total moving 155 tokens a second -- which is 1/6.27 ms (159.5), not 1/12.84 ms (77.9).
# The page used to say the opposite, from a pair implying 4.9 tokens a step, above the 1+2 ceiling the launch
# can reach. The fixture below is that shape: a 40 ms step turning out two accepted tokens.
BOTH = METRICS2 + ('\nvllm:inter_token_latency_seconds_sum{model_name="Qwen3.8"} 1.6\n'
                   'vllm:inter_token_latency_seconds_count{model_name="Qwen3.8"} 40.0\n')
bcur, bcap, binfo = W.parse_metrics(BOTH)
b = W.derive(bcur, prev, 10.0, bcap, binfo, "observe", 0, [])["rate"]
ck("the step family and the per-token family stay apart",
   abs(b["itl_ms"] - 40.0) < 1e-9 and abs(b["tpot_ms"] - 20.0) < 1e-9, f"{b['itl_ms']} / {b['tpot_ms']}")
ck("a stream's tok/s follows the per-token family, not the step one",
   abs(b["stream_tok_s"] - 50.0) < 1e-9, str(b["stream_tok_s"]))
ck("tokens per step reconciles the two (40 ms a step, 20 ms a token = 2)",
   abs(b["tokens_per_step"] - 2.0) < 1e-9, str(b.get("tokens_per_step")))

# The names vLLM 0.32.0+rocm723 actually publishes on this box (captured from a live native launch,
# 2026-11-02): kv_cache_usage_perc instead of gpu_cache_usage_perc, request_time_per_output_token_seconds,
# no num_requests_swapped at all, `_created` stamps beside every counter, and a hybrid GDN model whose real
# KV pool (497,588 tokens) is NOT num_gpu_blocks x block_size (1,230 x 448 = 551,040).
METRICS32 = """
vllm:kv_cache_usage_perc{engine="0",model_name="Qwen3.8"} 0.25
vllm:num_requests_running{engine="0",model_name="Qwen3.8"} 2.0
vllm:num_requests_waiting{engine="0",model_name="Qwen3.8"} 1.0
vllm:num_requests_waiting_by_reason{engine="0",model_name="Qwen3.8",reason="capacity"} 1.0
vllm:num_requests_waiting_by_reason{engine="0",model_name="Qwen3.8",reason="deferred"} 0.0
vllm:cache_config_info{block_size="448",cache_dtype="auto",engine="0",gpu_memory_utilization="0.97",\
    kv_cache_size_tokens="497588",model_name="Qwen3.8",num_gpu_blocks="1230"} 1.0
vllm:generation_tokens_total{engine="0",model_name="Qwen3.8"} 100.0
vllm:generation_tokens_created{engine="0",model_name="Qwen3.8"} 1762000000.0
vllm:inter_token_latency_seconds_sum{engine="0",model_name="Qwen3.8"} 3.0
vllm:inter_token_latency_seconds_count{engine="0",model_name="Qwen3.8"} 150.0
vllm:prefix_cache_queries_total{engine="0",model_name="Qwen3.8"} 0.0
vllm:prefix_cache_hits_total{engine="0",model_name="Qwen3.8"} 0.0
vllm:num_preemptions_total{engine="0",model_name="Qwen3.8"} 0.0
vllm:num_preemptions_created{engine="0",model_name="Qwen3.8"} 1762000000.0
"""
cur32, cap32, info32 = W.parse_metrics(METRICS32)
prev32, _, _ = W.parse_metrics(METRICS32.replace('model_name="Qwen3.8"} 100.0', 'model_name="Qwen3.8"} 40.0')
                 .replace("inter_token_latency_seconds_sum{engine=\"0\",model_name=\"Qwen3.8\"} 3.0",
                          "inter_token_latency_seconds_sum{engine=\"0\",model_name=\"Qwen3.8\"} 1.0")
                 .replace("inter_token_latency_seconds_count{engine=\"0\",model_name=\"Qwen3.8\"} 150.0",
                          "inter_token_latency_seconds_count{engine=\"0\",model_name=\"Qwen3.8\"} 50.0"))
st32 = W.derive(cur32, prev32, 2.0, cap32, info32, "observe", 0, [])
ck("the 0.32 KV gauge name is read (kv_cache_usage_perc)", abs(st32["live"]["kv_used_pct"] - 25.0) < 1e-9,
   str(st32["live"]["kv_used_pct"]))
ck("the 0.32 step family is read (inter_token_latency_seconds)", abs(st32["rate"]["itl_ms"] - 20.0) < 1e-9,
   str(st32["rate"]["itl_ms"]))
ck("a build with no per-token family reports it absent, not equal", st32["rate"]["tpot_ms"] is None,
   str(st32["rate"]["tpot_ms"]))

# The two inter-token families are NOT the same quantity, and the gap is speculation: one observation per
# engine step against one per request of that request's mean per ACCEPTED token. This fixture's pair (a 15 ms
# step and 300 ms per accepted token) is deliberately unphysical -- per-token time cannot exceed the step that
# produced it -- because it exists only to prove the two families are never conflated. The physical shape, and
# which one is a stream's token rate, is the BOTH fixture above: measured on the live launch 2026-11-02, one
# stream in one window gave a 12.84 ms step against 6.27 ms per accepted token while generation_tokens_total
# moved 155 tokens a second, so the token rate follows the per-token family (1/6.27 = 159.5), not the step
# family (1/12.84 = 77.9). An earlier revision of this file asserted the opposite, off a pair that implied 4.9
# tokens a step where the launch's ceiling is 1 + 2.
STEPF = ('vllm:inter_token_latency_seconds_sum{model_name="Qwen3.8"} 30.0\n'
         'vllm:inter_token_latency_seconds_count{model_name="Qwen3.8"} 2000.0\n'
         'vllm:request_time_per_output_token_seconds_sum{model_name="Qwen3.8"} 6.0\n'
         'vllm:request_time_per_output_token_seconds_count{model_name="Qwen3.8"} 20.0\n')
STEPF0 = (STEPF.replace("} 30.0", "} 15.0").replace("} 2000.0", "} 1000.0")
          .replace("} 6.0", "} 3.0").replace("seconds_count{model_name=\"Qwen3.8\"} 20.0",
                                             "seconds_count{model_name=\"Qwen3.8\"} 10.0"))
scur, scam, sinfo = W.parse_metrics(STEPF)
sp, _, _ = W.parse_metrics(STEPF0)
ss = W.derive(scur, sp, 1.0, scam, sinfo, "observe", 0, [])
ck("a step family and a per-accepted-token family stay apart",
   abs(ss["rate"]["itl_ms"] - 15.0) < 1e-9 and abs(ss["rate"]["tpot_ms"] - 300.0) < 1e-9,
   str((ss["rate"]["itl_ms"], ss["rate"]["tpot_ms"])))
ck("a stream's tok/s follows the per-token family even when the step family is present",
   abs(ss["rate"]["stream_tok_s"] - 1e3 / 300.0) < 1e-9, str(ss["rate"]["stream_tok_s"]))
ck("and the unphysical pair is reported as 0.05 tokens a step rather than smoothed away",
   abs(ss["rate"]["tokens_per_step"] - 0.05) < 1e-9, str(ss["rate"].get("tokens_per_step")))
ck("a hybrid model's KV pool is kv_cache_size_tokens, not blocks x block_size", cap32 == 497588, str(cap32))
ck("a gauge the build does not have reads absent, not zero", st32["live"]["swapped"] is None,
   str(st32["live"]["swapped"]))
ck("_created stamps are not mistaken for series", "generation_tokens_created" not in cur32, str(sorted(cur32)))
ck("labels the newer build carries are harvested", info32.get("cache_dtype") == "auto"
   and info32.get("gpu_memory_utilization") == "0.97", str(sorted(info32)))
# The waiting queue arrives twice on 0.32: once whole, once by reason. Its own HELP text: "'capacity' = waiting for
# scheduling capacity; 'deferred' = deferred by transient constraints (LoRA budget, KV transfer, blocked status).
# Sum of all reasons equals vllm:num_requests_waiting." The split matters because the total alone misleads: measured
# on the live 262k launch, 8 x 30k queued with the KV pool 11.7% full -- scheduling capacity, not blocks.
ck("a labelled gauge is kept per label value too", cur32["num_requests_waiting_by_reason;capacity"][1] == 1.0
   and cur32["num_requests_waiting_by_reason;deferred"][1] == 0.0, str(sorted(cur32)))
ck("the queue split reaches the page, and a real 0 stays a 0",
   st32["live"]["waiting_capacity"] == 1.0 and st32["live"]["waiting_deferred"] == 0.0,
   str(st32["live"]))
ck("a build that publishes no reason labels reports it absent, not zero",
   st["live"]["waiting_capacity"] is None and st["live"]["waiting_deferred"] is None, str(st["live"]))
# Preemptions are the counter that says a long-context load has gone past what the KV pool can hold: 540,202 pool
# tokens against 160k prompts is 3.4 contexts, and the first two preemptions on this box came from exactly that.
ck("preemptions since launch ride the payload, and never-preempted reads as a real 0, not as unknown",
   st32["live"]["preemptions_total"] == 0.0, str(st32["live"]["preemptions_total"]))
_cur_noprem, _cap_n, _info_n = W.parse_metrics(METRICS32.replace(
    'vllm:num_preemptions_total{engine="0",model_name="Qwen3.8"} 0.0\n', ""))
ck("a build that does not publish the preempt counter leaves it unknown rather than claiming 0",
   W.derive(_cur_noprem, _cur_noprem, 2.0, _cap_n, _info_n, "observe", 0, [])["live"]["preemptions_total"]
   is None)

# A histogram is observed when a request FINISHES, so the window a long request spans holds no observation
# at all. The page falls back to the cumulative mean and says so; the first poll, which has no window, still
# gets the cumulative one.
first = W.derive(cur, {}, None, cap, info, "observe", 0, [])
ck("with no window the rates are None and the cumulative means are not",
   first["rate"]["ttft_ms"] is None and abs(first["since"]["ttft_ms"] - 500.0) < 1e-9,
   f"{first['rate']['ttft_ms']} {first['since']['ttft_ms']}")
ck("cumulative acceptance is over the whole run", abs(first["since"]["accept_pct"] - 75.0) < 1e-9,
   str(first["since"]["accept_pct"]))
ck("cumulative steps per request", abs(first["since"]["iters_per_req"] - 16.0) < 1e-9,
   str(first["since"]["iters_per_req"]))
ck("requests finished since the engine started", first["since"]["requests"] == 8.0)

# ...and the scaling bug that only a live engine shows: request_prompt_tokens counts TOKENS. Scaling it
# like a seconds family made the first real /api/status report a mean prompt of 2,028,442 tokens.
TOKF = ('vllm:request_prompt_tokens_sum{model_name="Qwen3.8"} 4000.0\n'
        'vllm:request_prompt_tokens_count{model_name="Qwen3.8"} 8.0\n'
        'vllm:request_generation_tokens_sum{model_name="Qwen3.8"} 400.0\n'
        'vllm:request_generation_tokens_count{model_name="Qwen3.8"} 8.0\n'
        'vllm:e2e_request_latency_seconds_sum{model_name="Qwen3.8"} 16.0\n'
        'vllm:e2e_request_latency_seconds_count{model_name="Qwen3.8"} 8.0\n')
TOKF0 = (TOKF.replace("} 4000.0", "} 2000.0").replace("} 400.0", "} 200.0")
        .replace("} 16.0", "} 8.0").replace("count{model_name=\"Qwen3.8\"} 8.0",
                                            "count{model_name=\"Qwen3.8\"} 4.0"))
tcur, tcap, tinfo = W.parse_metrics(TOKF)
tp, _, _ = W.parse_metrics(TOKF0)
ts = W.derive(tcur, tp, 10.0, tcap, tinfo, "observe", 0, [])
ck("a token family is not scaled like a seconds family",
   ts["rate"]["prompt_tokens_mean"] == 500.0 and ts["rate"]["generation_tokens_mean"] == 50.0
   and ts["since"]["prompt_tokens_mean"] == 500.0, str((ts["rate"]["prompt_tokens_mean"],
                                                        ts["since"]["prompt_tokens_mean"])))
ck("a seconds family still is", abs(ts["rate"]["e2e_ms"] - 2000.0) < 1e-9, str(ts["rate"]["e2e_ms"]))


# ------------------------------------------------------------------ (2) the request ring ---------
ring = W.Records(keep=3)
recs = [ring.start("/v1/chat/completions", "Qwen3.8", json.dumps({"model": "Qwen3.8", "stream": True}), True)
        for _ in range(5)]
for r in recs:
    ring.note(r, "hello")
    ring.finish(r, "completed", status=200)
ck("ring evicts at --keep", len(ring.items) == 3, f"{len(ring.items)} of 5")
ck("newest first", ring.listing()[0]["started_at"] >= ring.listing()[-1]["started_at"])
big = ring.start("/v1/completions", "Qwen3.8", "x" * (W.CAP + 10), False)
ring.note(big, "y" * (W.CAP + 10))
ring.finish(big, "completed", status=200)
d = ring.get(f"req-{big['seq']}")
ck("detail found by generated id", d is not None)
ck("input marked truncated", d["input"].endswith("[truncated]") and len(d["input"]) < W.CAP + 64)
ck("output capped at 256 KiB", d["output_truncated"] and len(d["output"]) == W.CAP)
ck("listing carries no bodies", "input" not in ring.listing()[0])
# ... but it does carry what the cards and rows read. A field that lives only in the detail makes a card
# fall back to something else while its label still describes the field it lost (see ROW_KEYS).
ck("a row carries the token counts and the timings the cards read",
   all(k in ring.listing()[0] for k in ("usage", "timings", "queue_s", "load_s")), str(sorted(ring.listing()[0])))
src = (Path(ROOT) / "serve" / "webui.py").read_text()
ck("both row builders list from one key list, so the page cannot lose a field on one backend",
   src.count("for k in ROW_KEYS") >= 2 and "usage" in W.ROW_KEYS and "timings" in W.ROW_KEYS
   and "boot" in W.ROW_KEYS, str(W.ROW_KEYS))
use = ring.start("/v1/chat/completions", "Qwen3.8", "{}", True)
use["usage"] = {"prompt_tokens": 11, "completion_tokens": 20}
use["first_token_s"] = 0.0
time.sleep(0.05)
ring.finish(use, "completed", status=200)
d = ring.get(f"req-{use['seq']}")
ck("decode speed from usage", (d["timings"]["predicted_per_second"] or 0) > 100, str(d["timings"]))
ck("estimate flagged only when it is one", d["timings"]["tokens_are_estimate"] is False
   and ring.get(f"req-{recs[-1]['seq']}")["timings"]["tokens_are_estimate"] is True)
# The same per-request prefill mean the engine ring computes, built from what a relay can time.
pre = ring.start("/v1/completions", "Qwen3.8", "{}", False)
pre["usage"], pre["first_token_s"], pre["queue_s"] = {"prompt_tokens": 1000, "completion_tokens": 4}, 0.5, 0.1
ring.finish(pre, "completed", status=200)
pt = ring.get(f"req-{pre['seq']}")["timings"]
ck("a relay record computes its prefill speed and names what it timed",
   abs((pt["prefill_tok_s"] or 0) - 2500) < 1 and "through this page" in pt["prefill_basis"], str(pt))
ck("no prefill speed is claimed when the first token was never seen",
   ring.get(f"req-{use['seq']}")["timings"]["prefill_tok_s"] is None, str(d["timings"]))


# ---------------------------------------------------------------- (2b) the request journal -------
# The point is the box, not the feature: this machine hard resets without ever writing a shutdown record, and
# the engine's request list dies with the launch. The rows describing what it was doing at the moment of death
# are worth more than the disk they sit on.
class Fin:                                                    # a FinishedRequestStats stand-in
    def __init__(s, i, prompt=50000, gen=8):
        s.request_id, s.num_prompt_tokens, s.num_generation_tokens = f"chatcmpl-{i}", prompt, gen
        s.num_cached_tokens, s.num_preemptions, s.max_tokens_param = 0, 0, 1024
        s.queued_time, s.prefill_time, s.inference_time, s.decode_time = 0.02, 5.0, 5.2, 0.2
        s.finish_reason = "stop"


# (one run at a time: these journal paths are fixed names, so two suites in flight at once read each
# other's half-written files and report a journal test failing that is not)
jpath = Path("/tmp/r9700-webui-journal.jsonl")
jpath.unlink(missing_ok=True)
j1 = W.Journal(jpath, keep=5)
e1 = W.EngineRecords(keep=5, journal=j1)
for i in range(7):
    e1.add(Fin(i))
ck("finished requests are journalled, one line each",
   len(jpath.read_text().splitlines()) == 7, str(len(jpath.read_text().splitlines())))
ck("the journal holds rows, not bodies", "input" not in jpath.read_text() and "_in" not in jpath.read_text())
j2 = W.Journal(jpath, keep=5)
e2 = W.EngineRecords(keep=5, journal=j2)
ck("a new process replays the last rows, newest first",
   [r["id"] for r in e2.listing()] == [f"chatcmpl-{i}" for i in (6, 5, 4, 3, 2)],
   str([r["id"] for r in e2.listing()]))
ck("replay is capped at --keep, so the file can outgrow the page without the page growing", len(e2) == 5, str(len(e2)))
ck("seq continues past the restored rows instead of colliding with them",
   e2.add(Fin(99)).get("seq") == 8, str(e2.listing()[0]["seq"]))
ck("a restored row carries the launch it came from, so two boots do not read as one odd afternoon",
   jpath.read_text().splitlines()[-1].find('"boot"') > 0
   and e2.listing()[0]["boot"] == j1.boot, str(e2.listing()[0])[:160])
jpath.write_text(jpath.read_text() + '{"seq": 4242, "state": "compl')      # the write a reset cut in half
ck("a torn trailing line is dropped, not rendered as a request",
   len(W.Journal(jpath, keep=5).replay()) == 5 and all(r["seq"] != 4242 for r in W.Journal(jpath, keep=5).replay()),
   str([r["seq"] for r in W.Journal(jpath, keep=5).replay()]))
# A relay row restored from the journal has no captured bodies -- the detail pane must say so instead of
# raising on the missing keys, which is what the first version of this did.
j3 = W.Journal("/tmp/r9700-webui-journal-relay.jsonl", keep=5)
r1 = W.Records(keep=5, journal=j3)
c1 = r1.start("/v1/chat/completions", "Qwen3.8", '{"hello": 1}', True)
c1["usage"] = {"prompt_tokens": 11, "completion_tokens": 20}
r1.finish(c1, "completed", status=200)
r2 = W.Records(keep=5, journal=W.Journal("/tmp/r9700-webui-journal-relay.jsonl", keep=5))
det = r2.get(f"req-{c1['seq']}")
ck("a relay row restored from the journal still opens, and says its text did not survive",
   det is not None and det["input"] is None and "journal" in (det.get("input_note") or ""), str(det)[:160])
t0 = time.perf_counter()
for i in range(20):
    r1.finish(r1.start("/v1/chat/completions", "Qwen3.8", "{}", True), "completed", status=200)
app_ms = (time.perf_counter() - t0) / 20 * 1e3
# 2.0 ms measured on this box per journalled request. The bound is loose on purpose -- it is here to catch the
# journal becoming a per-token cost or a stall, not to police disk jitter. Requests finish seconds apart; the
# fsync is what makes the row survive a hard reset, which is the whole reason the file exists.
ck("journal cost stays one fsync per finished request, not per token", app_ms < 25.0, f"{app_ms:.2f} ms each")
bad = W.Journal("/proc/definitely-not-writable/journal.jsonl", keep=5)
ck("a journal that cannot be opened degrades to memory and says why, rather than breaking the serve",
   bad.f is None and bad.error and bad.append({"seq": 1}) is False and bad.replay() == [], str(bad.error))
_web = Path(__file__).resolve().parent.parent / "serve" / "web"
ck("the page is told about the journal, so a restored list is not mistaken for this launch's work",
   all(x in (_web / "monitor.js").read_text()
       for x in ("request journal", "carried over", "journal FAILED", "this launch only"))
   and 'id="journal-line"' in (_web / "monitor.html").read_text())
# The launch and the plugin meet only through this variable's name, so a rename on one side is a silent
# memory-only Monitor -- the kind of break that shows up weeks later when someone asks what died.
ck("the launch and the plugin agree on the journal's env var name",
   "R9700_WEBUI_JOURNAL" in (Path(ROOT) / "serve/native-flashnext.sh").read_text()
   and "R9700_WEBUI_JOURNAL" in (Path(ROOT) / "serve/plugin/r9700_webui/__init__.py").read_text())


# ------------------------------------------------------- (2c) shared_decode: what a decode rate means --
# Measured off 1,400 real requests: a stream runs 158 tok/s alone, 119 at two in flight, 23 at three or four
# (notes/context-262k.md). A decode number printed without the concurrency beside it is unreadable -- the reader
# cannot tell a slow request from a busy box -- so the listing carries the concurrency, computed rather than
# guessed. These are the windows that computation has to get right.
def _rq(s, w, dec=None, first=None, q=None):
    t = {} if dec is None and q is None else {"decode_s": dec, "queue_s": q}
    return {"started_at": s, "wallclock_s": w, "first_token_s": first, "timings": t or None}


ck("a request alone in the engine shares with itself only", W.shared_decode([_rq(0, 10, dec=4.0)])[0] == 1.0,
   str(W.shared_decode([_rq(0, 10, dec=4.0)])))
ck("two requests with identical decode spans each report x2",
   W.shared_decode([_rq(0, 10, dec=4.0), _rq(0, 10, dec=4.0)]) == [2.0, 2.0],
   str(W.shared_decode([_rq(0, 10, dec=4.0), _rq(0, 10, dec=4.0)])))
_v = W.shared_decode([_rq(0, 10, dec=4.0), _rq(0, 14, dec=8.0)])
ck("a partial overlap reports the mean across its own span, not the peak it touched", _v == [2.0, 1.5], str(_v))
# The clock-basis trap, from a real row: an engine row's wallclock_s is vLLM's inference_time (queue NOT
# included) while its first_token_s is queue + prefill -- first token 21.8 s against a 10.8 s wallclock, because
# it queued 12.3 s before the engine started. A window built as [start + first_token, start + wallclock] is
# inverted there, which is why the span is anchored at the END minus decode_s instead.
_q = [{"started_at": 100.0, "wallclock_s": 10.8, "first_token_s": 21.8,
       "timings": {"decode_s": 1.25, "queue_s": 12.3, "prefill_s": 9.55}},
      {"started_at": 100.0, "wallclock_s": 11.0, "first_token_s": 5.0, "timings": {"decode_s": 10.0}}]
_qv = W.shared_decode(_q)
ck("a queue longer than the wallclock cannot invert an engine row's decode window",
   abs(_qv[0] - 2.0) < 1e-9 and abs(_qv[1] - 1.12) < 0.02, str(_qv))
# (these seven fixture rows all land within the same second, so they really do share: the check is that the
# field is there and at least counts the row itself, not that it reads 1.0)
ck("the listing carries the derived field, and the journal does not write it to disk",
   isinstance(e1.listing()[0].get("shared_decode"), float) and e1.listing()[0]["shared_decode"] >= 1.0
   and "shared_decode" not in jpath.read_text(), str(e1.listing()[0])[:160])
ck("so does the proxy path's listing, which is the other way a row arrives",
   r1.listing()[0].get("shared_decode") is not None, str(r1.listing()[0])[:120])

# ------------------------------------------------------------------ (3) the sysfs read -----------
fake = Path("/tmp/r9700-webui-sysfs")
shutil.rmtree(fake, ignore_errors=True)                       # built from scratch every run
for card, (busy, used, total, temp, bus) in {"card0": (87, 31 * 2**30, 32 * 2**30, 61000, "c3:00.0"),
                                             "card3": (0, 2 * 2**30, 32 * 2**30, 33000, "06:00.0")}.items():
    dev = fake / card / "device"
    dev.mkdir(parents=True, exist_ok=True)
    (dev / "uevent").write_text(f"DRIVER=amdgpu\nPCI_SLOT_NAME=0000:{bus}\nMAJOR=226\n")
    (dev / "gpu_busy_percent").write_text(f"{busy}\n")
    (dev / "mem_busy_percent").write_text("41\n")
    (dev / "mem_info_vram_used").write_text(f"{used}\n")
    (dev / "mem_info_vram_total").write_text(f"{total}\n")
    (dev / "product_name").write_text("Radeon AI PRO R9700\n")
    if card != "card0":
        continue                                            # card3's sensors: the nested layout, below
    hm = dev / "hwmon0"                                      # card0: the older device/hwmonN layout
    hm.mkdir(exist_ok=True)
    (hm / "name").write_text("amdgpu\n")
    (hm / "temp1_input").write_text(f"{temp}\n")
    (hm / "power1_average").write_text("199870000\n")            # this kernel: average, not input
    (hm / "freq1_label").write_text("sclk\n")
    (hm / "freq1_input").write_text("1180000000\n")
    (hm / "freq2_label").write_text("mclk\n")
    (hm / "freq2_input").write_text("2520000000\n")
(fake / "card1").mkdir(parents=True, exist_ok=True)          # a DRM node with no amdgpu counters
# card3's sensors live in the OTHER layout this box actually runs: device/hwmon/hwmonN, not device/hwmonN.
# Globbing only the flat form is how the Monitor showed a blank temperature, power and clock for weeks on a
# machine whose files read fine (found 2026-11-02: temp1_input 65000, power1_average 205000000 sitting there
# under hwmon/hwmon2 while the page drew "—").
deep = fake / "card3" / "device" / "hwmon" / "hwmon2"
deep.mkdir(parents=True, exist_ok=True)
(deep / "name").write_text("amdgpu\n")
(deep / "temp1_input").write_text("33000\n")
# ...and labelled, because this box's kernel labels all three of its temperature sensors (edge / junction /
# mem), and the headline sensor has to be the junction: measured on the live box, one card's edge ran 9-11 C
# above its neighbours while its junction matched theirs and its GDDR7 ran 10-13 C hotter. temp1 alone cannot
# see that at all.
(deep / "temp1_label").write_text("edge\n")
(deep / "temp2_label").write_text("junction\n")
(deep / "temp2_input").write_text("45000\n")
(deep / "temp2_crit").write_text("110000\n")
(deep / "temp3_label").write_text("mem\n")
(deep / "temp3_input").write_text("41000\n")
(deep / "temp3_crit").write_text("108000\n")
(deep / "temp3_emergency").write_text("113000\n")
(deep / "fan1_input").write_text("4400\n")
(deep / "pwm1").write_text("186\n")
(deep / "power1_average").write_text("199870000\n")
(deep / "power1_cap").write_text("205000000\n")
(deep / "freq1_label").write_text("sclk\n")
(deep / "freq1_input").write_text("1180000000\n")
# and one that refuses to answer: on this box gpu_busy_percent and the whole hwmon set come back EBUSY
# -- with no compute context open as well as with one -- so that is a state the reader has to survive.
# It must read as "not readable", never as "0% busy".
stub = fake / "card5" / "device"
stub.mkdir(parents=True, exist_ok=True)
stub.joinpath("uevent").write_text("DRIVER=amdgpu\nPCI_SLOT_NAME=0000:08:00.0\n")
(stub / "gpu_busy_percent").mkdir()                          # reading it raises, like EBUSY does
(stub / "mem_info_vram_total").write_text(f"{32 * 2**30}\n")

# ------------------------------------------------------------------ the PCI tree the link card reads
# Built the way sysfs actually looks: the devices live in a physical topology under devices/pciNNNN:BB and
# /sys/bus/pci/devices/<bdf> is a symlink into it -- that is how the chain of bridges above a card is found
# without parsing lspci. The fixture carries a four-level chain (endpoint, downstream port, upstream port, root
# port), a second chain that came up x8 on an x16 link, and a root port with no AER files at all.
fakepci = Path("/tmp/r9700-webui-sysfs-pci")
shutil.rmtree(fakepci, ignore_errors=True)
pnodevs = fakepci / "bus" / "pci" / "devices"
pnodevs.mkdir(parents=True)


def pci_dev(parent, bdf, driver="pcieport", speed="32.0 GT/s PCIe", width="16",
            max_speed="32.0 GT/s PCIe", max_width="16", aer=None, state="D0"):
    d = parent / bdf
    d.mkdir(parents=True, exist_ok=True)
    (d / "uevent").write_text(f"DRIVER={driver}\nPCI_SLOT_NAME=0000:{bdf}\n")
    (d / "current_link_speed").write_text(speed + "\n")
    (d / "current_link_width").write_text(width + "\n")
    (d / "max_link_speed").write_text(max_speed + "\n")
    (d / "max_link_width").write_text(max_width + "\n")
    (d / "power_state").write_text(state + "\n")
    for name, rows in (aer or {}).items():
        (d / name).write_text("".join(f"{k} {v}\n" for k, v in rows))
    (pnodevs / bdf).symlink_to(d, target_is_directory=True)     # sysfs links the same way, absolute here
    return d


AER_COR = [("RxErr", 0), ("BadTLP", 0), ("BadDLLP", 0), ("TOTAL_ERR_COR", 0)]
AER_NF = [("DLP", 0), ("TOTAL_ERR_NONFATAL", 0)]
AER_FT = [("DLP", 0), ("TOTAL_ERR_FATAL", 0)]
root_a = pci_dev(fakepci / "devices" / "pci0000:c0", "0000:c0:03.1")          # root port, no AER files here
# chain A idles: the endpoint is runtime-suspended (D3hot) and its upstream port has dropped to Gen1, which
# is what every chain on this box reads between launches (measured 2026-11-03 03:00, nothing running).
up_a = pci_dev(root_a, "0000:c4:00.0", speed="2.5 GT/s PCIe", aer={"aer_dev_correctable":
                                            [("RxErr", 0), ("BadTLP", 24), ("NonFatalErr", 1),
                                             ("TOTAL_ERR_COR", 24)],
                                            "aer_dev_nonfatal": AER_NF, "aer_dev_fatal": AER_FT})
down_a = pci_dev(up_a, "0000:c5:00.0", aer={"aer_dev_correctable":
                                            [("BadTLP", 7), ("TOTAL_ERR_COR", 7)],
                                            "aer_dev_nonfatal": AER_NF, "aer_dev_fatal": AER_FT})
endpoint_a = pci_dev(down_a, "0000:c6:00.0", driver="amdgpu", state="D3hot", aer={"aer_dev_correctable":
                                                                   [("BadTLP", 1), ("TOTAL_ERR_COR", 1)],
                                                                   "aer_dev_nonfatal": AER_NF,
                                                                   "aer_dev_fatal": AER_FT})
root_b = pci_dev(fakepci / "devices" / "pci0000:00", "0000:00:03.1", aer={"aer_dev_correctable": AER_COR,
                                                                          "aer_dev_nonfatal": AER_NF,
                                                                          "aer_dev_fatal": AER_FT})
up_b = pci_dev(root_b, "0000:04:00.0", aer={"aer_dev_correctable": AER_COR,
                                            "aer_dev_nonfatal": AER_NF, "aer_dev_fatal": AER_FT})
down_b = pci_dev(up_b, "0000:05:00.0", aer={"aer_dev_correctable": AER_COR,
                                            "aer_dev_nonfatal": AER_NF, "aer_dev_fatal": AER_FT})
endpoint_b = pci_dev(down_b, "0000:06:00.0", driver="amdgpu", width="8",      # came up x8 on an x16 link
                     aer={"aer_dev_correctable": AER_COR, "aer_dev_nonfatal": AER_NF,
                          "aer_dev_fatal": AER_FT})
pci_dev(fakepci / "devices" / "pci0000:02", "0000:07:00.0", driver="nvme")     # not a GPU chain, ignore it
gpus = W.read_gpus(fake)
ck("both hwmon layouts are read (the kernel here uses device/hwmon/hwmonN)",
   any(g["card"] == "card3" and g["temp_c"] == 45.0 and g["power_w"] == 199.87 and g["power_cap_w"] == 205.0
       for g in gpus), str([(g["card"], g["temp_c"], g["power_w"], g.get("power_cap_w")) for g in gpus]))
# The three temperature sensors are read by label, and the headline is the junction -- the limit-bearing,
# fan-relevant one. card0's fixture has no labels at all, which is what an older kernel looks like, and keeps
# temp1 as the headline rather than reporting nothing.
_c3 = next(g for g in gpus if g["card"] == "card3")
ck("temperatures are read by label: junction is temp_c, edge and mem ride along, and the crit limit comes with them",
   _c3["temp_c"] == 45.0 and _c3["temp_edge_c"] == 33.0 and _c3["temp_mem_c"] == 41.0
   and _c3["temp_crit_c"] == 110.0 and _c3["temp_mem_crit_c"] == 108.0
   and _c3["temp_mem_emerg_c"] == 113.0, str(_c3))
ck("an unlabelled kernel still gets its temp1 as the headline",
   gpus[0]["temp_c"] == 61.0 and gpus[0]["temp_junction_c"] is None, str(gpus[0]))
ck("fan rpm and duty ride the payload, because the fan map follows the junction and can be calm while the "
   "memory is the hottest thing on the box",
   _c3["fan_rpm"] == 4400.0 and _c3["pwm"] == 186.0, str(_c3))
ck("only cards with amdgpu counters", [g["card"] for g in gpus] == ["card0", "card3", "card5"],
   str([g["card"] for g in gpus]))
ck("busy, VRAM and temperature read", gpus[0]["busy_pct"] == 87 and gpus[0]["vram_total"] == 32 * 2**30
   and gpus[0]["temp_c"] == 61.0)
ck("card name", gpus[0]["name"] == "Radeon AI PRO R9700")
ck("power read from power1_average", abs(gpus[0]["power_w"] - 199.87) < 1e-6, str(gpus[0]["power_w"]))
ck("sclk and mclk by label, not by index", gpus[0]["sclk_mhz"] == 1180 and gpus[0]["mclk_mhz"] == 2520,
   f"{gpus[0]['sclk_mhz']} / {gpus[0]['mclk_mhz']}")
ck("an unreadable attribute is named, not zeroed", "busy_pct" not in gpus[2]
   and gpus[2]["unread"] == ["gpu_busy_percent"], str(gpus[2].get("unread")))
ck("a card that answers some things still shows", gpus[2]["vram_total"] == 32 * 2**30)
ck("real sysfs is readable without raising", isinstance(W.read_gpus("/sys/class/drm"), list))
ck("the PCI bus id is carried for joining", next(g for g in gpus if g["card"] == "card0")["bus"] == "c3:00.0",
   str(next(g for g in gpus if g["card"] == "card0")["bus"]))

# ---- rocm-smi as the fallback, joined by PCI bus id (opt-in: it opens /dev/kfd, so never automatic) --
smi = fake / "smi"
smi.mkdir(exist_ok=True)
(smi / "rocm-smi").write_text("""#!/bin/sh
cat <<'EOF'
==================== the banner rocm-smi prints before the JSON ====================
{"card0": {"Device Name": "Radeon AI PRO R9700", "PCI Bus": "0000:C3:00.0", "GPU use (%)": "55",
           "VRAM Total Memory (B)": "34208743424", "VRAM Total Used Memory (B)": "7000000000",
           "Temperature (sensor 'edge' 0) (C)": "66", "Average Graphics Package Power (W)": "N/A",
           "Current Clock Frequency (sclk)": "1200", "Current Clock Frequency (mclk)": "2520"},
 "card7": {"Device Name": "Radeon AI PRO R9700", "PCI Bus": "0000:05:00.0", "GPU use (%)": "12",
           "VRAM Total Used Memory (B)": "1000", "Average Graphics Package Power (W)": "43"},
 "card8": {"PCI Bus": "0000:08:00.0", "GPU use (%)": "77"}}
EOF
""")
os.chmod(smi / "rocm-smi", 0o755)
by_bus, err = W.read_smi([str(smi / "rocm-smi")])
ck("rocm-smi JSON read out through its banner", by_bus and err is None, str(err))
ck("joined by PCI bus, not by index", "c3:00.0" in by_bus and "05:00.0" in by_bus, str(sorted(by_bus)))
ck("an N/A answer is dropped, not zeroed", by_bus["c3:00.0"].get("power_w") is None)
ck("an unknown command is an error string, not a crash", W.read_smi([str(smi / "nope")])[1] is not None)
sys_cards = W.read_gpus(fake)
W.merge_smi(sys_cards, by_bus)
c0 = next(g for g in sys_cards if g["card"] == "card0")
ck("the sysfs answer wins where it answered", c0["busy_pct"] == 87 and c0["temp_c"] == 61.0
   and c0["sclk_mhz"] == 1180.0, str({k: c0.get(k) for k in ("busy_pct", "temp_c", "sclk_mhz")}))
ck("rocm-smi fills only the gaps", c0["from_smi"] == [], str(c0["from_smi"]))
stub0 = next(g for g in sys_cards if g["card"] == "card5")
ck("rocm-smi answers what the sysfs refused", stub0["busy_pct"] == 77.0 and stub0["from_smi"] == ["busy_pct"],
   f"{stub0.get('busy_pct')} {stub0.get('from_smi')}")
ck("a card only rocm-smi knows about turns up as its own row",
   any(g["card"] == "card7" and g["busy_pct"] == 12 and "busy_pct" in g["from_smi"] for g in sys_cards),
   str([g["card"] for g in sys_cards]))


# ------------------------------------------------------------------ (4) end to end ---------------
class FakeVLLM(BaseHTTPRequestHandler):
    """A stand-in serve: /health, /v1/models, /metrics with moving counters, and a streaming and a
    non-streaming completion."""
    n_metrics = 0

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        raw = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, "")
        elif self.path == "/v1/models":
            self._send(200, json.dumps({"object": "list", "data": [
                {"id": "Qwen3.8", "object": "model", "max_model_len": 32768}]}))
        elif self.path == "/metrics":
            FakeVLLM.n_metrics += 1
            self._send(200, METRICS if FakeVLLM.n_metrics == 1 else METRICS2, "text/plain")
        else:
            self._send(404, json.dumps({"error": {"message": "no " + self.path}}))

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if body.get("boom"):
            self._send(500, json.dumps({"error": {"message": "engine is gone"}}))
        elif body.get("slow"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for i in range(4):
                self.wfile.write(("data: " + json.dumps({"id": "chatcmpl-slow",
                                                          "choices": [{"index": 0, "delta": {"content": f" s{i}"}}]})
                                  + "\n\n").encode())
                self.wfile.flush()
                time.sleep(0.12)                      # a stream that arrives over time, on purpose
            self.wfile.write(('data: ' + json.dumps({"id": "chatcmpl-slow", "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 4}}) + "\n\n\ndata: [DONE]\n\n").encode())
            self.wfile.flush()
            self.close_connection = True              # no length, no chunking: the body ends at the socket
        elif body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for i, tok in enumerate(["Hello", " there", "!"]):
                self.wfile.write(("data: " + json.dumps({"id": "chatcmpl-1", "object": "chat.completion.chunk",
                                                         "choices": [{"index": 0, "delta": {"content": tok}}]})
                                  + "\n\n").encode())
                self.wfile.flush()
                time.sleep(0.02)
            self.wfile.write(('data: ' + json.dumps({"id": "chatcmpl-1", "object": "chat.completion.chunk",
                                                     "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                                                     "usage": {"prompt_tokens": 12, "completion_tokens": 3}})
                              + "\n\n\ndata: [DONE]\n\n").encode())
            self.wfile.flush()
        else:
            self._send(200, json.dumps({"id": "chatcmpl-2", "object": "chat.completion",
                                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"},
                                                     "finish_reason": "stop"}],
                                        "usage": {"prompt_tokens": 7, "completion_tokens": 1}}))


up = ThreadingHTTPServer(("127.0.0.1", 0), FakeVLLM)
up_port = up.server_address[1]
threading.Thread(target=up.serve_forever, daemon=True).start()


_jsid = iter(range(1000))


def sidecar(mode="proxy", key="", keep=50, journal=None):
    # Each instance gets its own journal file: two sidecars sharing one path would replay each other's rows,
    # which is the confusion the launch avoids by naming the file after the port it serves.
    jp = journal or f"/tmp/r9700-webui-journal-{mode}-{next(_jsid)}.jsonl"
    if journal is None:
        Path(jp).unlink(missing_ok=True)                    # a named journal is reused on purpose
    cfg = {"port": 0, "upstream": f"http://127.0.0.1:{up_port}", "mode": mode, "key": key, "poll": 0.2,
           "keep": keep, "meta": "", "gpus": str(fake), "pci": str(pnodevs), "journal": jp, "smi": None,
           "timeout": 10.0, "verbose": False, "version": "test"}
    hub, srv = W.build(cfg)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return hub, srv, srv.server_address[1]


def call(port, path, body=None, key=None, raw_stream=False):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {key}"} if key else {})})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = r.read()
        return r.status, (data if raw_stream else json.loads(data or b"null"))


# A fresh journal for the running sidecar, or every run of this file would replay the last run's rows and the
# counts below would be about a launch that happened an hour ago.
MAIN_JOURNAL = "/tmp/r9700-webui-journal-main.jsonl"
Path(MAIN_JOURNAL).unlink(missing_ok=True)
hub, srv, port = sidecar("proxy", journal=MAIN_JOURNAL)
time.sleep(0.6)                                            # let the poll threads land a window

status, body = call(port, "/healthz")
ck("healthz", status == 200 and body["ok"] and body["upstream_reachable"], str(body))

status, s = call(port, "/api/status")
ck("status reads the engine", s["reachable"] and s["model_name"] == "Qwen3.8", str(s)[:120])
ck("status carries config labels", s["config"].get("block_size") == "1088")
ck("status carries cards", len(s["gpus"]) == 3 and s["gpus"][0]["busy_pct"] == 87, str(len(s.get("gpus", []))))
# Per-card history, so a card's busy% and VRAM are waveforms like everything else. Two things the test
# insists on: the ring is capped (a page open for a day must not grow the payload), and a card that refuses
# to answer (the fixture's card5, standing in for EBUSY on a live box) is a null in the ring, not a 0 --
# otherwise a card that stops answering plots as a card that went idle.
_gh = s["gpus_history"]
ck("per-card history rides along, capped at the same ring",
   _gh and all(len(e["cards"]) == 3 for e in _gh) and len(_gh) <= W.HIST,
   f"{len(_gh)} samples x {len(_gh[0]['cards']) if _gh else 0} cards (HIST={W.HIST})")
ck("a card that refuses sysfs reads is a gap in its own waveform, not a zero",
   all(c["busy_pct"] is None for e in _gh for c in e["cards"] if c["card"] == "card5")
   and any(c["busy_pct"] == 87 for e in _gh for c in e["cards"] if c["card"] == "card0"),
   str(_gh[-1]["cards"]) if _gh else "no history")
ck("the slot carries the leak question too (vram_used, and the card's own total)",
   all(k in _gh[-1]["cards"][0] for k in ("vram_used", "vram_total", "bus", "sclk_mhz")), str(_gh[-1]["cards"][0]))
# The thermal sensors ride the ring too, which is what the per-card W and °C waves are drawn from.
ck("temperature, power and the card's power cap ride the ring",
   all(k in _gh[-1]["cards"][0] for k in ("temp_c", "power_w", "power_cap_w")), str(_gh[-1]["cards"][0]))

# ---- PCIe: the link the cards hang off. Byte counters answer N/A on these cards, so what is collected is
# negotiated link state and the kernel's AER error counters -- and the incident's evidence.
_pc = W.read_pcie(pnodevs)
_ck_ch = {c["bus"]: c for c in _pc}
ck("PCIe chains walk the sysfs topology: endpoint, downstream port, upstream port, root port",
   len(_pc) == 2 and [d["bdf"] for d in _ck_ch["c6:00.0"]["devices"]]
   == ["0000:c6:00.0", "0000:c5:00.0", "0000:c4:00.0", "0000:c0:03.1"],
   str([(c["bus"], [d["bdf"] for d in c["devices"]]) for c in _pc]))
ck("only GPU endpoints start a chain (an nvme on the box is not one)",
   set(_ck_ch) == {"c6:00.0", "06:00.0"}, str(sorted(_ck_ch)))
ck("multi-digit AER counts parse -- 24 BadTLP is exactly the number that matters on the live box",
   _ck_ch["c6:00.0"]["devices"][2]["cor"] == 24
   and _ck_ch["c6:00.0"]["devices"][2]["notable"].get("BadTLP") == 24,
   str(_ck_ch["c6:00.0"]["devices"][2]))
ck("a bridge whose AER files are missing reads as unknown, never as zero",
   _ck_ch["c6:00.0"]["devices"][3]["cor"] is None, str(_ck_ch["c6:00.0"]["devices"][3]))
ck("a link that came up x8 on an x16 lane is flagged, and the chain reports its weakest link",
   _ck_ch["06:00.0"]["degraded"] and _ck_ch["06:00.0"]["width"] == 8 and not _ck_ch["06:00.0"]["parked"],
   str([(c["bus"], c["width"], c["degraded"]) for c in _pc]))
# Parked is not damaged. With the GPUs runtime-suspended the switch links drop to Gen1 and EVERY chain reads
# "below maximum" -- the panel alarming on all four at an idle box is how a warning stops being read at all.
ck("a suspended card is reported as parked, so an idle Gen1 link is not painted as a fault",
   _ck_ch["c6:00.0"]["degraded"] and _ck_ch["c6:00.0"]["parked"] and not _ck_ch["06:00.0"]["parked"]
   and _ck_ch["c6:00.0"]["devices"][0]["power_state"] == "D3hot",
   str([(c["bus"], c["degraded"], c["parked"]) for c in _pc]))
_d1, _last = W.aer_deltas(_pc, {})
ck("the parked flag rides the panel rows, which is where the colour decision is made",
   all("parked" in d for d in _d1), str([(d["bus"], d.get("parked")) for d in _d1]))
ck("the first window has no delta to show, which is a gap rather than a zero",
   all(d["cor_d"] is None for d in _d1), str([(d["bus"], d["cor_d"]) for d in _d1]))
(down_a / "aer_dev_correctable").write_text("BadTLP 12\nTOTAL_ERR_COR 12\n")      # five more, one bridge
_d2, _last2 = W.aer_deltas(W.read_pcie(pnodevs), _last)
ck("the AER wave is the per-window delta, not the cumulative counter",
   [d for d in _d2 if d["bus"] == "c6:00.0"][0]["cor_d"] == 5,
   str([(d["bus"], d["cor_d"]) for d in _d2]))
(down_a / "aer_dev_correctable").write_text("BadTLP 1\nTOTAL_ERR_COR 1\n")        # kernel zeroed it
_d3, _ = W.aer_deltas(W.read_pcie(pnodevs), _last2)
ck("a counter that went down is a reset: a gap and a flag, never a bar of -11",
   [d for d in _d3 if d["bus"] == "c6:00.0"][0]["cor_d"] is None
   and [d for d in _d3 if d["bus"] == "c6:00.0"][0]["reset"],
   str([(d["bus"], d["cor_d"], d["reset"]) for d in _d3]))
_named, _ev = {}, []
W.aer_deltas(W.read_pcie(pnodevs), {}, _named, _ev, 100.0)
ck("a counter already non-zero when we start watching sets a baseline, it is not an event we witnessed",
   not _ev, str(_ev[:3]))
(down_a / "aer_dev_correctable").write_text("BadTLP 4\nBadDLLP 1\nTOTAL_ERR_COR 5\n")
W.aer_deltas(W.read_pcie(pnodevs), {}, _named, _ev, 137.0)
ck("an AER event carries its time, its device and which counter moved",
   len(_ev) == 2 and {e["kind"] for e in _ev} == {"BadTLP", "BadDLLP"}
   and all(e["bdf"] == "0000:c5:00.0" and e["at"] == 137.0 and e["bus"] == "c6:00.0" for e in _ev),
   str(_ev))
ck("PCIe chain state, its ring and its event list ride the payload, the ring capped at the same HIST",
   "pcie_events" in s and s.get("pcie") and s.get("pcie_history")
   and len(s["pcie_history"]) <= W.HIST
   and all(len(h["chains"]) == len(s["pcie"]) for h in s["pcie_history"]),
   f"{len(s.get('pcie_history') or [])} samples x {len(s.get('pcie') or [])} chains")
ck("the payload names the journal so the page can say where a restored row came from",
   bool(s.get("journal")) and s["journal"]["ok"] and s["journal"]["path"].endswith("journal-main.jsonl"),
   str(s.get("journal")))
ck("status carries models", s["models"][0]["id"] == "Qwen3.8")
ck("status engine version", s["engine"]["vllm_version"] == "test")
ck("the sparkline ring carries the live rates too",
   s["history"] and all(k in s["history"][-1] for k in ("gen_tok_s", "stream_tok_s", "kv_tok_s", "running")),
   str(s["history"][-1] if s["history"] else "no history"))
ck("the payload says how often it is polled",
   isinstance(s.get("poll_s"), float) and s["poll_s"] == 0.2, str(s.get("poll_s")))

status, r = call(port, "/v1/chat/completions", {"model": "Qwen3.8", "stream": True,
                                                "messages": [{"role": "user", "content": "hi"}]}, raw_stream=True)
sse = r.decode()
ck("stream relayed byte for byte", "data: [DONE]" in sse and sse.count("delta") == 4, f"{len(sse)} bytes")
time.sleep(0.1)
status, lst = call(port, "/api/requests")
rec = lst["requests"][0]
ck("stream recorded", rec["state"] == "completed" and rec["id"] == "chatcmpl-1" and rec["http_status"] == 200, str(rec))
ck("first token stamped", isinstance(rec["first_token_s"], float) and 0.0 <= rec["first_token_s"] <= rec["wallclock_s"])
ck("wall-clock closed", isinstance(rec["wallclock_s"], float) and rec["wallclock_s"] >= rec["first_token_s"])
status, det = call(port, f"/api/requests?id={rec['id']}")
ck("detail: input kept", json.loads(det["input"])["messages"][0]["content"] == "hi")
ck("detail: streamed text concatenated", det["output"] == "Hello there!")
ck("detail: usage from the final chunk", det["usage"] == {"prompt_tokens": 12, "completion_tokens": 3})
ck("detail: response body kept", "chatcmpl-1" in det["response"])
ck("detail: decode speed computed", det["timings"]["predicted_per_second"] > 0 and not det["timings"]["tokens_are_estimate"])

call(port, "/v1/chat/completions", {"model": "Qwen3.8", "messages": []})
time.sleep(0.1)
status, det = call(port, "/api/requests?id=chatcmpl-2")
ck("non-stream recorded", det["state"] == "completed" and det["output"] == "Hi")
ck("non-stream usage read", det["usage"]["completion_tokens"] == 1)

try:
    status, err = call(port, "/v1/chat/completions", {"model": "Qwen3.8", "boom": True})
except urllib.error.HTTPError as e:
    status, err = e.code, json.loads(e.read() or b"{}")
ck("a 500 relayed as it was", status == 500 and err["error"]["message"] == "engine is gone", str(status))
time.sleep(0.1)
bad = [r for r in call(port, "/api/requests")[1]["requests"] if r["state"] == "error"]
ck("a 500 recorded as an error", len(bad) == 1 and bad[0]["http_status"] == 500, str(bad))
with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as r:
    ck("a non-API GET is relayed", r.status == 200 and r.read().startswith(b"# HELP vllm:"))
ck("and not recorded", len([x for x in call(port, "/api/requests")[1]["requests"]
                            if x["path"] == "/metrics"]) == 0)

# The end the journal exists for: restart the process and the rows come back. This box does it on its own -- a
# hard reset, no shutdown record, and a Monitor that used to come back claiming nothing had ever run.
_, _srv_j, _port_j = sidecar("proxy", journal=MAIN_JOURNAL)
time.sleep(0.6)
_j = call(_port_j, "/api/status")[1]["journal"]
_restored = call(_port_j, "/api/requests")[1]["requests"]
ck("a restarted process comes back with the previous launch's rows",
   len(_restored) >= 3 and all(r.get("boot") for r in _restored) and _j["restored"] == len(_restored),
   f"{len(_restored)} rows restored, journal says {_j['restored']}")
ck("restored rows name the launch they came from, which is not this one",
   _restored[0]["boot"] != _j["boot"], str([_restored[0]["boot"], _j["boot"]]))
_srv_j.shutdown()

# the stream must be pumped as it arrives, not buffered to the end: if the first bytes reach the client
# only when the upstream finishes, every first-token number on the page -- and the page's whole reason
# for sitting in the path -- is wrong
conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
conn.request("POST", "/v1/chat/completions",
             body=json.dumps({"model": "Qwen3.8", "slow": True,
                              "messages": [{"role": "user", "content": "hi"}], "stream": True}),
             headers={"Content-Type": "application/json"})
resp = conn.getresponse()
t0, first = time.time(), None
while True:
    piece = resp.read1(4096)
    if not piece:
        break
    if first is None:
        first = time.time() - t0
whole = time.time() - t0
conn.close()
ck("a stream is relayed as it arrives, not buffered to the end",
   first is not None and first < whole - 0.15, f"first bytes at {first:.2f}s of {whole:.2f}s")

hub2, srv2, port2 = sidecar("observe")
time.sleep(0.3)
status, body = call(port2, "/v1/chat/completions", {"model": "Qwen3.8", "stream": True, "messages": []},
                     raw_stream=True)
ck("observe mode still relays", "data: [DONE]" in body.decode())
ck("observe mode records nothing", call(port2, "/api/requests")[1]["requests"] == [])
ck("observe mode says so", call(port2, "/api/status")[1]["mode"] == "observe")

_, srv3, port3 = sidecar("proxy", key="sekret")
try:
    call(port3, "/api/status")
    ck("keyed sidecar refuses unkeyed /api", False)
except urllib.error.HTTPError as e:
    ck("keyed sidecar refuses unkeyed /api", e.code == 401, str(e.code))
try:
    call(port3, "/v1/models")
    ck("keyed sidecar refuses an unkeyed relay", False)
except urllib.error.HTTPError as e:
    ck("keyed sidecar refuses an unkeyed relay", e.code == 401, str(e.code))
ck("keyed sidecar answers with the key", call(port3, "/api/status", key="sekret")[0] == 200)
with urllib.request.urlopen(f"http://127.0.0.1:{port3}/monitor", timeout=5) as r:
    page = r.read().decode()
ck("/monitor is the Monitor", r.status == 200 and "<h1>Monitor</h1>" in page and "/web/monitor.js" in page)
with urllib.request.urlopen(f"http://127.0.0.1:{port3}/", timeout=5) as r:
    ck("/ shows the Monitor", r.status == 200 and "<h1>Monitor</h1>" in r.read().decode())
with urllib.request.urlopen(f"http://127.0.0.1:{port3}/about", timeout=5) as r:
    ck("/about is the About page", r.status == 200 and "<h1>About</h1>" in r.read().decode())
for asset, ctype in (("monitor.js", "text/javascript"), ("web.css", "text/css"), ("tokens.css", "text/css"),
                     ("favicon.svg", "image/svg+xml"), ("fonts/outfit-latin-wght.woff2", "font/woff2")):
    with urllib.request.urlopen(f"http://127.0.0.1:{port3}/web/{asset}", timeout=5) as r:
        ck(f"/web/{asset}", r.status == 200 and r.headers["Content-Type"].startswith(ctype.split(";")[0]),
           r.headers["Content-Type"])
try:
    urllib.request.urlopen(f"http://127.0.0.1:{port3}/web/%2e%2e/webui.py", timeout=5)
    ck("no traversal above serve/web", False)
except urllib.error.HTTPError as e:
    ck("no traversal above serve/web", e.code == 404, str(e.code))
for a, b in (("/monitor", "Monitor"), ("/about", "About")):
    with urllib.request.urlopen(f"http://127.0.0.1:{port3}{a}", timeout=5) as r:
        t = r.read().decode()
    ck(f"{a} links tokens.css and web.css", "/web/tokens.css" in t and "/web/web.css" in t)
    ck(f"{a} sets a theme", 'data-theme="dark"' in t)

# ------------------------------------------------------------------ (5) the pages themselves -----
# A duplicated `const` in a page script kills every card on the page, silently, in the browser only: no
# server error, no failed request, just a Monitor that stops updating. There is no bundler here, so when
# node is on the box, use it as a parser. This caught a real duplicate on 2026-11-02 -- the rates row
# introduced a second `const inflight` in the same function.
import subprocess                                        # noqa: E402
_node = shutil.which("node")
for _js in sorted(p.name for p in (Path(ROOT) / "serve" / "web").glob("*.js")):
    if _node:
        _r = subprocess.run([_node, "--check", str((Path(ROOT) / "serve" / "web" / _js))],
                            capture_output=True, text=True)
        ck(f"{_js} parses as JavaScript", _r.returncode == 0, (_r.stderr or "").strip()[:200])
    else:
        ck(f"{_js} parses as JavaScript (no node on this box - skipped)", True)
# Static pages rot quietly: a text("kv_pct") that no longer matches an id in the HTML renders a blank
# card forever with no error anywhere. These three checks are the cheapest way to keep that from being
# discovered by eye.
WEB = Path(WEB_DIR := os.path.join(ROOT, "serve", "web"))
for js, html in (("monitor.js", "monitor.html"), ("about.js", "about.html")):
    src, page = (WEB / js).read_text(), (WEB / html).read_text()
    used = set(re.findall(r'\$\("([a-zA-Z0-9_-]+)"\)', src)) | set(re.findall(r'\btext\("([a-zA-Z0-9_-]+)"', src))
    have = set(re.findall(r'\bid="([a-zA-Z0-9_-]+)"', page))
    ck(f"{js} only touches ids {html} has", not (used - have), f"missing: {sorted(used - have)}")
    ck(f"{html} links assets that exist", all((WEB / a).is_file() for a in
       re.findall(r'(?:href|src)="/web/([^"]+)"', page)), str(re.findall(r'(?:href|src)="/web/([^"]+)"', page)))
tokens = (WEB / "tokens.css").read_text() + "".join(p.read_text() for p in WEB.glob("*.css") if p.name != "tokens.css")
used = set()
for p in list(WEB.glob("*.css")) + [WEB / "monitor.html", WEB / "about.html", WEB / "monitor.js", WEB / "about.js"]:
    used |= set(re.findall(r"var\((--[a-z0-9-]+)", p.read_text()))
defined = set(re.findall(r"(--[a-z0-9-]+)\s*:", tokens))
ck("no CSS variable is used without being defined", not (used - defined), f"{sorted(used - defined)}")
ck("every page carries a theme switch and an API key control",
   all('data-theme-toggle' in (WEB / h).read_text() and 'id="key-btn"' in (WEB / h).read_text()
       for h in ("monitor.html", "about.html")))
monjs = (WEB / "monitor.js").read_text()
# A request row is scanned, not read, so its order is the design: the request's own prefill and decode speeds
# first and largest, the supporting detail after them, and the things that never carried information gone.
_i_pre, _i_dec = monjs.find('rqSpeed("prefill"'), monjs.find('rqSpeed("decode"')
_i_supp = monjs.find("first token ${seconds(")
ck("a request row leads with its own prefill and decode speeds and puts the supporting detail after them",
   0 < _i_pre < _i_dec < _i_supp and "b.append(row, sp, size, meta)" in monjs
   and ".rq-speed b{" in (WEB / "monitor.css").read_text(), f"{_i_pre} {_i_dec} {_i_supp}")
ck("both speeds are labelled as this request's own, with the basis in the tooltip",
   "this request's own prompt tokens over its own prefill span" in monjs
   and "queueing and prefill excluded" in monjs)
# Absence claims are checked against the CODE, not the prose: the comment above the row renderer names the
# strings it dropped, which is exactly what a naive substring check would then find and call a bug.
moncode = "\n".join(l for l in monjs.split("\n") if not l.strip().startswith("//"))
ck("the clutter is gone: no endpoint apology, no note about how the monitor obtained the row",
   "no endpoint in its stats" not in moncode and '"engine record"' not in moncode
   and moncode.count("tag tag--${") == 1, f"{moncode.count('tag tag--${')} state tags")
ck("and an ordinary row carries no state tag -- 1,429 of 1,429 engine rows on this box read completed",
   'if (r.state && r.state !== "completed")' in monjs
   and "hit max_tokens" in monjs and "preempted ${t.preemptions}" in monjs)
# The concurrency a decode rate has to be read against: derived, one decimal, and titled with the measurement
# that makes it meaningful.
ck("a row shows how many requests shared its decode window, and the tooltip says what the number is for",
   "shared ×${fmt(r.shared_decode, 1)}" in monjs and "158 tok/s alone" in monjs
   and "isnum(r.shared_decode)" in monjs)
# The clock basis, spelled out on the page instead of assumed: an engine row's wallclock excludes the queue, so
# a queued request used to print a first token LATER than its whole wall-clock, and the note claimed the opposite
# of what the number held.
ck("idle link states are named instead of alarmed: parked (D3hot, Gen1) and sagging (D0, nothing served)",
   "parked, max \u00d7" in monjs and "links sag while nothing runs: that is power management" in monjs
   and "a link is below its maximum while work is running" in monjs
   and "2.5 GT/s x16" in monjs and "16 GT/s on the c6 chain at 3% busy" in monjs)
ck("wall-clock is summed per path, and the note says which sum it is",
   "of it queue" in monjs and "all in ${seconds(total)}" in monjs
   and "plus the queue wait before them" in monjs
   and "From the engine: queue, prefill and decode together" not in monjs
   and "Wall-clock (queue included)" in (WEB / "monitor.html").read_text())
# The fixture tool renders the page over real journal rows -- including while the box is down, which is when
# those rows matter most. It observes: journal file plus a GET.
_fx = (Path(__file__).resolve().parent.parent / "tools" / "monitor-fixture.py").read_text()
compile(_fx, "tools/monitor-fixture.py", "exec")
ck("tools/monitor-fixture.py reads the journal and a GET, sends nothing, and survives a dead serve",
   "urlopen(" in _fx and "data=" not in _fx and "POST" not in _fx
   and "no live serve" in _fx and "shared_decode" in _fx)
# The queue line says WHY requests wait when the engine says why, and says nothing invented when it does not.
ck("the queue line splits waiting into scheduling capacity and deferred when the engine publishes it",
   "for scheduling capacity" in monjs and "deferred" in monjs
   and 'live.waiting_capacity' in monjs and 'live.waiting_deferred' in monjs
   and 'live.preemptions_total' in monjs and "preempted in the window" in monjs
   and 'title="The engine' in (WEB / "monitor.html").read_text())
ck("preemptions since launch are shown next to the window count, because a preempt costs a whole re-prefill",
   "live.preemptions_total" in monjs and "re-prefills" in monjs)
# vllm:iteration_tokens_total calls itself "tokens per engine step" and is not one on 0.32.0: a whole chunked
# prefill arrives in a single record (measured: a 29,969-token prompt produced records 29,969 / 4 / 1). The page
# has to name the counter's unit, not a chunk size it cannot verify -- and the measurement goes in the tooltip.
ck("the iteration line names the counter's unit instead of claiming a step size",
   "iteration record" in monjs and "prompt whole, decode per step" in monjs
   and "tokens per engine step" not in monjs and "steps per request" not in monjs)
ck("and the tooltip carries the measurement that made the wording change",
   "29,969" in (WEB / "monitor.html").read_text()
   and "max-num-batched-tokens" in (WEB / "monitor.html").read_text())
ck("a cumulative mean is labelled as one on the page",
   monjs.count("(since launch)") >= 2 and "const pick = (k) =>" in monjs,
   f"{monjs.count('(since launch)')} labels")
# the pane used to open empty on a serve doing 100 tok/s: an empty panel next to a busy engine is the one
# UI lie nobody notices. It follows the newest visible row -- active first, since that is the one moving --
# until you click a row, and the button that gets it back is in the bar, not hidden in a keyboard shortcut.
ck("the request list follows the newest row until you click one",
   'follow = true' in monjs and 'id="follow"' in (WEB / "monitor.html").read_text()
   and "rows.find(active) || rows[0]" in monjs and 'aria-pressed="true"' in (WEB / "monitor.css").read_text(),
   monjs[monjs.find("follow = true") - 40:monjs.find("follow = true") + 20])
# Two ways the per-request numbers misled on the live merged page. (a) The "Last decode speed ... One stream"
# card substituted the engine-wide 1/step when the record's own speed was missing: 27.4 tok/s next to a
# request that decoded at 87.8. (b) A request's prefill was not shown per request at all, so a 250k-token
# prompt looked indistinguishable from a 22-token one, and the only prefill on the page was a 2 s mean.
ck("the Last decode speed card shows that request's own speed, or says it has none",
   "lt.predicted_per_second" in monjs and "1e3 / rate.itl_ms : null" not in monjs
   and "no speed for that record" in monjs, monjs[monjs.find('text("speed-note"') - 200:monjs.find('text("speed-note"')])
ck("a row names its token counts and its own prefill, in seconds and in tok/s",
   "prefill ${seconds(t.prefill_s)}" in monjs and "fmt(u.prompt_tokens ?? 0, 0)} → " in monjs
   and 'rqSpeed("prefill", t.prefill_tok_s' in monjs)
ck("the wall-clock card splits that one request into queue, prefill and decode",
   'id="wall-split"' in (WEB / "monitor.html").read_text() and "queue ${seconds(lt.queue_s)}" in monjs)
# vLLM credits prompt_tokens_total at request END: a 180,066-token prefill read 0 tok/s for its whole duration
# and then 89,580 in one 2 s poll. The line has to say that rather than look live.
ck("the prompt tok/s line says it is credited at request end, not measured live",
   "(credited at request end)" in monjs)
# The rates people ask the page for -- single-stream and aggregate decode, prefill -- lead it, at display
# size, and each one says what it is computed from. Row 0 comes before the per-request cards.
monhtml = (WEB / "monitor.html").read_text()
ck("the four rates lead the Monitor, before the per-request cards",
   monhtml.index("grid--rates") < monhtml.index('id="speed"'), "rates row is not first")
ck("they are rendered at display size",
   'class="value value--xl"' in monhtml and ".value--xl{font-size:var(--st-fs-display)}" in (WEB / "web.css").read_text())
ck("all four rate slots exist and are written",
   all(f'id="{i}"' in monhtml and f'("{i}"' in monjs
       for i in ("rate-stream", "rate-gen", "rate-prefill", "rate-step")))
ck("the headline decode rate is the live aggregate-over-running one",
   "rate.stream_tok_s_live" in monjs and 'id="rate-stream"' in monhtml)
ck("the headline line reconciles the step rate with the token rate",
   "tokens a step" in monjs and "rate.tokens_per_step" in monjs)
ck("the prefill rate names which of its two numbers is showing",
   "estimated from KV growth" in monjs and "exact, from prompt_tokens_total" in monjs)
ck("the page says how old its numbers are", 'id="fresh"' in monhtml and "snapshot " in monjs)
ck("the page refetches on the engine's beat", "setTimeout(poll, 1000)" in monjs)
# The waveforms belong in one row, not sprinkled through the value cards: four series over the same span,
# each with last/peak/avg attached. (The aggregate series used to be drawn twice -- once big, once small.)
_trend = monhtml.split('grid--trends')[1].split("</section>")[0]
_rates = monhtml.split('grid--rates')[1].split("</section>")[0]
ck("every activity waveform is in the one trends row",
   all(f'id="{i}"' in _trend for i in ("stream-spark", "gen-spark", "prefill-spark", "running-spark")))
ck("no waveform is left behind in the value cards",
   'class="sparkline' not in _rates
   and monhtml.split('class="grid grid--4 engine"')[1].split("</section>")[0]
       .count('id="gen-spark"') == 0)
ck("the KV pool draws a waveform on a 0-100 axis, and the progress bar is gone",
   'id="kv-spark"' in monhtml and 'id="kv-bar"' not in monhtml
   and "{ baseLo: 0, baseHi: 100 }" in monjs
   and ".sparkline[data-tone=warn] .spark{color:var(--st-warn)}" in (WEB / "web.css").read_text())
ck("sparklines take an axis from the caller instead of always autoscaling",
   "baseLo = null" in (WEB / "common.js").read_text()
   and 'lo: 0, hi: "peak"' in monjs)
# An anchored series with no samples drew nothing at all: an empty box reads as a broken page, when the truth is
# "nothing is running". The axis's zero now gets drawn (faint, dashed, under the data) so a series at zero reads
# as a line lying on the axis and a series with no samples still shows where nothing-happening is. For the
# autoscaled series the line goes where zero is, because crossing it (the pool draining) is the event there.
ck("the axis zero is drawn, under the data, and styled faint",
   'class: "spark-base"' in (WEB / "common.js").read_text()
   and ".spark .spark-base{stroke:var(--st-ink-muted);opacity:.5}" in (WEB / "web.css").read_text())
ck("a series with no samples keeps its axis line instead of an empty box",
   "if (xs.length < 2) { el.replaceChildren(svg); return; }"
   in (WEB / "common.js").read_text())
# "no series yet" made the reader guess between nothing-running, counter-dead and page-broken. Each waveform now
# says which, in its own words, and names the axis the flat line sits on.
ck("an empty waveform says why in the series' own terms, not 'no series yet'",
   '"no series yet"' not in monjs
   and "has no denominator" in monjs and "0 streams running" in monjs and "did not grow" in monjs
   and "` · axis ${fmt(lo, 0)}..${fmt(top, 0)} ${unit}`" in monjs)
ck("a peak axis is never computed as Math.max() of nothing (which is -Infinity)",
   'Math.max(...xs, 0)' in monjs and "isFinite(lo) || !isFinite(hi)" in (WEB / "common.js").read_text())
# The per-card cards: busy and VRAM as waveforms on honest axes (0..100, and 0..the card's own VRAM), and
# aligned by bus id / card name so a card vanishing from sysfs cannot plot its neighbour's numbers.
_ck2 = (WEB / "common.js").read_text()
ck("a caveat on a headline number is note-sized, not headline-sized",
   'function qual(t) {' in _ck2 and '.qual{font-size:var(--st-fs-xs)' in (WEB / "web.css").read_text())
# "(since launch)" sat inside value--sm elements, set in the headline's own size -- three words louder than
# the measurement they qualify, which is backwards. The qualifier helper carries them instead.
ck("the tokens-per-step clause falls back to the since-launch ratio, labelled, when the window has none",
   "tokens a step (since launch)" in monjs and "sn.itl_ms / sn.tpot_ms" in monjs)
ck("the since-launch caveats on the headline stats ride the qualifier",
   monjs.count("qualText(") >= 2 and '"since launch"' in monjs
   and 'text(id, ms(pick(k)) + (isnum(rate[k])' not in monjs)

ck("a GPU card draws one labelled wave per sensor, and no busy bar",
   _ck2.count('spark--gpu') >= 1 and 'class="bar"' not in _ck2
   and all(f'["{lab}"' in _ck2 for lab in ("busy", "VRAM", "sclk", "W", "mem"))
   and "rows.push([tlab," in _ck2 and "{ baseLo: 0, baseHi: ceils[j] }" in _ck2 and "/ 2 ** 30" in _ck2)
ck("edge is text only, and each sensor row says what file it comes from",
   "const rows = [[\"busy\", \"gpu_busy_percent\"]" in _ck2 and "GDDR7" in _ck2
   and "temp*_crit" in _ck2)
ck("each wave gets its own ceiling, not an autoscaled one, and the temperature axis comes from the card's own "
   "published limit",
   "ceils = [100, vramCeil, sclkPeak, powerCeil]" in _ck2 and "ceils.push(tempCeil)" in _ck2
   and "const tempCeil = g.temp_crit_c || 110;" in _ck2
   and "Math.max(g.power_cap_w || 0, powerPeak)" in _ck2)
ck("each temperature wave is graded against its own sensor's published crit, and power never gets coloured",
   "g[kt] > 0.98 * kc" in _ck2 and "g[kt] > 0.95 * kc" in _ck2
   and "j === memIdx ? \"temp_mem_c\" : \"temp_c\"" in _ck2
   and "memCeil = g.temp_mem_crit_c || tempCeil" in _ck2)
# A page newer than the serve it reads must not name a sensor that build never took: an in-serve Monitor started
# before the three-sensor read reports temp1 only, so the headline row is labelled from the payload and the mem
# row is drawn only when there is a mem number behind it.
ck("the headline temperature row is labelled by what the payload carries, and the mem row appears only when the "
   "build reports mem",
   'isnum(g.temp_junction_c) ? "junction"' in _ck2 and 'isnum(g.temp_edge_c) ? "edge"' in _ck2
   and "const hasMem = isnum(g.temp_mem_c) || ghist.some" in _ck2 and "if (hasMem)" in _ck2)
ck("each card's waveform is matched by bus/card id, not by position",
   "c.bus === g.bus" in _ck2 and "c.card === g.card" in _ck2)
ck("both pages feed the card history to it",
   "gpuCard(g, s.gpus_history || [])" in monjs and "gpuCard(g, s.gpus_history || [])" in (WEB / "about.js").read_text())
ck("a card's caption carries the thermal trio and busy, each as peak and average, with the span named",
   all(x in _ck2 for x in ('"temp_c"', '"temp_mem_c"', '"power_w"', '"sclk_mhz"', "busy avg", "pts over",
                           "crit")))
# The PCIe panel. Two things it can report, and neither is a curve: link state only moves when a link
# renegotiates, and the AER counters are eventual (measured on the live box: 0 events in 90 idle samples, 0
# through an eight-stream burst, ~12 clustered around two serve restarts) -- so four flat waveforms per chain
# took a third of the page to say nothing. Rows state the numbers; the event, with its time, is what gets listed.
ck("the PCIe panel exists and says what it is not (no byte counters on these cards)",
   'id="pcie"' in monhtml and 'id="pcie-note"' in monhtml and "pcieRow(" in monjs
   and "N/A" in monjs and "no root and no /dev/kfd" in monjs)
ck("a chain is one row: card, device chain, negotiated link, totals since boot, which counters, this window",
   'class="pcie-rows"' in monhtml and all(x in monjs for x in
                                          ('"pcie-row"', 'class="chain"', 'class="link"',
                                           'class="cnt"', 'class="who"', 'class="win"')))
ck("no per-chain waveforms are drawn any more -- a flat line for a static or eventual number reads as a fault",
   "spark" not in monjs.split("function pcieRow")[1].split("\nfunction")[0],
   str([l.strip() for l in monjs.split("function pcieRow")[1].split("\nfunction")[0].splitlines()
        if "spark" in l][:3]))
ck("one box-wide spark of AER events per second stays, on a zero-anchored axis, and it is warned when it moves",
   'id="pcie-spark"' in monhtml and "baseLo: 0, baseHi: Math.max(...perPoll, 1)" in monjs
   and 'perPoll.some((v) => v > 0) ? "warn"' in monjs)
ck("an event is listed with its wall-clock time, device and counter, and only events we witnessed",
   "s.pcie_events" in monjs and "${when(e.at)} ${shortb(e.bdf)} ${name}" in monjs
   and "nothing moved while this page has been watching" in monjs)
ck("a page talking to a serve build too old to report events says so, instead of claiming nothing happened",
   "s.pcie_events === undefined" in monjs and "does not report events" in monjs)
# The kernel names a CORRECTABLE counter "NonFatalErr" (Advisory Non-Fatal Error), and the webui tags the real
# uncorrectable ones #nf/#ft. Printed bare, "NonFatalErr 3" next to "0 uncorr" looks like the panel disagrees
# with itself, so the counter's kind is stated wherever a name is.
ck("an AER counter name is shown with its kind, so a correctable NonFatalErr is not read as uncorrectable",
   "function aerKind(" in monjs and '/#(nf|ft)$/.test(k)' in monjs and '<span class=\"q\">corr</span>' in monjs
   and 'class="unc"' in monjs)
ck("a link below its maximum names what it fell short of, and only work-in-progress turns it to danger",
   'c.parked ? "parked, max \u00d7"' in monjs and 'sag ? "idle, max \u00d7" : "max \u00d7"' in monjs
   and '(c.degraded && !c.parked && busyNow) || unc ? "danger"' in monjs
   and "g.busy_pct >= 25" in monjs)
# tools/watch-live.py is the page's questions asked from a terminal. Its one hard property is that it observes:
# measuring someone else's concurrency by adding your own requests would contaminate exactly what is being read.
_wl = open(os.path.join(ROOT, "tools", "watch-live.py")).read()
compile(_wl, "watch-live.py", "exec")            # syntax only -- running it would sample and print
ck("the live watcher only reads: it never builds a request or sends a body to the serve it is watching",
   "urlopen(" in _wl and "Request(" not in _wl and "data=" not in _wl and "POST" not in _wl)
# gpu_busy_percent on this box is coarse, not finely readable: refused with nothing running, 0 stopped, 24-29
# idle, 46-100 under load, 95-99 through long decode. The page says so, because otherwise a flat 99 reads as
# "fully loaded" and a 0 as "healthy idle" when either could also mean "the launch let go".
ck("the cards line says busy % is coarse here, with the observed spread",
   "busy % is coarse here" in monjs and "amdgpu sysfs" in monjs)
ck("each waveform carries last, peak and average",
   'avg ${fmt' in monjs and 'peak ${fmt' in monjs
   and all(f'id="{i}"' in monhtml for i in ("stream-spark-note", "gen-spark-note",
                                            "prefill-spark-note", "running-spark-note", "kv-spark-note")))
# A null is "no sample in this window", not "the series was at zero": the line has to break, and the filled
# area under it has to stop, or the gap reads as a measurement at the baseline.
_ck = (WEB / "common.js").read_text()
ck("a gap in a series is drawn as a gap, and its fill is dropped",
   "if (!isnum(v)) { if (pen) gaps++; pen = false; return; }" in _ck and "if (fill && !gaps)" in _ck)
ck("the span is named in points and seconds, since the ring counts points",
   "pts over" in monjs and "s.history" in monjs)
ck("a rate waveform averages only the windows that carried a rate, and the estimate only the growing ones",
   'keep = (v) => v !== 0' in monjs and 'keep: (v) => v > 0' in monjs
   and 'label: " while the pool grew"' in monjs)
ck("the pool's own waveform is plotted straight, with no sample filtering",
   'spark($("kv-spark"), hist.map((h) => h.kv_used_pct)' in monjs)
ck("the sidecar polls the engine at 1 s by default, both modes",
   'os.environ.get("R9K_WEB_POLL", 1.0)' in (Path(ROOT) / "serve" / "webui.py").read_text()
   and 'os.environ.get("R9700_WEBUI_POLL", "1")'
       in (Path(ROOT) / "serve" / "plugin" / "r9700_webui" / "__init__.py").read_text())
ck("a record shows its own prefill speed with the basis it was computed from, and the row leads with it",
   'id="detail-prefill-speed"' in (WEB / "monitor.html").read_text() and "t.prefill_tok_s" in monjs
   and "t.prefill_basis" in monjs and "${fmt(t.prefill_tok_s, 0)} tok/s" in monjs
   and 'rqSpeed("prefill", t.prefill_tok_s' in monjs)
# About exists so a client can be configured off it. Two things made that fail before: the base URL printed was
# the sidecar's own upstream (127.0.0.1 on the serve's box -- dead wrong pasted anywhere else, and this box is
# reached over the network), and it had no /v1, which is what LiteLLM and the OpenAI SDK need. Both blocks are
# generated from location.origin, and the key is named, never inlined.
abjs, abhtml = (WEB / "about.js").read_text(), (WEB / "about.html").read_text()
ck("About carries a pasteable LiteLLM proxy config and a direct call",
   'id="litellm-yaml"' in abhtml and 'id="litellm-py"' in abhtml
   and "model: openai/" in abjs and 'api_base: ${apiBase}' in abjs and 'api_base="${apiBase}"' in abjs)
ck("the base a client gets is the address that answered this page, with /v1; the internal one is labelled",
   "const origin = (location.origin" in abjs and 'const apiBase = `${origin}/v1`' in abjs
   and '["OpenAI / LiteLLM base_url", apiBase, true]' in abjs
   and '["Upstream this page dials", s.upstream' in abjs)
ck("no API key is ever written into a pasteable block",
   all(k in abjs for k in ('api_key: ${s.key_needed ? "os.environ/LITELLM_API_KEY" : "none"}',))
   and not re.search(r"api_key[=:]\s*\$?\{?\s*(s\.key\b|cfg\[.key.\])", abjs),
   [m.group(0) for m in re.finditer(r"api_key.{0,40}", abjs)][:2])
ck("the checkpoint path is on the page but says it is not a model name",
   'Weights live at' in abjs and "not a name a client can ask for" in abjs)
# Measured with litellm 1.106.6 against this serve: the pasted snippet as first written returned
# content=None, because every one of max_tokens went into reasoning. The snippet reads both fields, and the
# parser name only reaches the page through the recipe file -- so the ARGS line and the meta line have to
# name the same variable, or About describes a serve that is not running.
ck("the LiteLLM snippet reads reasoning where this serve puts it",
   "reasoning_content" in abjs and "max_tokens=128" in abjs and "reasoning" in abhtml)
for shl in ("serve/native-flashnext.sh", "serve/serve.sh"):
    src = open(ROOT + "/" + shl).read()
    ck(f"{shl} passes and reports the same reasoning parser",
       '--reasoning-parser "${REASONING:-qwen3}"' in src or "--reasoning-parser ${REASONING:-qwen3}" in src,
       [m.group(0) for m in re.finditer(r"reasoning-parser.{0,24}", src)][:1])
    ck(f"{shl} records that parser in the recipe file About reads",
       'reasoning="${REASONING:-qwen3}"' in src)

# VLLM_PLUGINS is a STRICT allowlist, not an "also load these" list. Writing VLLM_PLUGINS=r9700_webui by
# hand (the first version) turned "load everything discoverable" into "load exactly this", which silently
# disabled this repo's own r9700 plugin -- the compressed-tensors scheme for this checkpoint -- and the
# engine died four minutes into the load with NotImplementedError on ...in_proj_qkvz. So the value is
# built from discovery, and a test says the plugin that must survive does.
import importlib.util                                     # noqa: E402
import subprocess                                         # noqa: E402
spec = importlib.util.spec_from_file_location("webui_plugins", ROOT + "/serve/webui-plugins.py")
WP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(WP)
ck("an existing allowlist survives being added to",
   {"r9700", "r9700_webui"} <= set(WP.allowlist(["r9700"]).split(",")), WP.allowlist(["r9700"]))
ck("and the monitor's own name is always there", "r9700_webui" in WP.allowlist([]).split(","))
native = Path.home() / "venv-r9700" / "bin" / "python"
if native.exists():
    got = subprocess.run([str(native), ROOT + "/serve/webui-plugins.py", "--quiet"],
                         capture_output=True, text=True).stdout.strip()
    ck("discovered in the venv that runs the serve: this repo's r9700 plugin is NOT dropped",
       "r9700" in got.split(",") and "r9700_webui" in got.split(","), got)
else:
    ck("native venv absent here: discovery not checked", True)

# The recipe file is the About page's only source for "how this box was launched", and it is named after the
# port the PAGES are on. WEBUI=same has no second port; naming the file after PORT+1 anyway wrote it to
# -8081.json while the plugin read -8080.json, so the card vanished and nothing anywhere reported an error.
lf = (ROOT + "/serve/native-flashnext.sh")
nf = open(lf).read()
ck("the launcher names the recipe file in exactly one place, and merged mode names it after its own port",
   nf.count("r9700-web-meta-") == 1 and 'same ] && echo "${PORT' in nf,
   f"{nf.count('r9700-web-meta-')} default meta paths in {lf}")

for srvx in (srv, srv2, srv3):
    srvx.shutdown()
up.shutdown()

print("PASS" if fail == 0 else "FAIL", f"({fail} failing)")
sys.exit(1 if fail else 0)
