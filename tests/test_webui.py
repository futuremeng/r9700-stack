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
from collections import deque
import re
import shutil
import tempfile
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
# A request COUNT summed across series, a pool FULLNESS taken at its worst. Two engines running 1 and 3 requests
# is 4 requests on the box: max would report 3 and the page's "N running" would be a per-engine number wearing a
# box-level label. The percentage is the other way -- summing 0.40 and 0.72 into 112% is the classic way a gauge
# becomes impossible, so `kv_used_pct_mean` divides by the series count and the count itself travels so the page
# can say which it is looking at.
ck("running/waiting are summed across engines, not the max of them", st["live"]["running"] == 4.0
   and st["live"]["waiting"] == 2.0, str({k: st["live"][k] for k in ("running", "waiting")}))
ck("the pool mean is a MEAN over series (56%, not the 112% the sum gives), and the series count rides along",
   abs(st["live"]["kv_used_pct_mean"] - 56.0) < 1e-9 and st["live"]["kv_series"] == 2,
   str({k: st["live"][k] for k in ("kv_used_pct", "kv_used_pct_mean", "kv_series")}))
ck("prefix hit rate", abs(st["live"]["prefix_hit_pct"] - 100.0 / 3.0) < 1e-6)
ck("a counter that did not move reads None, not 0", W.derive(cur2, cur2, 10.0, cap2, info2, "observe", 0, [])
   ["rate"]["ttft_ms"] is None)

# The two rates the headline row is built from are the two that move WHILE a request runs. Measured on this
# box (0.32.0+rocm723, 2026-11-02): generation_tokens_total advances every engine step (one 900-token stream
# moved it ~120 tokens per 0.5 s sample) and so does the KV gauge (0.036 -> 0.1007 across a 51,859-token
# prefill), while prompt_tokens_total, iteration_tokens_total_* and every latency histogram are credited when
# the request finishes (that same prefill held the prompt counter flat for 10 s, then one jump 129 -> 51,988).
ck("single-stream rate = aggregate over running, live",
   abs(st["rate"]["stream_tok_s_live"] - 15.0) < 1e-9,
   f"{st['rate']['stream_tok_s_live']} for 60 tok/s across 4 running (1 + 3 across two engines)")
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

# The per-poll decode fields, on the SAME basis as the spans block: the denominator is the ITL family's sum, which
# is stream-seconds (one observation per request x step), so the rate it gives is a per-stream rate and the field
# is named for that. Here 600 tokens over 1.6 s of stream-decode-seconds is 375 tok/s a stream, and 1.6 s of
# stream-seconds inside a 10 s window says 0.16 streams were decoding -- 375 x 0.16 = 60, the summed figure. That
# multiplication is the card's whole basis, so it is pinned here and again over a span below.
_dv = W.derive(bcur, prev, 10.0, bcap, binfo, "observe", 0, [])["rate"]
ck("a poll's decode rate names its denominator: tokens over STREAM-seconds, plus the stream count that closes it",
   abs(_dv["decode_stream_tok_s"] - 600.0 / 1.6) < 1e-9 and abs(_dv["decode_time_s"] - 1.6) < 1e-9
   and abs(_dv["decode_streams"] - 0.16) < 1e-9
   and abs(_dv["decode_stream_tok_s"] * _dv["decode_streams"] - _dv["generation_tok_s"]) < 1e-9,
   str({k: _dv.get(k) for k in ("decode_stream_tok_s", "decode_time_s", "decode_streams", "generation_tok_s")}))
ck("the old field names are gone rather than kept as aliases, because the alias is what gets read as the truth",
   "decode_tok_s" not in open(os.path.join(ROOT, "serve", "webui.py")).read()
   and "decode_time_pct" not in open(os.path.join(ROOT, "serve", "webui.py")).read())
ck("per-poll streams per record come from the two counters that mean it (ITL observations over engine records)",
   abs(_dv["streams_per_record"] - 40.0 / 4.0) < 1e-9, str(_dv.get("streams_per_record")))

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
        # vLLM computes this itself (v1/metrics/stats.py:606-610): decode_time / (gen - 1), because decode_time
        # spans first token -> last token and the first token came out of the prefill step. The Monitor's row
        # rate is 1/this, so the stand-in has to carry it or the test is checking the fallback, not the engine's
        # own number.
        s.mean_time_per_output_token = s.decode_time / (s.num_generation_tokens - 1)
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
# The per-request decode rate, on the ENGINE's denominator. gen 8 over a 0.2 s first-to-last span: the engine's
# own mean is 0.2/7 = 28.6 ms a token, so the row must read 35 tok/s. Dividing gen by decode_time -- what this
# line used to do while calling its result "the engine's own mean" -- charges the prefill's first token to the
# decode span: 40 tok/s here, i.e. 14% too fast, and a 2-token request came out at double its real rate. Its own
# ring, no journal: the replay checks below count the rows that went to that file.
_er = W.EngineRecords(keep=3)
_own = _er.add(Fin(1))["timings"]
ck("a request row's decode rate is 1 / the engine's mean time per output token, not gen over the whole span",
   abs(_own["predicted_per_second"] - 35.0) < 1e-9 and abs(_own["tpot_ms"] - 1000.0 * 0.2 / 7) < 1e-6
   and "mean 28.6 ms" in _own["speed_basis"], str(_own)[:220])
_nomp = Fin(99)                                               # a build that does not hand the mean over
del _nomp.mean_time_per_output_token
_fallback = _er.add(_nomp)
ck("without the engine's field the fallback is its formula (gen-1 over the span), not gen over the span",
   abs(_fallback["timings"]["predicted_per_second"] - 35.0) < 1e-9, str(_fallback["timings"]["speed_basis"]))
_two = Fin(98, gen=2)
_two.mean_time_per_output_token = 0.1 / 1
_two.decode_time = 0.1
_short = _er.add(_two)
ck("a 2-token request reads 10 tok/s, not the 20 tok/s that counts its first token twice",
   abs(_short["timings"]["predicted_per_second"] - 10.0) < 1e-9, str(_short["timings"]))
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
# The row must arrive on disk COMPLETE. Journal.append serialises ROW_KEYS the moment it is called, so an append
# placed before wallclock_s/timings were computed wrote nulls for both: the live list looked right, and after a
# hard reset every restored relay row had no wall, no decode rate and no span for shared_decode to count. This is
# the check that the old ordering failed.
_line = json.loads(Path("/tmp/r9700-webui-journal-relay.jsonl").read_text().splitlines()[-1])
ck("a journalled relay row carries the timings and the wall it computed at finish, not nulls",
   _line["wallclock_s"] is not None and _line["timings"]
   and _line["timings"]["predicted_per_second"] is not None, str(_line)[:220])
ck("so a restored relay row can still be shared-counted (a null wall would drop it out of the decode spans)",
   r2.listing()[0]["shared_decode"] is not None, str(r2.listing()[0])[:160])
# When the client did not ask for usage, a streamed relay row's ONLY token count is the number of SSE deltas --
# and vLLM sends one delta per scheduled step, which with MTP carries 1 + accepted draft tokens (2.05 a stream-step
# live). So that count is a floor, and the row has to say so instead of putting a ~ on the number.
_rs = W.Records(keep=2)
_r = _rs.start("/v1/chat/completions", "m", "{}", True)
_r["first_token_s"], _r["tokens_est"], _r["stream_chars"] = 0.1, 2, 11     # what _sse accumulates per delta
_r["usage"] = {"prompt_tokens": 5, "completion_tokens": None}
_rs.finish(_r, "completed", 200)
ck("a usage-less stream says its token count is a floor on deltas, with the characters beside it",
   _r["timings"]["tokens_are_estimate"] and "floor" in (_r["timings"].get("gen_basis") or "")
   and "2 streamed deltas over 11 characters" in _r["timings"]["gen_basis"], str(_r["timings"]))
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
# The firmware's slot table, as this box lays it out: the slot is named at the card's SWITCH UPSTREAM port
# (slot 6 -> 0000:c4:00, slot 8 -> 0000:04:00), never at the GPU endpoint -- so "which slot is this card in" is
# a question about the chain, and answering it from the endpoint's own BDF would find nothing.
slotsroot = fakepci / "bus" / "pci" / "slots"
for slotname, addr in (("6", "0000:c4:00"), ("8", "0000:04:00")):
    sd = slotsroot / slotname
    sd.mkdir(parents=True)
    (sd / "address").write_text(addr + "\n")

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
ck("status carries cards, delivered in chassis order and numbered",
   len(s["gpus"]) == 3 and any(g["card"] == "card0" and g["busy_pct"] == 87 for g in s["gpus"])
   and [g["pos"] for g in s["gpus"]] == [1, 2, 3]          # 1-based, and the delivered order IS slot order
   and s["gpus"] == sorted(s["gpus"], key=lambda g: W.slot_key(g.get("slot"), g.get("bus"))),
   str([(g.get("pos"), g.get("slot"), g.get("card")) for g in s.get("gpus", [])])[:160])
# Was `s["gpus"][0]["busy_pct"] == 87`: position-dependent, and the order is no longer enumeration order.
# A card's identity is its bus/card name -- which is also how the history ring is aligned.
ck("the payload carries the chassis mapping the page prints (slot, driver name, GPU bus, chain)",
   [(m["pos"], m["slot"], m["bus"]) for m in s.get("slot_map", [])] == [(1, "6", "c6:00.0"), (2, "8", "06:00.0")]
   and all(m["chain"] for m in s["slot_map"])
   and s["slot_map"][0]["slot_at"] == "0000:c4:00.0"
   # `card` may be None when the DRM tree and the PCI tree do not overlap (as in these fixtures): the page
   # prints "unnamed" rather than inventing a driver name for a chain it cannot match.
   and all(m["card"] is None or isinstance(m["card"], str) for m in s["slot_map"]),
   str(s.get("slot_map"))[:200])
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
# The pages poll themselves, and how they poll is load-bearing: a fetch that never returns used to leave the
# tab showing its last render forever (the loop was `await refresh(); setTimeout(poll, 1000)`, so the tick that
# stalled was the last tick there ever was), and a background tab throttled to one tick a minute looked exactly
# like a quiet machine. All three JS files are syntax-checked when node is present, and the loop shape is
# asserted, because every one of those failures is silent on screen.
# A correctable count that stopped moving must stop being painted as if it were moving. The clock for that is
# aer_deltas' quiet_s: this box's c6 chain carried 42 correctables from a load window that ended at 09:00:03 and
# then sat at exactly 42 for 42 minutes at 32 GT/s. Pure function, so no server and no GPU needed.
def _ch(cor, notable):
    return {"bus": "c6:00.0", "card": "card2", "speed": 32.0, "width": 16, "degraded": False, "parked": False,
            "devices": [{"bdf": "0000:c4:00.0", "role": "upstream port", "cor": cor, "nonfatal": 0, "fatal": 0,
                         "notable": dict(notable)}]}
_l, _n, _e, _q = {}, {}, [], {}
_d, _l = W.aer_deltas([_ch(42, {"BadTLP": 27})], _l, _n, _e, 1000.0, _q)
ck("a chain found with counters already raised gets a watch clock starting now, and no invented events",
   _d[0]["quiet_s"] == 0.0 and _e == [], str(_e))
_d, _l = W.aer_deltas([_ch(43, {"BadTLP": 28})], _l, _n, _e, 1060.0, _q)
ck("when a counter moves the clock resets and the event is dated",
   _d[0]["quiet_s"] == 0.0 and _d[0]["cor_d"] == 1 and _e and _e[-1]["kind"] == "BadTLP"
   and _e[-1]["n"] == 1, str(_e[-1:]))
_d, _l = W.aer_deltas([_ch(43, {"BadTLP": 28})], _l, _n, _e, 3600.0, _q)
ck("nothing moving for 42 minutes reads as 42 minutes, not as calm-by-default",
   _d[0]["quiet_s"] == 2540.0 and _d[0]["cor_d"] == 0, str(_d[0]))
_d, _l = W.aer_deltas([_ch(1, {"BadTLP": 1})], _l, _n, _e, 3660.0, _q)
ck("the kernel zeroing a counter is an event, not a long quiet spell (the delta is unknowable)",
   _d[0]["reset"] and _d[0]["quiet_s"] == 0.0 and _d[0]["cor_d"] is None, str(_d[0]))
_rq, _rr, _rn = {}, {}, {}
_rl, _rw = {}, {}
_d1, _rl = W.aer_deltas([_ch(1, {"BadTLP": 1})], _rl, _rn, [], 1000.0, _rq, _rr, watch=_rw)   # baseline poll
_d2, _rl = W.aer_deltas([_ch(46, {"BadTLP": 46})], _rl, _rn, [], 1030.0, _rq, _rr, watch=_rw)  # +45 in 30 s
# The window is the 60 s floor, not the nominal 300 s: the watcher has only seen 30 s of this chain, and dividing
# 45 counters by five minutes reports 9/min for a link that is doing 90/min. The page keys its colouring off this
# number, so an understated rate dims a storm; it may dilute, it must never invent calm.
ck("45 counters inside the first 30 s of watching average 45/min over the 60 s floor, and the page is told the span",
   _d2[0]["rate_per_min"] == 45.0 and _d2[0]["window_s"] == 60.0 and _d2[0]["watched_s"] == 30.0,
   str(_d2[0]))
_wo2, _wq2, _wn2, _wr2, _wl2 = {}, {}, {}, {}, {}          # watch / quiet / named / recent / last
_o1, _wl2 = W.aer_deltas([_ch(2, {"BadTLP": 2})], _wl2, _wn2, [], 5000.0, _wq2, _wr2, watch=_wo2)
_o2, _wl2 = W.aer_deltas([_ch(6, {"BadTLP": 6})], _wl2, _wn2, [], 5020.0, _wq2, _wr2, watch=_wo2)
ck("a chain that raises 4 counters in its first 20 s reads 4.0/min, not the diluted 0.8/min a flat 300 s divisor "
   "gives -- 0.8 is what put the quiet glyph over a real storm",
   _o2[0]["rate_per_min"] == 4.0 and _o2[0]["window_s"] == 60.0 and _o2[0]["watched_s"] == 20.0, str(_o2[0]))
_d3, _rl = W.aer_deltas([_ch(46, {"BadTLP": 46})], _rl, _rn, [], 4700.0, _rq, _rr, watch=_rw)  # an hour later
ck("an hour later the burst has aged out and the rate is 0 -- a trailing rate relaxes, a lifetime average never does",
   _d3[0]["rate_per_min"] == 0.0 and _d3[0]["quiet_s"] == 3670.0 and _d3[0]["window_s"] == 300.0, str(_d3[0]))
ck("quiet_s is asked for by the hub, not just defined (the running serve must actually fill it in)",
   "self.aer_quiet" in open(os.path.join(ROOT, "serve", "webui.py")).read())
_js = {p.name: p.read_text() for p in WEB.glob("*.js")}
ck("every api() call carries a deadline (a stalled request must become one visible error and the next tick)",
   all(x in _js["common.js"] for x in ("AbortController", "signal: ctl.signal", "clearTimeout", "FETCH_MS",
                                       "did not answer within")),
   sorted(x for x in ("AbortController", "signal: ctl.signal", "clearTimeout", "FETCH_MS",
                      "did not answer within") if x not in _js["common.js"]))
_m = _js["monitor.js"]
ck("the poll loop reschedules in a finally, so no tick can be the last one",
   bool(re.search(r"finally\s*\{.*setTimeout\(poll", _m, re.S)) and "await refresh(); setTimeout(poll" not in _m)
ck("a page with no answer for 5 s says how old its numbers are (short in the slot, the sentence in the title), "
   "and stays quiet while deliberately paused",
   bool(re.search(r"!paused.*s stale", _m, re.S)) and "has not had an answer for" in _m)
ck("returning to a throttled background tab refetches at once instead of waiting for the timer",
   all("visibilitychange" in _js[f] for f in ("monitor.js", "about.js")))
# the de-emphasis itself: colour follows activity, not a counter that has stopped moving
_m2 = _js["monitor.js"]
ck("a static correctable count no longer holds the warn tone (the tone keys off liveErrs now)",
   ': cor ? "warn"' not in _m2 and "(cor && liveErrs) ? \"warn\"" in _m2)
ck("the sidecar clocks are read off the history entry, and an unknown clock counts as LOUD (an older sidecar "
   "cannot fake calm)",
   "last?.quiet_s" in _m2 and "last?.rate_per_min" in _m2 and "!last ||" in _m2 and "rate === null" in _m2
   and "c.quiet_s" not in _m2 and "c.rate_per_min" not in _m2,
   "payload['pcie'] is read_pcie chains, which carry no deltas, no quiet_s and no rate_per_min")
# third level: at most one counter a minute over the trailing hour collapses to an info glyph
_sr = open(os.path.join(ROOT, "serve", "webui.py")).read()
ck("the pump counts characters as well as deltas, so a usage-less stream can call its token count a floor",
   'rec["stream_chars"] += len(txt)' in _sr and '"stream_chars": 0' in _sr)
ck("the sidecar publishes a trailing-window per-minute rate, pruned as events age out, over the span it watched",
   "self.aer_recent" in _sr and 'd["rate_per_min"]' in _sr and "window_s=300.0" in _sr
   and 'd["window_s"] = round(span, 1)' in _sr and "self.aer_watch" in _sr)
ck("the row collapses to an info glyph only below ~1 counter/minute, and never while uncorrectable is nonzero",
   "rate <= 1" in _m2 and "!unc" in _m2.split("const sparse")[1][:40] and 'class="p-info' in _m2)
ck("the rate, not the time since the last event, decides: a one-per-few-minutes trickle gets the glyph, and a "
   "single fresh event still flashes to digits",
   "justNow || !last || rate === null || rate > 1" in _m2 and "!justNow && rate !== null && rate <= 1" in _m2
   and "quiet < 300" not in _m2, "quiet_s must not gate the colour: measured 1 BadTLP per ~3 min under ~100k-token "
   "traffic, a 300 s gate would leave the row coloured forever")
ck("the window cell carries the rate and the calm span once it is sparse",
   '/min${quiet !== null ? ` \u00b7 quiet' in _m2 or "/min${quiet !== null ? ` · quiet" in _m2)
ck("the demoted numbers survive in pinned panel text, not in a tooltip that cannot be selected or copied",
   'put("since boot", `${cor} correctable' in _m2 and 'put("rate",' in _m2
   and 'put("correctable", cn.join' in _m2 and "function pcieDetailLines" in _m2)
ck("a chain with nothing raised is a dash, not the sentence 'no errors since boot'",
   "no errors" not in _js["monitor.js"]
   and 'title="no AER counter on this chain has moved since boot">' in _js["monitor.js"]
   and r">\u2014</span>" in _js["monitor.js"], "the dash must be there as a dash, with the meaning in the title")
_mc = open(os.path.join(ROOT, "serve", "web", "monitor.css")).read()
ck("the sparse icon and the panel have styles, built from tokens this page defines",
   ".p-glyph{" in _mc and "var(--st-line)" in _mc and "var(--st-ink-muted)" in _mc
   and ".pcie-detail{" in _mc and "var(--st-r-sm)" in _mc)
ck("the quiet row says how long it has been quiet instead of pretending there is nothing there",
   re.search(r"\+0 · \$\{rate[^}]*\}/min", _m2) and re.search(r"quiet \$\{ago\(quiet\)\}", _m2)
   and "const ago = (s2) =>" in _m2)
ck("the About page loop cannot die on one bad tick either",
   bool(re.search(r"finally\s*\{.*setTimeout\(loop", _js["about.js"], re.S)))
tokens = (WEB / "tokens.css").read_text() + "".join(p.read_text() for p in WEB.glob("*.css") if p.name != "tokens.css")
used = set()
for p in list(WEB.glob("*.css")) + [WEB / "monitor.html", WEB / "about.html", WEB / "monitor.js", WEB / "about.js"]:
    used |= set(re.findall(r"var\((--[a-z0-9-]+)", p.read_text()))
defined = set(re.findall(r"(--[a-z0-9-]+)\s*:", tokens))
ck("no CSS variable is used without being defined", not (used - defined), f"{sorted(used - defined)}")
ck("every page carries a theme switch and an API key control",
   all('data-theme-toggle' in (WEB / h).read_text() and 'id="key-btn"' in (WEB / h).read_text()
       for h in ("monitor.html", "about.html")))
# ------------------------------------------------- (3c) what ended the PREVIOUS boot (ACPI BERT / MCE)
# A hard reset on this box leaves almost nothing, but not nothing: the firmware keeps one record in ACPI BERT
# and the kernel replays it at the start of the next boot. These lines are verbatim from this machine's own
# journal at 2026-11-03 02:50:38 -- the death that ended the 15.7 h launch. All 18 deaths the journal holds
# have this shape, and every one names the same bank and the same physical core.
DEATH_LOG = [
    "2026-11-03T02:50:38+00:00 jiulu-9700 kernel: BERT: Error records from previous boot:",
    "2026-11-03T02:50:38+00:00 jiulu-9700 kernel: [Hardware Error]: event severity: fatal",
    "2026-11-03T02:50:38+00:00 jiulu-9700 kernel: [Hardware Error]:  Error 0, type: fatal",
    "2026-11-03T02:50:38+00:00 jiulu-9700 kernel: [Hardware Error]:  fru_text: ProcessorError",
    "2026-11-03T02:50:38+00:00 jiulu-9700 kernel: mce: [Hardware Error]: Machine check events logged",
    "2026-11-03T02:50:38+00:00 jiulu-9700 kernel: mce: [Hardware Error]: CPU 17: Machine Check: 0 Bank 5:"
    " aea0000001000108",
    "2026-11-03T02:50:38+00:00 jiulu-9700 kernel: TSC 0 ",
    "2026-11-03T02:50:38+00:00 jiulu-9700 kernel: ADDR 142d6f3 ",
    "2026-11-03T02:50:38+00:00 jiulu-9700 kernel: MISC d0140ff600000000 ",
    "2026-11-03T02:50:38+00:00 jiulu-9700 kernel: PPIN 2b61290dcb2002b SYND 4d000000 IPID 500b024029a00 ",
    "2026-11-03T02:50:38+00:00 jiulu-9700 kernel: mce: [Hardware Error]: PROCESSOR 2:a10f11 TIME 1793674220"
    " SOCKET 0 APIC 82 microcode a101158",
    "2026-11-03T02:50:38+00:00 jiulu-9700 kernel: BERT: Total records found: 1",
]
_death = W.parse_death(DEATH_LOG)
ck("the record the firmware kept is parsed: severity, uncorrectable, bank, CPU, time, fru_text",
   _death and _death["severity"] == "fatal" and _death["uncorrectable"] and _death["bank"] == 5
   and _death["cpu"] == 17 and _death["at"] == 1793674220 and _death["fru"] == "ProcessorError"
   and _death["bert_records"] == 1, str(_death))
ck("the IPID is resolved to the block it names, the way the kernel's own smca table does",
   _death["bank_name"] == "execution unit", str(_death))
ck("a memory-controller IPID is named as one, so the page cannot call a UMC fault a core fault",
   W.smca_bank(0x00000096 << 32) == "memory controller" and W.smca_bank(0x00000046 << 32) == "PCIe")
ck("an unknown IPID says so with its numbers instead of guessing",
   W.smca_bank(0x00001234 << 32).startswith("bank hwid"), W.smca_bank(0x00001234 << 32))
ck("MCi_STATUS bit 61 is what makes a record uncorrectable, and a record without it is not",
   W.parse_death([l.replace("aea0000001000108", "8ea0000001000108") for l in DEATH_LOG])["uncorrectable"]
   is False)
ck("a torn record is not invented into a death",
   W.parse_death(["kernel: mce: [Hardware Error]: TSC 0 ", "kernel: PPIN x SYND y IPID 500b024029a00"]) is None)
# The CPU number is only half the story: 17 and 113 are the two SMT threads of ONE physical core, and that is
# what turns "two CPUs failed" into "one core failed 18 times". Read from sysfs, so fixture it.
cpu = Path("/tmp/r9700-webui-cpu")
shutil.rmtree(cpu, ignore_errors=True)
(cpu / "cpu17" / "topology").mkdir(parents=True)
(cpu / "cpu17" / "topology" / "thread_siblings_list").write_text("17,113\n")
(cpu / "cpu17" / "topology" / "core_id").write_text("65\n")
ck("a CPU number resolves to its physical core and SMT siblings",
   W.core_of(17, root=str(cpu)) == {"cpu": 17, "siblings": "17,113", "core_id": "65"},
   str(W.core_of(17, root=str(cpu))))
ck("a CPU with no topology node still returns, with nothing claimed",
   W.core_of(999, root=str(cpu)) == {"cpu": 999, "siblings": None, "core_id": None})
_hist = W.death_history(lines=DEATH_LOG)
ck("the count, the span and the block every record blames are derived for the page",
   _hist and _hist["records"] == 1 and _hist["named"] == {"execution unit": 1}
   and _hist["severities"] == {"fatal": 1} and _hist["first_at"] == _hist["last_at"] == 1793674220,
   str(_hist))
ck("the expensive scan is per boot, not `-b all`: the boot list is what bounds the work",
   "journalctl" in open(os.path.join(ROOT, "serve", "webui.py")).read()
   and '"-b", "all"' not in open(os.path.join(ROOT, "serve", "webui.py")).read())

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
# The page's own poll wraps its render in try/catch, so a broken section reads as an empty one, and a
# const used above its declaration (temporal dead zone) is exactly that class of break: string checks pass
# and the section never renders. Order is therefore asserted, not just presence.
mon_note = monjs[monjs.index('function renderStatus'):]
ck("a const the PCIe note uses is declared before the note is built (no temporal-dead-zone ReferenceError)",
   mon_note.index("const busyNow =") < mon_note.index("const pse = $(\"pcie-spark\")")
   and mon_note.index("const busyNow =") < mon_note.index("busyNow)"))
ck("idle link states are named instead of alarmed: parked (D3hot, Gen1) and sagging (D0, nothing served)",
   "parked, max \u00d7" in monjs and "below its maximum with nothing running" in monjs
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
ck("the page says what ended the previous boot, in the firmware record's own terms",
   'id="death-line"' in monhtml and "machine check in the" in monjs and "who they blame" in monjs
   and "the boot before this one ended with a" in monjs
   and '#death-line[data-tone="danger"]' in (WEB / "monitor.css").read_text())
ck("they are rendered at display size",
   'class="value value--xl"' in monhtml and ".value--xl{font-size:var(--st-fs-display)}" in (WEB / "web.css").read_text())
ck("all four rate slots exist and are written",
   all(f'id="{i}"' in monhtml and f'("{i}"' in monjs
       for i in ("rate-gentot", "rate-gen", "rate-prefill", "rate-step")))
# The per-stream mean used to be the first headline on this page. It is not any more, and the reason is on the
# record rather than in a design note: it divides the summed rate by the running gauge, which counts a request
# in the middle of a chunked prefill as running, and it invited the comparison "why is the aggregate smaller
# than one stream" that the step census in notes/context-262k.md answers (47.6% of wall seconds hold no request,
# 5.9% of records carry a ~1 s prefill chunk, 1.62 streams decode per record). A count cannot be misread that way.
ck("the decode headline row shows COUNTS, not a per-stream average",
   'big("rate-gentot", sn.generation_tokens_total, "tok out")' in monjs
   and "generation_tokens_total" in (monhtml + monjs) and 'id="rate-gentot"' in monhtml
   and 'id="rate-stream"' not in monhtml)
# Two generations of the same mistake have lived in this one slot, so both are named here. `tok/s` (a per-stream
# mean, killed because it divided by the running gauge, which counts a request sitting in a chunked prefill as
# decoding), and `decode_tok_s` (2026-11-04: generated tokens over the ITL family's sum, labelled "summed, decode
# time" when that sum is STREAM-seconds -- a per-stream rate in a summed card's clothes, 17.3 tok/s where the same
# span summed to 45.1 over wall seconds). The payload keeps the per-stream figures for the request rows and the
# titles; they must never reach a card's value, slot list or waveform. The prefill card is the deliberate
# exception: its headline IS a per-prefill-second rate, and its label and note line say so.
ck("the decode card's headline is DECODE output per DECODE second, and its label names the denominator",
   'big("rate-gen", perS, "tok/s")' in monjs
   and "A.decode_stream_tok_s : rate.decode_stream_tok_s" in monjs
   and "Decode · per stream-second" in monhtml
   and "Decode · summed, decode time" not in monhtml
   and '"decode_tok_s"' not in monjs and '"stream_tok_s"' not in monjs
   and "decode_time_pct" not in monjs and "prefill_tok_s_credited" not in monjs
   and "decode_tok_s" not in monhtml)
ck("the wall-second sum is NOT dropped: it sits on the note line labelled 'summed', with the fan-out as the "
   "multiplier that reaches it from the headline",
   'slot("note-gen", isnum(wallV)' in monjs
   and "A.generation_tok_s : rate.generation_tok_s" in monjs
   and "summed`" in monjs and "\\u00d7${streamsV" in monjs and "% dec" not in monjs,
   "the multiplier is now conditional (it has no denominator before the first decode step), but the summed "
   "figure never goes to a dash with it: wall seconds always exist")
ck("the card's spans line and poll line stay on the headline's basis (no silent switch to the wall sum mid-card), "
   "and a poll with no decode step says so instead of a 0",
   'spanList("decode_stream_tok_s")' in monjs and 'spanList("generation_tok_s")' not in monjs
   and "poll \\u00b7 no decode step" in monjs and '"poll \\u00b7 no window"' not in monjs)
ck("the trends waveform keeps the summed-wall basis and says why (a per-stream denominator does not exist in a "
   "window that was all prefill)",
   "hist.map((h) => h.gen_tok_s)" in monjs and "summed over WALL seconds" in monjs
   and "Decode · summed, wall" in monhtml)
ck("the payload keeps the per-record figures the new card and its waveform read",
   "hist.map((h) => h.gen_per_step)" in monjs
   and open(os.path.join(ROOT, "serve/webui.py")).read().count("gen_per_step") >= 4)
ck("the step card reconciles the step rate with the token rate (xN tokens a step, five characters in a slot)",
   "rate.tokens_per_step" in monjs and "\\u00d7${tps.toFixed(2)}" in monjs and "tokens a step" in monjs)
ck("one poll's read rate subtracts the cached prefix tokens too",
   abs(st["rate"]["prompt_read_tok_s"]
       - ((cur2["prompt_tokens_total"][0] - prev["prompt_tokens_total"][0])
          - (cur2["prefix_cache_hits_total"][0] - prev["prefix_cache_hits_total"][0])) / 10.0) < 1e-9,
   str(st["rate"]["prompt_read_tok_s"]))
ck("and the page says so when the cache is serving: `N% cache` in the prefill card's note slot",
   "% cache`" in monjs and "prompt_read_tok_s" in monjs and "prompt tok/s read" in monjs)
ck("the prefill rate names which of its two numbers is showing",
   "from KV growth" in monjs and "exact \\u00b7 credits at end" in monjs)
ck("the page says how old its numbers are (the sentence is the slot's title, in the markup and in what the page "
   "writes into it)", 'id="fresh"' in monhtml and " ago" in monjs
   and "How old these numbers are" in (monhtml + monjs))
ck("the page refetches on the engine's beat", "setTimeout(poll, 1000)" in monjs)
# The waveforms belong in one row, not sprinkled through the value cards: four series over the same span,
# each with last/peak/avg attached. (The aggregate series used to be drawn twice -- once big, once small.)
_trend = monhtml.split('grid--trends')[1].split("</section>")[0]
_rates = monhtml.split('grid--rates')[1].split("</section>")[0]
ck("every activity waveform is in the one trends row",
   all(f'id="{i}"' in _trend for i in ("perstep-spark", "gen-spark", "prefill-spark", "running-spark")))
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
   and "no records this window" in monjs and "0 streams" in monjs and "did not grow" in monjs
   and "nothing generated" in monjs
   and "drawn on a ${dnum(lo)}..${dnum(top)} axis" in monjs)
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
   "sn.itl_ms / sn.tpot_ms" in monjs and "Since launch - this window holds no" in monjs)
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
ck("a chain is one row: slot, driver, device chain, negotiated link, totals since boot, which counters, window",
   'class="pcie-rows"' in monhtml and all(x in monjs for x in
                                          ('"pcie-row"', 'class="chain p-chain"', 'class="link p-link"',
                                           'class="cnt p-cnt"', 'class="who p-who"', 'class="win p-win"')))
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
# tools/probe-load-link.py is the opposite instrument: it sends exactly one request, on purpose, to watch
# whether a PCIe link renegotiates around real card work. Its honesty line matters as much as its action: a short
# request cannot show whether a hop is marginal under a heavy transfer, and a reader who forgets that will
# over-read a clean result.
_pl = open(os.path.join(ROOT, "tools", "probe-load-link.py")).read()
compile(_pl, "probe-load-link.py", "exec")        # syntax only -- running it would load the serving box
ck("the link probe sends exactly one request, samples at a stated rate, and states what it cannot show",
   _pl.count("urllib.request.Request(") == 1 and "max_tokens" in _pl and "HZ = 10.0" in _pl
   and "what it cannot show is whether the hop is marginal under a heavy" in _pl)
# tools/recipe-check.py exists because measuring things here needs config changes, and a test config left running
# silently changes the meaning of every number the Monitor shows. So: get "production" from the launcher itself (a
# copied default goes stale the day native-flashnext.sh changes), and answer "改回来没有" in one command.
_rcs = open(os.path.join(ROOT, "tools", "recipe-check.py")).read()
import importlib.util as _ilu                                                   # noqa: E402
_spec = _ilu.spec_from_file_location("recipe_check", os.path.join(ROOT, "tools", "recipe-check.py"))
rc = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(rc)
_PROD_CMD = ("/venv/bin/vllm serve /models/M --served-model-name Qwen3.8 --host 0.0.0.0 --port 8080 "
             "--tensor-parallel-size 4 --max-model-len 262144 --max-num-seqs 32 --max-num-batched-tokens 8192 "
             "--gpu-memory-utilization 0.97 --kv-cache-dtype fp8_e4m3 --no-enable-prefix-caching "
             '--speculative-config {"method": "mtp", "num_speculative_tokens": 2} '
             '--compilation-config {"cudagraph_capture_sizes": [1,2,4], "max_cudagraph_capture_size": 256}')
_META = {"maxlen": "262144", "nseq": "32", "nbt": "8192", "kvdtype": "fp8_e4m3", "prefix_cache": "0", "mtp": "2"}
_CFG = {"enable_prefix_caching": "False", "cache_dtype": "fp8_e4m3"}
ck("the recipe check keeps no copy of the recipe: it asks the launcher (DRY=1) and reads the live serve",
   '"DRY": "1"' in _rcs and "FORCE" in _rcs
   and not any(n in _rcs for n in ("262144", "8192", "fp8_e4m3")),          # no copied defaults, anywhere
   "a default hardcoded outside TRUTH goes stale the day native-flashnext.sh changes")
ck("the production launch, taken from this tree's own launcher, passes the check",
   rc.compare(_PROD_CMD, _CFG, _META) == [], str(rc.compare(_PROD_CMD, _CFG, _META)))
ck("a PREFIX_CACHE=1 test config left running is caught and named (the launcher says off, the box says on)",
   [d["key"] for d in rc.compare(_PROD_CMD, {**_CFG, "enable_prefix_caching": "True"},
                                 {**_META, "prefix_cache": "1"})] == ["enable_prefix_caching"])
ck("an NBT left at a probe value is caught, even though the engine publishes no such field (the meta carries it)",
   [d["key"] for d in rc.compare(_PROD_CMD, _CFG, {**_META, "nbt": "4096"})] == ["max_num_batched_tokens"])
ck("MTP raised for a probe is caught, and so is a KV dtype the box ended up running without asking for",
   [d["key"] for d in rc.compare(_PROD_CMD, _CFG, {**_META, "mtp": "3"})] == ["num_speculative_tokens"]
   and [d["key"] for d in rc.compare(_PROD_CMD, {**_CFG, "cache_dtype": "auto"},
                                     {**_META, "kvdtype": "auto"})] == ["kv_cache_dtype"])
ck("a launch line's JSON-valued flags survive parsing (the launcher echoes ARGS unquoted, so the braces arrive "
   "spread across words)",
   rc.parse_args(_PROD_CMD)[0]["--speculative-config"].startswith('{"method"')
   and "num_speculative_tokens" in rc.parse_args(_PROD_CMD)[0]["--speculative-config"]
   and rc.parse_args(_PROD_CMD)[1] == {"--no-enable-prefix-caching"})
# tools/journal-pcie.py asks the page's questions from a terminal, per boot: AER by port and name, and the
# crash record the firmware kept. Read-only, and it must stay that way -- a tool that pokes the box while
# diagnosing it changes what it diagnoses.
_jp = open(os.path.join(ROOT, "tools", "journal-pcie.py")).read()
compile(_jp, "journal-pcie.py", "exec")            # syntax only -- running it reads the whole journal
ck("the journal tool reads the log and the kernel's read-only sysfs: it sends no requests, and its only writes "
   "are its own archive and topology snapshot -- never a control file (the reason it exists is to not poke the "
   "box it is diagnosing)",
   "journalctl" in _jp and "subprocess.run" in _jp
   and "urlopen(" not in _jp and "Request(" not in _jp
   and "w" not in re.findall(r'open\([^)]*["\'](\w*)["\']\)', _jp)      # no open(..., 'w') on anything
   and "echo" not in _jp and "sudo" not in _jp                          # nothing echoed into /sys, no privileges
   and "online" in _jp and (".read_text()" in _jp)                      # sysfs read, not written
   and 'p.write_text' in _jp and "cpu-topology" in _jp)   # the only two writes it makes, both its own
# tools/brief.py is the one-line-per-call state log (the series that outlives any one conversation). Same
# property as the other watchers: it observes, it does not participate.
_bf = open(os.path.join(ROOT, "tools", "brief.py")).read()
compile(_bf, "brief.py", "exec")
ck("the brief line only reads: one /health GET, no request bodies, nothing echoed into /sys",
   "/health" in _bf and "Request(" not in _bf and "data=" not in _bf and "POST" not in _bf
   and "sudo" not in _bf and "write_text" in _bf and "series.log" in _bf)
# gpu_busy_percent on this box is coarse, not finely readable: refused with nothing running, 0 stopped, 24-29
# idle, 46-100 under load, 95-99 through long decode. The page says so, because otherwise a flat 99 reads as
# "fully loaded" and a 0 as "healthy idle" when either could also mean "the launch let go".
ck("the cards line says busy % is coarse here, with the observed spread",
   "busy % is coarse here" in monjs and "amdgpu sysfs" in monjs)
ck("each waveform carries the level and the spread, in two slots that never wrap",
   'avg ${dnum' in monjs and 'peak ${dnum' in monjs
   and all(f'id="{i}"' in monhtml for i in ("perstep-note1", "perstep-note2", "gen-note1", "gen-note2",
                                             "prefill-note1", "prefill-note2", "running-note1", "running-note2",
                                             "kv-note1", "kv-note2")))
# A null is "no sample in this window", not "the series was at zero": the line has to break, and the filled
# area under it has to stop, or the gap reads as a measurement at the baseline.
_ck = (WEB / "common.js").read_text()
ck("a gap in a series is drawn as a gap, and its fill is dropped",
   "if (!isnum(v)) { if (pen) gaps++; pen = false; return; }" in _ck and "if (fill && !gaps)" in _ck)
ck("the span is named in points and seconds, since the ring counts points",
   "pts over" in monjs and "s.history" in monjs)
ck("a rate waveform averages only the windows that carried a rate, and the estimate only the growing ones",
   'keep = (v) => v !== 0' in monjs and 'keep: (v) => v > 0' in monjs
   and "count only the windows where the pool grew" in monjs)
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


# ------------------------------------------------- slots, attached journals, the serve-down page
ck("slots come from the firmware's table and are named on the chain, not the endpoint",
   (lambda ch: ch["c6:00.0"]["slot"] == "6" and ch["c6:00.0"]["slot_at"] == "0000:c4:00.0"
    and ch["c6:00.0"]["slot_role"] == "upstream port" and ch["06:00.0"]["slot"] == "8")(
       {c["bus"]: c for c in W.read_pcie(fakepci / "bus" / "pci" / "devices")}))
ck("a board with no slot table is not an error, it just has no slot to show",
   W.read_slots(fakepci / "bus" / "pci" / "nosuch") == {}
   and W.slot_of([], {}) == (None, None, None))

with tempfile.TemporaryDirectory() as td:
    jpath = Path(td) / "requests-8080.jsonl"
    owner = W.Journal(jpath, keep=10)
    for i in range(4):
        owner.append({"seq": i, "started_at": time.time(), "path": "/v1/chat/completions",
                      "prompt_tokens": 10, "completion_tokens": 2, "ok": True,
                      "wallclock_s": 0.5, "finished_at": time.time()})
    before = jpath.read_bytes()
    att = W.Journal(jpath, keep=10, readonly=True)
    ck("an ATTACHED journal replays the owner's rows and writes not one byte back",
       att.info()["readonly"] is True and att.info()["ok"] is True
       and [r["seq"] for r in att.replay()] == [3, 2, 1, 0]
       and att.append({"seq": 99, "started_at": time.time(), "path": "/x", "prompt_tokens": 1,
                       "completion_tokens": 1, "ok": True, "wallclock_s": 1.0,
                       "finished_at": time.time()}) is False
       and jpath.read_bytes() == before)
    missing = W.Journal(Path(td) / "nope.jsonl", keep=5, readonly=True)
    ck("an attached journal that cannot be read says so instead of reading as empty",
       missing.info()["ok"] is False and bool(missing.info()["error"]))

mjs = open(os.path.join(ROOT, "serve/web/monitor.js")).read()
ck("the page says what to do when the serve behind it stops answering",
   all(k in mjs for k in ("down-line", "the serve behind this page is not answering",
                          "attached read-only", "slot ${esc(c.slot)}")))
wsrc = open(os.path.join(ROOT, "serve/webui.py")).read()
ck("the launch hint is built after the meta file is read (before it, the page showed launch: None)",
   wsrc.index("self.meta = json.loads(") < wsrc.index("self.launch = launch_hint(self.meta)"))
ck("the hint reproduces the standing recipe from the launch's own parameters",
   W.launch_hint({}) is None
   and W.launch_hint({"maxlen": "262144", "nseq": "32", "nbt": "8192", "prefix_cache": "0"})
   == "MAXLEN=262144 NSEQ=32 NBT=8192 PREFIX_CACHE=0 WEBUI=same bash serve/native-flashnext.sh")
cjs = open(os.path.join(ROOT, "serve/web/common.js")).read()
ck("card rows carry the identity a person at the machine needs (BDF and slot), and every one of them is escaped",
   "PCI ${esc(g.bus)}" in cjs and "slot ${esc(g.slot)}" in cjs and "[g.card && esc(g.card), where]" in cjs
   and "const at = (s) =>" in cjs)                    # history alignment still by bus/card id
sh = open(os.path.join(ROOT, "tools/restart-sidecar.sh")).read()
ck("the sidecar -- the page that survives the serve -- gets the meta file and the journal attached",
   all(k in sh for k in ("--meta", "--journal-attach", "requests-${UP##*:}.jsonl")))
rsrc = open(os.path.join(ROOT, "tools/reseat-check.py")).read()
compile(rsrc, "tools/reseat-check.py", "exec")
ck("the reseat check reads sysfs only (no requests to the serve) and carries its own caveats",
   "urlopen(" not in rsrc
   and all(k in rsrc for k in ("--save", "--diff", "read_pcie", "MISSING after", "per boot",
                                 "different physical slot")))

# --------------------------------------------- chassis order: the panel's order is the machine's order
ck("slot ordering is numeric and handles sub-slots, where string order would put slot 10 second",
   sorted(["10", "2", "1-1", "1", "3"], key=lambda x: W.slot_key(x, "")) == ["1", "1-1", "2", "3", "10"])
ck("a card the firmware names no slot for sorts last, never into the middle of chassis order",
   [x["slot"] for x in sorted([{"slot": None, "bus": "99:00.0"}, {"slot": "10", "bus": "a:0"},
                               {"slot": "2", "bus": "c:0"}], key=lambda x: W.slot_key(x["slot"], x["bus"]))]
   == ["2", "10", None])

_g = [{"card": "card1", "bus": "c3:00.0", "slot": "3"}, {"card": "card2", "bus": "c6:00.0", "slot": "6"},
      {"card": "card3", "bus": "03:00.0", "slot": "1"}, {"card": "card4", "bus": "06:00.0", "slot": "8"}]
_c = [{"bus": x["bus"], "card": x["card"], "slot": x["slot"], "devices": [{"bdf": "0000:x", "max_speed": 32.0,
       "cor": 0, "nonfatal": 0, "fatal": 0}]} for x in _g]
W.order_by_slot(_g, _c)
ck("order_by_slot numbers the cards in chassis order and keeps the driver name as the key",
   [(x["pos"], x["slot"], x["card"]) for x in _g] == [(1, "1", "card3"), (2, "3", "card1"),
                                                      (3, "6", "card2"), (4, "8", "card4")]
   and [(x["pos"], x["slot"]) for x in _c] == [(1, "1"), (2, "3"), (3, "6"), (4, "8")])
ck("the chain ceiling is a speed, not the truthiness helper's bool (the page prints it as a number)",
   all(isinstance(x["devices"][0]["max_speed"], float) for x in _c)
   and max([(d.get("max_speed") or 0) for d in _c[0]["devices"]]) == 32.0)
ck("gpu_poll orders both lists and publishes the mapping, and both pages read the same order",
   "order_by_slot(self.gpus, chains)" in wsrc and 'payload["slot_map"] = self.slotmap' in wsrc
   and "slot_map=self.slotmap" in wsrc)
ck("the page leads with the slot and says which numbering it is using",
   all(k in mjs for k in ("slot-map", "chassis order", "firmware slot numbers, not driver names",
                          "slot ${esc(c.slot)}"))
   and all(k in open(os.path.join(ROOT, "serve/web/common.js")).read()
           for k in ("slot ${esc(g.slot)}", "#${g.pos} in slot order")))

# ---- the PCIe grid: seven columns, and the widths are a promise the rows have to keep
_css = open(os.path.join(ROOT, "serve/web/monitor.css")).read()
_mjs2 = open(os.path.join(ROOT, "serve/web/monitor.js")).read()


def _tracks(decl):
    """Count grid tracks, respecting minmax(...)/(180px,34%) so a comma inside a function is not a column."""
    out, depth, cur = [], 0, ""
    for ch in decl:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch.isspace() and depth == 0:
            if cur:
                out.append(cur)
                cur = ""
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


_pblk = _css.split(".pcie-row{display:grid;")[1].split("\n}")[0]
_ptk = _tracks(_pblk.split("grid-template-columns:")[1].split(";")[0])
_par = _pblk.split('grid-template-areas:"')[1].split('"')[0].split()
ck("the PCIe row grid declares as many tracks as areas, and every area has a class to land in",
   len(_ptk) == 7 and len(_par) == 7 and _par == ["slot", "drv", "chain", "link", "cnt", "who", "win"]
   and all(f".p-{a}{{grid-area:{a}}}" in _css for a in _par)
   and all(f'p-{a}' in _mjs2 for a in _par), f"{len(_ptk)} tracks / {len(_par)} areas")

# Track widths are arithmetic, not taste. The strings below are what the page actually renders -- run
# `node tests/render-monitor.mjs` and read the rows -- and the px-per-glyph comes from the sizes monitor.css
# itself sets for each cell (chain/who are 11px mono, link/cnt/win 13px mono, the slot label at the row's 16px).
# A track narrower than the longest string it must hold is not a narrow column, it is a wrapped row.
_MONO_ADV = 0.6          # a mono advance, as a fraction of the font size
_needs = {"slot": ("slot 6", 16.0), "drv": ("unnamed \u00b7 #12", 11.0),
          "chain": ("c6:00.0 \u2192 c5:00.0 \u2192 c4:00.0 \u2192 c0:03.1", 11.0),
          "link": ("16.0 GT/s \u00d716 (idle, max \u00d716)", 13.0),
          "cnt": ("144 corr \u00b7 0 uncorr", 13.0),
          "who": ("corr BadTLP 30 \u00b7 Timeout 13", 11.0),
          "win": ("+12 \u00b7 45.00/min \u00b7 quiet 4.2 h", 13.0)}


def _minem(track):
    """minmax(16em,1.7fr) or 5.2em -> the fixed lower bound in em: the width the layout is promised."""
    return float(track.split("(")[1].split("em")[0] if track.startswith("minmax(")
                 else track.split("em")[0])


for _a in _par:
    _s, _fs = _needs[_a]
    _need = len(_s) * _MONO_ADV * _fs
    ck("the %s track holds its longest live string (%d glyphs, %.0fpx)" % (_a, len(_s), _need),
       _minem(_ptk[_par.index(_a)]) * 16 >= _need,
       "track %s = %.0fpx vs %.0fpx needed" % (_ptk[_par.index(_a)],
                                               _minem(_ptk[_par.index(_a)]) * 16, _need))
ck("seven minimums plus the column gaps still fit below the 1360px breakpoint, so the single-line claim holds",
   sum(_minem(t) for t in _ptk) * 16 + 6 * 10 <= 1360,
   "%.0fpx at the narrowest" % (sum(_minem(t) for t in _ptk) * 16 + 60))
# The count cell's own text, with title attributes removed: a tooltip may spell the span out, rendered text may
# not repeat what the column header already says (11 glyphs, in a track sized for 19).
_cnt = re.sub(r'title="[^"]*"', "", _mjs2.split('class="cnt p-cnt"')[1].split('class="who p-who"')[0])
ck("no per-cell 'since boot' is repeated when the column header already says it",
   "since boot" not in _cnt and "errors since boot</span>" in _mjs2, _cnt.strip()[:160])
ck("the render harness stubs the globals the page has started to use (a missing one silences every row)",
   "AbortController" in open(os.path.join(ROOT, "tests", "render-monitor.mjs")).read())

# The page's body font ships a Latin subset (serve/web/fonts/outfit-latin-wght.woff2), so a code point outside
# that subset falls back to a system font -- and when nothing has it, the browser draws a missing-glyph box.
# That is exactly how the sparse-state icon arrived: U+24D8 (ⓘ) rendered as a "?"-looking box, which reads as
# "what is this?" rather than "there is information here". The icon is now drawn in CSS around an ASCII `i`.
_weball = "".join(open(os.path.join(ROOT, "serve", "web", f), encoding="utf-8").read()
                  for f in ("common.js", "monitor.js", "about.js", "monitor.html", "about.html",
                            "monitor.css", "web.css", "tokens.css"))
_cps = {int(m.group(1), 16) for m in re.finditer(r"\\u([0-9a-fA-F]{4})", _weball)} | {ord(c) for c in _weball
                                                                                     if ord(c) > 127}
# Everything below U+0100 is in the subset; the rest are the characters the pages have been rendering for weeks
# (em dash, curly quotes, ellipsis, arrow, almost-equal) -- present in every system fallback on this box.
_SAFE = {cp for cp in _cps if cp <= 0x00FF} | {0x2014, 0x201C, 0x201D, 0x2026, 0x2192, 0x2248}
ck("every glyph the pages emit is one their fonts actually have (U+24D8 came back as a missing-glyph box)",
   _cps <= _SAFE, sorted("U+%04X" % cp for cp in _cps - _SAFE))

# The detail has to be DOM text, not a `title` tooltip. A tooltip cannot be selected, cannot be copied, and
# vanishes when the pointer leaves -- which is exactly what the reader asked for after the demotion moved the
# numbers into one. So the count cell is a button, the panel is outside #pcie (which is rebuilt every tick), and
# the click is delegated to the container for the same reason.
ck("a chain's account opens as real text (the cell is a button, the panel is outside the rebuilt rows)",
   'class="p-glyph" data-bus="${esc(c.bus)}"' in _mjs2 and 'class="p-info" data-bus="${esc(c.bus)}"' in _mjs2
   and "pcie-detail" in _mjs2 and 'id="pcie-detail"' in open(os.path.join(ROOT, "serve", "web", "monitor.html")).read()
   and "paintPcieDetail(s);" in _mjs2,
   "the count cell, the container outside #pcie, and the call that keeps a pinned panel live each tick")
ck("the click is delegated to the row container, and Esc closes what opened",
   '$("pcie").onclick = (e) =>' in _mjs2 and 'closest("[data-bus]")' in _mjs2
   and 'e.key === "Escape" && detailBus' in _mjs2,
   "#pcie is replaced wholesale every second: a handler on a row would be garbage within the tick")
ck("the panel is selectable, monospace-aligned text (pre-wrap, so what is seen is what is copied)",
   ".pcie-detail{" in _css and "white-space:pre-wrap" in _css and "user-select:text" in _css
   and ".pcie-detail[hidden]{display:none}" in _css
   and "k.padEnd(14," in _mjs2,
   "[hidden] needs the display:none guard because .pcie-detail sets display itself")
ck("a pinned panel only writes the lines that changed (re-writing them would destroy the selection)",
   "if (el.textContent !== t) el.textContent = t;" in _mjs2)
ck("the sparse icon is still drawn, not typed (a Latin-subset font has no U+24D8)",
   'class="p-glyph"' in _mjs2 and ">i</button>" in _mjs2 and r"\u24d8" not in _mjs2
   and chr(0x24d8) not in _mjs2 and ".p-glyph{" in _css and "border-radius:50%" in _css)
_narrow = [x for x in _css.split("@media")[1:] if "pcie-row" in x and "grid-template-areas" in x]
ck("both narrow layouts keep all seven areas, spread over more lines (a dropped column is dropped information)",
   len(_narrow) >= 2 and all(sorted(set(x.split("grid-template-areas:")[1].split("}")[0]
                                         .replace('"', " ").split())) == sorted("slot drv chain link cnt who win".split())
                             for x in _narrow),
   f"{len(_narrow)} narrow layouts checked")

ck("the PCIe table has a header row on the same grid, naming each column",
   'phead.className = "pcie-row pcie-head"' in _mjs2
   and all(f'class="{cls} p-{a}"' in _mjs2 or f'p-{a}"' in _mjs2 for a, cls in
           [("slot", "b"), ("drv", "small"), ("chain", "chain"), ("link", "link"), ("cnt", "cnt"),
            ("who", "who"), ("win", "win")]))
ck("slot labels and driver names are separate columns, so one can be long without shoving the other",
   '<b class="p-slot">' in _mjs2 and 'class="small p-drv"' in _mjs2)

# --- which build is this page? ---------------------------------------------------------------------------
# "Is the live serve the new one?" used to cost an ssh and a look at the launch log. The header answers it, and
# About lists it as a fact, so the answer survives into a screenshot and into a bug report.
st = W.stack_info()
try:
    import r9700_vllm
    want = r9700_vllm.__version__
except Exception:
    want = None
ck("the pages can name the plugin build they are talking to", bool(st.get("plugin")), str(st)[:140])
ck("and it is the version this tree declares", want is None or st.get("plugin") == want,
   f"page {st.get('plugin')!r} vs tree {want!r}")
ck("and the commit, whenever the code came from a checkout rather than an install",
   bool(st.get("commit")) or not (Path(ROOT) / ".git").is_dir(), str(st)[:140])
ck("a dirty checkout is told, because a bare commit would claim a build the repository does not contain",
   'out["dirty"]' in (Path(ROOT) / "serve" / "webui.py").read_text()
   and 'stack.dirty' in (Path(ROOT) / "serve" / "web" / "common.js").read_text()
   and '(dirty)' in (Path(ROOT) / "serve" / "web" / "about.js").read_text())
for _pg in ("monitor", "about"):
    ck(f"the {_pg} header carries the id the build line is written into",
       'id="brand-line"' in (Path(ROOT) / "serve" / "web" / f"{_pg}.html").read_text())
ck("and both pages call it, from one shared helper that remembers the header's base text so a per-second "
   "repaint cannot append the version again",
   "function brandStack" in (Path(ROOT) / "serve" / "web" / "common.js").read_text()
   and all("brandStack(s);" in (Path(ROOT) / "serve" / "web" / f"{_pg}.js").read_text()
           for _pg in ("monitor", "about")))
ck("About lists the stack as a fact, not only in the header",
   '["Stack", s.stack?.plugin' in (Path(ROOT) / "serve" / "web" / "about.js").read_text())

# --- the trailing averages --------------------------------------------------------------------------
# The poll window is 1 s and since() is the whole run, and the question people actually ask sits between them:
# "what has this box been doing for the last quarter of an hour?" A span mean must be a delta of the engine's
# cumulative counters, not an average of the 1 s samples -- that would weight an idle second the same as a second
# with six streams running, which on this launch is most seconds.
def _cur(t, run=6):
    ser = lambda v: [v, v, 1]
    # SELF-CONSISTENT on purpose, because the decode denominators get checked against the token rate rather than
    # just for being present. The shape: 50 engine records a second at 20 ms each (which fills the second), 2
    # streams decoding in every one of them -> 100 ITL observations a second summing to 2.0 s of STREAM-seconds,
    # and request_decode_time_seconds carries the same 2.0 s once per request (that equality is the identity
    # vLLM's two families have -- see the ITL_STEP note). 2 accepted tokens per stream-step = 200 generated
    # tokens a second; 0.05 requests a second finishing owns 4,000 of those tokens over a 40 s decode span and a
    # 140,000-token prompt over a 20 s prefill span (7,000 prompt tok/s), with 0.8 s of queue in front.
    return {"generation_tokens_total": ser(200.0 * t), "prompt_tokens_total": ser(7000.0 * t),
            "e2e_request_latency_seconds_count": ser(0.05 * t),
            "e2e_request_latency_seconds_sum": ser(0.05 * 60.8 * t),       # queue .8 + prefill 20 + decode 40
            "request_prefill_time_seconds_sum": ser(1.0 * t), "request_prefill_time_seconds_count": ser(0.05 * t),
            "request_decode_time_seconds_sum": ser(2.0 * t), "request_decode_time_seconds_count": ser(0.05 * t),
            "request_prompt_tokens_sum": ser(7000.0 * t), "request_prompt_tokens_count": ser(0.05 * t),
            "request_generation_tokens_sum": ser(200.0 * t), "request_generation_tokens_count": ser(0.05 * t),
            # 10 ms per ACCEPTED token: a 20 ms step turning out 2 tokens, so tokens_per_step reconciles to 2
            "request_time_per_output_token_seconds_sum": ser(0.05 * 0.01 * t),
            "request_time_per_output_token_seconds_count": ser(0.05 * t),
            # ONE OBSERVATION PER (REQUEST x STEP): 100 a second at 20 ms = 2.0 s of stream-seconds a second, i.e.
            # TWO streams in their decode phase at an average moment. Deliberately above 1: a fixture that decodes
            # one stream at a time cannot tell a per-stream denominator from a share of the span, and that
            # confusion is exactly the bug this block exists to keep out of the page.
            "inter_token_latency_seconds_sum": ser(2.0 * t), "inter_token_latency_seconds_count": ser(100.0 * t),
            "time_to_first_token_seconds_sum": ser(0.05 * 20.8 * t), "time_to_first_token_seconds_count": ser(0.05 * t),
            "request_queue_time_seconds_sum": ser(0.05 * 0.8 * t), "request_queue_time_seconds_count": ser(0.05 * t),
            "spec_decode_num_drafts_total": ser(100.0 * t),
            "spec_decode_num_accepted_tokens_total": ser(100.0 * t),
            "spec_decode_num_draft_tokens_total": ser(200.0 * t),
            # a cache that hits on 10% of the prompt tokens. The EXACT skipped count (prompt_tokens_cached_total,
            # which also covers an external KV transfer) is 800 a second while the local prefix-cache counter says
            # 700, and the read rate must follow the exact one: read is 6,200 a second, not 6,300.
            "prefix_cache_queries_total": ser(7000.0 * t), "prefix_cache_hits_total": ser(700.0 * t),
            "prompt_tokens_cached_total": ser(800.0 * t),
            # the iteration RECORD counter: 50 records a second, each carrying 144 tokens (7,000 prompt + 200
            # generated over 50 records) -- the fixture's own version of a prefill-heavy launch
            "iteration_tokens_total_count": ser(50.0 * t), "iteration_tokens_total_sum": ser(144.0 * t),
            "num_requests_running": ser(float(run)), "num_requests_waiting": ser(0.0),
            # two series 40% full each, exactly as parse_metrics would report them: sum 0.8, max 0.4, count 2.
            # The ring's means must be 40%, never the 80% the sum gives.
            "gpu_cache_usage_perc": [0.8, 0.4, 2]}

# 13 snapshots at the ring's own spacing, with running alternating 6 / 2 so a mean of 4 is the right answer and
# a single poll's running is provably the wrong one.
_snaps = [W.avg_sample(_cur(t, 6 if (t // 5) % 2 == 0 else 2), float(t)) for t in range(60, 125, 5)]
_avg = W.trailing(_snaps, _cur(124.0), 124.0)["1m"]
ck("a span rate is the delta of the cumulative counter over the span",
   abs(_avg["generation_tok_s"] - 200.0) < 1e-9 and abs(_avg["prompt_tok_s"] - 7000.0) < 1e-6, str(_avg)[:200])
ck("the two speeds the box is judged on, token-weighted from the engine's own seconds, both on a STREAM basis",
   abs(_avg["prefill_stream_tok_s"] - 6200.0) < 1e-6 and abs(_avg["decode_stream_tok_s"] - 100.0) < 1e-9,
   f"prefill-read {_avg['prefill_stream_tok_s']} decode-per-stream {_avg['decode_stream_tok_s']}")
# The prefill rate counts what was READ. vLLM credits cached prefix tokens to prompt_tokens_total, so a rate over
# credited tokens is not a read speed -- it is high by the hit rate. Strata's run average subtracts `reused` for
# the same reason (serve/web/app.js), and both numbers are kept here so the difference is visible, not implied.
# The subtraction uses prompt_tokens_cached_total (the exact skipped count, local hit + external transfer) when the
# build publishes it; the local prefix_cache_hits counter alone would have said 6,300 and left an external hit
# inside the numerator looking like read work.
ck("the prefill rate's numerator is read tokens (credited minus SKIPPED, preferring the exact counter), with the "
   "credited version alongside",
   abs(_avg["prompt_read_tokens"] - 6200.0 * 64.0) < 1e-6
   and abs(_avg["prefill_stream_tok_s_credited"] - 7000.0) < 1e-6
   and abs(_avg["prefix_hit_pct"] - 10.0) < 1e-9,
   f"read {_avg['prompt_read_tokens']} credited/s {_avg['prefill_stream_tok_s_credited']} hit {_avg['prefix_hit_pct']}%")
# The decode denominator is the ITL family's sum, which is STREAM-seconds -- one observation per request per step,
# so a step with two streams decoding accrues twice. Tokens over stream-seconds is therefore a PER-STREAM rate,
# and stream-seconds over the span is a STREAM COUNT (2.0 here), not a share of the span and not capped at 100.
# The identity the page prints -- summed = per-stream x streams -- is what this pins.
ck("decode time from the per-(request x step) family is STREAM-seconds: 2.0 s a second means 2.0 streams decoding, "
   "not 200% of the span and not a share",
   abs(_avg["decode_time_s"] - 2.0 * 64.0) < 1e-9 and abs(_avg["decode_streams"] - 2.0) < 1e-9
   and abs(_avg["decode_stream_tok_s"] * _avg["decode_streams"] - _avg["generation_tok_s"]) < 1e-9,
   f"{_avg['decode_time_s']} s over {_avg['span_s']} s = x{_avg['decode_streams']}, "
   f"{_avg['decode_stream_tok_s']} tok/stream-s")
ck("the per-record fan-out comes from the two counters that mean it (100 ITL observations over 50 records = 2 streams a step)",
   abs(_avg["streams_per_record"] - 2.0) < 1e-9, str(_avg["streams_per_record"]))
ck("the request-end family reports the SAME stream-seconds (the two families are one sum), so both denominators agree",
   abs(_avg["decode_time_req_s"] - 100.0) < 1e-9, str(_avg["decode_time_req_s"]))
# One poll, same rule: the divisor is stream-seconds, and a window with no decode step in it has no denominator.
# The poll block's own derive(), kept under a name: the same 1 s window is what three of the checks below read.
_pt = W.derive(_cur(200.0), _cur(199.0), 1.0, 1_000_000, {}, "observe", 0, [])["rate"]
# A rate of 0 there would claim a measured throughput of nothing when the truth is "this second was prefill, or
# idle" -- which is a fact about the window, not a number to average with the others.
ck("a poll's per-stream decode rate is its tokens over its stream-seconds, with the stream count beside it",
   abs(_pt["decode_stream_tok_s"] - 100.0) < 1e-9 and abs(_pt["decode_streams"] - 2.0) < 1e-9
   and abs(_pt["generation_tok_s"] - 200.0) < 1e-9,
   str(_pt)[:200])
ck("counters that did not move give no per-stream decode rate rather than a rate of zero",
   W.derive(_cur(200.0), _cur(200.0), 1.0, 1_000_000, {}, "observe", 0, [])
   ["rate"]["decode_stream_tok_s"] is None)
_nostep = dict(_cur(200.0)); _nostep.pop("inter_token_latency_seconds_sum")
_nostep_b = dict(_cur(199.0)); _nostep_b.pop("inter_token_latency_seconds_sum")
ck("without the per-step family the request family stands in (2.0 s of stream-seconds -> 100 tok/s)",
   abs(W.derive(_nostep, _nostep_b, 1.0, 1_000_000, {}, "observe", 0, [])
      ["rate"]["decode_stream_tok_s"] - 100.0) < 1e-9,
   str(W.derive(_nostep, _nostep_b, 1.0, 1_000_000, {}, "observe", 0, [])["rate"]["decode_stream_tok_s"]))
ck("a per-stream average over a span divides by the MEAN concurrency of the span, not by this poll's running",
   abs(_avg["running_mean"] - 4.0) < 1e-9 and abs(_avg["stream_tok_s"] - 50.0) < 1e-9,
   f"running_mean {_avg['running_mean']} stream (gauge-divided) {_avg['stream_tok_s']}")
# Two per-stream means live in the payload and only one of them is right. `stream_tok_s(_live)` divides the summed
# rate by the RUNNING gauge, which counts a request sitting in the middle of a chunked prefill as a producer; the
# 1,284 s step census (notes/context-262k.md: 5.9% of records carry a prefill chunk, 47.6% of wall seconds hold no
# request at all) is what makes that wrong. `decode_stream_tok_s` divides by STREAM-SECONDS, which only a request
# that actually got a token in a step accrues. Same payload, 33.3 (200 tok/s over a 6-request gauge) against 100.
ck("the running-gauge mean and the stream-seconds mean are different numbers, and the second is the one with a "
   "checked multiplier",
   abs(_pt["stream_tok_s_live"] - 200.0 / 6) < 1e-9 and abs(_pt["decode_stream_tok_s"] - 100.0) < 1e-9,
   f"gauge {_pt['stream_tok_s_live']} stream-seconds {_pt['decode_stream_tok_s']}")
ck("span latencies are delta(sum)/delta(count) of the histogram families",
   abs(_avg["itl_ms"] - 20.0) < 1e-9 and abs(_avg["ttft_ms"] - 20800.0) < 1e-9
   and abs(_avg["queue_ms"] - 800.0) < 1e-9 and abs(_avg["prompt_tokens_mean"] - 140000.0) < 1e-6
   and abs(_avg["accepted_per_draft"] - 1.0) < 1e-9 and abs(_avg["tokens_per_step"] - 2.0) < 1e-9,
   str(_avg)[:240])
ck("a span can answer in COUNTS as well as rates, and per record as well as per second -- the decode headline "
   "row shows these because a count cannot be misread as an average",
   abs(_avg["generation_tokens"] - 12800.0) < 1e-6 and abs(_avg["prompt_tokens"] - 448000.0) < 1e-3
   and abs(_avg["records"] - 3200.0) < 1e-6 and abs(_avg["gen_per_step"] - 4.0) < 1e-9
   and abs(_avg["drafts_per_step"] - 2.0) < 1e-9, str(_avg)[:300])
ck("and the pool's mean over the span comes from the same gauge the sparkline draws -- a MEAN over its series "
   "(two ranks 40% full is a 40% pool, not an 80% one)",
   abs(_avg["kv_used_pct_mean"] - 40.0) < 1e-9 and abs(_avg["requests_per_min"] - 3.0) < 1e-9, str(_avg)[:240])
ck("every field the span is asked to report is fed by a name the ring actually stores",
   all(_avg.get(k) is not None for k in
       ("generation_tok_s", "prompt_tok_s", "prefill_stream_tok_s", "decode_stream_tok_s", "stream_tok_s",
        "requests_per_min", "itl_ms", "tpot_ms", "stream_tok_s_hist", "ttft_ms", "queue_ms", "e2e_ms",
        "prefill_ms", "decode_ms", "prompt_tokens_mean", "generation_tokens_mean", "tokens_per_step",
        "accept_pct", "accepted_per_draft", "running_mean", "waiting_mean", "kv_used_pct_mean",
        "generation_tokens", "prompt_tokens", "records", "gen_per_step", "drafts_per_step", "decode_time_s",
        "decode_streams", "streams_per_record", "decode_time_req_s", "prefill_streams", "prompt_read_tokens",
        "prefill_stream_tok_s_credited", "prefix_hit_pct")),
   str({k: _avg.get(k) for k in _avg if _avg.get(k) is None})[:220])
# How many prefills are in flight, as its own field: prefill_tok_s and prompt_tok_s agreed to 4% on this launch
# because long prefills serialize, and a reader cannot tell that coincidence from a law without the factor.
ck("the prefill span reports its own stream factor (1.0 prefill-second a second = prefills serialize here)",
   abs(_avg["prefill_streams"] - 1.0) < 1e-9, str(_avg["prefill_streams"]))

_bad = _cur(124.0); _bad["generation_tokens_total"] = [10.0, 10.0, 1]      # the engine restarted inside the span
_r = W.trailing(_snaps, _bad, 124.0)["1m"]
ck("a counter that went DOWN is reported as a restart, not as a large negative rate",
   _r["reset"] and _r["generation_tok_s"] is None and _r["decode_stream_tok_s"] is None
   and _r["accept_pct"] is None, str(_r)[:200])
# The families the page never SHOWED are where a negative used to leak out: a restart that leaves the generated
# token counter ahead but the request count behind produced a `requests_per_min` of -8.4 and means built on a
# negative observation count.
_rn2 = _cur(124.0)
_rn2["e2e_request_latency_seconds_count"] = [0.0, 0.0, 1]
_r2 = W.trailing(_snaps, _rn2, 124.0)["1m"]
ck("a request-count decrease is a restart too: no negative requests_per_min, no negative means",
   _r2["reset"] and _r2["requests_per_min"] is None and _r2["ttft_ms"] is None, str(_r2)[:200])
_rn3 = _cur(124.0)
_rn3["request_prefill_time_seconds_count"] = [0.0, 0.0, 1]
_rn3["request_prefill_time_seconds_sum"] = [-40.0, -40.0, 1]
_r3 = W.trailing(_snaps, _rn3, 124.0)["1m"]
ck("a negative OBSERVATION count never divides (a negative latency in ms is not a measurement)",
   _r3["prefill_ms"] is None, str(_r3["prefill_ms"]))

_tiny = [W.avg_sample(_cur(t), float(t)) for t in (120, 122, 124)]
_t = W.trailing(_tiny, _cur(124.0), 124.0)
ck("a ring younger than the span says what it averaged instead of lending the span its name",
   _t["1h"]["span_s"] < 20 and _t["1h"]["generation_tok_s"] is None
   and _t["15m"]["span_s"] == _t["1m"]["span_s"], str(_t["1h"])[:160])

_mid = [W.avg_sample(_cur(t), float(t)) for t in range(40, 125, 5)]
_m = W.trailing(_mid, _cur(124.0), 124.0)
ck("and the same page at the same moment can have a real 1 m and a still-labelled 15 m",
   abs(_m["1m"]["span_s"] - 64.0) < 1e-6 and _m["15m"]["generation_tok_s"] is not None
   and _m["15m"]["span_s"] < 900 and _m["1h"]["span_s"] < 90,
   f"1m {_m['1m']['span_s']} 15m {_m['15m']['span_s']} 1h {_m['1h']['span_s']}")

_partial = {k: v for k, v in _cur(60.0).items() if not k.startswith("inter_token_latency")}
_p = W.trailing([W.avg_sample(_partial, 60.0)], _cur(124.0), 124.0)["1m"]
ck("a family the base snapshot does not have is a dash, not a division by the whole cumulative counter",
   _p["itl_ms"] is None and abs(_p["generation_tok_s"] - 200.0) < 1e-9,
   f"itl {_p['itl_ms']} gen {_p['generation_tok_s']}")
ck("the payload carries all three spans so the page can label what it has and wait for the rest",
   sorted(s.get("avg", {})) == ["15m", "1h", "1m"], str(sorted(s.get("avg", {}))))

# --- the hour-long waveforms -----------------------------------------------------------------------
# A point every 15 s, each holding the rate OVER that interval. Sampling a 1 s window once every 15 s would draw a
# heartbeat and miss the work between the beats -- which is the same mistake the trailing averages exist to fix.
class _Slow:
    slowhist, hist = deque(maxlen=W.SLOW_KEEP), []
    slow_point = W.Hub.slow_point

_sh = _Slow()
_p1 = _sh.slow_point(_cur(200.0), 1_000_000, 200.0)
ck("the first point has no rate to report (there is no interval yet)",
   _p1["gen_tok_s"] is None and _p1["running"] == 6.0 and _p1["c"]["gen"] == 40000.0, str(_p1)[:160])
_sh.slowhist.append(_p1)
_sh.hist = [{"at": 188.0, "running": 4.0}, {"at": 199.0, "running": 8.0}]      # 4 and 8 running in the interval
_p2 = _sh.slow_point(_cur(215.0), 1_000_000, 215.0)
ck("a point's rate is the interval's delta, not a sample of one poll",
   abs(_p2["gen_tok_s"] - 200.0) < 1e-9 and abs(_p2["prompt_tok_s"] - 7000.0) < 1e-6, str(_p2)[:200])
ck("and its per-stream rate divides by the mean running ACROSS the interval (4 and 8 running -> 6)",
   abs(_p2["running"] - 6.0) < 1e-9 and abs(_p2["stream_tok_s"] - 200.0 / 6.0) < 1e-9,
   f"running {_p2['running']} stream {_p2['stream_tok_s']}")
ck("the KV-growth series is the same interval difference of the gauge",
   abs(_p2["kv_tok_s"] - 0.0) < 1e-6 and abs(_p2["kv_used_pct"] - 40.0) < 1e-9, str(_p2)[:200])
ck("the per-record series is a ratio of two deltas (3,000 generated over 750 records), not a rate",
   abs(_p2["gen_per_step"] - 4.0) < 1e-9 and _p2["c"]["iter"] == 10750.0, str(_p2)[:220])
_sh.slowhist.clear()
_sh.slowhist.append({"at": 300.0, "gen_tok_s": 1.0, "prompt_tok_s": 1.0, "kv_tok_s": 1.0, "kv_used_pct": 1.0,
                     "running": 1.0, "c": {"gen": 9e9, "prompt": 9e9, "kv": 9e9}})   # a counter ahead of now
_back = _sh.slow_point(_cur(315.0), 1_000_000, 315.0)
ck("counters that went backwards (an engine restart inside the interval) read None, not a negative rate",
   _back["gen_tok_s"] is None and _back["stream_tok_s"] is None and _back["prompt_tok_s"] is None,
   str(_back)[:200])
ck("but the KV-growth series is not guarded that way: a negative there is the pool draining, which is its use",
   _back["kv_tok_s"] is not None and _back["kv_tok_s"] < 0, str(_back["kv_tok_s"]))
ck("a slow point taken before the record counter existed reports no ratio rather than dividing by nothing",
   _back["gen_per_step"] is None, str(_back)[:160])
ck("the payload carries the slow ring and says how far apart its points are, capped like every other ring",
   len(s.get("history_slow", [])) <= W.SLOW_KEEP and s.get("slow_step_s") == W.SLOW_STEP,
   f"{len(s.get('history_slow', []))} pts, step {s.get('slow_step_s')}")

# --- the layout cannot move when a number changes width -----------------------------------------
# Every number line in the rate and trend rows is a SLOT: one line, ellipsized, full text in the title. The
# reason is arithmetic: these lines carry numbers whose width changes every poll ("2.0" then "8,481" then
# "16.2k"), a line that re-wraps changes its card's height, the card's height changes the grid row, and the row
# moves everything below it. The checks below keep the mechanism honest; tests/render-monitor.mjs enforces the
# character budget on what the live page actually renders.
_wh = open(os.path.join(ROOT, "serve/web/monitor.html")).read()
_wc = open(os.path.join(ROOT, "serve/web/web.css")).read()
_slot_html = {m.group(2) for m in re.finditer(r'class="([^"]*)"[^>]*?\bid="([^"]+)"', _wh)
              if "slot" in m.group(1).split()}
_slot_js = set(re.findall(r'slot\("([^"]+)"', _mjs2)) | {
    x for pair in re.findall(r'trend\("[^"]+", "([^"]+)", "([^"]+)"', _mjs2) for x in pair}
ck("every slot the page writes is declared in the markup (an undeclared one silently stops rendering)",
   _slot_js <= _slot_html, f"missing from HTML: {sorted(_slot_js - _slot_html)}")
ck("and every slot in the markup is written (a stale one shows a dash forever)",
   _slot_html <= _slot_js, f"never written: {sorted(_slot_html - _slot_js)}")
ck("a slot is one line: nowrap, ellipsis, tabular digits",
   ".slot{" in _wc and all(x in _wc.split(".slot{")[1][:260]
                           for x in ("white-space:nowrap", "text-overflow:ellipsis",
                                     "font-variant-numeric:tabular-nums")),
   _wc.split(".slot{")[1][:180] if ".slot{" in _wc else "no .slot rule in web.css")
ck("the rate row is three slots per card and the trend row two, so the two rows are the same shape",
   len(re.findall(r'id="note-[a-z]+"', _wh)) == 4 and len(re.findall(r'id="spans-[a-z]+"', _wh)) == 4
   and len(re.findall(r'id="poll-[a-z]+"', _wh)) == 4
   and len(re.findall(r'id="(perstep|gen|prefill|running)-note[12]"', _wh)) == 8,
   f"{len(re.findall(chr(34) + 'id=' + chr(34) + 'note-', _wh))} note slots")
_cjs2 = open(os.path.join(ROOT, "serve/web/common.js")).read()
ck("numbers in those slots are printed as DIGITS (dnum): grouped by thousands, no kilo or mega anywhere",
   "const dnum = " in _cjs2 and "knum" not in _cjs2 and _mjs2.count("dnum(") >= 12 and "knum(" not in _mjs2
   and "1e6)" not in _cjs2.split("const pct")[0] and "10000 ?" not in _cjs2,
   f"{_mjs2.count('dnum(')} uses in monitor.js; a k/M branch in common.js would put 16.2k back")
ck("an integer prints as an integer, so a measured 0 is 0 and two streams are 2 (not 0.0 and 2.0)",
   "Number.isInteger(n) ? fmt(n)" in _cjs2)
ck("the width discipline moved from the formatter into the copy, and the harness measures the copy: no slot may "
   "abbreviate, and no slot may print a bare signed number",
   '/\\d(?:[.,]\\d+)?[kM]\\b/' in open(os.path.join(ROOT, "tests/render-monitor.mjs")).read()
   and 'shows a bare signed number' in open(os.path.join(ROOT, "tests/render-monitor.mjs")).read())
ck("the pool-draining case is named in words instead of a signed rate under a prefill label",
   'now draining ${dnum(-last, d)}' in monjs and 'const drain = last < 0 && !keep(last)' in monjs)
ck("a unit on a spans line belongs to each number, so a span that has reset does not inherit one",
   'spanAll("kv_used_pct_mean", ["1m"], "%")' in monjs and "${dnum(v)}${unit}" in monjs
   and ") %`" not in monjs)
ck("whether the prefix cache is ON is stated, from the engine's own config -- not left to the reader to infer "
   "from a clause that may or may not appear",
   'const pcState = String(((s.config || {}).enable_prefix_caching ?? "")).toLowerCase()' in monjs
   and 'const pcOff = pcState === "false" || pcState === "0"' in monjs
   and "` cache off`" not in monjs and "cache off" in monjs
   and "This launch runs with the prefix cache OFF" in monjs
   and '["Prefix caching", yesno(c.enable_prefix_caching)]' in (WEB / "about.js").read_text())
ck("and the state is only claimed when the engine published it: an unknown config keeps the silence, not an "
   "invention",
   monjs.count("pcOff") == 3 and "pcState === \"unknown\"" not in monjs)
ck("a poll window with nothing credited says so in words, not as a bare duration like `poll 1s`",
   'nothing in ${Math.round((rate.window_s || 0) * 10) / 10}s' in monjs)
ck("and the old always-wraps note class is gone, so nothing outside the slot mechanism re-wraps",
   "small--avg" not in _wc and "small--avg" not in _wh)

# -- the socket and the key: what the pages promise, the server keeps ------------------------------------
# The audit of 2026-11-04: the sidecar listened on every interface while every URL it prints says
# 127.0.0.1, and both guards compared the key with ==. The socket address is checked on the live
# sidecar of this run, not on the source text: a promise the source makes and the socket breaks is
# exactly the bug being pinned.
ck("the sidecar binds the loopback every URL it prints promises; 0.0.0.0 stays an operator's choice",
   srv.server_address[0] == "127.0.0.1"
   and 'cfg.get("bind", "127.0.0.1")' in wsrc and "--bind" in wsrc and "R9K_WEB_BIND" in wsrc,
   str(srv.server_address))
ck("both guards compare the key in constant time: a plain == answers with the first differing byte",
   "hmac.compare_digest" in wsrc and 'got == self.hub.cfg["key"]' not in wsrc
   and "hmac.compare_digest" in (Path(ROOT) / "serve/plugin/r9700_webui/__init__.py").read_text())
ck("the card head escapes what it interpolates, like the file's own rule says (sysfs labels are not markup)",
   "slot ${esc(g.slot)}" in (_web / "common.js").read_text()
   and "esc(g.name || g.card)" in (_web / "common.js").read_text())

print("PASS" if fail == 0 else "FAIL", f"({fail} failing)")
sys.exit(1 if fail else 0)
