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
ck("stream tok/s is 1/ITL", abs(st["rate"]["stream_tok_s"] - 50.0) < 1e-9)
ck("spec acceptance 90/120", abs(st["rate"]["accept_pct"] - 75.0) < 1e-9, str(st["rate"]["accept_pct"]))
ck("tokens per engine step", abs(st["rate"]["tokens_per_iter"] - 5.0) < 1e-9)
ck("KV uses the worst rank", abs(st["live"]["kv_used_pct"] - 72.0) < 1e-9, str(st["live"]["kv_used_pct"]))
ck("running is per-rank max", st["live"]["running"] == 3.0)
ck("prefix hit rate", abs(st["live"]["prefix_hit_pct"] - 100.0 / 3.0) < 1e-6)
ck("a counter that did not move reads None, not 0", W.derive(cur2, cur2, 10.0, cap2, info2, "observe", 0, [])
   ["rate"]["ttft_ms"] is None)

# The names vLLM 0.32.0+rocm723 actually publishes on this box (captured from a live native launch,
# 2026-11-02): kv_cache_usage_perc instead of gpu_cache_usage_perc, request_time_per_output_token_seconds,
# no num_requests_swapped at all, `_created` stamps beside every counter, and a hybrid GDN model whose real
# KV pool (497,588 tokens) is NOT num_gpu_blocks x block_size (1,230 x 448 = 551,040).
METRICS32 = """
vllm:kv_cache_usage_perc{engine="0",model_name="Qwen3.8"} 0.25
vllm:num_requests_running{engine="0",model_name="Qwen3.8"} 2.0
vllm:num_requests_waiting{engine="0",model_name="Qwen3.8"} 1.0
vllm:cache_config_info{block_size="448",cache_dtype="auto",engine="0",gpu_memory_utilization="0.97",\
    kv_cache_size_tokens="497588",model_name="Qwen3.8",num_gpu_blocks="1230"} 1.0
vllm:generation_tokens_total{engine="0",model_name="Qwen3.8"} 100.0
vllm:generation_tokens_created{engine="0",model_name="Qwen3.8"} 1762000000.0
vllm:inter_token_latency_seconds_sum{engine="0",model_name="Qwen3.8"} 3.0
vllm:inter_token_latency_seconds_count{engine="0",model_name="Qwen3.8"} 150.0
vllm:prefix_cache_queries_total{engine="0",model_name="Qwen3.8"} 0.0
vllm:prefix_cache_hits_total{engine="0",model_name="Qwen3.8"} 0.0
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
# engine step against one per request of that request's mean per ACCEPTED token. Measured on the live
# native launch 2026-11-02: 15.6 ms a step, 3.19 ms per accepted token (4.9 tokens a step at MTP=2). A
# stream's tok/s is 1/step.
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
ck("a stream's tok/s is 1 / step", abs(ss["rate"]["stream_tok_s"] - 1e3 / 15.0) < 1e-9,
   str(ss["rate"]["stream_tok_s"]))
ck("a hybrid model's KV pool is kv_cache_size_tokens, not blocks x block_size", cap32 == 497588, str(cap32))
ck("a gauge the build does not have reads absent, not zero", st32["live"]["swapped"] is None,
   str(st32["live"]["swapped"]))
ck("_created stamps are not mistaken for series", "generation_tokens_created" not in cur32, str(sorted(cur32)))
ck("labels the newer build carries are harvested", info32.get("cache_dtype") == "auto"
   and info32.get("gpu_memory_utilization") == "0.97", str(sorted(info32)))

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
   src.count("for k in ROW_KEYS") == 2 and "usage" in W.ROW_KEYS and "timings" in W.ROW_KEYS, str(W.ROW_KEYS))
use = ring.start("/v1/chat/completions", "Qwen3.8", "{}", True)
use["usage"] = {"prompt_tokens": 11, "completion_tokens": 20}
use["first_token_s"] = 0.0
time.sleep(0.05)
ring.finish(use, "completed", status=200)
d = ring.get(f"req-{use['seq']}")
ck("decode speed from usage", (d["timings"]["predicted_per_second"] or 0) > 100, str(d["timings"]))
ck("estimate flagged only when it is one", d["timings"]["tokens_are_estimate"] is False
   and ring.get(f"req-{recs[-1]['seq']}")["timings"]["tokens_are_estimate"] is True)


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
    hm = dev / "hwmon0"
    hm.mkdir(exist_ok=True)
    (hm / "name").write_text("amdgpu\n")
    (hm / "temp1_input").write_text(f"{temp}\n")
    (hm / "power1_average").write_text("199870000\n")            # this kernel: average, not input
    (hm / "freq1_label").write_text("sclk\n")
    (hm / "freq1_input").write_text("1180000000\n")
    (hm / "freq2_label").write_text("mclk\n")
    (hm / "freq2_input").write_text("2520000000\n")
(fake / "card1").mkdir(parents=True, exist_ok=True)          # a DRM node with no amdgpu counters
# and one that refuses to answer: on this box gpu_busy_percent and the whole hwmon set come back EBUSY
# -- with no compute context open as well as with one -- so that is a state the reader has to survive.
# It must read as "not readable", never as "0% busy".
stub = fake / "card5" / "device"
stub.mkdir(parents=True, exist_ok=True)
stub.joinpath("uevent").write_text("DRIVER=amdgpu\nPCI_SLOT_NAME=0000:08:00.0\n")
(stub / "gpu_busy_percent").mkdir()                          # reading it raises, like EBUSY does
(stub / "mem_info_vram_total").write_text(f"{32 * 2**30}\n")
gpus = W.read_gpus(fake)
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


def sidecar(mode="proxy", key="", keep=50):
    cfg = {"port": 0, "upstream": f"http://127.0.0.1:{up_port}", "mode": mode, "key": key, "poll": 0.2,
           "keep": keep, "meta": "", "gpus": str(fake), "smi": None, "timeout": 10.0, "verbose": False,
           "version": "test"}
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


hub, srv, port = sidecar("proxy")
time.sleep(0.6)                                            # let the poll threads land a window

status, body = call(port, "/healthz")
ck("healthz", status == 200 and body["ok"] and body["upstream_reachable"], str(body))

status, s = call(port, "/api/status")
ck("status reads the engine", s["reachable"] and s["model_name"] == "Qwen3.8", str(s)[:120])
ck("status carries config labels", s["config"].get("block_size") == "1088")
ck("status carries cards", len(s["gpus"]) == 3 and s["gpus"][0]["busy_pct"] == 87, str(len(s.get("gpus", []))))
ck("status carries models", s["models"][0]["id"] == "Qwen3.8")
ck("status engine version", s["engine"]["vllm_version"] == "test")

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
ck("a row names its token counts and its own prefill",
   "prefill ${seconds(t.prefill_s)}" in monjs and "fmt(u.prompt_tokens ?? 0, 0)} -> " in monjs)
ck("the wall-clock card splits that one request into queue, prefill and decode",
   'id="wall-split"' in (WEB / "monitor.html").read_text() and "queue ${seconds(lt.queue_s)}" in monjs)
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
