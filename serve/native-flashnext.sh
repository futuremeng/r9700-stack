#!/bin/bash
# Native (no-container) launch of Qwen3.8-Flash-Next on stock vLLM + the r9700_vllm plugin.
# Same knobs as serve/serve.sh, but vllm runs straight out of a venv instead of the pinned ROCm 10 image.
# Why: this box has no docker and no root (serve.sh's `sudo docker run` is not available).
#   VENV=~/venv-r9700  GPUS=0,1,2,3 TP=4 OFFLOAD_GB=0 NSEQ=16 bash serve/native-flashnext.sh
MODEL=${MODEL:-$HOME/models/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ}
VENV=${VENV:-$HOME/venv-r9700}
REPO=${REPO:-$(dirname "$(realpath "$0")")/..}
PY=$VENV/bin/python3

# Machine-local launch defaults, deliberately kept outside the repo. The repo's own defaults are the conservative
# ones: HOST=127.0.0.1, which is also upstream's (`args.host or "127.0.0.1"`). This box is not served over loopback
# -- `ss` shows 192.168.1.130 and 192.168.1.200 holding :8080 open -- so the bind address is a fact about the box,
# not about the code, and it lives here where it survives a checkout reset and gets named in the log at every
# launch instead of being remembered by whoever last typed the command. LOCAL_ENV= overrides the path.
# notes/launch-methodology.md section 10.
LOCAL_ENV=${LOCAL_ENV:-$HOME/.config/r9700-stack/native.env}
if [ -f "$LOCAL_ENV" ]; then
  # `set -a` so the file's facts reach the child processes as well as this script: sourcing alone only makes shell
  # variables, and anything a child reads from the environment (serve/webui.py's R9K_WEB_BIND default, for one)
  # would silently keep its own default. serve/webui.sh sources the same file the same way.
  set -a; . "$LOCAL_ENV"; set +a; echo "local env: $LOCAL_ENV"
fi
: ${HOST:=127.0.0.1}
# stock ROCm 7.2-era vLLM wheels link OpenMPI; on a box without root we unpacked the runtime debs locally
OPENMPI=${OPENMPI:-$HOME/opt/openmpi/usr/lib/x86_64-linux-gnu}
[ -d "$OPENMPI" ] && export LD_LIBRARY_PATH=$OPENMPI${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
LOG=${LOG:-$HOME/logs/vllm-$(date +%H%M%S).log}; mkdir -p "$(dirname "$LOG")"

# the plugin's kernels, built by kernels/build.sh for gfx1201
SO=$REPO/r9700_vllm/kernels/libr9k.so
[ -f "$SO" ] || { echo "missing $SO (cd kernels && HIPCFLAGS=-I\$HOME/opt/rocm-dev/opt/rocm/core-10.1/include ./build.sh)" >&2; exit 1; }

# plugin defaults from serve/flashnext.sh (measured; the reason for each is in that file)
: ${R9K_FOLD:=1}; : ${R9K_AR_QUANT:=1}
: ${R9K_HC_FP8:=r9k}; : ${R9K_HC_FP8_DECODE:=1}; : ${R9K_HC_FP8_DECODE_MAXM:=255}
: ${R9K_HC_FP8_DOWN:=1}; : ${R9K_HC_FREE_BF16:=1}
: ${R9K_FP8_BLOCK:=block}; : ${R9K_FP8_QA:=1}
: ${R9K_DRAFT_LMHEAD:=mxfp4}; : ${R9K_TARGET_LMHEAD:=mxfp4}
export R9K_FOLD R9K_AR_QUANT R9K_HC_FP8 R9K_HC_FP8_DECODE R9K_HC_FP8_DECODE_MAXM \
       R9K_HC_FP8_DOWN R9K_HC_FREE_BF16 R9K_FP8_BLOCK R9K_FP8_QA R9K_DRAFT_LMHEAD R9K_TARGET_LMHEAD
for v in $(env | grep -o '^R9K_[A-Z0-9_]*'); do export "$v"; done

export HIP_VISIBLE_DEVICES=${GPUS:-0,1,2,3} VLLM_ROCM_USE_AITER=${AITER:-0} HSA_ENABLE_IPC_MODE_LEGACY=0
export GPU_MAX_HW_QUEUES=${HWQ:-1} HSA_ENABLE_MWAITX=${MWAITX:-1} OMP_NUM_THREADS=8
: ${R9K_MTP_QUANT_ROWS:=32768}   # bf16->MXFP4 requant chunk: 2.68 GiB of fp32 temporaries at 2^18 rows, 0.34 GiB here
export R9K_MTP_QUANT_ROWS PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export R9K_LIB=$SO VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$HOME/vllm-native-cache} TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$HOME/vllm-native-triton}

# Pre-flight. Measured 2026-10-31: SIGTERM does not stop the ranks -- pkill'ing the API server leaves four
# VLLM::Worker_TP* holding 27-31 GiB a card, and the next launch dies with "Free memory on device cuda:0
# (0.0/31.86 GiB)". Refuse rather than eat a 2-minute failure; STOP=1 clears them, FORCE=1 launches anyway.
# (notes/launch-methodology.md: this check, why a launch is started in tmux, and what notices a death.)
if [ "${STOP:-0}" = 1 ]; then
  pkill -9 -f 'VLLM[::]' 2>/dev/null; pkill -9 -f 'vll[m] serve' 2>/dev/null; sleep 5
  echo "leftover vLLM processes after STOP: $(pgrep -cf 'VLLM[::]|vll[m] serve')"
  echo "STOP=1 stops and exits: nothing was launched. Rerun the same command without STOP=1 to bring it back."
  exit 0
fi
LEFT=$(pgrep -cf 'VLLM[::]' || true)
# Reading the recipe is not a launch, so GPU occupancy is not its business: making ENVRECIPE need FORCE=1 would
# mean the gate could not be run while a serve is up, which is exactly the moment it gets asked.
if [ "${ENVRECIPE:-0}" != 1 ] && [ "${FORCE:-0}" != 1 ] && [ "${LEFT:-0}" -gt 0 ]; then
  echo "refusing to launch: $LEFT vLLM process(es) still holding the GPUs (SIGTERM does not stop the ranks)." >&2
  echo "  clear them: STOP=1 bash serve/native-flashnext.sh    (FORCE=1 to launch anyway)" >&2
  exit 1
fi

# Defaults by tensor-parallel size. NBT=8192 / NSEQ=32 are measured at TP4 with everything resident in VRAM:
# the chunk is +5.4% prefill at 7.8k (9,331 -> 9,831) and +3.5% at 31k, +4.4% aggregate at conc 16, for 64k KV
# tokens (563,692 -> 499,815, still 15.2x at 32k); the slots are +18% aggregate at 32 clients in flight
# (1,216 -> 1,433, 48 is worse at 1,366), with single-stream decode and long-context prefill untouched, and the
# per-stream rate reads 50.5 tok/s at 32 in flight against 79.8 at 16. At TP2 neither default is measured and
# both cost KV the two ranks do not have -- serve/flashnext.sh keeps 8 / 4096 there (conc-8 283 -> 401 tok/s),
# so use those until someone repeats the sweep on a two-card box.
: ${NSEQ:=$([ "${TP:-4}" = 4 ] && echo 32 || echo 8)}
: ${NBT:=$([ "${TP:-4}" = 4 ] && echo 8192 || echo 4096)}

# Context. The checkpoint itself declares text_config.max_position_embeddings = 262144, and this box's own
# request ledger says 262144 is the traffic it carries: 1,483 kept requests, p99 prompt 245,792 tokens, 1,195 of
# them at or above 32,000. A 32,768 default is therefore not conservative, it refuses four requests in five.
# Raising it does not cost KV either -- README.md measures the pool getting BIGGER at the full context (540,202
# against 497,588 tokens) because the sizing run is bounded by NBT, not by MAXLEN. Note that every launch logged
# on this box before 2026-11-03 came up at 32,768 even though README.md's own "what this box runs" line says
# MAXLEN=262144: the default was the trap, so it is now that line's value. MAXLEN= still overrides.
: ${MAXLEN:=262144}

# Main KV in e4m3. Twelve of the 48 layers do full attention, and their K/V is 12,288 of the 14,348 bytes a token
# costs a card, so quantizing exactly those -- the side/indexer caches QSA selects with stay bf16, so top-k selection
# does not move -- is a capacity change. Measured on this box 2026-11-03 with two paired fresh instances
# (notes/kv-fp8-qsa.md): pool 540,202 -> 1,021,931 tokens, 262,144-token concurrency 2.06x -> 3.90x, GSM8K no-think
# 800 @8 95.6% -> 96.0%, HumanEval 164 96.95% -> 98.17% (McNemar b=2, c=0, p=0.50), needle intact cold and warm out
# to 250,078-token prompts, prefill flat to +-0.5% at every size to 77,811 tokens. The price: 1.5-4% of aggregate
# decode, and it is MTP acceptance (2.61 -> 2.55 tokens a step), not the kernel, which measures 7-11% FASTER by
# itself; one 32 s AOT recompile on the first fp8 launch (the next one is 0.5 s); and an attention block size of
# 880 instead of 448, so a freed block is twice as coarse. With this box's ledger (p99 prompt 245,792, KV usage seen
# at 82% under three streams) that trade is worth taking, so it is the default. It is also two settings that travel
# together: stock vLLM's AMD QSA raises on a non-bf16 main cache unless the plugin's reader is installed, so the
# launcher refuses the combination that cannot start. bf16 rollback:
# R9K_QSA_KVF8=0 KVDTYPE= bash serve/native-flashnext.sh
: ${R9K_QSA_KVF8:=1}
if [ "${R9K_QSA_KVF8}" = 1 ]; then
  : ${KVDTYPE:=fp8_e4m3}
elif [ -n "${KVDTYPE:-}" ]; then
  echo "R9K_QSA_KVF8=0 with KVDTYPE=$KVDTYPE: stock QSA refuses a non-bf16 main KV cache." >&2
  echo "  bf16: R9K_QSA_KVF8=0 KVDTYPE= bash serve/native-flashnext.sh" >&2
  exit 1
fi
export R9K_QSA_KVF8

# The recipe includes the ENVIRONMENT, not only the flags. Half of this stack's switches are env names (R9K_*),
# they decide which kernel runs and therefore what a measured number means, and a gate that reads only the
# engine's measured config cannot see any of them: on 2026-11-04 two launches died inside r9k_router_gemm_kernel
# and the experiment that followed was a single env name (R9K_ROUTER_MAX_M=4) that recipe-check said nothing
# about. ENVRECIPE=1 prints the block as JSON and touches nothing -- the same no-write rule as DRY, because a
# preview that writes the recipe of a running serve is a page that lies (see the DRY note further down).
# The snapshot sits after the last default this script sets: taken earlier it missed R9K_QSA_KVF8, and the gate
# then blamed the live serve for a name the launcher itself exports.
r9k_env_json() { env | sed -n 's/^\(R9K_[A-Z0-9_]*\|R9700_[A-Z0-9_]*\)=\(.*\)$/\1\t\2/p' \
  | python3 -c 'import json, sys
d = {}
for ln in sys.stdin:
    k, _, v = ln.rstrip("\n").partition("\t")
    d[k] = v
print(json.dumps(d, sort_keys=True))'; }
if [ "${ENVRECIPE:-0}" = 1 ]; then echo "env recipe: $(r9k_env_json)"; exit 0; fi

# The prefix cache, ON -- which is upstream's default (`config/cache.py`: `enable_prefix_caching: bool = True`).
# This launcher has been passing --no-enable-prefix-caching since it was written, inherited from
# serve/flashnext.sh, where the choice WAS measured: 5056943 (2026-09-25, "prefix caching off by default for
# Flash-Next; short-prefill root cause"). The mechanism serve/serve.sh documents next to its own ON default: on a
# hybrid GDN/attention model caching moves the recurrent state to "align" mode, the scheduler then aligns every
# prefill chunk end to the mamba block size, and a prompt that is not a multiple of it pays an extra forward pass
# (~285 ms of CPU-bound Python each; a 1.5k prompt ran as two), which read 2.8k tok/s of 2k-token prefill against
# ~4.7k with the cache off. That is a real cost and it is a SHORT-prompt cost.
# What this box carries is not short prompts: p50 in the 30k tier and p99 245,792 prompt tokens in its own request
# ledger. Measured against this box's traffic, tools/prefix-replay.py replays 6,560 model calls / 581.9M prompt
# tokens out of the local conversation records through the engine's own rules (880-token hybrid blocks, contiguous
# prefix reuse only, LRU-evictable retained blocks, this pool's 1,175 blocks): 98.2% of prompt tokens are already
# resident, 77,851 s of prefill back over four days, and zero admission queueing at every load tried, while OFF
# (which has 186 blocks free at its peak) starts queueing at 2x. The endpoint those same conversations go through
# reports cacheRead for 99.1% of their prompt tokens: measured on the identical calls, not simulated.
# The price, measured on this box rather than modelled: mamba_cache_mode goes "none" -> "align", the attention
# block 880 -> 832, and the pool reads 989,534 tokens instead of 1,009,899 (-2.0%). notes/release-v0.2.4.md's
# one-page GDN state is Stage-1 for mode "none" only, so with the cache on this runs on vLLM's stock pages -- and
# the -2.0% above is that cost as the engine published it here, not the 27B table's +18% quoted there.
# What owed a measurement was taken on 2026-11-04, on the cache-on launch, 15 distinct prompts a tier:
# `ITEMS=0,66,152 bench/ttft_breakdown.py http://127.0.0.1:8080 cacheon` -> TTFT 37 ms at 45 tokens, 188 ms at
# 1,412, 371 ms at 3,323 -- 3x better than 2026-09-25's cache-on numbers and better than that day's cache-off ones --
# with EXACTLY 1.00 iterations per request at every tier, i.e. the align-mode extra pass no longer happens on this
# stack. notes/prefix-cache-decision.md. PREFIX_CACHE=0 still turns it off, and tools/recipe-check.py fails the
# launch if the running engine disagrees with this line.
: ${PREFIX_CACHE:=1}

# Text-only by default, and it is a choice rather than an accident: `--language-model-only` puts every multimodal
# limit at 0 (the engine says so at boot: "All limits of multimodal modalities supported by the model are set to 0,
# running in text-only mode", and an image request comes back 400 "At most 0 image(s) may be provided"). The
# checkpoint does carry a vision tower (config.json has vision_config / image_token_id / vision_{start,end}_token_id,
# and the directory has preprocessor_config.json and video_preprocessor_config.json), so this flag -- not the
# weights -- is what keeps the box reading text.
# Why text-only stays the default: every number in this repo (262k context, fp8 KV, NBT 8192, MTP-2, the R9K_*
# kernels for MoE/GDN/QSA) was measured on the language path; the vision tower would take card memory out of the KV
# pool (1,242 blocks / 977,726 tokens at util 0.97), boot enables a different attention mode (`mm_prefix`, disabled
# today because mm is configuration-disabled), and mm + prefix caching + align has never been measured here.
# MM=1 turns it on for a named test; MM_IMAGE / MM_VIDEO set the per-prompt limits. The live answer is not in the
# engine's measured config, so tools/recipe-check.py compares the launch FLAGS: launcher truth against the live
# process's own argv, published by the merged plugin in /api/status.
: ${MM:=0}; : ${MM_IMAGE:=4}; : ${MM_VIDEO:=0}
# `--limit-mm-per-prompt` caps HOW MANY images, not how big they are, and on this checkpoint the default size cap is
# enormous: preprocessor_config.json says size.longest_edge = 16,777,216 pixels (with shortest_edge = 65,536). With
# patch 16 and merge 2, one image token covers 32x32 = 1,024 pixels, so that default is 16,384 image tokens per
# image -- and boot agrees: "Encoder cache will be initialized with a budget of 16384 tokens". Measured on
# 2026-11-05: a 2,560x6,236 screenshot arrived as 15,664 prompt tokens, 6% of a 262,144-token window, from one image.
# MM_MAX_PIX is therefore the knob that bounds what an image can cost; 4,194,304 px is ~4,096 tokens, which keeps a
# 1,200x654 documentation PNG (760 tokens) untouched and shrinks the giant screenshot to 4,059. MM_MIN_PIX is left at
# the checkpoint's own shortest_edge (tiny images get upscaled to ~64 tokens).
: ${MM_MAX_PIX:=4194304}; : ${MM_MIN_PIX:=65536}

ARGS=(--served-model-name "${SERVED_NAME:-Qwen3.8}" --host "$HOST" --port "${PORT:-8080}"
      --tensor-parallel-size "${TP:-4}" --max-model-len "${MAXLEN}"
      --max-num-seqs "$NSEQ" --max-num-batched-tokens "$NBT"
      --gpu-memory-utilization "${UTIL:-$([ "${TP:-4}" = 4 ] && echo 0.97 || echo 0.94)}"
      --reasoning-parser "${REASONING:-qwen3}" --tool-call-parser "${TOOLCALL:-qwen3_coder}" --enable-auto-tool-choice)
if [ "$MM" = 1 ]; then
  ARGS+=(--limit-mm-per-prompt "{\"image\": ${MM_IMAGE}, \"video\": ${MM_VIDEO}}"
         --mm-processor-kwargs "{\"size\": {\"longest_edge\": ${MM_MAX_PIX}, \"shortest_edge\": ${MM_MIN_PIX}}}")
else
  ARGS+=(--language-model-only)
fi
[ "${OFFLOAD_GB:-0}" != 0 ] && ARGS+=(--cpu-offload-gb "$OFFLOAD_GB" --cpu-offload-params experts)
[ -n "${KVDTYPE:-}" ] && ARGS+=(--kv-cache-dtype "$KVDTYPE")
[ "$PREFIX_CACHE" = 0 ] && ARGS+=(--no-enable-prefix-caching)
[ "${EAGER:-0}" = 1 ] && ARGS+=(--enforce-eager)
# Draft depth, measured 2026-11-01 on this box (fresh instances, 3 x 300-token @32 samples each side, plus ITL):
#   MTP=2: @32 1,762 / 1,538 / 1,750 -> median 1,755 aggregate at 57.6 tok/s a stream; @16 1,265 / 83.4;
#          median ITL 13.0 ms (1 stream), 20.8 (8), 37.1 (32); single-stream 179.8 tok/s; GSM8K 98/100, needles 3/3
#   MTP=3: @32 1,385 / 1,488 / 1,509 -> median 1,488 at 48.8 a stream; @16 1,202 / 80.7;
#          median ITL 14.0 / 23.4 / 49.8 ms; single-stream 197.6 tok/s; GSM8K 97/100, needles 3/3
# So 2 wins everything except one-stream latency: +18% aggregate and +18% per-stream at 32 in flight, ITL at 32
# down 26%, and it frees 10,922 KV tokens (491,520 -> 502,442). It costs ~9% of single-stream decode (197.6 ->
# 179.8; bench.py's decode-only metric reads 180.1 -> 170.4). The mechanism is the row budget: a decode step runs
# NSEQ x (MTP+1) rows -- 128 at MTP=3, 96 at MTP=2 -- while the third draft row only adds ~0.3 accepted tokens a
# step (2.54 -> 2.21 accepted at conc 1). Above ~16 streams the four cards are sitting on the ~300 W envelope, so
# those 32 rows a step are exactly what is scarce. MTP=3 is the better choice for a single interactive stream:
# MTP=3 bash serve/native-flashnext.sh.
MTP=${MTP:-2}
[ -n "$MTP" ] && ARGS+=(--speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": $MTP${SPEC_EXTRA:+, $SPEC_EXTRA}}")
[ -n "${ATTN:-}" ] && ARGS+=(--attention-backend "$ATTN")
CC=()
: ${CGSIZES:=1,2,4,8,16,24,32,48,64,96,128,192,256}   # graphs for prefill chunks <= 256 (serve/flashnext.sh)
[ -n "${CGSIZES:-}" ] && CC+=("\"cudagraph_capture_sizes\": [$CGSIZES], \"max_cudagraph_capture_size\": ${CGSIZES##*,}")
[ ${#CC[@]} -gt 0 ] && ARGS+=(--compilation-config "{$(IFS=,; echo "${CC[*]}")}")

# WEBUI=1: the Monitor/About pages on this same host at PORT+1 (serve/webui.py; notes/webui.md). observe mode
# by default -- it only GETs /health, /v1/models and /metrics, so the launch itself is untouched -- plus this
# launch's recipe as a meta file, which is what the About page shows as the recipe. Started before vllm is up
# on purpose: the sidecar says "engine unreachable" until it answers, rather than dying with the launch.
#
# WEBUI=same: the same pages INSIDE this serve, on its own port, from serve/plugin (an endpoint plugin for the
# routes and a stat-logger plugin for the per-request records). Cheaper and richer: /metrics comes from this
# process's own registry instead of a socket, and every finished request carries the engine's own
# queued/prefill/decode split instead of a window mean. It costs two things a route cannot invent: the pages
# die with the launch, and no request TEXT is ever seen (the engine reports tokens and timings, not bodies).
# Needs the plugin installed in $VENV: pip install -e serve/plugin. Needs one API server: with
# --api-server-count N>1 each server has its own registry, its own poller and its own request ring.
if [ "${WEBUI:-0}" = 1 ] || [ "${WEBUI:-}" = same ]; then
  # The recipe file is named after the port the PAGES are served on, which is PORT+1 for the sidecar and
  # PORT itself for WEBUI=same: the plugin reads the file the launch wrote, and there is no second port in
  # merged mode for WEBUI_PORT to mean. Naming it after PORT+1 in both modes is how the merged About page
  # lost its "how this box was launched" card: written to -8081.json, read from -8080.json.
  WPORT=${WEBUI_PORT:-$([ "${WEBUI:-}" = same ] && echo "${PORT:-8080}" || echo "$(( ${PORT:-8080} + 1 ))")}
  META=${WEBUI_META:-$HOME/r9700-web-meta-$WPORT.json}
  D="$(dirname "$(realpath "$0")")"
  VER=$($VENV/bin/python -c "import importlib.metadata as m; print('native vllm '+m.version('vllm'))" 2>/dev/null \
        || echo "native")
  # DRY must not write anything, including this file. It bit on 2026-11-04: a DRY=1 WEBUI=same run meant to read back
  # the launch line overwrote the recipe of the engine that was actually running, so the About page of a cache-off
  # serve described a cache-on launch, and tools/recipe-check.py's meta-vs-engine agreement became a lie until a
  # restart. METAONLY=1 is the deliberate version of the same thing: refresh the recipe file for a launch that is
  # already up (or was started by hand) without starting anything.
  if [ "${DRY:-0}" != 1 ]; then
    META=$META bash "$D/webmeta.sh" venv="$VENV" model="$MODEL" served_name="${SERVED_NAME:-Qwen3.8}" \
      gpus="${GPUS:-0,1,2,3}" tp="${TP:-4}" maxlen="$MAXLEN" nseq="$NSEQ" nbt="$NBT" \
      reasoning="${REASONING:-qwen3}" toolcall="${TOOLCALL:-qwen3_coder}" \
      util="${UTIL:-$([ "${TP:-4}" = 4 ] && echo 0.97 || echo 0.94)}" offload_gb="${OFFLOAD_GB:-0}" \
      kvdtype="${KVDTYPE:-}" mtp="$MTP" attn="${ATTN:-}" cg_sizes="${CGSIZES:-}" prefix_cache="$PREFIX_CACHE" \
      host="$HOST" \
      eager="${EAGER:-0}" log="$LOG" fold="${R9K_FOLD:-1}" ar_quant="${R9K_AR_QUANT:-1}" hc_fp8="${R9K_HC_FP8:-r9k}" \
      hc_fp8_decode="${R9K_HC_FP8_DECODE:-1}" fp8_block="${R9K_FP8_BLOCK:-block}" fp8_qa="${R9K_FP8_QA:-1}" \
      target_lmhead="${R9K_TARGET_LMHEAD:-mxfp4}" draft_lmhead="${R9K_DRAFT_LMHEAD:-mxfp4}" \
      mtp_quant_rows="${R9K_MTP_QUANT_ROWS:-}" mm="$MM" \
      mm_effective="$([ "$MM" = 1 ] && echo "on" || echo "off")" \
      mm_image="$([ "$MM" = 1 ] && echo "$MM_IMAGE" || echo 0)" \
      mm_video="$([ "$MM" = 1 ] && echo "$MM_VIDEO" || echo 0)" \
      mm_max_pix="$([ "$MM" = 1 ] && echo "$MM_MAX_PIX" || echo 0)" \
      mm_min_pix="$([ "$MM" = 1 ] && echo "$MM_MIN_PIX" || echo 0)" \
      mm_max_tok="$([ "$MM" = 1 ] && echo "$((MM_MAX_PIX / 1024))" || echo 0)" \
      mm_if_on="MM_IMAGE=${MM_IMAGE} MM_VIDEO=${MM_VIDEO} MM_MAX_PIX=${MM_MAX_PIX} MM_MIN_PIX=${MM_MIN_PIX}" \
      mm_limits="$([ "$MM" = 1 ] && echo "image ${MM_IMAGE}, video ${MM_VIDEO}, max ${MM_MAX_PIX} px (~$((MM_MAX_PIX / 1024)) tokens each)" \
                     || echo "image 0, video 0 (--language-model-only)")" \
      env="$(r9k_env_json)" extra="${EXTRA:-}"
  fi
  if [ "${DRY:-0}" = 1 ] || [ "${METAONLY:-0}" = 1 ]; then
    [ "${DRY:-0}" = 1 ] && echo "  recipe file NOT written (preview): $META"
    if [ "${METAONLY:-0}" = 1 ]; then echo "  recipe file refreshed for the launch that is running, nothing started: $META"; exit 0; fi
  fi
  if [ "${WEBUI:-}" = same ]; then
    D="$(dirname "$(realpath "$0")")"
    if ! "$VENV/bin/python" -c 'import sys; from importlib.metadata import entry_points
sys.exit(0 if any(e.name == "r9700_webui" for e in entry_points(group="vllm.endpoint_plugins")) else 1)'; then
      echo "WARN WEBUI=same: the monitor plugin is not installed in $VENV; starting WITHOUT the pages."
      echo "WARN   install it with:  $VENV/bin/pip install -e $D/plugin   (or run ./setup.sh)"
    else
      # VLLM_PLUGINS is a STRICT allowlist, not an "also load these" list. Naming only ours silently
      # disables this repo's own r9700 plugin -- the compressed-tensors scheme for this checkpoint -- and
      # the engine dies four minutes into the load with NotImplementedError on in_proj_qkvz. Build it from
      # what is discoverable (serve/webui-plugins.py), never by hand.
      PLG=$("$VENV/bin/python" "$D/webui-plugins.py" --require r9700 --quiet)
      VLLM_PLUGINS=$PLG; export VLLM_PLUGINS
      export WEBUI_META=$META                                  # the About page's recipe file
      # The request journal, on by default with the pages: this box hard resets without a shutdown record, and
      # the engine's request list dies with the launch, so the rows describing what it was doing at the moment
      # of death are worth more than the memory it costs. ~0.5 kB per finished request, fsynced once each.
      JOURNAL=${R9700_WEBUI_JOURNAL:-$HOME/logs/requests-$WPORT.jsonl}
      mkdir -p "$(dirname "$JOURNAL")"; export R9700_WEBUI_JOURNAL=$JOURNAL
      echo "  request journal: $JOURNAL (the list survives a restart or a reset)"
      echo "Monitor and About pages on http://127.0.0.1:${PORT:-8080}/monitor (inside the serve itself)"
      echo "  VLLM_PLUGINS=$VLLM_PLUGINS"
      echo "  recipe file the About page reads: $META"
    fi
  else
    # A preview writes nothing and starts nothing: the sidecar stays down for DRY and METAONLY too, or a DRY run
    # meant to read the launch line would occupy PORT+1 and the next real launch would find the port taken.
    if [ "${DRY:-0}" != 1 ] && [ "${METAONLY:-0}" != 1 ]; then
      WEBUI_PORT=$WPORT UPSTREAM=http://127.0.0.1:${PORT:-8080} META=$META VERSION="$VER" \
      JOURNAL=${JOURNAL:-$HOME/logs/requests-$WPORT.jsonl} bash "$D/webui.sh"
    fi
  fi
fi
[ "${DRY:-0}" = 1 ] && { echo "launch: $VENV/bin/vllm serve $MODEL ${ARGS[*]} $EXTRA"; exit 0; }
echo "launch: $VENV/bin/vllm serve $MODEL ${ARGS[*]} $EXTRA  (log: $LOG)"
exec "$VENV/bin/vllm" serve "$MODEL" "${ARGS[@]}" $EXTRA
