#!/bin/bash
# Native (no-container) launch of Qwen3.8-Flash-Next on stock vLLM + the r9700_vllm plugin.
# Same knobs as serve/serve.sh, but vllm runs straight out of a venv instead of the pinned ROCm 10 image.
# Why: this box has no docker and no root (serve.sh's `sudo docker run` is not available).
#   VENV=~/venv-r9700  GPUS=0,1,2,3 TP=4 OFFLOAD_GB=0 NSEQ=16 bash serve/native-flashnext.sh
MODEL=${MODEL:-$HOME/models/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ}
VENV=${VENV:-$HOME/venv-r9700}
REPO=${REPO:-$(dirname "$(realpath "$0")")/..}
PY=$VENV/bin/python3
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
if [ "${STOP:-0}" = 1 ]; then
  pkill -9 -f 'VLLM[::]' 2>/dev/null; pkill -9 -f 'vll[m] serve' 2>/dev/null; sleep 5
  echo "leftover vLLM processes after STOP: $(pgrep -cf 'VLLM[::]|vll[m] serve')"; exit 0
fi
LEFT=$(pgrep -cf 'VLLM[::]' || true)
if [ "${FORCE:-0}" != 1 ] && [ "${LEFT:-0}" -gt 0 ]; then
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

ARGS=(--served-model-name "${SERVED_NAME:-Qwen3.8}" --host 0.0.0.0 --port "${PORT:-8080}"
      --tensor-parallel-size "${TP:-4}" --max-model-len "${MAXLEN:-32768}"
      --max-num-seqs "$NSEQ" --max-num-batched-tokens "$NBT"
      --gpu-memory-utilization "${UTIL:-$([ "${TP:-4}" = 4 ] && echo 0.97 || echo 0.94)}"
      --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice
      --language-model-only)
[ "${OFFLOAD_GB:-0}" != 0 ] && ARGS+=(--cpu-offload-gb "$OFFLOAD_GB" --cpu-offload-params experts)
[ -n "${KVDTYPE:-}" ] && ARGS+=(--kv-cache-dtype "$KVDTYPE")
[ "${PREFIX_CACHE:-0}" = 0 ] && ARGS+=(--no-enable-prefix-caching)
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
  WPORT=${WEBUI_PORT:-$(( ${PORT:-8080} + 1 ))}
  META=${WEBUI_META:-$HOME/r9700-web-meta-$WPORT.json}
  D="$(dirname "$(realpath "$0")")"
  VER=$($VENV/bin/python -c "import importlib.metadata as m; print('native vllm '+m.version('vllm'))" 2>/dev/null \
        || echo "native")
  META=$META bash "$D/webmeta.sh" venv="$VENV" model="$MODEL" served_name="${SERVED_NAME:-Qwen3.8}" \
    gpus="${GPUS:-0,1,2,3}" tp="${TP:-4}" maxlen="${MAXLEN:-32768}" nseq="$NSEQ" nbt="$NBT" \
    util="${UTIL:-$([ "${TP:-4}" = 4 ] && echo 0.97 || echo 0.94)}" offload_gb="${OFFLOAD_GB:-0}" \
    kvdtype="${KVDTYPE:-}" mtp="$MTP" attn="${ATTN:-}" cg_sizes="${CGSIZES:-}" prefix_cache="${PREFIX_CACHE:-0}" \
    eager="${EAGER:-0}" log="$LOG" fold="${R9K_FOLD:-1}" ar_quant="${R9K_AR_QUANT:-1}" hc_fp8="${R9K_HC_FP8:-r9k}" \
    hc_fp8_decode="${R9K_HC_FP8_DECODE:-1}" fp8_block="${R9K_FP8_BLOCK:-block}" fp8_qa="${R9K_FP8_QA:-1}" \
    target_lmhead="${R9K_TARGET_LMHEAD:-mxfp4}" draft_lmhead="${R9K_DRAFT_LMHEAD:-mxfp4}" \
    mtp_quant_rows="${R9K_MTP_QUANT_ROWS:-}" extra="${EXTRA:-}"
  if [ "${WEBUI:-}" = same ]; then
    D="$(dirname "$(realpath "$0")")"
    META=${WEBUI_META:-$HOME/r9700-web-meta-${PORT:-8080}.json}
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
      echo "Monitor and About pages on http://127.0.0.1:${PORT:-8080}/monitor (inside the serve itself)"
      echo "  VLLM_PLUGINS=$VLLM_PLUGINS"
    fi
  else
    WEBUI_PORT=$WPORT UPSTREAM=http://127.0.0.1:${PORT:-8080} META=$META VERSION="$VER" bash "$D/webui.sh"
  fi
fi
[ "${DRY:-0}" = 1 ] && { echo "launch: $VENV/bin/vllm serve $MODEL ${ARGS[*]} $EXTRA"; exit 0; }
echo "launch: $VENV/bin/vllm serve $MODEL ${ARGS[*]} $EXTRA  (log: $LOG)"
exec "$VENV/bin/vllm" serve "$MODEL" "${ARGS[@]}" $EXTRA
