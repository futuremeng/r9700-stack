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

ARGS=(--served-model-name "${SERVED_NAME:-Qwen3.8}" --host 0.0.0.0 --port "${PORT:-8080}"
      --tensor-parallel-size "${TP:-4}" --max-model-len "${MAXLEN:-32768}"
      --max-num-seqs "${NSEQ:-32}" --max-num-batched-tokens "${NBT:-8192}"   # TP4 on this box, 2026-10-31: 4096 -> 8192
      # is +5.4% prefill at 7.8k (9,331 -> 9,831) and +3.5% at 31k (8,909 -> 9,220), +4.4% aggregate at conc 16
      # (1,163 -> 1,214), decode and conc-8 unchanged; it costs 64k KV tokens (563,692 -> 499,815, still 15.2x
      # at 32k). serve/flashnext.sh keeps 4096: at TP2 with offloaded experts that chunk has no KV room.
      # NSEQ 16 -> 32: +18% aggregate decode at 32 in flight (1,207/1,226 -> 1,429/1,437), single-stream and
      # long-context prefill untouched (9,376/9,311 -> 9,323/9,298 tok/s at 16/24 x ~20k prompts), KV -1.2k
      # tokens. 48 in flight is worse (1,366), so 32 is the knee here, not a limit: the KV pool still admits
      # only 15.2 requests at the full 32k, and vLLM queues the rest (prefill throughput does not change).
      --gpu-memory-utilization "${UTIL:-$([ "${TP:-4}" = 4 ] && echo 0.97 || echo 0.94)}"
      --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice
      --language-model-only)
[ "${OFFLOAD_GB:-0}" != 0 ] && ARGS+=(--cpu-offload-gb "$OFFLOAD_GB" --cpu-offload-params experts)
[ -n "${KVDTYPE:-}" ] && ARGS+=(--kv-cache-dtype "$KVDTYPE")
[ "${PREFIX_CACHE:-0}" = 0 ] && ARGS+=(--no-enable-prefix-caching)
[ "${EAGER:-0}" = 1 ] && ARGS+=(--enforce-eager)
MTP=${MTP:-3}
[ -n "$MTP" ] && ARGS+=(--speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": $MTP${SPEC_EXTRA:+, $SPEC_EXTRA}}")
[ -n "${ATTN:-}" ] && ARGS+=(--attention-backend "$ATTN")
CC=()
: ${CGSIZES:=1,2,4,8,16,24,32,48,64,96,128,192,256}   # graphs for prefill chunks <= 256 (serve/flashnext.sh)
[ -n "${CGSIZES:-}" ] && CC+=("\"cudagraph_capture_sizes\": [$CGSIZES], \"max_cudagraph_capture_size\": ${CGSIZES##*,}")
[ ${#CC[@]} -gt 0 ] && ARGS+=(--compilation-config "{$(IFS=,; echo "${CC[*]}")}")

echo "launch: $VENV/bin/vllm serve $MODEL ${ARGS[*]} $EXTRA  (log: $LOG)"
exec "$VENV/bin/vllm" serve "$MODEL" "${ARGS[@]}" $EXTRA
