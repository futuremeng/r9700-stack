#!/bin/bash
# serve/embed.sh - the CPU embedding sidecar: llama-server + Qwen3-Embedding-0.6B q8_0, mmap only.
#
# Why this shape (the reasoning lives in the box's own notes):
# * The 4 R9700 cards carry ~1.35 GiB of spare VRAM at util 0.96 (README "VRAM per card"), so the
#   embedding model runs on the EPYC. ~11 of 192 cores are busy with the engine
#   (notes/native-4xr9700.md); the rest are free, and CORES/THREADS keep this off their paths.
# * mmap and NOTHING else: the GGUF pages are page cache -- reclaimable, never swap, never anon.
#   An extra 4 GiB of anon commit is the exact recipe of the 2026-11-01 SVM wedge
#   (notes/load-oom-svm-wedge.md: Committed_AS 122.6 against 93.8 GiB, kswapd blocked on the amdgpu
#   notifier, box unrecoverable). So: never --mlock, never --no-mmap, and stay at 0.6B here --
#   the 4B q8_0 is still mmap-safe but 6x slower; do that only if a golden set says 0.6B falls short.
# * Qwen3-Embedding is instruction-tuned: the DOCUMENT side is plain text, the QUERY side carries
#   the Instruct/Query prompt -- in the request (see the curl below) or baked in with CHAT_TEMPLATE.
#
#   bash serve/embed.sh                    # start on 8090, download the GGUF if missing
#   STOP=1 bash serve/embed.sh             # stop it
#   DRY=1  bash serve/embed.sh             # print what would start, start and write nothing
#   FORE=1 bash serve/embed.sh             # foreground (no nohup), for watching a first bring-up
# Monitor: serve/embed-watch.sh (its own page on :8091 -- webui.py is vLLM-shaped and cannot read
# llama.cpp's /metrics; --metrics is on in the ARGS below for exactly that page).
# Knobs: MODEL, HF_REPO, HF_FILE, LLAMA_SERVER, PORT, EMBED_BIND, CORES (taskset range), THREADS,
#        CTX, BATCH, SERVED_NAME, CHAT_TEMPLATE, LOG, LOCAL_ENV (default the same
#        ~/.config/r9700-stack/native.env the other launchers read; only EMBED_BIND is honoured here).

MODEL=${MODEL:-$HOME/models/qwen3-embedding-0.6b-q8_0.gguf}
HF_REPO=${HF_REPO:-unsloth/Qwen3-Embedding-0.6B-GGUF}
HF_FILE=${HF_FILE:-qwen3-embedding-0.6b-q8_0.gguf}
LLAMA_SERVER=${LLAMA_SERVER:-$(command -v llama-server)}
PORT=${PORT:-8090}
LOCAL_ENV=${LOCAL_ENV:-$HOME/.config/r9700-stack/native.env}
[ -f "$LOCAL_ENV" ] && { set -a; . "$LOCAL_ENV"; set +a; echo "local env: $LOCAL_ENV"; }
BIND=${EMBED_BIND:-${R9K_WEB_BIND:-127.0.0.1}}
CORES=${CORES:-16-31}                      # taskset range: stay off the engine's ~11 cores
THREADS=${THREADS:-8}
CTX=${CTX:-8192}
BATCH=${BATCH:-512}
LOG=${LOG:-$HOME/logs/embed-$PORT.log}; mkdir -p "$(dirname "$LOG")"
PAT="llama-server.*--port $PORT"

if [ "${STOP:-0}" = 1 ]; then
  pkill -f "$PAT" && echo "stopped the embedding server on $PORT" || echo "nothing running on $PORT"
  exit 0
fi
[ -z "$LLAMA_SERVER" ] && { echo "llama-server not found. On this box (no root): git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp && cmake -B ~/llama.cpp/build -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release && cmake --build ~/llama.cpp/build -j16 --config Release, then LLAMA_SERVER=~/llama.cpp/build/bin/llama-server bash serve/embed.sh" >&2; exit 1; }

# the GGUF by mmap path: one file, page cache, zero setup-time anon. DRY writes nothing -- the same
# rule native-flashnext.sh learned the hard way on 2026-11-04 (a DRY run must not touch the state of
# the thing that is actually running).
if [ ! -f "$MODEL" ]; then
  if [ "${DRY:-0}" = 1 ]; then echo "  (DRY: the file is missing and would be downloaded to $MODEL)"
  else
    echo "fetching $HF_REPO -> $MODEL"
    mkdir -p "$(dirname "$MODEL")"
    PIP=$(command -v pip3 || command -v pip) || { echo "need pip3 once for the download: pip3 install -U 'huggingface_hub[cli]', or fetch the GGUF by hand" >&2; exit 1; }
    "$PIP" install -q -U "huggingface_hub[cli]" \
      && "$HOME/.local/bin/hf" download "$HF_REPO" "$HF_FILE" --local-dir "$(dirname "$MODEL")" --local-dir-use-symlinks False \
      && mv "$(dirname "$MODEL")/$HF_FILE" "$MODEL" \
      || { echo "download failed -- fetch $HF_REPO/$HF_FILE by hand to $MODEL" >&2; exit 1; }
  fi
fi

ARGS=(-m "$MODEL" --embedding --pooling last --metrics --host "$BIND" --port "$PORT"
      -t "$THREADS" -c "$CTX" -b "$BATCH" -ngl 0)
# the name /v1/models shows and what clients put in "model"; llama.cpp answers any value either way.
[ -n "$SERVED_NAME" ] && ARGS+=(--alias "$SERVED_NAME")
[ -n "$CHAT_TEMPLATE" ] && ARGS+=(--chat-template "$CHAT_TEMPLATE")
# mmap is the whole memory argument of this launcher: no mlock, no no-mmap anywhere. If LLAMA_EXTRA
# ever says --mlock, this script is no longer the incident-safe one.
[ -n "$LLAMA_EXTRA" ] && ARGS+=($LLAMA_EXTRA)
if [ -n "$CORES" ] && command -v taskset >/dev/null; then
  ARGS=(taskset -c "$CORES" "$LLAMA_SERVER" "${ARGS[@]}")
else
  ARGS=("$LLAMA_SERVER" "${ARGS[@]}")
fi

if [ "${DRY:-0}" = 1 ]; then echo "launch: ${ARGS[*]}  (log: $LOG)"; exit 0; fi
echo "launch: ${ARGS[*]}  (log: $LOG)"
if [ "${FORE:-0}" = 1 ]; then exec "${ARGS[@]}"; fi
nohup "${ARGS[@]}" >>"$LOG" 2>&1 &
PID=$!

# wait for the model to map and the tokenizer to come up -- /health says 200 only when it answers
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null) || true
  [ "$code" = 200 ] && break
  kill -0 $PID 2>/dev/null || { echo "died during startup, see $LOG" >&2; tail -5 "$LOG" >&2; exit 1; }
  sleep 1
done
[ "$code" = 200 ] || { echo "not healthy after 60 s, see $LOG" >&2; exit 1; }

echo "embedding up: pid $PID on http://$BIND:$PORT (threads $THREADS on cores $CORES, ctx $CTX, model $(du -h "$MODEL" | cut -f1) mmap)"
echo "  docs : curl http://$BIND:$PORT/v1/embeddings -H 'Content-Type: application/json' -d '{\"model\":\"qwen3-embed\",\"input\":[\"文档文本\"]}'"
echo "  query: the instruct goes in the input, e.g. input=[[{\"role\":\"user\",\"content\":\"Instruct: Retrieve passages that answer the question\\nQuery: ...\"}]]"
echo "  rerank variant: a Qwen3-Reranker-0.6B q8_0 GGUF with --rerank instead of --embedding (LLAMA_EXTRA, same launcher)"

