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
# The official Qwen GGUF, not a third-party repack: unsloth/Qwen3-Embedding-0.6B-GGUF, the name the first
# version of this launcher carried, answers 401 from an unauthenticated hub API -- on HF it is gated, so the
# default would have died at the download step on any box without a token. Qwen/Qwen3-Embedding-0.6B-GGUF is
# public (q8_0 639,150,592 B, fetched on this box 2026-11-15).
HF_REPO=${HF_REPO:-Qwen/Qwen3-Embedding-0.6B-GGUF}
HF_FILE=${HF_FILE:-Qwen3-Embedding-0.6B-Q8_0.gguf}
LLAMA_SERVER=${LLAMA_SERVER:-$(command -v llama-server)}
PORT=${PORT:-8090}
LOCAL_ENV=${LOCAL_ENV:-$HOME/.config/r9700-stack/native.env}
[ -f "$LOCAL_ENV" ] && { set -a; . "$LOCAL_ENV"; set +a; echo "local env: $LOCAL_ENV"; }
BIND=${EMBED_BIND:-${R9K_WEB_BIND:-127.0.0.1}}
CORES=${CORES:-16-31}                      # taskset range: stay off the engine's ~11 cores. On THIS box cpu 17
                                          # and 113 are offline (host/offline-core.cpus, notes/incident-2026-11-03)
                                          # -- sched_setaffinity drops them silently, so the range runs 15 cores.
cpus_in() {                                # "16-31" or "16-17,20" -> one cpu number per line
  local r lo hi
  for r in $(printf '%s' "$1" | tr ',' ' '); do
    case "$r" in *-*) lo=${r%%-*}; hi=${r##*-};; *) lo=$r; hi=$r;; esac
    seq "$lo" "$hi" 2>/dev/null
  done
}
# THREADS defaults to the number of ONLINE cpus inside CORES, and that is the whole point of the count: llama.cpp's
# workers spin, so one thread more than there are cores costs a third of the throughput. Measured on this box
# 2026-11-15, cores 16-31 (cpu 17 offline -> 15 online), a saturated 0.6B q8_0 indexing run:
#   -t 15 -> 1,112 tok/s   -t 16 -> 748 tok/s   (-t 8 -> 689, -t 31 over cores 16-63 -> 2,052)
# The engine is unpinned over 190 cores and its ITL and throughput did not move under any of these.
ONLINE_LIST=$(cpus_in "$(cat /sys/devices/system/cpu/online 2>/dev/null || echo 0-$(( $(nproc) - 1 )))")
N_ONLINE=$(cpus_in "$CORES" | grep -Fx -f <(printf '%s\n' "$ONLINE_LIST") 2>/dev/null | wc -l)
THREADS=${THREADS:-${N_ONLINE:-8}}
CTX=${CTX:-8192}
BATCH=${BATCH:-512}
LOG=${LOG:-$HOME/logs/embed-$PORT.log}; mkdir -p "$(dirname "$LOG")"
PAT="llama-server.*--port $PORT"

if [ "${STOP:-0}" = 1 ]; then
  # pkill returns as soon as the signal is sent; llama.cpp takes a moment to close its sockets, and a start run
  # inside that moment used to see the dying serve's /health and print "embedding up" for a pid that never bound
  # the port. Wait for it to be gone, then say so.
  if pkill -f "$PAT"; then
    for i in $(seq 1 15); do pgrep -f "$PAT" >/dev/null 2>&1 || break; sleep 1; done
    echo "stopped the embedding server on $PORT"
  else
    echo "nothing running on $PORT"
  fi
  exit 0
fi
[ -z "$LLAMA_SERVER" ] && { echo "llama-server not found. On this box (no root): git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp && cmake -S ~/llama.cpp -B ~/llama.cpp/build -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release && cmake --build ~/llama.cpp/build -j16, then LLAMA_SERVER=~/llama.cpp/build/bin/llama-server bash serve/embed.sh (no cmake on the box and no sudo? pip install cmake into a venv -- that is what this box did, .venv/bin/cmake). Or let setup.py do it: ./setup.sh --embedding" >&2; exit 1; }

# the GGUF by mmap path: one file, page cache, no anon for the weights. DRY writes nothing -- the same
# rule native-flashnext.sh learned the hard way on 2026-11-04 (a DRY run must not touch the state of
# the thing that is actually running).
# Measured on this box 2026-11-15, so the claim is stated at its real size: the GGUF mapping reads 618,368 kB
# resident, ALL of it clean (Private_Clean, 0 dirty, 0 anon, Locked 0) -> reclaimable page cache, never swap.
# What is anon is the KV pool and the scratch: 977,404 kB at -c 8192 (4 slots), and the process takes
# +2.77 GB of Committed_AS (85.15 -> 87.92 GB of 93.8 GiB). STOP gives all of it back.
if [ ! -f "$MODEL" ]; then
  if [ "${DRY:-0}" = 1 ]; then echo "  (DRY: the file is missing and would be downloaded to $MODEL)"
  else
    echo "fetching $HF_REPO -> $MODEL"
    mkdir -p "$(dirname "$MODEL")"
    # use an hf that is already installed; a launcher must not pip-install into someone else's environment.
    # (The first version pip3-installed huggingface_hub[cli] and then reached for ~/.local/bin/hf: on this box
    # pip3 is the system one, externally managed, and hf lives in the venv -- 2026-11-15.)
    HF=$(command -v hf || command -v huggingface-cli || (test -x "$HOME/.local/bin/hf" && echo "$HOME/.local/bin/hf"))
    [ -z "$HF" ] && { echo "no hf CLI on PATH: pip install 'huggingface_hub[cli]' into the venv you already use, or fetch $HF_REPO/$HF_FILE to $MODEL by hand (on a box that reaches HF through a proxy, https_proxy must be set for this shell -- git carries its own http.proxy, curl and hf do not)" >&2; exit 1; }
    "$HF" download "$HF_REPO" "$HF_FILE" --local-dir "$(dirname "$MODEL")" --local-dir-use-symlinks False \
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
if [ "$THREADS" -gt "${N_ONLINE:-$THREADS}" ]; then
  echo "note: THREADS=$THREADS but CORES=$CORES holds only $N_ONLINE online cpus -- llama.cpp's workers spin, so"
  echo "      oversubscribing costs real throughput (measured: 15 cpus, -t 15 = 1,112 tok/s, -t 16 = 748)."
fi
# one serve per port, said out loud: a second launch dies in the bind ("couldn't bind HTTP server socket")
# while the loop below sees the FIRST serve's /health answering 200 and prints "embedding up: pid <dead>" --
# seen on this box 2026-11-15, with rc=0. Refuse instead, the way the GPU launchers refuse a busy card.
ALREADY=$(pgrep -f "$PAT" 2>/dev/null | head -1)
if [ -n "$ALREADY" ]; then
  echo "already running on :$PORT (pid $ALREADY) -- not starting a second one that cannot bind the port"
  echo "  restart it with: STOP=1 bash serve/embed.sh && bash serve/embed.sh"
  exit 0
fi
echo "launch: ${ARGS[*]}  (log: $LOG)"
if [ "${FORE:-0}" = 1 ]; then exec "${ARGS[@]}"; fi
nohup "${ARGS[@]}" >>"$LOG" 2>&1 &
PID=$!

# wait for the model to map and the tokenizer to come up -- /health says 200 only when it answers. Our own death
# is checked FIRST: an older serve on the same port answers 200 no matter what this launch did.
for i in $(seq 1 60); do
  kill -0 $PID 2>/dev/null || { echo "died during startup, see $LOG" >&2; tail -5 "$LOG" >&2; exit 1; }
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null) || true
  [ "$code" = 200 ] && break
  sleep 1
done
[ "$code" = 200 ] || { echo "not healthy after 60 s, see $LOG" >&2; exit 1; }

echo "embedding up: pid $PID on http://$BIND:$PORT (threads $THREADS on cores $CORES, ctx $CTX, model $(du -h "$MODEL" | cut -f1) mmap)"
echo "  docs : curl http://$BIND:$PORT/v1/embeddings -H 'Content-Type: application/json' -d '{\"model\":\"qwen3-embed\",\"input\":[\"文档文本\"]}'"
echo "  query: the instruct goes in the input, e.g. input=[[{\"role\":\"user\",\"content\":\"Instruct: Retrieve passages that answer the question\\nQuery: ...\"}]]"
echo "  rerank variant: a Qwen3-Reranker-0.6B q8_0 GGUF with --rerank instead of --embedding (LLAMA_EXTRA, same launcher)"

