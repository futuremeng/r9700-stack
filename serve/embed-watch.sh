#!/bin/bash
# serve/embed-watch.sh - start (or stop) the embedding sidecar's Monitor: serve/embed-watch.py on WPORT.
# Observe only, like webui.sh's default mode: it GETs :8090/health and :8090/metrics every POLL_S and reads
# /proc -- nothing is written, nothing sits in the inference path. The page's first card is the memory
# pressure the GGUF's mmap must not add (notes/load-oom-svm-wedge.md); the second is llama.cpp's own
# /metrics deltas; the third is embed.sh's recipe, read from its ${VAR:=default} lines.
#
#   bash serve/embed-watch.sh                  # watch :8090 from :8091
#   STOP=1 bash serve/embed-watch.sh           # stop it
#   DRY=1  bash serve/embed-watch.sh           # print what would start, start nothing
#   FORE=1 bash serve/embed-watch.sh           # foreground
# Knobs: EMBED_PORT (the llama-server port, 8090), WATCH_PORT (8091), WATCH_BIND (loopback default, same
#        chain as webui.sh: flag > LOCAL_ENV's EMBED_WATCH_BIND > repo default), POLL_S, KEY, LOCAL_ENV, LOG.
EMBED_PORT=${EMBED_PORT:-8090}
WPORT=${WATCH_PORT:-8091}
LOCAL_ENV=${LOCAL_ENV:-$HOME/.config/r9700-stack/native.env}
if [ -f "$LOCAL_ENV" ]; then set -a; . "$LOCAL_ENV"; set +a; echo "local env: $LOCAL_ENV"; fi
BIND=${WATCH_BIND:-${EMBED_WATCH_BIND:-127.0.0.1}}
LOG=${WATCH_LOG:-$HOME/logs/embed-watch-$WPORT.log}; mkdir -p "$(dirname "$LOG")"
DIR=$(dirname "$(realpath "$0")")
PAT="embed-watch.py --port $WPORT"

if [ "${STOP:-0}" = 1 ]; then
  pkill -f "$PAT" && echo "stopped the embedding monitor on $WPORT" || echo "nothing running on $WPORT"
  exit 0
fi

ARGS=(--port "$WPORT" --bind "$BIND" --embed "http://127.0.0.1:$EMBED_PORT")
[ -n "$POLL_S" ] && ARGS+=(--poll "$POLL_S")
[ -n "$KEY" ]    && ARGS+=(--key "$KEY")
ARGS+=(--launcher "$DIR/embed.sh")

if [ "${DRY:-0}" = 1 ]; then echo "launch: python3 $DIR/embed-watch.py ${ARGS[*]}  (log: $LOG)"; exit 0; fi
echo "launch: python3 $DIR/embed-watch.py ${ARGS[*]}  (log: $LOG)"
[ "${FORE:-0}" = 1 ] && exec python3 "$DIR/embed-watch.py" "${ARGS[@]}"
nohup python3 "$DIR/embed-watch.py" "${ARGS[@]}" >>"$LOG" 2>&1 &
PID=$!
for i in $(seq 1 10); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$WPORT/health" 2>/dev/null) || true
  [ "$code" = 200 ] && break
  kill -0 $PID 2>/dev/null || { echo "died during startup, see $LOG" >&2; tail -5 "$LOG" >&2; exit 1; }
  sleep 1
done
[ "$code" = 200 ] || { echo "not healthy after 10 s, see $LOG" >&2; exit 1; }
echo "embedding monitor up: pid $PID on http://$BIND:$WPORT/  (observing :$EMBED_PORT; STOP=1 to stop)"
