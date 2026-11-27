#!/bin/bash
# serve/percept-watch.sh - start (or stop) the perception layer's Monitor: serve/percept-watch.py on WPORT.
# Observe only, like embed-watch.sh: the perception layer is CLI-shaped, so instead of polling a port this
# watcher reads the run journal serve/percept.sh appends and reports the same live inventory the launcher's
# guards check (binary, models, core range, MemAvailable vs MEM_FLOOR) -- nothing written, nothing in any
# diarization's path.
#
#   bash serve/percept-watch.sh                  # the page on :8094
#   STOP=1 bash serve/percept-watch.sh           # stop it
#   DRY=1  bash serve/percept-watch.sh           # print what would start, start nothing
#   FORE=1 bash serve/percept-watch.sh           # foreground
# Knobs: WATCH_PORT (8094), WATCH_BIND (loopback default, same chain as embed-watch.sh), KEY, JOURNAL,
#        LOCAL_ENV, LOG.
# The machine-local file FIRST: it stays a DEFAULT -- what the caller exported wins (serve/lib/localenv.sh).
. "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/lib/localenv.sh"
WPORT=${WATCH_PORT:-8094}
LOCAL_ENV=${LOCAL_ENV:-$HOME/.config/r9700-stack/native.env}
BIND=${WATCH_BIND:-${PERCEPT_WATCH_BIND:-127.0.0.1}}
LOG=${WATCH_LOG:-$HOME/logs/percept-watch-$WPORT.log}; mkdir -p "$(dirname "$LOG")"
DIR=$(dirname "$(realpath "$0")")
PAT="percept-watch.py --port $WPORT"

if [ "${STOP:-0}" = 1 ]; then
  pkill -f "$PAT" && echo "stopped the perception monitor on $WPORT" || echo "nothing running on $WPORT"
  exit 0
fi

ARGS=(--port "$WPORT" --bind "$BIND" --launcher "$DIR/percept.sh")
# the native.env paths the launcher would resolve at run time -- pass them through so the inventory card
# reports what a real run would find, not just what the regex can see in the script.
[ -n "$SHERPA_BIN" ] && ARGS+=(--sherpa-bin "$SHERPA_BIN")
[ -n "$SEG_MODEL" ]  && ARGS+=(--seg-model "$SEG_MODEL")
[ -n "$SPK_MODEL" ]  && ARGS+=(--spk-model "$SPK_MODEL")
[ -n "$JOURNAL" ] && ARGS+=(--journal "$JOURNAL")
[ -n "$KEY" ]     && ARGS+=(--key "$KEY")

if [ "${DRY:-0}" = 1 ]; then echo "launch: python3 $DIR/percept-watch.py ${ARGS[*]}  (log: $LOG)"; exit 0; fi
echo "launch: python3 $DIR/percept-watch.py ${ARGS[*]}  (log: $LOG)"
[ "${FORE:-0}" = 1 ] && exec python3 "$DIR/percept-watch.py" "${ARGS[@]}"
nohup python3 "$DIR/percept-watch.py" "${ARGS[@]}" >>"$LOG" 2>&1 &
PID=$!
for i in $(seq 1 10); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$WPORT/health" 2>/dev/null) || true
  [ "$code" = 200 ] && break
  kill -0 $PID 2>/dev/null || { echo "died during startup, see $LOG" >&2; tail -5 "$LOG" >&2; exit 1; }
  sleep 1
done
[ "$code" = 200 ] || { echo "not healthy after 10 s, see $LOG" >&2; exit 1; }
echo "perception monitor up: pid $PID on http://$BIND:$WPORT/  (journal-driven; STOP=1 to stop)"
