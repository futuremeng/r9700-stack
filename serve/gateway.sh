#!/bin/bash
# serve/gateway.sh - the one port in front of the engine and the CPU sidecars (serve/gateway.py).
#
# What it is for: one address for clients and for the Monitor page. Path routing, so a client keeps using
# /v1/chat/completions, /v1/embeddings and /v1/rerank on the SAME port and never learns which process answers.
# It is a byte pipe (see serve/gateway.py for what it does NOT do): no body parsing, no buffering, SSE and
# keep-alive untouched.
#
#   bash serve/gateway.sh                  # start it (port GATEWAY_PORT, default 8080)
#   STOP=1 bash serve/gateway.sh           # stop it
#   DRY=1  bash serve/gateway.sh           # print what would start, start nothing
#   FORE=1 bash serve/gateway.sh           # foreground
#
# It assumes the ENGINE sits on a loopback port BELOW this one (this box: 127.0.0.1:8180) -- if the engine
# itself holds 8080, starting this one on 8080 just fails to bind, and this script says so rather than
# reporting the engine's own /health as success.
#
# Knobs: GATEWAY_PORT, GATEWAY_BIND, ENGINE, EMBED, RERANK, PAGES, SIDECAR_STATE, PYTHON, LOG, LOCAL_ENV.

GATEWAY_PORT=${GATEWAY_PORT:-8080}
LOCAL_ENV=${LOCAL_ENV:-$HOME/.config/r9700-stack/native.env}
[ -f "$LOCAL_ENV" ] && { set -a; . "$LOCAL_ENV"; set +a; echo "local env: $LOCAL_ENV"; }
BIND=${GATEWAY_BIND:-${R9K_WEB_BIND:-127.0.0.1}}
ENGINE_PORT=${ENGINE_PORT:-8180}
EMBED_PORT=${EMBED_PORT:-8090}
RERANK_PORT=${RERANK_PORT:-8092}
WATCH_PORT=${WATCH_PORT:-8091}
RERANK_WATCH_PORT=${RERANK_WATCH_PORT:-8093}
ENGINE=${ENGINE:-http://127.0.0.1:$ENGINE_PORT}
EMBED=${EMBED:-http://127.0.0.1:$EMBED_PORT}
RERANK=${RERANK:-http://127.0.0.1:$RERANK_PORT}
PAGES=${PAGES:-$ENGINE}                     # or the webui sidecar (http://127.0.0.1:8081) for a page that
                                           # survives the engine dying -- see notes/launch-methodology.md §5
# name=url pairs: the Monitor's card labels itself from the name (serve/gateway.py splits on the first =
# before ://, so the URLs may carry query strings)
SIDECAR_STATE=${SIDECAR_STATE:-embed=http://127.0.0.1:$WATCH_PORT/state.json,rerank=http://127.0.0.1:$RERANK_WATCH_PORT/state.json}
PYTHON=${PYTHON:-$HOME/r9700-stack/.venv/bin/python}
[ -x "$PYTHON" ] || PYTHON=$(command -v python3)
LOG=${LOG:-$HOME/logs/gateway-$GATEWAY_PORT.log}; mkdir -p "$(dirname "$LOG")"
PIDF=${PIDF:-$HOME/.r9700-state/gateway-$GATEWAY_PORT.pid}; mkdir -p "$(dirname "$PIDF")"

if [ "${STOP:-0}" = 1 ]; then
  if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
    kill "$(cat "$PIDF")"
    for i in $(seq 1 10); do kill -0 "$(cat "$PIDF")" 2>/dev/null || break; sleep 1; done
    echo "stopped the gateway on :$GATEWAY_PORT"
  else
    echo "nothing running on :$GATEWAY_PORT (no live pid in $PIDF)"
  fi
  rm -f "$PIDF"
  exit 0
fi

ARGS=()
[ "${DRY:-0}" = 1 ] && { echo "launch: GATEWAY_PORT=$GATEWAY_PORT GATEWAY_BIND=$BIND ENGINE=$ENGINE EMBED=$EMBED RERANK=$RERANK PAGES=$PAGES $PYTHON $HOME/r9700-stack/serve/gateway.py  (log: $LOG)"; exit 0; }

# bind first, then health: an engine that still owns this port answers /health on it, so a /health-only check
# would congratulate a process that never bound (the exact mistake serve/embed.sh learned on 2026-11-15).
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":$GATEWAY_PORT\$"; then
  echo "something already listens on :$GATEWAY_PORT -- the engine cannot be moved off it by this script."
  echo "  stop the launch (STOP=1 bash serve/native-flashnext.sh), then start this, then relaunch the engine"
  echo "  on ENGINE_PORT=$ENGINE_PORT."
  exit 1
fi

echo "launch: GATEWAY_PORT=$GATEWAY_PORT GATEWAY_BIND=$BIND ENGINE=$ENGINE EMBED=$EMBED RERANK=$RERANK PAGES=$PAGES $PYTHON serve/gateway.py  (log: $LOG)"
if [ "${FORE:-0}" = 1 ]; then
  exec env GATEWAY_PORT="$GATEWAY_PORT" GATEWAY_BIND="$BIND" ENGINE="$ENGINE" EMBED="$EMBED" RERANK="$RERANK" \
      PAGES="$PAGES" SIDECAR_STATE="$SIDECAR_STATE" "$PYTHON" "$HOME/r9700-stack/serve/gateway.py"
fi
nohup env GATEWAY_PORT="$GATEWAY_PORT" GATEWAY_BIND="$BIND" ENGINE="$ENGINE" EMBED="$EMBED" RERANK="$RERANK" \
    PAGES="$PAGES" SIDECAR_STATE="$SIDECAR_STATE" "$PYTHON" "$HOME/r9700-stack/serve/gateway.py" >>"$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PIDF"
for i in $(seq 1 15); do
  kill -0 $PID 2>/dev/null || { echo "died during startup, see $LOG" >&2; tail -5 "$LOG" >&2; rm -f "$PIDF"; exit 1; }
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$GATEWAY_PORT/health" 2>/dev/null) || true
  [ "$code" = 200 ] && break
  sleep 1
done
[ "$code" = 200 ] || { echo "not healthy after 15 s (the engine behind it may not be up yet): $LOG" >&2; exit 1; }
echo "gateway up: pid $PID on http://$BIND:$GATEWAY_PORT  (engine $ENGINE, embed $EMBED, rerank $RERANK, pages $PAGES)"
echo "  clients:  http://<box>:$GATEWAY_PORT/v1/chat/completions | /v1/embeddings | /v1/rerank"
echo "  monitor:  http://<box>:$GATEWAY_PORT/monitor"
