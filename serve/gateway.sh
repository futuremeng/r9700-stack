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
# Knobs: GATEWAY_PORT, GATEWAY_BIND, ENGINE, EMBED, RERANK, PAGES, SIDECAR_STATE, PYTHON, LOG, LOCAL_ENV,
# and the admission gate's GATE_* set (below) -- passed through only when set, because the gate's defaults
# live in serve/gateway.py and two copies of a default is how a launcher starts lying about the running one.

# The machine-local file FIRST: it is where this box says where things live, and it stays a DEFAULT --
# what the caller exported wins over it (serve/lib/localenv.sh, and the same order setup.py reads).
# Sourcing it below a `${NAME:-default}` line makes the script's own default an override: gateway.sh's
# GATEWAY_PORT and embed.sh's LLAMA_SERVER were both frozen before the file ever got a vote.
. "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/lib/localenv.sh"
GATEWAY_PORT=${GATEWAY_PORT:-8080}
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
# name=url pairs. The name IS the route the serve answers on (/v1/embeddings=..., /v1/rerank=...), so the
# gateway is not asked to keep a second copy of its own routing table: the Monitor card and the About page
# label themselves from it. A bare name (embed=...) still works -- it just cannot be read as a route.
SIDECAR_STATE=${SIDECAR_STATE:-/v1/embeddings=http://127.0.0.1:$WATCH_PORT/state.json,/v1/rerank=http://127.0.0.1:$RERANK_WATCH_PORT/state.json}
PYTHON=${PYTHON:-$HOME/r9700-stack/.venv/bin/python}
[ -x "$PYTHON" ] || PYTHON=$(command -v python3)
LOG=${LOG:-$HOME/logs/gateway-$GATEWAY_PORT.log}; mkdir -p "$(dirname "$LOG")"
PIDF=${PIDF:-$HOME/.r9700-state/gateway-$GATEWAY_PORT.pid}; mkdir -p "$(dirname "$PIDF")"

# The gate (serve/gateway.py): since 2026-11-16 it OBSERVES by default. It reads the engine's harm signals -- its
# preemption counter moving, or its KV pool at GATE_KV_HIGH -- and reports them on the Monitor and at /api/gate,
# counting the requests it WOULD have held (arrived_shut, split by declared size class), but it holds, refuses and
# delays nothing. That change was bought with the numbers in notes/admission-gate.md §7: the door could not
# re-open (a preemption every 20 s re-arms the window, and a pool whose steady state sits above GATE_KV_LOW latches
# the pool axis), a 5 s hold cannot free one block that requests already running are holding, and the traffic that
# filled the pool that day was the traffic the small-request escape waved through. GATE_POLICY=hold re-arms the
# intervention; the defaults below are still the thresholds it would use, and they stay measured, not guessed.
# Defaults are in gateway.py; these are the overrides, and what is live gets printed rather than implied. A knob
# missing from GATENAMES is a knob the tmux launch cannot set, so tests/test_setup.py holds this list against the
# names gateway.py actually reads.
GATENAMES="GATE_DISABLE GATE_POLICY GATE_KV_HIGH GATE_KV_LOW GATE_PREEMPT_S GATE_WAIT GATE_MAX_INFLIGHT GATE_SMALL_BYTES GATE_MAX_HELD GATE_HOLD_S GATE_METRICS_S GATE_STALE_S GATE_HIST GATE_POOL_TOKENS"
GATEENV=(); GATEDESC=""
for n in $GATENAMES; do
  v=${!n:-}
  [ -n "$v" ] && { GATEENV+=("$n=$v"); GATEDESC="$GATEDESC $n=$v"; }
done

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
[ "${DRY:-0}" = 1 ] && { echo "launch: GATEWAY_PORT=$GATEWAY_PORT GATEWAY_BIND=$BIND ENGINE=$ENGINE EMBED=$EMBED RERANK=$RERANK PAGES=$PAGES${GATEDESC:+ gate:$GATEDESC} $PYTHON $HOME/r9700-stack/serve/gateway.py  (log: $LOG)"; exit 0; }

# bind first, then health: an engine that still owns this port answers /health on it, so a /health-only check
# would congratulate a process that never bound (the exact mistake serve/embed.sh learned on 2026-11-15).
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":$GATEWAY_PORT\$"; then
  echo "something already listens on :$GATEWAY_PORT -- the engine cannot be moved off it by this script."
  echo "  stop the launch (STOP=1 bash serve/native-flashnext.sh), then start this, then relaunch the engine"
  echo "  on ENGINE_PORT=$ENGINE_PORT."
  exit 1
fi

echo "launch: GATEWAY_PORT=$GATEWAY_PORT GATEWAY_BIND=$BIND ENGINE=$ENGINE EMBED=$EMBED RERANK=$RERANK PAGES=$PAGES${GATEDESC:+ gate:$GATEDESC} $PYTHON serve/gateway.py  (log: $LOG)"
if [ "${FORE:-0}" = 1 ]; then
  exec env GATEWAY_PORT="$GATEWAY_PORT" GATEWAY_BIND="$BIND" ENGINE="$ENGINE" EMBED="$EMBED" RERANK="$RERANK" \
      PAGES="$PAGES" SIDECAR_STATE="$SIDECAR_STATE" ${GATEENV[@]+"${GATEENV[@]}"} \
      "$PYTHON" "$HOME/r9700-stack/serve/gateway.py"
fi
nohup env GATEWAY_PORT="$GATEWAY_PORT" GATEWAY_BIND="$BIND" ENGINE="$ENGINE" EMBED="$EMBED" RERANK="$RERANK" \
    PAGES="$PAGES" SIDECAR_STATE="$SIDECAR_STATE" ${GATEENV[@]+"${GATEENV[@]}"} \
    "$PYTHON" "$HOME/r9700-stack/serve/gateway.py" >>"$LOG" 2>&1 &
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
# The page SOURCE is a choice, and a choice can point at a process that is not running. PAGES moved to a standalone
# sidecar on 2026-11-15 so the Monitor would outlive the engine, but nothing here guarantees the sidecar is up:
# after a reboot, or on a box where nobody starts it, every /monitor through this door is a failure. Say that, and
# name both ways out. What must NOT happen is quietly pointing the pages back at the engine -- that reinstates the
# exact state in which a wedged serve takes its own report down with it (notes/supervision.md §4).
if ! curl -fsS -m 2 "$PAGES/health" >/dev/null 2>&1; then
  echo "  WARN  the page source $PAGES is not answering: /monitor and /about through this door will fail." >&2
  echo "        start it:  bash serve/webui.sh" >&2
  echo "        or serve pages from the engine again: PAGES=$ENGINE in ${LOCAL_ENV:-$HOME/.config/r9700-stack/native.env}" >&2
fi
echo "  gate:     http://127.0.0.1:$GATEWAY_PORT/api/gate  -- the harm signals, what the door would have done, and what it is set to"
echo "  clients:  http://<box>:$GATEWAY_PORT/v1/chat/completions | /v1/embeddings | /v1/rerank"
echo "  monitor:  http://<box>:$GATEWAY_PORT/monitor"
