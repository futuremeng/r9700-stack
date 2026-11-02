#!/bin/bash
# serve/webui.sh - start (or stop) the Monitor/About sidecar for a running serve/ launch.
#
# It is a separate host process: it never runs inside the container and, in the default mode, it never
# touches the inference path. Its whole footprint is GET /health, /v1/models and /metrics every
# POLL_S seconds, plus reads of the amdgpu sysfs. Requests appear in the Monitor only when traffic is
# routed through it, which is what WEBUI_MODE=proxy says yes to -- one extra TCP hop, on purpose.
#
#   bash serve/webui.sh                      # watch the serve on PORT (8080) from PORT+1 (8081)
#   PORT=8100 bash serve/webui.sh            # a second launch on 8100 -> monitor on 8101
#   WEBUI_MODE=proxy bash serve/webui.sh     # also relay /v1/* so the request list fills up
#   META=~/launch.json bash serve/webui.sh   # show that recipe on the About page
#   STOP=1 bash serve/webui.sh               # stop the sidecar for this port
# Knobs: PORT (the vLLM port), WEBUI_PORT, UPSTREAM (full URL, wins over both), WEBUI_MODE=observe|proxy,
#        POLL_S, KEEP, GPUS (sysfs root), SMI=1 (fill the sysfs gaps from rocm-smi: it opens /dev/kfd, so
#        leave it off while measuring; SMI="<command>" overrides the default invocation),
#        VERSION (string to show as the engine version), FORE=1 (no nohup).
PORT=${PORT:-8080}
WPORT=${WEBUI_PORT:-$((PORT + 1))}
UP=${UPSTREAM:-http://127.0.0.1:$PORT}
DIR=$(dirname "$(realpath "$0")")
LOG=${WEBUI_LOG:-$HOME/r9700-webui-$WPORT.log}
PAT="webui.py --port $WPORT"

if [ "${STOP:-0}" = 1 ]; then
  pkill -f "$PAT" && echo "stopped the sidecar on $WPORT" || echo "nothing running on $WPORT"
  exit 0
fi

ARGS=(--port "$WPORT" --upstream "$UP")
[ -n "$WEBUI_KEY" ]  && ARGS+=(--key "$WEBUI_KEY")
[ -n "$META" ]       && ARGS+=(--meta "$META")
[ -n "$VERSION" ]    && ARGS+=(--version "$VERSION")
[ "${WEBUI_MODE:-observe}" = proxy ] && ARGS+=(--proxy)
[ -n "$POLL_S" ]     && ARGS+=(--poll "$POLL_S")
[ -n "$KEEP" ]       && ARGS+=(--keep "$KEEP")
[ -n "$GPUS" ]       && ARGS+=(--gpus "$GPUS")
[ -n "$SMI" ]        && ARGS+=(--smi "$SMI")
[ -n "$JOURNAL" ]    && ARGS+=(--journal "$JOURNAL")

pkill -f "$PAT" 2>/dev/null
if [ "${FORE:-0}" = 1 ]; then exec python3 "$DIR/webui.py" "${ARGS[@]}"; fi
nohup python3 "$DIR/webui.py" "${ARGS[@]}" >> "$LOG" 2>&1 &
sleep 1
if pgrep -f "$PAT" > /dev/null; then
  echo "monitor: http://127.0.0.1:$WPORT/monitor   about: http://127.0.0.1:$WPORT/about"
  echo "upstream $UP · mode ${WEBUI_MODE:-observe} · log $LOG · stop: STOP=1 bash serve/webui.sh"
else
  echo "the sidecar died; last lines of $LOG:" >&2
  tail -5 "$LOG" >&2
  exit 1
fi
