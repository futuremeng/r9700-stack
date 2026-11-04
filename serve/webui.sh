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
#   DRY=1 bash serve/webui.sh              # print what would start, start nothing, stop nothing
#   STOP=1 bash serve/webui.sh               # stop the sidecar for this port
# Knobs: PORT (the vLLM port), WEBUI_PORT, UPSTREAM (full URL, wins over both), WEBUI_MODE=observe|proxy,
#        WEBUI_BIND (default 127.0.0.1: the API carries the engine's config and the request records, so
#        the sidecar binds loopback like every URL it prints; 0.0.0.0 answers on every interface),
#        R9K_WEB_BIND (the same choice one level down, read by webui.py as its --bind default; the machine-local
#        env file below normally holds it), LOCAL_ENV (path to that file, default
#        ~/.config/r9700-stack/native.env -- the same file serve/native-flashnext.sh reads),
#        POLL_S, KEEP, GPUS (sysfs root), SMI=1 (fill the sysfs gaps from rocm-smi: it opens /dev/kfd, so
#        leave it off while measuring; SMI="<command>" overrides the default invocation),
#        VERSION (string to show as the engine version), FORE=1 (no nohup).
PORT=${PORT:-8080}
WPORT=${WEBUI_PORT:-$((PORT + 1))}
UP=${UPSTREAM:-http://127.0.0.1:$PORT}
DIR=$(dirname "$(realpath "$0")")
# A box fact is not a repo fact (notes/launch-methodology.md section 10). The repo default binds the Monitor to
# loopback, because /api/* carries the engine's config and its request records; a box that serves the Monitor to
# other machines says so in this file rather than in the repo, and the choice is then written down instead of
# being whatever the last hand-typed command happened to inherit. Same file the engine launcher reads.
LOCAL_ENV=${LOCAL_ENV:-$HOME/.config/r9700-stack/native.env}
if [ -f "$LOCAL_ENV" ]; then set -a; . "$LOCAL_ENV"; set +a
  # `set -a` because the file holds facts for the CHILD processes too, not only for this script. Sourcing alone
  # makes shell variables: webui.py reads R9K_WEB_BIND from the environment as its --bind default, and without the
  # export a sourced line does nothing at all -- measured the same day, a sidecar restarted from a hand-typed
  # command that sourced the file came up on 127.0.0.1:8099 with the file saying 0.0.0.0.
  echo "local env: $LOCAL_ENV"; fi
LOG=${WEBUI_LOG:-$HOME/r9700-webui-$WPORT.log}
PAT="webui.py --port $WPORT"

if [ "${STOP:-0}" = 1 ]; then
  pkill -f "$PAT" && echo "stopped the sidecar on $WPORT" || echo "nothing running on $WPORT"
  exit 0
fi

ARGS=(--port "$WPORT" --upstream "$UP")
# Always explicit: the sidecar's bind address is then readable in the process's own command line instead of
# inferred from which env happened to be set. The chain is the flag, then the machine-local file, then the repo
# default -- and the repo default is loopback, which is also what every URL this script prints promises.
BIND=${WEBUI_BIND:-${R9K_WEB_BIND:-127.0.0.1}}
ARGS+=(--bind "$BIND")
[ -n "$WEBUI_KEY" ]  && ARGS+=(--key "$WEBUI_KEY")
[ -n "$META" ]       && ARGS+=(--meta "$META")
[ -n "$VERSION" ]    && ARGS+=(--version "$VERSION")
[ "${WEBUI_MODE:-observe}" = proxy ] && ARGS+=(--proxy)
[ -n "$POLL_S" ]     && ARGS+=(--poll "$POLL_S")
[ -n "$KEEP" ]       && ARGS+=(--keep "$KEEP")
[ -n "$GPUS" ]       && ARGS+=(--gpus "$GPUS")
[ -n "$SMI" ]        && ARGS+=(--smi "$SMI")
[ -n "$JOURNAL" ]    && ARGS+=(--journal "$JOURNAL")
[ -n "$JOURNAL_ATTACH" ] && ARGS+=(--journal-attach "$JOURNAL_ATTACH")

if [ "${DRY:-0}" = 1 ]; then
  # A preview must not touch state -- the rule serve/native-flashnext.sh learned the hard way. Without this branch
  # a preview would run the pkill below and take down the very sidecar it was only printing, and its 1 m / 15 m /
  # 1 h windows would start over from empty.
  echo "sidecar: python3 $DIR/webui.py ${ARGS[*]}   (DRY: nothing started, nothing stopped)"
  exit 0
fi

pkill -f "$PAT" 2>/dev/null
if [ "${FORE:-0}" = 1 ]; then exec python3 "$DIR/webui.py" "${ARGS[@]}"; fi
nohup python3 "$DIR/webui.py" "${ARGS[@]}" >> "$LOG" 2>&1 &
sleep 1
if pgrep -f "$PAT" > /dev/null; then
  echo "monitor: http://127.0.0.1:$WPORT/monitor   about: http://127.0.0.1:$WPORT/about"
  echo "upstream $UP · mode ${WEBUI_MODE:-observe} · bind $BIND · log $LOG · stop: STOP=1 bash serve/webui.sh"
else
  echo "the sidecar died; last lines of $LOG:" >&2
  tail -5 "$LOG" >&2
  exit 1
fi
