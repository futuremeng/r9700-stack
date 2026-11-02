#!/usr/bin/env bash
# tools/restart-sidecar.sh -- restart the Monitor sidecar without the classic self-kill.
#
# The obvious `pkill -f serve/webui.py` run from inside a shell command that also contains the launch line
# matches that shell's own command line, and takes the calling command down with exit 143. This one tracks the
# pid in a file and kills only that.
set -u
PORT="${1:-8099}"
UP="${2:-http://127.0.0.1:8080}"
VENV="${VENV:-$HOME/venv-r9700}"
PIDF="/tmp/r9700-sidecar-${PORT}.pid"
if [ -f "$PIDF" ]; then
  kill "$(cat "$PIDF")" 2>/dev/null || true
  sleep 1
fi
cd "$(dirname "$0")/.." || exit 1
nohup "$VENV/bin/python" serve/webui.py --port "$PORT" --upstream "$UP" \
  > "/tmp/r9700-sidecar-${PORT}.out" 2>&1 &
echo $! > "$PIDF"
sleep 6
curl -sS -o /dev/null -w "sidecar :$PORT -> HTTP %{http_code}\n" "http://127.0.0.1:$PORT/monitor"
