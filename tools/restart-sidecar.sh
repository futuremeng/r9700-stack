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
# Two things the sidecar carries that the in-serve Monitor cannot, both for the moment the serve is gone
# (2026-11-03 04:36: the serve stopped answering on :8080 for minutes while its pid lived, and the in-serve
# Monitor -- the page people actually have bookmarked -- went away with it):
#   --meta            the launch's own meta file, so the page can print the command that would bring it back
#   --journal-attach  the merged monitor's request journal, READ-ONLY: the dead launch's rows stay on screen,
#                     and "read-only" is what keeps two Monitors from replaying each other's rows.
ARGS=(serve/webui.py --port "$PORT" --upstream "$UP")
META="${META:-}"
if [ -z "$META" ]; then for m in "$HOME"/r9700-web-meta-*.json; do [ -f "$m" ] && META="$m" && break; done; fi
[ -n "$META" ] && ARGS+=(--meta "$META")
JAT="${JOURNAL_ATTACH:-}"
if [ -z "$JAT" ]; then j="$HOME/logs/requests-${UP##*:}.jsonl"; [ -f "$j" ] && JAT="$j"; fi
if [ -n "$JAT" ] && [ "$PORT" != "${UP##*:}" ]; then ARGS+=(--journal-attach "$JAT"); fi
nohup "$VENV/bin/python" "${ARGS[@]}" > "/tmp/r9700-sidecar-${PORT}.out" 2>&1 &
echo $! > "$PIDF"
sleep 6
curl -sS -o /dev/null -w "sidecar :$PORT -> HTTP %{http_code}\n" "http://127.0.0.1:$PORT/monitor"
