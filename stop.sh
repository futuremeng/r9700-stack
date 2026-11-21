#!/bin/bash
# stop.sh - stop what this repository started, in the order that makes stopping quiet, and say what it did.
#
# Every launcher already answers STOP=1, so why a file for this:
#
# (1) ORDER. The monitors poll. A sidecar that is watched answers /health every few seconds, so taking the
#     watched thing down first makes its watcher print errors for the rest of its life, and makes the log you
#     then read about the engine full of someone else's failures. The front door comes next: it is a byte pipe
#     with no state and no cost to bring back, and it should be CLOSED before what it fronts goes down, so no
#     client is mid-request into a sidecar that is being stopped. Then the sidecars, then the engine -- the
#     expensive thing last.
# (2) GRACE. The engine's own STOP is a pkill -9, and it has to be: measured 2026-10-31, SIGTERM does not stop
#     the ranks, so pkill'ing the API server leaves four VLLM::Worker_TP* holding 27-31 GiB a card and the next
#     launch dies on "Free memory on device cuda:0 (0.0/31.86 GiB)" (notes/launch-methodology.md). But -9 as
#     the FIRST move is ruder than the engine needs: it skips the API server's own shutdown. So this sends a
#     TERM, waits for the ranks to leave on their own, and only then calls the launcher's STOP to clear what is
#     left -- and reports the leftover count either way, because "stopped" and "stopped and still holding the
#     cards" are two different sentences.
# (3) THE SESSION. A native launch started by setup.py lives in a tmux session, and that session IS the
#     liveness probe the next run asks (notes/launch-methodology.md §3). A bare pkill leaves the pane and its
#     shell standing, and the next report then describes a launch that is not a launch.
#
# Nothing here invents a kill pattern. The patterns belong to the launchers -- they are read out of their own
# PAT= lines and their own pkill lines -- because a pattern kept in two files is a pattern that drifts, and
# this one is the difference between a clean stop and a card that stays allocated.
#
#   ./stop.sh                    stop everything: the monitors, the door, the sidecars, the engine
#   ./stop.sh --embedding        the CPU sidecars and their monitors only (no GPU, no door touched)
#   ./stop.sh --engine           the engine, its Monitor page and its tmux session only
#   ./stop.sh --check            report what is running; stop nothing
#   DRY=1 ./stop.sh              print what would run, run nothing
#   STATE=path/to/launch.json    stop FROM that record instead of .r9700-state/launch.json
#   WAIT=20                      how long to give the ranks to leave on their own after the TERM (default 20s)
#   ./stop.sh --yes              no question before the engine -- which is asked because coming back up costs
#                                the four minutes of loading that the launch log is full of
#
# Why --engine and --embedding do NOT take the door down: it fronts the engine AND both sidecars, so whichever
# half you stop with one of those flags, the other half still answers through it. A door routing to a stopped
# engine is honest -- its /health is the engine's own, and it goes down with it (setup.py's counters read the
# same truth when a rank dies).
#
# What it does NOT do: remove anything. The weights in ~/models, the launch caches, the image, the stored launch
# configurations and the state record all stay put -- the same contract as update.sh.

cd "$(dirname "$0")" || exit 1
ENGINE=1; EMB=1; GATE=1; CHECK=${CHECK:-0}; DRY=${DRY:-0}; YES=${YES:-0}; WAIT=${WAIT:-20}
for a in "$@"; do
  case "$a" in
    --engine) ENGINE=1; EMB=0; GATE=0 ;;
    --embedding) EMB=1; ENGINE=0; GATE=0 ;;
    --check|--report) CHECK=1 ;;
    --dry) DRY=1 ;;
    --yes|-y) YES=1 ;;
    *) echo "stop.sh: $a is not one of --engine --embedding --check --yes (or DRY=1)" >&2; exit 2 ;;
  esac
done
[ "$CHECK" = 1 ] && DRY=1

run() { echo "      \$ $*"; [ "$DRY" = 1 ] || { "$@" || true; }; }
say() { echo "  $*"; }

# Patterns go to pkill after --: a pattern read out of a file could begin with a dash, and pkill would then read
# it as its own options -- rc=2, nothing killed, and a usage error where a stop was promised.
#
# pat_any is the launcher's own pattern with its port stripped off, so one pass catches every instance of the
# thing rather than the one instance the recorded port happens to name.
pat_any() { grep -m1 '^PAT=' "$1" | sed 's/^[^"]*"//; s/"$//; s/ \$[A-Z_0-9]*$//'; }
# ... and the patterns its STOP block kills with, for the TERM that comes before that STOP.
pats() { grep -o "pkill -9 -f '[^']*'" "$1" 2>/dev/null | sed "s/.*'\(.*\)'/\1/" | sort -u; }
# Which sidecar tasks the launcher knows. Read from the file, not from a list kept here: a third task added
# there is a third process this has to stop, and a list in two files is how one gets left running.
tasks() { sed -n '/case "\$TASK"/,/^esac/p' serve/embed.sh | sed -n 's/^  \([a-z][a-z0-9]*\)).*/\1/p'; }

# A count, not a match: this file's report is numbers, and "stopped" versus "stopped with four ranks still
# holding the cards" has to be distinguishable in it. pgrep -c is GNU, so the launchers can assume it and this
# file cannot -- it gets run by whoever is cleaning up, from whatever box they happen to be typing on.
count() { local n; n=$(pgrep -cf "$1" 2>/dev/null); [ -n "$n" ] || n=$(pgrep -f "$1" 2>/dev/null | wc -l | tr -d ' ')
  echo "${n:-0}"; }
left() { count 'VLLM[::]|vll[m] serve'; }
# Where the last launch recorded itself. A knob so a stop can be aimed at one record instead of the box's
# live state -- "stop what THIS launch.json names", asked of a file that is not .r9700-state's own.
STATE=${STATE:-.r9700-state/launch.json}
# The port a sidecar task listens on. Asked of the launcher in DRY mode rather than parsed out of it: the task
# blocks do not all carry their own default the same way, and a parsed guess is how a stop ends up aimed at a
# port nothing is listening on. DRY starts nothing, downloads nothing, and honours the same native.env the
# real launch would (serve/lib/localenv.sh).
task_port() {
  DRY=1 TASK="$1" LLAMA_SERVER=${LLAMA_SERVER:-/bin/true} bash serve/embed.sh 2>/dev/null \
    | grep -o -- '--port [0-9]*' | grep -o '[0-9]*' | head -1
}
REC() { [ -f "$STATE" ] && python3 -c 'import json,sys
try: d = json.load(open(sys.argv[1]))
except Exception: d = {}
v = d.get(sys.argv[2]); print("" if v is None else v)' "$STATE" "$1" 2>/dev/null; }

recipe=$(REC recipe); sess=$(REC session); eport=$(REC port); epath=$(REC path)
say "stopping what this checkout started ($(pwd))$([ "$CHECK" = 1 ] && echo ' -- check only, nothing is stopped')"

# ------------------------------------------------------- 1. the monitors, because they poll -------
mons=""
[ "$ENGINE" = 1 ] && mons="webui.sh"          # the Monitor page polls the engine
[ "$EMB" = 1 ] && mons="$mons embed-watch.sh" # and this one polls the sidecar it watches
if [ -n "$mons" ]; then
  say "monitors first: their errors would otherwise fill the log you are about to read"
  for s in $mons; do
    if ! grep -q '^PAT=' "serve/$s" 2>/dev/null; then
      say "  serve/$s has no PAT= line: skipped rather than guessed at"; continue; fi
    p=$(pat_any "serve/$s"); n=$(count "$p")
    if [ "$n" = 0 ]; then say "  $s: not running"; continue; fi
    say "  $s: $n instance(s) matching \"$p\""     # the pattern's port is stripped, so this catches both
    run pkill -TERM -f -- "$p"                    # embed-watch pages (:8091 and the reranker's :8093)
  done
fi

# ------------------------------------------------------------- 2. the front door, cheap and stateless
if [ "$GATE" = 1 ]; then
  say "the door next: closing it stops traffic before what it fronts goes down"
  g=$(count "$(pat_any serve/gateway.sh)")
  if [ "$g" = 0 ]; then
    say "  gateway: not running"
  else
    say "  gateway: $g instance(s); its STOP is a pidfile kill, so it comes after that in case the pidfile died"
    say "    before the process"
    run bash serve/gateway.sh STOP=1
    g=$(pat_any serve/gateway.sh); g=$(count "$g")
    [ "$g" != 0 ] && [ "$DRY" != 1 ] && { say "    still up with a dead pidfile: TERM by the launcher's own pattern"; run pkill -TERM -f -- "$(pat_any serve/gateway.sh)"; }
  fi
fi

# ---------------------------------------- 3. the CPU sidecars, mmap and cheap (embed and the reranker)
if [ "$EMB" = 1 ]; then
  say "embedding and reranking sidecars: no GPU, no state, nothing to lose by stopping them"
  for t in $(tasks); do
    tp=$(task_port "$t")
    if [ -z "$tp" ]; then
      say "  TASK=$t: the launcher answered no port (a missing GGUF, or its own guards tripped) -- stopping by task anyway"
      run env STOP=1 TASK="$t" bash serve/embed.sh
    elif pgrep -f "llama-server.*--port $tp( |$)" >/dev/null 2>&1; then
      say "  TASK=$t (:${tp}, the port the launcher says it would use)"
      run env STOP=1 TASK="$t" PORT="$tp" bash serve/embed.sh
    else
      say "  $t: not running on :$tp"
    fi
  done
fi

# --------------------------------------------------------------- 4. the engine, the expensive one
if [ "$ENGINE" = 1 ]; then
  say "the engine last: it is the one thing that costs minutes to bring back"
  n=$(left); n=${n:-0}; go=1
  if [ "$n" = 0 ] && [ -z "$sess" ] && [ "$epath" != docker ]; then
    say "  no vLLM process, no recorded tmux session, no container recorded: nothing to stop"; go=0
  elif [ -t 0 ] && [ "$YES" != 1 ] && [ "$CHECK" != 1 ]; then
    printf '    stop the engine too (%s vLLM process(es) up, recipe %s)? [Y/n] ' "$n" "${recipe:-unknown}"
    read -r a
    case "$a" in
      [nN]*) say "  engine left running: rerun with --embedding only, or STOP=1 bash serve/${recipe:-native-flashnext}.sh"
             go=0 ;;
    esac
  fi
  if [ "$go" = 1 ]; then
    # The recorded recipe, but only if that launcher actually knows how to stop itself -- a container recipe
    # has no pkill line, and calling STOP=1 on it would print a launch usage instead of clearing anything.
    script=serve/native-flashnext.sh
    [ -n "$recipe" ] && [ -n "$(pats "serve/$recipe" 2>/dev/null)" ] && script=serve/$recipe
    say "  $script: TERM first; the launcher's own STOP only for what the TERM does not clear"
    pats "$script" | while read -r p; do
      say "    TERM $(count "$p") process(es) matching $p"
      run pkill -TERM -f -- "$p"
    done
    if [ "$DRY" != 1 ]; then
      for i in $(seq 1 "$WAIT"); do [ "$(left)" = 0 ] && break; sleep 1; done
      [ "$(left)" != 0 ] && say "  $(left) still up after ${WAIT}s: the ranks ignore SIGTERM, so this next part is a -9 by design"
    fi
    run env STOP=1 bash "$script"
    if [ -n "$sess" ] && command -v tmux >/dev/null && tmux has-session -t "$sess" 2>/dev/null; then
      say "  the tmux session is the launch's liveness probe, so it goes too: $sess"
      run tmux kill-session -t "$sess"
    fi
    if command -v docker >/dev/null 2>&1 && [ -n "$(docker ps -q -f "name=${NAME:-vllmstock}" 2>/dev/null)" ]; then
      say "  the container path is up as well: serve.sh runs it with sudo, so this does too (sudo asks)"
      run sudo docker rm -f "${NAME:-vllmstock}"
    fi
  fi
fi

# ------------------------------------------------------------------------ the honest tail -------
n=$(left); n=${n:-0}
m=$(count "llama-server")
w=$(count "$(pat_any serve/embed-watch.sh)")
u=$(count "$(pat_any serve/webui.sh)")
d=$(count "$(pat_any serve/gateway.sh)")
if [ "$n$m$w$u$d" = 00000 ]; then
  say "nothing left running: 0 vLLM, 0 llama-server, 0 embedding monitor, 0 Monitor page, 0 door"
  [ "$DRY" = 1 ] && say "  (a preview: nothing above was run, and those counts are what is up right now)"
  say "bring it back: ./setup.sh   (with a stored launch configuration it re-asks nothing)"
else
  say "still up: $n vLLM, $m llama-server, $w embedding monitor, $u Monitor page, $d door$([ "$DRY" = 1 ] && echo '  (a preview: nothing above was run)')"
  if [ "${n:-0}" != 0 ] && [ "$DRY" != 1 ]; then
    say "  those vLLM processes are what the next launch refuses to start over: it reads them as your cards"
    say "  being allocated -- rerun this, or STOP=1 bash serve/${recipe:-native-flashnext}.sh, or FORCE=1 anyway"
  fi
fi
