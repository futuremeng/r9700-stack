#!/bin/bash
# Relaunch until a slow post-load window actually happens, so the attribution that is now in place -- per-class and
# per-module wall / user CPU / sys CPU / minor+major faults, plus the thread-state and PSI census -- catches one with
# its brackets on. 4 of the last 10 instances had a tail (worst rank 24.9-61.8 s against 4.4-5.9 s), so this is just
# automating the wait; every round prints the same fingerprints, and the round that catches one keeps its server up.
#   usage: BASE=45 bash bench/catch_tail.sh [rounds] [threshold_s]   # default: 4 rounds, a tail is >= 8 s on one rank
cd /home/jiulu/r9700-stack || exit 1
ROUNDS=${1:-4}
THRESH=${2:-8}
BASE=${BASE:-43}

for r in $(seq 1 "$ROUNDS"); do
    LABEL="tail$((BASE + r))"
    LOG=$HOME/logs/relaunch-$LABEL.log
    pkill -f "bench/relaunch.sh" 2>/dev/null          # previous round's reporter, not its server (setsid detached it)
    setsid bash bench/relaunch.sh "$LABEL" > "$HOME/logs/relaunch-$LABEL.out" 2>&1 &
    for _ in $(seq 1 32); do                          # the window is attributed before the load line prints
        sleep 10
        n=$(grep -c "target post-load:" "$LOG" 2>/dev/null)
        [ "${n:-0}" -ge 4 ] && break
    done
    windows=$(grep -h "target post-load:" "$LOG" 2>/dev/null | sed -E 's/.*: ([0-9.]+) s.*/\1/' | sort -rn | tr '\n' ' ')
    worst=$(echo "$windows" | awk '{print $1}')
    kv=$(grep -m1 -oE "GPU KV cache size: [0-9,]+ tokens" "$LOG" | grep -oE "[0-9,]+ tokens")
    load=$(grep -m1 -oE "Model loading took [0-9.]+ GiB memory and [0-9.]+ seconds" "$LOG" | grep -oE "and [0-9.]+ seconds")
    echo "== $LABEL: windows (s) [$windows] worst=${worst:-?} KV=${kv:-?} ${load:-}"
    awk -v w="${worst:-0}" -v t="$THRESH" 'BEGIN { exit !(w + 0 >= t) }' && {
        echo "caught a window >= ${THRESH} s on $LABEL; attribution:"
        grep -h "post-load" "$LOG" | sed 's/.*\] //' | cut -c1-140
        exit 0
    }
done
echo "no window >= ${THRESH} s in $ROUNDS rounds; last instance is up"
