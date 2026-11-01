#!/bin/bash
# The only sanctioned way to bring the native server back up: verified gate, then launch, then wait for the load
# line and print the fingerprints that make two instances comparable (load seconds, KV pool, MTP depth, and the
# box's own counters at the moment the load finished).
#   usage: PROBE32=1 bash bench/relaunch.sh [label]
cd /home/jiulu/r9700-stack || exit 1
LABEL=${1:-inst}
LOG=$HOME/logs/relaunch-$LABEL.log
GATE=$HOME/logs/relaunch-$LABEL.gate

bash bench/gate.sh > "$GATE" 2>&1
tail -2 "$GATE"
grep -q "^gate: PASS" "$GATE" || { echo "refusing to launch: gate failed"; exit 3; }

R9K_LOAD_PROF=1 GPUS=0,1,2,3 TP=4 OFFLOAD_GB=0 NSEQ=32 MAXLEN=32768 \
  setsid bash serve/native-flashnext.sh > "$LOG" 2>&1 &

load=""
for _ in $(seq 1 40); do
  sleep 10
  load=$(grep -m1 -oE "Model loading took [0-9.]+ GiB memory and [0-9.]+ seconds" "$LOG" | grep -oE "and [0-9.]+ seconds" | grep -oE "[0-9.]+")
  [ -n "$load" ] && break
done
[ -n "$load" ] || { echo "load did not finish in 400 s -- see $LOG"; exit 4; }

echo "== fingerprints ($LABEL) =="
grep -m1 "Model loading took" "$LOG" | sed 's/^/  load: /'
grep -m1 "load profile (target)" "$LOG" | sed 's/.*r9700 load profile (target): /  place: /'
grep -m1 -oE "GPU KV cache size: [0-9,]+ tokens, Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" "$LOG" | sed 's/^/  kv: /'
grep -m1 -oE "'speculative_config': \{[^}]*\}" "$LOG" | sed 's/^/  spec: /'
grep -m1 -oE "Available RAM: [0-9.]+ GiB" "$LOG" | sed 's/^/  at loader start: /'
grep -m1 -oE "Available KV cache memory: [0-9.]+ GiB" "$LOG" | sed 's/^/  /'
echo "  after load: $(grep -E 'MemAvailable|SwapFree' /proc/meminfo | tr -d '\n' | sed 's/Mem/ Mem/g')"
for _ in $(seq 1 30); do
  timeout 15 curl -s -o /dev/null -w "" http://localhost:8080/health && { echo "  health: 200"; break; }
  sleep 5
done
[ "${PROBE32:-0}" = 1 ] && PROBE_OUT=$HOME/logs/probe-$LABEL.txt bash bench/probe32.sh
