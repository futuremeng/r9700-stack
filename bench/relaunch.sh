#!/bin/bash
# The only sanctioned way to bring the native server back up: verified gate, then launch, then wait for the load
# line and print the fingerprints that make two instances comparable (load seconds, KV pool, MTP depth, and the
# box's own counters at the moment the load finished).
#   usage: PROBE32=1 bash bench/relaunch.sh [label]        (MAXLEN= defaults to 32768; this box's standing recipe is
#           MAXLEN=262144 NSEQ=32 NBT=8192 WEBUI=same -- see notes/context-262k.md; setup.py's context check exists
#           precisely because a launch that comes up at 1/8 of the checkpoint's context still answers /health)
cd /home/jiulu/r9700-stack || exit 1
LABEL=${1:-inst}
LOG=$HOME/logs/relaunch-$LABEL.log
GATE=$HOME/logs/relaunch-$LABEL.gate

bash bench/gate.sh > "$GATE" 2>&1
tail -2 "$GATE"
grep -q "^gate: PASS" "$GATE" || { echo "refusing to launch: gate failed"; exit 3; }

R9K_LOAD_PROF=1 R9K_PRESTAGE_MTP=${R9K_PRESTAGE_MTP:-1} R9K_LOAD_STACKS=${R9K_LOAD_STACKS:-0} R9K_LOAD_COPIES=${R9K_LOAD_COPIES:-0} R9K_PLE_PIN_LATE=${R9K_PLE_PIN_LATE:-1} R9K_MOE_BATCH_H2D=${R9K_MOE_BATCH_H2D:-0} R9K_PRESTAGE_AHEAD=${R9K_PRESTAGE_AHEAD:-0} R9K_CACHE_WINDOW=${R9K_CACHE_WINDOW:-0} GPUS=0,1,2,3 TP=4 OFFLOAD_GB=0 NSEQ=32 MAXLEN=${MAXLEN:-32768} \
  setsid bash serve/native-flashnext.sh > "$LOG" 2>&1 &

load=""
for _ in $(seq 1 120); do   # 1200 s, not 400: a first launch on a cold VLLM_CACHE_ROOT/TRITON_CACHE_DIR (a pin
                        # move, a new flavor, an AOT recompile) spends minutes compiling before the load line appears
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
# Load anatomy, printed once the service is up so every worker has finished both passes. This is what the
# load-path experiments are measured on: prestage queueing, the draft pass's own bytes and wall time, and the
# phase snapshots whose read_bytes deltas say who re-read the checkpoint from disk.
echo "== load anatomy ($LABEL) =="
grep -oE "r9700 prestage mtp.*" "$LOG" | sed -E 's/.*(r9700)/  \1/' | sort -u
grep -oE "load profile .draft.*" "$LOG" | sed 's/^/  draft: /'
grep -oE "r9700 copies +[0-9]+ calls.*" "$LOG" | sed 's/^/  /' | sort -u | cut -c1-165
grep -oE "r9700 stacks leaf +[0-9.]+%.*" "$LOG" | sed 's/^/  /' | sort -u | head -5 | cut -c1-125
grep -oE "snap (target end|prestage mtp end|draft end).*" "$LOG" | sed -E 's/.*(snap )/  \1/' | sort -u | cut -c1-170
[ "${PROBE32:-0}" = 1 ] && PROBE_OUT=$HOME/logs/probe-$LABEL.txt bash bench/probe32.sh
