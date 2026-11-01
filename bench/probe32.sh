#!/bin/bash
# Throughput anchor + power-envelope check for the 4 x R9700 native stack. Written after the 2026-11-01 reboot to
# answer "is @16 1,210 against the recorded 1,265 a regression?" -- answer: the hardware is identical, the
# *protocol* was never pinned down, and it matters by more than the delta being asked about.
#
# Two things this must do right (both learned the hard way):
#   1. Sample power only while the box is actually loaded. A 32 x 300-token wave is ~6 s; sampling for a minute
#      around it returned a median of 258 W and looked like the envelope had moved. Gated on gpu_busy_percent==100
#      it reads a flat **300.0 W a card, four cards** -- exactly the envelope in notes/native-4xr9700.md.
#   2. State the wave length. At 32 in flight, aggregate reads 1,573-1,602 with 300-token requests and 1,843 with
#      1,200-token requests (per-stream 50-53 vs 57.6-61.7): short waves pay TTFT + queue ramp, so a
#      "3 x 300-token sample" baseline and a "300 x 300 in one wave" baseline are not comparable.
cd /home/jiulu/r9700-stack || exit 1
PY=~/venv-r9700/bin/python3
OUT=~/logs/probe32.txt; PW=~/logs/probe32.pwr
: > "$OUT"; : > "$PW"

power_gated () {   # $1: sample power+busy every 2 s until told to stop
  while [ -f "$PW.run" ]; do
    local p b
    p=$(timeout 4 rocm-smi --showpower 2>/dev/null | awk -F': ' '/Package Power/{printf "%s ", $NF}')
    b=$(timeout 4 sh -c 'for f in /sys/class/drm/card*/device/gpu_busy_percent; do printf "%s " "$(cat $f)"; done')
    echo "$(date +%H:%M:%S) busy=[$b] W=[$p]" >> "$PW"
    sleep 2
  done
}

wave () {   # $1 conc, $2 tokens, $3 label
  $PY - "$@" <<'PY' >> "$OUT" 2>&1
import sys, statistics
sys.path.insert(0, "bench")
import bench
conc, toks, label = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
agg, n, st = bench.aggregate(conc, toks)
print(f"{label}: agg {agg:.1f} tok/s, {n} ok, per-stream median {statistics.median(st):.1f} "
      f"[{min(st):.1f}..{max(st):.1f}]", flush=True)
PY
}

touch "$PW.run"; power_gated &
SAMP=$!
for n in 1 2 3; do wave 32 300  "@32 x300  sample $n"; done
for n in 1 2;   do wave 32 1200 "@32 x1200 sample $n"; done
for n in 1 2 3; do wave 16 300  "@16 x300  sample $n"; done
rm -f "$PW.run"; wait $SAMP 2>/dev/null

echo "--- power while busy==100 only ---" >> "$OUT"
grep -o 'W=\[[^]]*\]' "$PW" | sed 's/W=\[//;s/\]//' | tr ' ' '\n' | grep -E '^[0-9]+\.?[0-9]*$' | awk '$1 > 150' \
  | sort -n | awk '{v[NR]=$1} END { if (NR) printf "n=%d  min=%.0f  median=%.0f  max=%.0f W a card\n", NR, v[1], v[int((NR+1)/2)], v[NR] }' >> "$OUT"
grep -c '100 100 100 100' "$PW" | sed 's/^/busy-100 samples: /' >> "$OUT"
cat "$OUT"
