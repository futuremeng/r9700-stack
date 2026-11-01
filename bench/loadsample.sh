#!/bin/bash
# Footprint time series during a load, to size the load-time memory fix (E3). prof.snap only gives phase
# boundaries; what matters here is WHEN the ranks take RAM and when Cached falls, on the same clock.
#   usage: bash bench/loadsample.sh [outdir]      (run alongside bench/relaunch.sh; stop with pkill -f loadsample)
OUT=${1:-$HOME/logs}
F=$OUT/loadsample-$(date +%H%M%S).csv
echo "ts,memavail_gib,cached_gib,mlocked_mib,swapused_mib,workers,rss_sum_gib,rss_max_gib" > "$F"
while :; do
  avail=$(awk '/MemAvailable/{print $2/1048576}' /proc/meminfo)
  cached=$(awk '/^Cached:/{print $2/1048576}' /proc/meminfo)
  mlocked=$(awk '/^Mlocked:/{print $2/1024}' /proc/meminfo)
  swapused=$(awk '/^SwapTotal:/{t=$2} /^SwapFree:/{printf "%.0f", (t-$2)/1024}' /proc/meminfo)
  # RSS of the four vLLM workers from /proc/<pid>/status (smaps_rollup is cheaper but needs perms)
  # bracketed pattern on purpose: an unbracketed 'VLLM::Worker' in this line's argv was matched by serve/native-flashnext.sh's
  # own guard (pgrep -cf 'VLLM[::]'), which then refused a launch that had just passed the gate (2026-11-01 05:09).
  read -r nw sum mx < <(pgrep -f 'VLLM[:][:]Worker' | while read -r p; do
      awk '/^VmRSS/{print $2}' /proc/$p/status 2>/dev/null | awk -v OFS=' ' '{print 1, $1, $1}'
    done | awk '{n+=$1; s+=$2; if($3>m) m=$3} END {printf "%d %.1f %.1f\n", n, s/1048576, m/1048576}')
  printf "%s,%s,%s,%s,%s,%s,%s,%s\n" "$(date +%H:%M:%S)" "$avail" "$cached" "$mlocked" "$swapused" "${nw:-0}" "${sum:-0}" "${mx:-0}" >> "$F"
  sleep 2
done
