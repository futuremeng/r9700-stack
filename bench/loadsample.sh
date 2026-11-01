#!/bin/bash
# Footprint time series during a load, to size the load-time memory fix (E3). prof.snap only gives phase
# boundaries; what matters here is WHEN the ranks take RAM and when Cached falls, on the same clock.
#   usage: bash bench/loadsample.sh [outdir]      (run alongside bench/relaunch.sh; stop with pkill -f loadsample)
OUT=${1:-$HOME/logs}
F=$OUT/loadsample-$(date +%H%M%S).csv
rank_prev=0
echo "ts,memavail_gib,cached_gib,mlocked_mib,swapused_mib,workers,rss_sum_gib,rss_max_gib,busy,psi_mem_s,psi_io_s,disk_rd_mbs,disk_wr_mbs,rank_rd_mbs" > "$F"
prev_r=0; prev_w=0
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
  busy=$(timeout 3 sh -c 'for f in /sys/class/drm/card*/device/gpu_busy_percent; do printf "%s " "$(cat $f 2>/dev/null)"; done' | tr ' ' '+' )
  # PSI: memory stall is NOT the same as IO stall -- the copy_ stalls showed wchan=folio_wait_bit_common, which is
  # a wait on a page being read, so both are needed.
  psi_mem=$(awk '/^some/{split($2,a,"=");print a[2]}' /proc/pressure/memory 2>/dev/null)
  psi_io=$(awk '/^some/{split($2,a,"=");print a[2]}' /proc/pressure/io 2>/dev/null)
  # diskstats: sectors read/written on the data device (rd, the one holding the checkpoint)
  read -r r w < <(awk '$3 ~ /^sd|^nvme|^vd/ {r+=$6; w+=$10} END {print r+0, w+0}' /proc/diskstats)
  rd=$(( (r - prev_r) / 2 / 2048 )); wr=$(( (w - prev_w) / 2 / 2048 ))   # MiB/s over the 2 s interval
  prev_r=$r; prev_w=$w
  # what the vLLM workers themselves are reading from block devices right now
  rdb=0
  for p in $(pgrep -f 'VLLM[:][:]Worker'); do
    v=$(awk '/^read_bytes/{print $2}' /proc/$p/io 2>/dev/null)
    [ -n "$v" ] && rdb=$(( rdb + v ))
  done
  endrank=$(( (rdb - rank_prev) / 2 / 1048576 )); rank_prev=$rdb
  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" "$(date +%H:%M:%S)" "$avail" "$cached" "$mlocked" "$swapused" \
    "${nw:-0}" "${sum:-0}" "${mx:-0}" "$busy" "${psi_mem:-?}" "${psi_io:-?}" "$rd" "$wr" "$endrank" >> "$F"
  sleep 2
done
