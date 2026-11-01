#!/bin/bash
# Verified stop + launch gate. Use this instead of typing STOP/pgrep patterns inline in a tool call.
#
# WHY A FILE: serve/native-flashnext.sh's STOP path runs `pkill -9 -f 'VLLM[::]'` and `pkill -9 -f 'vll[m] serve'`.
# If you inline those literals in the command you are running, pkill matches YOUR OWN command line and SIGKILLs
# the caller (observed 2026-11-01: a tool call carrying the string "VLLM::" exited 137 while the STOP itself
# worked). Running the patterns from a script file keeps them out of the caller's argv.
#
# WHY THE GATE: STOP signals and sleeps 5 s; it does not wait for the ranks' ~50 GiB of anon to be unmapped, and
# relaunching before that is what put the box into the amdgpu SVM/GPU-fence deadlock (notes/load-oom-svm-wedge.md).
# Measured on a healthy boot-cold TP4 load: it alone reaches Available RAM 39.8 GiB at loader start, ends with
# Cached 42.1 GiB, MemAvailable 39.5 GiB and **4.2 GiB of swap used**. Teardown gives all of it back
# (MemAvailable 91.5 GiB, SwapFree 8.35 of 8.39 GiB) -- but only once the ranks are really gone.
cd /home/jiulu/r9700-stack || exit 1
MIN_AVAIL_KB=$((55 * 1048576))
MIN_SWAPFREE_KB=$((6 * 1048576))

STOP=1 bash serve/native-flashnext.sh 2>&1 | sed 's/^/  stop: /'

gate () {   # print one line: procs, tasks in D, MemAvailable, SwapFree
  local n d
  n=$(pgrep -cf 'VLLM[::]|vll[m] serve' || true)
  d=0
  for p in $(pgrep -d' ' 'VLLM::' 2>/dev/null); do
    [ "$(awk '{print $3}' /proc/$p/stat 2>/dev/null)" = D ] && d=$((d + 1))
  done
  printf '  t+%3ss procs=%s inD=%s MemAvail=%s GiB SwapFree=%s GiB\n' "$1" "${n:-0}" "$d" \
    "$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)" \
    "$(awk '/SwapFree/{printf "%.1f", $2/1048576}' /proc/meminfo)"
  GATE_PROCS=${n:-0}; GATE_D=$d
  GATE_AVAIL=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  GATE_SWAPFREE=$(awk '/SwapFree/{print $2}' /proc/meminfo)
}

for i in $(seq 1 36); do
  sleep 5
  gate "$((i * 5))"
  [ "$GATE_PROCS" = 0 ] || continue
  [ "$GATE_AVAIL" -ge "$MIN_AVAIL_KB" ] && [ "$GATE_SWAPFREE" -ge "$MIN_SWAPFREE_KB" ] && { echo "gate: PASS"; exit 0; }
done
echo "gate: FAIL -- procs stuck in D or memory not returned; treat the box as wedged (notes/load-oom-svm-wedge.md)"
exit 3
