#!/bin/bash
# Cold-start A/B: the axis that actually exists on this box is host page-cache state, not vLLM prefetch knobs
# (see bench/prefetch_ab.sh for why those are inert). Same config, two arms:
#   cold : drop_caches immediately before launch
#   warm : read the 23 shards into the page cache immediately before launch
# Protocol: two launches per arm, only the second one is reported (one config alone spans 143.8-179.9 s).
#
# HARD RULE -- the memory gate in wait_down() is not cosmetic. On 2026-11-01 a variant loop launched the next
# 4-rank load while the previous server's ~54 GiB of worker RSS was still unreclaimed (its processes were stuck
# in exit_mm on the amdgpu SVM notifier). RAM exhausted, kswapd0 blocked in svm_range_cpu_invalidate_pagetables
# on a mutex a worker held in amdgpu_hmm_range_get_pages, and the machine became unrecoverable without a reboot
# (notes/load-oom-svm-wedge.md). Never relax MIN_AVAIL_GIB.
cd /home/jiulu/r9700-stack || exit 1
OUT=$HOME/logs/coldstart-ab.txt
CKPT=/home/jiulu/models/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ
MIN_AVAIL_GIB=55          # gate: refuse to launch below this (4 ranks x ~14 GiB peak against 93.8 GiB total)
MIN_TEARDOWN_S=20
: > "$OUT"

avail_gib () { awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo; }

wait_down () {
  # (1) no vllm process left, (2) teardown settled, (3) MemAvailable back above the gate.
  local t0=$(date +%s)
  for _ in $(seq 1 36); do
    timeout 20 pgrep -f "vllm serve|VLLM::" >/dev/null || break
    sleep 5
  done
  if timeout 20 pgrep -f "vllm serve|VLLM::" >/dev/null; then
    echo "ABORT: vllm processes still present after 180 s (D-state? see notes/load-oom-svm-wedge.md)"; exit 3
  fi
  sleep "$MIN_TEARDOWN_S"
  for _ in $(seq 1 36); do
    awk -v m="$MIN_AVAIL_GIB" '/MemAvailable/{if ($2/1048576) >= m) exit 0}' /proc/meminfo && break
    echo "  waiting for RAM: MemAvailable=$(avail_gib) GiB"
    sleep 5
  done
  awk -v m="$MIN_AVAIL_GIB" '/MemAvailable/{if ($2/1048576) >= m) exit 0}' /proc/meminfo || {
    echo "ABORT: MemAvailable=$(avail_gib) GiB below gate ${MIN_AVAIL_GIB} GiB"; exit 3; }
}

run_once () {
  local variant="$1" n="$2"
  local log
  log="$HOME/logs/cs-${variant}-${n}.log"          # NOTE: separate `local` lines -- one `local a=.. b=$a..`
  STOP=1 bash serve/native-flashnext.sh >/dev/null 2>&1   # statement expands before the assignments land.
  wait_down
  case "$variant" in
    cold) sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null ;;
    warm) cat "$CKPT"/*.safetensors >/dev/null ;;
  esac
  R9K_LOAD_PROF=1 GPUS=0,1,2,3 TP=4 OFFLOAD_GB=0 MAXLEN=32768 \
    setsid bash serve/native-flashnext.sh > "$log" 2>&1 &
  local load="" place=""
  for _ in $(seq 1 45); do
    sleep 10
    load=$(grep -m1 -oE "Model loading took [0-9.]+ GiB memory and [0-9.]+ seconds" "$log" \
             | grep -oE "and [0-9.]+ seconds" | grep -oE "[0-9.]+")
    [ -n "$load" ] && break
  done
  place=$(grep "load profile (target)" "$log" | grep -m1 -oE "experts [0-9]+t/[0-9.]+GiB/[0-9.]+s" \
            | grep -oE "/[0-9.]+s" | tr -d '/s')
  echo "$(date +%H:%M:%S) variant=$variant run=$n load_s=${load:-FAILED} experts_place_s=${place:-?} \
avail_gib=$(avail_gib) swapfree_gib=$(awk '/SwapFree/{printf "%.1f",$2/1048576}' /proc/meminfo)" >> "$OUT"
}

for variant in cold warm; do
  run_once "$variant" 1
  run_once "$variant" 2
done
STOP=1 bash serve/native-flashnext.sh >/dev/null 2>&1
echo "done $(date +%H:%M:%S)" >> "$OUT"
