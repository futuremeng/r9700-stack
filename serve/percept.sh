#!/bin/bash
# serve/percept.sh - the CPU perception sidecars (Phase 1 of the cpu-sidecar plan, serve/cpu-sidecars.md):
# speaker diarization (pyannote segmentation-3.0 + CAM++ embedding) and the forced aligner. mmap-first,
# taskset-pinned, no GPU, no PyTorch at run time.
#
# Why this shape (the reasoning lives in serve/cpu-sidecars.md; the short version):
# * The cards carry the generation models and have ~1.35 GiB spare a card at util 0.96, so the sub-15 ms
#   perception models belong on the EPYC, not on a launch reload. Measured on this box: a saturated 0.6B
#   sidecar moved the running launch's ITL by 0.0-0.2 ms and its throughput by 0.3 %
#   (notes/embed-sidecar-box-test.md) -- the same headroom this file spends.
# * CORES: embed.sh owns 16-31 (embed) and 32-47 (rerank), so this file defaults to 48-63 -- one range per
#   serve, because two servers on one range share threads and neither gets the throughput its card says
#   (embed.sh's own note, and the ONNX-thread thrash measured in kokoro-js #361 upstream: default pools
#   size to every core on the box; pin or cap intra-op threads, do not do neither).
# * The 2026-11-01 red line applies here too: no --mlock anywhere, no big anon commits, and the script
#   refuses below MEM_FLOOR GiB of MemAvailable so a bring-up can never compete with the engine's
#   checkpoint stream (that window is swap-I/O bound, notes/native-4xr9700.md C-sections).
#
#   TASK=diarize INPUT=meeting.wav bash serve/percept.sh       # diarized, timestamped transcript
#   TASK=align INPUT=clip.wav TEXT=... bash serve/percept.sh    # word/char timestamps (runtime: pending, DRY shows the plan)
#   STOP=1 bash serve/percept.sh                                # kill any in-flight percept run
#   DRY=1  bash serve/percept.sh TASK=diarize                   # print the pinned command, run nothing
#
# Knobs: TASK (diarize | align), INPUT (audio file), TEXT, MODEL_DIR, SHERPA_BIN, SEG_MODEL, SPK_MODEL,
#        CORES (taskset range, default 48-63), THREADS, MEM_FLOOR (GiB, default 10), LOG.

. "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/lib/localenv.sh"
TASK=${TASK:-diarize}
MODEL_DIR=${MODEL_DIR:-$HOME/models/percept}
INPUT=${INPUT:-}
TEXT=${TEXT:-}

cpus_in() {                                # same range parser as embed.sh: keep the two files honest
  local r lo hi
  for r in $(printf '%s' "$1" | tr ',' ' '); do
    case "$r" in *-*) lo=${r%%-*}; hi=${r##*-};; *) lo=$r; hi=$r;; esac
    seq "$lo" "$hi" 2>/dev/null
  done
}

case "$TASK" in
  diarize)
    # catalog #25 (pyannote segmentation-3.0) and #26 (CAM++) in one binary: sherpa-onnx's diarization path
    # is segmentation + speaker-embedding + clustering, which is what those two rows are.
    SEG_MODEL=${SEG_MODEL:-$MODEL_DIR/segmentation-3.0.onnx}
    SPK_MODEL=${SPK_MODEL:-$MODEL_DIR/campplus.onnx}
    BIN_CANDIDATES=(sherpa-onnx-offline-speaker-diarization)
    PAT='sherpa-onnx-offline-speaker-diarization'
    RUNTIME=pinned
    ;;
  align)
    # Qwen3-ForcedAligner: no CPU runtime is pinned in this repo yet -- the options are an ONNX export on
    # sherpa-onnx (its alignment support first) or the model's own package served in the venv. DRY shows
    # the intended shape; a real run refuses until ALIGN_CMD says what the runtime is.
    BIN_CANDIDATES=()
    PAT='forced-align'
    RUNTIME=pending
    ;;
  *) echo "TASK must be diarize or align (got $TASK)" >&2; exit 1 ;;
esac

CORES=${CORES:-48-63}
ONLINE_LIST=$(cpus_in "$(cat /sys/devices/system/cpu/online 2>/dev/null || seq 0 $(( $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8) - 1 )))")
N_ONLINE=$(cpus_in "$CORES" | grep -Fx -f <(printf '%s\n' "$ONLINE_LIST") 2>/dev/null | wc -l | tr -d ' ')
# a range with nothing online in it -- or a box where the online-list trick finds nothing -- still has to
# answer a thread count: the pinned range's own size is the only safe upper bound there (llama.cpp's
# spinning workers -- see embed.sh's -t measurement). ${N_ONLINE:-8} alone is not the guard: 0 is set.
[ "$N_ONLINE" -lt 1 ] 2>/dev/null && N_ONLINE=$(cpus_in "$CORES" | wc -l | tr -d ' ')
THREADS=${THREADS:-${N_ONLINE:-8}}
MEM_FLOOR=${MEM_FLOOR:-10}
LOG=${LOG:-$HOME/logs/percept-$TASK.log}; mkdir -p "$(dirname "$LOG")"

if [ "${STOP:-0}" = 1 ]; then
  if pkill -f "$PAT"; then echo "stopped the in-flight $TASK run(s)"; else echo "no $TASK run in flight"; fi
  exit 0
fi

# MemAvailable guard: below the floor the answer is "later", not "swap". The floor is a knob because this
# file is also what Phase 3's threshold daemon calls to stop pulling sidecars up under pressure.
avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo 2>/dev/null)
avail_gb=$(( ${avail_kb:-0} / 1024 / 1024 ))
if [ "${DRY:-0}" != 1 ] && [ -n "$avail_kb" ] && [ "$avail_gb" -lt "$MEM_FLOOR" ]; then
  echo "refused: MemAvailable ${avail_gb} GiB < floor ${MEM_FLOOR} GiB -- bring this up after the engine settles" \
    "(that window is the swap-I/O one: notes/native-4xr9700.md)" >&2
  exit 1
fi

BIN=${SHERPA_BIN:-}
[ -z "$BIN" ] && for b in "${BIN_CANDIDATES[@]}"; do BIN=$(command -v "$b" 2>/dev/null) && [ -n "$BIN" ] && break; done

if [ "${DRY:-0}" = 1 ]; then
  echo "task: $TASK (runtime: $RUNTIME)"
  echo "cores: $CORES (${N_ONLINE} online) -- threads $THREADS; memory floor ${MEM_FLOOR} GiB (now ${avail_gb} GiB)"
  case "$TASK" in
    diarize) echo "would run: taskset -c $CORES ${BIN:-sherpa-onnx-offline-speaker-diarization} \
--segmentation-model $SEG_MODEL --embedding-model $SPK_MODEL --num-threads $THREADS ${INPUT:-AUDIO}";;
    align)   echo "would run: taskset -c $CORES ${ALIGN_CMD:-<ALIGN_CMD unset>} --audio ${INPUT:-AUDIO} --text ${TEXT:-TEXT}";;
  esac
  exit 0
fi

if [ "$RUNTIME" = pending ]; then
  echo "TASK=$TASK has no pinned CPU runtime yet -- options in serve/cpu-sidecars.md; set ALIGN_CMD when one is chosen." >&2
  exit 1
fi
[ -z "$BIN" ] && { echo "no sherpa-onnx binary on PATH -- build it once: cmake -B build -DSHERPA_ONNX_ENABLE_... (see serve/cpu-sidecars.md), then SHERPA_BIN=/path bash serve/percept.sh" >&2; exit 1; }
[ -n "$INPUT" ] || { echo "INPUT=<audio.wav> is required for TASK=$TASK (16k mono PCM; ffmpeg -ar 16000 -ac 1 first)" >&2; exit 1; }
[ -f "$INPUT" ] || { echo "INPUT not a file: $INPUT" >&2; exit 1; }
[ -f "$SEG_MODEL" ] || { echo "missing $SEG_MODEL -- fetch the pyannote segmentation-3.0 sherpa export into $MODEL_DIR (serve/cpu-sidecars.md has the checklist)" >&2; exit 1; }
[ -f "$SPK_MODEL" ] || { echo "missing $SPK_MODEL -- fetch the CAM++ sherpa export into $MODEL_DIR" >&2; exit 1; }

echo "launch: taskset -c $CORES $BIN ... --num-threads $THREADS $INPUT  (log: $LOG)"
PINNED=(); [ -n "$CORES" ] && command -v taskset >/dev/null 2>&1 && PINNED=(taskset -c "$CORES")
[ "${#PINNED[@]}" = 0 ] && echo "note: no taskset here (dev box?); running unpinned -- pinning is part of the acceptance on the real box"
# the run journal: the perception layer is CLI-shaped, so the Monitor has no port to poll -- it reads
# this ledger instead (serve/percept-watch.py). One line per run, the audio's length and the wall time
# are what make the RT factor the acceptance row asks for, and the journal lives where launch.json
# lives so STOP/clean contracts stay in one directory.
JOURNAL=${JOURNAL:-$(cd "$(dirname "$0")/.." && pwd)/.r9700-state/percept-runs.jsonl}
audio_s=$(python3 - "$INPUT" <<'PYEOF' 2>/dev/null
import sys, wave
try:
    w = wave.open(sys.argv[1]); print(round(w.getnframes() / w.getframerate(), 2))
except Exception:
    print("null")
PYEOF
) || audio_s=null
[ -n "$audio_s" ] || audio_s=null
t0=$SECONDS
"${PINNED[@]}" "$BIN" \
  --segmentation-model "$SEG_MODEL" --embedding-model "$SPK_MODEL" \
  --num-threads "$THREADS" "$INPUT" >>"$LOG" 2>&1
rc=$?
wall=$(( SECONDS - t0 ))
rt=$(awk -v a="${audio_s}" -v w="$wall" 'BEGIN { if (a != "null" && w > 0) printf "%.1f", a / w; else print "null" }')
mkdir -p "$(dirname "$JOURNAL")" 2>/dev/null
printf '{"ts": %s, "task": "%s", "audio_s": %s, "wall_s": %s, "rt": %s, "rc": %s, "threads": %s, "cores": "%s", "input": "%s"}\n' \
  "$(date +%s)" "$TASK" "${audio_s:-null}" "$wall" "$rt" "$rc" "$THREADS" "$CORES" "$INPUT" >>"$JOURNAL" 2>/dev/null
[ "$rc" = 0 ] && echo "done: $INPUT diarized (threads $THREADS on cores $CORES, ${audio_s:-?} s audio in ${wall}s; journal + output in $LOG)" \
              || { echo "rc=$rc, see $LOG" >&2; tail -5 "$LOG" >&2; exit "$rc"; }
