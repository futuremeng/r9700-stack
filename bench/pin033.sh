#!/bin/bash
# One arm of the native 0.32-vs-0.33 comparison this box owes (notes/vllm-0.33.md, "On this box: the comparison
# to run"). One launch, one log directory, the gates in the order the note sets them, printed as it goes.
#
#   usage: LABEL=A VENV=$HOME/venv-r9700      CALIBRATE=0 STEPS=full  bash bench/pin033.sh
#          LABEL=B VENV=$HOME/venv-r9700-033  CALIBRATE=0 STEPS=full  bash bench/pin033.sh
#          LABEL=Bc VENV=$HOME/venv-r9700-033 CALIBRATE=1 STEPS=probe bash bench/pin033.sh
#
#   LABEL      name of the arm; everything lands in ~/logs/pin033-LABEL/
#   VENV       the vLLM under test (the plugin tree is shared by every arm -- it is this checkout, editable-installed
#              in each venv, which is why the launch-time collective calibration has to be pinned by CALIBRATE and
#              not assumed: the 0.32 baseline numbers predate it)
#   CALIBRATE  R9K_ARN_CALIBRATE (comm/calibrate.py). 0 = the fixed PLX cutoffs, which is what every recorded
#              0.32 number was taken with; 1 = measure them at launch. An arm pair is only comparable at one value.
#   STEPS      probe  = gate + launch + probe32 + itl                      (~15 min)
#              quick  = probe + the retrieval ladder + GSM8K/needle gate   (~30 min)
#              full   = quick + sanity_stress at NSEQ+1/1.5x/2x and with long prefills first + a 1500 s soak
#
#   PC         PREFIX_CACHE (the launcher's default is 1). The bands this comparison is measured against --
#              1,576 / 1,816 / 1,137 and the retrieval ladder at 8.6k / 8.0k / 7.4k tok/s -- were all taken with
#              PC=0, so the arms run PC=0 unless someone changes it on both sides at once.
#
# Why the gate is the first line and not an afterthought: stopping a native launch signals the ranks, and relaunching
# before their ~50 GiB of anon is really unmapped is what wedged this box (notes/load-oom-svm-wedge.md). bench/gate.sh
# is the verified stop, and it runs the pkill patterns from a file so they never match this caller's own argv.
#
# Why the CLIENT is not $VENV's python: the client is part of the protocol, so it stays the production venv's on
# every arm. Only the server changes between arms.
set -uo pipefail
cd /home/jiulu/r9700-stack || exit 1

LABEL=${LABEL:?LABEL=<arm name> is required}
VENV=${VENV:-$HOME/venv-r9700}
CACHE=${VLLM_CACHE_ROOT:-$HOME/vllm-native-cache}
TRITON=${TRITON_CACHE_DIR:-$HOME/vllm-native-triton}
CAL=${CALIBRATE:-0}
PC=${PC:-0}
STEPS=${STEPS:-full}
CLIENT=${CLIENT:-$HOME/venv-r9700/bin/python3}
# The ROCm vLLM wheels link libmpi_cxx, which this box has unpacked rather than installed system-wide. This is
# serve/native-flashnext.sh's OPENMPI= line, and the launcher's LD_LIBRARY_PATH is inherited by the arm: without it
# every gate fails with "libmpi_cxx.so.40: cannot open shared object file" and looks like a broken venv instead of a
# missing path (the exact mistake setup.py's own import test made once).
OPENMPI=${OPENMPI:-$HOME/opt/openmpi/usr/lib/x86_64-linux-gnu}
[ -d "$OPENMPI" ] && export LD_LIBRARY_PATH=$OPENMPI${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
BASE=${BASE:-http://localhost:8080}
TIERS=${TIERS:-60000,120000,180000}
D=$HOME/logs/pin033-$LABEL
mkdir -p "$D"
log () { echo "[$(date +%H:%M:%S)] $*"; }

[ -x "$VENV/bin/python3" ] || { echo "no venv at $VENV"; exit 2; }
[ -x "$CLIENT" ] || { echo "no client python at $CLIENT"; exit 2; }

# What this arm actually runs, recorded before anything else: the version string is what compat/gate.py compares
# its TESTED lists against, so the arm that passes a gate has to be nameable afterwards.
"$VENV/bin/python3" - <<'PY' | tee "$D/vllm.txt"
import vllm, torch, os
print("venv:", os.path.dirname(os.path.dirname(vllm.__file__)))
print("vllm:", vllm.__version__, "| torch:", torch.__version__, "| hip:", getattr(torch.version, "hip", "?"))
try:
    import importlib.metadata as md
    for p in ("amd-aiter", "triton", "orjson"):
        print(f"{p}:", md.version(p))
except Exception as e:
    print("deps:", e)
PY

log "gate: verified stop of whatever is up, then the memory gate (bench/gate.sh)"
if ! bash bench/gate.sh > "$D/gate.log" 2>&1; then
  echo "gate: FAIL -- refusing to launch on a box that has not given its memory back"; tail -4 "$D/gate.log"; exit 3
fi
tail -2 "$D/gate.log"

log "launch: VENV=$VENV VLLM_CACHE_ROOT=$CACHE TRITON_CACHE_DIR=$TRITON R9K_ARN_CALIBRATE=$CAL STEPS=$STEPS"
VENV=$VENV VLLM_CACHE_ROOT=$CACHE TRITON_CACHE_DIR=$TRITON \
  MAXLEN=262144 NSEQ=32 NBT=8192 WEBUI=same PREFIX_CACHE=$PC R9K_ARN_CALIBRATE=$CAL \
  setsid bash serve/native-flashnext.sh > "$D/launch.log" 2>&1 &
UP=0
for _ in $(seq 1 90); do
  sleep 10
  curl -s -m 5 -o /dev/null "$BASE/health" && { UP=1; break; }
  grep -qE "Traceback|ENGINE DEAD|raise .*Error" "$D/launch.log" && { echo "launch is failing -- see $D/launch.log"; exit 4; }
done
[ "$UP" = 1 ] || { echo "health never came up -- see $D/launch.log"; exit 4; }

# The fingerprints that make two arms comparable (the same set bench/relaunch.sh prints), plus the two numbers this
# comparison exists for: did the KV pool move, and did the collective calibration run.
{ echo "== fingerprints ($LABEL, vllm $("$VENV/bin/python3" -c 'import vllm;print(vllm.__version__)')) =="
  grep -m1 -oE "Model loading took [0-9.]+ GiB memory and [0-9.]+ seconds" "$D/launch.log" | sed 's/^/  load: /'
  grep -m1 -oE "GPU KV cache size: [0-9,]+ tokens, Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" "$D/launch.log" | sed 's/^/  kv: /'
  grep -m1 -oE "Available KV cache memory: [0-9.]+ GiB" "$D/launch.log" | sed 's/^/  /'
  grep -m1 "load profile (target)" "$D/launch.log" | sed 's/.*profile (target): /  place: /'
  grep -m1 -oE "'speculative_config': \{[^}]*\}" "$D/launch.log" | sed 's/^/  spec: /'
  grep -m1 -oE "max_model_len=[0-9]+" "$D/launch.log" | head -1 | sed 's/^/  /'
  grep -m1 "collective calibration" "$D/launch.log" | sed 's/^/  calibrate: /' || echo "  calibrate: (no line -- fixed cutoffs)"
  grep -c "untested on vLLM" "$D/launch.log" | sed 's/^/  gate.py warnings (untested patches): /'
  # Which gate kernel this arm actually ran, per launch-methodology.md section 12: grep ALL the install lines (the
  # first pass is the target model's 48 gates, the second the MTP draft's 1), and read the cap off the live process
  # rather than trusting that a default travelled.
  grep -o "r9k router GEMM installed.*" "$D/launch.log" | sort | uniq -c | sed 's/^/  router: /'
  _pid=$(pgrep -f 'vll[m] serve' | head -1); tr '\0' '\n' < /proc/$_pid/environ 2>/dev/null | grep -E '^R9K_ROUTER_MAX_M=' | sed 's/^/  env: /'
} | tee "$D/fingerprints.txt"

# A launch that dies mid-arm makes every later row a lie, and on this box that is not hypothetical: the first arm of
# this comparison took probe32 + itl + the ladder clean and then died at 11:34:59 inside GSM8K with
# HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION (the family in notes/exl3-drafter-heisenbug.md), and the script kept
# going. It stops now, and keeps the crash with the arm's logs.
alive () {   # every phase ends with this; a dead serve ends the arm, it does not become a row of zeros
  if ! curl -s -m 5 -o /dev/null "$BASE/health"; then
    echo "SERVE IS DOWN after $1 -- the arm stops here. Last device errors:"
    grep -m4 -E 'APERTURE_VIOLATION|illegal memory access|Queue error|EngineCore encountered' "$D/launch.log" | sed 's/^/  /'
    grep -m1 -B2 -A2 'GPU HANG ANALYSIS' "$D/launch.log" | sed 's/^/  /'
    cp "$D/launch.log" "$D/launch.dead.log" 2>/dev/null
    echo "  full log kept: $D/launch.dead.log"
    exit 5
  fi
}

log "probe32: 32x300 x3, 32x1200 x2, 16x300 x3 (the recorded 0.32 band is 1,576 / 1,816 / 1,137)"
PROBE_OUT="$D/probe32.txt" bash bench/probe32.sh > "$D/probe32.out" 2>&1
sed 's/^/  /' "$D/probe32.txt"

alive "itl"
log "itl: median inter-token latency at 1 / 2 / 4 in flight"
timeout 900 "$CLIENT" bench/itl.py 1 2 4 > "$D/itl.txt" 2>&1; sed 's/^/  /' "$D/itl.txt"
[ "$STEPS" = probe ] && { log "done (STEPS=probe) -- launch left up; stop it with bench/gate.sh"; exit 0; }

alive "retrieval ladder"
log "retrieval ladder: $TIERS-token prompts with a needle at each tier (bounds #60084's new ptr_range assertion)"
timeout 2400 "$CLIENT" bench/needle_long.py "$TIERS" "$BASE" > "$D/needle.txt" 2>&1
grep -E 'tokens|found|prefill|TTFT|reuse' "$D/needle.txt" | head -24 | sed 's/^/  /'

alive "quality gate"
log "quality gate: GSM8K 100 at conc 8 + needles"
timeout 1800 "$CLIENT" bench/quality.py 100 8 > "$D/quality.txt" 2>&1
tail -8 "$D/quality.txt" | sed 's/^/  /'
[ "$STEPS" = quick ] && { log "done (STEPS=quick) -- launch left up; stop it with bench/gate.sh"; exit 0; }

log "sanity_stress: strict sanity at NSEQ+1 / 1.5x / 2x in flight, then long prefills first"
for c in 33 48 64; do
  timeout 3600 "$CLIENT" bench/sanity_stress.py "$BASE" --pre "" --conc "$c" --rounds 60 > "$D/sanity-$c.txt" 2>&1   # 60 rounds = the 1,980 / 2,880 / 3,840-request protocol the 0.33 note records
  alive "sanity_stress conc $c"
  log "  conc $c: $(grep -oE '[0-9]+ bad[^\n]*' "$D/sanity-$c.txt" | tail -1) ($(grep -coE 'round|ok' "$D/sanity-$c.txt") lines)"
  tail -3 "$D/sanity-$c.txt" | sed 's/^/    /'
done
timeout 3600 "$CLIENT" bench/sanity_stress.py "$BASE" --pre docs,bb --depth 24000 --conc 33 --rounds 10 > "$D/sanity-long.txt" 2>&1
tail -3 "$D/sanity-long.txt" | sed 's/^/    /'

alive "soak"
log "soak: 16 clients, 200-30k tokens, 1500 s (with per-card VRAM sampled alongside -- the leak check)"
touch "$D/soak.run"   # before the sampler starts: the loop's condition is this file, so touching it after means
                      # the sampler exits on its first test and the VRAM series comes back empty (it did, on arm A1)
( while [ -f "$D/soak.run" ]; do
    printf '%s ' "$(date +%H:%M:%S)" >> "$D/vram.txt"
    rocm-smi --showmeminfo vram 2>/dev/null | awk -F': ' '/Used Memory/{printf "%.2f ", $NF/1073741824}' >> "$D/vram.txt"
    echo >> "$D/vram.txt"; sleep 15
  done ) &
SAMPLER=$!
timeout 2000 "$CLIENT" bench/soak.py "$BASE" --conc 16 --seconds 1500 --lo 200 --hi 30000 > "$D/soak.txt" 2>&1
rm -f "$D/soak.run"; wait "$SAMPLER" 2>/dev/null   # NOT a bare wait: a bare wait also waits for the setsid'd
                       # server job, which never exits -- arm A2 hung here for 20 min with the soak already over
tail -12 "$D/soak.txt" | sed 's/^/  /'
{ echo "  VRAM samples: $(grep -c . "$D/vram.txt")"; awk 'NR>0{for(i=2;i<=NF;i++) if($i+0>m[i])m[i]=$i+0} END{printf "  high-water a card:"; for(i=2;i<=10;i++) if(m[i]) printf " %.2f", m[i]; print ""}' "$D/vram.txt"; } | tee -a "$D/fingerprints.txt"

log "arm $LABEL done -- launch left up; stop it with: bash bench/gate.sh"
