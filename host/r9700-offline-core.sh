#!/bin/bash
# Take ONE physical core offline -- the core whose L2 is producing the fatal machine checks -- without
# letting anyone offline the wrong pair by accident. Needs root.
#
#   sudo host/r9700-offline-core.sh              # the pair in host/offline-core.cpus (cpu17 + cpu113)
#   sudo host/r9700-offline-core.sh 17 113       # or spelled out
#
# What it refuses, and why:
#   - a pair that is not the two SMT threads of one core (thread_siblings_list). Offlining "two CPUs" that turn
#     out to be two different cores takes twice the compute and, worse, hides which core is actually sick.
#   - cpu0, which the kernel will not offline anyway.
#   - a core whose sysfs topology is already gone: that means it is offline (idempotent, reported) or already
#     reset -- and a reset brings every core back, which is why the .service unit exists next to this file.
# Before touching anything it snapshots the topology into .r9700-state/cpu-topology.json, because sysfs drops
# a cpu's topology directory the moment it goes offline: without the snapshot, tools/journal-pcie.py can no
# longer say that the two cpu numbers in the MCA records are one physical core.

set -u
D="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$D/.." && pwd)"
CONF="$D/offline-core.cpus"
CPUS=("$@")
if [ ${#CPUS[@]} -eq 0 ] && [ -f "$CONF" ]; then
  read -r -a CPUS < <(grep -v '^#' "$CONF" | grep -v '=' | head -1)
fi
[ ${#CPUS[@]} -ge 1 ] || { echo "no cpus given and $CONF has none" >&2; exit 1; }
[ "$(id -u)" = 0 ] || { echo "needs root: sudo host/r9700-offline-core.sh ${CPUS[*]}" >&2; exit 1; }

# 1. snapshot while still online
"$REPO/.venv/bin/python" - "$REPO/.r9700-state/cpu-topology.json" <<'PY' 2>/dev/null || true
import json, sys
from pathlib import Path
out, p = {}, Path(sys.argv[1])
for d in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*")):
    n = d.name[3:]
    if not n.isdigit() or not (d / "online").exists():
        continue
    t = d / "topology"
    try:
        out[n] = {"core": (t / "core_id").read_text().strip(),
                  "package": (t / "physical_package_id").read_text().strip(),
                  "siblings": (t / "thread_siblings_list").read_text().strip()}
    except OSError:
        pass
if out:
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(out))
    print(f"topology snapshot: {len(out)} cpu(s) -> {p}")
PY

# 2. verify the pair is one core
SIB=""
for c in "${CPUS[@]}"; do
  [ "$c" = 0 ] && { echo "cpu0 cannot be offlined" >&2; exit 1; }
  f=/sys/devices/system/cpu/cpu$c/online
  [ -f "$f" ] || { echo "cpu$c: no such cpu" >&2; exit 1; }
  s=$(cat /sys/devices/system/cpu/cpu$c/topology/thread_siblings_list 2>/dev/null)
  [ -n "$s" ] || { echo "cpu$c: topology already gone (offline?) -- nothing to do" >&2; exit 0; }
  [ -z "$SIB" ] && SIB="$s"
  [ "$s" = "$SIB" ] || { echo "refusing: cpu$c siblings $s, but the pair given shares $SIB -- those are
different cores. The MCA records name ONE core (see notes/incident-2026-11-03)." >&2; exit 1; }
done
CORE=$(cat /sys/devices/system/cpu/cpu${CPUS[0]}/topology/core_id 2>/dev/null)
PKG=$(cat /sys/devices/system/cpu/cpu${CPUS[0]}/topology/physical_package_id 2>/dev/null)
L2=$(cat /sys/devices/system/cpu/cpu${CPUS[0]}/cache/index2/size 2>/dev/null)
echo "cpu ${CPUS[*]} = core $CORE of package $PKG (siblings $SIB, L2 ${L2:-?})"

# 3. offline the whole sibling set, so the scheduler has no thread of that core left to land on
CHANGED=0
for c in ${SIB//,/ }; do
  f=/sys/devices/system/cpu/cpu$c/online
  [ -f "$f" ] || continue
  if [ "$(cat $f)" = 1 ]; then echo 0 > "$f" && CHANGED=1; fi
  printf "  cpu%-5s online=%s\n" "$c" "$(cat $f)"
done
[ $CHANGED = 1 ] && echo "core $CORE taken out of use: $(grep -c processor /proc/cpuinfo) of $(nproc --all) threads online" \
                 || echo "already offline (nothing changed)"
echo "This removes the failing block from use; it does not repair it. Keep the pairing in $CONF and the unit"
echo "host/r9700-offline-core.service enabled, or a reset puts the sick core back in service."
