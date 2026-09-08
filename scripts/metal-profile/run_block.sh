#!/bin/zsh
# Interleaved decode block: runs the cells listed in a spec file, ABAB..., with cooldown +
# thermal gate + GPU-quiet gate between launches. Spec lines: <label> <trace:yes|no> <driver args...>
#   run_block.sh <spec> <outdir> <tracedir> <rounds>
# Each round runs every spec line once (label gets _r<N>); "yes" lines are traced in round 1
# only (attach delay 3 s, 6 s window). Runs from the repo root so spec paths can be relative.
set -u
SPEC=$1; OUT=$2; TRACEDIR=$3; ROUNDS=$4
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd); cd "$REPO"
thermal_gate() { for _ in $(seq 1 24); do lim=$(pmset -g therm 2>/dev/null | awk '/CPU_Speed_Limit/ {print $3}'); [ -z "$lim" ] && return 0; [ "$lim" = "100" ] && return 0; sleep 5; done; }
quiet_gate() { for _ in $(seq 1 20); do u=$(ioreg -r -d 1 -c IOAccelerator 2>/dev/null | grep -o '"Device Utilization %"=[0-9]*' | head -1 | tr -dc 0-9); [ "${u:-0}" -le 2 ] && return 0; sleep 3; done; echo "WARN gpu busy before launch: $u"; }
mkdir -p "$OUT"
for r in $(seq 1 $ROUNDS); do
  while IFS= read -r line; do
    [ -z "$line" ] && continue; [[ "$line" = \#* ]] && continue
    label=$(echo "$line" | awk '{print $1}'); trace=$(echo "$line" | awk '{print $2}')
    args=$(echo "$line" | cut -d' ' -f3-)
    lab="${label}_r${r}"
    sleep 15; thermal_gate; quiet_gate
    if [ "$trace" = "yes" ] && [ "$r" = "1" ]; then td="$TRACEDIR"; dl=3; ts=6; else td="-"; dl=0; ts=0; fi
    echo "[$(date +%T)] $lab trace=$td"
    eval "$HERE/profile_cell.sh $lab $OUT $td $dl $ts -- $args --out $OUT/$lab.json --tag $lab --quiet"
  done < "$SPEC"
done
echo "BLOCK DONE $(date +%T)"
