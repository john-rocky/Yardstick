#!/bin/zsh
# Run one decode cell and (optionally) attach a Metal System Trace mid-decode.
#
#   profile_cell.sh <label> <outdir> <tracedir|-> <delay_s> <trace_s> -- <driver command...>
#
# The driver must print `PID <pid>` then `DECODE_START` (litert_decode_driver.py /
# mlx_decode_driver.py do). With tracedir "-" no trace is taken (plain timing run).
# Writes <outdir>/<label>.log (driver stdout+stderr) and <outdir>/<label>.env.txt
# (contamination gate: GPU utilization, top CPU consumers, thermal state, power).
# The .trace bundle (~1-2 GB) and the gpu-intervals XML export go to <tracedir>.
set -u
LABEL=$1; OUT=$2; TRACEDIR=$3; DELAY=$4; TRACE_S=$5; shift 5
[ "$1" = "--" ] && shift
mkdir -p "$OUT"; [ "$TRACEDIR" != "-" ] && mkdir -p "$TRACEDIR"
LOG="$OUT/$LABEL.log"; ENV="$OUT/$LABEL.env.txt"
{
  echo "label $LABEL"; echo "date $(date -u +%FT%TZ) epoch $(date +%s)"
  echo "gpu_util_before $(ioreg -r -d 1 -c IOAccelerator 2>/dev/null | grep -o '"Device Utilization %"=[0-9]*' | head -1)"
  echo "top_cpu_before:"; ps -Ao pcpu,pid,comm -r | head -6
  echo "therm: $(pmset -g therm 2>/dev/null | grep -E 'CPU_Speed_Limit' | tr -s ' ')"
  echo "power: $(pmset -g batt | head -1)"
  echo "cmd: $*"
} > "$ENV"
"$@" > "$LOG" 2>&1 &
CMDPID=$!
# wait for DECODE_START (max 600 s: engine init + prefill)
for i in $(seq 1 1200); do grep -q '^DECODE_START' "$LOG" 2>/dev/null && break; kill -0 $CMDPID 2>/dev/null || break; sleep 0.5; done
DRVPID=$(grep -m1 '^PID ' "$LOG" | awk '{print $2}')
if [ "$TRACEDIR" != "-" ] && grep -q '^DECODE_START' "$LOG"; then
  sleep "$DELAY"
  echo "trace_attach_epoch $(date +%s) pid ${DRVPID:-$CMDPID}" >> "$ENV"
  xcrun xctrace record --template 'Metal System Trace' --time-limit "${TRACE_S}s" \
    --output "$TRACEDIR/$LABEL.trace" --attach "${DRVPID:-$CMDPID}" >> "$ENV" 2>&1
  echo "trace_done_epoch $(date +%s)" >> "$ENV"
fi
wait $CMDPID; RC=$?
echo "exit $RC" >> "$ENV"
echo "gpu_util_after $(ioreg -r -d 1 -c IOAccelerator 2>/dev/null | grep -o '"Device Utilization %"=[0-9]*' | head -1)" >> "$ENV"
grep -E '^RESULT' "$LOG" | tail -1
if [ "$TRACEDIR" != "-" ] && [ -d "$TRACEDIR/$LABEL.trace" ]; then
  xcrun xctrace export --input "$TRACEDIR/$LABEL.trace" \
    --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]' \
    --output "$TRACEDIR/$LABEL.gpu.xml" >> "$ENV" 2>&1 && echo "exported $TRACEDIR/$LABEL.gpu.xml ($(du -h "$TRACEDIR/$LABEL.gpu.xml" | cut -f1))"
fi
exit $RC
