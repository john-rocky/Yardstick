#!/bin/bash
# 0.14 vs 0.15 Mac GPU speed delta — single instrument (pip CLI `litert-lm benchmark`),
# official gemma-4-E2B-it.litertlm @9262660, ctx 2048, --cache no, GPU backend.
# Protocol mapping of the 2026-07-28 Mac capture: warm n=8 (one launch = built-in warmup
# + 1 measured run, 8 launches per cell), ABAB version interleave per round to cancel
# thermal drift, cooldown + thermal gate between launches, strictly serial.
# NOTE: this is a VERSION-DELTA instrument, not the xcodebuild yardstick — 0.15 has no
# xcframework release assets, so the pip CLI is the only instrument that runs both sides.
set -u
BUNDLE="$HOME/.cache/huggingface/hub/models--litert-community--gemma-4-E2B-it-litert-lm/snapshots/9262660a1676eed6d0c477ab1a86344430854664/gemma-4-E2B-it.litertlm"
OUT="$HOME/code/apple-silicon-llm-bench/results/raw/2026-08-04-litert-015-mac-speed"
CSV="$OUT/summary.csv"
mkdir -p "$OUT"
echo "version,p,d,round,prefill_tps,decode_tps,init_s,ttft_s" > "$CSV"

declare -A CLI=( [014]="$HOME/venvs/lt092run/bin/litert-lm" [015]="$HOME/venvs/lt0150run/bin/litert-lm" )
CONFIGS=("64 256" "256 256" "1024 256")

thermal_gate() {
  # wait until CPU speed limit is back to 100 (nominal); give up after 120 s
  for _ in $(seq 1 24); do
    lim=$(pmset -g therm 2>/dev/null | awk '/CPU_Speed_Limit/ {print $3}')
    [ -z "$lim" ] && return 0
    [ "$lim" = "100" ] && return 0
    sleep 5
  done
}

run_one() {
  local ver=$1 p=$2 d=$3 round=$4
  local log="$OUT/run_${ver}_p${p}d${d}_r${round}.log"
  "${CLI[$ver]}" benchmark "$BUNDLE" -p "$p" -d "$d" --backend gpu --cache no \
      --max-num-tokens 2048 > "$log" 2>&1
  local pf dc init ttft
  pf=$(awk '/Prefill speed:/ {print $3}' "$log")
  dc=$(awk '/Decode speed:/ {print $3}' "$log")
  init=$(awk '/Init time:/ {print $3}' "$log")
  ttft=$(awk '/Time to first token:/ {print $5}' "$log")
  echo "${ver},${p},${d},${round},${pf:-NA},${dc:-NA},${init:-NA},${ttft:-NA}" >> "$CSV"
  echo "[$(date +%H:%M:%S)] $ver p$p d$d r$round -> prefill ${pf:-NA} decode ${dc:-NA}"
  sleep 15
  thermal_gate
}

for round in 1 2 3 4 5 6 7 8; do
  for cfg in "${CONFIGS[@]}"; do
    set -- $cfg
    p=$1; d=$2
    if [ $((round % 2)) -eq 1 ]; then order="014 015"; else order="015 014"; fi
    for ver in $order; do
      run_one "$ver" "$p" "$d" "$round"
    done
  done
done
echo "DONE $(date)"
