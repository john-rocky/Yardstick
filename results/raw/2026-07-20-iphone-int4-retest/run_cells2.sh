#!/bin/zsh
# Retest pass 2: the 3 cells that started thermal=fair in pass 1.
# Temp-gated: wait for battery skin temp <= 30C (max 30 min) before each cell.
set -u
DEVICE=A6F3E849-1947-5202-9AD1-9C881CA58EEF
APP=com.example.CoreMLLLMChat
OUT="$(cd "$(dirname "$0")" && pwd)"
MODELS=(
  "own/TinySwallow-1.5B-int4-BOCTAV4"
  "own/VibeThinker-1.5B-int4-BOCTAV4"
  "own/Phi-4-mini-int4-BOCTAV4-128"
)
cool_wait() {
  local waited=0
  while (( waited < 1800 )); do
    local t=$(pymobiledevice3 diagnostics battery single 2>/dev/null | grep '"AverageBattVirtualTemp"' | head -1 | tr -dc '0-9')
    echo "$(date '+%T') temp=${t:-?}C" >> "$OUT/driver2.log"
    if [[ -n "$t" && "$t" -le 30 ]]; then return 0; fi
    sleep 120; waited=$((waited+120))
  done
  echo "$(date '+%T') COOL_TIMEOUT (proceeding anyway)" >> "$OUT/driver2.log"
}
for m in "${MODELS[@]}"; do
  fname="${m//\//_}"
  fname="${fname//./_}"
  log="$OUT/console2_litert-lm_${fname}.txt"
  cool_wait
  echo "=== $(date '+%F %T') START $m ===" >> "$OUT/driver2.log"
  gtimeout 7200 xcrun devicectl device process launch --console --terminate-existing \
    --device "$DEVICE" "$APP" -- \
    --yardstick-autorun --runtime litert-lm --model-id "$m" --task short-chat --runs 4 \
    </dev/null > "$log" 2>&1
  rc=$?
  ok=$(grep -c YARDSTICK_RUN_OK "$log")
  echo "=== $(date '+%F %T') END $m exit=$rc run_ok=$ok ===" >> "$OUT/driver2.log"
done
echo "=== $(date '+%F %T') PASS2 DONE ===" >> "$OUT/driver2.log"
