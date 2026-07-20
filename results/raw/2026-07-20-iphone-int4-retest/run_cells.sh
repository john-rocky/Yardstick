#!/bin/zsh
# Sequential driver: 4 own-int4 LiteRT cells on iPhone 17 Pro, --runs 4 each
# (cold run1 + warm median r2-4, same methodology as 2026-07-14-iphone-int4).
set -u
DEVICE=A6F3E849-1947-5202-9AD1-9C881CA58EEF
APP=com.example.CoreMLLLMChat
OUT="$(cd "$(dirname "$0")" && pwd)"
MODELS=(
  "own/DeepSeek-R1-1.5B-int4-BOCTAV4"
  "own/TinySwallow-1.5B-int4-BOCTAV4"
  "own/VibeThinker-1.5B-int4-BOCTAV4"
  "own/Phi-4-mini-int4-BOCTAV4-128"
)
for m in "${MODELS[@]}"; do
  fname="${m//\//_}"
  fname="${fname//./_}"
  log="$OUT/console_litert-lm_${fname}.txt"
  echo "=== $(date '+%F %T') START $m ===" >> "$OUT/driver.log"
  pymobiledevice3 diagnostics battery single 2>/dev/null | grep -E '"CurrentCapacity"|"ExternalConnected"' | head -2 >> "$OUT/driver.log"
  gtimeout 7200 xcrun devicectl device process launch --console --terminate-existing \
    --device "$DEVICE" "$APP" -- \
    --yardstick-autorun --runtime litert-lm --model-id "$m" --task short-chat --runs 4 \
    </dev/null > "$log" 2>&1
  rc=$?
  ok=$(grep -c YARDSTICK_RUN_OK "$log")
  echo "=== $(date '+%F %T') END $m exit=$rc run_ok=$ok ===" >> "$OUT/driver.log"
  sleep 60
done
echo "=== $(date '+%F %T') ALL DONE ===" >> "$OUT/driver.log"
