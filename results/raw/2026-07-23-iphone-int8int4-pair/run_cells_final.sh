#!/bin/zsh
# Final 2 cells (Phi int4 + DeepSeek int4 re-run). Retry WITHOUT killall —
# killing CoreDeviceService drops the WiFi tunnel for >5 min (19:26 lesson).
set -u
DEVICE=A6F3E849-1947-5202-9AD1-9C881CA58EEF
APP=com.example.CoreMLLLMChat
OUT="$(cd "$(dirname "$0")" && pwd)"
CELLS=(
  "own/Phi-4-mini-int4-BOCTAV4-128"
  "own/DeepSeek-R1-1.5B-int4-BOCTAV4"
)
run_cell() {
  local fname="${1//\//_}"; fname="${fname//./_}"
  local log="$OUT/console_litert-lm_${fname}.txt"
  echo "=== $(date '+%F %T') START $1 ===" >> "$OUT/driver.log"
  gtimeout 900 xcrun devicectl device process launch --console --terminate-existing \
    --device "$DEVICE" "$APP" -- \
    --yardstick-autorun --runtime litert-lm --model-id "$1" --task short-chat --runs 4 \
    </dev/null > "$log" 2>&1
  local rc=$?
  local ok=$(grep -c YARDSTICK_RUN_OK "$log")
  echo "=== $(date '+%F %T') END $1 exit=$rc run_ok=$ok ===" >> "$OUT/driver.log"
  [[ "$ok" == "4" ]]
}
for c in "${CELLS[@]}"; do
  ok=0
  for attempt in 1 2 3; do
    run_cell "$c" && { ok=1; break; }
    echo "$(date '+%T') attempt $attempt failed for $c; waiting 60s (no killall)" >> "$OUT/driver.log"
    sleep 60
  done
  (( ok )) || echo "=== GAVE UP: $c after 3 attempts ===" >> "$OUT/driver.log"
  sleep 180
done
echo "=== $(date '+%F %T') PAIR SESSION DONE ===" | tee -a "$OUT/driver.log"
