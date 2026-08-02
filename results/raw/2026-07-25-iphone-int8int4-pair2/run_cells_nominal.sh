#!/bin/zsh
# Nominal re-take of the DeepSeek + Phi pairs: 25 min cooldown first (the 04:01
# pass started thermal=fair — 300 s post-charge settle was not enough), then the
# 4 cells with 240 s gaps. No killall on retry.
set -u
DEVICE=A6F3E849-1947-5202-9AD1-9C881CA58EEF
APP=com.example.CoreMLLLMChat
OUT="$(cd "$(dirname "$0")" && pwd)"
CELLS=(
  "litert-community/DeepSeek-R1-Distill-Qwen-1.5B"
  "own/DeepSeek-R1-1.5B-int4-BOCTAV4"
  "litert-community/Phi-4-mini-instruct"
  "own/Phi-4-mini-int4-BOCTAV4-128"
)
run_cell() {
  local fname="${1//\//_}"; fname="${fname//./_}"
  local log="$OUT/console2_litert-lm_${fname}.txt"
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
echo "--- nominal retake: cooling 1500s ---" | tee -a "$OUT/driver.log"
sleep 1500
for c in "${CELLS[@]}"; do
  ok=0
  for attempt in 1 2 3; do
    run_cell "$c" && { ok=1; break; }
    echo "$(date '+%T') attempt $attempt failed for $c; waiting 60s (no killall)" >> "$OUT/driver.log"
    sleep 60
  done
  (( ok )) || echo "=== GAVE UP: $c ===" >> "$OUT/driver.log"
  sleep 240
done
echo "=== $(date '+%F %T') NOMINAL RETAKE DONE ===" | tee -a "$OUT/driver.log"
