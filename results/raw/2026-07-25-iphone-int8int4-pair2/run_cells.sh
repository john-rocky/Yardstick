#!/bin/zsh
# Pair session 2 (2026-07-25): the two pairs left incomplete on 07-23 — q8 side
# re-captured TODAY so each ratio stays intra-session (07-23 partners are stale
# for pairing; each pair must be same-session). No pushes needed. No killall on
# retry (07-23 lesson). Waits for unplug + 300 s settle before cells.
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
echo "session $(date '+%F %T') pair2 (DeepSeek + Phi pairs, q8 re-captured same-day)" > "$OUT/driver.log"
echo "--- waiting for UNPLUG (set Auto-Lock=Never first, then pull the cable) ---" | tee -a "$OUT/driver.log"
while pymobiledevice3 diagnostics battery single 2>/dev/null | grep -q '"ExternalConnected": true'; do sleep 20; done
echo "$(date '+%T') unplugged; settling 300s" | tee -a "$OUT/driver.log"
sleep 300
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
echo "=== $(date '+%F %T') PAIR2 SESSION DONE ===" | tee -a "$OUT/driver.log"
