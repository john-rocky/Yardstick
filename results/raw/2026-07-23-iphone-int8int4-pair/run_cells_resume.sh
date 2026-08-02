#!/bin/zsh
# Resume after the 18:28-18:31 CoreDevice tunnel wedge: remaining 7 cells with
# per-cell retry (killall CoreDeviceService + reconnect wait + one retry).
# DeepSeek q8 (18:22, run_ok=4) is already captured; DeepSeek int4 re-runs last.
set -u
DEVICE=A6F3E849-1947-5202-9AD1-9C881CA58EEF
APP=com.example.CoreMLLLMChat
OUT="$(cd "$(dirname "$0")" && pwd)"
CELLS=(
  "litert-community/TinySwallow-1.5B-Instruct"
  "own/TinySwallow-1.5B-int4-BOCTAV4"
  "litert-community/VibeThinker-1.5B"
  "own/VibeThinker-1.5B-int4-BOCTAV4"
  "litert-community/Phi-4-mini-instruct"
  "own/Phi-4-mini-int4-BOCTAV4-128"
  "own/DeepSeek-R1-1.5B-int4-BOCTAV4"
)
wait_tunnel() {
  local n=0
  until xcrun devicectl list devices 2>/dev/null | grep "iPhone18,1" | grep -v simulated | grep -q " connected "; do
    sleep 10; n=$((n+1)); (( n >= 30 )) && { echo "$(date '+%T') TUNNEL TIMEOUT" >> "$OUT/driver.log"; return 1; }
  done
}
run_cell() {  # $1 = catalog id; returns 0 if run_ok=4
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
  if ! run_cell "$c"; then
    echo "$(date '+%T') cell failed -> tunnel recovery + retry" >> "$OUT/driver.log"
    killall CoreDeviceService 2>/dev/null; sleep 10
    wait_tunnel || { echo "=== ABORT: tunnel gone at $c ===" >> "$OUT/driver.log"; exit 1; }
    sleep 60
    run_cell "$c" || echo "=== RETRY ALSO FAILED: $c (continuing) ===" >> "$OUT/driver.log"
  fi
  sleep 180
done
echo "=== $(date '+%F %T') PAIR SESSION DONE ===" | tee -a "$OUT/driver.log"
