#!/usr/bin/env bash
# iPhone battery-delta energy — Gemma-4-E2B part 2 ("does wNa8o8's int8 activation path
# also win J/token on its native path?"). Companion to the Mac decode-window table.
#
# Device prep (all three required — the run is invalid otherwise):
#   1. UNPLUG the phone (devicectl drives it over Wi-Fi; pairing must already exist).
#   2. Battery <= 90% at start (95%+ sticks at the same percent for a long time -> 0 delta).
#   3. Settings > Display & Brightness > Auto-Lock = Never, phone unlocked.
#
# Three arms x 600 s sustained decode (~10 min each, ~10% battery each) + 300 s cooldowns:
# ~1 h total, ~30-35% battery. The app measures via the 1%-step battery API and reports
# J/token + average W + tokens/Wh in the result JSON (energySource "battery-1pct").
set -uo pipefail

UDID="${1:-A6F3E849-1947-5202-9AD1-9C881CA58EEF}"
BUNDLE_ID="${YARDSTICK_BUNDLE_ID:-com.daisukemajima.llmbench}"
SUSTAIN="${SUSTAIN_SECONDS:-600}"
PULL="/tmp/gemma4-energy-part2"
REPO_RAW="$HOME/code/apple-silicon-llm-bench/results/raw/2026-07-18-gemma4-bestquant"

log(){ printf '\n=== %s\n' "$*"; }

ARMS=(
  "litert-lm  litert-community/gemma-4-E2B-it-litert-lm"    # wNa8o8, native path — the question
  "mlx-swift  mlx-community/gemma-4-e2b-it-4bit"            # PTQ, the Mac efficiency king
  "core-ai    core-ai/gemma4-e2b-gpu"                       # patched engine (reference)
)

for a in "${ARMS[@]}"; do
  set -- $a
  log "$1 energy (sustain ${SUSTAIN}s — leave the phone alone, screen on, unplugged)"
  xcrun devicectl device process launch --terminate-existing --device "$UDID" "$BUNDLE_ID" -- \
    --yardstick-autorun --runtime "$1" --model-id "$2" --task energy \
    --sustain-seconds "$SUSTAIN" --runs 1 >/dev/null
  sleep $((SUSTAIN + 150))
  log "cooldown 300s"
  sleep 300
done

log "pull + import"
rm -rf "$PULL"; mkdir -p "$PULL"
xcrun devicectl device copy from --device "$UDID" --domain-type appDataContainer \
  --domain-identifier "$BUNDLE_ID" --source "Documents/results" --destination "$PULL" >/dev/null 2>&1
cp "$PULL"/*energy*.json "$REPO_RAW/" 2>/dev/null
find "$PULL" -name "*energy*.json" | sed 's/.*\//  /'
echo "verify each record: energySource=battery-1pct and joules != nil (plugged/charging runs report nil)"
