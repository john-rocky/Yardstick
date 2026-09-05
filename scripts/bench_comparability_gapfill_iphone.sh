#!/usr/bin/env bash
#
# bench_comparability_gapfill_iphone.sh — finish the 2026-07-26 comparability re-capture.
#
# The first pass (bench_comparability_recapture_iphone.sh, 00:22–01:15) landed the cell the
# whole audit turned on — LiteRT-LM at p=1024, footprint 818 MB against the 92 MB the report
# printed — but three groups came back empty or dirty:
#
#   llama.cpp            0 of 6 cells   (fixed sleeps too short, or the GGUF was evicted and
#                                        the launch went into a 2.6 GB re-download)
#   mlx-swift short-chat 0 of 3 cells   (same)
#   mlx-swift p=1024     3 cells, but captured at thermal `fair` while the phone charged;
#                        LiteRT's were `nominal`, so the cross-arm speed rows are not matched
#   litert-lm short-chat 2 of 3 cells
#
# This pass fills those and nothing else. Two changes from the first script:
#
#   1. A discarded pre-warm launch per arm with a long sleep, so a cold model load or a
#      re-download happens OUTSIDE the timed cells instead of eating one of them.
#   2. No build/install — the app on the device already carries the 2026-07-26 metrics
#      (wall-clock + resident). Set REBUILD=1 if the app was replaced since.
#
# Run it UNPLUGGED with Auto-Lock = Never: the first pass showed MLX entering `fair` while
# charging, which is exactly the thermal regime this pass exists to avoid.
#
set -euo pipefail

UDID="${1:-A6F3E849-1947-5202-9AD1-9C881CA58EEF}"
BUNDLE_ID="${YARDSTICK_BUNDLE_ID:-com.daisukemajima.llmbench}"
TEAM="${YARDSTICK_TEAM:-MFN25KNUGJ}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PROJ="$REPO/ios/BenchmarkApp/BenchmarkApp.xcodeproj"
DD="$HOME/Library/Developer/Xcode/DerivedData/BenchmarkApp-bestquant"
APP="$DD/Build/Products/Release-iphoneos/BenchmarkApp.app"
PULL="/tmp/gemma4-comparability-gapfill"

log(){ printf '\n=== %s\n' "$*"; }

if [ "${REBUILD:-0}" = "1" ]; then
  log "build + install"
  xcodebuild -project "$PROJ" -scheme BenchmarkApp -configuration Release \
    -destination "generic/platform=iOS" -derivedDataPath "$DD" \
    -skipPackagePluginValidation -skipMacroValidation -allowProvisioningUpdates \
    PRODUCT_BUNDLE_IDENTIFIER="$BUNDLE_ID" \
    DEVELOPMENT_TEAM="$TEAM" CODE_SIGN_STYLE=Automatic build > /tmp/xcb_gapfill.log 2>&1 || {
      echo "BUILD FAILED — cause:"; grep -iE "error:" /tmp/xcb_gapfill.log | head -5; exit 1; }
  xcrun devicectl device install app --device "$UDID" "$APP"
fi

run_cell() {  # runtime model task label
  log "$1 — $3 ($4)"
  xcrun devicectl device process launch --terminate-existing --device "$UDID" "$BUNDLE_ID" -- \
    --yardstick-autorun --runtime "$1" --model-id "$2" --task "$3" --runs 1 >/dev/null
}

LITERT="litert-community/gemma-4-E2B-it-litert-lm"
MLX="mlx-community/gemma-4-e2b-it-4bit"
LLAMA="unsloth/gemma-4-E2B-it-GGUF/Q4_K_M"

# --- llama.cpp: nothing landed last pass, so pre-warm generously first ---------------
run_cell "llama.cpp" "$LLAMA" short-chat "pre-warm, discarded"
sleep 420
for r in 1 2 3; do run_cell "llama.cpp" "$LLAMA" short-chat "cold $r/3"; sleep 150; done
for r in 1 2 3; do run_cell "llama.cpp" "$LLAMA" long-context-1024 "cold $r/3"; sleep 210; done

# --- MLX: short-chat missing; p=1024 re-taken at nominal ------------------------------
run_cell "mlx-swift" "$MLX" short-chat "pre-warm, discarded"
sleep 300
for r in 1 2 3; do run_cell "mlx-swift" "$MLX" short-chat "cold $r/3"; sleep 150; done
for r in 1 2 3; do run_cell "mlx-swift" "$MLX" long-context-1024 "cold $r/3 (nominal re-take)"; sleep 210; done

# --- LiteRT: one short-chat cell short of n=3 -----------------------------------------
run_cell "litert-lm" "$LITERT" short-chat "cold 3/3"
sleep 150

log "pull results"
rm -rf "$PULL"; mkdir -p "$PULL"
xcrun devicectl device copy from --device "$UDID" --domain-type appDataContainer \
  --domain-identifier "$BUNDLE_ID" --source "Documents/results" --destination "$PULL"

log "verdicts (all cells on the device, both passes)"
python3 "$REPO/scripts/analyze_comparability.py" "$PULL"
echo
echo "raw results in $PULL"
echo "NOTE: check initialThermalState per cell before using any cross-arm speed row."
