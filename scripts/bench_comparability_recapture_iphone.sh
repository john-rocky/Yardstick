#!/usr/bin/env bash
#
# bench_comparability_recapture_iphone.sh — re-capture the Gemma-4-E2B iPhone axes on ONE
# stated basis, after the 2026-07-26 audit found the published table was comparing numbers
# that are not comparable.
#
# What the audit found (methodology/comparability-recapture-runbook.md has the full list):
#
#   F1 memory  — the deep-context cell printed LiteRT's *added* memory (92 MB) next to other
#                arms' full footprints; llama.cpp's p=1024 footprint (300 MB) is in fact LOWER
#                than LiteRT's; the "mmap'd weights, not comparable" caveat was applied to
#                llama.cpp and Core AI but not to LiteRT, whose own model card documents that
#                it memory-maps 1.12 GB of embeddings; and our 487 MB sits ~3x under the
#                1,450 MB the card reports for the same device and the same phys_footprint
#                metric.
#   F2 decode  — the decode column mixes instruments: LiteRT-LM and MLX report their own
#                engine counters (host-side detokenize/stream cost excluded), llama.cpp and
#                Core AI are harness wall-clock. The +14% decode crown is measured across
#                that seam.
#   F3 prefill — LiteRT was measured through its own benchmark() entry point forced to 1024,
#                every other arm through this app's long-context task with a real templated
#                prompt (~1,082 tokens).
#
# The app now records both sides of each seam (BenchmarkRunner + MemorySampler, 2026-07-26):
#   decodeTokensPerSecond          engine-reported where the engine exposes counters
#   decodeTokensPerSecondWallClock harness wall-clock, identical basis for every arm
#   memoryPeakDuringDecodeMB       phys_footprint (jetsam basis, excludes clean file pages)
#   memoryPeakResidentMB           resident_size (includes mapped-and-resident pages)
#
# So this run needs no new analysis code on the device side: it just has to produce, for every
# arm, one capture per task with all four fields populated. `scripts/analyze_comparability.py`
# prints the verdicts straight from the pulled JSON.
#
# Usage:  scripts/bench_comparability_recapture_iphone.sh [UDID]
#
set -euo pipefail

UDID="${1:-A6F3E849-1947-5202-9AD1-9C881CA58EEF}"   # DaisukeのiPhone (iPhone 17 Pro)
BUNDLE_ID="${YARDSTICK_BUNDLE_ID:-com.daisukemajima.llmbench}"
TEAM="${YARDSTICK_TEAM:-MFN25KNUGJ}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PROJ="$REPO/ios/BenchmarkApp/BenchmarkApp.xcodeproj"
DD="$HOME/Library/Developer/Xcode/DerivedData/BenchmarkApp-bestquant"
APP="$DD/Build/Products/Release-iphoneos/BenchmarkApp.app"
PULL="/tmp/gemma4-comparability-recapture"

log(){ printf '\n=== %s\n' "$*"; }

if [ ! -d "$REPO/ios/BenchmarkApp/Vendored/Anemll/anemll-swift-cli" ]; then
  log "bootstrap (fetch vendored deps)"
  (cd "$REPO/ios/BenchmarkApp" && ./scripts/bootstrap.sh)
fi

log "build + install"
xcodebuild -project "$PROJ" -scheme BenchmarkApp -configuration Release \
  -destination "generic/platform=iOS" -derivedDataPath "$DD" \
  -skipPackagePluginValidation -skipMacroValidation -allowProvisioningUpdates \
  PRODUCT_BUNDLE_IDENTIFIER="$BUNDLE_ID" \
  DEVELOPMENT_TEAM="$TEAM" CODE_SIGN_STYLE=Automatic build > /tmp/xcb_comparability.log 2>&1 || {
    echo "BUILD FAILED — cause:"; grep -iE "error:" /tmp/xcb_comparability.log | head -5; exit 1; }
xcrun devicectl device uninstall app --device "$UDID" "$BUNDLE_ID" 2>/dev/null || true
xcrun devicectl device install app --device "$UDID" "$APP"

# Models are already resident on the device from the 7/18 campaign, so the sleeps only have to
# cover the run itself. If a model was evicted, the first cell of that arm will come back empty
# — re-run that arm alone rather than lengthening every sleep.
ENGINES=(
  "litert-lm   litert-community/gemma-4-E2B-it-litert-lm"     # the arm every finding is about
  "mlx-swift   mlx-community/gemma-4-e2b-it-4bit"             # PTQ — the decode runner-up (46.4)
  "llama.cpp   unsloth/gemma-4-E2B-it-GGUF/Q4_K_M"            # the arm that is actually smallest at depth
)

# short-chat  → F2 (engine vs wall-clock decode) and F1 (footprint vs resident, small KV)
# long-context-1024 → F1 (LiteRT's real p=1024 footprint, first ever on this instrument),
#                     F3 (LiteRT on the same task and prompt as every other arm),
#                     and the card reconciliation: same model, big KV, compare to 1,450 MB.
for e in "${ENGINES[@]}"; do
  set -- $e
  for run in 1 2 3; do
    log "$1 $2 — short-chat (cold $run/3)"
    xcrun devicectl device process launch --terminate-existing --device "$UDID" "$BUNDLE_ID" -- \
      --yardstick-autorun --runtime "$1" --model-id "$2" --task short-chat --runs 1 >/dev/null
    sleep 120
  done
  for run in 1 2 3; do
    log "$1 $2 — long-context-1024 (cold $run/3)"
    xcrun devicectl device process launch --terminate-existing --device "$UDID" "$BUNDLE_ID" -- \
      --yardstick-autorun --runtime "$1" --model-id "$2" --task long-context-1024 --runs 1 >/dev/null
    sleep 180
  done
done

log "pull results"
rm -rf "$PULL"; mkdir -p "$PULL"
xcrun devicectl device copy from --device "$UDID" --domain-type appDataContainer \
  --domain-identifier "$BUNDLE_ID" --source "Documents/results" --destination "$PULL"

log "verdicts"
python3 "$REPO/scripts/analyze_comparability.py" "$PULL"
echo
echo "raw results in $PULL"
