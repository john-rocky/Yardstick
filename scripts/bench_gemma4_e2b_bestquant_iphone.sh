#!/usr/bin/env bash
#
# bench_gemma4_e2b_bestquant_iphone.sh — Gemma-4-E2B on iPhone 17 Pro, EVERY ARM AT ITS BEST
# AVAILABLE QUANTIZATION, with the quantization stated per row.
#
# Why this script exists
# ---------------------
# The published Gemma-4-E2B table crowned LiteRT-LM on decode and memory. Audited 2026-07-17,
# the four arms were on four different checkpoints — and two of them were NOT at their best:
#
#   litert-lm   litert-community/gemma-4-E2B-it-litert-lm   wNa8o8 QAT   <- its best
#   mlx-swift   mlx-community/gemma-4-e2b-it-4bit           PTQ          <- NOT its best
#   llama.cpp   unsloth/gemma-4-E2B-it-GGUF Q4_K_M          PTQ          <- NOT its best
#   coreml-llm  mlboydaisuke/gemma-4-E2B-coreml             INT4 palett.
#
# So the trophy was measuring who used the better checkpoint, not which runtime is faster.
# Google ships an official QAT build for every one of these ecosystems; this script uses them.
# Measured on M4 Max, the PTQ->QAT swap alone is worth 9 points of GSM8K (78% -> 87%).
#
# Matching the WEIGHTS across arms is not achievable and we stopped trying: LiteRT's wNa8o8 is a
# co-designed weights+runtime package (2-bit decode layers, optimized KV cache, static int8
# activations) that no fp16-activation runtime can execute properly — the same weights score
# 85% on LiteRT and 48% elsewhere. "Each arm at its best, stated explicitly" is the comparison
# that can actually be made, and it is the one a user choosing a runtime today wants.
#
# Build artifacts FIRST (Mac, no device needed):
#   Core AI : scripts/export_coreai_gemma4.sh e2b   (or reuse exports/gemma4_e2b_qat_decode_int4lin_tbl)
# Everything else is pulled from HF by the app on-device.
#
set -euo pipefail

UDID="${1:-A6F3E849-1947-5202-9AD1-9C881CA58EEF}"   # DaisukeのiPhone (iPhone 17 Pro)
BUNDLE_ID="com.iosllmbenchmark.benchmarkapp"; TEAM="MFN25KNUGJ"; DEVICE="iphone17pro"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PROJ="$REPO/ios/BenchmarkApp/BenchmarkApp.xcodeproj"
DD="$HOME/Library/Developer/Xcode/DerivedData/BenchmarkApp-coreai"
APP="$DD/Build/Products/Release-iphoneos/BenchmarkApp.app"
PULL="/tmp/gemma4-e2b-bestquant-results"

log(){ printf '\n=== %s\n' "$*"; }

# Vendored/ (llama.xcframework + Anemll) is git-ignored and fetched on demand, so a fresh
# clone fails package resolution with a bare "doesn't exist in file system". Fetch it first.
# bootstrap.sh lives under ios/BenchmarkApp/scripts/ and must run from ios/BenchmarkApp/.
if [ ! -d "$REPO/ios/BenchmarkApp/Vendored/Anemll/anemll-swift-cli" ]; then
  log "bootstrap (fetch vendored deps — first run on this clone)"
  (cd "$REPO/ios/BenchmarkApp" && ./scripts/bootstrap.sh)
fi

log "build + install"
# -skipPackagePluginValidation: mlx-swift vendors a "CudaBuild" SwiftPM plugin, and plugin
#   trust is a GUI prompt — headless builds fail validation without this.
# Do NOT send xcodebuild to /dev/null: on failure you get a bare "** BUILD FAILED **" with no
#   cause. Keep the log and grep it.
# Signing needs the App ID to carry Extended Virtual Addressing + Increased Memory Limit.
#   Increased Memory Limit is what raises the jetsam ceiling to ~6.44 GB — the whole point of
#   benchmarking multi-GB models on a phone. If signing errors mention them, open the project
#   in Xcode once and let automatic signing register the capabilities; do not strip them.
xcodebuild -project "$PROJ" -scheme BenchmarkApp -configuration Release \
  -destination "generic/platform=iOS" -derivedDataPath "$DD" \
  -skipPackagePluginValidation -skipMacroValidation \
  DEVELOPMENT_TEAM="$TEAM" CODE_SIGN_STYLE=Automatic build > /tmp/xcb_gemma4.log 2>&1 || {
    echo "BUILD FAILED — cause:"; grep -iE "error:" /tmp/xcb_gemma4.log | head -5; exit 1; }
xcrun devicectl device uninstall app --device "$UDID" "$BUNDLE_ID" 2>/dev/null || true
xcrun devicectl device install app --device "$UDID" "$APP"

# runtime, model-id  — each arm at its BEST available quantization.
# These are the four arms in the published table. Core AI has no E2B entry in the catalog
# (only E4B); adding it needs the h18p bundle published or sideloaded — separate task.
ENGINES=(
  "litert-lm   litert-community/gemma-4-E2B-it-litert-lm"        # wNa8o8 QAT — unchanged, already its best
  "mlx-swift   mlx-community/gemma-4-e2b-it-qat-OptiQ-4bit"      # QAT        — was PTQ
  "llama.cpp   google/gemma-4-E2B-it-qat-q4_0-gguf"              # official QAT — was 3rd-party PTQ
  "coreml-llm  coreml-llm/gemma4-e2b"                            # INT4 palettized — unchanged
)

for e in "${ENGINES[@]}"; do
  set -- $e
  for run in 1 2 3; do
    log "run $1 $2 short-chat (cold $run/3)"
    xcrun devicectl device process launch --terminate-existing --device "$UDID" "$BUNDLE_ID" -- \
      --yardstick-autorun --runtime "$1" --model-id "$2" --task short-chat --runs 1 >/dev/null
    sleep 150   # E2B: first run also downloads the model from HF
  done
  log "run $1 $2 quality"
  xcrun devicectl device process launch --terminate-existing --device "$UDID" "$BUNDLE_ID" -- \
    --yardstick-autorun --runtime "$1" --model-id "$2" --task quality --runs 1 >/dev/null
  sleep 150
done

log "pull results"
rm -rf "$PULL"; mkdir -p "$PULL"
xcrun devicectl device copy from --device "$UDID" --domain-type appDataContainer \
  --domain-identifier "$BUNDLE_ID" --source "Documents/results" --destination "$PULL"
echo "raw results in $PULL"
