#!/usr/bin/env bash
# Build the CANONICAL Mac yardstick: the xcodebuild target, not `swift build`.
# The SwiftPM binary defines YARDSTICK_SPM, which compiles OUT llama-cpp /
# coreml-llm / executorch / anemll (flavor "spm-lite" in `yardstick version`);
# the xcodebuild target also carries project.yml's pinned mlx-swift-lm revision.
# Prints the binary path on success.
#
# Env: DD_MAC (derivedData, default ~/bench-dd-mac)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DD_MAC="${DD_MAC:-$HOME/bench-dd-mac}"
YS="$DD_MAC/Build/Products/Release/yardstick"

( cd "$REPO/ios/BenchmarkApp" && xcodegen generate >/dev/null )
# ARCHS=arm64: Release otherwise also builds the x86_64 slice of every package,
# and CoreML-LLM uses Float16, which does not exist on x86_64 macOS.
xcodebuild -project "$REPO/ios/BenchmarkApp/BenchmarkApp.xcodeproj" -scheme yardstick \
  -configuration Release -destination "platform=macOS,arch=arm64" -derivedDataPath "$DD_MAC" \
  -skipPackagePluginValidation -skipMacroValidation ARCHS=arm64 ONLY_ACTIVE_ARCH=YES \
  build 2>&1 | grep -E "\.swift:[0-9]+:[0-9]+: error|BUILD (SUCCEEDED|FAILED)" | sort -u | head -20
[ -x "$YS" ] || { echo "build produced no binary at $YS" >&2; exit 1; }
echo "$YS"
