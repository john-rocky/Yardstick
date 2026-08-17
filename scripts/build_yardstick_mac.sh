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
# The grep is display-only; the build verdict is xcodebuild's own exit code —
# a resolution error ("xcodebuild: error: Could not resolve...") matches no
# .swift: pattern and used to vanish, letting a stale binary pass the -x check.
LOG="$(mktemp)"
if ! xcodebuild -project "$REPO/ios/BenchmarkApp/BenchmarkApp.xcodeproj" -scheme yardstick \
  -configuration Release -destination "platform=macOS,arch=arm64" -derivedDataPath "$DD_MAC" \
  -skipPackagePluginValidation -skipMacroValidation ARCHS=arm64 ONLY_ACTIVE_ARCH=YES \
  build > "$LOG" 2>&1; then
  grep -E "\.swift:[0-9]+:[0-9]+: error|xcodebuild: error|BUILD FAILED" "$LOG" | sort -u | head -20 >&2
  echo "xcodebuild failed — full log: $LOG" >&2
  exit 1
fi
grep -E "BUILD (SUCCEEDED|FAILED)" "$LOG" | sort -u
rm -f "$LOG"
[ -x "$YS" ] || { echo "build produced no binary at $YS" >&2; exit 1; }
echo "$YS"
