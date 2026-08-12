#!/usr/bin/env bash
#
# run_mac_energy.sh — one short line for the Mac J/token capture.
#
# The energy harness needs a yardstick binary and a sudo prompt it can own, which makes the
# invocation long enough that pasting it into a terminal mangles the path (2026-07-26: the
# leading /tmp/ was lost and zsh tried to exec `yardstick-dd2/Build/...`). This wrapper finds
# the binary, states what it is going to measure, and hands off.
#
# Run it from YOUR terminal, without sudo — the harness prompts once and keeps the
# credential for the powermetrics children it spawns.
#
#   bash scripts/run_mac_energy.sh
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"

if [ -z "${YARDSTICK_BIN:-}" ]; then
  for c in /tmp/yardstick-dd2/Build/Products/Release/yardstick \
           /tmp/yardstick-dd/Build/Products/Release/yardstick \
           /tmp/yardstick-verify/Build/Products/Release/yardstick \
           "$REPO/yardstick" "$(command -v yardstick || true)"; do
    [ -x "$c" ] && YARDSTICK_BIN="$c" && break
  done
fi
if [ -z "${YARDSTICK_BIN:-}" ] || [ ! -x "$YARDSTICK_BIN" ]; then
  echo "no yardstick binary found. Build it with:"
  echo "  xcodebuild -project ios/BenchmarkApp/BenchmarkApp.xcodeproj -scheme yardstick \\"
  echo "    -configuration Release -derivedDataPath /tmp/yardstick-dd2 \\"
  echo "    -skipPackagePluginValidation -skipMacroValidation build"
  exit 1
fi

echo "yardstick : $YARDSTICK_BIN"
echo "convention: whole bench invocation (load + prefill + decode) / generated tokens —"
echo "            the documented one. Do not re-derive a decode-only window afterwards;"
echo "            that is what made the published Mac row unreproducible."
echo "output    : results/raw/*-sustained-energy.jsonl (traceable by verify_published_numbers.py)"
echo
echo "Close heavy apps first — powermetrics measures the whole system."
echo
export YARDSTICK_BIN
cd "$REPO"
exec bash scripts/measure_energy_gemma_e2b_v2.sh
