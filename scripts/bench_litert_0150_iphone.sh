#!/usr/bin/env bash
#
# bench_litert_0150_iphone.sh — LiteRT-LM arm re-capture on the v0.15.0 release
# (CLiteRTLM.xcframework via the vendored package) + the thinking cell.
#
# Protocol: the 2026-07-28/29 agreed conditions — ctx 2048, warm n=8 per cell
# (8 separate launches; run 1 of a launch is the launch's own warmup inside the
# app), thermal-nominal starts (in-app gate), UNPLUGGED, quiet device, serial.
# Thinking cells: --litert-thinking, thinking-length budget (max-tokens 1400),
# n=4 (the Core AI reference row is n=3). The result JSON self-identifies via
# harnessStamp "+litert-thinking".
#
# PRECONDITIONS (physical, cannot be scripted): phone unplugged, battery <=90%,
# Auto-Lock Never, no other use during the run (~75 min).
#
# Usage: scripts/bench_litert_0150_iphone.sh [UDID]
#   Off this machine: BENCH_UDID (or arg 1) selects the device — see
#   `xcrun devicectl list devices`. OUT overrides the output dir; the default is
#   date-stamped so a re-run can never clobber the published 2026-08-04 campaign
#   (results/raw/2026-08-04-litert-0150-iphone).
set -euo pipefail

UDID="${1:-${BENCH_UDID:-A6F3E849-1947-5202-9AD1-9C881CA58EEF}}"
BUNDLE_ID="com.daisukemajima.llmbench"
MODEL="litert-community/gemma-4-E2B-it-litert-lm"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT:-$REPO/results/raw/$(date +%Y-%m-%d)-litert-0150-iphone}"
xcrun devicectl list devices 2>/dev/null | grep -q "$UDID" || {
  echo "device $UDID not visible — pass a UDID or set BENCH_UDID (xcrun devicectl list devices)" >&2
  exit 1
}
mkdir -p "$OUT"

log(){ printf '\n=== %s [%s]\n' "$*" "$(date +%H:%M:%S)"; }

launch(){ # launch <task> <sleep_s> [extra args...]
  local task=$1 pause=$2; shift 2
  xcrun devicectl device process launch --terminate-existing --device "$UDID" "$BUNDLE_ID" -- \
    --yardstick-autorun --runtime litert-lm --model-id "$MODEL" --task "$task" --runs 1 "$@" >/dev/null
  sleep "$pause"
}

# battery/charging preflight is recorded per-cell by the app (device.batteryState);
# the analyzer rejects any cell that recorded "charging".

log "short-chat OFF x8"
for i in 1 2 3 4 5 6 7 8; do
  log "short-chat OFF $i/8"
  launch short-chat 120
done

log "long-context-1024 OFF x8"
for i in 1 2 3 4 5 6 7 8; do
  log "long-context-1024 OFF $i/8"
  launch long-context-1024 180
done

log "short-chat THINKING x4 (max-tokens 1400)"
for i in 1 2 3 4; do
  log "thinking $i/4"
  launch short-chat 150 --max-tokens 1400 --litert-thinking
done

log "pull results"
PULL="$OUT/pulled"
rm -rf "$PULL"; mkdir -p "$PULL"
xcrun devicectl device copy from --device "$UDID" --domain-type appDataContainer \
  --domain-identifier "$BUNDLE_ID" --source "Documents/results" --destination "$PULL"

log "done — raw in $PULL"
ls "$PULL" | tail -25
