#!/usr/bin/env bash
#
# bench_litert_resident_ab_iphone.sh — same-session resident-memory A/B:
# LiteRT-LM v0.14.0 vs v0.15.0, two app installs (bundle ids …llmbench014 / …llmbench),
# freshly rebooted + unplugged phone, long-context-1024 cells, ABAB interleave n=4/side.
# Purpose: decide whether the +17% deep-context resident (849 → 996 MB vs the July
# session) is a 0.15 change or cross-session drift. Footprint is expected flat.
# Off this machine: BENCH_UDID (or arg 1) selects the device. OUT overrides the
# output dir; the default is date-stamped in THIS repo (the original pointed at the
# historical ~/code/apple-silicon-llm-bench clone — audit gap 1-3 — and a re-run
# must not clobber the published 2026-08-04 campaign either way).
set -euo pipefail
UDID="${1:-${BENCH_UDID:-A6F3E849-1947-5202-9AD1-9C881CA58EEF}}"
B015="com.daisukemajima.llmbench"
B014="com.daisukemajima.llmbench014"
MODEL="litert-community/gemma-4-E2B-it-litert-lm"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT:-$REPO/results/raw/$(date +%Y-%m-%d)-litert-resident-ab}"
xcrun devicectl list devices 2>/dev/null | grep -q "$UDID" || {
  echo "device $UDID not visible — pass a UDID or set BENCH_UDID (xcrun devicectl list devices)" >&2
  exit 1
}
mkdir -p "$OUT"

log(){ printf '\n=== %s [%s]\n' "$*" "$(date +%H:%M:%S)"; }
cell(){ # cell <bundle-id>
  xcrun devicectl device process launch --terminate-existing --device "$UDID" "$1" -- \
    --yardstick-autorun --runtime litert-lm --model-id "$MODEL" --task long-context-1024 --runs 1 >/dev/null
  sleep 180
}

for round in 1 2 3 4; do
  if [ $((round % 2)) -eq 1 ]; then order="$B014 $B015"; else order="$B015 $B014"; fi
  for b in $order; do
    log "round $round — $b"
    cell "$b"
  done
done

log "pull both containers"
for b in $B014 $B015; do
  rm -rf "$OUT/pulled-$b"; mkdir -p "$OUT/pulled-$b"
  xcrun devicectl device copy from --device "$UDID" --domain-type appDataContainer \
    --domain-identifier "$b" --source "Documents/results" --destination "$OUT/pulled-$b"
done
log "done"
