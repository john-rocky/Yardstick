#!/usr/bin/env bash
#
# bench_cactus_iphone.sh — the Cactus arm of the Gemma-4-E2B best-available table,
# measured at the SAME protocols as the other six rows (short-chat 128 tok / 3
# thermal-nominal colds; deep-context p=1024/g=256; quality sanity). Energy is a
# separate unplugged run — see bench_energy_iphone_gemma_part2.sh for the device prep.
#
# Row build: Cactus-Compute/gemma-4-E2B-it `gemma-4-e2b-it-cq4-uncalibrated.zip` —
# the BEST USABLE CQ4 (GSM8K n=100 one-harness: 87.0% vs the shipped default
# "calibrated" cq4.zip's 3.0%; both measured 2026-07-20). The calibrated default is
# captured too, as the footnote row (its speed ≈ same format, its quality is the
# shipped-default finding).
#
# Prereqs (this script does NOT do them):
#   * app built + installed (CactusRuntime arm; Vendored/cactus-ios.xcframework)
#   * both bundles sideloaded to
#       Documents/models/cactus/Cactus-Compute__gemma-4-E2B-it/gemma-4-e2b-it-cq4/
#       Documents/models/cactus/Cactus-Compute__gemma-4-E2B-it/gemma-4-e2b-it-cq4-uncalibrated/
#     devicectl's directory copy has been observed to SILENTLY DROP files on a
#     3.8 GB / 2030-file tree (219 missing, exit 0, 2026-07-20) — verify with
#     `devicectl device info files` against the local listing and re-push the diff
#     before trusting a bundle.
#
set -uo pipefail

UDID="${1:-A6F3E849-1947-5202-9AD1-9C881CA58EEF}"   # DaisukeのiPhone (iPhone 17 Pro) — SHARED device
BUNDLE_ID="${YARDSTICK_BUNDLE_ID:-com.daisukemajima.llmbench}"
PULL="${PULL:-/tmp/cactus-arm-results}"

UNCAL="Cactus-Compute/gemma-4-E2B-it-cq4-uncalibrated"   # row build (best usable)
CAL="Cactus-Compute/gemma-4-E2B-it-cq4"                  # shipped default (footnote)

log(){ printf '\n=== %s %s\n' "$(date +%H:%M:%S)" "$*"; }

run_task(){ # runtime model task
  xcrun devicectl device process launch --terminate-existing --device "$UDID" "$BUNDLE_ID" -- \
    --yardstick-autorun --runtime cactus --model-id "$1" --task "$2" --runs 1 >/dev/null 2>&1
}

# Cactus short-chat ≈ load(~5 s) + 128 tok @ ~36 tok/s ≈ <15 s; 150 s spacing doubles
# as the >=100 s thermal cooldown (fairness rules). long-context prefills 1024 ≈ +2 s.
log "row build ($UNCAL): first-ever + 3 colds, short-chat"
for run in 0 1 2 3; do   # run 0 = first-ever (engine/Metal caches build) — reported separately
  run_task "$UNCAL" short-chat; sleep 150
done

log "row build: deep-context p=1024/g=256 x3"
for run in 1 2 3; do
  run_task "$UNCAL" long-context-1024; sleep 180
done

log "row build: quality sanity (9-question)"
run_task "$UNCAL" quality; sleep 150

log "shipped default ($CAL): 3 colds short-chat (footnote row)"
for run in 1 2 3; do
  run_task "$CAL" short-chat; sleep 150
done

log "shipped default: quality sanity"
run_task "$CAL" quality; sleep 150

log "pull results"
rm -rf "$PULL"; mkdir -p "$PULL"
xcrun devicectl device copy from --device "$UDID" --domain-type appDataContainer \
  --domain-identifier "$BUNDLE_ID" --source "Documents/results" --destination "$PULL" >/dev/null 2>&1
ls "$PULL" | grep -i cactus || echo "NO CACTUS RESULTS — check --console output of a manual launch"
echo DONE
