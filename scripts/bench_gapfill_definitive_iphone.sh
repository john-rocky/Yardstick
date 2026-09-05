#!/usr/bin/env bash
#
# bench_gapfill_definitive_iphone.sh — the two cells the 2026-07-27 definitive run could not
# produce. Everything else from that run is complete and already imported to
# results/raw/2026-07-27-definitive (46 captures, n=7-9 per cell, MAD 0-1%, none above nominal).
#
# Gap 1 — Cactus produced nothing. Eight launches, zero result files, no diagnosis yet. The
#   arm ran fine on 7/20, so this is most likely its bundle having been evicted (it is fetched
#   from HF like every other model) rather than the runtime. The first cell here is a single
#   diagnostic launch with the console attached, so the failure is visible rather than inferred.
#
# Gap 2 — LiteRT-LM has no prefill number. `long-context-1024-gen256` stops at the 256-token
#   cap, and MediaPipeRuntime only reads LiteRT's own counters on a natural EOS finish
#   (`let bench = capped ? nil : ...`, MediaPipeRuntime.swift:162). Capped runs fall back to
#   chunk counting, which reports promptTokenCount = 0. So the task that fixed decode-at-depth
#   took prefill away for that one arm. The fix needs no code change: the older
#   `long-context-1024` task ends on EOS, so it still yields LiteRT's prefill counters — the
#   same cells the 7/26 pass captured at 1,672 tok/s. Run both tasks and quote each for what
#   it can measure.
#
# Unplugged, Auto-Lock Never, phone left alone. ~25 min.
#
set -euo pipefail

UDID="${1:-A6F3E849-1947-5202-9AD1-9C881CA58EEF}"
BUNDLE_ID="${YARDSTICK_BUNDLE_ID:-com.daisukemajima.llmbench}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PULL="/tmp/gemma4-gapfill-definitive"
REPS="${REPS:-7}"
RUN_START="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

log(){ printf '\n=== [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
TIMEOUT_BIN="$(command -v gtimeout || command -v timeout || true)"

cell() {  # runtime model task  [show-console]
  local guard=()
  [ -n "$TIMEOUT_BIN" ] && guard=("$TIMEOUT_BIN" "${CELL_TIMEOUT:-900}")
  if [ "${4:-}" = "show" ]; then
    "${guard[@]}" xcrun devicectl device process launch \
      --terminate-existing --console --device "$UDID" "$BUNDLE_ID" -- \
      --yardstick-autorun --runtime "$1" --model-id "$2" --task "$3" --runs 1 2>&1 | tail -25 || true
  else
    "${guard[@]}" xcrun devicectl device process launch \
      --terminate-existing --console --device "$UDID" "$BUNDLE_ID" -- \
      --yardstick-autorun --runtime "$1" --model-id "$2" --task "$3" --runs 1 >/dev/null 2>&1 || true
  fi
}

CACTUS="Cactus-Compute/gemma-4-E2B-it-cq4-uncalibrated"
LITERT="litert-community/gemma-4-E2B-it-litert-lm"

# Cactus needs its CQ bundle sideloaded — it is not fetched from HF like the other arms, and
# reinstalling the app wipes the Data container that holds it (diagnosed 2026-07-27: "No Cactus
# bundle (config.txt) under .../models--Cactus-Compute--gemma-4-E2B-it/snapshots/b305fe22...").
# The bundle is 3.9 GB at ~/code/cactus/weights/gemma-4-e2b-it-cq4-uncalibrated. Push it with
# `devicectl device copy to` before setting SKIP_CACTUS=0.
if [ "${SKIP_CACTUS:-0}" = "1" ]; then
  log "skipping Cactus (SKIP_CACTUS=1)"
else

log "gap 1 — Cactus, one diagnostic launch WITH console output"
cell "cactus" "$CACTUS" short-chat show

log "if that printed YARDSTICK_RUN_OK, the arm works and the definitive run simply lost it;"
log "if it printed a FATAL or nothing, the reason is now on screen. Continuing either way."

for r in $(seq 1 "$REPS"); do
  log "Cactus rep $r/$REPS — short-chat"
  cell "cactus" "$CACTUS" short-chat
  sleep "${COOLDOWN:-150}"
  log "Cactus rep $r/$REPS — long-context-1024-gen256"
  cell "cactus" "$CACTUS" long-context-1024-gen256
  sleep "${COOLDOWN:-150}"
done

fi

log "gap 2 — LiteRT prefill via the EOS-terminating task (counters finalize only on EOS)"
for r in $(seq 1 "$REPS"); do
  log "LiteRT rep $r/$REPS — long-context-1024 (prefill)"
  cell "litert-lm" "$LITERT" long-context-1024
  sleep "${COOLDOWN:-150}"
done

log "pull"
rm -rf "$PULL"; mkdir -p "$PULL"
xcrun devicectl device copy from --device "$UDID" --domain-type appDataContainer \
  --domain-identifier "$BUNDLE_ID" --source "Documents/results" --destination "$PULL"

log "verdicts (this run only)"
python3 "$REPO/scripts/analyze_comparability.py" "$PULL" --since="$RUN_START"
echo
echo "raw results in $PULL   (this run = captures at/after $RUN_START)"
echo "import with:  python3 - <<'PY'
import json,glob,shutil
from pathlib import Path
dst=Path('results/raw/2026-07-27-definitive'); dst.mkdir(parents=True,exist_ok=True)
for f in glob.glob('$PULL/**/*.json',recursive=True):
    o=json.load(open(f))
    if o.get('timestamp','')>='$RUN_START': shutil.copy2(f,dst/Path(f).name)
PY"
