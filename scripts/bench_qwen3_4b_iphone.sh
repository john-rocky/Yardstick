#!/usr/bin/env bash
#
# bench_qwen3_4b_iphone.sh — Qwen3-4B on iPhone 17 Pro:
#   Core AI (GPU lin-INT4 + ANE palettized-4bit) vs LiteRT-LM vs MLX, short-chat, 3 iso-cold.
#
# Core AI 4B bundles are pre-assembled (coreai-matrix-completion-RESULTS.md); litert+mlx 4B are
# HF-cached. Rebuilds Release (catalog now carries the 4B core-ai ids), FRESH-installs (uninstall
# first) so the device's Documents/results doesn't accumulate across sessions, side-loads, runs,
# pulls + imports with generator-compatible names (strip mlx -4bit, keep core-ai -gpu/-ane).
#
# Usage: scripts/bench_qwen3_4b_iphone.sh [udid]
set -euo pipefail
UDID="${1:-A6F3E849-1947-5202-9AD1-9C881CA58EEF}"
SIZE="4b"
LITERT_REPO="litert-community/Qwen3-4B"; LITERT_FILE="qwen3_4b_mixed_int4.litertlm"
MLX_REPO="mlx-community/Qwen3-4B-4bit"
BUNDLE_ID="${YARDSTICK_BUNDLE_ID:-com.iosllmbenchmark.benchmarkapp}"; TEAM="${YARDSTICK_TEAM:-MFN25KNUGJ}"; DEVICE="iphone17pro"   # override both for your own Apple ID / App ID
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PROJ="$REPO/ios/BenchmarkApp/BenchmarkApp.xcodeproj"
DD="$HOME/Library/Developer/Xcode/DerivedData/BenchmarkApp-coreai"
APP="$DD/Build/Products/Release-iphoneos/BenchmarkApp.app"
EXPORTS="$HOME/code/coreai/coreai-models/exports"
HF="$HOME/.cache/huggingface/hub"
STAGE="/tmp/q${SIZE}-sideload"; PULL="/tmp/q${SIZE}-results"

log(){ printf '\n=== %s ===\n' "$*"; }
copy_to(){ xcrun devicectl device copy to --device "$UDID" --domain-type appDataContainer \
  --domain-identifier "$BUNDLE_ID" --source "$1" --destination "$2" 2>&1 | grep -iE "File on Device|error" | tail -1; }

log "build Release (catalog has core-ai qwen3-${SIZE} ids) + FRESH install"
xcodebuild -project "$PROJ" -scheme BenchmarkApp -configuration Release \
  -destination "platform=iOS,id=$UDID" -derivedDataPath "$DD" \
  -allowProvisioningUpdates CODE_SIGN_STYLE=Automatic DEVELOPMENT_TEAM="$TEAM" build
xcrun devicectl device uninstall app --device "$UDID" "$BUNDLE_ID" 2>/dev/null || true
xcrun devicectl device install app --device "$UDID" "$APP"

log "side-load Qwen3-${SIZE}: core-ai gpu+ane, litert, mlx"
copy_to "$EXPORTS/qwen3_${SIZE}_gpu"          "Documents/CoreAIModels/qwen3_${SIZE}_gpu"
copy_to "$EXPORTS/qwen3_${SIZE}_ane_pure4bit" "Documents/CoreAIModels/qwen3_${SIZE}_ane"
rm -rf "$STAGE"; mkdir -p "$STAGE/litert" "$STAGE/mlx/blobs" "$STAGE/mlx/refs"
LITSNAP="$(ls -d "$HF"/models--$(echo "$LITERT_REPO" | sed 's|/|--|g')/snapshots/*/ | head -1)"
cp -L "$LITSNAP/$LITERT_FILE" "$STAGE/litert/"          # deref symlink -> real .litertlm
copy_to "$STAGE/litert" "Documents/models/litert-lm/$(echo "$LITERT_REPO" | sed 's|/|__|g')"
MHUB="$HF/models--$(echo "$MLX_REPO" | sed 's|/|--|g')"
ln "$MHUB"/blobs/* "$STAGE/mlx/blobs/" 2>/dev/null || cp "$MHUB"/blobs/* "$STAGE/mlx/blobs/"
cp "$MHUB"/refs/main "$STAGE/mlx/refs/main"
copy_to "$STAGE/mlx" "Library/Caches/huggingface/hub/models--$(echo "$MLX_REPO" | sed 's|/|--|g')"

log "run short-chat (3 iso-cold) + quality (degeneracy/correctness gate, 1x) per engine"
# Core AI 4B/8B output PARITY is still being verified by the export session; the on-device
# quality task is the complementary degeneracy guardrail (the one that caught MiniCPM garbage).
ENGINES=(
  "core-ai core-ai/qwen3-${SIZE}-gpu"
  "core-ai core-ai/qwen3-${SIZE}-ane"
  "litert-lm $LITERT_REPO"
  "mlx-swift $MLX_REPO"
)
for e in "${ENGINES[@]}"; do
  set -- $e
  for run in 1 2 3; do
    log "run $1 $2 short-chat (cold $run/3)"
    xcrun devicectl device process launch --terminate-existing --device "$UDID" "$BUNDLE_ID" -- \
      --yardstick-autorun --runtime "$1" --model-id "$2" --task short-chat --runs 1 >/dev/null
    sleep 130   # 4B: allow load + 128-token decode + teardown
  done
  log "run $1 $2 quality (8 checkable Qs + degeneracy)"
  xcrun devicectl device process launch --terminate-existing --device "$UDID" "$BUNDLE_ID" -- \
    --yardstick-autorun --runtime "$1" --model-id "$2" --task quality --runs 1 >/dev/null
  sleep 170   # quality = 8 questions x up to 256 tokens — longer than short-chat
done

log "pull + import (generator-compatible names)"
rm -rf "$PULL"; mkdir -p "$PULL"
xcrun devicectl device copy from --device "$UDID" --domain-type appDataContainer \
  --domain-identifier "$BUNDLE_ID" --source "Documents/results" --destination "$PULL"
REPO="$REPO" DEVICE="$DEVICE" PULL="$PULL" python3 - <<'PY'
import json, os, re, glob, pathlib
repo = pathlib.Path(os.environ["REPO"]); dev = os.environ["DEVICE"]
raw = repo / "results" / "raw"
def short_rt(r): return {"mlx-swift": "mlx"}.get(r, r)
def short_m(mid):
    s = re.sub(r"[^a-z0-9.\-]+", "-", mid.split("/")[-1].lower())
    return re.sub(r"-4bit$", "", s)   # strip mlx -4bit; keep core-ai -gpu/-ane
c = {}
for f in sorted(glob.glob(os.path.join(os.environ["PULL"], "**", "*.json"), recursive=True)):
    try: d = json.loads(pathlib.Path(f).read_text())
    except Exception as ex: print("skip", f, ex); continue
    rt = d.get("runtime", "?"); mid = (d.get("model") or {}).get("id", "?"); task = d.get("task", "?")
    k = (rt, mid, task); c[k] = c.get(k, 0) + 1
    (raw / f"{dev}-{short_rt(rt)}-{short_m(mid)}-{task}-run{c[k]}.jsonl").write_text(json.dumps(d))
print("imported", sum(c.values()), "rows:", {f"{r}/{short_m(m)}": n for (r, m, t), n in c.items()})
PY
log "done — run: python3 scripts/litert_lm_report.py"
