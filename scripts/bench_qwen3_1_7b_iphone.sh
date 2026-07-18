#!/usr/bin/env bash
#
# bench_qwen3_1_7b_iphone.sh — Qwen3-1.7B on iPhone 17 Pro:
#   Core AI (GPU lin-INT4 + ANE palettized-4bit) vs LiteRT-LM vs MLX, short-chat ×3 iso-cold + quality.
#
# 1.7B is the iPhone-ceiling probe: Qwen3-0.6B invokes on LiteRT-LM iOS, 4B invoke-fails — 1.7B sits
# between. litert-community ships NO 1.7B, so the LiteRT row is OUR int8 conversion (side-loaded,
# disclosed as int8 vs the official 0.6B/4B int4-QAT). Build the artifacts FIRST (Mac, no device):
#   1) Core AI : scripts/export_coreai_qwen3.sh 1_7b Qwen/Qwen3-1.7B 4096 h18p
#   2) LiteRT  : (cd ~/code/litertlm-convert && FORCE_SPM=1 ./.venv/bin/python \
#                  scripts/export_simple_template.py Qwen/Qwen3-1.7B out/qwen3_1_7b_int8 \
#                  templates/chatml_simple.jinja dynamic_wi8_afp32)
#   3) verify  : swift run --package-path ~/code/litert-mac-verify litert-mac-verify \
#                  ~/code/litertlm-convert/out/qwen3_1_7b_int8/*.litertlm "Hello"  # coherence, no device
#
# Then plug in the iPhone (unlocked, trusted) and run this. As always: --console the LiteRT row once
# by hand first (a non-interactive launch can't attach) to confirm it INVOKES before trusting JSONL.
#
# Usage: scripts/bench_qwen3_1_7b_iphone.sh [udid]
set -euo pipefail
UDID="${1:-A6F3E849-1947-5202-9AD1-9C881CA58EEF}"   # DaisukeのiPhone (iPhone 17 Pro)
SIZE_FS="1_7b"          # export-folder convention (qwen3_1_7b_*)
SIZE_ID="1.7b"          # catalog-id convention   (core-ai/qwen3-1.7b-*)
MLX_REPO="mlx-community/Qwen3-1.7B-4bit"
LITERT_ID="litert-local/qwen3-1.7b"
LITERT_DEVDIR="litert-local__Qwen3-1.7B"            # hfRepoId 'litert-local/Qwen3-1.7B' -> '/'→'__'
LITERT_SRC_DIR="$HOME/code/litertlm-convert/out/qwen3_1_7b_int8"
BUNDLE_ID="${YARDSTICK_BUNDLE_ID:-com.iosllmbenchmark.benchmarkapp}"; TEAM="${YARDSTICK_TEAM:-MFN25KNUGJ}"; DEVICE="iphone17pro"   # override both for your own Apple ID / App ID
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PROJ="$REPO/ios/BenchmarkApp/BenchmarkApp.xcodeproj"
DD="$HOME/Library/Developer/Xcode/DerivedData/BenchmarkApp-coreai"
APP="$DD/Build/Products/Release-iphoneos/BenchmarkApp.app"
EXPORTS="$HOME/code/coreai/coreai-models/exports"
HF="$HOME/.cache/huggingface/hub"
STAGE="/tmp/q${SIZE_FS}-sideload"; PULL="/tmp/q${SIZE_FS}-results"

log(){ printf '\n=== %s ===\n' "$*"; }
copy_to(){ xcrun devicectl device copy to --device "$UDID" --domain-type appDataContainer \
  --domain-identifier "$BUNDLE_ID" --source "$1" --destination "$2" 2>&1 | grep -iE "File on Device|error" | tail -1; }

# ---- preflight: every artifact must exist before we touch the device --------
log "preflight — Core AI bundles + local .litertlm + MLX cache"
for d in "$EXPORTS/qwen3_${SIZE_FS}_gpu" "$EXPORTS/qwen3_${SIZE_FS}_ane_pure4bit"; do
  [ -f "$d/metadata.json" ] || { echo "MISSING $d — run scripts/export_coreai_qwen3.sh 1_7b Qwen/Qwen3-1.7B" >&2; exit 1; }
done
LITERT_FILE="$(ls "$LITERT_SRC_DIR"/*.litertlm 2>/dev/null | head -1 || true)"
[ -n "$LITERT_FILE" ] || { echo "MISSING .litertlm in $LITERT_SRC_DIR — run the export_simple_template.py step" >&2; exit 1; }
MHUB="$HF/models--$(echo "$MLX_REPO" | sed 's|/|--|g')"
[ -d "$MHUB/blobs" ] || { echo "MISSING MLX cache $MHUB — huggingface-cli download $MLX_REPO" >&2; exit 1; }
echo "OK: core-ai gpu+ane, litert=$(basename "$LITERT_FILE"), mlx cache"

log "build Release (catalog has core-ai qwen3-${SIZE_ID} + litert-local + mlx ids) + FRESH install"
xcodebuild -project "$PROJ" -scheme BenchmarkApp -configuration Release \
  -destination "platform=iOS,id=$UDID" -derivedDataPath "$DD" \
  -allowProvisioningUpdates CODE_SIGN_STYLE=Automatic DEVELOPMENT_TEAM="$TEAM" build
xcrun devicectl device uninstall app --device "$UDID" "$BUNDLE_ID" 2>/dev/null || true
xcrun devicectl device install app --device "$UDID" "$APP"

log "side-load Qwen3-${SIZE_ID}: core-ai gpu+ane, litert (local int8), mlx"
copy_to "$EXPORTS/qwen3_${SIZE_FS}_gpu"          "Documents/CoreAIModels/qwen3_${SIZE_FS}_gpu"
copy_to "$EXPORTS/qwen3_${SIZE_FS}_ane_pure4bit" "Documents/CoreAIModels/qwen3_${SIZE_FS}_ane"
rm -rf "$STAGE"; mkdir -p "$STAGE/litert" "$STAGE/mlx/blobs" "$STAGE/mlx/refs"
cp -L "$LITERT_FILE" "$STAGE/litert/"
copy_to "$STAGE/litert" "Documents/models/litert-lm/$LITERT_DEVDIR"
ln "$MHUB"/blobs/* "$STAGE/mlx/blobs/" 2>/dev/null || cp "$MHUB"/blobs/* "$STAGE/mlx/blobs/"
cp "$MHUB"/refs/main "$STAGE/mlx/refs/main"
copy_to "$STAGE/mlx" "Library/Caches/huggingface/hub/models--$(echo "$MLX_REPO" | sed 's|/|--|g')"

log "run short-chat (3 iso-cold) + quality (degeneracy/correctness gate, 1x) per engine"
ENGINES=(
  "core-ai core-ai/qwen3-${SIZE_ID}-gpu"
  "core-ai core-ai/qwen3-${SIZE_ID}-ane"
  "litert-lm $LITERT_ID"
  "mlx-swift $MLX_REPO"
)
for e in "${ENGINES[@]}"; do
  set -- $e
  for run in 1 2 3; do
    log "run $1 $2 short-chat (cold $run/3)"
    xcrun devicectl device process launch --terminate-existing --device "$UDID" "$BUNDLE_ID" -- \
      --yardstick-autorun --runtime "$1" --model-id "$2" --task short-chat --runs 1 >/dev/null
    sleep 90    # 1.7B: load + 128-token decode + teardown (between 0.6B's ~70s and 4B's ~130s)
  done
  log "run $1 $2 quality (8 checkable Qs + degeneracy)"
  xcrun devicectl device process launch --terminate-existing --device "$UDID" "$BUNDLE_ID" -- \
    --yardstick-autorun --runtime "$1" --model-id "$2" --task quality --runs 1 >/dev/null
  sleep 130   # quality = 8 questions x up to 256 tokens
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
log "done — verify results/raw/${DEVICE}-*qwen3-1.7b* then run: python3 scripts/litert_lm_report.py && python3 scripts/quality_check.py"
