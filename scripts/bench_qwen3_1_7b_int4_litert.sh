#!/usr/bin/env bash
#
# bench_qwen3_1_7b_int4_litert.sh — capture ONLY the LiteRT-LM int4-mixed Qwen3-1.7B row
# on iPhone 17 Pro (short-chat x3 iso-cold + quality), to complete the 1.7B int4 GPU 3-way
# (Core AI int4 + MLX int4 already captured). The int4 .litertlm is OUR MIXED4 conversion
# (int4 body + int8 tied-embedding; the embedding-at-int4 is what collapsed pure-int4).
#
# Build the artifact FIRST (Mac, no device):
#   cd ~/code/litertlm-convert && FORCE_SPM=1 .venv/bin/python scripts/export_simple_template.py \
#     Qwen/Qwen3-1.7B out/qwen3_1_7b_mixed4 templates/chatml_simple.jinja MIXED4
#
# Usage: scripts/bench_qwen3_1_7b_int4_litert.sh [udid]
set -euo pipefail
UDID="${1:-A6F3E849-1947-5202-9AD1-9C881CA58EEF}"   # iPhone 17 Pro
MODEL_ID="litert-local/qwen3-1.7b-int4"
DEVDIR="litert-local__Qwen3-1.7B-int4"              # hfRepoId 'litert-local/Qwen3-1.7B-int4' -> '/'→'__'
LITERT_SRC="$HOME/code/litertlm-convert/out/qwen3_1_7b_mixed4/model.litertlm"
BUNDLE_ID="${YARDSTICK_BUNDLE_ID:-com.iosllmbenchmark.benchmarkapp}"; TEAM="${YARDSTICK_TEAM:-MFN25KNUGJ}"; DEVICE="iphone17pro"   # override both for your own Apple ID / App ID
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PROJ="$REPO/ios/BenchmarkApp/BenchmarkApp.xcodeproj"
DD="$HOME/Library/Developer/Xcode/DerivedData/BenchmarkApp-coreai"
APP="$DD/Build/Products/Release-iphoneos/BenchmarkApp.app"
STAGE="/tmp/q1_7b-int4-sideload"; PULL="/tmp/q1_7b-int4-results"

log(){ printf '\n=== %s ===\n' "$*"; }
copy_to(){ xcrun devicectl device copy to --device "$UDID" --domain-type appDataContainer \
  --domain-identifier "$BUNDLE_ID" --source "$1" --destination "$2" 2>&1 | grep -iE "File on Device|error" | tail -1; }

log "preflight — int4 .litertlm"
[ -f "$LITERT_SRC" ] || { echo "MISSING $LITERT_SRC — run the MIXED4 export first" >&2; exit 1; }
echo "OK: $(du -h "$LITERT_SRC" | cut -f1) $LITERT_SRC"

log "build Release (catalog has litert-local/qwen3-1.7b-int4) + FRESH install"
xcodebuild -project "$PROJ" -scheme BenchmarkApp -configuration Release \
  -destination "platform=iOS,id=$UDID" -derivedDataPath "$DD" \
  -allowProvisioningUpdates CODE_SIGN_STYLE=Automatic DEVELOPMENT_TEAM="$TEAM" build
xcrun devicectl device uninstall app --device "$UDID" "$BUNDLE_ID" 2>/dev/null || true
xcrun devicectl device install app --device "$UDID" "$APP"

log "side-load int4 .litertlm -> $DEVDIR"
rm -rf "$STAGE"; mkdir -p "$STAGE/litert"
cp -L "$LITERT_SRC" "$STAGE/litert/"
copy_to "$STAGE/litert" "Documents/models/litert-lm/$DEVDIR"

log "run short-chat (3 iso-cold) + quality"
for run in 1 2 3; do
  log "short-chat (cold $run/3)"
  xcrun devicectl device process launch --terminate-existing --device "$UDID" "$BUNDLE_ID" -- \
    --yardstick-autorun --runtime litert-lm --model-id "$MODEL_ID" --task short-chat --runs 1 >/dev/null
  sleep 90
done
log "quality (8 checkable Qs + degeneracy)"
xcrun devicectl device process launch --terminate-existing --device "$UDID" "$BUNDLE_ID" -- \
  --yardstick-autorun --runtime litert-lm --model-id "$MODEL_ID" --task quality --runs 1 >/dev/null
sleep 130

log "pull + import"
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
    return re.sub(r"-4bit$", "", s)
c = {}
for f in sorted(glob.glob(os.path.join(os.environ["PULL"], "**", "*.json"), recursive=True)):
    try: d = json.loads(pathlib.Path(f).read_text())
    except Exception as ex: print("skip", f, ex); continue
    rt = d.get("runtime", "?"); mid = (d.get("model") or {}).get("id", "?"); task = d.get("task", "?")
    if "1.7b-int4" not in short_m(mid): continue   # only the new int4 row
    k = (rt, mid, task); c[k] = c.get(k, 0) + 1
    (raw / f"{dev}-{short_rt(rt)}-{short_m(mid)}-{task}-run{c[k]}.jsonl").write_text(json.dumps(d))
print("imported", sum(c.values()), "rows:", {f"{r}/{short_m(m)}/{t}": n for (r, m, t), n in c.items()})
PY
log "done — verify results/raw/${DEVICE}-litert-lm-qwen3-1.7b-int4-* then re-run the report scripts"
