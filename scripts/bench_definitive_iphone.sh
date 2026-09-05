#!/usr/bin/env bash
#
# bench_definitive_iphone.sh — the capture that settles the Gemma-4-E2B iPhone table.
#
# Written 2026-07-26 after two partial passes exposed both a comparability problem and a
# reproducibility problem. What this script does differently, and why each choice exists:
#
#  1. ROUND-ROBIN, not arm-by-arm. The earlier passes ran all of one arm's cells, then the
#     next arm's. The phone warms over a session, so a block design charges that drift to
#     whichever arm ran last — MLX entered `fair` while LiteRT-LM had run at `nominal`.
#     Interleaving spreads any drift evenly over every arm.
#
#  2. n=7, not n=3. Measured spread across three identical runs on 2026-07-26:
#     footprint 1.1-1.3% (fine at n=3) but prefill 6-113% and decode 17-38%. One cold-cache
#     outlier (MLX prefill 292 against a 2,058 median) moves a 3-run median and cannot move
#     a 7-run one.
#
#  3. `long-context-1024-gen256`, not `long-context-1024`. The original task's tail asks for
#     one sentence, so every arm hit EOS after 15-33 tokens and "decode tok/s at p=1024" was
#     a rate over a couple of dozen tokens. The new task keeps the same ~1,081-token prompt
#     and forces the model to fill the 256-token budget. The old id is left untouched so the
#     7/18 captures keep their meaning.
#
#  4. A discarded pre-warm per arm, and every cell now waits on `--console` instead of a fixed
#     sleep. llama.cpp appeared to produce nothing in both earlier passes; it had in fact
#     produced all six cells, 10-30 minutes after the script pulled the results. The bug was
#     the clock, not the arm.
#
#  5. Long spacing (240 s after the long-context cells). Fairness rule 2 wants
#     `initialThermalState == nominal` per cell; the analyzer filters on it post-hoc, and
#     that only works if most cells actually come back nominal.
#
# Requires the 2026-07-26 app build (new task + wall-clock + median-resident metrics), so it
# rebuilds by default. Budget ~3.5-4 h for 4 arms at REPS=7 — each cell now costs its real
# duration (~40-90 s) plus a 150 s cooldown, instead of a padded fixed sleep — and ~40% battery. Run UNPLUGGED from a full charge, Auto-Lock = Never, phone left
# alone. REPS=5 trims it to ~3.5 h if the battery is the binding constraint.
#
set -euo pipefail

UDID="${1:-A6F3E849-1947-5202-9AD1-9C881CA58EEF}"
BUNDLE_ID="${YARDSTICK_BUNDLE_ID:-com.daisukemajima.llmbench}"
TEAM="${YARDSTICK_TEAM:-MFN25KNUGJ}"
REPS="${REPS:-7}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PROJ="$REPO/ios/BenchmarkApp/BenchmarkApp.xcodeproj"
DD="$HOME/Library/Developer/Xcode/DerivedData/BenchmarkApp-bestquant"
APP="$DD/Build/Products/Release-iphoneos/BenchmarkApp.app"
PULL="/tmp/gemma4-definitive"

log(){ printf '\n=== [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# The device keeps every capture ever taken and the pull copies all of them; stamp the
# start so the analyzer can exclude tonight's partial passes and older app builds.
RUN_START="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [ "${REBUILD:-1}" = "1" ]; then
  log "build + install (new task + metrics)"
  xcodebuild -project "$PROJ" -scheme BenchmarkApp -configuration Release \
    -destination "generic/platform=iOS" -derivedDataPath "$DD" \
    -skipPackagePluginValidation -skipMacroValidation -allowProvisioningUpdates \
    PRODUCT_BUNDLE_IDENTIFIER="$BUNDLE_ID" \
    DEVELOPMENT_TEAM="$TEAM" CODE_SIGN_STYLE=Automatic build > /tmp/xcb_definitive.log 2>&1 || {
      echo "BUILD FAILED — cause:"; grep -iE "error:" /tmp/xcb_definitive.log | head -5; exit 1; }
  xcrun devicectl device install app --device "$UDID" "$APP"
fi

ARMS=(
  "litert-lm|litert-community/gemma-4-E2B-it-litert-lm"
  "mlx-swift|mlx-community/gemma-4-e2b-it-4bit"
  "llama.cpp|unsloth/gemma-4-E2B-it-GGUF/Q4_K_M"
  # Cactus is in the matrix because it is the decode runner-up (50.6 published) and it times
  # with wall-clock. If LiteRT-LM's wall-clock decode lands lower than its engine-reported
  # figure, the ranking has to be re-drawn — and this table's own rule forbids doing that
  # across sessions. Capturing it here keeps that comparison in-session.
  "cactus|Cactus-Compute/gemma-4-E2B-it-cq4-uncalibrated"
)

# `--console` attaches and waits for the app to terminate, so a cell takes exactly as long as
# it takes. The earlier passes used fixed sleeps and the final pull outran the runs: llama.cpp
# looked like it had produced nothing at all, when in fact all six of its cells landed on the
# device 10-30 minutes after the script had already copied the results off. Nothing was broken
# but the clock. gtimeout guards the one real stall we know of — breaking a LiteRT stream at a
# cap can wedge the next run for ~10 minutes.
TIMEOUT_BIN="$(command -v gtimeout || command -v timeout || true)"
cell() {  # runtime model task
  local guard=()
  [ -n "$TIMEOUT_BIN" ] && guard=("$TIMEOUT_BIN" "${CELL_TIMEOUT:-900}")
  "${guard[@]}" xcrun devicectl device process launch \
    --terminate-existing --console --device "$UDID" "$BUNDLE_ID" -- \
    --yardstick-autorun --runtime "$1" --model-id "$2" --task "$3" --runs 1 >/dev/null 2>&1 || true
}

# --- pre-flight: refuse to spend 4.5 h on cells the analyzer will throw away ----------
# Charging drives the phone to `fair` within about two cells (measured 2026-07-26: every MLX
# cell of the first pass came back charging/fair and was dropped from the speed rows). One
# throwaway capture up front costs three minutes and catches it.
if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
  log "pre-flight: one capture to check power + thermal state"
  cell "litert-lm" "litert-community/gemma-4-E2B-it-litert-lm" short-chat
  PRE="/tmp/gemma4-preflight"; rm -rf "$PRE"; mkdir -p "$PRE"
  xcrun devicectl device copy from --device "$UDID" --domain-type appDataContainer \
    --domain-identifier "$BUNDLE_ID" --source "Documents/results" --destination "$PRE" >/dev/null 2>&1 || true
  python3 - "$PRE" "$RUN_START" <<'PYEOF' || exit 1
import json, sys, glob
root, since = sys.argv[1], sys.argv[2]
rows = []
for f in glob.glob(f"{root}/**/*.json", recursive=True):
    try: o = json.load(open(f))
    except Exception: continue
    if isinstance(o, dict) and "metrics" in o and (o.get("timestamp") or "") >= since:
        rows.append(o)
if not rows:
    print("PRE-FLIGHT FAILED: no capture came back. The app may not be installed, or the "
          "model needs a longer first load — re-run with SKIP_PREFLIGHT=1 to push through.")
    sys.exit(1)
o = sorted(rows, key=lambda r: r["timestamp"])[-1]
d, m = o.get("device", {}), o["metrics"]
state, thermal = d.get("batteryState"), m.get("initialThermalState")
print(f"   battery={d.get('batteryLevel')} state={state} thermal={thermal}")
if state == "charging":
    print("\nPRE-FLIGHT FAILED: the phone is charging. Every speed cell will start above "
          "nominal and the analyzer will drop it — the first pass lost all of MLX that way.\n"
          "Unplug (drive over the network pairing) and re-run. SKIP_PREFLIGHT=1 overrides, "
          "and is only honest if you intend to publish memory cells alone.")
    sys.exit(1)
if thermal != "nominal":
    print(f"\nPRE-FLIGHT FAILED: started at '{thermal}', not nominal. Let the phone rest.")
    sys.exit(1)
print("   ok — unplugged and nominal")
PYEOF
fi

log "pre-warm every arm (discarded — absorbs cold load / any re-download)"
for a in "${ARMS[@]}"; do
  IFS='|' read -r rt model <<< "$a"
  log "pre-warm $rt"
  cell "$rt" "$model" short-chat
done

for rep in $(seq 1 "$REPS"); do
  for a in "${ARMS[@]}"; do
    IFS='|' read -r rt model <<< "$a"
    log "rep $rep/$REPS — $rt short-chat"
    cell "$rt" "$model" short-chat
    sleep "${COOLDOWN:-150}"      # thermal rest between cells, not a completion wait
    log "rep $rep/$REPS — $rt long-context-1024-gen256"
    cell "$rt" "$model" long-context-1024-gen256
    sleep "${COOLDOWN:-150}"
  done
done

log "pull results"
rm -rf "$PULL"; mkdir -p "$PULL"
xcrun devicectl device copy from --device "$UDID" --domain-type appDataContainer \
  --domain-identifier "$BUNDLE_ID" --source "Documents/results" --destination "$PULL"

log "verdicts"
python3 "$REPO/scripts/analyze_comparability.py" "$PULL" --since="$RUN_START"
echo
echo "raw results in $PULL   (this run = captures at/after $RUN_START)"
