#!/usr/bin/env bash
# Warm-matrix driver for iPhone 17 Pro — fixes the two v1 defects that made the
# 2026-07-13 Tier-1 capture unusable (results/raw/2026-07-13-iphone-warm-partial/):
#   1. device-jsonl copy-back (v1 kept console only; the app's per-run JSON under
#      Documents/results/ was never pulled) — pattern from bench_cactus_parity_iphone.sh.
#   2. thermal gate: every run of a cell must report initialThermalState==nominal
#      (fairness-rules.md §2); a cell that ran hot cools 240 s and re-runs once.
#
# Protocol per cell: ONE process launch with --runs 4. Run 1 = cold (fresh process,
# caches on disk), runs 2-4 = warm; warm headline = median of runs 2-4 (coldRun=false
# in the device JSON is authoritative, not the run index).
#
#   scripts/bench_warm_matrix_iphone.sh stage    # side-load Core AI 0.6B/1.7B + LiteRT 1.7B int4
#   scripts/bench_warm_matrix_iphone.sh run      # tier-1 cells (or CELLS_FILE=... for others)
#   scripts/bench_warm_matrix_iphone.sh pull     # copy-back only (recover an aborted session)
#   scripts/bench_warm_matrix_iphone.sh report   # cold/warm table from the campaign dir
#
# Custom cells: CELLS_FILE=path (lines: "<runtime> <model-id>", # comments ok).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEV="${DEV:-A6F3E849-1947-5202-9AD1-9C881CA58EEF}"   # devicectl id, iPhone 17 Pro
APP="${APP:-com.example.CoreMLLLMChat}"              # borrowed App ID (memory entitlements)
RUNS="${RUNS:-4}"
TASK="${TASK:-short-chat}"
BASE_COOLDOWN="${BASE_COOLDOWN:-100}"                # s between cells (fairness-rules §2)
THERMAL_COOLDOWN="${THERMAL_COOLDOWN:-240}"          # s before the one thermal re-run
CAMPAIGN="${CAMPAIGN:-$(date +%F)-iphone-warm}"
OUT="$REPO/results/raw/$CAMPAIGN"
EXPORTS="$HOME/code/coreai/coreai-models/exports"
LITERT_1_7B="$HOME/code/litertlm-convert/out/qwen3_1_7b_mixed4/model.litertlm"

# Tier 1: the cells whose cold rows are already published (0.6B / 1.7B).
# core-ai/qwen3-1.7b-ane is intentionally absent: documented invoke-fail (bd71203).
TIER1_CELLS='
mlx-swift mlx-community/Qwen3-0.6B-4bit
litert-lm litert-community/Qwen3-0.6B
core-ai core-ai/qwen3-0.6b-ane
core-ai core-ai/qwen3-0.6b-gpu
mlx-swift mlx-community/Qwen3-1.7B-4bit
litert-lm litert-local/qwen3-1.7b-int4
core-ai core-ai/qwen3-1.7b-gpu
'

log(){ printf '\n=== %s ===\n' "$*"; }

copy_to(){ xcrun devicectl device copy to --device "$DEV" --domain-type appDataContainer \
  --domain-identifier "$APP" --source "$1" --destination "$2" 2>&1 | grep -iE "File on Device|error" | tail -1; }

# Pull Documents/results and copy any device JSON newer than the campaign stamp
# into $OUT/device-jsonl/ (basenames embed a UTC timestamp, so they are unique
# keys). Time gate = `-newer <stamp file>`: BSD find has no @epoch syntax.
STAMP(){ echo "$OUT/.campaign_start"; }
pull_new(){
  local tmp; tmp="$(mktemp -d)"
  xcrun devicectl device copy from --device "$DEV" --domain-type appDataContainer \
    --domain-identifier "$APP" --source Documents/results --destination "$tmp" >/dev/null 2>&1
  mkdir -p "$OUT/device-jsonl"
  local n=0 f base
  while IFS= read -r -d '' f; do
    base="$(basename "$f")"
    [ -e "$OUT/device-jsonl/$base" ] || { cp "$f" "$OUT/device-jsonl/$base"; n=$((n+1)); }
  done < <(find "$tmp" -name "*.json" -newer "$(STAMP)" -print0 2>/dev/null)
  rm -rf "$tmp"
  echo "$n"
}

# Thermal + completeness verdict for the newest $RUNS files of a cell.
# Prints "OK" / "HOT <states>" / "SHORT <count>"; relies on metrics.coldRun ordering.
cell_verdict(){ # <runtime> <model-id>
  RT="$1" MID="$2" OUT="$OUT" RUNS="$RUNS" python3 - <<'PY'
import json, os, glob
rt, mid, runs = os.environ["RT"], os.environ["MID"], int(os.environ["RUNS"])
pat = rt + "_" + mid.replace("/", "_") + "_" + os.environ.get("TASK", "short-chat") + "_*.json"
files = sorted(glob.glob(os.path.join(os.environ["OUT"], "device-jsonl", pat)))[-runs:]
if len(files) < runs:
    print(f"SHORT {len(files)}"); raise SystemExit
states = [json.load(open(f))["metrics"].get("initialThermalState", "?") for f in files]
print("OK" if all(s == "nominal" for s in states) else "HOT " + ",".join(states))
PY
}

run_cell(){ # <runtime> <model-id>
  local rt="$1" mid="$2" logf="$OUT/console_$(echo "$1_$2" | tr '/.' '__').txt"
  log "CELL $rt / $mid ($(date +%H:%M:%S))"
  # </dev/null: devicectl --console forwards stdin, which would otherwise swallow
  # the rest of the cell list feeding cmd_run's while-read loop.
  # gtimeout: LiteRT-LM can hang at teardown (callback_thread_pool DEADLINE_EXCEEDED)
  # and never exit, which would block --console forever; completed runs are already
  # persisted on-device, so killing the console after CELL_TIMEOUT is lossless.
  gtimeout --kill-after=30 "${CELL_TIMEOUT:-900}" \
    xcrun devicectl device process launch --console --terminate-existing --device "$DEV" "$APP" \
    -- --yardstick-autorun --runtime "$rt" --model-id "$mid" --task "$TASK" --runs "$RUNS" </dev/null 2>&1 \
    | tee -a "$logf" | grep -E "YARDSTICK_(BEGIN|RUN_OK|RUN_FAIL|FATAL|ALL_DONE)" || true
}

cmd_stage(){
  log "side-load Core AI 0.6B/1.7B bundles + LiteRT 1.7B int4"
  for b in qwen3_0_6b_ane_pure4bit:qwen3_0_6b_ane qwen3_0_6b_gpu:qwen3_0_6b_gpu \
           qwen3_1_7b_gpu:qwen3_1_7b_gpu; do
    src="$EXPORTS/${b%%:*}"; dst="Documents/CoreAIModels/${b##*:}"
    [ -f "$src/metadata.json" ] || { echo "SKIP $src (not assembled yet)"; continue; }
    echo "-> $dst"; copy_to "$src" "$dst"
  done
  if [ -f "$LITERT_1_7B" ]; then
    stage="$(mktemp -d)/m"; mkdir -p "$stage"; cp -L "$LITERT_1_7B" "$stage/"
    # devicectl copies the source dir's CONTENTS; destination must be the full repo dir.
    echo "-> models/litert-lm/litert-local__Qwen3-1.7B-int4"
    copy_to "$stage" "Documents/models/litert-lm/litert-local__Qwen3-1.7B-int4"
  else
    echo "SKIP litert 1.7B ($LITERT_1_7B missing)"
  fi
}

cmd_run(){
  mkdir -p "$OUT/device-jsonl"
  # Stamp once per campaign (kept across re-invocations, so an interrupted campaign
  # resumed later still picks up every cell captured since the campaign began).
  [ -f "$(STAMP)" ] || touch "$(STAMP)"
  export TASK
  local cells; cells="$(grep -vE '^\s*(#|$)' <<<"${CELLS:-$(cat "${CELLS_FILE:-/dev/null}" 2>/dev/null || echo "$TIER1_CELLS")}")"
  local first=1
  while read -r rt mid; do
    [ "$first" = 1 ] && first=0 || { log "cooldown ${BASE_COOLDOWN}s"; sleep "$BASE_COOLDOWN"; }
    run_cell "$rt" "$mid"
    local pulled verdict
    pulled="$(pull_new)"; verdict="$(cell_verdict "$rt" "$mid")"
    echo "pulled=$pulled verdict=$verdict"
    if [[ "$verdict" == HOT* ]]; then
      log "thermal gate: $verdict — cooldown ${THERMAL_COOLDOWN}s, re-run once"
      sleep "$THERMAL_COOLDOWN"
      run_cell "$rt" "$mid"
      pulled="$(pull_new)"; verdict="$(cell_verdict "$rt" "$mid")"
      echo "pulled=$pulled verdict=$verdict"
      [[ "$verdict" == HOT* ]] && echo "THERMAL_FAIL $rt $mid (kept both captures; exclude from warm table)" | tee -a "$OUT/THERMAL_FAIL.txt"
    fi
  done <<<"$cells"
  cmd_report
}

cmd_pull(){
  mkdir -p "$OUT"
  # No stamp yet (recovery of an aborted session) -> pull everything (1970 stamp).
  [ -f "$(STAMP)" ] || touch -t 197001010000 "$(STAMP)"
  echo "pulled $(pull_new) new files -> $OUT/device-jsonl"
}

cmd_report(){
  OUT="$OUT" python3 - <<'PY'
import json, glob, os, re, statistics, collections
out = os.environ["OUT"]
cells = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(out, "device-jsonl", "*.json"))):
    d = json.load(open(f))
    key = (d["runtime"], d["model"]["id"], d["task"])
    cells[key].append(d)
lines = ["| runtime | model | task | cold (run1) | warm (med r2-4) | n | thermal |",
         "|---|---|---|---|---|---|---|"]
for (rt, mid, task), rows in sorted(cells.items()):
    rows.sort(key=lambda d: d["timestamp"])
    cold = [r for r in rows if r["metrics"].get("coldRun")]
    warm = [r for r in rows if not r["metrics"].get("coldRun")]
    cold_s = f"{cold[-1]['metrics']['decodeTokensPerSecond']:.1f}" if cold else "—"
    wtps = [r["metrics"]["decodeTokensPerSecond"] for r in warm[-3:]]
    warm_s = f"{statistics.median(wtps):.1f}" if wtps else "—"
    therm = sorted({r["metrics"].get("initialThermalState", "?") for r in rows})
    lines.append(f"| {rt} | {mid} | {task} | {cold_s} | {warm_s} | {len(rows)} | {','.join(therm)} |")
report = "\n".join(lines) + "\n"
open(os.path.join(out, "summary.md"), "w").write(report)
print(report)
PY
}

case "${1:-}" in
  stage)  cmd_stage ;;
  run)    cmd_run ;;
  pull)   cmd_pull ;;
  report) cmd_report ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
