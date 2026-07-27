#!/usr/bin/env bash
# Gemma-4-E2B iPhone re-capture under the protocol agreed with the LiteRT team.
# Spec: methodology/agreed-protocol-gemma4.md (prefill 1024 via benchmark(), decode 256,
# context forced to 2048, default sampler, card memory definitions, cool start, iOS Metal).
#
#   scripts/bench_gemma4_e2b_protocol_iphone.sh inventory     # what is on device (run FIRST)
#   scripts/bench_gemma4_e2b_protocol_iphone.sh install       # build + install (resets kernel caches!)
#   scripts/bench_gemma4_e2b_protocol_iphone.sh stage         # push MLX + GGUF  ** USB REQUIRED **
#   COREAI_BUNDLE=<dir> ... stage-coreai                     # push the recompiled Core AI bundle ** USB **
#   scripts/bench_gemma4_e2b_protocol_iphone.sh unplug-wait   # poll for unplug, then settle
#   scripts/bench_gemma4_e2b_protocol_iphone.sh warmup        # build kernel caches; DISCARDED
#   scripts/bench_gemma4_e2b_protocol_iphone.sh run-native    # LiteRT vendor-benchmark row (standalone)
#   scripts/bench_gemma4_e2b_protocol_iphone.sh run-cross [arms] # cross-arm columns for the arms given
#                                                              # (default all three; e.g. `run-cross litert llamacpp`)
#   scripts/bench_gemma4_e2b_protocol_iphone.sh run           # cross-arm columns + the Core AI probe
#   scripts/bench_gemma4_e2b_protocol_iphone.sh pull          # pull results (--since filtered)
#   scripts/bench_gemma4_e2b_protocol_iphone.sh analyze       # comparability + reproducibility
#   scripts/bench_gemma4_e2b_protocol_iphone.sh finalize      # pull + filter + import + audit, in one step
#
# Ordering is not cosmetic:
#   * `install` wipes the ML Drift kernel cache, so the runs right after it are FIRST-EVER,
#     not cold — LiteRT prefill reads ~1,650 tok/s there against ~3,234 once the cache is on
#     disk. `warmup` exists to get past that, and its output is thrown away.
#   * `stage` needs the cable; the capture needs it out (charging drives thermal `fair`
#     within ~2 cells). Hence the explicit `unplug-wait` between them.
#   * every cell is launched fresh with `--console`, which returns when the app exits, so
#     nothing is pulled before the runs it is waiting on have landed.
#
# Validation is post-hoc, per the 07-20 lesson: battery-temperature telemetry is a lagging
# long-window average, so gate on the app-recorded initialThermalState/batteryState in the
# result JSON (that is what `analyze` reads), never on a temperature poll here.
set -uo pipefail

# REPO must be overridable. Deriving it from $0 breaks the moment the driver is invoked from
# a copy — which is the recommended way to run it, because editing a script while bash is
# executing it corrupts the running job (bash reads incrementally). Measured 2026-07-27: a
# scratchpad copy silently wrote three cells' console logs into <scratchpad>/results/raw/
# instead of the repo, and the campaign summariser reported the MLX arm as missing.
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
DEV="${DEV:-A6F3E849-1947-5202-9AD1-9C881CA58EEF}"
ECID="${ECID:-00008150-0018713A0207801C}"
APP="${APP:-com.example.CoreMLLLMChat}"
DD="${DD:-$HOME/bench-dd}"
CAMPAIGN="${CAMPAIGN:-$(date +%F)-gemma4-e2b-protocol}"
OUT="$REPO/results/raw/$CAMPAIGN"
STAMP_FILE="$OUT/.session-start"

# Agreed protocol constants — change these only with the doc, not to make a run fit.
CTX=2048            # forced context (Marissa, 7/14: "For the Gemma 4 models, we have been using 2048")
PREFILL=1024        # forced prefill via LiteRT-LM benchmark()
DECODE=256          # decode length
RUNS="${RUNS:-4}"   # per launch; run 1 discarded, median of 2-4 is the warm figure
LAUNCHES="${LAUNCHES:-3}"        # 3 launches x 3 warm runs = n=9 per cell (protocol wants n>=7)
COOLDOWN="${COOLDOWN:-180}"      # idle gap between cells
CELL_TIMEOUT="${CELL_TIMEOUT:-3600}"

LITERT_ID="litert-community/gemma-4-E2B-it-litert-lm"
MLX_ID="mlx-community/gemma-4-e2b-it-4bit"
# MLX checkpoint revision. This is NOT a free choice — only one revision loads, and which one
# flipped when mlx-swift-lm gained Gemma-4 KV sharing:
#   2c3e507 (2026-05-19, "pre-7/6")  2,649 keys — still ships v_proj for the 20 KV-shared
#                                    layers (15..34). Loads on the June checkout ONLY.
#   2387675 (2026-07-06 re-upload)   2,511 keys — v_proj dropped for those layers.
# `Gemma4Text.swift:245` builds `vProj` only when `!useKeqV`, so on a KV-shared layer the
# module has no slot and the extra key fails the load with
#   "Key language_model.model.layers.15.self_attn.v_proj.weight not found in ... Linear"
# — which reads like a MISSING weight but is an EXTRA one. Verified 2026-07-27 by diffing the
# two revisions' safetensors index (138-key delta = 20 layers x 7 quantized-linear keys).
# NOTE: every published MLX E2B cell was measured on 2c3e507. Moving to 2387675 is a
# checkpoint change and has to be labelled as one.
MLX_REV="238767527555cb75a05732a84dff5d6ba0dd6809"
GGUF_ID="unsloth/gemma-4-E2B-it-GGUF/Q4_K_M"
GGUF_REPO="unsloth/gemma-4-E2B-it-GGUF"
GGUF_FILE="gemma-4-E2B-it-Q4_K_M.gguf"
COREAI_ID="core-ai/gemma4-e2b-gpu"
HUB="Library/Caches/huggingface/hub"

have() { xcrun devicectl device info files --device "$DEV" --domain-type appDataContainer \
  --domain-identifier "$APP" --subdirectory "$1" 2>/dev/null; }
copy_to() { xcrun devicectl device copy to --device "$DEV" --domain-type appDataContainer \
  --domain-identifier "$APP" --source "$1" --destination "$2" 2>&1 | grep -iE "File on Device|error" | tail -1; }

# Refuse to measure on the cable. Charging heat drives the thermal state to `fair` within
# ~2 cells, and a fair cell is not comparable with a nominal one — a whole session's worth of
# runs can be lost to this and only discovered post-hoc in the JSON. Set ALLOW_WIRED=1 to
# override deliberately (a plugged run is still a valid capture of something, just not this).
require_unplugged() {
  local transport
  transport="$(xcrun devicectl device info details --device "$DEV" 2>/dev/null \
    | grep 'Transport Type' | awk -F': ' '{print $2}' | tr -d ' ')"
  if [ "$transport" = "wired" ] && [ "${ALLOW_WIRED:-0}" != "1" ]; then
    echo "REFUSING: device is still on the cable (transport=wired)."
    echo "Pull it, wait ~300 s for the thermal state to settle, then re-run."
    echo "(ALLOW_WIRED=1 overrides — the cells will read thermal 'fair' and cannot be ranked.)"
    exit 1
  fi
}

# One cell = one app launch. `--console` blocks until the app exits, which is the only
# reliable completion signal; a fixed sleep pulled results mid-run in earlier passes and
# produced a "llama.cpp failed" conclusion for six cells that landed minutes later.
launch() {
  local label="$1"; shift
  mkdir -p "$OUT"
  echo "--- cell $label"
  gtimeout "$CELL_TIMEOUT" xcrun devicectl device process launch --terminate-existing \
    --console --device "$DEV" "$APP" -- --yardstick-autorun "$@" 2>&1 \
    | tee "$OUT/console_${label}.txt" \
    | grep -E "YARDSTICK_(BEGIN|RUN_OK|NATIVE_OK|RUN_FAIL|FATAL|WARN|ALL_DONE)" || true
}

cmd_inventory() {
  echo "== device"
  xcrun devicectl list devices 2>/dev/null | grep -E "$DEV|connected" | head -3
  xcrun devicectl device info details --device "$DEV" 2>/dev/null | grep -E "OS Build Update|OS Version|Marketing Name"
  echo "== LiteRT .litertlm"
  have "Documents/models/litert-lm/litert-community__gemma-4-E2B-it-litert-lm" | grep -i litertlm || echo "   MISSING"
  echo "== llama.cpp GGUF"
  have "Documents/models/llama.cpp/unsloth__gemma-4-E2B-it-GGUF" | grep -i gguf || echo "   MISSING"
  echo "== MLX hub cache"
  have "$HUB/models--mlx-community--gemma-4-e2b-it-4bit/refs" | grep -iE "main" || echo "   MISSING"
  echo "== Core AI bundle"
  have "Documents/CoreAIModels/gemma4_e2b_gpu" | grep -i metadata || echo "   MISSING (arm will report unavailable)"
  echo "== ML Drift kernel caches in tmp/ (absent => next runs are FIRST-EVER, discard them)"
  have "tmp" | grep -i "gemma-4-E2B.*mldrift" || echo "   none for E2B"
  echo "== XNNPACK cache beside the Mac-side model (must be empty)"
  ls "$HOME"/.cache/huggingface/hub/models--litert-community--*/snapshots/*/*xnnpack_cache* 2>/dev/null || echo "   clean"
}

cmd_install() {
  (cd "$REPO/ios/BenchmarkApp" && xcodegen generate >/dev/null)
  xcodebuild -project "$REPO/ios/BenchmarkApp/BenchmarkApp.xcodeproj" -scheme BenchmarkApp \
    -configuration Release -destination "platform=iOS,id=$ECID" -derivedDataPath "$DD" \
    -skipPackagePluginValidation -skipMacroValidation \
    PRODUCT_BUNDLE_IDENTIFIER="$APP" DEVELOPMENT_TEAM=MFN25KNUGJ CODE_SIGN_STYLE=Automatic \
    build 2>&1 | grep -E "error:|BUILD (SUCCEEDED|FAILED)"
  xcrun devicectl device install app --device "$DEV" \
    "$DD/Build/Products/Release-iphoneos/BenchmarkApp.app" 2>&1 | grep -iE "bundleID|error"
  echo "NOTE: kernel caches are now cold-on-disk. Run 'warmup' before 'run'."
}

cmd_stage() {
  local mac_hub="$HOME/.cache/huggingface/hub" T real f blob
  # MLX: blobs + refs only. The snapshots dir is a farm of relative symlinks and devicectl
  # rejects '..' in a source path; the on-device HF client rebuilds snapshots from refs+blobs.
  local d="$mac_hub/models--mlx-community--gemma-4-e2b-it-4bit"
  local snap="$d/snapshots/$MLX_REV"
  if [ -d "$snap" ]; then
    # Only the blobs THIS revision points at. The Mac cache also holds the 2026-07-06
    # re-upload, and copying the whole blobs/ dir would push both checkpoints (6.7 GB) and
    # leave the wrong one available on device.
    T=$(mktemp -d)/blobs; mkdir -p "$T"
    for f in "$snap"/*; do
      blob="$(readlink "$f" || true)"
      [ -n "$blob" ] || continue                      # plain file, not a cache symlink
      cp -L "$f" "$T/$(basename "$blob")"
    done
    echo "-> MLX blobs for $MLX_REV ($(du -sh "$T" | cut -f1), $(ls "$T" | wc -l | tr -d ' ') files)"
    copy_to "$T" "$HUB/models--mlx-community--gemma-4-e2b-it-4bit/blobs"
    # Pin refs/main to the pre-7/6 revision: the 2026-07-06 re-upload is the checkpoint the
    # vendored mlx-swift-lm loader rejects, and it is also a different checkpoint from the
    # one every published MLX cell was measured on.
    T=$(mktemp -d); printf '%s' "$MLX_REV" > "$T/main"
    echo "-> pin MLX refs/main = $MLX_REV"
    copy_to "$T" "$HUB/models--mlx-community--gemma-4-e2b-it-4bit/refs"
  else
    echo "MISSING Mac cache for MLX — hf download $MLX_ID --revision $MLX_REV"
  fi
  # llama.cpp: single GGUF into the runtime's own directory. Resolve the symlink; devicectl
  # copy-to rejects a '..'-relative symlink as a source (the 4faacef lesson).
  real="$(ls "$mac_hub"/models--unsloth--gemma-4-E2B-it-GGUF/snapshots/*/"$GGUF_FILE" 2>/dev/null | head -1)"
  if [ -n "$real" ]; then
    T=$(mktemp -d)/g; mkdir -p "$T"; cp -L "$real" "$T/"
    echo "-> GGUF $(du -h "$T/$GGUF_FILE" | cut -f1)"
    copy_to "$T" "Documents/models/llama.cpp/unsloth__gemma-4-E2B-it-GGUF"
  else
    echo "MISSING Mac cache for GGUF — hf download $GGUF_REPO $GGUF_FILE"
  fi
}

# Core AI: replace the compiled bundle inside the existing catalog folder.
#
# The 2026-07-03 bundle on device aborts at LOAD on iOS 24A5380h — `expected AICode versioned
# location` / `cannot unwrap empty odiec_module_t`, signal 6 — because coreai-torch 0.4.0 baked
# PyTorch stack traces into the IR as nested MLIR `fused` locations and the beta-2+ compiler no
# longer parses them (apple/coreai-torch#37). The fix used here is Apple's in-place workaround
# (#44): strip the debug locations from the SAME 0.4.0 asset and recompile, rather than
# re-exporting, so the weights and the conversion stay the ones the published 2026-07-18 row
# was measured on. See COREAI_BUNDLE below for how it was produced.
#
# Caveat that must travel with any number from this bundle: the COMPILE stage is not the same.
# The 07-02 bundle carries an AOT-specialized module (MPSGraph package 7.0.56,
# `specialized_model_1.mpsgraph`, 816 KB); the 7.0.63 toolchain emits none. Same source IR,
# different compiled artifact — a lineage note is required, and a speed comparison with the
# 07-18 row is not.
COREAI_BUNDLE="${COREAI_BUNDLE:-}"    # dir containing <name>.h18p.aimodelc
COREAI_DEVICE_DIR="Documents/CoreAIModels/gemma4_e2b_gpu"

cmd_stage_coreai() {
  local b="$COREAI_BUNDLE" name
  [ -n "$b" ] || { echo "set COREAI_BUNDLE=<dir containing the .aimodelc>"; exit 1; }
  name="$(basename "$(ls -d "$b"/*.aimodelc 2>/dev/null | head -1)")"
  [ -n "$name" ] || { echo "no .aimodelc under $b"; exit 1; }
  echo "-> Core AI $name ($(du -sh "$b/$name" | cut -f1))"
  # devicectl cannot delete, and the two bundles' file sets differ (the new one has
  # reflection.fb, the old one specialized_model_1.mpsgraph), so an overwrite leaves the old
  # specialized module behind. It is inert — the new manifest's "Optimized Modules" is empty
  # and never names it — but it IS a stale file, so say so rather than let it be discovered
  # later as a mystery.
  echo "   note: overwrite leaves the old specialized_model_1.mpsgraph in place (unreferenced)"
  copy_to "$b/$name" "$COREAI_DEVICE_DIR/$name"
}

cmd_unplug_wait() {
  echo "Pull the cable now. Charging drives thermal 'fair' within ~2 cells."
  local n=0 transport
  while :; do
    # devicectl keeps the device "connected" over the WiFi tunnel after the cable is out,
    # so `connected` is NOT the unplug signal — Transport Type is (wired -> localNetwork).
    # The authoritative check is still the batteryState recorded in each result JSON, which
    # `analyze` reports; this loop only decides when to start.
    transport="$(xcrun devicectl device info details --device "$DEV" 2>/dev/null \
      | grep 'Transport Type' | awk -F': ' '{print $2}' | tr -d ' ')"
    [ "$transport" = "wired" ] || break
    n=$((n+1)); printf "\r  waiting for unplug… %ds (transport=%s)" $((n*5)) "$transport"; sleep 5
    [ $n -gt 240 ] && { echo; echo "still wired after 20 min — continuing anyway, POST-HOC CHECK REQUIRED"; break; }
  done
  echo "  transport now: ${transport:-unknown}"
  echo; echo "settling 300 s"; sleep 300
  mkdir -p "$OUT"; date -u +%Y-%m-%dT%H:%M:%SZ > "$STAMP_FILE"
  echo "session start stamped: $(cat "$STAMP_FILE")  (use as --since)"
}

# Throwaway. Builds the ML Drift / Metal kernel caches so the measured runs are cold-or-warm
# rather than first-ever. Never imported.
# Every arm, not just LiteRT-LM. Fairness rule 2 splits first-ever from cold for all of them:
# MLX compiles Metal shaders on its first load and llama.cpp builds its own pipeline state, so
# an arm that has never run in this install is in a different regime from one that has. Warming
# only the arm whose cache we happen to know about would bake that asymmetry into the table.
# LiteRT's ML Drift cache lives in the app's tmp/ and SURVIVES an app update — re-warming it is
# cheap and harmless, and skipping it on that assumption is how the regime gets guessed wrong.
cmd_warmup() {
  require_unplugged
  launch "WARMUP_litert" --runtime litert-lm --model-id "$LITERT_ID" \
    --task short-chat --context-tokens "$CTX" --runs 2
  sleep "$COOLDOWN"
  launch "WARMUP_litert_native" --runtime litert-lm --model-id "$LITERT_ID" \
    --litert-native-benchmark "${PREFILL}x${DECODE}" --context-tokens "$CTX"
  sleep "$COOLDOWN"
  launch "WARMUP_mlx" --runtime mlx-swift --model-id "$MLX_ID" \
    --task short-chat --context-tokens "$CTX" --runs 2
  sleep "$COOLDOWN"
  launch "WARMUP_llamacpp" --runtime llama.cpp --model-id "$GGUF_ID" \
    --task short-chat --context-tokens "$CTX" --runs 2
  sleep "$COOLDOWN"
  launch "WARMUP_coreai" --runtime core-ai --model-id "$COREAI_ID" \
    --task short-chat --context-tokens "$CTX" --runs 1
  sleep "$COOLDOWN"
  echo "warmup done — its numbers are FIRST-EVER and must not be imported"
}

# Split out on purpose. The vendor-`benchmark()` row is compared against the model card, not
# against the other arms, so it has no interleaving requirement and no dependency on any other
# arm being staged — it can run while something else is still being sorted out. The cross-arm
# columns are the opposite: they are only meaningful if the arms are captured adjacently, so
# they must not be started until every arm in them is ready.
cmd_run_native() {
  require_unplugged
  mkdir -p "$OUT"
  [ -f "$STAMP_FILE" ] || date -u +%Y-%m-%dT%H:%M:%SZ > "$STAMP_FILE"
  local i
  for i in $(seq 1 7); do
    launch "NATIVE_litert_$i" --runtime litert-lm --model-id "$LITERT_ID" \
      --litert-native-benchmark "${PREFILL}x${DECODE}" --context-tokens "$CTX"
    sleep "$COOLDOWN"
  done
  echo "ALL CELLS DONE (native)"
}

# Run the cross-arm columns for a SUBSET of arms, interleaved within each round.
#
#   cmd_run_cross litert llamacpp        # the two that are staged
#   cmd_run_cross litert mlx llamacpp    # all three
#
# Splitting the set is defensible for the memory column and NOT for a speed ranking, so the
# distinction is written here rather than left to whoever reads the output: footprint is
# session-stable to ~1% (measured across sessions: 3,387 on 07-18 vs 3,388 on 07-26), while
# decode rate is the session- and thermal-sensitive axis. Arms captured in different rounds of
# the SAME session are fine for memory; a decode ranking should come from rounds where the arms
# being compared ran adjacently. `analyze_comparability.py --since` is what decides, per cell.
run_arm() {   # run_arm <label> <runtime> <model-id> <task>
  launch "$1" --runtime "$2" --model-id "$3" --task "$4" --context-tokens "$CTX" --runs "$RUNS"
  sleep "$COOLDOWN"
}

arm_runtime() { case "$1" in litert) echo litert-lm;; mlx) echo mlx-swift;; llamacpp) echo llama.cpp;; coreai) echo core-ai;; esac; }
arm_model()   { case "$1" in litert) echo "$LITERT_ID";; mlx) echo "$MLX_ID";; llamacpp) echo "$GGUF_ID";; coreai) echo "$COREAI_ID";; esac; }

cmd_run_cross() {
  require_unplugged
  mkdir -p "$OUT"
  [ -f "$STAMP_FILE" ] || date -u +%Y-%m-%dT%H:%M:%SZ > "$STAMP_FILE"
  local i a
  local arms=("$@")
  [ ${#arms[@]} -gt 0 ] || arms=(litert mlx llamacpp)
  echo "cross-arm columns for: ${arms[*]}  (${LAUNCHES} rounds x ${RUNS} runs)"

  # TASKS lets a top-up round add launches for one column without re-running the other.
  # A 4-run launch yields only 2 usable runs under the positional rule, so reaching n>=7 needs
  # a 4th launch per cell — and there is no reason to repeat a column that already has enough.
  local tasks=(${TASKS:-depth chat})
  local t
  for t in "${tasks[@]}"; do
    case "$t" in
      depth) for i in $(seq 1 "$LAUNCHES"); do
               for a in "${arms[@]}"; do
                 run_arm "DEPTH_${a}_${ROUND_OFFSET:-0}$i" "$(arm_runtime "$a")" "$(arm_model "$a")" long-context-1024-gen256
               done
             done ;;
      chat)  for i in $(seq 1 "$LAUNCHES"); do
               for a in "${arms[@]}"; do
                 run_arm "CHAT_${a}_${ROUND_OFFSET:-0}$i" "$(arm_runtime "$a")" "$(arm_model "$a")" short-chat
               done
             done ;;
      *) echo "unknown TASKS entry: $t"; exit 1 ;;
    esac
  done
  echo "ALL CELLS DONE (cross-arm: ${arms[*]})"
}

cmd_run() {
  require_unplugged
  mkdir -p "$OUT"
  [ -f "$STAMP_FILE" ] || date -u +%Y-%m-%dT%H:%M:%SZ > "$STAMP_FILE"
  local i

  # 1. The cross-arm deep-context column: one instrument for every arm, p~1024 / gen 256,
  #    same forced context. LiteRT-LM is included here too — a column is only comparable if
  #    every cell in it came from the same harness.
  for i in $(seq 1 "$LAUNCHES"); do
    launch "DEPTH_litert_$i"   --runtime litert-lm --model-id "$LITERT_ID" \
      --task long-context-1024-gen256 --context-tokens "$CTX" --runs "$RUNS"
    sleep "$COOLDOWN"
    launch "DEPTH_mlx_$i"      --runtime mlx-swift --model-id "$MLX_ID" \
      --task long-context-1024-gen256 --context-tokens "$CTX" --runs "$RUNS"
    sleep "$COOLDOWN"
    launch "DEPTH_llamacpp_$i" --runtime llama.cpp --model-id "$GGUF_ID" \
      --task long-context-1024-gen256 --context-tokens "$CTX" --runs "$RUNS"
    sleep "$COOLDOWN"
  done

  # 2. short-chat companions at the same forced context, so the ① column and the ④ column
  #    differ only in prompt depth rather than in context budget as well.
  for i in $(seq 1 "$LAUNCHES"); do
    launch "CHAT_litert_$i"   --runtime litert-lm --model-id "$LITERT_ID" \
      --task short-chat --context-tokens "$CTX" --runs "$RUNS"
    sleep "$COOLDOWN"
    launch "CHAT_mlx_$i"      --runtime mlx-swift --model-id "$MLX_ID" \
      --task short-chat --context-tokens "$CTX" --runs "$RUNS"
    sleep "$COOLDOWN"
    launch "CHAT_llamacpp_$i" --runtime llama.cpp --model-id "$GGUF_ID" \
      --task short-chat --context-tokens "$CTX" --runs "$RUNS"
    sleep "$COOLDOWN"
  done

  # 3. Core AI: the published claim is that it cannot finish this job. A claim of failure
  #    needs a captured failure, not an absence of rows — one launch, whatever it prints.
  launch "DEPTH_coreai_1" --runtime core-ai --model-id "$COREAI_ID" \
    --task long-context-1024-gen256 --context-tokens "$CTX" --runs 1

  echo "ALL CELLS DONE — pull, then analyze"
}

cmd_pull() {
  mkdir -p "$OUT/pulled"
  # The device keeps every capture it has ever taken and a pull copies all of them; the
  # --since filter in `analyze` is what keeps earlier sessions out of the medians.
  xcrun devicectl device copy from --device "$DEV" --domain-type appDataContainer \
    --domain-identifier "$APP" --source "Documents/results" --destination "$OUT/pulled" \
    2>&1 | tail -2
  echo "pulled $(find "$OUT/pulled" -name '*.json' | wc -l | tr -d ' ') result files"
  [ -f "$STAMP_FILE" ] && echo "session start: $(cat "$STAMP_FILE")"
}

# Pull -> filter -> import -> audit, in the order that keeps contamination out.
#
# The filtering is the point. A pull copies every capture the device has ever taken (576 files
# predating this session on 2026-07-27), and the session's own warm-up cells carry an older
# harness stamp than its data. Two filters, not one: `--since` for other sessions, the stamp
# for other contracts inside this one.
cmd_finalize() {
  local stamp="${HARNESS_STAMP:-2026-07-27-agreed-protocol-r2}"
  local since; since="$(cat "$STAMP_FILE" 2>/dev/null || echo 1970-01-01T00:00:00Z)"
  mkdir -p "$OUT/pulled" "$OUT/app-path"

  echo "== pull"
  xcrun devicectl device copy from --device "$DEV" --domain-type appDataContainer \
    --domain-identifier "$APP" --source "Documents/results" --destination "$OUT/pulled" \
    2>&1 | tail -1

  echo "== keep only this session's cells on this harness"
  python3 - "$OUT/pulled" "$OUT/app-path" "$since" "$stamp" <<'PY'
import json, shutil, sys
from pathlib import Path
src, dst, since, stamp = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
kept = dropped = 0
for p in src.rglob("*.json"):
    try:
        o = json.loads(p.read_text())
    except Exception:
        continue
    if not (isinstance(o, dict) and "metrics" in o):
        continue
    if (o.get("timestamp") or "") < since or o["metrics"].get("harnessStamp") != stamp:
        dropped += 1
        continue
    shutil.copy(p, dst / p.name)
    kept += 1
print(f"   kept {kept}, dropped {dropped} (earlier sessions / other harness stamps)")
PY

  echo "== lift the native rows out of their console logs"
  python3 "$REPO/scripts/import_native_benchmark.py" "$OUT"/console_NATIVE_*.txt 2>&1 | tail -1

  echo "== comparability + reproducibility"
  python3 "$REPO/scripts/analyze_comparability.py" "$OUT/app-path" "--harness=$stamp"

  echo "== campaign summary (console logs, positional rule)"
  python3 "$REPO/scripts/summarize_gemma4_campaign.py" "$OUT"

  echo "== every number in the campaign README must trace to a file"
  python3 "$REPO/scripts/verify_published_numbers.py" "$OUT/README.md" --results "$REPO/results" \
    2>&1 | tail -5
}

cmd_analyze() {
  local since=""
  [ -f "$STAMP_FILE" ] && since="--since=$(cat "$STAMP_FILE")"
  python3 "$REPO/scripts/analyze_comparability.py" "$OUT/pulled" $since
}

case "${1:-}" in
  inventory)    cmd_inventory ;;
  run-native)   cmd_run_native ;;
  run-cross)    shift; cmd_run_cross "$@" ;;
  install)      cmd_install ;;
  stage)        cmd_stage ;;
  stage-coreai) cmd_stage_coreai ;;
  unplug-wait)  cmd_unplug_wait ;;
  warmup)       cmd_warmup ;;
  run)          cmd_run ;;
  pull)         cmd_pull ;;
  analyze)      cmd_analyze ;;
  finalize)     cmd_finalize ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
