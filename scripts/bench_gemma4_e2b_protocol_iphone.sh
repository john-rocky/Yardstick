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
#   CACTUS_SRC=<dir> ... stage-cactus                        # push the CQ bundles ** USB **
#   scripts/bench_gemma4_e2b_protocol_iphone.sh run-block A1 # ONE comparison group (~90 min)
#        blocks: A1 depth[litert mlx optiq]      A2 chat[same]
#                B1 depth[litert llamacpp cactus] B2 chat[same]
#                C  chat[litert coreai cactus-shipped]   N native[litert]   E energy[all]
#        LiteRT-LM is the anchor in every block; cross-block ratios go through it.
#        Run ONE block per sitting and let the device cool -- do not chain two.
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
# RUNS=3, not 4. The positional rule keeps runs 2-3 and drops run 1 (cold) and run 4+ (the
# thermally degraded tail), so a 4-run launch and a 3-run launch yield the SAME two usable runs
# -- the 4th is measured, discarded, and its heat carried into the next cell. Measured
# 2026-07-27: run 4 read -12.9%, -23.3%, -25.0% (LiteRT-LM) and -7.1%, -11.0% (llama.cpp)
# against run 2 of its own launch. Dropping it costs nothing and buys ~25% of the session's
# thermal budget, which is the binding constraint on how many arms fit in one block.
RUNS="${RUNS:-3}"                # per launch; run 1 cold, runs 2-3 are the warm pair
LAUNCHES="${LAUNCHES:-4}"        # 4 launches x 2 warm runs = n=8 per cell (protocol wants n>=7)
COOLDOWN="${COOLDOWN:-300}"      # base idle gap; grows with session age, see cool()
COOLDOWN_MAX="${COOLDOWN_MAX:-600}"
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
OPTIQ_ID="mlx-community/gemma-4-e2b-it-qat-OptiQ-4bit"
# Cactus bundles are sideloaded, not fetched: the CQ zips are not usefully per-file
# downloadable through the HF snapshot API, and reinstalling the app wipes the Data container
# that holds them. `stage-cactus` pushes them; without it the arm reports no bundle and the
# cell is lost silently 40 minutes later.
CACTUS_ID="Cactus-Compute/gemma-4-E2B-it-cq4-uncalibrated"
CACTUS_SHIPPED_ID="Cactus-Compute/gemma-4-E2B-it-cq4"
CACTUS_SRC="${CACTUS_SRC:-$HOME/code/cactus/weights}"
GGUF_ID="unsloth/gemma-4-E2B-it-GGUF/Q4_K_M"
GGUF_REPO="unsloth/gemma-4-E2B-it-GGUF"
GGUF_FILE="gemma-4-E2B-it-Q4_K_M.gguf"
COREAI_ID="core-ai/gemma4-e2b-gpu"
HUB="Library/Caches/huggingface/hub"

# NOTE: never pipe `have` into `grep -q` while pipefail is set. Big listings (the cactus
# bundle is ~1,500 files, well past the 64 KB pipe buffer) make devicectl take SIGPIPE when
# grep -q exits at first match, and pipefail then reports the check as FAILED — a present
# model reads as MISSING (cost three phone-chain relaunches on 07-29). Plain `grep` (which
# drains the stream) or a `case "$(have …)" in *pat*)` substring check are both safe.
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
    | grep -E "YARDSTICK_(BEGIN|RUN_OK|NATIVE_OK|RUN_FAIL|FATAL|WARN|ALL_DONE|ENERGY|THERMAL_DEFER)" || true
}

# Energy cells must start `nominal` (agreed protocol; the 07-19 headline cells started `fair`
# and read ~25% low). The gate lives in the app — it checks its own thermal state at launch
# and defers (sentinel + exit 7) BEFORE loading a model, because battery-temperature telemetry
# here lags by ~30 min (the 07-20 lesson). This wrapper retries a deferred cell after a full
# cool; three deferrals mean the sitting is spent — record the skip, never substitute a number.
# Retries get their own console files so the deferrals stay on disk for the audit.
launch_gated() {   # launch_gated <label> <launch args...>
  local label="$1"; shift
  local attempt alabel
  for attempt in 1 2 3; do
    alabel="$label"; [ "$attempt" -gt 1 ] && alabel="${label}_retry$((attempt-1))"
    launch "$alabel" "$@"
    grep -q "YARDSTICK_THERMAL_DEFER" "$OUT/console_${alabel}.txt" 2>/dev/null || return 0
    echo "    thermal gate: $label deferred (attempt $attempt) — cooling ${COOLDOWN_MAX}s"
    sleep "$COOLDOWN_MAX"
  done
  echo "    thermal gate: $label SKIPPED after 3 deferrals (device will not reach nominal this sitting)"
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
  echo "== MLX-OptiQ hub cache (blocks A1/A2 need it)"
  have "$HUB/models--mlx-community--gemma-4-e2b-it-qat-OptiQ-4bit/refs" | grep -iE "main" || echo "   MISSING"
  echo "== Cactus CQ bundles (blocks C/E need them; sideloaded, wiped with the Data container)"
  have "Documents/models/cactus/Cactus-Compute__gemma-4-E2B-it" | grep -iE "config.txt|cq4" | head -4 || echo "   MISSING"
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
  # Refuse to install a product whose harness stamp is not the source's. This function has
  # no set -e: on 2026-07-30 an iOS-unavailable API (homeDirectoryForCurrentUser) failed the
  # build, the install step then pushed the STALE r3 .app still sitting in DerivedData, and
  # only the stamp on the warmup console caught it — a device sitting later. Check the
  # binary, not the build log.
  local src_stamp
  src_stamp="$(sed -n 's/.*harnessStamp = "\([^"]*\)".*/\1/p' \
    "$REPO/ios/BenchmarkApp/Sources/Benchmark/BenchmarkRunner.swift" | head -1)"
  # (plain grep, not -q: strings output is way past the 64 KB pipe buffer — see the have() note)
  if [ -n "$src_stamp" ] && ! strings "$DD/Build/Products/Release-iphoneos/BenchmarkApp.app/BenchmarkApp" \
      2>/dev/null | grep -F "$src_stamp" >/dev/null; then
    echo "REFUSING install: source stamp '$src_stamp' not found in the built binary (stale build?)"
    return 1
  fi
  xcrun devicectl device install app --device "$DEV" \
    "$DD/Build/Products/Release-iphoneos/BenchmarkApp.app" 2>&1 | grep -iE "bundleID|error"
  echo "NOTE: kernel caches are now cold-on-disk. Run 'warmup' before 'run'."
}

stage_hub_model() {   # stage_hub_model <repo-id> <revision>
  # Blobs + refs only. The snapshots dir is a farm of relative symlinks and devicectl
  # rejects '..' in a source path; the on-device HF client rebuilds snapshots from refs+blobs.
  # Only the blobs THE revision points at: a cache can hold several checkpoints under one repo
  # id (the MLX repo holds both the pre-7/6 upload and the 07-06 re-upload, 6.7 GB together),
  # and copying the whole blobs/ dir would push all of them and leave the wrong one loadable.
  # Pinning refs/main is what makes the load deterministic AND what the result JSON's
  # modelRevision field reads back, so a staged cell records the checkpoint it really ran.
  local repo_id="$1" rev="$2"
  local cache_name="models--$(echo "$repo_id" | sed 's|/|--|')"
  local snap="$HOME/.cache/huggingface/hub/$cache_name/snapshots/$rev"
  local T f blob
  if [ ! -d "$snap" ]; then
    echo "MISSING Mac cache for $repo_id — hf download $repo_id --revision $rev"
    return 1
  fi
  T=$(mktemp -d)/blobs; mkdir -p "$T"
  for f in "$snap"/*; do
    blob="$(readlink "$f" || true)"
    [ -n "$blob" ] || continue                      # plain file, not a cache symlink
    cp -L "$f" "$T/$(basename "$blob")"
  done
  echo "-> $repo_id blobs for $rev ($(du -sh "$T" | cut -f1), $(ls "$T" | wc -l | tr -d ' ') files)"
  copy_to "$T" "$HUB/$cache_name/blobs"
  T=$(mktemp -d); printf '%s' "$rev" > "$T/main"
  echo "-> pin $repo_id refs/main = $rev"
  copy_to "$T" "$HUB/$cache_name/refs"
}

# OptiQ revision, pinned for the same reason as MLX_REV: an unpinned load records nothing
# and re-resolves on HF's schedule, not ours.
OPTIQ_REV="${OPTIQ_REV:-ed4ba3d83bcf2ad004369bc5c9b2bb53aadbdb1f}"

cmd_stage() {
  local mac_hub="$HOME/.cache/huggingface/hub" T real
  # MLX_REV is the 2026-07-06 re-upload (2387675…). This checkout's vendored mlx-swift-lm has
  # Gemma-4 KV sharing, which builds no vProj slot on the 20 KV-shared layers — so the OLDER
  # 2c3e507 upload (which still ships v_proj for them) fails its load with an extra-key error,
  # and 2387675 is the only revision that loads here. Every pre-07-27 published MLX cell was
  # measured on 2c3e507; the move is a checkpoint change and is labelled as one.
  stage_hub_model "$MLX_ID" "$MLX_REV"
  # MLX-OptiQ — the same runtime's QAT recipe row (defect 4 needs both MLX rows in one session).
  stage_hub_model "$OPTIQ_ID" "$OPTIQ_REV"
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

# Cactus is the only arm that is not fetched from HF by the app. Its CQ bundles are zips of a
# directory (graph + manifest + config.txt) and are not usefully per-file downloadable through
# the snapshot API, so they are pushed by hand -- and `install` wipes the Data container that
# holds them, so this has to run AFTER every reinstall. Diagnosed 2026-07-27 the hard way: eight
# launches produced zero result files and the reason ("No Cactus bundle (config.txt) under
# .../models--Cactus-Compute--gemma-4-E2B-it/snapshots/...") was only visible with the console
# attached. HFDownloader.snapshot short-circuits on a non-empty directory, which is what makes
# the sideload work at all.
#
#   CACTUS_SRC=~/code/cactus/weights scripts/... stage-cactus     ** USB REQUIRED, ~7.6 GB **
cmd_stage_cactus() {
  # The app's layout is HFDownloader.modelDirectory: models/<runtime>/<repo-id with "/"
  # replaced by "__">. The first version of this command pushed to a NESTED
  # Cactus-Compute/gemma-4-E2B-it/ path, which HFDownloader never looks at — the arm then
  # fell through to a hub download and failed exactly like the 2026-07-27 silent loss this
  # command exists to prevent (caught in warmup, 2026-07-28 16:0x, 7.7 GB re-push).
  local dst_root="Documents/models/cactus/Cactus-Compute__gemma-4-E2B-it"
  local b found=0
  for b in gemma-4-e2b-it-cq4-uncalibrated gemma-4-e2b-it-cq4; do
    local src="$CACTUS_SRC/$b"
    if [ ! -d "$src" ]; then
      echo "MISSING: $src"
      echo "  the bundle is a directory (config.txt + graph + manifest), not a file."
      echo "  unzip the CQ bundle from huggingface.co/Cactus-Compute/gemma-4-E2B-it into \$CACTUS_SRC."
      continue
    fi
    if [ ! -f "$src/config.txt" ]; then
      echo "REFUSING $b: no config.txt at its root -- that is the file the runtime probes for,"
      echo "  so pushing this directory would reproduce the silent 2026-07-27 failure."
      continue
    fi
    echo "--- pushing $b ($(du -sh "$src" | cut -f1))"
    copy_to "$src" "$dst_root/$b"
    found=$((found+1))
  done
  [ "$found" -gt 0 ] || { echo "nothing staged"; exit 1; }
  echo "staged $found Cactus bundle(s); verify with: $0 inventory"
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
  # OptiQ and Cactus too — they were absent from the original warmup list because they were
  # absent from the checkout, and a first-ever load inside block A1/C would put one arm in a
  # different regime from its rivals mid-block. OptiQ shares the MLX shader cache but not the
  # weight load; Cactus builds its own engine cache. A launch that fails because the model is
  # not staged yet is loud in the console and harmless — output is discarded either way.
  launch "WARMUP_optiq" --runtime mlx-swift --model-id "$OPTIQ_ID" \
    --task short-chat --context-tokens "$CTX" --runs 2
  sleep "$COOLDOWN"
  launch "WARMUP_cactus" --runtime cactus --model-id "$CACTUS_SHIPPED_ID" \
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

arm_runtime() { case "$1" in
  litert)   echo litert-lm ;;
  mlx)      echo mlx-swift ;;
  optiq)    echo mlx-swift ;;
  llamacpp) echo llama.cpp ;;
  coreai)   echo core-ai ;;
  cactus|cactus-shipped) echo cactus ;;
  *) echo "unknown arm: $1" >&2; return 1 ;;
esac; }
arm_model() { case "$1" in
  litert)   echo "$LITERT_ID" ;;
  mlx)      echo "$MLX_ID" ;;
  optiq)    echo "$OPTIQ_ID" ;;
  llamacpp) echo "$GGUF_ID" ;;
  coreai)   echo "$COREAI_ID" ;;
  cactus)   echo "$CACTUS_ID" ;;
  cactus-shipped) echo "$CACTUS_SHIPPED_ID" ;;
  *) echo "unknown arm: $1" >&2; return 1 ;;
esac; }

# Which arms can appear in which column. Encoded here rather than left to whoever writes the
# block list, because two of the exclusions are findings and must not be silently re-attempted:
#   * Core AI cannot take a ~1,098-token prompt at all -- iOS caps its dynamic-KV capacity at
#     1024 upstream (coreai-models CoreAIPipelinedEngine.swift:1416, apple/coreai-models#124),
#     so the deep-context cell is refused before allocation, not slow.
#   * llama.cpp's official QAT q4_0 build aborts on a vocab defect (empty token) through b10064.
arm_can_depth() { case "$1" in coreai) return 1 ;; *) return 0 ;; esac; }
# Core AI cannot run the EQUALIZED energy protocol either (measured 2026-07-29 05:5x):
# the sustain task's per-call maxTokens=2048 exceeds the iOS KV cap (1024, same upstream
# limit as the depth exclusion) and the app dies at setup with signal 9 (jetsam) before
# emitting a single run. The 2026-07-19 Core AI energy row only existed because that
# session gave it maxTokens=192 while every other arm ran 2048 — the exact per-arm
# asymmetry this block was rebuilt to remove. A finding, not a failure; do not retry.
arm_can_energy() { case "$1" in coreai) return 1 ;; *) return 0 ;; esac; }

# A fixed inter-cell gap is wrong for a long session and the 2026-07-27 capture proved it:
# 180 s was enough for the first ~90 minutes, and after ~2 hours the device no longer recovered
# inside it -- LiteRT-LM's run 3, which had stayed nominal in every earlier round, throttled to
# 47.4 and the whole launch had to be discarded. That is session-level heat, and the positional
# rule cannot catch it because run 3 is inside the keep-window. So the gap grows with how long
# this block has been running.
BLOCK_START_EPOCH=""
cool() {
  local base="$COOLDOWN" age=0
  [ -n "$BLOCK_START_EPOCH" ] && age=$(( $(date +%s) - BLOCK_START_EPOCH ))
  # +60 s per 30 minutes of block age, capped.
  local grown=$(( base + (age / 1800) * 60 ))
  [ "$grown" -gt "$COOLDOWN_MAX" ] && grown="$COOLDOWN_MAX"
  echo "    (cooling ${grown}s; block age $((age/60)) min)"
  sleep "$grown"
}

# One block = one comparison group. Everything inside it is captured adjacently, so its cells
# may be ranked against each other; cells in different blocks may not, EXCEPT through the
# anchor.
#
# THE ANCHOR. LiteRT-LM runs in every block. That is not redundancy -- it is the only thing
# that makes two blocks comparable at all. Device state drifts between sessions by far more
# than the effects being measured (measured on this repo: the same binary and pin read 126-133
# tok/s in June and 159-180 in July), so a cross-block ratio has to go through an arm that ran
# in both, and the anchor's own drift is then a measured quantity rather than an assumption.
# The published table already does this once, in footnote (g), for the Cactus row.
#
#   run-block A1     # depth : anchor + the two MLX builds
#   run-block A2     # chat  : same three
#   run-block B1     # depth : anchor + llama.cpp + Cactus
#   run-block B2     # chat  : same three
#   run-block C      # chat  : anchor + Core AI + Cactus-shipped  (Core AI has no depth cell)
#   run-block N      # LiteRT vendor benchmark() row
#   run-block E      # energy, every arm -- LAST, see below
#
# Blocks are sized to ~90 minutes so they end before the thermal budget does. Run one per
# sitting and let the device cool between them; do not chain two.
#
# Energy is deliberately last. At 600 s per run it is the most expensive axis and the least
# comparative one, so if the campaign runs out of time the right outcome is that every energy
# cell keeps its 2026-07-19 date, not that half of them are new.
block_arms() { case "$1" in
  A1|A2) echo "litert mlx optiq" ;;
  B1|B2) echo "litert llamacpp cactus" ;;
  C)     echo "litert coreai cactus-shipped" ;;
  E)     echo "litert mlx optiq llamacpp coreai cactus cactus-shipped" ;;
  *) return 1 ;;
esac; }
block_task() { case "$1" in
  A1|B1) echo depth ;;
  A2|B2|C) echo chat ;;
  E) echo energy ;;
  *) return 1 ;;
esac; }

cmd_run_block() {
  local blk="${1:-}"
  [ -n "$blk" ] || { echo "usage: run-block <A1|A2|B1|B2|C|N|E>"; exit 2; }
  require_unplugged
  mkdir -p "$OUT"
  [ -f "$STAMP_FILE" ] || date -u +%Y-%m-%dT%H:%M:%SZ > "$STAMP_FILE"
  BLOCK_START_EPOCH=$(date +%s)

  if [ "$blk" = "N" ]; then cmd_run_native; return; fi

  local arms task i a
  arms="$(block_arms "$blk")" || { echo "unknown block: $blk"; exit 2; }
  task="$(block_task "$blk")"
  echo "=== block $blk : task=$task arms=[$arms] launches=$LAUNCHES runs=$RUNS"
  echo "    anchor = litert (present in every block; cross-block ratios go through it)"

  local tid
  case "$task" in
    depth)  tid=long-context-1024-gen256 ;;
    chat)   tid=short-chat ;;
    energy) tid=energy ;;
  esac

  for i in $(seq 1 "$LAUNCHES"); do
    for a in $arms; do
      if [ "$task" = depth ] && ! arm_can_depth "$a"; then
        echo "--- skip ${a} at depth (structurally impossible, not a failure -- see arm_can_depth)"
        continue
      fi
      if [ "$task" = energy ] && ! arm_can_energy "$a"; then
        echo "--- skip ${a} at energy (structurally impossible, not a failure -- see arm_can_energy)"
        continue
      fi
      # Energy is a 600 s sustained measurement; one run per launch, not RUNS, and it
      # goes through the thermal gate (nominal start or defer-and-retry). One task, one
      # default 600 s sustain, no per-arm override — on 07-19 the only real per-arm
      # asymmetry was Core AI's per-call maxTokens=192 (others 2048); the differing
      # battery deltas (5% vs 10%) were an OUTCOME of power draw at 1% battery
      # resolution, not a per-arm window setting (audited 2026-07-28: all six cells ran
      # the same ~600 s sustain).
      local r="$RUNS"
      if [ "$task" = energy ]; then
        r=1
        launch_gated "${blk}_${a}_${i}" --runtime "$(arm_runtime "$a")" --model-id "$(arm_model "$a")" \
          --task "$tid" --context-tokens "$CTX" --runs "$r"
      else
        launch "${blk}_${a}_${i}" --runtime "$(arm_runtime "$a")" --model-id "$(arm_model "$a")" \
          --task "$tid" --context-tokens "$CTX" --runs "$r"
      fi
      cool
    done
    [ "$task" = energy ] && [ "$i" -ge 2 ] && { echo "    (energy: stopping at n=$i per arm)"; break; }
  done
  echo "BLOCK $blk DONE"
}

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
  # The stamp default is read out of the app source, not hardcoded: a hardcoded r2 default
  # outlived the r3 harness and would have silently dropped every cell of the session that
  # measured on r3. HARNESS_STAMP still overrides for cross-checking an older capture.
  local src_stamp
  src_stamp="$(sed -n 's/.*harnessStamp = "\([^"]*\)".*/\1/p' \
    "$REPO/ios/BenchmarkApp/Sources/Benchmark/BenchmarkRunner.swift" | head -1)"
  local stamp="${HARNESS_STAMP:-$src_stamp}"
  [ -n "$stamp" ] || { echo "cannot determine harness stamp (set HARNESS_STAMP)"; exit 1; }
  echo "== harness stamp: $stamp"
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
  python3 "$REPO/scripts/import_native_benchmark.py" --device iPhone18,1 \
    "$OUT"/console_NATIVE_*.txt 2>&1 | tail -1

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
  run-block)    shift; cmd_run_block "$@" ;;
  stage-cactus) cmd_stage_cactus ;;
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
