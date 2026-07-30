#!/usr/bin/env bash
# Cactus-parity bench on iPhone 17 Pro — everything that still needs the device.
#
#   scripts/bench_cactus_parity_iphone.sh check     # device reachable + app installed?
#   scripts/bench_cactus_parity_iphone.sh install   # (re)build + install BenchmarkApp
#   scripts/bench_cactus_parity_iphone.sh stage     # side-load the LiteRT bundle over USB
#   scripts/bench_cactus_parity_iphone.sh run       # the measurement runs
#   scripts/bench_cactus_parity_iphone.sh pull      # copy result JSONL off the device
#   scripts/bench_cactus_parity_iphone.sh cleanup   # remove the borrowed CactusTest app
#   scripts/bench_cactus_parity_iphone.sh all       # install → stage → run → pull
#
# Protocol mirrors `cactus benchmark` (cactus-engine/tests/test_benchmark.cpp):
# a ~1K-token prompt, 100 decode tokens, greedy, one warmup run then three warm runs.
# Run 1 is cold (model load) and is dropped by scripts/cactus_parity_report.py.
#
# Signing: this Mac has no Xcode account, so a fresh App ID cannot be registered.
# Both apps are built under existing MFN25KNUGJ App IDs whose cached profiles already
# carry the memory entitlements. See methodology/next-session-brief.md.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEV="${DEV:-A6F3E849-1947-5202-9AD1-9C881CA58EEF}"   # devicectl id, iPhone 17 Pro (iPhone18,1)
ECID="${ECID:-00008150-0018713A0207801C}"            # xcodebuild -destination id
APP="${APP:-com.example.CoreMLLLMChat}"              # borrowed App ID (both memory caps)
CACTUS_APP="${CACTUS_APP:-com.coreai.gemmaplebench}" # borrowed App ID for CactusTest
DD="${DD:-/tmp/bench-dd}"
OUT="${OUT:-$REPO/results/raw/2026-07-10-cactus-parity}"
TASK="cactus-parity"
RUNS=4

MLX_MODEL="mlx-community/gemma-4-e2b-it-4bit"
LITERT_MODEL="litert-community/gemma-4-E2B-it-litert-lm"
LLAMA_MODEL="unsloth/gemma-4-E2B-it-GGUF/Q4_K_M"
COREML_MODEL="coreml-llm/gemma4-e2b"

launch() {  # launch <runtime> <model-id> [extra args…]
    local rt="$1" model="$2"; shift 2
    echo "── $rt / $model"
    xcrun devicectl device process launch --console --terminate-existing --device "$DEV" "$APP" \
        -- --yardstick-autorun --runtime "$rt" --model-id "$model" "$@" 2>&1 \
        | grep -E "YARDSTICK_(BEGIN|RUN_OK|NATIVE_OK|RUN_FAIL|FATAL|ALL_DONE)" || true
}

cmd_check() {
    xcrun devicectl list devices 2>/dev/null | grep -E "connected|available" | head -3
    echo "--- installed apps ---"
    xcrun devicectl device info apps --device "$DEV" 2>/dev/null | grep -E "$APP|$CACTUS_APP" \
        || echo "(neither app installed — run: $0 install)"
}

cmd_install() {
    (cd "$REPO/ios/BenchmarkApp" && xcodegen generate >/dev/null)
    xcodebuild -project "$REPO/ios/BenchmarkApp/BenchmarkApp.xcodeproj" -scheme BenchmarkApp \
        -configuration Release -destination "platform=iOS,id=$ECID" -derivedDataPath "$DD" \
        PRODUCT_BUNDLE_IDENTIFIER="$APP" DEVELOPMENT_TEAM=MFN25KNUGJ CODE_SIGN_STYLE=Automatic \
        build 2>&1 | grep -E "error:|BUILD (SUCCEEDED|FAILED)"
    xcrun devicectl device install app --device "$DEV" "$DD/Build/Products/Release-iphoneos/BenchmarkApp.app" \
        2>&1 | grep -E "bundleID|ERROR"
}

cmd_stage() {
    # LiteRT is the only large `download` row we side-load; on-device HF pulls stall past ~32 MB.
    local snap stage
    snap="$(ls -d "$HOME"/.cache/huggingface/hub/models--litert-community--gemma-4-E2B-it-litert-lm/snapshots/*/ | head -1)"
    [ -f "$snap/gemma-4-E2B-it.litertlm" ] || { echo "run first: hf download $LITERT_MODEL"; exit 1; }
    stage=$(mktemp -d)/litert-community__gemma-4-E2B-it-litert-lm
    mkdir -p "$stage"; cp -L "$snap/gemma-4-E2B-it.litertlm" "$stage/"
    # Destination must be the full per-repo directory: devicectl copies the source dir's
    # *contents*, so passing the parent flattens the file into models/litert-lm/.
    xcrun devicectl device copy to --device "$DEV" --domain-type appDataContainer \
        --domain-identifier "$APP" --source "$stage" \
        --destination "Documents/models/litert-lm/litert-community__gemma-4-E2B-it-litert-lm" >/dev/null
    echo "staged $LITERT_MODEL"
}

cmd_run() {
    mkdir -p "$OUT"
    # LiteRT's own benchmark entry point: prefill exactly 1024 tokens (8×128, on the
    # prefill-chunk boundary so it stays fixed under future chunk-selection changes),
    # decode 256 — the LiteRT team's standard decode length (Marissa, 2026-07-15).
    # It uses createConversation()'s default sampler, which also matches their setup.
    # Context is NOT settable here: LiteRTLM.benchmark hardcodes
    # maxNumTokens = max(prefill, decode) + 32 = 1056, where the team forces 2048 for
    # Gemma-4 internally. That bounds KV-cache size (a memory effect), not decode rate.
    # Three cold process launches (it builds its own Engine each time), so no warm/cold
    # split to drop.
    for i in 1 2 3; do
        launch litert-lm "$LITERT_MODEL" --litert-native-benchmark 1024x256
    done | tee "$OUT/litert_native_benchmark.log"

    # App-level parity runs: same prompt, same 100-token budget, every runtime.
    # Runtime ids are RuntimeKind raw values — `mlx-swift`, not `mlx`. An unknown id
    # makes specFromLaunchArgs return nil, and the app silently opens its normal UI
    # instead of benchmarking.
    {
        launch litert-lm  "$LITERT_MODEL" --task "$TASK" --runs "$RUNS"
        launch mlx-swift  "$MLX_MODEL"    --task "$TASK" --runs "$RUNS"
        launch llama.cpp  "$LLAMA_MODEL"  --task "$TASK" --runs "$RUNS"
        launch coreml-llm "$COREML_MODEL" --task "$TASK" --runs "$RUNS"
    } | tee "$OUT/parity_runs.log"
}

cmd_pull() {
    mkdir -p "$OUT/device-jsonl"
    xcrun devicectl device copy from --device "$DEV" --domain-identifier "$APP" \
        --domain-type appDataContainer --source Documents/results \
        --destination "$OUT/device-jsonl" 2>&1 | tail -1
    echo "pulled → $OUT/device-jsonl"
}

cmd_cleanup() {
    # CactusTest was installed under a borrowed App ID and carries a 3.8 GB payload.
    xcrun devicectl device uninstall app --device "$DEV" "$CACTUS_APP" 2>&1 | tail -1
    echo "removed $CACTUS_APP"
}

case "${1:-}" in
    check)   cmd_check ;;
    install) cmd_install ;;
    stage)   cmd_stage ;;
    run)     cmd_run ;;
    pull)    cmd_pull ;;
    cleanup) cmd_cleanup ;;
    all)     cmd_install; cmd_stage; cmd_run; cmd_pull ;;
    *) sed -n '2,12p' "$0"; exit 1 ;;
esac
