#!/usr/bin/env bash
# Gemma-4-E2B on Mac, under the same protocol as the iPhone capture.
# Spec: methodology/agreed-protocol-gemma4.md. Same tasks, same forced context, same
# positional rule — so an iPhone cell and a Mac cell differ by the device and nothing else.
#
#   scripts/bench_gemma4_e2b_protocol_mac.sh build      # build yardstick
#   scripts/bench_gemma4_e2b_protocol_mac.sh chat       # short-chat column
#   scripts/bench_gemma4_e2b_protocol_mac.sh depth      # deep-context column
#   scripts/bench_gemma4_e2b_protocol_mac.sh native     # LiteRT vendor benchmark() row
#   scripts/bench_gemma4_e2b_protocol_mac.sh energy     # sustained-generation column
#   scripts/bench_gemma4_e2b_protocol_mac.sh coreai     # Core AI via Apple's llm-benchmark
#   scripts/bench_gemma4_e2b_protocol_mac.sh all        # chat + depth + native, in order
#   scripts/bench_gemma4_e2b_protocol_mac.sh analyze
#
# WHY THIS EXISTS AS A SEPARATE SCRIPT, AND WHY IT IS THE CHEAP HALF OF THE CAMPAIGN
#
# The iPhone capture is rate-limited by heat: the device stops recovering inside a 300 s gap
# after roughly two hours, which is what forces the campaign into ~90-minute blocks spread over
# several sittings. A Mac has none of that. It stays on mains power, it has fans, there is no
# staging step, no unplug discipline, and no thermal budget to spend. So the Mac half can run
# *while the phone is cooling* and costs essentially nothing in wall-clock terms.
#
# That asymmetry was missed on 2026-07-27: the Mac table was never re-measured because the
# session was scoped to an iPhone question, even though re-measuring it would have been free.
#
# CORE AI ON MAC RUNS THROUGH A DIFFERENT HARNESS — `coreai` below
#
# The macOS `yardstick` target does not compile CoreAIRuntime, which is easy to mistake for
# "Core AI cannot be measured on Mac". It can: the published Mac row (82.4 prefill / 75.9 decode
# / GSM8K 88.0, medians of 3, 2026-07-17) came from Apple's own `llm-benchmark` CLI, not from
# this repo's harness. Method of record: `~/code/coreai/GEMMA4_LU_BENCH_HANDOFF.md` §2 and §5.
#
# That means the Mac Core AI cell is NOT protocol-identical with the other Mac cells and cannot
# be made so: `llm-benchmark` has no `--context-tokens`, sizes its own KV, and reports its own
# timing. Its prefill in particular is not a speed result — Gemma-4's per-layer embeddings force
# S=1 on both Core AI routes, so prefill/decode is 1.09x where a batched arm is ~50x. Quote it
# as an architectural fact, never as a slow-vs-fast comparison.
#
# GPU CONTENTION IS A REAL CONFOUND HERE, unlike on the phone. Everything shares unified
# memory, so a browser doing WebGL, a build, or a second benchmark will move these numbers.
# The guard below refuses to start if a known heavy pipeline is running; it cannot see
# everything, so close other GPU work by hand.
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
# The XCODEBUILD yardstick, not `swift build`. The SwiftPM binary defines YARDSTICK_SPM,
# which compiles the llama-cpp case OUT (no llama.xcframework in the SPM package), so every
# llamacpp cell dies with "unknown runtime" — hit 2026-07-28 one launch into the campaign.
# The xcodebuild target also carries project.yml's pinned mlx-swift-lm revision (60bd0d78…),
# the one the iPhone cells were measured on; the SPM manifest floated `branch: main` until
# 2026-07-28 and resolved a loader that cannot read either MLX checkpoint. One harness, all
# four arms, same loader as the phone — that is the point of the campaign.
DD_MAC="${DD_MAC:-$HOME/bench-dd-mac}"
YS="${YS:-$DD_MAC/Build/Products/Release/yardstick}"
CAMPAIGN="${CAMPAIGN:-$(date +%F)-gemma4-e2b-protocol-mac}"
OUT="$REPO/results/raw/$CAMPAIGN"

# Protocol constants — identical to the iPhone driver on purpose. Changing one here and not
# there is how a "cross-device comparison" stops being one.
CTX=2048
PREFILL=1024
DECODE=256
RUNS="${RUNS:-3}"            # run 1 cold, runs 2-3 the warm pair (positional rule)
LAUNCHES="${LAUNCHES:-4}"    # 4 x 2 = n=8 per cell
COOLDOWN="${COOLDOWN:-30}"   # no thermal budget to protect; this is just GPU settle time

LITERT_ID="litert-community/gemma-4-E2B-it-litert-lm"
MLX_ID="mlx-community/gemma-4-e2b-it-4bit"
OPTIQ_ID="mlx-community/gemma-4-e2b-it-qat-OptiQ-4bit"
GGUF_ID="unsloth/gemma-4-E2B-it-GGUF/Q4_K_M"

# The published Mac table carries MLX, LiteRT-LM and Core AI. OptiQ and llama.cpp are added
# here because they cost nothing on this device and because the MLX pair (PTQ vs OptiQ) is read
# against itself in every table — capturing one without the other is what left the iPhone table
# with two MLX rows nine days apart.
ARMS="${ARMS:-litert mlx optiq llamacpp}"

arm_runtime() { case "$1" in
  litert) echo litert-lm ;; mlx|optiq) echo mlx-swift ;;
  # The Mac CLI's runtime name is `llama-cpp`, not the iOS raw value `llama.cpp` —
  # passing the latter fails every llamacpp cell with "unknown runtime" (hit 2026-07-28).
  llamacpp) echo llama-cpp ;;
  *) echo "unknown arm: $1" >&2; return 1 ;;
esac; }
arm_model() { case "$1" in
  litert) echo "$LITERT_ID" ;; mlx) echo "$MLX_ID" ;; optiq) echo "$OPTIQ_ID" ;;
  llamacpp) echo "$GGUF_ID" ;;
  *) echo "unknown arm: $1" >&2; return 1 ;;
esac; }

guard_gpu_only() {
  # Pattern must not match the repo name itself ("ios-llm-benchmark") in a task path.
  if ps aux | grep -E "coreai\.llm\.export|release/llm-benchmark |export_simple_template\.py" \
      | grep -v grep >/dev/null; then
    echo "refusing to start: a heavy CPU/GPU pipeline is still running (unified-memory contention)" >&2
    exit 1
  fi
}

guard() {
  [ -x "$YS" ] || { echo "build first:  $0 build" >&2; exit 1; }
  guard_gpu_only
  # An XNNPACK cache beside the model silently costs 13% of GPU decode and doubles the tail
  # latency (measured 2026-07-27: 155->135 tok/s, ITL p95 6.6->12.3 ms). One CPU-backend run
  # leaves one behind, so this is checked every time rather than trusted.
  local stale
  stale=$(ls -d "$HOME"/.cache/huggingface/hub/models--litert-community--*/snapshots/*/*xnnpack_cache* 2>/dev/null | head -3)
  if [ -n "$stale" ]; then
    echo "REFUSING: XNNPACK cache present beside a litert model — it degrades every later GPU run." >&2
    echo "$stale" | sed 's/^/  /' >&2
    echo "delete those files, then re-run." >&2
    exit 1
  fi
  mkdir -p "$OUT"
}

cell() {  # cell <label> <runtime> <model> <task-args...>
  local label="$1" rt="$2" model="$3"; shift 3
  echo "--- $label"
  "$YS" run --runtime "$rt" --model-id "$model" --runs "$RUNS" \
      --context-tokens "$CTX" --output "$OUT/${label}.jsonl" "$@" 2>&1 \
    | tail -6
  sleep "$COOLDOWN"
}

cmd_build() {
  echo "building yardstick (xcodebuild Release; scheme from project.yml)..."
  ( cd "$REPO/ios/BenchmarkApp" && xcodegen generate >/dev/null )
  # ARCHS=arm64: Release otherwise also builds the x86_64 slice of every package, and the
  # local CoreML-LLM package uses Float16, which does not exist on x86_64 macOS — the whole
  # build fails on a slice no benchmark will ever run (hit 2026-07-28).
  xcodebuild -project "$REPO/ios/BenchmarkApp/BenchmarkApp.xcodeproj" -scheme yardstick \
    -configuration Release -destination "platform=macOS,arch=arm64" -derivedDataPath "$DD_MAC" \
    -skipPackagePluginValidation -skipMacroValidation ARCHS=arm64 ONLY_ACTIVE_ARCH=YES \
    build 2>&1 | grep -E "\.swift:[0-9]+:[0-9]+: error|BUILD (SUCCEEDED|FAILED)" | sort -u | head -20
  [ -x "$YS" ] && echo "yardstick: $YS" || { echo "build produced no binary at $YS" >&2; exit 1; }
}

run_column() {  # run_column <task-id> <label-prefix>
  guard
  local i a
  echo "=== $2 : arms=[$ARMS] launches=$LAUNCHES runs=$RUNS ctx=$CTX"
  for i in $(seq 1 "$LAUNCHES"); do
    for a in $ARMS; do
      cell "${2}_${a}_${i}" "$(arm_runtime "$a")" "$(arm_model "$a")" --task "$1"
    done
  done
  echo "$2 DONE"
}

cmd_chat()  { run_column short-chat CHAT ; }
cmd_depth() { run_column long-context-1024-gen256 DEPTH ; }

# Energy is one 600 s run per launch, not RUNS — and it is the expensive column here too, so
# it is kept out of `all`. n=2 per arm is the historical depth of this row.
cmd_energy() {
  guard
  local i a
  for i in 1 2; do
    for a in $ARMS; do
      echo "--- ENERGY_${a}_${i}"
      "$YS" run --runtime "$(arm_runtime "$a")" --model-id "$(arm_model "$a")" \
        --runs 1 --context-tokens "$CTX" --task energy \
        --output "$OUT/ENERGY_${a}_${i}.jsonl" 2>&1 | tail -4
      sleep "$COOLDOWN"
    done
  done
  echo "ENERGY DONE"
}

# LiteRT-LM only: no other runtime has this entry point, and it is the number the HF model card
# quotes. --context-tokens is required even here — without it the vendor helper hardcodes
# maxNumTokens to max(prefill,decode)+32 = 1056, not the agreed 2048.
cmd_native() {
  guard
  local i
  for i in $(seq 1 7); do
    echo "--- NATIVE_litert_$i"
    "$YS" run --runtime litert-lm --model-id "$LITERT_ID" --runs 1 \
      --context-tokens "$CTX" --litert-native-benchmark "${PREFILL}x${DECODE}" \
      --output "$OUT/NATIVE_litert_${i}.jsonl" 2>&1 | tail -4
    sleep "$COOLDOWN"
  done
  echo "NATIVE DONE"
}

# Core AI, via Apple's CLI. Separate command because it is a separate harness with separate
# caveats — folding it into `all` would imply a comparability it does not have.
#
# THE ROUTE THAT PRODUCED THE PUBLISHED 82.4 / 75.9, re-established 2026-07-28:
#   * bundle = fleet_exports/gemma4_e2b_qat_decode_int4lin_tbl (NOT the device_b2 h18p AOT
#     bundle — that is compiled for h18p and this Mac is h16c);
#   * --raw-dir = the PLE table dump (embed_per_layer.i8 + .scale.f32). A `tbl` bundle
#     gathers the per-layer-embedding table in-graph and the table rides as a STATIC input
#     the harness must bind; without --raw-dir the load dies asking a garbage per-token
#     'ple_table' allocation (the "9.6 TB" failure) — that error means a missing flag, not
#     a broken bundle;
#   * COREAI_CHUNK_THRESHOLD=1, matching the iOS harness, so prefill steps S=1 (Gemma-4's
#     PLE forces S=1; the ~1.09x prefill/decode ratio is that architecture fact).
#   Verified loading + running on this Mac (h16c): p=64/g=32 probe -> 101.9 / 90.0 tok/s.
COREAI_BIN="${COREAI_BIN:-$HOME/code/coreai-models-020-bench/.build/out/Products/Release/llm-benchmark}"
COREAI_BUNDLE="${COREAI_BUNDLE:-$HOME/code/coreai/leaderboard/models/fleet_exports/gemma4_e2b_qat_decode_int4lin_tbl}"
COREAI_RAWDIR="${COREAI_RAWDIR:-$HOME/code/coreai/ondevice/artifacts/gemma4_qat_ple_raw}"
cmd_coreai() {
  [ -x "$COREAI_BIN" ] || { echo "no llm-benchmark at $COREAI_BIN" >&2
    echo "  build it in ~/code/coreai-models-020-bench (note: .build/out/Products/Release, not .build/release)" >&2; exit 1; }
  [ -d "$COREAI_BUNDLE" ] || { echo "no model bundle at $COREAI_BUNDLE" >&2; exit 1; }
  [ -f "$COREAI_RAWDIR/embed_per_layer.i8" ] || { echo "no PLE dump at $COREAI_RAWDIR" >&2; exit 1; }
  mkdir -p "$OUT"
  # The GPU must be idle. The published row was under-reported by 8-12% once (73.5/70.2 against
  # a true 82.4/75.9) for exactly this reason, so the guard is not decorative.
  guard_gpu_only
  echo "=== Core AI (Apple llm-benchmark) p=$PREFILL g=$DECODE trials=5"
  COREAI_CHUNK_THRESHOLD=1 "$COREAI_BIN" --model "$COREAI_BUNDLE" --raw-dir "$COREAI_RAWDIR" \
    -p "$PREFILL" -g "$DECODE" -n 5 \
    --output-json "$OUT/COREAI_llm-benchmark.json" 2>&1 | tail -12
  echo
  echo "NOTE: this row is not protocol-identical with the yardstick rows above."
  echo "      llm-benchmark has no --context-tokens and reports its own timing."
  echo "      GSM8K for this arm comes from ~/code/hf-to-litertlm/scripts/parity_gsm8k.py."
}

cmd_all() { cmd_chat && cmd_depth && cmd_native; }

cmd_analyze() {
  python3 "$REPO/scripts/analyze_comparability.py" "$OUT"
  echo
  echo "raw: $OUT"
}

case "${1:-}" in
  build)   cmd_build ;;
  chat)    cmd_chat ;;
  depth)   cmd_depth ;;
  energy)  cmd_energy ;;
  native)  cmd_native ;;
  coreai)  cmd_coreai ;;
  all)     cmd_all ;;
  analyze) cmd_analyze ;;
  *) sed -n '2,30p' "$0"; exit 2 ;;
esac
