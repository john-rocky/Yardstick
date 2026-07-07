#!/usr/bin/env bash
#
# bench_e4b_iphone.sh — reproducible on-device Gemma-4-E4B 3-way (Core AI / MLX / LiteRT),
# all QAT-int4, one app (BenchmarkApp), matched conditions (short-chat, greedy, 3 cold launches).
# The whole point: anyone with the app + the public models can re-run this and get the table —
# no cited numbers from other harnesses.
#
# Prereq: BenchmarkApp installed on the device (built in Xcode; signing needs the GUI once).
#   The app must include the Core AI E4B PLE path (CoreAIRuntime.staticPLEBuffers) + the
#   `core-ai/gemma4-e4b-gpu` and `mlx-community/gemma-4-e4b-it-qat-OptiQ-4bit` catalog entries.
#
# Usage:
#   bench_e4b_iphone.sh stage    # side-load the Core AI E4B bundle + PLE tables (USB, ~5 GB)
#   bench_e4b_iphone.sh speed    # run the 3-way, write results/raw/<date>-gemma4-e4b-iphone/speed.jsonl
#   bench_e4b_iphone.sh all      # stage then speed
set -uo pipefail
DEV="${DEV:-A6F3E849-1947-5202-9AD1-9C881CA58EEF}"          # iPhone 17 Pro
APP=com.iosllmbenchmark.benchmarkapp
ROOT=~/Downloads/ios-llm-benchmark
OUT="$ROOT/results/raw/2026-07-06-gemma4-e4b-iphone"; mkdir -p "$OUT"
# Core AI E4B: iPhone-AOT (h18p) decode bundle + the PLE gather tables (mmap'd as ple_table/ple_scale).
H18P="${H18P:-$ROOT/../e4b_coreai_h18p/gpu-pipelined/gemma4_e4b_qat_decode_int4lin_aotc_h18p}"
PLE_DIR="${PLE_DIR:-$HOME/code/coreai/ondevice/artifacts/gemma4_e4b_qat_gather_raw}"

# runtime  model-id                                          (all QAT-int4)
ROWS=(
  "core-ai   core-ai/gemma4-e4b-gpu"
  "mlx       mlx-community/gemma-4-e4b-it-qat-OptiQ-4bit"
  "litert-lm litert-community/gemma-4-E4B-it-litert-lm"
)

dev_ok() { xcrun devicectl list devices 2>/dev/null | grep -qiE "$DEV.*(available|connected)" \
  || { echo "⚠ device $DEV not available"; return 1; }; }
copy() { xcrun devicectl device copy to --device "$DEV" --domain-type appDataContainer \
  --domain-identifier "$APP" --source "$1" --destination "$2" 2>&1 | tail -1; }

cmd_stage() {
  dev_ok || return 1
  [ -d "$H18P/metadata.json" -o -f "$H18P/metadata.json" ] || { echo "✗ AOT bundle missing: $H18P"; return 1; }
  echo "stage Core AI E4B decode (h18p) → Documents/CoreAIModels/gemma4_e4b_gpu"
  copy "$H18P" "Documents/CoreAIModels/gemma4_e4b_gpu"
  echo "stage PLE tables → gemma4_e4b_gpu/ple/  (embed_per_layer.i8 2.6 GB + scale)"
  copy "$PLE_DIR/embed_per_layer.i8"       "Documents/CoreAIModels/gemma4_e4b_gpu/ple/embed_per_layer.i8"
  copy "$PLE_DIR/embed_per_layer.scale.f32" "Documents/CoreAIModels/gemma4_e4b_gpu/ple/embed_per_layer.scale.f32"
  echo "staged. (MLX + LiteRT download on-device on first run — keep WiFi on.)"
}

cmd_speed() {
  dev_ok || return 1
  local jsonl="$OUT/speed.jsonl"; : > "$jsonl"
  echo "writing → $jsonl"
  for row in "${ROWS[@]}"; do
    set -- $row; local rt="$1" mid="$2"
    echo "##### $mid ($rt) #####"
    for run in 1 2 3; do
      line=$(timeout 600 xcrun devicectl device process launch --console --terminate-existing \
        --device "$DEV" "$APP" -- --yardstick-autorun --runtime "$rt" --model-id "$mid" \
        --task short-chat --runs 1 </dev/null 2>&1 \
        | grep -oE "decode_tok_s=[0-9.]+ ttft_ms=[0-9]+ peak_mb=[0-9]+|YARDSTICK_[A-Z_]+[^\"]{0,60}|Cannot allocate memory")
      echo "  run$run :: ${line:-<no verdict>}"
      d=$(echo "$line" | grep -oE "decode_tok_s=[0-9.]+" | cut -d= -f2)
      t=$(echo "$line" | grep -oE "ttft_ms=[0-9]+" | cut -d= -f2)
      p=$(echo "$line" | grep -oE "peak_mb=[0-9]+" | cut -d= -f2)
      [ -n "$d" ] && echo "{\"family\":\"Gemma4-E4B\",\"runtime\":\"$rt\",\"model_id\":\"$mid\",\"quant\":\"int4-qat\",\"run\":$run,\"decode_tps\":$d,\"ttft_ms\":${t:-null},\"peak_mb\":${p:-null}}" >> "$jsonl"
      sleep 5
    done
  done
  echo "DONE → $jsonl"; echo "rows: $(wc -l < "$jsonl")"
}

case "${1:-}" in
  stage) cmd_stage ;;
  speed) cmd_speed ;;
  all)   cmd_stage && cmd_speed ;;
  *) echo "usage: $0 {stage|speed|all}"; exit 2 ;;
esac
