#!/usr/bin/env bash
DEV=A6F3E849-1947-5202-9AD1-9C881CA58EEF; APP=com.iosllmbenchmark.benchmarkapp
STG="/private/tmp/claude-501/-Users-majimadaisuke-Downloads-ios-llm-benchmark/8778f82f-8fdb-4a64-97ed-838c42f8a72c/scratchpad/mlx_llama3b"
DEST="Documents/models/mlx/mlx-community__Llama-3.2-3B-Instruct-4bit"
RES=~/Downloads/ios-llm-benchmark/results/raw/2026-06-24-coreai-iphone/extemb-isolation
echo "=== side-load MLX Llama-3.2-3B-4bit ($(du -sh $STG | awk '{print $1}')) ==="
xcrun devicectl device copy to --device $DEV --domain-type appDataContainer --domain-identifier $APP \
  --source "$STG" --destination "$DEST" 2>&1 | tail -1
echo "on-device: $(xcrun devicectl device info files --device $DEV --domain-type appDataContainer --domain-identifier $APP --subdirectory "$DEST" 2>/dev/null | grep -i safetensors | head -1)"
for run in 1 2 3; do
  f="$RES/console_mlx_llama3b_${run}.txt"
  timeout 360 xcrun devicectl device process launch --console --terminate-existing --device $DEV $APP -- \
    --yardstick-autorun --runtime mlx --model-id "mlx-community/Llama-3.2-3B-Instruct-4bit" --task short-chat --runs 1 </dev/null > "$f" 2>&1
  v=$(grep -hoE "YARDSTICK_RUN_OK[^\"]*decode_tok_s=[0-9.]*|YARDSTICK_RUN_FAIL run=1 error=[^\"]{0,70}|Cannot allocate memory" "$f" | head -1)
  echo "  MLX-Llama3B run$run :: ${v:-<no verdict>}"
  sleep 6
done
echo "MLX_LLAMA3B_DONE"
