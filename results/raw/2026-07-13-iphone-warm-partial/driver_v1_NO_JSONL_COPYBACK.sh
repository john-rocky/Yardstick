#!/bin/bash
# Warm re-capture campaign Tier 1 — iPhone 17 Pro
# Protocol: --runs 4 per cell (run1=cold(b), runs2-4=warm(a) median); 100s cooldown between cells.
DEV=A6F3E849-1947-5202-9AD1-9C881CA58EEF
APP=com.example.CoreMLLLMChat
CELLS=(
  "mlx-swift mlx-community/Qwen3-0.6B-4bit"
  "litert-lm litert-community/Qwen3-0.6B"
  "core-ai core-ai/qwen3-0.6b-ane"
  "core-ai core-ai/qwen3-0.6b-gpu"
  "mlx-swift mlx-community/Qwen3-1.7B-4bit"
  "litert-lm litert-local/qwen3-1.7b-int4"
  "core-ai core-ai/qwen3-1.7b-gpu"
  "core-ai core-ai/qwen3-1.7b-ane"
)
for cell in "${CELLS[@]}"; do
  set -- $cell
  echo "==== CELL $1 / $2 $(date +%H:%M:%S) ===="
  xcrun devicectl device process launch --console --terminate-existing --device "$DEV" "$APP" \
    -- --yardstick-autorun --runtime "$1" --model-id "$2" --task short-chat --runs 4 2>&1 \
    | grep -E "YARDSTICK_(BEGIN|RUN_OK|RUN_FAIL|FATAL)" || echo "CELL_FAILED $1 $2"
  echo "---- cooldown 100s ----"
  sleep 100
done
echo "TIER1_DONE"
