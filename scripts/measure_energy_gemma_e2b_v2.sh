#!/usr/bin/env bash
# Mac (M4 Max) J/token for the 2026-07-18 Gemma-4-E2B best-available lineup — the energy
# companion to the iPhone table (same arms, same builds, labelled the same way).
#
# Run this IN YOUR TERMINAL (powermetrics needs sudo; the script validates it up front and
# must not itself run under sudo). Idle desktop, no other heavy apps — powermetrics measures
# the whole system.
#
#   YARDSTICK_BIN=/tmp/yardstick-dd2/Build/Products/Release/yardstick \
#     bash scripts/measure_energy_gemma_e2b_v2.sh
#
# Window convention (same as every existing energy row): joules integrate over the whole
# bench invocation (load + prefill + decode) and divide by generated tokens; the sustained /
# long-decode configs below keep decode dominant so load amortizes.
set -e

PY=python3
SCRIPT=$(dirname "$0")/measure_energy.py
COREAI_BIN="$HOME/code/coreai-models-020-bench/.build/out/Products/Release/llm-benchmark"
COREAI_BUNDLE="$HOME/code/coreai/leaderboard/models/fleet_exports/gemma4_e2b_qat_decode_int4lin_tbl"
COREAI_PLE="$HOME/code/coreai/ondevice/artifacts/gemma4_ple_raw"

echo "=== 1/5: LiteRT-LM (wNa8o8 QAT, sustained) ==="
$PY "$SCRIPT" run --task sustained --runtime litert-lm \
    --model litert-community/gemma-4-E2B-it-litert-lm --device m4max

echo "=== 2/5: MLX-Swift (QAT OptiQ int4 — quality-best, sustained) ==="
$PY "$SCRIPT" run --task sustained --runtime mlx-swift \
    --model mlx-community/gemma-4-e2b-it-qat-OptiQ-4bit --device m4max

echo "=== 3/5: MLX-Swift (PTQ 4-bit — speed-best, sustained; refresh of the old row) ==="
$PY "$SCRIPT" run --task sustained --runtime mlx-swift \
    --model mlx-community/gemma-4-e2b-it-4bit --device m4max

echo "=== 4/5: llama.cpp (Q4_K_M — best that loads; official QAT gguf is unloadable) ==="
$PY "$SCRIPT" run --task sustained --runtime llama.cpp \
    --model unsloth/gemma-4-E2B-it-GGUF/Q4_K_M --device m4max

echo "=== 5/5: Core AI (own int4, PATCHED ENGINE reference — via llm-benchmark) ==="
# -g 512 x -n 3 = 1536 generated tokens; S=1 prefill kept short (-p 32) so decode dominates.
$PY "$SCRIPT" run --task sustained --runtime core-ai \
    --model core-ai/gemma4-e2b-gpu --device m4max \
    --cmd "COREAI_CHUNK_THRESHOLD=1 '$COREAI_BIN' --model '$COREAI_BUNDLE' --raw-dir '$COREAI_PLE' -p 32 -g 512 -n 3" \
    --tokens 1536 \
    --quant "int4 q4_0 (QAT, own export; patched engine — see methodology/core-ai-arm-provenance.md)"

echo "=== Done. Re-rendering RESULTS.md ==="
$PY "$(dirname "$0")/render_results.py"
echo "Energy rows appended under results/raw/m4max-*-sustained-energy.jsonl (litert/core-ai new)."
