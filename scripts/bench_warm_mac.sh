#!/usr/bin/env bash
# Mac M4 Max warm sweep — REPO harness (`yardstick run --runs N`), fairness §2.
# One JSONL per (runtime, model) with N runs (run 1 cold, 2..N warm, coldRun-flagged).
# GPU must be idle (no other GPU work / browser automation). Results stay local;
# nothing is published without explicit user GO.
#
#   scripts/bench_warm_mac.sh [litert|mlx|all]   (default: all)
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
YS="$REPO/.build/release/yardstick"
RUNS="${RUNS:-5}"
COOLDOWN="${COOLDOWN:-30}"
OUT="$REPO/results/raw/${CAMPAIGN:-$(date +%F)-mac-warm}"
mkdir -p "$OUT"

[ -x "$YS" ] || { echo "build first: GIT_LFS_SKIP_SMUDGE=1 swift build -c release --product yardstick" >&2; exit 1; }
# NB: pattern must not match the repo name "ios-llm-benchmark" in task paths.
if ps aux | grep -E "coreai\.llm\.export|release/llm-benchmark |export_simple_template\.py" | grep -v grep >/dev/null; then
  echo "refusing to start: heavy CPU pipeline still running (unified-memory contention)" >&2; exit 1
fi

LITERT_MODELS=(
  litert-community/Qwen3-0.6B
  litert-local/qwen3-1.7b-int4
  litert-community/Qwen3-4B
  litert-community/gemma-4-E2B-it-litert-lm
  litert-community/DeepSeek-R1-Distill-Qwen-1.5B
  litert-community/Phi-4-mini-instruct
  litert-community/TinySwallow-1.5B-Instruct
  litert-community/VibeThinker-1.5B
  litert-community/Gemma3-1B-IT
  mlboydaisuke/OLMo-2-1B-Instruct-LiteRT
  mlboydaisuke/SmolLM3-3B-LiteRT
  mlboydaisuke/Llama-3.2-3B-Instruct-LiteRT
  mlboydaisuke/Ministral-3-3B-Instruct-2512-LiteRT
)
MLX_MODELS=(
  mlx-community/Qwen3-0.6B-4bit
  mlx-community/Qwen3-1.7B-4bit
  mlx-community/Qwen3-4B-4bit
  mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit
  mlx-community/gemma-3-1b-it-4bit
  mlx-community/Phi-4-mini-instruct-4bit
  mlx-community/TinySwallow-1.5B-Instruct-4bit
  mlx-community/Llama-3.2-3B-Instruct-4bit
  mlx-community/SmolLM3-3B-4bit
)

sweep(){ # <runtime> <model...>
  local rt="$1"; shift
  for m in "$@"; do
    local slug; slug="$(echo "$rt-$m" | tr '/.' '__')"
    echo "== [$rt] $m ($(date +%H:%M:%S))"
    if ! "$YS" run --task short-chat --runtime "$rt" --model "$m" --runs "$RUNS" \
         --output "$OUT/$slug.jsonl" 2>"$OUT/$slug.stderr.log"; then
      echo "   FAILED — see $OUT/$slug.stderr.log" | tee -a "$OUT/FAILURES.txt"
    fi
    sleep "$COOLDOWN"
  done
}

case "${1:-all}" in
  litert) sweep litert-lm "${LITERT_MODELS[@]}" ;;
  mlx)    sweep mlx-swift "${MLX_MODELS[@]}" ;;
  all)    sweep litert-lm "${LITERT_MODELS[@]}"; sweep mlx-swift "${MLX_MODELS[@]}" ;;
esac
echo "MAC_SWEEP_DONE -> $OUT"
