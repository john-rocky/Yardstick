#!/bin/bash
# Muse-Glimmer-30B text decoder — Core AI vs ExecuTorch (Metal) vs raw MLX, one Mac.
#
# Interleaved CA/ET/MLX per prompt with a cooldown between every run (fairness rule 11):
# a 30B heats the GPU in seconds and absolute tok/s swings ~30% with thermal state, so
# block ordering measures the order, not the engines. Raw MLX is the arm that makes the
# ExecuTorch number interpretable: their `metal` backend is MLX-native (their README), so
# beating ExecuTorch could mean beating MLX or beating the wrapper — only raw MLX separates
# the two.
#
# Published capture: results/raw/2026-08-15-muse-glimmer-30b-3way/ (pins in its ENV.md and
# in environment.lock.json). Every path below is env-overridable; the OUT default is
# date-stamped with a -rerun suffix so a re-run never clobbers the published capture.
#
# Prereqs (external checkouts, not in this repo — see ./reproduce mac muse-glimmer-30b-3way):
#   - ExecuTorch solo_runner built from pytorch/executorch @ abc5586
#     + patches/executorch-abc5586-empty-data-path.diff
#   - Core AI llm-runner built from the fork state in
#     patches/coreai-models-58aab35-muse-bench-session.diff (NOT the pinned 0.2.0 arm)
#   - a python with mlx-vlm (spec: tools/python-envs/mlx-vlm-muse.requirements.txt)
set -u

ET_ROOT=${ET_ROOT:-$HOME/code/et-pr/executorch}
ET_SOLO=${ET_SOLO:-$ET_ROOT/cmake-out/examples/models/muse-glimmer/solo_runner}
ET_VARIANT=${ET_VARIANT:-muse-glimmer-k-quant-17G-128K-text-solo-metal}
ET_PTE=${ET_PTE:-$(ls -d "$HOME"/.cache/huggingface/hub/models--meta-models--Muse-Glimmer-30B-ExecuTorch-PTE/snapshots/* 2>/dev/null | head -1)/$ET_VARIANT/$ET_VARIANT.pte}
ET_TOK=${ET_TOK:-$(ls -d "$HOME"/.cache/huggingface/hub/models--meta-models--Muse-Glimmer-30B/snapshots/* 2>/dev/null | head -1)/tokenizer.json}

CA_RUNNER=${CA_RUNNER:-$HOME/code/coreai/coreai-models/.build/release/llm-runner}
CA_BUNDLE=${CA_BUNDLE:-$HOME/code/coreai/coreai-models/exports/muse_glimmer_30b_decode_int4hu_block32_sym}

MLX_PY=${MLX_PY:-python3}
MLX_MODEL=${MLX_MODEL:-mlx-community/Muse-Glimmer-30B-4bit}

N=${N:-192}
COOL=${COOL:-45}
OUT=${OUT:-results/raw/$(date +%Y-%m-%d)-muse-glimmer-30b-3way-rerun/threeway-interleaved-$N.log}

missing=0
for pair in "ET solo_runner:$ET_SOLO" "ET .pte:$ET_PTE" "tokenizer.json:$ET_TOK" \
            "Core AI llm-runner:$CA_RUNNER" "Core AI bundle:$CA_BUNDLE"; do
  label=${pair%%:*}; path=${pair#*:}
  if [ ! -e "$path" ]; then echo "MISSING $label: $path" >&2; missing=1; fi
done
"$MLX_PY" -c 'import mlx_vlm' 2>/dev/null || { echo "MISSING mlx-vlm in MLX_PY=$MLX_PY" >&2; missing=1; }
[ "$missing" -eq 0 ] || exit 1

P1="Write a Python function that merges two sorted lists into one sorted list, then explain its time complexity."
P2="A farmer has 17 sheep. All but 9 run away. How many are left? Think step by step."

run_ca() {
  "$CA_RUNNER" --model "$CA_BUNDLE" --prompt "$1" --max-tokens "$N" --temperature 0.0 \
               --inference-engine-variant coreai-pipelined --warmup off 2>&1 \
    | grep -E "^Generation:" | sed 's/^/    CA   /'
}

run_et() {
  "$ET_SOLO" --model_path "$ET_PTE" --tokenizer_path "$ET_TOK" \
             --prompt "$1" --max_new_tokens "$N" --temperature 0 --ignore_eos=true 2>&1 \
    | grep -E "^Decode:" | sed 's/^/    ET   /'
}

run_mlx() {
  "$MLX_PY" -m mlx_vlm.generate --model "$MLX_MODEL" --prompt "$1" \
            --max-tokens "$N" --temperature 0 --verbose 2>&1 \
    | grep -iE "generation:|tokens-per-sec|tokens per second" | sed 's/^/    MLX  /'
}

mkdir -p "$(dirname "$OUT")"
{
  echo "protocol: $N new tokens, greedy, batch 1, interleaved CA/ET/MLX, ${COOL}s cooldown between runs"
  sleep "$COOL"
  for i in 1 2; do
    eval "P=\$P$i"
    echo "=== prompt $i"
    for round in 1 2; do
      echo "  round $round"
      run_ca  "$P"; sleep "$COOL"
      run_et  "$P"; sleep "$COOL"
      run_mlx "$P"; sleep "$COOL"
    done
  done
} | tee "$OUT"
echo "wrote $OUT" >&2
