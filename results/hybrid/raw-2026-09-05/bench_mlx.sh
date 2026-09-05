#!/bin/zsh
# usage: bench_mlx.sh <tag> <model-repo-or-path>
set -u
tag=$1; model=$2; out=/private/tmp/odp-bench/results; mkdir -p $out
export HF_HOME=/private/tmp/odp-bench/hf HF_HUB_OFFLINE=1
cd /private/tmp/odp-bench
for i in 1 2 3; do
  ./venv/bin/python -m mlx_lm generate --model "$model" --prompt "$(cat prompt.txt)" --max-tokens 256 --temp 0 > $out/$tag.mlx.$i.txt 2>&1
  grep -E "^(Prompt|Generation|Peak memory)" $out/$tag.mlx.$i.txt | tr '\n' ' '; echo
done
echo "--- output head (run 3):"; sed -n '/^==========/,/^==========/p' $out/$tag.mlx.3.txt | head -c 500; echo
