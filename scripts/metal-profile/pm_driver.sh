#!/bin/zsh
# Sequential decode runs with epoch markers, to be aligned with a concurrent
# `sudo powermetrics --samplers gpu_power` capture.
set -u
WORK=~/Downloads/meeting/metal-profile-work
TL=$WORK/pm_timeline.txt
V=~/code/litert-mac-verify/.build/release/litert-mac-verify
PY=~/code/litertlm-convert/.venv/bin/python
QWEN=~/.cache/huggingface/hub/models--litert-community--Qwen3-4B/snapshots/84cc5a35c9c65cd18fcd65bb1f3a7d77a4acfe6e/qwen3_4b_mixed_int4.litertlm
Q8=~/.cache/huggingface/hub/models--litert-community--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/2f8b8ee90d8f93b15305b699e8772b277d074a9a/DeepSeek-R1-Distill-Qwen-1.5B_multi-prefill-seq_q8_ekv4096.litertlm
I4=~/.cache/huggingface/hub/models--mlboydaisuke--DeepSeek-R1-Distill-Qwen-1.5B-LiteRT/snapshots/3c8f42f9415b899c43a5211d6a42e2d342f238ea/model.litertlm
PROMPT="Write a detailed essay about the history of computing, covering mainframes, personal computers, and mobile devices."

: > $TL
mark() { echo "$(date +%s) $(date '+%H:%M:%S') $1" >> $TL; }

mark BASELINE_START
sleep 8
mark LITERT_QWEN3_4B_START
$V "$QWEN" "$PROMPT" --max-tokens 900 --backend gpu > $WORK/pm_litert_q4.log 2>&1
mark LITERT_QWEN3_4B_END
sleep 8
mark MLX_QWEN3_4B_START
$PY -m mlx_lm generate --model mlx-community/Qwen3-4B-4bit --prompt "$PROMPT" --max-tokens 1200 --temp 0.0 > $WORK/pm_mlx_q4.log 2>&1
mark MLX_QWEN3_4B_END
sleep 8
mark LITERT_DS_Q8_START
$V "$Q8" "$PROMPT" --max-tokens 1200 --backend gpu > $WORK/pm_litert_dsq8.log 2>&1
mark LITERT_DS_Q8_END
sleep 8
mark LITERT_DS_INT4_START
$V "$I4" "$PROMPT" --max-tokens 1200 --backend gpu > $WORK/pm_litert_dsi4.log 2>&1
mark LITERT_DS_INT4_END
sleep 8
mark MLX_DS_START
$PY -m mlx_lm generate --model mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit --prompt "$PROMPT" --max-tokens 2500 --temp 0.0 > $WORK/pm_mlx_ds.log 2>&1
mark MLX_DS_END
mark ALL_DONE
