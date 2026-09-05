#!/bin/zsh
cd /private/tmp/odp-bench/results/litertlm
M=/data/local/tmp/llmbench/models/qwen25_15b_q8_ekv4096.litertlm
for be in cpu gpu; do for it in 1 2 3; do
  sleep 45
  adb -s RFGL80R6A6H shell "cd /data/local/tmp/llmbench && LD_LIBRARY_PATH=/data/local/tmp/llmbench ./litert_lm_main --model_path=$M --backend=$be --input_prompt_file=/data/local/tmp/llmbench/p512.txt --max_output_tokens=256 2>&1" > s26_${be}_p512_${it}.txt
  echo "== $be #$it: $(grep -E 'Prefill Turn 1|Decode Turn 1|Prefill Speed|Decode Speed' s26_${be}_p512_${it}.txt | tr -s ' ' | tr '\n' ' ')"
done; done
adb -s RFGL80R6A6H shell "dumpsys thermalservice 2>/dev/null | grep -iE 'status|mValue' | head -4"
echo "S26 BENCH DONE $(date +%H:%M:%S)"
