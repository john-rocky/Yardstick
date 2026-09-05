#!/bin/zsh
# usage: bench_llama.sh <tag> <gguf>
set -u
tag=$1; gguf=$2; out=/private/tmp/odp-bench/results; mkdir -p $out
echo "== $tag: $gguf" ; ls -l "$gguf"
llama-bench -m "$gguf" -p 512 -n 256 -ngl 99 -fa 1 -r 3 -o json > $out/$tag.llama-bench.json 2> $out/$tag.llama-bench.stderr
llama-bench -m "$gguf" -p 512 -n 256 -ngl 99 -fa 1 -r 3 2>/dev/null | tee $out/$tag.llama-bench.txt
{ /usr/bin/time -l llama-bench -m "$gguf" -p 512 -n 256 -ngl 99 -fa 1 -r 3 ; } > $out/$tag.llama-bench-rss.txt 2>&1
grep -E "maximum resident|tg256|pp512" $out/$tag.llama-bench-rss.txt
llama-completion -m "$gguf" -ngl 99 -fa on --temp 0 -n 160 -st --simple-io -p "In two or three sentences, what is unified memory on Apple Silicon and why does it matter for running LLMs locally?" > $out/$tag.completion.txt 2> $out/$tag.completion.stderr
grep -E "eval time|prompt eval time" $out/$tag.completion.stderr | tail -2
echo "--- completion head:"; head -c 400 $out/$tag.completion.txt; echo
