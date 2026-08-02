#!/usr/bin/env bash
set -e
EX=~/code/coreai/coreai-models/exports; ARCH=h18p
DEV=A6F3E849-1947-5202-9AD1-9C881CA58EEF; APP=com.iosllmbenchmark.benchmarkapp
RES=~/Downloads/ios-llm-benchmark/results/raw/2026-06-24-coreai-iphone
IRDIR=$EX/vibethinker_1_5b_ios_pure4bit
IR=$IRDIR/vibethinker_1_5b_ios_pure4bit.aimodel
OUT=$EX/vibethinker_1_5b_ane_pure4bit
TMP=$(mktemp -d)
echo "=== compile ANE bundle (neural-engine, $ARCH) ==="
xcrun coreai-build compile "$IR" --platform iOS --preferred-compute neural-engine --architecture $ARCH --output "$TMP" 2>&1 | tail -3
rm -rf "$OUT"; mkdir -p "$OUT"
cp -R "$TMP/vibethinker_1_5b_ios_pure4bit.$ARCH.aimodelc" "$OUT/"
cp -R "$IRDIR/tokenizer" "$OUT/"
python3 -c "import json; m=json.load(open('$IRDIR/metadata.json')); m.setdefault('assets',{})['main']='vibethinker_1_5b_ios_pure4bit.$ARCH.aimodelc'; json.dump(m,open('$OUT/metadata.json','w'),indent=2)"
echo "assembled: $(ls $OUT)"
echo "=== side-load → Documents/CoreAIModels/vibethinker_1_5b_ane ==="
xcrun devicectl device copy to --device $DEV --domain-type appDataContainer --domain-identifier $APP \
  --source "$OUT" --destination "Documents/CoreAIModels/vibethinker_1_5b_ane" 2>&1 | tail -1
echo "=== run core-ai/vibethinker-1.5b-ane 3x ==="
for run in 1 2 3; do
  v=$(timeout 400 xcrun devicectl device process launch --console --terminate-existing --device $DEV $APP -- \
    --yardstick-autorun --runtime core-ai --model-id "core-ai/vibethinker-1.5b-ane" --task short-chat --runs 1 </dev/null 2>&1 \
    | grep -hoE "YARDSTICK_RUN_OK[^\"]*decode_tok_s=[0-9.]*|YARDSTICK_RUN_FAIL[^\"]{0,60}|not found" | head -1)
  echo "  VibeThinker-ANE run$run :: ${v:-<no verdict>}"; sleep 6
done
echo "VIBETHINKER_ANE_DONE"
