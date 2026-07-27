#!/usr/bin/env bash
# One-shot device staging for the FULL warm matrix. Run when the iPhone is back:
#
#   scripts/stage_device_full.sh inventory   # what already survives on device (run FIRST)
#   scripts/stage_device_full.sh install     # install the pre-built app (harness update!)
#   scripts/stage_device_full.sh coreai      # push Core AI bundles (skips ones already there)
#   scripts/stage_device_full.sh litert      # push .litertlm files
#   scripts/stage_device_full.sh mlx         # push MLX caches + pin E2B to the pre-7/6 revision
#   scripts/stage_device_full.sh all         # install + coreai + litert + mlx
#
# Then:  CAMPAIGN=$(date +%F)-iphone-full-warm CELL_TIMEOUT=1200 BASE_COOLDOWN=180 \
#          CELLS_FILE=scripts/cells_full_warm_matrix.txt scripts/bench_warm_matrix_iphone.sh run
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEV="${DEV:-A6F3E849-1947-5202-9AD1-9C881CA58EEF}"
APP="${APP:-com.example.CoreMLLLMChat}"
B="$HOME/bench-staging"
# WARNING (2026-07-27): this pin is for the JUNE mlx-swift-lm only. Once the checkout has
# Gemma-4 KV sharing, 2c3e507 fails to load ("...layers.15.self_attn.v_proj.weight not found
# in ...Linear" — an EXTRA key, not a missing one) and the 2026-07-06 re-upload
# 238767527555cb75a05732a84dff5d6ba0dd6809 is the one that loads. See the revision note in
# scripts/bench_gemma4_e2b_protocol_iphone.sh before staging E2B for MLX.
E2B_OLD_REV="2c3e507453b4f218d05fe3cc97bea5c5a654257e"

copy_to(){ xcrun devicectl device copy to --device "$DEV" --domain-type appDataContainer \
  --domain-identifier "$APP" --source "$1" --destination "$2" 2>&1 | grep -iE "File on Device|error" | tail -1; }
have(){ xcrun devicectl device info files --device "$DEV" --domain-type appDataContainer \
  --domain-identifier "$APP" --subdirectory "$1" 2>/dev/null; }

cmd_inventory(){
  echo "== CoreAIModels on device:"; have Documents/CoreAIModels | awk '{print $1}' | grep -E "^[^/]+$|^[^/]+/[^/]+$" | grep -v "^Name\|^---" | awk -F/ 'NF<=1' | sort -u
  echo "== litert-lm dirs:"; have Documents/models/litert-lm | awk '{print $1, $(NF-1)}' | grep -iE "litertlm GB|litertlm MB"
  echo "== mlx gemma e2b cache snapshots:"; have "Library/Caches/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots" | awk '{print $1}' | awk -F/ 'NF<=1' | grep -v "^Name\|^---"
}

cmd_install(){
  local app="/tmp/bench-dd/Build/Products/Release-iphoneos/BenchmarkApp.app"
  [ -d "$app" ] || { echo "pre-built app missing — build first (see methodology/warm-full-matrix-prep.md)"; exit 1; }
  xcrun devicectl device install app --device "$DEV" "$app" 2>&1 | grep -iE "bundleID|error"
}

cmd_coreai(){
  # <staging bundle> -> <device catalog folder>
  local real
  while read -r src dst; do
    [ -e "$B/coreai/$src" ] || { echo "MISSING staging: $src"; continue; }
    if have "Documents/CoreAIModels/$dst" | grep -q metadata.json; then echo "skip $dst (on device)"; continue; fi
    # devicectl rejects a symlink as --source (error 21) — resolve to the real dir.
    real="$(readlink "$B/coreai/$src" || echo "$B/coreai/$src")"
    echo "-> $dst"; copy_to "$real" "Documents/CoreAIModels/$dst"
  done <<'MAP'
qwen3_0_6b_ane_ct041 qwen3_0_6b_ane
qwen3_0_6b_gpu_ct041 qwen3_0_6b_gpu
qwen3_0_6b_ane_JUNE qwen3_0_6b_ane_june
qwen3_1_7b_ane_ct041 qwen3_1_7b_ane
qwen3_1_7b_gpu_ct041 qwen3_1_7b_gpu
qwen3_1_7b_gpu_JUNE qwen3_1_7b_gpu_june
deepseek_r1_1_5b_ane_ct041 deepseek_r1_1_5b_ane
deepseek_r1_1_5b_gpu_ct041 deepseek_r1_1_5b_gpu
tinyswallow_1_5b_ane_ct041 tinyswallow_1_5b_ane
tinyswallow_1_5b_gpu_ct041 tinyswallow_1_5b_gpu
vibethinker_1_5b_ane_ct041 vibethinker_1_5b_ane
vibethinker_1_5b_gpu_ct041 vibethinker_1_5b_gpu
olmo2_1b_ane_ct041 olmo2_1b_ane
olmo2_1b_gpu_ct041 olmo2_1b_gpu
smollm3_3b_ane_ct041 smollm3_3b_ane
smollm3_3b_gpu_ct041 smollm3_3b_gpu
llama32_3b_ane_ct041 llama32_3b_ane
llama32_3b_gpu_ct041 llama32_3b_gpu
gemma3_1b_gpu_ct041 gemma3_1b_gpu
ministral3_3b_gpu_ct041 ministral3_3b_gpu
MAP
  # NOTE: qwen3_4b_{ane,gpu} already on device (June lineage) — left as-is.
  # NOTE: qwen3_0_6b_ane_june / qwen3_1_7b_gpu_june need catalog ids before use — see
  # cells file notes; primary cells use the standard folders above.
}

cmd_litert(){
  local T
  while read -r file dst; do
    [ -e "$B/litert/$file" ] || { echo "MISSING staging: $file"; continue; }
    if have "Documents/models/litert-lm/$dst" | grep -qiE "litertlm +GB|litertlm +MB"; then echo "skip $dst"; continue; fi
    T=$(mktemp -d)/m; mkdir -p "$T"; cp -L "$B/litert/$file" "$T/"
    echo "-> $dst"; copy_to "$T" "Documents/models/litert-lm/$dst"
  done <<'MAP'
qwen3-1.7b-int8.litertlm litert-local__Qwen3-1.7B
OLMo-2-1B.litertlm litert-local__OLMo-2-1B
SmolLM3-3B.litertlm litert-local__SmolLM3-3B
Llama-3.2-3B.litertlm litert-local__Llama-3.2-3B
Ministral-3-3B.litertlm litert-local__Ministral-3-3B
gemma3-1b-it-int4.litertlm litert-community__Gemma3-1B-IT
Phi-4-mini-instruct_multi-prefill-seq_q8_ekv4096.litertlm litert-community__Phi-4-mini-instruct
DeepSeek-R1-Distill-Qwen-1.5B_multi-prefill-seq_q8_ekv4096.litertlm litert-community__DeepSeek-R1-Distill-Qwen-1.5B
TinySwallow-1.5B-Instruct.litertlm litert-community__TinySwallow-1.5B-Instruct
VibeThinker-1.5B.litertlm litert-community__VibeThinker-1.5B
MAP
}

cmd_mlx(){
  local HUB="Library/Caches/huggingface/hub"
  for r in Qwen3-4B-4bit DeepSeek-R1-Distill-Qwen-1.5B-4bit gemma-3-1b-it-4bit Phi-4-mini-instruct-4bit TinySwallow-1.5B-Instruct-4bit Llama-3.2-3B-Instruct-4bit SmolLM3-3B-4bit; do
    local d="$HOME/.cache/huggingface/hub/models--mlx-community--$r"
    [ -d "$d/blobs" ] || { echo "MISSING mac cache: $r"; continue; }
    echo "-> mlx $r"
    # blobs + refs only (June sideload pattern): the snapshots dir is a farm of
    # relative symlinks and devicectl rejects '..' in paths; the on-device HF client
    # reconstructs snapshots from refs + blobs.
    copy_to "$d/blobs" "$HUB/models--mlx-community--$r/blobs"
    copy_to "$d/refs"  "$HUB/models--mlx-community--$r/refs"
  done
  # E2B: the Mac could NOT re-download the pre-7/6 revision (CDN refused; see prep runbook).
  # The device's own June cache should still hold the old snapshot — pin refs/main to it so
  # the pinned loader resolves the compatible weights. Verify with `inventory` first.
  local T; T=$(mktemp -d); printf '%s' "$E2B_OLD_REV" > "$T/main"
  echo "-> pin device E2B refs/main = $E2B_OLD_REV (loader-compatible pre-7/6 snapshot)"
  copy_to "$T" "$HUB/models--mlx-community--gemma-4-e2b-it-4bit/refs"
}

case "${1:-}" in
  inventory) cmd_inventory ;;
  install)   cmd_install ;;
  coreai)    cmd_coreai ;;
  litert)    cmd_litert ;;
  mlx)       cmd_mlx ;;
  all)       cmd_install; cmd_coreai; cmd_litert; cmd_mlx ;;
  *) sed -n '2,14p' "$0"; exit 1 ;;
esac
