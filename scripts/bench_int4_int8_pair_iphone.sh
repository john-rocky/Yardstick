#!/bin/zsh
# Same-session int8-vs-int4 pair capture (iPhone 17 Pro) for the all-int4 ask:
# 4 pairs x (official litert-community int8 cell + own BOCTAV4 int4 cell), --runs 4 each,
# interleaved per model so each ratio is computed from adjacent cells.
#
# Phases:
#   1. PUSH (device plugged, USB — >1GB WiFi pushes fail with socket error 54):
#      re-push any zeroed/missing model file from the Mac HF cache / conversion outputs.
#   2. UNPLUG WAIT: polls until the cable is pulled (charging heat drives thermal=fair
#      within ~2 cells; see results/raw/2026-07-20-iphone-int4-retest/summary.md), then
#      settles 300 s.
#   3. CELLS: 8 launches over the WiFi devicectl tunnel, 180 s idle gaps.
#
# Validation is post-hoc: every imported run must have initialThermalState == nominal
# AND batteryState == unplugged in its device JSON (battery-temp telemetry lags — do
# not add a temp gate here).
#
# Env: SKIP_PUSH=1 (skip phase 1) · SKIP_PHI=1 (drop the 3.8B pair; q8 is 4 GB)
set -u
DEVICE=A6F3E849-1947-5202-9AD1-9C881CA58EEF
APP=com.example.CoreMLLLMChat
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO_ROOT/results/raw/$(date +%F)-iphone-int8int4-pair"
mkdir -p "$OUT"

# pair layout: int8 catalog id | int8 device dir/file | int8 Mac source (resolved below) | int4 catalog id
INT8_IDS=(
  "litert-community/DeepSeek-R1-Distill-Qwen-1.5B"
  "litert-community/TinySwallow-1.5B-Instruct"
  "litert-community/VibeThinker-1.5B"
  "litert-community/Phi-4-mini-instruct"
)
INT8_FILES=(
  "DeepSeek-R1-Distill-Qwen-1.5B_multi-prefill-seq_q8_ekv4096.litertlm"
  "TinySwallow-1.5B-Instruct.litertlm"
  "VibeThinker-1.5B.litertlm"
  "Phi-4-mini-instruct_multi-prefill-seq_q8_ekv4096.litertlm"
)
INT4_IDS=(
  "own/DeepSeek-R1-1.5B-int4-BOCTAV4"
  "own/TinySwallow-1.5B-int4-BOCTAV4"
  "own/VibeThinker-1.5B-int4-BOCTAV4"
  "own/Phi-4-mini-int4-BOCTAV4-128"
)
# NOTE: DeepSeek own-int4 has NO Mac copy (2026-07-20 audit) — the only copy is the
# device file at own__DeepSeek-R1-1.5B-int4-BOCTAV4/model.litertlm (1.03 GB, healthy).
# FIRST thing on device return: back it up Mac-side (see the backup block below).
INT4_SRCS=(
  "$HOME/code/litertlm-convert/out/deepseek-r1-1.5b-int4-BOCTAV4/model.litertlm"
  "$HOME/code/litertlm-convert/out/tinyswallow-1.5b-int4-v2/model.litertlm"
  "$HOME/code/litertlm-convert/out/vibethinker-1.5b-int4-v2/model.litertlm"
  "$HOME/code/litertlm-convert/out/phi4-mini-instruct-int4/model.litertlm"
)
NPAIR=4
[[ "${SKIP_PHI:-0}" == "1" ]] && NPAIR=3

device_size() {  # $1 = device dir under Documents/models/litert-lm, $2 = filename
  xcrun devicectl device info files --device "$DEVICE" --domain-type appDataContainer \
    --domain-identifier "$APP" --subdirectory "Documents/models/litert-lm/$1" 2>/dev/null \
    | grep -F "$2" | head -1
}

push_if_needed() {  # $1 = device dir, $2 = filename, $3 = Mac source path
  local line=$(device_size "$1" "$2")
  if echo "$line" | grep -q " GB "; then
    echo "  OK on device: $1/$2" | tee -a "$OUT/driver.log"; return 0
  fi
  [[ -f "$3" ]] || { echo "  MISSING Mac source: $3" | tee -a "$OUT/driver.log"; return 1; }
  echo "  pushing $(du -h "$3" | cut -f1) -> $1/$2" | tee -a "$OUT/driver.log"
  xcrun devicectl device copy to --device "$DEVICE" \
    --source "$3" --destination "Documents/models/litert-lm/$1/$2" \
    --domain-type appDataContainer --domain-identifier "$APP" >/dev/null 2>&1
  device_size "$1" "$2" | grep -q " GB " || { echo "  PUSH FAILED: $1/$2" | tee -a "$OUT/driver.log"; return 1; }
}

run_cell() {  # $1 = catalog id
  local fname="${1//\//_}"; fname="${fname//./_}"
  local log="$OUT/console_litert-lm_${fname}.txt"
  echo "=== $(date '+%F %T') START $1 ===" >> "$OUT/driver.log"
  gtimeout 900 xcrun devicectl device process launch --console --terminate-existing \
    --device "$DEVICE" "$APP" -- \
    --yardstick-autorun --runtime litert-lm --model-id "$1" --task short-chat --runs 4 \
    </dev/null > "$log" 2>&1
  local rc=$?
  local ok=$(grep -c YARDSTICK_RUN_OK "$log")
  echo "=== $(date '+%F %T') END $1 exit=$rc run_ok=$ok ===" >> "$OUT/driver.log"
  grep YARDSTICK_RUN_OK "$log"
}

echo "session $(date '+%F %T') int8int4-pair" > "$OUT/driver.log"
xcrun devicectl list devices 2>/dev/null | grep -q "iPhone 17 Pro (iPhone18,1).*physical" || { echo "device not connected"; exit 1; }

# phase 0: back up the device-only DeepSeek own-int4 to the Mac (skipped once it exists)
if [[ ! -f "${INT4_SRCS[1]}" ]]; then
  echo "--- phase 0: backing up DeepSeek own-int4 from device (only copy) ---" | tee -a "$OUT/driver.log"
  mkdir -p "$(dirname "${INT4_SRCS[1]}")"
  xcrun devicectl device copy from --device "$DEVICE" \
    --source "Documents/models/litert-lm/own__DeepSeek-R1-1.5B-int4-BOCTAV4/model.litertlm" \
    --destination "${INT4_SRCS[1]}" \
    --domain-type appDataContainer --domain-identifier "$APP" >/dev/null 2>&1
  [[ -f "${INT4_SRCS[1]}" ]] && echo "  backed up: $(du -h "${INT4_SRCS[1]}" | cut -f1)" | tee -a "$OUT/driver.log" \
    || echo "  BACKUP FAILED (continue; device copy is the working source)" | tee -a "$OUT/driver.log"
fi

if [[ "${SKIP_PUSH:-0}" != "1" ]]; then
  echo "--- phase 1: push (USB required) ---" | tee -a "$OUT/driver.log"
  for i in {1..$NPAIR}; do
    repo_dir="${INT8_IDS[$i]//\//__}"
    src=$(hf download "${INT8_IDS[$i]}" "${INT8_FILES[$i]}" 2>/dev/null | tail -1)
    src="${src:A}"  # resolve the HF-cache symlink — devicectl rejects symlinks containing ".."
    push_if_needed "$repo_dir" "${INT8_FILES[$i]}" "$src" || exit 1
    int4_dir="${INT4_IDS[$i]//\//__}"
    push_if_needed "$int4_dir" "model.litertlm" "${INT4_SRCS[$i]}" || exit 1
  done
fi

echo "--- phase 2: waiting for UNPLUG (pull the cable now) ---" | tee -a "$OUT/driver.log"
while pymobiledevice3 diagnostics battery single 2>/dev/null | grep -q '"ExternalConnected": true'; do sleep 20; done
echo "$(date '+%T') unplugged; settling 300s" | tee -a "$OUT/driver.log"
sleep 300

echo "--- phase 3: cells (pairs interleaved, 180s gaps) ---" | tee -a "$OUT/driver.log"
for i in {1..$NPAIR}; do
  run_cell "${INT8_IDS[$i]}"; sleep 180
  run_cell "${INT4_IDS[$i]}"; sleep 180
done
echo "=== $(date '+%F %T') PAIR SESSION DONE ===" | tee -a "$OUT/driver.log"
