#!/usr/bin/env bash
# Core AI Mac arm for the matrix runner — wraps Apple's external `llm-benchmark`
# CLI and imports its JSON into schema-v1 records.
#
#   coreai_mac_wrapper.sh <model-id> <task> <trials> <out-dir>
#
# Only native-benchmark-<P>x<D> tasks are meaningful here: llm-benchmark is a
# synthetic prefill/decode tool with its own timing and no --context-tokens
# (the documented comparability caveat — its rows are never protocol-identical
# with yardstick rows). Prompt tasks for Core AI run on iOS via the app.
#
# Env (defaults follow bench_gemma4_e2b_protocol_mac.sh):
#   COREAI_BIN      llm-benchmark binary (best-effort external arm — the engine
#                   needs the unpublished COREAI_STATIC_INPUTS patch for PLE models)
#   COREAI_EXPORTS  dir holding bundles; <model-id> basename resolves under it
#   COREAI_BUNDLE   full bundle path (overrides COREAI_EXPORTS resolution)
#   COREAI_RAWDIR   PLE raw dir (Gemma-4 PLE bundles only)
#   COREAI_QUANT    quantization label for the record (quant-per-arm-rule)
set -euo pipefail

MID="${1:?model-id}" TASK="${2:?task}" TRIALS="${3:-5}" OUT="${4:?out-dir}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

COREAI_BIN="${COREAI_BIN:-$HOME/code/coreai-models-020-bench/.build/out/Products/Release/llm-benchmark}"
COREAI_EXPORTS="${COREAI_EXPORTS:-$HOME/code/coreai/coreai-models/exports}"
BUNDLE="${COREAI_BUNDLE:-$COREAI_EXPORTS/$(basename "$MID")}"

case "$TASK" in
  native-benchmark-*x*) ;;
  *) echo "SKIPPED core-ai $MID $TASK reason=llm-benchmark-is-native-only" | tee -a "$OUT/SKIPPED.txt"; exit 0 ;;
esac
spec="${TASK#native-benchmark-}"
P="${spec%x*}"; G="${spec#*x}"

if [ ! -x "$COREAI_BIN" ]; then
  echo "SKIPPED core-ai $MID $TASK reason=external-binary-missing ($COREAI_BIN)" | tee -a "$OUT/SKIPPED.txt"
  exit 0
fi
[ -d "$BUNDLE" ] || { echo "SKIPPED core-ai $MID $TASK reason=bundle-missing ($BUNDLE)" | tee -a "$OUT/SKIPPED.txt"; exit 0; }

mkdir -p "$OUT"
slug="$(echo "core-ai_${MID}_${TASK}" | tr '/.' '__')"
args=(--model "$BUNDLE" -p "$P" -g "$G" -n "$TRIALS" --output-json "$OUT/${slug}.llm-benchmark.json")
[ -n "${COREAI_RAWDIR:-}" ] && args+=(--raw-dir "$COREAI_RAWDIR")

echo "=== Core AI (Apple llm-benchmark) $MID p=$P g=$G n=$TRIALS"
COREAI_CHUNK_THRESHOLD=1 "$COREAI_BIN" "${args[@]}" 2>&1 | tail -6

# Observed engine pin: the external checkout's git state (registry/witness convention).
ENGINE_SRC="${COREAI_SRC:-${COREAI_BIN%/.build/*}}"
ENGINE_PIN="$(git -C "$ENGINE_SRC" describe --always --dirty 2>/dev/null || echo unknown)"
python3 "$REPO/scripts/import_coreai_llm_benchmark.py" \
  "$OUT/${slug}.llm-benchmark.json" --model-id "$MID" --task "$TASK" \
  --engine-version "coreai-models-020-bench@$ENGINE_PIN (external, static-inputs patch unpublished)" \
  --quant "${COREAI_QUANT:-unrecorded (set COREAI_QUANT)}" --out-dir "$OUT/app-path-import"
