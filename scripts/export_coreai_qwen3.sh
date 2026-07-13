#!/usr/bin/env bash
#
# export_coreai_qwen3.sh — export + AOT-assemble the two iPhone Core AI bundles
# (GPU dynamic-INT4 + ANE static palettized-4bit) for ANY Qwen3 dense size.
#
# Generalises the 0.6B `prep_coreai` step in bench_coreai_iphone.sh so a new
# size (1.7B, …) is one command. Runs on the Mac against the coreai-models repo;
# the finished bundles land in $EXPORTS, ready for the device side-load step in
# bench_qwen3_iphone.sh. The Qwen3 architecture is size-agnostic (dense GQA),
# so the same export pipeline that produced 0.6B/4B/8B is reused verbatim.
#
#   ANE row : iOS static export, 4bit_weight_palettized_group32 (LUT). Linear
#             INT4 SIGSEGVs the ANE pre-compiler — palettised is the leanest
#             scheme that runs on the ANE (docs/litert-lm/COREAI_INT4_EXPORT.md).
#   GPU row : macOS dynamic export, linear INT4 → coreai-pipelined GPU engine.
#
# Usage: scripts/export_coreai_qwen3.sh <size_us> <hf-id> [ane_ctx] [arch]
#   e.g. scripts/export_coreai_qwen3.sh 1_7b Qwen/Qwen3-1.7B 4096 h18p
# <size_us> uses underscores for the dot (1.7B -> 1_7b), matching the existing
# qwen3_0_6b / qwen3_4b export-folder convention.
set -euo pipefail

SIZE_US="${1:?usage: export_coreai_qwen3.sh <size_us> <hf-id> [ane_ctx] [arch]}"
HF_ID="${2:?missing hf-id, e.g. Qwen/Qwen3-1.7B}"
ANE_CTX="${3:-4096}"
ARCH="${4:-h18p}"

COREAI="$HOME/code/coreai/coreai-models"
EXPORTS="$COREAI/exports"
BASE_DYN="qwen3_${SIZE_US}_dynamic"            # GPU IR (macOS dynamic)
BASE_ANE="qwen3_${SIZE_US}_ios_pure4bit"       # ANE IR (iOS static palettized)
OUT_GPU="$EXPORTS/qwen3_${SIZE_US}_gpu"
OUT_ANE="$EXPORTS/qwen3_${SIZE_US}_ane_pure4bit"

log(){ printf '\n=== %s ===\n' "$*"; }

# Assemble a loadable bundle from a compiled .aimodelc: device-arch file +
# tokenizer + a metadata.json whose assets.main points at the compiled file.
# (Verbatim from bench_coreai_iphone.sh — tokenizer/metadata live NEXT TO the IR.)
assemble() { # <ir.aimodel> <compute: gpu|neural-engine> <out-bundle-dir> <base>
  local ir="$1" compute="$2" out="$3" base="$4" tmp srcdir tok md c
  tmp="$(mktemp -d)"
  xcrun coreai-build compile "$ir" --platform iOS --preferred-compute "$compute" \
    --architecture "$ARCH" --output "$tmp"
  rm -rf "$out"; mkdir -p "$out"
  cp -R "$tmp/${base}.${ARCH}.aimodelc" "$out/"
  srcdir="$(dirname "$ir")"
  tok=""; for c in "$srcdir/tokenizer" "$srcdir/../tokenizer" "$srcdir/../../tokenizer"; do
    [ -d "$c" ] && { tok="$c"; break; }; done
  [ -n "$tok" ] || { echo "assemble: no tokenizer found near $ir" >&2; rm -rf "$tmp"; return 1; }
  cp -R "$tok" "$out/"
  md=""; for c in "$srcdir/metadata.json" "$srcdir/../metadata.json"; do
    [ -f "$c" ] && { md="$c"; break; }; done
  [ -n "$md" ] || { echo "assemble: no metadata.json found near $ir" >&2; rm -rf "$tmp"; return 1; }
  python3 - "$md" "$out/metadata.json" "${base}.${ARCH}.aimodelc" <<'PY'
import json,sys
m=json.load(open(sys.argv[1])); m["assets"]["main"]=sys.argv[3]
json.dump(m,open(sys.argv[2],"w"),indent=2)
PY
  rm -rf "$tmp"
  echo "assembled -> $out"
}

cd "$COREAI"

log "export GPU (macOS dynamic 4bit) -> $BASE_DYN"
# `.aimodel` is a directory (contains main.mlirb), so the guard must be -e, not -f.
[ -e "$EXPORTS/$BASE_DYN/$BASE_DYN.aimodel" ] || \
  uv run coreai.llm.export "$HF_ID" --platform macOS --compression 4bit \
    --compute-precision float16 --experimental --output-name "$BASE_DYN"

log "export ANE (iOS static palettized g32, ctx $ANE_CTX) -> $BASE_ANE"
[ -e "$EXPORTS/$BASE_ANE/$BASE_ANE.aimodel" ] || \
  uv run coreai.llm.export "$HF_ID" --platform iOS \
    --compression 4bit_weight_palettized_group32 --compute-precision float16 \
    --max-context-length "$ANE_CTX" --experimental --output-name "$BASE_ANE"

log "assemble GPU bundle (compile $ARCH, preferred-compute gpu)"
[ -f "$OUT_GPU/metadata.json" ] || \
  assemble "$EXPORTS/$BASE_DYN/$BASE_DYN.aimodel" gpu "$OUT_GPU" "$BASE_DYN"

log "assemble ANE bundle (compile $ARCH, preferred-compute neural-engine)"
[ -f "$OUT_ANE/metadata.json" ] || \
  assemble "$EXPORTS/$BASE_ANE/$BASE_ANE.aimodel" neural-engine "$OUT_ANE" "$BASE_ANE"

log "done — bundles:"
ls -d "$OUT_GPU" "$OUT_ANE"
