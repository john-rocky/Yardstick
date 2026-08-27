#!/usr/bin/env bash
# Generic Mac matrix runner — loops `yardstick run` over the mac cells of a
# cells file (matrices/README.md grammar).
#
#   scripts/bench_matrix_mac.sh run matrices/release-regression-litert.cells
#
# Binary resolution: $YS_BIN -> $DD_MAC/Build/Products/Release/yardstick
# (scripts/build_yardstick_mac.sh) -> error. The runner REFUSES the spm-lite
# flavor (SwiftPM build; four runtimes compiled out — silently wrong matrix)
# unless YS_ALLOW_SPM=1.
#
# core-ai cells dispatch to scripts/coreai_mac_wrapper.sh (external Apple
# llm-benchmark binary; own timing, no --context-tokens — documented
# comparability caveat). A missing external binary logs SKIPPED, not fatal:
# the arm is best-effort until Apple publishes the static-inputs API.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
. "$REPO/scripts/lib/matrix_common.sh"

# Repo-local build dir: an out-of-repo default (~/bench-dd-mac) let a fresh
# clone silently run whatever stale yardstick another checkout had built —
# an unpinned harness under this clone's protocol (2026-08-26 rehearsal).
DD_MAC="${DD_MAC:-$REPO/.build/dd-mac}"
YS="${YS_BIN:-$DD_MAC/Build/Products/Release/yardstick}"
# Engine-pin witness: without this the CLI stamps nothing and rows read
# "pre-stamp" (found 2026-08-27 — the convention lived in the lockfile but
# no runner wired it). stamp_engine_pins.sh regenerates the file at build.
export BENCH_ENGINE_PINS_FILE="${BENCH_ENGINE_PINS_FILE:-$REPO/ios/BenchmarkApp/Vendored/engine-pins.json}"
DEFAULT_RUNS="${RUNS:-4}"
BASE_COOLDOWN="${BASE_COOLDOWN:-30}"
CAMPAIGN="${CAMPAIGN:-$(date +%F)-mac-matrix}"
OUT="$REPO/results/raw/$CAMPAIGN"

log(){ printf '\n=== %s ===\n' "$*"; }

run_ys_cell(){
  # One capture attempt for the current loop cell (bash dynamic scope: rt/mid/
  # task_arg/runs/extra/slug are cmd_run locals).
  "$YS" run --runtime "$rt" --model-id "$mid" --task "$task_arg" --runs "$runs" \
    ${extra[@]+"${extra[@]}"} --output "$OUT/${slug}.jsonl" 2>&1 | tail -4
}

guard(){
  # Unified memory: a heavy CPU/GPU pipeline moves these numbers (spread-rule).
  # NB: pattern must not match the repo name "ios-llm-benchmark" in task paths.
  if ps aux | grep -E "coreai\.llm\.export|release/llm-benchmark |export_simple_template\.py" | grep -v grep >/dev/null; then
    echo "refusing to start: heavy pipeline running (unified-memory contention)" >&2; exit 1
  fi
}

check_binary(){
  [ -x "$YS" ] || { echo "no yardstick at $YS — run scripts/build_yardstick_mac.sh" >&2; exit 1; }
  local ver flavor
  ver="$("$YS" version 2>/dev/null || true)"
  flavor="$(sed -n 's/.*flavor=\([a-z-]*\).*/\1/p' <<<"$ver")"
  if [ -z "$flavor" ]; then
    echo "WARNING: $YS predates 'yardstick version' — cannot verify build flavor" >&2
  elif [ "$flavor" = "spm-lite" ] && [ "${YS_ALLOW_SPM:-0}" != "1" ]; then
    echo "refusing spm-lite yardstick ($YS): llama-cpp/coreml/executorch/anemll are" >&2
    echo "compiled out of the SwiftPM build. Use scripts/build_yardstick_mac.sh" >&2
    echo "(or YS_ALLOW_SPM=1 if every mac cell is mlx/litert/apple-fm)." >&2
    exit 1
  fi
  echo "yardstick: $YS ${ver:+($ver)}"
}

cmd_run(){
  local cells_file="${1:?usage: run <cells-file>}"
  python3 "$REPO/scripts/validate_cells.py" "$cells_file" || exit 1
  check_binary
  guard
  mkdir -p "$OUT"
  { sw_vers; date "+session start %F %T"; echo "cells: $cells_file"; } >> "$OUT/session_provenance.txt"

  local first=1
  while read -r rt mid task rest; do
    read -r -a opts <<<"${rest:-}"
    local runs cool ctx maxtok slug
    runs="$(cell_opt runs "$DEFAULT_RUNS" ${opts[@]+"${opts[@]}"})"
    cool="$(cell_opt cooldown "$BASE_COOLDOWN" ${opts[@]+"${opts[@]}"})"
    ctx="$(cell_opt context-tokens "" ${opts[@]+"${opts[@]}"})"
    maxtok="$(cell_opt max-tokens "" ${opts[@]+"${opts[@]}"})"
    slug="$(echo "${rt}_${mid}_${task}" | tr '/.' '__')"

    [ "$first" = 1 ] && first=0 || { log "cooldown ${cool}s"; sleep "$cool"; }

    if [ "$rt" = "core-ai" ]; then
      "$REPO/scripts/coreai_mac_wrapper.sh" "$mid" "$task" "$runs" "$OUT" \
        || echo "FAIL core-ai $mid $task" >> "$OUT/FAILURES.txt"
      continue
    fi
    if [ "$rt" = "cactus" ]; then
      echo "SKIPPED $rt $mid $task reason=no-mac-arm" | tee -a "$OUT/SKIPPED.txt"
      continue
    fi

    # Resume-safe: a JSONL that already holds >= runs records is a finished cell.
    if [ "${FORCE:-0}" != "1" ] && [ -f "$OUT/${slug}.jsonl" ] \
       && [ "$(grep -c '"task"' "$OUT/${slug}.jsonl" 2>/dev/null || echo 0)" -ge "$runs" ]; then
      log "SKIP $slug (already captured; FORCE=1 to redo)"
      continue
    fi

    local extra=() task_arg="$task"
    [ -n "$ctx" ] && extra+=(--context-tokens "$ctx")
    [ -n "$maxtok" ] && extra+=(--max-tokens "$maxtok")
    case "$task" in native-benchmark-*)
      # The native benchmark runs INSTEAD of a task (yardstick resolves --task
      # before the native branch, so it must still name a real task id).
      extra+=(--litert-native-benchmark "${task#native-benchmark-}")
      task_arg="short-chat" ;;
    esac

    log "CELL $rt / $mid / $task runs=$runs ($(date +%H:%M:%S))"
    if ! run_ys_cell; then
      echo "FAIL $rt $mid $task" >> "$OUT/FAILURES.txt"
    fi
    # Post-capture gate (scripts/cell_gate.py): a HOT or wide-spread capture is
    # quarantined (.jsonl.attempt1 — kept in raw, outside build_summary's
    # *.jsonl glob) and the cell re-runs ONCE after a real cooldown. SHORT is
    # never retried here — that is a failure, and failed runs stay.
    if [ -f "$OUT/${slug}.jsonl" ] && [ "${GATE_RETRY:-1}" = "1" ]; then
      gate="$(python3 "$REPO/scripts/cell_gate.py" --runs "$runs" --jsonl "$OUT/${slug}.jsonl")" || true
      case "$gate" in HOT*|SPREAD*)
        log "gate: $gate — quarantine + cooldown ${GATE_COOLDOWN:-180}s, re-run once"
        mv "$OUT/${slug}.jsonl" "$OUT/${slug}.jsonl.attempt1"
        sleep "${GATE_COOLDOWN:-180}"
        run_ys_cell || echo "FAIL $rt $mid $task (gate retry)" >> "$OUT/FAILURES.txt"
        gate2="$(python3 "$REPO/scripts/cell_gate.py" --runs "$runs" --jsonl "$OUT/${slug}.jsonl" 2>/dev/null)" || true
        case "$gate2" in HOT*|SPREAD*)
          echo "GATE_FAIL $rt $mid $task first='$gate' retry='$gate2' (retry kept; ⚠ downstream)" \
            | tee -a "$OUT/FLAGGED.txt" ;;
        esac ;;
      esac
    fi
  done < <(cells_for mac "$cells_file" 2> >(tee -a "$OUT/SKIPPED.txt" >&2))

  log "campaign dir: $OUT"
  [ -f "$OUT/FAILURES.txt" ] && { echo "failures (failed-runs-stay — keep in the table):"; cat "$OUT/FAILURES.txt"; }
  [ -f "$OUT/SKIPPED.txt" ] && { echo "skipped:"; cat "$OUT/SKIPPED.txt"; }
  exit 0
}

case "${1:-}" in
  run) shift; cmd_run "$@" ;;
  *) sed -n '2,18p' "$0"; exit 1 ;;
esac
