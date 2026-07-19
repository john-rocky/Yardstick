#!/usr/bin/env python3
"""Import the AUDITED captures of the 2026-07-20 Cactus-arm session into the flat
results/raw convention that render_results.py / generate_charts.py consume.

The selection below IS the audit decision (see SUMMARY.txt):
  - Only post-fix captures count (the key-order bug invalidated the first pass:
    prompt_tokens=9 "empty user turn" records — those stay in this directory
    as the audit trail and are never imported);
  - per-bundle first-ever runs (fresh-binary Metal compile: uncal 18-08-58Z
    decode 36.9 load 2.4 s; cal 18-28-36Z decode 36.4 load 2.5 s) are excluded
    from cold medians, reported as first-ever in SUMMARY;
  - every imported capture is initialThermalState=nominal (verified via
    audit_pull.py);
  - the litert-lm re-anchor captures land under their own slug — they are a
    cross-session comparability CONTROL, not a replacement for the published
    2026-07-18 LiteRT row.

Idempotent: re-running overwrites the same run files.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE.parent

SELECTION = {
    # slug: ([short-chat timestamps in run order], quality_ts, device-file prefix)
    "cactus-gemma-4-e2b-cq4-uncal": (
        ["2026-07-19T18-09-24Z", "2026-07-19T18-11-55Z", "2026-07-19T18-14-26Z"],
        "2026-07-19T18-26-00Z",
        "cactus_Cactus-Compute_gemma-4-E2B-it-cq4-uncalibrated",
    ),
    "cactus-gemma-4-e2b-cq4": (
        ["2026-07-19T18-31-05Z", "2026-07-19T18-33-36Z", "2026-07-19T18-36-07Z"],
        "2026-07-19T18-38-37Z",
        "cactus_Cactus-Compute_gemma-4-E2B-it-cq4",
    ),
}

LONG_CONTEXT = {
    "cactus-gemma-4-e2b-cq4-uncal": [
        "2026-07-19T18-16-58Z", "2026-07-19T18-19-59Z", "2026-07-19T18-23-00Z",
    ],
}

ENERGY = {
    "cactus-gemma-4-e2b-cq4-uncal": "2026-07-19T18-55-27Z",  # 600 s standard deep protocol
}

# NOTE: the same-session LiteRT control colds (the cross-session bridge for the
# 2026-07-18 table) are deliberately NOT imported to flat — the flat pivots pool
# by (runtime, logical model), so importing them would pool a 07-20 capture into
# the published 07-18 LiteRT medians (cross-session pooling ban). The control
# values live in SUMMARY.txt and the raw JSONs stay in this directory.


def copy(src: Path, dest: Path) -> None:
    data = json.loads(src.read_text())
    dest.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"  {src.name} -> {dest.name}")


for slug, (shorts, quality_ts, prefix) in SELECTION.items():
    for i, ts in enumerate(shorts, start=1):
        copy(HERE / f"{prefix}_short-chat_{ts}.json",
             RAW / f"iphone17pro-{slug}-short-chat-run{i}.jsonl")
    copy(HERE / f"{prefix}_quality_{quality_ts}.json",
         RAW / f"iphone17pro-{slug}-quality-run1.jsonl")
    for i, ts in enumerate(LONG_CONTEXT.get(slug, []), start=1):
        copy(HERE / f"{prefix}_long-context-1024_{ts}.json",
             RAW / f"iphone17pro-{slug}-long-context-1024-run{i}.jsonl")
    if (ets := ENERGY.get(slug)):
        copy(HERE / f"{prefix}_energy_{ets}.json",
             RAW / f"iphone17pro-{slug}-energy-run1.jsonl")
print("done")
