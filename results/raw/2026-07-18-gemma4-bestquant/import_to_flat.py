#!/usr/bin/env python3
"""Import the AUDITED captures of the 2026-07-18 Gemma-4-E2B best-available session into
the flat results/raw convention that render_results.py / generate_charts.py consume.

The selection below IS the audit decision (see SUMMARY.txt for the reasoning):
  - only thermal-nominal cold captures count toward medians;
  - the mlx-OptiQ fair/serious captures (post-download contamination) are excluded;
  - the core-ai 13:43 capture is excluded (broken-template probe, 5-token degenerate);
  - everything excluded here still lives in this directory for the audit trail.

Idempotent: re-running overwrites the same run files.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE.parent

SELECTION = {
    # slug: ([short-chat timestamps in run order], quality timestamp)
    "litert-lm-gemma-4-e2b": (
        ["2026-07-17T19-05-24Z", "2026-07-17T20-08-14Z", "2026-07-17T20-10-43Z"],
        "2026-07-17T19-06-04Z",
        "litert-lm_litert-community_gemma-4-E2B-it-litert-lm",
    ),
    "mlx-swift-gemma-4-e2b-optiq": (
        ["2026-07-17T21-02-08Z", "2026-07-17T21-04-38Z", "2026-07-17T21-07-42Z"],
        "2026-07-17T19-40-35Z",
        "mlx-swift_mlx-community_gemma-4-e2b-it-qat-OptiQ-4bit",
    ),
    "mlx-swift-gemma-4-e2b": (
        ["2026-07-18T00-52-05Z", "2026-07-18T00-57-29Z", "2026-07-18T05-16-23Z"],
        "2026-07-18T05-18-53Z",
        "mlx-swift_mlx-community_gemma-4-e2b-it-4bit",
    ),
    "llama-cpp-gemma-4-e2b": (
        ["2026-07-18T00-36-36Z", "2026-07-18T00-42-08Z", "2026-07-18T00-45-01Z"],
        # 2026-07-19: the 07-18T05-21-24Z quality capture was a bug artifact (adapter ran
        # every prompt untemplated; gemma-4's <|turn> markers made the first sampled token
        # the EOS on all tasks but short-chat -> empty output). Post-fix capture below;
        # the empty record stays in this directory for the audit trail.
        "2026-07-19T07-56-08Z",
        "llama.cpp_unsloth_gemma-4-E2B-it-GGUF_Q4_K_M",
    ),
    "core-ai-gemma-4-e2b": (
        ["2026-07-18T13-46-16Z", "2026-07-18T13-47-28Z", "2026-07-18T13-53-09Z"],
        "2026-07-18T13-57-19Z",
        "core-ai_core-ai_gemma4-e2b-gpu",
    ),
}


# Deep-context (long-context-1024, p=1024/g=256 protocol) and energy (600 s battery-delta)
# captures, added 2026-07-19. litert's deep-context row comes from its own benchmark
# counters and core-ai jetsams under the deep protocol, so neither has files here.
# llama.cpp's 07-19T05-* long-context captures were the adapter-bug empty-output runs
# (prefill-only); superseded by the post-fix 07-56/07-57/07-58 set — audit trail as above.
LONG_CONTEXT = {
    "mlx-swift-gemma-4-e2b": ["2026-07-19T05-27-21Z", "2026-07-19T05-29-25Z", "2026-07-19T05-31-30Z"],
    "mlx-swift-gemma-4-e2b-optiq": ["2026-07-19T05-33-44Z", "2026-07-19T05-35-50Z", "2026-07-19T05-37-55Z"],
    "llama-cpp-gemma-4-e2b": ["2026-07-19T07-56-28Z", "2026-07-19T07-57-17Z", "2026-07-19T07-58-05Z"],
}
ENERGY = {
    "litert-lm-gemma-4-e2b": "2026-07-19T03-00-21Z",
    "mlx-swift-gemma-4-e2b": "2026-07-19T03-18-56Z",
    "mlx-swift-gemma-4-e2b-optiq": "2026-07-19T06-29-29Z",
    "core-ai-gemma-4-e2b": "2026-07-19T04-09-29Z",   # shallow-rep variant (--max-tokens 192)
    "llama-cpp-gemma-4-e2b": "2026-07-19T08-21-57Z", # post-fix 600 s standard protocol
    # (the 08-05-04Z / 08-06-11Z energy files are the fix-validation preflights — 45 s
    #  sustain, no battery delta; audit trail only, never imported)
}


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
    if (ets := ENERGY.get(slug)) and "TS" not in ets:
        copy(HERE / f"{prefix}_energy_{ets}.json",
             RAW / f"iphone17pro-{slug}-energy-run1.jsonl")
print("done")
