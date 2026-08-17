#!/usr/bin/env python3
"""Convert Apple llm-benchmark --output-json into schema-v1 per-trial records.

One record per trial (spread stays visible — spread-rule). Trials repeat
in-process, so coldRun=false. llm-benchmark reports its own timing and has no
context budget flag; that caveat travels in provenance.note, and its rows are
never protocol-identical with yardstick rows.
"""
import argparse
import json
import os
import platform
import subprocess


def mac_model_identifier():
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "hw.model"], text=True).strip()
    except Exception:
        return "Mac"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--engine-version", required=True)
    ap.add_argument("--quant", default="unrecorded")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    d = json.load(open(args.json_path))
    os.makedirs(args.out_dir, exist_ok=True)
    stamp = os.path.basename(args.json_path).replace(".llm-benchmark.json", "")
    for i, trial in enumerate(d.get("trials", []), 1):
        rec = {
            "schemaVersion": 1,
            "id": f"{stamp}-trial{i}",
            "runtime": "core-ai",
            "engineVersion": args.engine_version,
            "model": {"id": args.model_id, "quantization": args.quant,
                      "bundle": d.get("model")},
            "task": args.task,
            "timestamp": None,  # llm-benchmark emits no per-trial clock
            "device": {"modelIdentifier": mac_model_identifier(),
                       "systemName": "macOS",
                       "systemVersion": platform.mac_ver()[0]},
            "conditions": {"promptTokens": d.get("prompt_tokens"),
                           "generationTokens": d.get("generation_tokens"),
                           "chunkThreshold": 1},
            "metrics": {
                "decodeTokensPerSecond": trial.get("gen_tps"),
                "promptTokensPerSecond": trial.get("prompt_tps"),
                "promptTokenCount": d.get("prompt_tokens"),
                "generatedTokenCount": d.get("generation_tokens"),
                "coldRun": False,  # in-process trial repeats
                "harnessStamp": "coreai-llm-benchmark-import-v1",
            },
            "provenance": {
                "rawLog": os.path.basename(args.json_path),
                "note": ("external Apple llm-benchmark harness: own timing, no "
                         "context-tokens flag — not protocol-identical with "
                         "yardstick rows; engine needs the unpublished "
                         "COREAI_STATIC_INPUTS patch for PLE models"),
            },
        }
        name = f"core-ai_{args.model_id.replace('/', '_')}_{args.task}_trial{i}.json"
        json.dump(rec, open(os.path.join(args.out_dir, name), "w"), indent=2)
    print(f"imported {len(d.get('trials', []))} trials -> {args.out_dir}")


if __name__ == "__main__":
    main()
