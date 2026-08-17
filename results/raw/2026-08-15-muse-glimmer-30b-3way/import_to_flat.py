#!/usr/bin/env python3
"""Synthesize schema-v1 per-run records from this campaign's interleaved CLI log
so the capture enters results/summary/ (it predates the matrix runners; the raw
log stays canonical — stored-report-rule).

Reads  threeway-interleaved-192.log   (numbers are parsed, never transcribed)
Writes app-path-import/*.json         (picked up by build_summary.py's app-path* glob)

Arm metadata below is copied from ENV.md, the audited capture record. Hardware
state (thermal/battery) was NOT instrumented at capture — those fields are
absent, not defaulted. The Core AI arm is the fork engine (reference), not the
repo's pinned 0.2.0; that disclosure travels in provenance.note.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "threeway-interleaved-192.log")
OUT = os.path.join(HERE, "app-path-import")

ARMS = {
    "CA": {
        "runtime": "core-ai",
        "engineVersion": "john-rocky/coreai-models@58aab35 + session diff (fork zoo-0.2, reference — NOT pinned 0.2.0)",
        "model": {"id": "mlboydaisuke/Muse-Glimmer-30B-CoreAI", "hfRevision": "dcda323",
                  "quantization": "int4 block-32 sym, absmax head (int4hu)"},
    },
    "MLX": {
        "runtime": "mlx-vlm",  # pip mlx-vlm CLI, not the in-app mlx-swift arm
        "engineVersion": "mlx-vlm 0.6.13 / mlx 0.32.0",
        "model": {"id": "mlx-community/Muse-Glimmer-30B-4bit", "hfRevision": "3e7677d",
                  "quantization": "community 4-bit"},
    },
    "ET": {
        "runtime": "executorch",
        "engineVersion": "pytorch/executorch@abc5586 + empty-data-path diff (patches/)",
        "model": {"id": "meta-models/Muse-Glimmer-30B-ExecuTorch-PTE", "hfRevision": "fc6fa93",
                  "quantization": "Meta official k-quant (17G text-solo-metal)"},
    },
}

RE_CA = re.compile(r"CA\s+Generation: [\d.]+ms, (\d+) tokens, ([\d.]+) tokens/sec")
RE_ET = re.compile(r"ET\s+Decode: (\d+) tokens in [\d.]+ ms \(([\d.]+) tok/s\)")
RE_MLX_P = re.compile(r"MLX\s+Prompt: (\d+) tokens, ([\d.]+) tokens-per-sec")
RE_MLX_G = re.compile(r"MLX\s+Generation: (\d+) tokens, ([\d.]+) tokens-per-sec")


def main():
    os.makedirs(OUT, exist_ok=True)
    prompt = rnd = 0
    mlx_prefill = None  # (prompt_tokens, prefill_tps) pending its Generation line
    records = []
    for line in open(LOG):
        m = re.match(r"=== prompt (\d+)", line)
        if m:
            prompt = int(m.group(1)); continue
        m = re.match(r"\s+round (\d+)", line)
        if m:
            rnd = int(m.group(1)); continue
        if m := RE_CA.search(line):
            records.append(("CA", prompt, rnd, int(m.group(1)), float(m.group(2)), None, None))
        elif m := RE_ET.search(line):
            records.append(("ET", prompt, rnd, int(m.group(1)), float(m.group(2)), None, None))
        elif m := RE_MLX_P.search(line):
            mlx_prefill = (int(m.group(1)), float(m.group(2)))
        elif m := RE_MLX_G.search(line):
            pt, ptps = mlx_prefill if mlx_prefill else (None, None)
            records.append(("MLX", prompt, rnd, int(m.group(1)), float(m.group(2)), pt, ptps))
            mlx_prefill = None

    assert len(records) == 12, f"expected 3 arms x 2 prompts x 2 rounds, got {len(records)}"
    for arm, prompt, rnd, gen, dtps, ptok, ptps in records:
        meta = ARMS[arm]
        metrics = {
            "decodeTokensPerSecond": dtps,
            "generatedTokenCount": gen,
            "coldRun": True,  # fresh CLI process per run (fairness cold-warm-split)
            "harnessStamp": "2026-08-15-muse-3way-log-import",
        }
        if ptok is not None:
            metrics["promptTokenCount"] = ptok
            metrics["promptTokensPerSecond"] = ptps
        rec = {
            "schemaVersion": 1,
            "id": f"muse3way-{arm.lower()}-p{prompt}r{rnd}",
            "runtime": meta["runtime"],
            "engineVersion": meta["engineVersion"],
            "model": meta["model"],
            "task": "decode-192-greedy",
            "timestamp": "2026-08-15",  # log carries no per-run clock; date precision only
            "device": {"modelIdentifier": "Mac16,9", "systemName": "macOS",
                       "systemVersion": "27.0", "osBuild": "26A5406e"},
            "conditions": {"sampler": "greedy", "maxOutputTokens": 192,
                           "interleaved": True, "cooldownSeconds": 45,
                           "position": f"p{prompt}r{rnd}"},
            "metrics": metrics,
            "provenance": {
                "rawLog": "threeway-interleaved-192.log",
                "note": ("synthesized from the CLI log by import_to_flat.py; "
                         "thermal/battery not instrumented at capture (ENV.md); "
                         "Core AI arm = fork engine (reference), not pinned 0.2.0"),
            },
        }
        name = f"{meta['runtime']}_{meta['model']['id'].replace('/', '_')}_decode-192-greedy_p{prompt}r{rnd}.json"
        json.dump(rec, open(os.path.join(OUT, name), "w"), indent=2)
    print(f"wrote {len(records)} records -> {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main()
