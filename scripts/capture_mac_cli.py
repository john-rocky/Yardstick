#!/usr/bin/env python3
"""Run a Mac benchmark CLI and store its result where the audit can find it.

    python3 scripts/capture_mac_cli.py \
        --runtime litert-lm --model litert-community/gemma-4-E2B-it-litert-lm \
        --task long-context-1024 --out results/raw/2026-07-26-mac \
        -- litert-mac-verify --which int4 --prefill 1024 --decode 256

Why this exists
---------------
The iPhone table is auditable cell by cell because the app writes a JSON per run. The Mac
table is not: verified 2026-07-26, **not one** of its six speed cells (8,505 / 171.3 / 7,305 /
153.2 / 82.4 / 75.9) or its five energy values appears anywhere in `results/`. They were read
off the consoles of three external tools — Core AI's `llm-benchmark`, `mlx_lm.benchmark`,
`litert-mac-verify` — and never written back. That is how the Mac energy row could carry a
WebGPU measurement under a native label for a week without anyone being able to check it.

This wrapper closes that path. It runs the tool, **always** stores the raw console output
verbatim, and writes a repo-shaped result JSON next to it. Parsing is best-effort; storage is
not. A cell whose numbers could not be parsed still leaves a log with the exact argv that
produced it, which is the difference between "check the log" and "nobody can reproduce this".

Recognised output today:
  * `--output-json` from Core AI's llm-benchmark (preferred — pass it and this reads the file)
  * `run N: decode X tok/s · prefill Y tok/s`     (litert-mac-verify, mlx bench wrappers)
  * `yardstick: TTFT=Nms decode=Xtok/s peakMB=N`  (the repo's own Mac CLI line)
Anything else is stored unparsed and flagged, rather than dropped.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import statistics as st
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

RUN_LINE = re.compile(r"decode\s+([\d.]+)\s*tok/s.*?prefill\s+([\d.]+)\s*tok/s", re.I)
YARDSTICK_LINE = re.compile(r"TTFT=(\d+)ms\s+decode=([\d.]+)tok/s\s+peakMB=([\d.]+)", re.I)
CAPTURE_STAMP = "2026-07-26-mac-cli"


def parse_console(text: str) -> dict:
    """Pull the per-run rates out of a console log. Medians, because every one of these
    tools prints one line per trial and a single line is a cold-cache artefact."""
    decodes, prefills, ttfts, peaks = [], [], [], []
    for m in RUN_LINE.finditer(text):
        decodes.append(float(m.group(1)))
        prefills.append(float(m.group(2)))
    for m in YARDSTICK_LINE.finditer(text):
        ttfts.append(float(m.group(1)))
        decodes.append(float(m.group(2)))
        peaks.append(float(m.group(3)))
    out = {}
    if decodes:
        out["decodeTokensPerSecond"] = st.median(decodes)
        out["decodeSamples"] = decodes
    if prefills:
        out["promptTokensPerSecond"] = st.median(prefills)
        out["prefillSamples"] = prefills
    if ttfts:
        out["firstTokenLatencyMS"] = int(st.median(ttfts))
    if peaks:
        out["memoryPeakDuringDecodeMB"] = st.median(peaks)
    return out


def parse_output_json(path: Path) -> dict:
    """Core AI's llm-benchmark --output-json. Key names differ by version, so map what we
    recognise and keep the whole document under `toolSummary` regardless."""
    try:
        doc = json.loads(path.read_text())
    except Exception:
        return {}
    flat = {}

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, (int, float)):
                    flat[k] = v
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(doc)
    out = {"toolSummary": doc}
    for src, dst in (("prefill_tokens_per_second", "promptTokensPerSecond"),
                     ("prefillTokensPerSecond", "promptTokensPerSecond"),
                     ("decode_tokens_per_second", "decodeTokensPerSecond"),
                     ("decodeTokensPerSecond", "decodeTokensPerSecond"),
                     ("peak_memory_mb", "memoryPeakDuringDecodeMB")):
        if src in flat:
            out[dst] = float(flat[src])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, help="e.g. long-context-1024, sustained-generation")
    ap.add_argument("--device", default="m4max")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--note", default="", help="anything the numbers need stated with them")
    ap.add_argument("--output-json", type=Path, default=None,
                    help="a summary JSON the tool itself wrote (llm-benchmark --output-json)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        print("no command given (put it after `--`)")
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    slug = f"{args.device}_{args.runtime}_{args.model.split('/')[-1]}_{args.task}_{stamp}"
    log_path = args.out / f"{slug}.log"

    print(f"$ {' '.join(shlex.quote(c) for c in cmd)}\n")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    console = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
    log_path.write_text(
        f"# command: {' '.join(shlex.quote(c) for c in cmd)}\n"
        f"# exit: {proc.returncode}   elapsed: {elapsed:.1f}s   captured: {stamp}\n\n{console}")
    print(console[-2000:] if console else "(no output)")

    metrics = parse_console(console)
    if args.output_json and args.output_json.exists():
        metrics.update(parse_output_json(args.output_json))

    record = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runtime": args.runtime,
        "task": args.task,
        "model": {"id": args.model},
        "device": {"modelIdentifier": args.device, "systemName": "macOS"},
        "metrics": metrics,
        "provenance": {
            "captureStamp": CAPTURE_STAMP,
            "tool": Path(cmd[0]).name,
            "argv": cmd,
            "exitCode": proc.returncode,
            "elapsedSeconds": round(elapsed, 1),
            "consoleLog": log_path.name,
            "note": args.note,
            "parsed": bool(metrics),
        },
    }
    json_path = args.out / f"{slug}.json"
    json_path.write_text(json.dumps(record, indent=2) + "\n")

    print(f"\nlog    -> {log_path}")
    print(f"result -> {json_path}")
    if metrics:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                print(f"   {k} = {v:,.2f}")
    else:
        print("   !! nothing parsed — the log is stored, but add a parser for this tool's "
              "format before quoting a number from it")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
