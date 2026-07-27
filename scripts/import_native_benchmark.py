#!/usr/bin/env python3
"""Turn `YARDSTICK_NATIVE_OK` console lines into result JSON so the native row is auditable.

    python3 scripts/import_native_benchmark.py results/raw/<campaign>/console_NATIVE_*.txt

Why this exists
---------------
`runNativeBenchmark` prints its numbers and returns — it never goes through `ResultStore`, so
the LiteRT-LM vendor-`benchmark()` row (prefill exactly 1024, the ONLY card-comparable prefill
figure we can produce) lands in a `.txt` and nowhere else. `analyze_comparability.py` reads
`*.json`; `verify_published_numbers.py` indexes `results/**/*.json{,l}`. A number that exists
only in a console log is exactly the "number that exists only in prose" those two scripts were
written to refuse. This lifts the console line into the same schema the app writes, so the
native row is audited by the same tools as everything else.

What it deliberately does NOT do
--------------------------------
Invent fields the native path never measured. `runNativeBenchmark` samples memory but starts
no `ThermalSampler` and no `EnergyMonitor`, so thermal/battery/energy are written as null
rather than as a plausible-looking "nominal". `analyze_comparability.py` gates its speed table
on `initialThermalState == "nominal"`, so these rows are excluded from it by default — which
is the correct outcome: we cannot attest the thermal regime of a cell that did not record one.
Pass `--all-thermal` to inspect them anyway, and read the exclusion as a TODO for the app
(the native path should start a ThermalSampler like `BenchmarkRunner` does).

The task id is `native-benchmark-<prefill>x<decode>`, never `long-context-*`: a forced-prefill
vendor entry point and a task-prompt run are different measurements and must not pool into one
median. That separation is the whole point of the protocol's two-row structure.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# One regex over the whole line; every field is optional so older console logs (which lack
# context_tokens / median_mb / harness) still import, with the missing fields left null
# instead of silently defaulted.
#
# The value pattern is a plain `\S+` on purpose. An earlier `([-\d.]+|\?|[\w.-]+)` truncated
# `harness=2026-07-27-agreed-protocol-r2` to `2026-07-27-`, because the numeric alternative
# matched the leading date and won. A stamp that silently loses its suffix is worse than no
# stamp at all — it reads as a valid, different contract. Numeric coercion happens in `num()`,
# where a non-numeric value returns None rather than a wrong number.
FIELD = re.compile(r"(\w+)=(\S+)")

MODEL_RE = re.compile(r"YARDSTICK_BEGIN native_benchmark model=(\S+)")
BEGIN_RE = re.compile(r"prefill=(\d+) decode=(\d+)")


def parse(path: Path):
    text = path.read_text(errors="replace").replace("\r", "\n")
    model = None
    prefill_cfg = decode_cfg = None
    for line in text.split("\n"):
        if m := MODEL_RE.search(line):
            model = m.group(1)
        if "native_benchmark" in line and (m := BEGIN_RE.search(line)):
            prefill_cfg, decode_cfg = int(m.group(1)), int(m.group(2))
        if "YARDSTICK_NATIVE_OK" not in line:
            continue
        f = dict(FIELD.findall(line.split("YARDSTICK_NATIVE_OK", 1)[1]))

        def num(k, cast=float):
            v = f.get(k)
            if v in (None, "?"):
                return None
            try:
                return cast(v)
            except ValueError:
                return None

        yield {
            "model": model,
            "prefill_cfg": prefill_cfg,
            "decode_cfg": decode_cfg,
            "fields": f,
            "num": num,
        }


def to_result(rec, source: Path, device_id: str):
    f, num = rec["fields"], rec["num"]
    prefill = num("prefill_tokens", int) or rec["prefill_cfg"]
    decode = num("decode_tokens", int) or rec["decode_cfg"]
    task = f"native-benchmark-{rec['prefill_cfg']}x{rec['decode_cfg']}"
    return {
        "runtime": "litert-lm",
        "task": task,
        "model": {"id": rec["model"]},
        "device": {"modelIdentifier": device_id},
        "outputSample": "",
        # Provenance is part of the record, not a comment: this row was lifted from a console
        # log by this script, and anyone auditing it should be able to go straight back to the
        # line it came from.
        "provenance": {
            "importedBy": "scripts/import_native_benchmark.py",
            "sourceFile": str(source),
            "entryPoint": "LiteRTLM.benchmark()",
            "note": "vendor force-prefill entry point; not comparable with task-prompt prefill",
        },
        "metrics": {
            "coldRun": None,
            "promptTokenCount": prefill,
            "promptTokensPerSecond": num("prefill_tok_s"),
            "generatedTokenCount": decode,
            "decodeTokensPerSecond": num("decode_tok_s"),
            "firstTokenLatencyMS": num("ttft_ms"),
            "loadTimeSeconds": num("init_s"),
            "memoryPeakDuringDecodeMB": num("peak_mb"),
            "memoryMedianMB": num("median_mb"),
            "memoryMedianResidentMB": num("median_resident_mb"),
            "memorySampleCount": num("samples", int),
            # The published 92 MB cell was this quantity. Kept as its own field so it can never
            # again be mistaken for the in-run peak sitting next to it.
            "memoryPostTeardownFootprintMB": num("teardown_footprint_mb"),
            "contextTokensConfigured": num("context_tokens", int),
            "harnessStamp": f.get("harness"),
            # Not measured by the native path — see the module docstring.
            "initialThermalState": None,
            "peakThermalState": None,
            "energyJoules": None,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--device", default="iPhone18,1")
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir (default: alongside each console log)")
    args = ap.parse_args()

    written = 0
    for log in args.logs:
        if not log.exists():
            print(f"warn: no such file: {log}", file=sys.stderr)
            continue
        for i, rec in enumerate(parse(log), 1):
            result = to_result(rec, log, args.device)
            outdir = args.out or log.parent
            outdir.mkdir(parents=True, exist_ok=True)
            stem = log.stem.replace("console_", "")
            out = outdir / f"native_{stem}_{i}.json"
            out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(f"wrote {out}", file=sys.stderr)
            written += 1

    if not written:
        print("no YARDSTICK_NATIVE_OK lines found", file=sys.stderr)
        return 1
    print(f"\n{written} native row(s) imported.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
