#!/usr/bin/env python3
"""Check that every measurement in a document traces back to a stored result file.

    python3 scripts/verify_published_numbers.py <doc.md> [doc2 ...] [options]

Why this exists
---------------
The 2026-07-19 Mac energy row (MLX-PTQ 0.090 / LiteRT 0.154 J/tok) was derived inside a
working session, quoted in the report, and never written back to a result file. Four months
of raw captures sit in `results/`, and not one of them contains those numbers — so nobody
downstream could reproduce them, and the row survived three revisions unchallenged. A number
that exists only in prose cannot be audited; this script refuses to let one ship.

What it does
------------
Indexes every numeric metric in `results/**/*.json{,l}`, pulls the measurement-shaped numbers
out of the document, and reports each one as traced or untraced. Unit-aware (GB<->MB) and
tolerant of the rounding a table applies.

Derived numbers — ratios, percentages, sums — legitimately do not appear in any result file.
Mark them at the end of the line so the intent is on the record:

    LiteRT uses 4.1x less memory than MLX   <!-- derived: 3388/818 -->

An unmarked, untraced number is the failure mode this script is looking for.

Exit code is 1 if any unmarked number fails to trace, so it can gate a commit.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import namedtuple
from pathlib import Path

Source = namedtuple("Source", "value file field runtime task")

# Every numeric metric is indexed, not a hand-picked list. The first cut of this script
# carried a curated METRIC_FIELDS tuple and omitted `energyJoulesPerTokenDecode` — so it
# reported the whole Mac energy row as untraceable when all four values were sitting in
# results/ under that exact key. A checker that decides in advance which fields count
# reproduces the failure it exists to catch.
SKIP_KEYS = {"promptTokenCount", "generatedTokenCount", "streamedChunkCount"}


def numeric_items(obj, prefix=""):
    """Every number anywhere under `metrics`, including nested objects and arrays."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from numeric_items(v, k)
    elif isinstance(obj, list):
        for v in obj:
            yield from numeric_items(v, prefix)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield prefix, float(obj)


# A number is measurement-shaped if it has a unit, a decimal point, or thousands grouping.
NUM = re.compile(
    r"(?<![\w.])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+)\s*"
    r"(GB|MB|KB|tok/s|tokens/s|J/tok|J|W|ms|s|%|×|x)?",
    re.I,
)
DERIVED = re.compile(r"<!--\s*derived\b", re.I)
EXTERNAL = re.compile(r"<!--\s*external\b", re.I)
# Numbers inside code spans are constants, shapes and buffer sizes, not measurements:
# `clamp(round(x/s),-128,127)*s`, a 145,920 B scratch heap, `262144 x 8960 x 64`.
CODE_SPAN = re.compile(r"`[^`]*`")


RUNTIME_HINTS = {
    "litert": "litert-lm", "wna8o8": "litert-lm", "mlx": "mlx-swift",
    "llama": "llama.cpp", "gguf": "llama.cpp", "q4_k_m": "llama.cpp",
    "core ai": "core-ai", "coreai": "core-ai", "cactus": "cactus",
}


def index_results(roots: list[Path]) -> list[Source]:
    out: list[Source] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.suffix not in (".json", ".jsonl") or not p.is_file():
                continue
            try:
                text = p.read_text()
            except Exception:
                continue
            for line in (text.splitlines() if p.suffix == ".jsonl" else [text]):
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                if not isinstance(o, dict):
                    continue
                m = o.get("metrics")
                if not isinstance(m, dict):
                    # quality reports: {"acc": .., "correct": ..}
                    tag = str(o.get("tag", "?")).lower()
                    rt = next((v for k, v in RUNTIME_HINTS.items() if k in tag), tag)
                    for k in ("acc", "correct", "n", "ok"):
                        if isinstance(o.get(k), (int, float)):
                            out.append(Source(float(o[k]), p.name, k, rt, "quality"))
                    continue
                rt = o.get("runtime", "?")
                task = o.get("task", "?")
                for field, v in numeric_items(m):
                    if v and field not in SKIP_KEYS:
                        out.append(Source(v, p.name, field, rt, task))
    return out


def candidates(value: float, unit: str | None) -> list[float]:
    """The same measurement may be printed in another unit or as a fraction."""
    c = [value]
    u = (unit or "").upper()
    if u == "GB":
        c.append(value * 1024)
    elif u == "MB":
        c.append(value / 1024)
    elif u == "%":
        c.append(value / 100)
    elif u == "S":
        c.append(value * 1000)
    elif u == "MS":
        c.append(value / 1000)
    return c


def trace(value: float, unit: str | None, srcs: list[Source], tol: float, context: str = ""):
    """Closest match within tolerance. A source whose runtime is named in the same line wins
    over an equally-close one that is not: a bare value match is easy to get by coincidence
    across four months of captures, and quoting the right number from the wrong arm is the
    error this whole exercise is about."""
    ctx = context.lower()
    wanted = {rt for k, rt in RUNTIME_HINTS.items() if k in ctx}
    best = None
    for cand in candidates(value, unit):
        for s in srcs:
            if s.value == 0:
                continue
            rel = abs(s.value - cand) / abs(s.value)
            if rel > tol:
                continue
            score = (0 if (wanted and s.runtime in wanted) else 1, rel)
            if best is None or score < best[0]:
                best = (score, s)
    if best is None:
        return None, False
    return best[1], bool(wanted) and best[0][0] == 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("docs", nargs="+", type=Path)
    ap.add_argument("--results", type=Path, action="append", default=None,
                    help="results root (repeatable); defaults to ./results")
    ap.add_argument("--tol", type=float, default=0.01, help="relative tolerance (default 1%%)")
    ap.add_argument("--quiet-traced", action="store_true", help="only list failures")
    args = ap.parse_args()

    roots = args.results or [Path("results")]
    srcs = index_results(roots)
    print(f"indexed {len(srcs):,} metric values from {', '.join(str(r) for r in roots)}\n")
    if not srcs:
        print("no result files found — check --results")
        return 1

    failures = 0
    for doc in args.docs:
        if not doc.exists():
            print(f"!! missing document: {doc}")
            failures += 1
            continue
        print(f"=== {doc}")
        for lineno, line in enumerate(doc.read_text().splitlines(), 1):
            if DERIVED.search(line) or EXTERNAL.search(line):
                continue
            line = CODE_SPAN.sub(lambda m: " " * len(m.group(0)), line)
            for m in NUM.finditer(line):
                raw, unit = m.group(1), m.group(2)
                value = float(raw.replace(",", ""))
                # Measurement-shaped: carries a unit, a decimal point, or thousands grouping.
                # A bare integer with no unit is a count, an n, or a year — not a measurement.
                # (An earlier cut also skipped unitless values below 10, which silently let
                # through exactly the row this script was written for: "MLX-PTQ 0.090".)
                shaped = bool(unit) or "." in raw or "," in raw
                if not shaped or value == 0:
                    continue
                if unit in ("x", "×") or (unit == "%" and "." not in raw):
                    continue          # multipliers and whole percents are derived, not stored
                hit, wrong_arm = trace(value, unit, srcs, args.tol, line)
                label = f"{raw}{' ' + unit if unit else ''}"
                if hit and not wrong_arm:
                    if not args.quiet_traced:
                        print(f"  ok    L{lineno:<4} {label:<14} -> {hit.field} "
                              f"({hit.runtime}/{hit.task}) {hit.file}")
                elif hit and wrong_arm:
                    failures += 1
                    print(f"  ARM?  L{lineno:<4} {label:<14} only matches {hit.runtime}/"
                          f"{hit.task} ({hit.file}) — the line names a different runtime")
                else:
                    failures += 1
                    ctx = line.strip()[:88]
                    print(f"  FAIL  L{lineno:<4} {label:<14} no stored capture within "
                          f"{args.tol:.0%}   | {ctx}")
        print()

    if failures:
        print(f"{failures} number(s) do not trace to a stored capture.")
        print("Either point them at a result file, re-measure them, or mark the line "
              "`<!-- derived: <inputs> -->` if the value is computed from other cells.")
        return 1
    print("every measurement-shaped number traces to a stored capture.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
