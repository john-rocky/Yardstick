#!/usr/bin/env python3
"""Summarise one Gemma-4 campaign's console logs into the table the doc needs.

    python3 scripts/summarize_gemma4_campaign.py results/raw/<campaign>

Reads the `console_*.txt` the driver tees, not the pulled JSON, so it works mid-run without
touching the device. Applies the same positional rule as `analyze_comparability.py`: run 1 is
cold, run 4+ is the thermally degraded tail, warm = the runs in between.

Written after an ad-hoc one-liner split `console_DEPTH_litert_1.txt` on the wrong underscore
and pooled two runtimes into a single row. The bogus figure (decode 46.55, footprint 484 MB —
neatly between the two arms) was obvious enough to catch, which is exactly why a less obvious
one would not have been. The arm name is parsed once, here.
"""

from __future__ import annotations

import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

FIELD = re.compile(r"(\w+)=(\S+)")


def parse_cells(root: Path):
    """-> {(phase, arm): {run_index: [row, ...]}}"""
    out = defaultdict(lambda: defaultdict(list))
    for p in sorted(root.glob("console_*.txt")):
        name = p.stem[len("console_"):]
        phase, rest = name.split("_", 1)
        if phase not in ("DEPTH", "CHAT"):
            continue                                  # WARMUP / PROBE / NATIVE are not data
        arm = rest.rsplit("_", 1)[0]                  # <arm>_<round>
        for line in p.read_text(errors="replace").replace("\r", "\n").split("\n"):
            if "YARDSTICK_RUN_OK" not in line:
                continue
            f = dict(FIELD.findall(line))
            try:
                out[(phase, arm)][int(f["run"])].append({
                    k: float(f[k]) for k in
                    ("decode_tok_s", "decode_tok_s_wall", "median_mb",
                     "median_resident_mb", "peak_mb", "tokens", "ctx")
                    if k in f
                })
            except (KeyError, ValueError):
                continue
    return out


def med(rows, key):
    v = [r[key] for r in rows if r.get(key)]
    return st.median(v) if v else None


def fmt(x, nd=1):
    return "—" if x is None else f"{x:,.{nd}f}"


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    cells = parse_cells(root)
    if not cells:
        print(f"no DEPTH/CHAT console logs under {root}")
        return 1

    for phase in ("DEPTH", "CHAT"):
        arms = sorted(a for (ph, a) in cells if ph == phase)
        if not arms:
            continue
        print(f"\n=== {phase} — warm = runs 2..3 (run 1 cold, run 4+ thermal tail) ===")
        print(f"{'arm':10s} {'n':>3s} {'decode eng':>11s} {'decode wall':>12s} "
              f"{'footprint':>10s} {'resident':>10s} {'ctx':>6s}   run-by-run (median)")
        for arm in arms:
            byrun = cells[(phase, arm)]
            warm = [r for i in (2, 3) for r in byrun.get(i, [])]
            allr = [r for i in sorted(byrun) for r in byrun[i]]
            if not warm:
                continue
            per_run = "  ".join(
                f"r{i}:{fmt(med(byrun[i], 'decode_tok_s'))}" + ("*" if i == 1 or i >= 4 else "")
                for i in sorted(byrun))
            print(f"{arm:10s} {len(warm):3d} {fmt(med(warm,'decode_tok_s')):>11s} "
                  f"{fmt(med(warm,'decode_tok_s_wall')):>12s} "
                  f"{fmt(med(allr,'median_mb'),0)+' MB':>10s} "
                  f"{fmt(med(allr,'median_resident_mb'),0)+' MB':>10s} "
                  f"{fmt(med(allr,'ctx'),0):>6s}   {per_run}")
        # A truncated run is a failure, not a fast cell — surface it next to the medians.
        for arm in arms:
            for i, rows in sorted(cells[(phase, arm)].items()):
                for r in rows:
                    if r.get("tokens", 0) and r["tokens"] < 32:
                        print(f"  !! {arm} run{i}: only {r['tokens']:.0f} tokens — degenerate, "
                              f"not a decode rate")
    print("\n* = excluded from the warm median by position, identically for every arm.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
