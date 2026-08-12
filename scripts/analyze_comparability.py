#!/usr/bin/env python3
"""Read a pulled results directory and say what the numbers are allowed to claim.

    python3 scripts/analyze_comparability.py /tmp/gemma4-definitive [--all-thermal]

Written for the 2026-07-26 re-capture. Two things go wrong in a cross-runtime table and this
prints both rather than a single tidy number:

  comparability  the same column measured different ways per arm — engine-reported vs
                 harness wall-clock for speed, footprint vs resident for memory.
  reproducibility  a cell whose scatter across identical runs is wider than the gap being
                 claimed. Every row carries n, relative MAD and the full range; a cell whose
                 MAD exceeds 5% is flagged and may not be published as a ranking.

Speed rows use only cells whose `initialThermalState` is `nominal` (fairness rule 2); memory
rows use every cell, since footprint is not thermally sensitive. `--all-thermal` disables the
filter and says so.
"""

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

# Relative MAD (median absolute deviation / median), in %. Above this a cell is too noisy
# to rank on. MAD is used rather than (max-min) because the range grows with n even for an
# identical population — simulated on a 5%-CV population, mean range is 8.5% at n=3, 13.5%
# at n=7 and 15.4% at n=10, so a range-based threshold would penalise exactly the extra
# repetitions this protocol prescribes. Relative MAD over the same population sits at
# 2.2 / 3.0 / 3.1%.
MAD_LIMIT = 5.0


def load(root: Path, since: str | None = None):
    """`since` is an ISO timestamp; the device keeps every capture ever taken, and a pull
    copies all of them, so a session that does not filter silently pools cells from earlier
    app builds and earlier protocols into one median."""
    out = []
    for p in sorted(root.rglob("*.json")):
        try:
            o = json.loads(p.read_text())
        except Exception:
            continue
        if not (isinstance(o, dict) and "metrics" in o):
            continue
        if since and (o.get("timestamp") or "") < since:
            continue
        out.append(o)
    return out


def stat(values):
    """median, n, relative MAD %, full range % — MAD decides, range is shown for context."""
    v = [x for x in values if x not in (None, 0)]
    if not v:
        return None, 0, None, None
    m = st.median(v)
    if not m:
        return m, len(v), None, None
    mad = st.median([abs(x - m) for x in v]) / m * 100
    return m, len(v), mad, (max(v) - min(v)) / m * 100


def cell(v, nd=1):
    m, n, mad, rng = v
    if m is None:
        return "—".rjust(24)
    if mad is None:
        return f"{m:>10,.{nd}f} n={n}".ljust(24)
    # MAD needs a handful of points to mean anything: at n=3 a single outlier leaves the
    # median absolute deviation near zero while the range is wide (measured 2026-07-26:
    # LiteRT decode [45.5, 33.8, 32.8] -> MAD 3%, range 37%). Below n=5, fall back to the
    # range so a small sample cannot look clean by construction.
    noisy = mad > MAD_LIMIT or (n < 5 and (rng or 0) > 20)
    flag = "!" if noisy else " "
    return f"{m:>10,.{nd}f} n={n}{flag}mad{mad:>4.0f}% r{rng:>4.0f}%"


def main(root: Path, all_thermal: bool, since: str | None):
    rows = load(root, since)
    if not rows:
        print(f"no result JSON under {root}")
        return

    hot = sum(1 for o in rows if o["metrics"].get("initialThermalState") != "nominal")
    print(f"{len(rows)} captures under {root}   ({hot} started above nominal)"
          + (f"   [only captures at/after {since}]" if since else "   [ALL captures on the device — "
             "pass --since to exclude earlier sessions]"))
    print(f"speed rows use nominal-only cells{' — DISABLED (--all-thermal)' if all_thermal else ''};"
          f" memory rows use all cells.  '!' = MAD > {MAD_LIMIT:.0f}%, do not rank on it.\n")

    speed = defaultdict(list)
    mem = defaultdict(list)
    for o in rows:
        m = o["metrics"]
        key = (o.get("runtime", "?"), o.get("task", "?"))
        mem[key].append(m)
        if all_thermal or m.get("initialThermalState") == "nominal":
            speed[key].append(m)

    print("=" * 104)
    print("SPEED — engine-reported vs harness wall-clock (the two are not the same column)")
    print("=" * 104)
    print(f"{'runtime':11s} {'task':26s} {'decode eng':>19s} {'decode wall':>19s} {'gen tok':>10s}")
    for k, ms in sorted(speed.items()):
        e = stat([m.get("decodeTokensPerSecond") for m in ms])
        w = stat([m.get("decodeTokensPerSecondWallClock") for m in ms])
        g = stat([m.get("generatedTokenCount") for m in ms])
        note = ""
        if g[0] is not None and g[0] < 64:
            note = f"  <- only {g[0]:.0f} tokens; not a decode rate"
        print(f"{k[0]:11s} {k[1]:24s} {cell(e)} {cell(w)} {('' if g[0] is None else f'{g[0]:.0f}'):>8s}{note}")
    print("\nRule: rank on the wall-clock column, or print both labelled. Never mix.\n")

    print("=" * 104)
    print("PREFILL — same task, same prompt, same instrument for every arm")
    print("=" * 104)
    print(f"{'runtime':11s} {'task':26s} {'prompt tok':>11s} {'prefill eng':>19s} {'prefill wall':>19s}")
    for k, ms in sorted(speed.items()):
        if not k[1].startswith("long-context"):
            continue
        p = stat([m.get("promptTokenCount") for m in ms])
        e = stat([m.get("promptTokensPerSecond") for m in ms])
        w = stat([m.get("promptTokensPerSecondWallClock") for m in ms])
        print(f"{k[0]:11s} {k[1]:26s} {('' if p[0] is None else f'{p[0]:,.0f}'):>11s} {cell(e, 0)} {cell(w, 0)}")
    print("\nRule: this replaces any cell measured through a runtime's own benchmark() entry\n"
          "point. A forced-1024 vendor-card method and this task are different measurements.\n")

    print("=" * 104)
    print("MEMORY — footprint vs resident, on medians (peaks are page-cache noise)")
    print("=" * 104)
    print(f"{'runtime':11s} {'task':26s} {'footprint med':>19s} {'resident med':>19s} {'mapped':>10s}")
    for k, ms in sorted(mem.items()):
        f = stat([m.get("memoryMedianMB") or m.get("memoryPeakDuringDecodeMB") for m in ms])
        r = stat([m.get("memoryMedianResidentMB") for m in ms])
        mapped = f"{r[0] - f[0]:+,.0f}" if (f[0] and r[0]) else "—"
        print(f"{k[0]:11s} {k[1]:26s} {cell(f, 0)} {cell(r, 0)} {mapped:>10s}")
    print("\nRule: rank on one column and name it. A positive 'mapped' means the runtime maps\n"
          "weights the footprint does not charge (LiteRT-LM maps 1.12 GB of embeddings per its\n"
          "card; llama.cpp maps the GGUF). A negative one means compressed pages the resident\n"
          "size does not see. Neither column ranks both kinds of runtime fairly on its own.\n")

    lit = [m for (rt, task), ms in mem.items() if rt == "litert-lm" for m in ms]
    if lit:
        print("=" * 104)
        print("CARD RECONCILIATION — LiteRT-LM footprint vs context budget")
        print("=" * 104)
        for (rt, task), ms in sorted(mem.items()):
            if rt != "litert-lm":
                continue
            f = stat([m.get("memoryMedianMB") or m.get("memoryPeakDuringDecodeMB") for m in ms])
            p = stat([m.get("promptTokenCount") for m in ms])
            print(f"  {task:26s} prompt={('' if p[0] is None else f'{p[0]:,.0f}'):>6s} tok   footprint={cell(f, 0)}")
        print("  card (iPhone 17 Pro, GPU, 2048 ctx, phys_footprint)                1,450 MB")
        print("\nRule: our footprint has to track the KV/context budget toward the card's figure.\n"
              "If it does not, the difference is unexplained and no memory claim may be built on\n"
              "our number.\n")


if __name__ == "__main__":
    argv = sys.argv[1:]
    args = [a for a in argv if not a.startswith("--")]
    since = next((a.split("=", 1)[1] for a in argv if a.startswith("--since=")), None)
    main(Path(args[0] if args else "/tmp/gemma4-definitive"), "--all-thermal" in argv, since)
