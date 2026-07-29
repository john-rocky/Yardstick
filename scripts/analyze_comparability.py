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

Speed rows use cells that STARTED nominal (fairness rule 2), minus run 1 and the run-4+ tail of
each launch — a positional rule applied identically to every arm (see `positional_verdict`).
Memory rows use every cell, since footprint is not thermally sensitive. `--all-thermal` disables
both filters and says so.

    --since=<iso>      drop captures from earlier sessions (a pull copies every capture the
                       device has ever taken)
    --harness=<stamp>  keep one measurement contract; a timestamp cutoff does not separate
                       warm-up cells captured minutes into the same session
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


def load(root: Path, since: str | None = None, harness: str | None = None):
    """`since` is an ISO timestamp; the device keeps every capture ever taken, and a pull
    copies all of them, so a session that does not filter silently pools cells from earlier
    app builds and earlier protocols into one median.

    `harness` filters on `metrics.harnessStamp`. A timestamp cutoff is not always enough:
    warm-up cells captured minutes into the same session can carry an older contract than the
    measured cells that follow them (2026-07-27: the `-r1` warm-up cells sat inside the
    `--since` window and dragged the LiteRT short-chat wall-clock median to 55% MAD, because
    `-r1` measured the decode window through the LiteRT stream drain). Filter on the stamp
    when the mixed-harness banner fires."""
    out = []
    # Both layouts: the iOS pull is one *.json per run; the Mac CLI writes one *.jsonl per
    # cell with a row per run. The analyzer read only the former until 2026-07-28, which is
    # one of the reasons no Mac capture was ever comparability-checked.
    def rows_of(p: Path):
        text = p.read_text()
        if p.suffix == ".jsonl":
            return [line for line in text.splitlines() if line.strip()]
        return [text]

    for p in sorted(list(root.rglob("*.json")) + list(root.rglob("*.jsonl"))):
        try:
            candidates = rows_of(p)
        except Exception:
            continue
        for line in candidates:
            try:
                o = json.loads(line)
            except Exception:
                continue
            if not (isinstance(o, dict) and "metrics" in o):
                continue
            if since and (o.get("timestamp") or "") < since:
                continue
            if harness and o["metrics"].get("harnessStamp") != harness:
                continue
            out.append(o)
    return out


def annotate_arms(rows):
    """Set `_arm` on every row: the runtime name, plus a model disambiguator whenever one
    runtime carries more than one model in the set. Keying cells on (runtime, task) alone
    pooled MLX-PTQ and MLX-OptiQ into one `mlx-swift` cell (caught 2026-07-28: the pooled
    memory row read 18-22% MAD because it mixed a ~3 GB and a ~4.5 GB model, and OptiQ's
    cells were misread as the thermal tail of MLX launches)."""
    import os
    tails = defaultdict(set)
    for o in rows:
        mid = ((o.get("model") or {}).get("id")) or "?"
        o["_mtail"] = mid.rsplit("/", 1)[-1]
        tails[o.get("runtime", "?")].add(o["_mtail"])
    for o in rows:
        rt = o.get("runtime", "?")
        ts = tails[rt]
        if len(ts) > 1:
            pre = os.path.commonprefix(sorted(ts))
            o["_arm"] = f"{rt}·{o['_mtail'][len(pre):] or o['_mtail']}"
        else:
            o["_arm"] = rt


def arm_of(o):
    return o.get("_arm") or o.get("runtime", "?")


# Runs inside one `--runs N` launch land seconds apart; launches are separated by the driver's
# cooldown (180 s+). Anything above this gap starts a new launch. The app does not record a run
# index, so it is reconstructed from the timestamps.
LAUNCH_GAP_SECONDS = 90


def group_launches(rows):
    """Annotate every row with `_run_index` (1-based, within its launch) and `_launch_size`.

    Needed because the warm convention and the thermal-degradation rule are both POSITIONAL,
    and a positional rule cannot be applied to an unordered bag of cells."""
    from datetime import datetime

    def ts(o):
        t = (o.get("timestamp") or "").replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(t)
        except ValueError:
            return None

    by_cell = defaultdict(list)
    for o in rows:
        by_cell[(arm_of(o), o.get("task"))].append(o)

    for cell_rows in by_cell.values():
        dated = [o for o in cell_rows if ts(o)]
        dated.sort(key=ts)
        launch = []
        launches = []
        for o in dated:
            if launch and (ts(o) - ts(launch[-1])).total_seconds() > LAUNCH_GAP_SECONDS:
                launches.append(launch)
                launch = []
            launch.append(o)
        if launch:
            launches.append(launch)
        for lch in launches:
            for i, o in enumerate(lch, 1):
                o["_run_index"] = i
                o["_launch_size"] = len(lch)
        for o in cell_rows:
            o.setdefault("_run_index", None)
            o.setdefault("_launch_size", None)
    return rows


def positional_verdict(o):
    """Why a run is or is not part of the warm speed median. Positional, never value-based.

    run 1  — cold: the model was just loaded. Fairness rule 2's warm convention.
    run 4+ — thermally degraded: measured 2026-07-27, the 4th back-to-back run of a launch
             dropped 12.9% and 23.3% (LiteRT) and 11.0% (llama.cpp) against run 2 of the same
             launch. Excluding it on the THERMAL FLAG instead would be asymmetric — LiteRT
             crossed into `fair` and llama.cpp did not, so the flag removes the fast arm's bad
             run and keeps the slow arm's, flattering the fast arm by ~1.8%. A positional rule
             cannot do that: it never looks at the number.
    """
    i, n = o.get("_run_index"), o.get("_launch_size")
    if i is None:
        return None, "no timestamp — cannot place in a launch"
    if i == 1:
        return False, "run 1 (cold)"
    if n and n >= 4 and i >= 4:
        return False, f"run {i} of {n} (thermally degraded tail)"
    return True, f"run {i} of {n}"


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


def main(root: Path, all_thermal: bool, since: str | None, harness: str | None = None):
    rows = load(root, since, harness)
    if not rows:
        print(f"no result JSON under {root}")
        return
    annotate_arms(rows)
    group_launches(rows)

    # "not nominal" and "never recorded" are different claims, and only one of them is a
    # statement about the device. The native-benchmark path starts no ThermalSampler, so its
    # rows carry a null thermal state; reporting those as "started above nominal" asserts a
    # regime nobody measured. Both are excluded from the speed table either way — but the
    # header has to say which reason applies.
    hot = sum(1 for o in rows
              if (t := o["metrics"].get("initialThermalState")) is not None and t != "nominal")
    unknown = sum(1 for o in rows if o["metrics"].get("initialThermalState") is None)
    throttled_mid_run = sum(
        1 for o in rows
        if o["metrics"].get("initialThermalState") == "nominal"
        and o["metrics"].get("peakThermalState") not in ("nominal", None))
    thermal_note = f"{hot} started above nominal"
    if throttled_mid_run:
        thermal_note += f", {throttled_mid_run} throttled mid-run"
    if unknown:
        thermal_note += f", {unknown} recorded no thermal state"
    print(f"{len(rows)} captures under {root}   ({thermal_note})"
          + (f"   [only captures at/after {since}]" if since else "   [ALL captures on the device — "
             "pass --since to exclude earlier sessions]"))
    print(f"speed rows use nominal-START cells, minus run 1 and the run-4+ tail by POSITION"
          f"{' — ALL FILTERS DISABLED (--all-thermal)' if all_thermal else ''};"
          f" memory rows use all cells.  '!' = MAD > {MAD_LIMIT:.0f}%, do not rank on it.")
    if throttled_mid_run:
        # Reported, not acted on. The exclusion is positional; if a throttled cell is NOT in
        # an excluded position that is an anomaly worth seeing, not something to drop quietly.
        print(f"  note: {throttled_mid_run} cell(s) started nominal and throttled mid-run."
              " They are handled by the positional rule below, not by their thermal flag —"
              " flag-based exclusion favours whichever arm happens to cross `fair` first."
              " The underlying cause is that a `--runs N` launch runs back-to-back, while the"
              " agreed protocol asks for a few minutes BETWEEN RUNS, not only between launches.")

    # More than one measurement contract in the set means the medians below pool cells that
    # do not measure the same thing. This is the check the 92 MB deep-context cell needed and
    # did not have: nothing on it said which harness produced it.
    stamps = defaultdict(int)
    for o in rows:
        stamps[o["metrics"].get("harnessStamp") or "pre-stamp (unknown harness)"] += 1
    if len(stamps) > 1:
        print("\n!! MIXED HARNESSES — these cells were not produced by the same measurement contract:")
        for s, c in sorted(stamps.items(), key=lambda kv: -kv[1]):
            print(f"     {c:4d}  {s}")
        print("   Narrow with --since before ranking anything below.")
    else:
        print(f"harness: {next(iter(stamps))}")
    print()

    speed = defaultdict(list)
    mem = defaultdict(list)
    degenerate = []
    for o in rows:
        m = o["metrics"]
        key = (arm_of(o), o.get("task", "?"))
        mem[key].append(m)
        # Gate on the PEAK, not just the initial, state. Gating on `initialThermalState` alone
        # passes a run that began nominal and throttled partway through — which is exactly the
        # shape a back-to-back `--runs N` block produces: measured 2026-07-27, run 4 of a
        # 4-run LiteRT launch started nominal, peaked `fair`, and read 43.7 tok/s against
        # 56.9 for run 2 of the same launch. Its decode rate is real, but it is a throttled
        # rate and pooling it into a burst median silently mixes two regimes.
        # A run that stopped far short of its token budget did not measure a decode rate; it
        # measured a failure. Measured 2026-07-27: a Core AI probe with a broken chat template
        # emitted 3 of 128 requested tokens and still reported `decode_tok_s=75.41` — the
        # fastest number in the whole session, produced by a run that said almost nothing.
        # Pooling that into a median would reward degeneracy. `parameters.maxTokens` is the
        # budget the task asked for; anything under a quarter of it is not a sample.
        budget = ((o.get("parameters") or {}).get("maxTokens")) or 0
        produced = m.get("generatedTokenCount") or 0
        if budget and produced < budget // 4:
            degenerate.append((arm_of(o), o.get("task"), produced, budget))
            if not all_thermal:
                continue

        started_clean = m.get("initialThermalState") == "nominal"
        if not started_clean and not all_thermal:
            continue
        # Positional, not value-based — see `positional_verdict`. Excluding on the mid-run
        # thermal FLAG was tried first and is asymmetric across arms; this is the fix.
        keep, _why = positional_verdict(o)
        if keep is False and not all_thermal:
            continue
        speed[key].append(m)

    # Show the effect the positional rule is correcting for, rather than applying it silently.
    # A reader has to be able to check that run 4 really is degraded and that the rule is not
    # just discarding inconvenient numbers.
    pos = defaultdict(lambda: defaultdict(list))
    for o in rows:
        i = o.get("_run_index")
        d = o["metrics"].get("decodeTokensPerSecond")
        if i and d:
            pos[(arm_of(o), o.get("task"))][i].append(d)
    if any(len(v) > 1 for v in pos.values()):
        print("=" * 104)
        print("RUN POSITION WITHIN LAUNCH — why the warm median drops run 1 and the run-4 tail")
        print("=" * 104)
        print(f"{'runtime':11s} {'task':26s} " + "".join(f"{'run'+str(i):>12s}" for i in range(1, 6))
              + "   used")
        for k in sorted(pos):
            cells = []
            for i in range(1, 6):
                v = pos[k].get(i)
                cells.append(f"{st.median(v):>11,.1f}" + ("*" if i == 1 or i >= 4 else " ") if v else " " * 12)
            used = ",".join(str(i) for i in sorted(pos[k]) if not (i == 1 or i >= 4))
            print(f"{k[0]:11s} {k[1]:26s} " + "".join(cells) + f"   {used or '—'}")
        print("\n* = excluded from the speed median by position, for every arm alike:\n"
              "  run 1 is cold; run 4+ is the thermally degraded tail of a back-to-back launch.\n"
              "  Excluding the tail on its thermal FLAG instead is asymmetric — the faster arm\n"
              "  crosses into `fair` and loses its bad run while the slower arm keeps its own.\n")

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

    lit = [m for (rt, task), ms in mem.items() if rt.startswith("litert-lm") for m in ms]
    if lit:
        print("=" * 104)
        print("CARD RECONCILIATION — LiteRT-LM footprint vs context budget")
        print("=" * 104)
        for (rt, task), ms in sorted(mem.items()):
            if not rt.startswith("litert-lm"):
                continue
            f = stat([m.get("memoryMedianMB") or m.get("memoryPeakDuringDecodeMB") for m in ms])
            p = stat([m.get("promptTokenCount") for m in ms])
            # The context budget is the axis the card's figure is quoted at, and it is now
            # recorded per run rather than inferred from the prompt length — a cell with no
            # `contextTokensConfigured` predates the 2026-07-27 harness and is off-protocol
            # by construction, so say so instead of silently comparing it with the card.
            ctx = stat([m.get("contextTokensConfigured") for m in ms])
            ctx_s = "pre-2026-07-27 (unrecorded)" if ctx[0] is None else f"{ctx[0]:,.0f}"
            print(f"  {task:26s} ctx={ctx_s:>27s}  prompt={('' if p[0] is None else f'{p[0]:,.0f}'):>6s} tok   footprint={cell(f, 0)}")
        print("  card (iPhone 17 Pro, GPU, 2048 ctx, phys_footprint)                1,450 MB"
              "   <-- STALE, confirmed 2026-07-27: not a target, do not read a gap from it")
        print("\nRule: a forced context is a CEILING, not an occupancy. LiteRT-LM grows into its KV\n"
              "rather than pre-allocating maxNumTokens — measured 2026-07-27, three cells all at\n"
              "ctx 2048 read 497 / 663 / 732 MB, tracking the tokens actually used (21 / 1,024 /\n"
              "1,098). So a memory cell must state the PROMPT LENGTH it was measured at; 'at a 2048\n"
              "context' does not pin a footprint. The card row above is stale and is printed for\n"
              "reference only — no claim may be built on the distance from it.\n")


if __name__ == "__main__":
    argv = sys.argv[1:]
    args = [a for a in argv if not a.startswith("--")]
    since = next((a.split("=", 1)[1] for a in argv if a.startswith("--since=")), None)
    harness = next((a.split("=", 1)[1] for a in argv if a.startswith("--harness=")), None)
    main(Path(args[0] if args else "/tmp/gemma4-definitive"), "--all-thermal" in argv, since, harness)
