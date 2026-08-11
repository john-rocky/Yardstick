#!/usr/bin/env python3
"""Release-regression differ over the accumulation layer (continuous-bench condition 3).

The v0.13.1->v0.15.0 re-measure was done by hand; this is the reusable half of it:
given two capture sets, join comparable cells and report deltas WITH the repo's
fairness rules applied as code, not discipline:

  rule 3   a budget/mode mismatch (max_tokens, thinking) is NOT a comparison —
           such pairs are marked NOT-COMPARABLE, never scored.
  rule 4   per-side trial spread is quoted; spread beyond --spread-limit makes the
           cell UNRELIABLE (throw the number out, don't average it away).
  session  device cells captured on different days are INFO-ONLY: same binary,
           same pins measured 126-133 tok/s in June and 159-180 in July
           (methodology: iphone-session-variance). Only same-session pairs
           (e.g. the resident A/B design) earn a REGRESSION/OK verdict.

Quality (GSM8K) pairs join on tag; device cells join on
(device, runtime, model_id, task, cold/warm) split by the requested selectors —
quantization is compared as a label, not a key (it has been corrected in place for
the same artifact), and a 0 tok/s field is treated as an unmeasured axis.

Usage:
  # fresh regression captures vs the published rows with the same tags
  python3 scripts/regression_diff.py quality --candidate-dir results/quality/regression/<dir>

  # explicit pair of published tags
  python3 scripts/regression_diff.py quality \
      --baseline-tag litertlm-gemma4-e2b-wna8o8-measured \
      --candidate-tag litert-gemma4-e2b-v0150-off

  # device cells between two campaigns (or engine versions once stamped rows exist)
  python3 scripts/regression_diff.py device \
      --baseline campaign:2026-07-30-gemma4-e2b-protocol \
      --candidate campaign:2026-08-04-litert-0150-iphone

Exit code 1 iff any cell's verdict is REGRESSION (loop/CI-usable). Reads
results/summary/*.csv, regenerating them first via build_summary.py (--no-rebuild
to skip). ./reproduce <platform> <table> --regress drives the capture+diff loop.
"""
import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(ROOT, "results", "summary")


def rebuild_summary():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import build_summary
    build_summary.main()


def load_csv(name):
    with open(os.path.join(SUMMARY, name)) as fh:
        return list(csv.DictReader(fh))


# ---------------- quality ----------------

def quality_row_by_tag(rows, tag, exclude_regression):
    hits = [r for r in rows if r["tag"] == tag]
    if exclude_regression:
        hits = [r for r in hits if "/regression/" not in r["source"]]
    return hits


def compare_quality_pair(base, cand, threshold_pts):
    """One verdict line for a (baseline row, candidate row) pair from quality.csv."""
    problems, cautions = [], []
    if base["max_tokens"] != cand["max_tokens"]:
        problems.append(f"max_tokens {base['max_tokens']} vs {cand['max_tokens']} (rule 3)")
    # thinking blocks only when BOTH sides recorded it and they differ; pre-v1 reports
    # never recorded it (the tag suffix carried the mode), so absence is noted, not fatal
    bt, ct = base.get("thinking") or "", cand.get("thinking") or ""
    if bt and ct and bt != ct:
        problems.append(f"thinking {bt} vs {ct} (rule 3)")
    elif bt != ct:
        cautions.append("thinking unrecorded on one side (pre-v1 report)")
    if problems:
        return "NOT-COMPARABLE", "; ".join(problems)
    b_acc, c_acc = float(base["acc"]), float(cand["acc"])
    b_n, c_n = int(base["n"]), int(cand["n"])
    delta = (c_acc - b_acc) * 100
    # binomial sd of the baseline accuracy at the candidate's n — the yardstick for
    # whether a delta is noise (n=100 at 90% => ~3 points)
    sd = 100 * math.sqrt(max(b_acc * (1 - b_acc), 1e-9) / max(c_n, 1))
    note = f"{b_acc*100:.1f} -> {c_acc*100:.1f} ({delta:+.1f} pts, ~1sd={sd:.1f})"
    if cautions:
        note += "  [" + "; ".join(cautions) + "]"
    if b_n != c_n:
        note += f"  [n {b_n} vs {c_n} — protocol pins n; verdict withheld]"
        return "NOT-COMPARABLE", note
    if delta < -threshold_pts:
        return "REGRESSION", note
    if delta > threshold_pts:
        return "IMPROVED", note
    return "OK", note


def run_quality(args):
    rows = load_csv("quality.csv")
    pairs = []
    if args.candidate_dir:
        cdir = os.path.relpath(os.path.abspath(args.candidate_dir), ROOT)
        cand_rows = [r for r in rows if r["source"].startswith(cdir + os.sep)
                     or os.path.dirname(r["source"]) == cdir]
        if not cand_rows:
            # candidate dir may hold reports newer than the last summary build —
            # read them directly (they are schema-v1, written by parity_gsm8k.py)
            for f in sorted(glob.glob(os.path.join(args.candidate_dir, "gsm8k_*.json"))):
                d = json.load(open(f))
                cand_rows.append({
                    "source": os.path.relpath(f, ROOT), "tag": d.get("tag"),
                    "n": str(d.get("n")), "correct": str(d.get("correct")),
                    "acc": str(d.get("acc")), "max_tokens": str(d.get("max_tokens")),
                    "thinking": str(d.get("conditions", {}).get("thinking", "")),
                    "engine_version": d.get("engineVersion") or "",
                })
        if not cand_rows:
            print(f"no candidate reports under {args.candidate_dir}", file=sys.stderr)
            return 2
        for c in cand_rows:
            bases = quality_row_by_tag(rows, c["tag"], exclude_regression=True)
            bases = [b for b in bases if b["source"] != c["source"]]
            pairs.append((bases[-1] if bases else None, c))
    else:
        if not (args.baseline_tag and args.candidate_tag):
            print("quality mode needs --candidate-dir or --baseline-tag/--candidate-tag",
                  file=sys.stderr)
            return 2
        b = quality_row_by_tag(rows, args.baseline_tag, exclude_regression=False)
        c = quality_row_by_tag(rows, args.candidate_tag, exclude_regression=False)
        if not b or not c:
            print(f"tag not found: {args.baseline_tag if not b else args.candidate_tag}",
                  file=sys.stderr)
            return 2
        pairs.append((b[-1], c[-1]))

    worst = 0
    print("\n== quality (GSM8K) ==")
    for base, cand in pairs:
        if base is None:
            print(f"NO-BASELINE      {cand['tag']}  (no published row with this tag)")
            continue
        verdict, note = compare_quality_pair(base, cand, args.threshold_points)
        eng = ""
        if cand.get("engine_version") or base.get("engine_version"):
            eng = f"  [{base.get('engine_version') or 'pre-stamp'} -> " \
                  f"{cand.get('engine_version') or 'pre-stamp'}]"
        print(f"{verdict:16} {cand['tag']}  {note}{eng}")
        if verdict == "REGRESSION":
            worst = 1
    return worst


# ---------------- device ----------------

def select(rows, sel):
    kind, _, val = sel.partition(":")
    if kind == "campaign":
        return [r for r in rows if val in r["campaign"]]
    if kind == "engine":
        return [r for r in rows if (r["engine_version"] or "").startswith(val)]
    raise SystemExit(f"bad selector {sel!r} (want campaign:<substr> or engine:<prefix>)")


# quantization is deliberately NOT in the join key: it is a prose label and this repo
# has corrected it in place ("INT4 (QAT)" -> "wNa8o8 (...)" for the SAME artifact, rule 2
# audit); model_id carries the recipe when it genuinely differs. A label mismatch is
# surfaced on the output line instead. cold_run IS in the key: cold and warm are
# different published cells, and pooling them inflates spread past rule 4.
GROUP = ("device", "runtime", "model_id", "task", "cold_run")


def cells(rows, metric):
    out = {}
    for r in rows:
        v = r.get(metric)
        if not v or float(v) == 0.0:  # a 0 tok/s field is an unmeasured axis, not a datum
            continue
        out.setdefault(tuple(r[k] for k in GROUP), []).append(
            (float(v), (r["timestamp"] or "")[:10]))
    return out


def quants(rows):
    out = {}
    for r in rows:
        out.setdefault(tuple(r[k] for k in GROUP), set()).add(r["quantization"])
    return out


def fmt_key(key):
    parts = [k for k in key[:-1] if k]
    if key[-1] == "True":
        parts.append("cold")
    elif key[-1] == "False":
        parts.append("warm")
    return " ".join(parts)


def run_device(args):
    rows = load_csv("device-runs.csv")
    base_rows, cand_rows = select(rows, args.baseline), select(rows, args.candidate)
    if not base_rows or not cand_rows:
        print(f"selector matched no rows: "
              f"{args.baseline if not base_rows else args.candidate}", file=sys.stderr)
        return 2
    worst = 0
    b_quants, c_quants = quants(base_rows), quants(cand_rows)
    any_common = False
    for metric, higher_is_better in (("decode_tps", True), ("prefill_tps", True),
                                     ("energy_j_per_tok", False)):
        b_cells, c_cells = cells(base_rows, metric), cells(cand_rows, metric)
        common = sorted(set(b_cells) & set(c_cells))
        if not common:
            continue
        any_common = True
        print(f"\n== device / {metric} ==")
        for key in common:
            bv = [v for v, _ in b_cells[key]]
            cv = [v for v, _ in c_cells[key]]
            bm, cm = statistics.median(bv), statistics.median(cv)
            bs = (max(bv) - min(bv)) / bm * 100 if bm and len(bv) > 1 else 0.0
            cs = (max(cv) - min(cv)) / cm * 100 if cm and len(cv) > 1 else 0.0
            delta = (cm - bm) / bm * 100 if bm else 0.0
            label = fmt_key(key)
            note = (f"{bm:.1f} (n={len(bv)}, spread {bs:.0f}%) -> "
                    f"{cm:.1f} (n={len(cv)}, spread {cs:.0f}%)  {delta:+.1f}%")
            if b_quants.get(key, set()) != c_quants.get(key, set()):
                note += (f"  [quant label: {'/'.join(sorted(b_quants.get(key, set())))} vs "
                         f"{'/'.join(sorted(c_quants.get(key, set())))}]")
            b_dates = {d for _, d in b_cells[key] if d}
            c_dates = {d for _, d in c_cells[key] if d}
            if bs > args.spread_limit or cs > args.spread_limit:
                # rule 4: contention halves decode and the only tell is spread
                print(f"UNRELIABLE       {label}  {note}  [spread > {args.spread_limit:.0f}% — throw out]")
                continue
            if b_dates and c_dates and b_dates.isdisjoint(c_dates):
                # different sittings: device-state drift dwarfs engine deltas
                print(f"INFO-ONLY        {label}  {note}  [cross-session — do not pool; "
                      f"use a same-session A/B for a verdict]")
                continue
            worse = delta < -args.threshold_pct if higher_is_better else delta > args.threshold_pct
            better = delta > args.threshold_pct if higher_is_better else delta < -args.threshold_pct
            verdict = "REGRESSION" if worse else ("IMPROVED" if better else "OK")
            print(f"{verdict:16} {label}  {note}")
            if worse:
                worst = 1
    if not any_common:
        # show why nothing joined (e.g. a task rename) instead of exiting silently
        print("no common cells between the two selections; distinct cell keys:")
        for side, sel_rows in (("baseline", base_rows), ("candidate", cand_rows)):
            for key in sorted(set(tuple(r[k] for k in GROUP) for r in sel_rows)):
                print(f"  {side}: {fmt_key(key)}")
        return 2
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["quality", "device"])
    ap.add_argument("--candidate-dir", help="quality: dir of fresh gsm8k_*.json reports")
    ap.add_argument("--baseline-tag")
    ap.add_argument("--candidate-tag")
    ap.add_argument("--baseline", help="device: campaign:<substr> or engine:<prefix>")
    ap.add_argument("--candidate", help="device: campaign:<substr> or engine:<prefix>")
    ap.add_argument("--threshold-points", type=float, default=3.0,
                    help="quality: accuracy delta (points) treated as real (default 3 ~ 1sd at n=100)")
    ap.add_argument("--threshold-pct", type=float, default=5.0,
                    help="device: median delta (%%) treated as real")
    ap.add_argument("--spread-limit", type=float, default=5.0,
                    help="device: per-side trial spread (%%) beyond which a cell is UNRELIABLE (rule 4)")
    ap.add_argument("--no-rebuild", action="store_true",
                    help="skip regenerating results/summary/ first")
    args = ap.parse_args()
    if not args.no_rebuild:
        rebuild_summary()
    if args.mode == "quality":
        return run_quality(args)
    if not (args.baseline and args.candidate):
        print("device mode needs --baseline and --candidate selectors", file=sys.stderr)
        return 2
    return run_device(args)


if __name__ == "__main__":
    sys.exit(main())
