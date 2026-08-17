#!/usr/bin/env python3
"""Release-regression differ over the accumulation layer (continuous-bench condition 3).

The v0.13.1->v0.15.0 re-measure was done by hand; this is the reusable half of it:
given two capture sets, join comparable cells and report deltas WITH the repo's
fairness rules applied as code, not discipline:

  budget-mode-rule   a budget/mode mismatch (max_tokens, thinking) is NOT a comparison —
           such pairs are marked NOT-COMPARABLE, never scored.
  spread-rule        per-side trial spread is quoted; spread beyond --spread-limit makes the
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
        problems.append(f"max_tokens {base['max_tokens']} vs {cand['max_tokens']} (budget-mode-rule)")
    # thinking blocks only when BOTH sides recorded it and they differ; pre-v1 reports
    # never recorded it (the tag suffix carried the mode), so absence is noted, not fatal
    bt, ct = base.get("thinking") or "", cand.get("thinking") or ""
    if bt and ct and bt != ct:
        problems.append(f"thinking {bt} vs {ct} (budget-mode-rule)")
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


RECORDS = []  # machine verdicts, written by --json-out


def record(mode, verdict, label, note, **extra):
    RECORDS.append({"mode": mode, "verdict": verdict, "cell": label,
                    "note": note, **extra})


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
            record("quality", "NO-BASELINE", cand["tag"], "no published row with this tag")
            continue
        verdict, note = compare_quality_pair(base, cand, args.threshold_points)
        eng = ""
        if cand.get("engine_version") or base.get("engine_version"):
            eng = f"  [{base.get('engine_version') or 'pre-stamp'} -> " \
                  f"{cand.get('engine_version') or 'pre-stamp'}]"
        print(f"{verdict:16} {cand['tag']}  {note}{eng}")
        record("quality", verdict, cand["tag"], note,
               metric="gsm8k_acc",
               base_acc=base.get("acc"), cand_acc=cand.get("acc"),
               base_n=base.get("n"), cand_n=cand.get("n"),
               base_engine=base.get("engine_version") or None,
               cand_engine=cand.get("engine_version") or None,
               model_id=cand.get("model_id") or None)
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
# has corrected it in place ("INT4 (QAT)" -> "wNa8o8 (...)" for the SAME artifact, quant-label-rule
# audit); model_id carries the recipe when it genuinely differs. A label mismatch is
# surfaced on the output line instead. cold_run IS in the key: cold and warm are
# different published cells, and pooling them inflates spread past the spread-rule.
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


def load_anchor_cells(path):
    """anchor=1 lines of a cells file -> [(runtime, model_id, task)]."""
    anchors = []
    for raw in open(path):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 4 and "anchor=1" in parts[4:]:
            anchors.append((parts[1], parts[2], parts[3]))
    return anchors


def anchor_median(side_cells, anchors, device, cold_run, exclude_runtime, spread_limit):
    """Median of a session-anchor cell within one selection. An anchor is
    usable only if (a) its runtime is not the engine under test — an anchor
    measured by the engine being bumped moves WITH the engine and normalizes
    the change away (first Mac run proved it: the litert anchor improved 14%
    and every mlx cell read a phantom -12%) — and (b) it has n>=2 within
    the spread gate (an n=1 anchor passes the spread check trivially and a
    single noisy cold value multiplies into every verdict; measured: a 31ms
    -> 611ms cold-TTFT anchor stamped +1620% on an actually-flat cell).
    Returns (median, label) or (None, why)."""
    for rt, mid, task in anchors:
        if rt == exclude_runtime:
            continue
        key = (device, rt, mid, task, cold_run)
        vals = [v for v, _ in side_cells.get(key, [])]
        if len(vals) < 2:
            continue
        med = statistics.median(vals)
        spread = (max(vals) - min(vals)) / med * 100 if med else 0.0
        if spread > spread_limit:
            continue  # an unreliable anchor normalizes nothing (spread-rule)
        return med, f"{rt} {mid} {task}"
    return None, "no usable anchor (need n>=2, in-spread, runtime != engine under test)"


def run_device(args):
    rows = load_csv("device-runs.csv")
    base_rows, cand_rows = select(rows, args.baseline), select(rows, args.candidate)
    if not base_rows or not cand_rows:
        print(f"selector matched no rows: "
              f"{args.baseline if not base_rows else args.candidate}", file=sys.stderr)
        return 2
    worst = 0
    b_quants, c_quants = quants(base_rows), quants(cand_rows)
    anchors = load_anchor_cells(args.anchors) if args.anchors else []
    any_common = False
    # ttft/memory: lower is better; both were columns without a diff until 2026-08-17
    for metric, higher_is_better in (("decode_tps", True), ("prefill_tps", True),
                                     ("ttft_ms", False),
                                     ("mem_footprint_median_mb", False),
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
            base_rec = dict(metric=metric, base_median=round(bm, 3),
                            cand_median=round(cm, 3), base_n=len(bv), cand_n=len(cv),
                            base_spread_pct=round(bs, 1), cand_spread_pct=round(cs, 1),
                            raw_delta_pct=round(delta, 2))
            if bs > args.spread_limit or cs > args.spread_limit:
                # spread-rule: contention halves decode and the only tell is spread
                print(f"UNRELIABLE       {label}  {note}  [spread > {args.spread_limit:.0f}% — throw out]")
                record("device", "UNRELIABLE", label, note, **base_rec)
                continue
            if b_dates and c_dates and b_dates.isdisjoint(c_dates):
                # Different sittings: device-state drift (16-25% across days) dwarfs
                # engine deltas. With --anchors, score the ANCHOR-NORMALIZED delta
                # instead — cross-session ratios are only ever computed through
                # anchors (continuous-benchmarking-proposal §2) — while still
                # printing the raw numbers.
                # Only the arm under test earns an anchor-normalized verdict:
                # other arms' engines are identical on both sides, so their
                # cross-session raw delta IS the session-drift signal (they are
                # anchors in spirit) — scoring them through another arm's
                # anchor manufactures phantom verdicts.
                eut = args.engine_under_test
                if anchors and (eut is None or key[1] == eut):
                    excl = eut if eut else key[1]
                    ba, ba_label = anchor_median(b_cells, anchors, key[0], key[-1],
                                                 excl, args.spread_limit)
                    ca, ca_label = anchor_median(c_cells, anchors, key[0], key[-1],
                                                 excl, args.spread_limit)
                elif anchors:
                    ba, ba_label = None, f"arm not under test ({eut}) — raw cross-session is the drift signal"
                    ca, ca_label = None, ba_label
                else:
                    ba, ba_label = None, "no --anchors"
                    ca, ca_label = None, ba_label
                if ba and ca:
                    ndelta = ((cm / ca) / (bm / ba) - 1) * 100
                    worse = ndelta < -args.threshold_pct if higher_is_better else ndelta > args.threshold_pct
                    better = ndelta > args.threshold_pct if higher_is_better else ndelta < -args.threshold_pct
                    verdict = "REGRESSION" if worse else ("IMPROVED" if better else "OK")
                    anote = (f"{note}  [anchor-normalized {ndelta:+.1f}% via {ba_label}; "
                             f"raw cross-session delta is informational]")
                    print(f"{verdict:16} {label}  {anote}")
                    record("device", verdict, label, anote, **base_rec,
                           anchor={"base_median": round(ba, 3), "cand_median": round(ca, 3),
                                   "cell": ba_label, "normalized_delta_pct": round(ndelta, 2)})
                    if worse:
                        worst = 1
                else:
                    print(f"INFO-ONLY        {label}  {note}  [cross-session — do not pool; "
                          f"anchor unavailable ({ba_label if not ba else ca_label}); "
                          f"use a same-session A/B for a verdict]")
                    record("device", "INFO-ONLY", label, note, **base_rec)
                continue
            worse = delta < -args.threshold_pct if higher_is_better else delta > args.threshold_pct
            better = delta > args.threshold_pct if higher_is_better else delta < -args.threshold_pct
            verdict = "REGRESSION" if worse else ("IMPROVED" if better else "OK")
            print(f"{verdict:16} {label}  {note}")
            record("device", verdict, label, note, **base_rec)
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
                    help="device: per-side trial spread (%%) beyond which a cell is UNRELIABLE (spread-rule)")
    ap.add_argument("--no-rebuild", action="store_true",
                    help="skip regenerating results/summary/ first")
    ap.add_argument("--anchors",
                    help="cells file whose anchor=1 cells normalize cross-session "
                         "device pairs (e.g. matrices/anchors.cells)")
    ap.add_argument("--engine-under-test",
                    help="runtime id whose release is being tested (e.g. litert-lm): "
                         "only its cells get anchor-normalized verdicts, and anchors "
                         "measured by it are excluded; other arms stay INFO-ONLY")
    ap.add_argument("--json-out",
                    help="write per-cell machine verdicts (regression_report.py "
                         "persists these as verdicts.json)")
    args = ap.parse_args()
    if not args.no_rebuild:
        rebuild_summary()
    if args.mode == "quality":
        rc = run_quality(args)
    else:
        if not (args.baseline and args.candidate):
            print("device mode needs --baseline and --candidate selectors", file=sys.stderr)
            return 2
        rc = run_device(args)
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        payload = {"mode": args.mode, "baseline": args.baseline or args.baseline_tag,
                   "candidate": args.candidate or args.candidate_tag or args.candidate_dir,
                   "thresholds": {"points": args.threshold_points,
                                  "pct": args.threshold_pct,
                                  "spread_limit": args.spread_limit},
                   "exit_code": rc, "verdicts": RECORDS}
        json.dump(payload, open(args.json_out, "w"), indent=2)
        print(f"\nwrote {args.json_out} ({len(RECORDS)} verdicts)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
