#!/usr/bin/env python3
"""Post-capture gate for one cell: SHORT / HOT / SPREAD / OK.

Automates the two flags the protocol already defines, instead of leaving them
to the operator's eye between cells (fairness cold-warm-split thermal guard;
spread-rule, same 5% bar as render_leaderboard and regression_diff):

  SHORT <n>        fewer records than --runs (crash/timeout — do NOT retry
                   here; failed-runs-stay owns that path)
  DEAD <n>         a judged record has no decode number (missing key or 0) —
                   the run completed as a record but not as a measurement
                   (audited 2026-08-27: zero-decode runs silently passed and
                   vanished from the spread list)
  HOT <states>     an initialThermalState outside --ok-thermal
  SPREAD <pct>     warm decode (max-min)/median exceeded --spread-flag
  COLLAPSE <pct>   cold-only capture (Android regime has no warm runs, so
                   SPREAD can never fire) whose slowest cold decode fell
                   under half the cold median — the contended-device
                   signature (measured 2026-08-27 Pixel: 0.3-1.4 tok/s junk
                   beside 22-26 clean in one session). The bar is 50%, not
                   --spread-flag: Android cold trials legitimately spread
                   15-30% and a 5% bar would flag every capture.
  OK

Input: schema-v1 records — --jsonl <cell.jsonl> (mac runner) or positional
per-run .json files (iPhone device-jsonl pulls). Only the newest --runs
records are judged: a retried cell is judged on its retry, not its history.

Runners act on the verdict by quarantining the flagged capture (mac:
<cell>.jsonl.attempt1, outside build_summary's *.jsonl glob; iPhone:
device-jsonl-flagged/) and re-running ONCE. The flagged capture always stays
on disk in raw — replaced wholesale, never mixed run-by-run (no-cherry-pick).
Exit: 0 = OK, 1 = flagged.
"""
import argparse
import json
import statistics
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, required=True)
    ap.add_argument("--spread-flag", type=float, default=5.0)
    ap.add_argument("--ok-thermal", default="nominal",
                    help="comma list; empty/missing states always pass")
    ap.add_argument("--jsonl")
    ap.add_argument("files", nargs="*")
    a = ap.parse_args()

    recs = []
    try:
        if a.jsonl:
            with open(a.jsonl) as fh:
                recs = [json.loads(ln) for ln in fh if ln.strip()]
        else:
            for f in a.files:
                with open(f) as fh:
                    recs.append(json.load(fh))
    except (OSError, json.JSONDecodeError) as e:
        print(f"SHORT 0 ({e.__class__.__name__})")
        return 1
    recs.sort(key=lambda d: d.get("timestamp") or "")
    recs = recs[-a.runs:]
    if len(recs) < a.runs:
        print(f"SHORT {len(recs)}")
        return 1

    dead = sum(1 for r in recs
               if not r.get("metrics", {}).get("decodeTokensPerSecond"))
    if dead:
        print(f"DEAD {dead}")
        return 1

    ok = set(a.ok_thermal.split(",")) | {"", None}
    states = [r.get("metrics", {}).get("initialThermalState") for r in recs]
    if any(s not in ok for s in states):
        print("HOT " + ",".join(str(s) for s in states))
        return 1

    warm = [m["decodeTokensPerSecond"] for r in recs
            for m in [r.get("metrics", {})]
            if not m.get("coldRun")]
    if len(warm) > 1:
        med = statistics.median(warm)
        spread = (max(warm) - min(warm)) / med * 100 if med else 0.0
        if spread > a.spread_flag:
            print(f"SPREAD {spread:.1f}")
            return 1
    if not warm:
        cold = [r["metrics"]["decodeTokensPerSecond"] for r in recs]
        if len(cold) > 1:
            med = statistics.median(cold)
            if med and min(cold) < med / 2:
                print(f"COLLAPSE {min(cold) / med * 100:.0f}")
                return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
