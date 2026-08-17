#!/usr/bin/env python3
"""Run regression_diff and persist the outcome as a release-regression report.

  python3 scripts/regression_report.py --engine litert-lm --version v0.16.0 \\
      device --baseline campaign:2026-08-04 --candidate campaign:2026-08-17 \\
      --anchors matrices/anchors.cells

Everything after the --engine/--version pair is forwarded to regression_diff.py
verbatim. Output dir: results/regression-reports/<date>-<engine>-<version>/
  report.md       the human verdict text (stdout of the diff)
  verdicts.json   per-cell machine verdicts (--json-out)
  invocation.txt  the exact command, for provenance
The committed verdicts.json files are the append-only canon behind
results/summary/history.csv (build_summary.py regenerates it from them).
Exit code mirrors regression_diff (1 = REGRESSION somewhere).
"""
import argparse
import datetime
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="engine under test, e.g. litert-lm")
    ap.add_argument("--version", required=True, help="candidate engine version, e.g. v0.16.0")
    ap.add_argument("diff_args", nargs=argparse.REMAINDER,
                    help="arguments forwarded to regression_diff.py")
    args = ap.parse_args()
    args.diff_args = [a for a in args.diff_args if a != "--"]
    if not args.diff_args:
        ap.error("pass regression_diff.py arguments after --engine/--version")

    date = datetime.date.today().isoformat()
    outdir = os.path.join(ROOT, "results", "regression-reports",
                          f"{date}-{args.engine}-{args.version}")
    os.makedirs(outdir, exist_ok=True)

    cmd = [sys.executable, os.path.join(ROOT, "scripts", "regression_diff.py"),
           *args.diff_args, "--json-out", os.path.join(outdir, "verdicts.json")]
    with open(os.path.join(outdir, "invocation.txt"), "w") as fh:
        fh.write(" ".join(cmd) + "\n")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)

    with open(os.path.join(outdir, "report.md"), "w") as fh:
        fh.write(f"# Regression report — {args.engine} {args.version} ({date})\n\n"
                 f"```\n{' '.join(cmd)}\n```\n\n```\n{proc.stdout}\n```\n")
        if proc.stderr.strip():
            fh.write(f"\nstderr:\n```\n{proc.stderr}\n```\n")
    print(f"report: {os.path.relpath(outdir, ROOT)} (exit {proc.returncode})")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
