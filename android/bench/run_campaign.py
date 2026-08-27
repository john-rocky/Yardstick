#!/usr/bin/env python3
"""Android campaign runner — consumes the unified cells grammar
(matrices/README.md) filtered to platform=android and drives run_cell.py.

  CAMPAIGN=<name> python3 android/bench/run_campaign.py matrices/release-regression-litert.cells

Order and discipline (fairness rules as code):
  - anchor=1 cells run FIRST (session-anchor normalization).
  - payload cells run INTERLEAVED PER ROUND across arms (interleave-arms):
    round 1 of every cell, then round 2 of every cell — never one arm's block.
  - >=COOLDOWN s between runs; the thermal gate waits for status 0 (nominal)
    up to THERMAL_WAIT s and records the state either way.
  - exclude=/manual= cells are skipped with the reason logged (SKIPPED.txt).
"""
import fcntl
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from device_probe import thermal_status  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COOLDOWN = int(os.environ.get("COOLDOWN", "120"))
THERMAL_WAIT = int(os.environ.get("THERMAL_WAIT", "600"))
SERIAL = os.environ.get("BENCH_ANDROID_SERIAL")


def parse_cells(path):
    anchors, payload, skipped = [], [], []
    for lineno, raw in enumerate(open(path), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if parts[0] != "android" or len(parts) < 4:
            continue
        cell = {"runtime": parts[1], "model_id": parts[2], "task": parts[3], "opts": {}}
        for kv in parts[4:]:
            k, _, v = kv.partition("=")
            cell["opts"][k] = v
        reason = cell["opts"].get("exclude") or ("manual" if cell["opts"].get("manual") else None)
        if reason:
            skipped.append((cell, reason))
        elif cell["opts"].get("anchor"):
            anchors.append(cell)
        else:
            payload.append(cell)
    return anchors, payload, skipped


def wait_nominal(out_dir):
    t0 = time.time()
    while time.time() - t0 < THERMAL_WAIT:
        raw, name = thermal_status(SERIAL)
        if raw == 0:
            return name
        print(f"thermal gate: status={name} — waiting…")
        time.sleep(30)
    with open(os.path.join(out_dir, "THERMAL_GATE.txt"), "a") as fh:
        fh.write(f"gate timeout after {THERMAL_WAIT}s at {time.strftime('%F %T')}; ran anyway\n")
    return name


def run_cell_once(cell, out_dir, runs):
    cmd = [sys.executable, os.path.join(ROOT, "android", "bench", "run_cell.py"),
           "--runtime", cell["runtime"], "--model-id", cell["model_id"],
           "--task", cell["task"], "--runs", str(runs), "--out", out_dir]
    if cell["opts"].get("backend"):
        cmd += ["--backend", cell["opts"]["backend"]]
    if cell["opts"].get("file"):
        cmd += ["--file", cell["opts"]["file"]]
    if cell["opts"].get("max-tokens"):
        cmd += ["--max-tokens", cell["opts"]["max-tokens"]]
    if cell["opts"].get("context-tokens"):
        cmd += ["--context-tokens", cell["opts"]["context-tokens"]]
    if SERIAL:
        cmd += ["--serial", SERIAL]
    print(f"\n=== {cell['runtime']}{'/' + cell['opts'].get('backend', '') if cell['opts'].get('backend') else ''} "
          f"{cell['model_id']} {cell['task']} ({time.strftime('%H:%M:%S')})")
    rc = subprocess.call(cmd)
    if rc:
        with open(os.path.join(out_dir, "FAILURES.txt"), "a") as fh:
            fh.write(f"{cell['runtime']} {cell['model_id']} {cell['task']} rc={rc}\n")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    cells_file = sys.argv[1]
    # One driver per device. Two campaigns interleaving on one phone poison
    # BOTH sets of numbers (measured 2026-08-27: a contended anchor read 0.4
    # tok/s against a clean 22-26, and a payload cell 5.0 against 25.9) — and
    # the round-robin runner outlives its last log line, so "looks finished"
    # is not finished. The lock makes the mistake impossible instead of rare.
    lock = open(f"/tmp/edge-llm-bench-android-{SERIAL or 'default'}.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"another campaign is already driving device {SERIAL or '(default)'} — "
              "refusing to start (two drivers corrupt both campaigns)", file=sys.stderr)
        return 3
    campaign = os.environ.get("CAMPAIGN", time.strftime("%Y-%m-%d") + "-android-matrix")
    out_dir = os.path.join(ROOT, "results", "raw", campaign, "app-path-android")
    os.makedirs(out_dir, exist_ok=True)
    anchors, payload, skipped = parse_cells(cells_file)
    for cell, reason in skipped:
        line = f"CELL_SKIP {cell['runtime']} {cell['model_id']} {cell['task']} reason={reason}"
        print(line)
        with open(os.path.join(out_dir, "SKIPPED.txt"), "a") as fh:
            fh.write(line + "\n")
    if not anchors and not payload:
        print("no android cells in this file")
        return 0

    with open(os.path.join(out_dir, "session_provenance.txt"), "a") as fh:
        fh.write(f"session start {time.strftime('%F %T')} cells={cells_file} "
                 f"cooldown={COOLDOWN}s\n")

    first = True
    for cell in anchors:  # anchors first, all their runs at once
        if not first:
            time.sleep(COOLDOWN)
        first = False
        wait_nominal(out_dir)
        run_cell_once(cell, out_dir, int(cell["opts"].get("runs", "3")))

    # payload: interleave per round (round-robin across cells)
    max_rounds = max((int(c["opts"].get("runs", "3")) for c in payload), default=0)
    for rnd in range(1, max_rounds + 1):
        for cell in payload:
            if rnd > int(cell["opts"].get("runs", "3")):
                continue
            time.sleep(int(cell["opts"].get("cooldown", COOLDOWN)))
            wait_nominal(out_dir)
            run_cell_once(cell, out_dir, 1)
        print(f"--- round {rnd}/{max_rounds} complete (publish every round)")
    print(f"\ncampaign dir: {os.path.dirname(out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
