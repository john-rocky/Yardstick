#!/usr/bin/env python3
"""
Import a warm-campaign capture (results/raw/<campaign>/device-jsonl/*.json) into
results/raw/ using the repo's `<device>-<runtime>-<model>-<task>-runN.jsonl`
convention, and retire the superseded old-session rows of the SAME
(device, runtime, model, task) cells into results/raw/superseded/<campaign>/.

Why retirement instead of pooling: cross-session medians are invalid on this
device — the same binary + pins measured 1.3x apart across sessions weeks apart
(results/raw/2026-07-13-mlx-variance/README.md). A re-captured cell must be
represented by ONE session (cold run1 + warm runs 2-4), per fairness-rules §2.

Usage: python3 scripts/import_warm_campaign.py results/raw/2026-07-13-iphone-warm [--dry-run]
"""
import json
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "results" / "raw"
DEVICE = "iphone17pro"


def short_rt(rt: str) -> str:
    return {"mlx-swift": "mlx"}.get(rt, rt)


def short_model(mid: str) -> str:
    s = re.sub(r"[^a-z0-9.\-]+", "-", mid.split("/")[-1].lower())
    return re.sub(r"-4bit$", "", s)


def main() -> None:
    campaign = Path(sys.argv[1])
    dry = "--dry-run" in sys.argv
    superseded = RAW / "superseded" / campaign.name
    excluded: set[str] = set()
    exc_file = campaign / "EXCLUDED.txt"
    if exc_file.exists():
        excluded = {
            line.strip()
            for line in exc_file.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
    files = [
        f for f in sorted((campaign / "device-jsonl").glob("*.json"))
        if f.name not in excluded
    ]
    print(f"excluded: {len(excluded)} files (EXCLUDED.txt)")

    # Group campaign files per cell, ordered by embedded timestamp (filename).
    cells: dict[tuple[str, str, str], list[Path]] = {}
    for f in files:
        d = json.loads(f.read_text())
        key = (d["runtime"], d["model"]["id"], d["task"])
        cells.setdefault(key, []).append(f)

    # Retire old-session rows by PARSED identity, not filename: naming drifted
    # across eras (mlx- vs mlx-swift-, with/without -4bit), so glob would miss.
    for old in sorted(RAW.glob(f"{DEVICE}-*.jsonl")):
        try:
            d = json.loads(old.read_text())
        except Exception:
            continue
        key = (d.get("runtime"), (d.get("model") or {}).get("id"), d.get("task"))
        if key in cells:
            print(f"retire  {old.name}")
            if not dry:
                superseded.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old), superseded / old.name)

    for (rt, mid, task), fs in sorted(cells.items()):
        base = f"{DEVICE}-{short_rt(rt)}-{short_model(mid)}-{task}"
        # Import the campaign session as run1..N (timestamp order = run order).
        for i, f in enumerate(fs, start=1):
            dst = RAW / f"{base}-run{i}.jsonl"
            print(f"import  {f.name} -> {dst.name}")
            if not dry:
                dst.write_text(json.dumps(json.loads(f.read_text())))

    if not dry and any(cells):
        note = superseded / "README.md"
        superseded.mkdir(parents=True, exist_ok=True)
        note.write_text(
            f"# Superseded by {campaign.name}\n\n"
            "Old-session rows for cells re-captured by the warm campaign. Kept for\n"
            "audit; excluded from RESULTS.md because cross-session pooling is invalid\n"
            "on this device (see results/raw/2026-07-13-mlx-variance/README.md).\n"
        )
    print(f"{'DRY RUN — ' if dry else ''}cells: {len(cells)}")


if __name__ == "__main__":
    main()
