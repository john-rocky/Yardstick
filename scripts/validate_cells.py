#!/usr/bin/env python3
"""Validate matrix cell files (matrices/*.cells). Grammar: matrices/README.md.

  python3 scripts/validate_cells.py matrices/*.cells
  python3 scripts/validate_cells.py --require-anchor matrices/release-regression-litert.cells
  python3 scripts/validate_cells.py --catalog catalog.json matrices/apple-warm-matrix.cells

--catalog: output of `yardstick list --json`; model ids are checked against it
(cells with local=1 or platform=android are exempt — android ids are HF repos
resolved by the android driver, not ModelCatalog entries).
Exit 1 on any error; warnings don't fail.
"""
import argparse
import json
import re
import sys

PLATFORMS = {"ios", "mac", "android"}
RUNTIMES = {"mlx-swift", "llama.cpp", "coreml-llm", "litert-lm", "executorch",
            "anemll", "apple-fm", "core-ai", "cactus"}
TASKS = {"short-chat", "long-context-512", "long-context-1024",
         "long-context-1024-gen256", "long-context", "long-context-3k",
         "long-context-8k", "long-context-32k", "cactus-parity", "sustained",
         "energy", "quality", "lifecycle"}
NATIVE_TASK = re.compile(r"^native-benchmark-\d+x\d+$")
INT_KEYS = {"runs", "context-tokens", "max-tokens", "cooldown"}
FLAG_KEYS = {"anchor", "manual", "local"}          # value must be 1
STR_KEYS = {"exclude", "file", "backend"}
BACKENDS = {"cpu", "gpu"}


def parse_line(line):
    """-> (platform, runtime, model_id, task, opts_dict) or raises ValueError."""
    parts = line.split()
    if len(parts) < 4:
        raise ValueError(f"need at least 4 columns, got {len(parts)}")
    plat, rt, mid, task = parts[:4]
    opts = {}
    for kv in parts[4:]:
        if "=" not in kv:
            raise ValueError(f"option {kv!r} is not key=value")
        k, v = kv.split("=", 1)
        if k in opts:
            raise ValueError(f"duplicate option {k!r}")
        if k in INT_KEYS:
            if not v.isdigit():
                raise ValueError(f"{k}={v!r} is not an integer")
        elif k in FLAG_KEYS:
            if v != "1":
                raise ValueError(f"{k}={v!r} (flags take only =1)")
        elif k in STR_KEYS:
            if not v:
                raise ValueError(f"{k}= needs a value")
            if k == "backend" and v not in BACKENDS:
                raise ValueError(f"backend={v!r} (want cpu|gpu)")
        else:
            raise ValueError(f"unknown option key {k!r}")
        opts[k] = v
    return plat, rt, mid, task, opts


def validate_file(path, catalog=None, require_anchor=False):
    errors, warnings = [], []
    seen = {}
    anchors_by_platform = set()
    platforms_in_file = set()
    with open(path) as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            where = f"{path}:{lineno}"
            try:
                plat, rt, mid, task, opts = parse_line(line)
            except ValueError as e:
                errors.append(f"{where}: {e}")
                continue
            if plat not in PLATFORMS:
                errors.append(f"{where}: unknown platform {plat!r}")
            if rt not in RUNTIMES:
                errors.append(f"{where}: unknown runtime {rt!r}")
            if task not in TASKS and not NATIVE_TASK.match(task):
                errors.append(f"{where}: unknown task {task!r}")
            if task == "energy" and opts.get("manual") != "1":
                errors.append(f"{where}: energy task requires manual=1 "
                              "(unplug discipline is a human step)")
            if opts.get("backend") and plat != "android":
                errors.append(f"{where}: backend= is android-only "
                              "(Apple arms encode backend in the model id)")
            if opts.get("max-tokens") and plat == "mac":
                errors.append(f"{where}: the Mac CLI has no --max-tokens flag "
                              "(BenchmarkRunner.Configuration carries no budget "
                              "override) — encode the budget in the task id "
                              "(e.g. long-context-1024-gen256)")
            key = (plat, rt, mid, task, opts.get("backend", ""))
            if key in seen:
                errors.append(f"{where}: duplicate cell (first at line {seen[key]})")
            else:
                seen[key] = lineno
            platforms_in_file.add(plat)
            if opts.get("anchor") == "1":
                anchors_by_platform.add(plat)
                if opts.get("exclude") or opts.get("manual"):
                    errors.append(f"{where}: an anchor cell cannot be "
                                  "excluded/manual")
            if (catalog is not None and plat != "android"
                    and opts.get("local") != "1" and opts.get("exclude") is None):
                if mid not in catalog.get(rt, []):
                    errors.append(f"{where}: model id {mid!r} not in the "
                                  f"{rt} catalog (side-loaded? add local=1)")
    if require_anchor:
        for plat in sorted(platforms_in_file - anchors_by_platform):
            errors.append(f"{path}: no anchor=1 cell for platform {plat!r} "
                          "(regression matrices need one per platform)")
    return errors, warnings


def load_catalog(path):
    """yardstick list --json -> {runtime: [model ids]}."""
    data = json.load(open(path))
    cat = {}
    for rt, models in data.get("models", data).items():
        cat[rt] = [m["id"] if isinstance(m, dict) else m for m in models]
    return cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--catalog", help="yardstick list --json output")
    ap.add_argument("--require-anchor", action="store_true")
    args = ap.parse_args()
    catalog = load_catalog(args.catalog) if args.catalog else None
    failed = False
    for path in args.files:
        errors, warnings = validate_file(path, catalog, args.require_anchor)
        for w in warnings:
            print(f"WARN {w}")
        for e in errors:
            print(f"ERROR {e}")
            failed = True
        if not errors:
            print(f"OK {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
