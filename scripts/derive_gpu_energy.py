#!/usr/bin/env python3
"""Derive the uniform GPU-only energy column for a Mac powermetrics energy campaign.

    python3 scripts/derive_gpu_energy.py results/raw/<campaign>/ENERGY_PM_*.jsonl

Why this exists
---------------
Measured 2026-07-28 on macOS 27 beta: powermetrics' CPU power sampler is flaky — it read
`CPU Power: 0 mW` on every sample for three cells of the same campaign and real values
(~23 W) for a fourth. `energyJoulesPerToken` as patched by `measure_energy.py` uses the
combined (CPU+GPU+ANE) average, so with a flaky CPU sampler the recorded column silently
mixes two different quantities across cells and CANNOT be ranked.

The GPU sampler was live in every cell, and `measure_energy.py` already stores its
window-clipped average as `averageGPUPowerW`. This script derives, per row:

    energyJoulesGPU          = averageGPUPowerW * energyMeasurementWindowSeconds
    energyJoulesPerTokenGPU  = energyJoulesGPU / generatedTokenCount

and writes them back into the JSONL (new fields; the combined-basis originals are kept,
labeled by `energyBasisNote`). The GPU-only column is the one uniform, rankable basis on
this OS build; where the CPU sampler did fire, the combined figure remains as an annotation
of how much the GPU-only basis understates whole-package energy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print(__doc__)
        return 2
    for p in paths:
        lines = [l for l in p.read_text().splitlines() if l.strip()]
        changed = False
        out = []
        for line in lines:
            o = json.loads(line)
            m = o.get("metrics", {})
            gpu_w = m.get("averageGPUPowerW")
            win = m.get("energyMeasurementWindowSeconds")
            tok = m.get("generatedTokenCount") or 0
            if gpu_w and win and tok:
                m["energyJoulesGPU"] = round(gpu_w * win, 4)
                m["energyJoulesPerTokenGPU"] = round(gpu_w * win / tok, 6)
                cpu_w = m.get("averageCPUPowerW")
                m["energyBasisNote"] = (
                    "GPU-only column derived by scripts/derive_gpu_energy.py; the combined "
                    f"energyJoulesPerToken sampled CPU at {cpu_w} W and is not cross-cell "
                    "comparable on this OS build (flaky CPU sampler)."
                )
                changed = True
            out.append(json.dumps(o))
        if changed:
            p.write_text("\n".join(out) + "\n")
            last = json.loads(out[-1])["metrics"]
            print(f"{p.name}: GPU J/tok = {last.get('energyJoulesPerTokenGPU')} "
                  f"(combined was {last.get('energyJoulesPerToken')}, "
                  f"CPU sampler {last.get('averageCPUPowerW')} W)")
        else:
            print(f"{p.name}: nothing to derive (missing GPU power / window / tokens)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
