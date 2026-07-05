#!/usr/bin/env python3
"""Gemma-4-12B on M4 Max (Mac Studio, 40-core GPU, 546 GB/s): Core AI vs MLX vs LiteRT-LM.
Reduces the raw per-trial decode rates to summary stats and derives the memory-bandwidth
roofline numbers that normalise for the int8-vs-int4 quant asymmetry. Writes summary.json."""
import json, statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
BW_GBPS = 546.0  # M4 Max, 40-core GPU bin

# Per-trial decode tok/s (greedy, n=5, cold).
TRIALS = {
    "MLX":              [55.077, 55.122, 55.466, 54.958, 55.049],  # int4 (mlx-vlm, gemma4_unified)
    "LiteRT-LM":        [48.4, 48.5, 48.6, 48.4, 48.6],            # int4 (litert-mac-verify, Apple GPU)
    "Core AI int4-q40": [37.8, 37.9, 37.9, 37.9, 37.9],           # int4 q4_0-snapped + msdpa (this work)
    "Core AI int8":     [24.754, 24.793, 24.788, 24.763, 24.737], # int8lin, GPU-pipelined (shipped)
}
# Weight bytes streamed per decode token — approximated by the on-disk quantised weight size
# (GB). Whole-artifact sizes; text decode reads a subset, so treat %-of-roofline as directional.
WEIGHT_GB = {"MLX": 6.74, "LiteRT-LM": 6.10, "Core AI int4-q40": 8.20, "Core AI int8": 14.67}
QUANT     = {"MLX": "int4", "LiteRT-LM": "int4", "Core AI int4-q40": "int4-q4_0", "Core AI int8": "int8lin"}
PREFILL   = {"MLX": 466.7, "LiteRT-LM": None, "Core AI int4-q40": None, "Core AI int8": None}
NOTE      = {"MLX": "coherent",
             "LiteRT-LM": "DEGENERATE output on Mac build (throughput valid, quality broken)",
             "Core AI int4-q40": "coherent (q4_0-snap == fp16 gen); 1.53x the int8 path; 8.2GB",
             "Core AI int8": "coherent; shipped; prefill N/A (decode-only export)"}

rows = {}
for rt, xs in TRIALS.items():
    mean = st.mean(xs)
    wb = WEIGHT_GB[rt]
    ach = mean * wb                 # achieved GB/s = tok/s * GB/token
    ceil = BW_GBPS / wb             # tok/s ceiling at this quant
    rows[rt] = {
        "quant": QUANT[rt], "weight_gb": wb,
        "decode_tps_mean": round(mean, 2), "decode_tps_sd": round(st.pstdev(xs), 3),
        "prefill_tps": PREFILL[rt],
        "achieved_gbps": round(ach, 1), "roofline_tps": round(ceil, 1),
        "pct_of_roofline": round(100 * mean / ceil, 1),
        "note": NOTE[rt],
    }

summary = {
    "model": "google/gemma-4-12B-it (unified, encoder-free multimodal)",
    "hardware": "Mac Studio · Apple M4 Max · 40-core GPU · 546 GB/s · 128 GB",
    "protocol": "greedy, n=5 cold; steady-state decode (512/445/640 gen tok for CoreAI/MLX/LiteRT)",
    "date": "2026-07-05",
    "runtimes": rows,
}
(HERE / "summary.json").write_text(json.dumps(summary, indent=2))

w = max(len(r) for r in rows)
print(f"{'runtime':<{w}}  quant    decode   sd     ach GB/s  ceil t/s  %roof  note")
for rt, r in sorted(rows.items(), key=lambda kv: -kv[1]["decode_tps_mean"]):
    print(f"{rt:<{w}}  {r['quant']:<7} {r['decode_tps_mean']:6.2f}  {r['decode_tps_sd']:.3f}  "
          f"{r['achieved_gbps']:6.1f}   {r['roofline_tps']:6.1f}   {r['pct_of_roofline']:4.1f}   {r['note']}")
print("\nwrote", HERE / "summary.json")
