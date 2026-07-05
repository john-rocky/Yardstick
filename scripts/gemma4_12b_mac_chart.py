#!/usr/bin/env python3
"""Gemma-4-12B on M4 Max — Core AI vs MLX vs LiteRT-LM decode, now at iso-int4.

Adds a quality-preserving int4 Core AI (q4_0 pre-snap: greedy generation is bit-identical to
fp16 — 'The capital of France is Paris.'), so the comparison is apples-to-apples 4-bit:
  - iso-int4 ranking: MLX 55.1 > LiteRT 48.5 > Core AI 37.9. MLX's engine is the most
    bandwidth-efficient (68% of the 546 GB/s roofline); Core AI's int4 dequant path is only
    57%, LOWER than its own int8 path (67%), so halving the bytes buys 1.53x, not 2x.
  - the win: a coherent int4 Core AI now exists and is 1.53x the shipped int8 (24.8 -> 37.9).
  - LiteRT is mid-pack on speed but emits degenerate output on this Mac build (unusable).

Left: absolute decode tok/s. Right: % of the 546 GB/s memory-bandwidth roofline (engine
efficiency). Outputs docs/charts/gemma4_12b_mac.png.
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "docs" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.alpha": 0.3,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlepad": 9,
    "figure.dpi": 140, "savefig.dpi": 140, "savefig.bbox": "tight",
})

# Mac Studio M4 Max, 40-core GPU (546 GB/s), 128 GB. Gemma-4-12B, greedy, n=5 cold.
# %roof = decode tok/s / (546 / on-disk weight GB). See results/raw/2026-07-05-gemma4-12b-mac.
RT      = ["MLX", "LiteRT-LM", "Core AI\nint4·q4_0", "Core AI\nint8"]
KEY     = ["MLX", "LiteRT-LM", "CoreAI4", "CoreAI8"]
COLOR   = {"MLX": "#7c3aed", "LiteRT-LM": "#0d9488", "CoreAI4": "#e11d48", "CoreAI8": "#fb7185"}
QUANT   = {"MLX": "int4", "LiteRT-LM": "int4", "CoreAI4": "int4", "CoreAI8": "int8"}
DECODE  = {"MLX": 55.13, "LiteRT-LM": 48.50, "CoreAI4": 37.88, "CoreAI8": 24.77}
PCTROOF = {"MLX": 68.1, "LiteRT-LM": 54.2, "CoreAI4": 56.9, "CoreAI8": 66.5}
DEGEN   = {"LiteRT-LM"}
NEWMARK = {"CoreAI4"}
INK, MUTED = "#1f2937", "#6b7280"
colors = [COLOR[k] for k in KEY]
x = np.arange(len(KEY))

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.9))

# --- Panel A: absolute decode tok/s ---
dec = [DECODE[k] for k in KEY]
axL.bar(x, dec, 0.64, color=colors, edgecolor="white",
        hatch=["//" if k in DEGEN else "" for k in KEY])
axL.set_ylim(0, 72)
for xi, k in zip(x, KEY):
    axL.text(xi, DECODE[k] + 1.0, f"{DECODE[k]:.1f}", ha="center", va="bottom",
             fontsize=11.5, fontweight="bold", color=INK)
    axL.text(xi, 1.9, QUANT[k], ha="center", va="bottom", fontsize=9, fontweight="bold", color="white")
# int8 -> int4 Core AI improvement arrow
axL.annotate("", xy=(2, 37.88), xytext=(3, 24.77),
             arrowprops=dict(arrowstyle="->", color="#7f1d1d", lw=1.6,
                             connectionstyle="arc3,rad=-0.25"))
axL.text(2.5, 46.5, "quality-preserving\nint4  →  1.53×", ha="center", fontsize=9.5,
         color="#7f1d1d", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.3", fc="#fee2e2", ec="none"))
axL.text(1.5, 67.5, "iso-int4:  MLX 55 > LiteRT 48 > Core AI 38", ha="center", fontsize=9.5,
         color="#3730a3", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.3", fc="#e0e7ff", ec="none"))
axL.set_xticks(x); axL.set_xticklabels(RT, fontsize=9.5)
axL.set_ylabel("decode tok/s  (↑ better)", fontsize=10)
axL.set_title("Absolute decode speed", loc="left")

# --- Panel B: % of memory-bandwidth roofline ---
pct = [PCTROOF[k] for k in KEY]
axR.bar(x, pct, 0.64, color=colors, edgecolor="white",
        hatch=["//" if k in DEGEN else "" for k in KEY])
axR.set_ylim(0, 100)
for xi, k in zip(x, KEY):
    axR.text(xi, PCTROOF[k] + 1.5, f"{PCTROOF[k]:.0f}%", ha="center", va="bottom",
             fontsize=11.5, fontweight="bold", color=INK)
axR.text(2.5, 88, "Core AI's int4 path (57%)\n< its int8 path (67%)", ha="center", fontsize=9,
         color="#7f1d1d", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.3", fc="#fee2e2", ec="none"))
axR.set_xticks(x); axR.set_xticklabels(RT, fontsize=9.5)
axR.set_ylabel("% of 546 GB/s roofline  (engine efficiency)", fontsize=10)
axR.set_title("Bandwidth efficiency", loc="left")

# per-bar quality flags
for ax in (axL, axR):
    ax.text(1, -0.135, "⚠ degenerate", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8, color="#b45309", fontweight="bold")
    ax.text(2, -0.135, "✓ coherent · NEW", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8, color="#047857", fontweight="bold")

fig.suptitle("Gemma-4-12B on M4 Max — iso-int4 decode: MLX's engine leads; "
             "a quality-preserving int4 lifts Core AI 1.53×",
             fontsize=12.5, fontweight="bold", y=1.02)
fig.text(0.5, -0.07,
         "Mac Studio · M4 Max 40-core GPU · 546 GB/s · greedy · n=5 cold   ·   "
         "Core AI int4=q4_0 pre-snap+msdpa (coherent, this work) / int8lin (shipped) · MLX int4 (mlx-vlm) · LiteRT int4 (Apple GPU)   ·   "
         "%roofline = decode tok/s ÷ (546 GB/s ÷ on-disk weight GB), directional",
         ha="center", fontsize=7.4, color=MUTED)
fig.tight_layout(w_pad=3.0)
out = OUT / "gemma4_12b_mac.png"
plt.savefig(out)
plt.close(fig)
print("wrote", out)
