#!/usr/bin/env python3
"""One-off article charts: iPhone battery (tok per 1%) + iPhone ITL p99.

Numbers hand-copied from RESULTS.md (energy + latency-profile tables).
Outputs to docs/charts/:

    iphone_battery_per_pct.png
    iphone_itl_jitter.png

    python3 scripts/iphone_battery_jitter_charts.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "charts"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "x",
    "grid.alpha": 0.3,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlepad": 14,
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "savefig.bbox": "tight",
})

C_LITERT = "#ef4444"     # LiteRT-LM = red (Google)
C_COREML = "#f59e0b"     # CoreML/ANE = amber (house palette)
C_MLX = "#7c3aed"        # MLX = violet (house palette)
C_LLAMA = "#0ea5e9"      # llama.cpp = sky (house palette)
C_CAI_GPU = "#e11d48"    # Core AI GPU = rose (coreai_chart.py)
C_CAI_ANE = "#fb7185"    # Core AI ANE = light rose


def battery():
    # (label, tok/1%, J/tok, color) — iPhone 17 Pro, Gemma 4 E2B, energy task
    rows = [
        ("LiteRT-LM (INT4 QAT)", 4074, 0.146, C_LITERT),
        ("CoreML-LLM · ANE (INT4)", 2867, 0.207, C_COREML),
        ("MLX (Q4)", 2560, 0.232, C_MLX),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ys = range(len(rows))[::-1]
    for y, (label, tok, j, c) in zip(ys, rows):
        ax.barh(y, tok, color=c, edgecolor="white", linewidth=0.6, height=0.62)
        ax.text(tok + 40, y, f"{tok:,} tok  ·  {j:.3f} J/tok",
                va="center", fontsize=9.5, color="#222")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows], fontsize=10)
    ax.set_xlim(0, 5300)
    ax.set_xlabel("generated tokens per 1% of battery (higher = better)")
    ax.set_title("Tokens per 1% battery — iPhone 17 Pro, Gemma 4 E2B\n"
                 "600 s sustained generation, unplugged, battery-delta method")
    fig.text(0.5, -0.16,
             "Avg draw is nearly identical (4.5–4.9 W) — the gap is tokens per joule, not watts. n=1 per row.",
             ha="center", fontsize=8.5, color="#555")
    fig.savefig(OUT / "iphone_battery_per_pct.png")
    print("saved:", OUT / "iphone_battery_per_pct.png")


def jitter():
    # (label, p99 ms, color) — iPhone 17 Pro, short-chat, ITL p99
    panels = [
        ("Qwen3-0.6B", [
            ("Core AI · GPU", 7.1, C_CAI_GPU),
            ("MLX", 10.2, C_MLX),
            ("Core AI · ANE", 23.6, C_CAI_ANE),
            ("CoreML-LLM · ANE", 29.4, C_COREML),
        ]),
        ("Gemma 4 E2B", [
            ("LiteRT-LM", 19.0, C_LITERT),
            ("MLX", 23.0, C_MLX),
            ("llama.cpp", 29.1, C_LLAMA),
            ("CoreML-LLM · ANE", 33.4, C_COREML),
        ]),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.6), sharex=True)
    for ax, (model, rows) in zip(axes, panels):
        ys = range(len(rows))[::-1]
        for y, (label, v, c) in zip(ys, rows):
            ax.barh(y, v, color=c, edgecolor="white", linewidth=0.6, height=0.62)
            ax.text(v + 0.4, y, f"{v:.1f} ms", va="center", fontsize=9.5, color="#222")
        ax.set_yticks(list(ys))
        ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
        ax.set_xlim(0, 38)
        ax.set_ylabel(model, fontsize=10.5, fontweight="bold")
    # 120 Hz ProMotion frame budget on the top panel
    axes[0].axvline(8.3, color="#7f1d1d", lw=1.4, ls=(0, (3, 2)))
    axes[0].text(8.7, 3.35, "120 Hz frame (8.3 ms)", fontsize=8.5, color="#7f1d1d")
    axes[1].set_xlabel("inter-token latency p99 (ms) — lower = smoother stream")
    axes[0].set_title("Streaming smoothness — iPhone 17 Pro, short-chat\n"
                      "worst-case gap between tokens (ITL p99)")
    fig.text(0.5, -0.02,
             "Core AI GPU's worst-case token gap beats a 120 Hz frame. Thermals nominal on all runs — the spread is engine cadence, not throttling.",
             ha="center", fontsize=8.5, color="#555")
    fig.savefig(OUT / "iphone_itl_jitter.png")
    print("saved:", OUT / "iphone_itl_jitter.png")


def jitter_hz():
    # Same data as jitter(), expressed as worst-case streaming rate (1000 / p99 ms).
    panels = [
        ("Qwen3-0.6B", [
            ("Core AI · GPU", 1000 / 7.1, C_CAI_GPU),
            ("MLX", 1000 / 10.2, C_MLX),
            ("Core AI · ANE", 1000 / 23.6, C_CAI_ANE),
            ("CoreML-LLM · ANE", 1000 / 29.4, C_COREML),
        ]),
        ("Gemma 4 E2B", [
            ("LiteRT-LM", 1000 / 19.0, C_LITERT),
            ("MLX", 1000 / 23.0, C_MLX),
            ("llama.cpp", 1000 / 29.1, C_LLAMA),
            ("CoreML-LLM · ANE", 1000 / 33.4, C_COREML),
        ]),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.6), sharex=True)
    for ax, (model, rows) in zip(axes, panels):
        ys = range(len(rows))[::-1]
        for y, (label, v, c) in zip(ys, rows):
            ax.barh(y, v, color=c, edgecolor="white", linewidth=0.6, height=0.62)
            ax.text(v + 1.5, y, f"{v:.0f} Hz", va="center", fontsize=9.5, color="#222")
        ax.set_yticks(list(ys))
        ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
        ax.set_xlim(0, 160)
        ax.set_ylabel(model, fontsize=10.5, fontweight="bold")
        ax.axvline(120, color="#7f1d1d", lw=1.4, ls=(0, (3, 2)))
    axes[0].text(122, 3.35, "120 Hz display", fontsize=8.5, color="#7f1d1d")
    axes[1].set_xlabel("worst-case streaming rate, 1 ÷ p99 token gap (Hz) — higher = smoother")
    axes[0].set_title("Streaming smoothness — iPhone 17 Pro, short-chat\n"
                      "how fast tokens arrive even at the slowest moment")
    fig.text(0.5, -0.02,
             "Core AI GPU outruns the 120 Hz display at its worst token. Thermals nominal on all runs — the spread is engine cadence, not throttling.",
             ha="center", fontsize=8.5, color="#555")
    fig.savefig(OUT / "iphone_itl_jitter_hz.png")
    print("saved:", OUT / "iphone_itl_jitter_hz.png")


if __name__ == "__main__":
    battery()
    jitter()
    jitter_hz()
