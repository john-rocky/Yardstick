#!/usr/bin/env python3
"""X-thread hero — Gemma-4-E2B on iPhone 17 Pro, the 2026-07-28/30 fairness re-capture.

Three cards (decode / memory / energy), one fixed arm order and one fixed hue per arm
across all three so the eye can track a runtime between panels. Second builds of a
family (MLX-OptiQ, Cactus-shipped) reuse the family hue with a hatch + direct label
(composite encoding; palettes validated with the dataviz six-checks script for both
surfaces). Numbers = the audited campaign records:
  results/raw/2026-07-29-gemma4-e2b-protocol/README.md   (decode/memory, warm protocol)
  results/raw/2026-07-30-gemma4-e2b-protocol/README.md   (energy, tick-window instrument)

    python3 scripts/x_hero_e2b_iphone_chart.py
      -> docs/charts/x_hero_e2b_iphone_dark.png   (production, X)
      -> docs/charts/x_hero_e2b_iphone.png        (light)
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUTDIR = Path(__file__).resolve().parent.parent / "docs" / "charts"

THEMES = {
    "dark": dict(
        out="x_hero_e2b_iphone_dark.png",
        BG="#0b1017", CARD="#141b26", EDGE="#1e2836",
        FG="#e6edf3", MUTED="#8b98a8", DIM="#5f6d7e", ACCENT="#fbbf24",
        # validated: node validate_palette.js "#f43f5e,#7c3aed,#65a30d,#3b82f6,#0ea5a0" --mode dark
        LITERT="#f43f5e", MLX="#7c3aed", COREAI="#65a30d", LLAMA="#3b82f6", CACTUS="#0ea5a0",
    ),
    "light": dict(
        out="x_hero_e2b_iphone.png",
        BG="#ffffff", CARD="#f6f8fa", EDGE="#d8dee6",
        FG="#1a1a1a", MUTED="#57606a", DIM="#848d97", ACCENT="#b45309",
        # validated: node validate_palette.js "#e11d48,#6d28d9,#4d7c0f,#2563eb,#0d9f97" --mode light
        LITERT="#e11d48", MLX="#6d28d9", COREAI="#4d7c0f", LLAMA="#2563eb", CACTUS="#0d9f97",
    ),
}

mpl.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "DejaVu Sans"],
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "hatch.linewidth": 1.4,
})


def render(theme: dict) -> Path:
    T = theme
    fig = plt.figure(figsize=(15.5, 8.7), facecolor=T["BG"])
    fig.patches.append(Rectangle((0, 0), 1, 1, transform=fig.transFigure,
                                 facecolor=T["BG"], zorder=-10))

    def text(x, y, s, size=12, color=None, weight="normal", ha="left"):
        fig.text(x, y, s, fontsize=size, color=color or T["FG"], fontweight=weight,
                 ha=ha, va="baseline")

    def card(x, y, w, h):
        fig.patches.append(FancyBboxPatch(
            (x, y), w, h, transform=fig.transFigure,
            boxstyle="round,pad=0,rounding_size=0.012",
            facecolor=T["CARD"], edgecolor=T["EDGE"], linewidth=1.0, zorder=0))

    def bars(x, top, w, rows, vmax, gap, bar_h, label_w):
        """rows: (label, color, hatch, value|None, value_text, sub|None)."""
        track_x = x + label_w
        track_w = w - label_w - 0.052
        for i, (label, color, hatch, value, value_text, sub) in enumerate(rows):
            cy = top - i * gap
            text(x, cy + bar_h / 2 - 0.005, label, size=11.5, color=T["MUTED"])
            if value is not None:
                bw = max(track_w * (value / vmax), 0.003)
                fig.patches.append(FancyBboxPatch(
                    (track_x, cy), bw, bar_h, transform=fig.transFigure,
                    boxstyle="round,pad=0,rounding_size=0.004",
                    facecolor=color, edgecolor=T["CARD"] if hatch else "none",
                    hatch=hatch or None, linewidth=0.0, zorder=2))
                tx = track_x + bw + 0.008
            else:
                tx = track_x
            text(tx, cy + bar_h / 2 - 0.005, value_text, size=12, weight="bold")
            if sub:
                text(tx, cy + bar_h / 2 - 0.021, sub, size=9.5, color=T["DIM"])

    # ------------------------------------------------------------- header
    text(0.035, 0.930, "Gemma-4-E2B on iPhone 17 Pro — the fair re-capture", size=29,
         weight="light")
    text(0.035, 0.884,
         "7 runtimes/builds · one warm protocol (runs 2–3 median, ctx 2048, thermal-nominal, "
         "n≥6) · one prefill instrument for every arm · July 28–30, 2026",
         size=12.5, color=T["MUTED"])
    text(0.035, 0.830,
         "LiteRT-LM leads decode, memory on both columns — and J/token on the rebuilt "
         "energy instrument.",
         size=18.5, color=T["ACCENT"])

    # ------------------------------------------------------------- cards
    CY, CH = 0.115, 0.655
    CW, GAPX = 0.302, 0.012
    CX = [0.035 + i * (CW + GAPX) for i in range(3)]
    for cx in CX:
        card(cx, CY, CW, CH)
    TOP = CY + CH

    ROW_GAP, BAR_H = 0.079, 0.030
    ROWS_TOP = TOP - 0.135
    LBL_W = 0.082

    L, M, C, LL, CA = T["LITERT"], T["MLX"], T["COREAI"], T["LLAMA"], T["CACTUS"]
    H = "////"  # second build of a family

    # --- card 1: decode
    ix = CX[0] + 0.024
    iw = CW - 0.048
    text(ix, TOP - 0.048, "Decode  tok/s", size=15)
    text(CX[0] + CW - 0.024, TOP - 0.048, "warm, n≥6", size=11, color=T["DIM"], ha="right")
    bars(ix, ROWS_TOP, iw, [
        ("LiteRT-LM",    L,  None, 61.1, "61.1", "wNa8o8 QAT · n=8"),
        ("Cactus uncal", CA, None, 50.6, "50.6", "7/20 cold ×3"),
        ("Cactus ship",  CA, H,    50.0, "50.0", "GSM8K 3% — reasoning-dead"),
        ("MLX PTQ",      M,  None, 49.1, "49.1", "n=6"),
        ("Core AI",      C,  None, 47.1, "47.1", "patched engine (ref)"),
        ("llama.cpp",    LL, None, 38.8, "38.8", "Q4_K_M · 7/27 n=10"),
        ("MLX OptiQ",    M,  H,    36.0, "36.0", "GSM8K 91% — quality build"),
    ], vmax=61.1, gap=ROW_GAP, bar_h=BAR_H, label_w=LBL_W)

    # --- card 2: memory
    ix = CX[1] + 0.024
    text(ix, TOP - 0.048, "Memory  MB", size=15)
    text(CX[1] + CW - 0.024, TOP - 0.048, "median charged footprint", size=11,
         color=T["DIM"], ha="right")
    bars(ix, ROWS_TOP, iw, [
        ("LiteRT-LM",    L,  None,  497, "497", "<1 GB on residency too"),
        ("Cactus uncal", CA, None, 1061, "1,061", "7/20, peak basis"),
        ("Cactus ship",  CA, H,     632, "632", None),
        ("MLX PTQ",      M,  None, 3010, "3,010", None),
        ("Core AI",      C,  None,  755, "755", "mmap †"),
        ("llama.cpp",    LL, None,  191, "191", "mmap † — 2.9 GB resident"),
        ("MLX OptiQ",    M,  H,    4592, "4,592", None),
    ], vmax=4592, gap=ROW_GAP, bar_h=BAR_H, label_w=LBL_W)

    # --- card 3: energy
    ix = CX[2] + 0.024
    text(ix, TOP - 0.048, "Energy  J/token", size=15)
    text(CX[2] + CW - 0.024, TOP - 0.048, "tick-window, audited", size=11,
         color=T["DIM"], ha="right")
    bars(ix, ROWS_TOP, iw, [
        ("LiteRT-LM",    L,  None, 0.147, "0.147", "fastest AND most efficient"),
        ("Cactus uncal", CA, None, 0.222, "0.222", "±10% cluster ▾"),
        ("Cactus ship",  CA, H,    0.226, "0.226", "±10% cluster ▾"),
        ("MLX PTQ",      M,  None, 0.182, "0.182", None),
        ("Core AI",      C,  None, None,  "cannot", "KV cap 1,024 < budget 2,048"),
        ("llama.cpp",    LL, None, 0.260, "0.260", None),
        ("MLX OptiQ",    M,  H,    0.231, "0.231", "±10% cluster ▾"),
    ], vmax=0.260, gap=ROW_GAP, bar_h=BAR_H, label_w=LBL_W)

    # ------------------------------------------------------------- footer
    text(0.035, 0.082,
         "energy: measured BETWEEN 5%-step battery-gauge transitions (iOS 27 quantizes "
         "start/end deltas ×2 — old J/tok figures retired)",
         size=10.5, color=T["DIM"])
    text(0.035, 0.060,
         "▾ 0.222/0.226/0.231 are one unresolved cluster (±10%) · † mmap'd weights: the "
         "footprint hides file-backed pages · hatch = second build of the same runtime",
         size=10.5, color=T["DIM"])
    text(0.035, 0.038,
         "Core AI decode = patched engine reference (Apple ships no Gemma-4 bundle) · "
         "GSM8K n=100, one Mac harness for every row",
         size=10.5, color=T["DIM"])
    text(0.035, 0.013,
         "full tables, raw logs, per-cell transition timestamps, repro:  "
         "github.com/john-rocky/apple-silicon-llm-bench",
         size=12, color=T["MUTED"])

    out = OUTDIR / T["out"]
    fig.savefig(out, facecolor=T["BG"])
    plt.close(fig)
    return out


if __name__ == "__main__":
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():
        print("wrote", render(theme))
