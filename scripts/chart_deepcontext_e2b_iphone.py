#!/usr/bin/env python3
"""Deep-context memory chart — Gemma-4-E2B on iPhone 17 Pro, 2026-07-27 capture.

Replaces the 2026-07-18 `x_deepcontext_e2b_iphone.png`, whose two inputs are both gone:
the 92 MB bar was a post-teardown footprint sample (not the in-run peak the other bars
showed), and the "MLX pays 37-54x LiteRT's memory" caption was built on it.

TWO PANELS, and that is the point. A single memory panel cannot show the finding, which is
that the ranking *inverts* depending on which memory column you read. llama.cpp is smallest
by charged footprint and largest by residency, because its GGUF is mapped rather than wired.
The one claim that survives both counts is that LiteRT-LM is the only arm under a gigabyte
on either — so that is the headline, not a ratio.

No single ratio is used as a headline on purpose: picking one column and building a multiple
is the shape of the error this chart exists to correct.

Numbers: results/raw/2026-07-27-gemma4-e2b-protocol/ (README.md for method and caveats).
One session, one device, unplugged, context forced to 2048 on every cell.

    python3 scripts/chart_deepcontext_e2b_iphone.py
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

REPO = Path(__file__).resolve().parent.parent
OUTS = [REPO / "docs" / "charts" / "x_deepcontext_e2b_iphone.png"]

BG = "#0b1017"
CARD = "#141b26"
FG = "#e6edf3"
MUTED = "#8b98a8"
DIM = "#5f6d7e"
ACCENT = "#fbbf24"

# Categorical hues, fixed per runtime and identical across both panels — colour follows the
# entity, never its rank, so the eye can track an arm as its bar length flips between panels.
# Validated with the dataviz skill's checker on the dark surface: lightness band, chroma floor,
# normal-vision separation and contrast all PASS. MLX vs llama.cpp sits at deutan dE 7.5, the
# 6-8 floor band, which is legal only with secondary encoding — every bar is direct-labelled
# with its runtime name and value, which is that encoding.
LITERT = "#f43f5e"
LLAMA = "#0284c7"
MLX = "#8b5cf6"
COREAI = "#4d7c0f"

mpl.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "DejaVu Sans"],
    "figure.dpi": 140,
    "savefig.dpi": 140,
})

fig = plt.figure(figsize=(15.5, 9.2), facecolor=BG)
fig.patches.append(Rectangle((0, 0), 1, 1, transform=fig.transFigure,
                             facecolor=BG, zorder=-10))


def text(x, y, s, size=12, color=FG, weight="normal", ha="left", style="normal"):
    fig.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha,
             va="baseline", style=style)


def card(x, y, w, h):
    fig.patches.append(FancyBboxPatch(
        (x, y), w, h, transform=fig.transFigure,
        boxstyle="round,pad=0,rounding_size=0.012",
        facecolor=CARD, edgecolor="#1e2836", linewidth=1.0, zorder=0))


# Both panels share one scale, so a bar's length means the same thing on either side and the
# inversion is visible as a swap rather than inferred from the axis labels.
VMAX = 3400.0
LABEL_W = 0.088
TRACK_W = 0.268
BAR_H = 0.030
GAP = 0.062


def bar_row(x, cy, label, color, value, value_text, note=None, note_color=None):
    text(x, cy + BAR_H / 2 - 0.005, label, size=13, color=MUTED)
    bw = max(TRACK_W * (value / VMAX), 0.004)
    fig.patches.append(FancyBboxPatch(
        (x + LABEL_W, cy), bw, BAR_H, transform=fig.transFigure,
        boxstyle="round,pad=0,rounding_size=0.005",
        facecolor=color, edgecolor="none", zorder=2))
    text(x + LABEL_W + bw + 0.010, cy + BAR_H / 2 - 0.005, value_text,
         size=14.5, color=FG, weight="bold")
    if note:
        text(x + LABEL_W, cy - 0.024, note, size=11, color=note_color or DIM)


def absent_row(x, cy, label, color, message):
    """An arm that produced no number. Drawn as an empty dashed track, never as a zero bar —
    a zero-length bar reads as 'used no memory', which is the opposite of what happened."""
    text(x, cy + BAR_H / 2 - 0.005, label, size=13, color=MUTED)
    # A short dashed stub, with the explanation OUTSIDE it. Sizing the box to the text made
    # it as long as MLX's real 3,367 MB bar — an absence that reads as a magnitude — and
    # sizing the text to the box clipped the words against the border.
    stub = 0.034
    fig.patches.append(FancyBboxPatch(
        (x + LABEL_W, cy), stub, BAR_H, transform=fig.transFigure,
        boxstyle="round,pad=0,rounding_size=0.005",
        facecolor="none", edgecolor=color, linewidth=1.2, linestyle=(0, (4, 3)),
        zorder=2))
    text(x + LABEL_W + stub + 0.012, cy + BAR_H / 2 - 0.005, message, size=12,
         color=color, style="italic")


# ---------------------------------------------------------------- header
text(0.035, 0.940, "Deep context on iPhone: which runtime uses less memory?",
     size=30, weight="light")
text(0.035, 0.898,
     "Gemma-4-E2B  ·  iPhone 17 Pro  ·  1,098-token prompt, 256 generated  ·  "
     "context forced to 2,048 on every arm  ·  2026-07-27",
     size=12.5, color=MUTED)
text(0.035, 0.848,
     "It depends which memory you count — and the order reverses.",
     size=20, color=ACCENT)

# ---------------------------------------------------------------- panels
CX1, CX2 = 0.035, 0.512
CW = 0.453
CY = 0.335
CH = 0.455
TOP = CY + CH

for cx in (CX1, CX2):
    card(cx, CY, CW, CH)

# --- left: charged footprint -----------------------------------------
ix = CX1 + 0.028
text(ix, TOP - 0.048, "Charged footprint", size=17)
text(CX1 + CW - 0.028, TOP - 0.048, "what jetsam bills the app", size=12,
     color=MUTED, ha="right")
text(ix, TOP - 0.082, "phys_footprint — dirty + compressed pages", size=11.5, color=DIM)

y = TOP - 0.170
bar_row(ix, y, "LiteRT-LM", LITERT, 732, "732 MB", "n=24, MAD 0%")
bar_row(ix, y - GAP, "llama.cpp", LLAMA, 239, "239 MB",
        "n=20, MAD 0%  ·  smallest here, largest on the right", ACCENT)
bar_row(ix, y - 2 * GAP, "MLX", MLX, 3367, "3,367 MB", "n=8, MAD 0%")
absent_row(ix, y - 3 * GAP, "Core AI", COREAI, "cannot take the prompt")
text(ix + LABEL_W, y - 3 * GAP - 0.024,
     "iOS dynamic-KV capped at 1,024 tokens upstream (apple/coreai-models#124)",
     size=11, color=DIM)

# --- right: resident --------------------------------------------------
ix = CX2 + 0.028
text(ix, TOP - 0.048, "Resident", size=17)
text(CX2 + CW - 0.028, TOP - 0.048, "pages actually in RAM", size=12,
     color=MUTED, ha="right")
text(ix, TOP - 0.082, "resident_size — counts mapped weights the footprint does not",
     size=11.5, color=DIM)

y = TOP - 0.170
bar_row(ix, y, "LiteRT-LM", LITERT, 849, "849 MB", "n=24, MAD 0%")
bar_row(ix, y - GAP, "llama.cpp", LLAMA, 3165, "3,165 MB",
        "n=20, MAD 0%  ·  2.9 GB of GGUF, mapped not wired", ACCENT)
absent_row(ix, y - 2 * GAP, "MLX", MLX, "no rankable median")
text(ix + LABEL_W, y - 2 * GAP - 0.024,
     "1,152–3,161 MB — swings 46% between launches of the identical cell (n=2)",
     size=11, color=DIM)
absent_row(ix, y - 3 * GAP, "Core AI", COREAI, "cannot take the prompt")

# ---------------------------------------------------------------- takeaway
card(CX1, 0.180, 0.930, 0.128)
text(CX1 + 0.028, 0.268, "LiteRT-LM is the only arm under a gigabyte on both counts.",
     size=19, color=FG)
text(CX1 + 0.028, 0.231,
     "llama.cpp's 239 MB is not a small working set — it is 2.9 GB of GGUF held as mapped "
     "pages, which phys_footprint never charges for. Counted as residency it is the largest "
     "arm here.",
     size=12.5, color=MUTED)
text(CX1 + 0.028, 0.204,
     "No single ratio describes this. Any multiple you quote depends on which column you "
     "picked, and the other column reverses it.",
     size=12.5, color=ACCENT)

# ---------------------------------------------------------------- footnotes
text(0.035, 0.138,
     "Method  ·  one session, one device, unplugged, harness 2026-07-27-agreed-protocol-r2. "
     "Context forced to 2,048 tokens on every cell — but a forced context is a ceiling, not "
     "an occupancy:",
     size=11.5, color=DIM)
text(0.035, 0.116,
     "LiteRT-LM grows into its KV, so the same arm reads 497 MB at a 21-token prompt and "
     "732 MB at 1,098. A memory figure needs its prompt length quoted with it.",
     size=11.5, color=DIM)
text(0.035, 0.088,
     "n and spread  ·  footprint n=24 (LiteRT-LM) / 20 (llama.cpp) / 8 (MLX), MAD 0% on all "
     "three. Medians over runs 2–3 of each launch; run 1 is cold and",
     size=11.5, color=DIM)
text(0.035, 0.066,
     "run 4+ is a thermally degraded tail, dropped by position for every arm alike. MLX and "
     "Core AI are below the protocol's n>=7 — both only became runnable late in the session.",
     size=11.5, color=DIM)
text(0.035, 0.044,
     "Core AI's short-chat footprint swings 686 to 873 MB inside a single launch, so it "
     "carries no rankable median either.",
     size=11.5, color=DIM)

text(0.035, 0.016,
     "raw logs, per-run JSON, method and the failures behind it:  "
     "results/raw/2026-07-27-gemma4-e2b-protocol/",
     size=12, color=MUTED)

for out in OUTS:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=BG)
    print(f"wrote {out}")
