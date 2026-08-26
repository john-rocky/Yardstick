#!/usr/bin/env python3
"""Generate docs/charts/*.png from the accumulation layer — never from literals.

Reads results/summary/device-runs.csv and results/regression-reports/*/verdicts.json.
Cell values come from render_leaderboard.arm_row — the same latest-session
aggregation that renders LEADERBOARD.md — so a chart and the leaderboard cannot
show two numbers for one cell. (Until 2026-08-26 charts pooled cold runs across
sessions: a few percent off the leaderboard on Pixel cells, and wrong by 2× on
any device whose history holds pre-fix sessions — the S26 mask-artifact rows.)

Palette: dataviz-validated defaults (categorical #2a78d6/#eb6834/#1baf7a passes
CVD+normal-vision checks on the #fcfcfb surface; the contrast WARN on aqua is
relieved by direct value labels on every mark). Diverging = blue/red with a
gray near-zero band. Identity is never color-alone: every bar carries its label.
"""
import csv
import json
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_leaderboard import arm_row  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "charts")
os.makedirs(OUT, exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#333330"
MUTED = "#6f6e6a"
GRID = "#e8e7e4"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RED, NEUTRAL = "#e34948", "#b0afac"

ARM_COLOR = {  # fixed categorical order — color follows the entity
    "litert-lm": BLUE, "litert-lm-cpu": BLUE, "litert-lm-gpu": BLUE,
    "llama.cpp": ORANGE, "mlx-swift": AQUA,
}


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# Every table in this file is a per-DEVICE view (its column headers name the
# device), so every filter must pin the device, not just the platform — the
# Galaxy S26's first session (platform=android) silently pooled into the
# "Pixel 8a" columns the day it landed. Same-device-class rule.
DEV = {"mac": "Mac16,9", "ios": "iPhone18,1", "android": "Pixel 8a"}



def load_rows():
    with open(os.path.join(ROOT, "results", "summary", "device-runs.csv")) as fh:
        return list(csv.DictReader(fh))


def cell(rows, plat, rt, msub, dev=None):
    """One (device, runtime, model) cell on the LEADERBOARD basis: arm_row over
    the cell's full history (latest-session selection happens inside arm_row).
    A substring matching two artifacts would silently pool them — refuse."""
    sel = [r for r in rows
           if r["platform"] == plat and r["device"] == (dev or DEV[plat])
           and r["runtime"] == rt and msub in r["model_id"]
           and r["task"] == "short-chat"]
    ids = {r["model_id"] for r in sel}
    if len(ids) > 1:
        raise SystemExit(f"ambiguous cell {plat}/{rt}/{msub!r}: {sorted(ids)}")
    return arm_row(sel) if sel else None


def cell_num(rows, plat, rt, msub, field="cold"):
    c = cell(rows, plat, rt, msub)
    v = c and c[field]
    return f"{v:.1f}" if v else "—"


def chart_regression():
    """v0.15/v0.13-era -> v0.16.0 scored verdicts, one bar per cell (diverging)."""
    cells = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "regression-reports",
                                           "*litert-lm-v0.16.0*", "verdicts.json"))):
        d = json.load(open(f))
        plat = ("iPhone 17 Pro" if "ios" in os.path.basename(os.path.dirname(f))
                else "Mac Studio M4 Max" if "mac" in os.path.basename(os.path.dirname(f))
                else "Pixel 8a")
        for v in d["verdicts"]:
            if v["verdict"] not in ("OK", "IMPROVED", "REGRESSION"):
                continue
            delta = (v.get("anchor") or {}).get("normalized_delta_pct",
                                                v.get("raw_delta_pct"))
            model = v["cell"].split()[2].split("/")[-1].replace("-litert-lm", "")
            label = f"{plat} · {model} · {v['metric'].replace('_', ' ')}"
            cells.append((label, delta, v["verdict"]))
    cells.sort(key=lambda c: c[1])
    fig, ax = plt.subplots(figsize=(8.6, 0.52 * len(cells) + 1.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    ys = range(len(cells))
    colors = [RED if v == "REGRESSION" else BLUE if abs(d) > 3 else NEUTRAL
              for _, d, v in cells]
    ax.barh(ys, [d for _, d, _ in cells], height=0.55, color=colors,
            edgecolor=SURFACE, linewidth=2)
    ax.axvline(0, color=MUTED, linewidth=1)
    for y, (label, d, verdict) in zip(ys, cells):
        # negative bars are tiny here — putting their label left of the bar
        # collides with the y tick labels; the region right of the zero line
        # is free, so all labels sit on the right side
        ax.text(max(d, 0) + 0.4, y, f"{d:+.1f}%  {verdict}",
                va="center", ha="left", fontsize=9, color=INK)
    lo = min(d for _, d, _ in cells)
    ax.set_xlim(min(lo * 1.5, -2.5), None)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([c[0] for c in cells], fontsize=9, color=INK)
    ax.set_xlabel("decode / prefill / TTFT delta vs previous capture (%, anchor-normalized "
                  "where cross-session)", fontsize=9, color=MUTED)
    ax.set_title("LiteRT-LM v0.16.0 — release regression verdicts (scored cells only; "
                 "spread- or n-gated cells not shown)", fontsize=11, color=INK,
                 loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "v0160_regression_verdicts.png"),
                facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return len(cells)


def chart_pixel_demo():
    """Pixel 8a — one panel per model, arms compared WITHIN each model only.
    A single sorted axis across models implied a cross-model race (a 1B and a
    1.5B are not competitors); small multiples remove that reading."""
    rows = load_rows()
    models = [  # (panel title, [(arm label, runtime, model substring, quant)])
        ("DeepSeek-R1-Distill-1.5B", [
            ("llama.cpp (Q4_K_M)", "llama.cpp", "DeepSeek-R1-Distill-Qwen-1.5B-GGUF", None),
            ("LiteRT-LM cpu (INT8)", "litert-lm-cpu", "DeepSeek-R1-Distill-Qwen-1.5B", None),
            ("LiteRT-LM gpu (INT8)", "litert-lm-gpu", "DeepSeek-R1-Distill-Qwen-1.5B", None),
        ]),
        ("LFM2.5-1.2B-Instruct", [
            ("LiteRT-LM cpu (int4)", "litert-lm-cpu", "LFM2.5-1.2B", None),
            ("LiteRT-LM gpu (int4_gpu)", "litert-lm-gpu", "LFM2.5-1.2B", None),
        ]),
        ("MiniCPM5-1B", [
            ("LiteRT-LM cpu (wi4b32_wi8)", "litert-lm-cpu", "MiniCPM5-1B", None),
            ("LiteRT-LM gpu (gpu-opt)", "litert-lm-gpu", "MiniCPM5-1B", None),
        ]),
    ]

    def med(rt, msub):
        c = cell(rows, "android", rt, msub)
        return c and c["cold"]

    heights = [len(arms) for _, arms in models]
    fig, axes = plt.subplots(len(models), 1, figsize=(8.6, 0.62 * sum(heights) + 2.4),
                             dpi=200, sharex=True,
                             gridspec_kw={"height_ratios": heights, "hspace": 0.75})
    fig.patch.set_facecolor(SURFACE)
    xmax = 0
    for ax, (title, arms) in zip(axes, models):
        style_ax(ax)
        labels, vals, colors = [], [], []
        for label, rt, msub, _ in arms:
            v = med(rt, msub)
            if v:
                labels.append(label); vals.append(v); colors.append(ARM_COLOR[rt])
        ys = range(len(vals))
        ax.barh(ys, vals, height=0.6, color=colors, edgecolor=SURFACE, linewidth=2)
        for y, v in zip(ys, vals):
            ax.text(v + 0.3, y, f"{v:.1f}", va="center", fontsize=9, color=INK)
            xmax = max(xmax, v)
        ax.set_yticks(list(ys))
        ax.set_yticklabels(labels, fontsize=9, color=INK)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10, color=INK, loc="left", pad=4)
    for ax in axes:
        ax.set_xlim(0, xmax * 1.12)
    axes[-1].set_xlabel("decode tok/s — short-chat, fresh process (cold); latest capture "
                        "session per cell, the same numbers as LEADERBOARD.md",
                        fontsize=9, color=MUTED)
    fig.suptitle("Pixel 8a — arms compared within each model (each model = one "
                 "config line; recipes stated per arm)", fontsize=11, color=INK,
                 x=0.02, ha="left")
    fig.savefig(os.path.join(OUT, "pixel8a_model_demo.png"),
                facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return sum(heights)


def chart_crossarm_table():
    """The cross-platform demo table as an image (for chat posts)."""
    rows = load_rows()

    def med(plat, rt, msub, field="cold"):
        return cell_num(rows, plat, rt, msub, field)

    def ios_warm(msub, label, suffix="", runtime="litert-lm"):
        # `suffix` states a per-cell budget deviation (e.g. LFM2.5's ctx 1024,
        # LiteRT-LM#3129) and only appears once a number exists to qualify.
        c = cell(rows, "ios", runtime, msub)
        v = c and c["warm"]
        if v:
            return (label + f" {v:.1f}" + (" " + suffix if suffix else "")).strip()
        return label + " —" if c else "—"
    data = [
        ["DeepSeek-R1-1.5B", "mlx 4bit " + med("mac", "mlx-swift", "DeepSeek"),
         ios_warm("DeepSeek", "LiteRT INT8"),
         "llama Q4_K_M " + med("android", "llama.cpp", "DeepSeek"),
         "LiteRT INT8 " + med("android", "litert-lm-cpu", "DeepSeek")],
        ["", "LiteRT INT8 " + med("mac", "litert-lm", "DeepSeek"), "", "",
         "LiteRT gpu " + med("android", "litert-lm-gpu", "DeepSeek")],
        # LFM2.5 iPhone runs at context-tokens=1024 (the file's exported prefill
        # plan; the 08-24 engine-create failure was this harness's own
        # max_num_tokens config, not the runtime — LiteRT-LM#3129, corrected
        # in matrices/lu-focus-litert-ios.cells).
        ["LFM2.5-1.2B", "—",
         ios_warm("LFM2.5", "LiteRT int4_gpu", "(ctx 1024)") + "\n" +
         ios_warm("lfm2.5-1.2b", "Core AI", "(S=1 export)", runtime="core-ai"), "—",
         "LiteRT int4 cpu " + med("android", "litert-lm-cpu", "LFM2.5") +
         " / gpu " + med("android", "litert-lm-gpu", "LFM2.5")],
        ["MiniCPM5-1B", "—",
         ios_warm("MiniCPM", "LiteRT gpu-opt") + "\n" +
         ios_warm("minicpm5-1b", "Core AI", runtime="core-ai"), "—",
         "LiteRT cpu " + med("android", "litert-lm-cpu", "MiniCPM") +
         " / gpu-opt " + med("android", "litert-lm-gpu", "MiniCPM")],
        ["Gemma-4-E2B", "LiteRT wNa8o8 " + med("mac", "litert-lm", "gemma-4-E2B", "warm"),
         ios_warm("gemma-4-E2B", "LiteRT wNa8o8"), "—", "—"],
        ["Qwen3-0.6B", "mlx " + med("mac", "mlx-swift", "Qwen3-0.6B", "warm"),
         ios_warm("Qwen3-0.6B", "LiteRT"),
         "llama " + med("android", "llama.cpp", "Qwen3-0.6B"),
         "LiteRT gpu " + med("android", "litert-lm-gpu", "Qwen3-0.6B")],
        ["", "", ios_warm("qwen3-0.6b", "Core AI", runtime="core-ai"), "", ""],
    ]
    cols = ["model", "Mac Studio M4 Max", "iPhone 17 Pro", "Pixel 8a (llama.cpp)",
            "Pixel 8a (LiteRT-LM)"]
    fig, ax = plt.subplots(figsize=(12.5, 0.55 * len(data) + 1.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.axis("off")
    t = ax.table(cellText=data, colLabels=cols, loc="center", cellLoc="left")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.6)
    for (r, c), tc in t.get_celld().items():
        tc.set_edgecolor(GRID)
        tc.set_text_props(color=INK)
        if r == 0:
            tc.set_text_props(color=MUTED, fontweight="bold")
            tc.set_facecolor("#f0efec")
        else:
            tc.set_facecolor(SURFACE)
    ax.set_title("decode tok/s per arm (warm where the protocol defines it, else cold; "
                 "each cell states its recipe — cross-recipe cells are different\n"
                 "deployment profiles, not one race. Cell values are LEADERBOARD.md's: "
                 "latest capture session per cell, never pooled across sessions — "
                 "warm = median of same-session warm runs, cold = that session's last cold run)",
                 fontsize=10, color=INK, loc="left", pad=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "crossarm_table.png"),
                facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def chart_demo_models_table():
    """Just the three demo models, every arm that has a number (for chat posts)."""
    rows = load_rows()

    def med(plat, rt, msub):
        # cold (fresh-process) — LEADERBOARD basis via cell(), the same as the
        # crossarm table's Mac/Pixel cells.
        return cell_num(rows, plat, rt, msub)

    def ios_warm(msub, suffix="", runtime="litert-lm"):
        # `suffix` states a per-cell budget deviation and only appears with a
        # number (LFM2.5's ctx 1024 — LiteRT-LM#3129).
        c = cell(rows, "ios", runtime, msub)
        v = c and c["warm"]
        if v:
            return f"{v:.1f}" + (" " + suffix if suffix else "")
        return "—"

    data = [
        ["DeepSeek-R1-Distill-1.5B",
         "mlx 4bit " + med("mac", "mlx-swift", "DeepSeek") +
         "\nLiteRT INT8 " + med("mac", "litert-lm", "DeepSeek"),
         "LiteRT INT8 " + ios_warm("DeepSeek") + "\nCore AI — (bundle pending export)",
         "llama.cpp Q4_K_M " + med("android", "llama.cpp", "DeepSeek"),
         "cpu " + med("android", "litert-lm-cpu", "DeepSeek") +
         " / gpu " + med("android", "litert-lm-gpu", "DeepSeek") + "  (INT8)"],
        # LFM2.5 iPhone runs at context-tokens=1024 (exported prefill plan;
        # 08-24's failure was the harness's own max_num_tokens config —
        # LiteRT-LM#3129, corrected in matrices/lu-focus-litert-ios.cells).
        # LFM Core AI: our adapter's ShortConv-hybrid binding limitation
        # (exclude row in matrices/lu-focus-litert-ios.cells), not the runtime's.
        ["LFM2.5-1.2B-Instruct", "—",
         "LiteRT int4_gpu " + ios_warm("LFM2.5", "(ctx 1024)") +
         "\nCore AI " + ios_warm("lfm2.5-1.2b", "(S=1 export)", runtime="core-ai"), "—",
         "cpu " + med("android", "litert-lm-cpu", "LFM2.5") + " (int4) / gpu " +
         med("android", "litert-lm-gpu", "LFM2.5") + " (int4_gpu)"],
        ["MiniCPM5-1B", "—",
         "LiteRT gpu-opt " + ios_warm("MiniCPM") +
         "\nCore AI INT8 " + ios_warm("minicpm5-1b", runtime="core-ai"), "—",
         "cpu " + med("android", "litert-lm-cpu", "MiniCPM") +
         " (wi4b32_wi8) / gpu " + med("android", "litert-lm-gpu", "MiniCPM") +
         " (gpu-opt)"],
    ]
    cols = ["model", "Mac Studio M4 Max", "iPhone 17 Pro",
            "Pixel 8a — llama.cpp", "Pixel 8a — LiteRT-LM"]
    fig, ax = plt.subplots(figsize=(14.8, 3.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.axis("off")
    t = ax.table(cellText=data, colLabels=cols, loc="center", cellLoc="left",
                 colWidths=[0.15, 0.19, 0.18, 0.16, 0.32])
    t.auto_set_font_size(False)
    t.set_fontsize(9.5)
    t.scale(1, 2.4)
    for (r, c), tc in t.get_celld().items():
        tc.set_edgecolor(GRID)
        tc.set_text_props(color=INK)
        if r == 0:
            tc.set_text_props(color=MUTED, fontweight="bold")
            tc.set_facecolor("#f0efec")
        else:
            tc.set_facecolor(SURFACE)
    ax.set_title("Three models added by one config line each — decode tok/s, short-chat, "
                 "fresh process (iPhone: warm in-process runs).\nRecipes differ per cell "
                 "and are stated in it; cross-recipe cells are deployment profiles, not "
                 "one race. Cell values are LEADERBOARD.md's: latest capture session per "
                 "cell — warm = same-session median, cold = the session's last cold run.",
                 fontsize=10, color=INK, loc="left", pad=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "demo_models_table.png"),
                facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    n1 = chart_regression()
    n2 = chart_pixel_demo()
    chart_crossarm_table()
    chart_demo_models_table()
    print(f"wrote {OUT}: v0160_regression_verdicts.png ({n1} cells), "
          f"pixel8a_model_demo.png ({n2} bars), crossarm_table.png")
