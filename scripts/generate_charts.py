#!/usr/bin/env python3
"""
Yardstick — generate the README hero charts from results/raw/*.jsonl.

Outputs to docs/charts/:

    decode_tok_per_s.png     # MLX vs llama.cpp vs CoreML, 5 models, M4 Max
    energy_per_token.png     # 4 backends, Gemma 4 E2B sustained, M4 Max
    itl_jitter.png           # Inter-token p99 ms, 4 backends
    tradeoff.png             # tok/s × J/tok scatter, the Pareto picture

Regenerate after every results/raw/ change:

    python scripts/generate_charts.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "results" / "raw"
OUT = REPO / "docs" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

# House style — slightly opinionated, but consistent across the chart set.
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "x",
    "grid.alpha": 0.3,
    "grid.linestyle": "-",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlepad": 14,
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "savefig.bbox": "tight",
})

PALETTE = {
    "mlx-swift": "#7c3aed",   # MLX = violet
    "llama.cpp": "#0ea5e9",   # llama.cpp = sky
    "coreml-llm": "#f59e0b",  # CoreML/ANE = amber
    "apple-fm":  "#10b981",   # Apple FM = emerald
}

LOGICAL_MODEL_ORDER = [
    "Qwen 2.5 0.5B",
    "Qwen 3.5 0.8B",
    "Qwen 3.5 2B",
    "Gemma 4 E2B",
    "Gemma 4 E4B",
]

LOGICAL_MODEL_MATCH = [
    ("Qwen 2.5 0.5B", ("qwen2.5-0.5b",)),
    ("Qwen 3.5 0.8B", ("qwen3.5-0.8b",)),
    ("Qwen 3.5 2B",   ("qwen3.5-2b",)),
    ("Gemma 4 E2B",   ("gemma-4-e2b", "gemma4-e2b")),
    ("Gemma 4 E4B",   ("gemma-4-e4b", "gemma4-e4b")),
]


def logical_model(model_id: str) -> str | None:
    s = model_id.lower()
    for label, needles in LOGICAL_MODEL_MATCH:
        if any(n in s for n in needles):
            return label
    return None


def load_runs(device: str, task: str | None = None) -> list[dict]:
    out = []
    for p in sorted(RAW.glob(f"{device}-*.jsonl")):
        # Two on-disk shapes coexist: pretty-printed single records (multi-line JSON,
        # the import_to_flat convention) and true JSONL where repeated measure_energy
        # passes APPEND one-line records. Whole-file parse first; fall back to the last
        # non-empty line for appended JSONL.
        try:
            text = p.read_text()
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                obj = json.loads([l for l in text.splitlines() if l.strip()][-1])
        except Exception:
            continue
        if task and obj.get("task") != task:
            continue
        out.append(obj)
    return out


def median(values: list[float]) -> float | None:
    xs = [v for v in values if v is not None]
    return statistics.median(xs) if xs else None


# ------------------------------------------------------------------ #
#  Chart 1 — decode tok/s, 3 runtimes × 5 models, M4 Max, short-chat
# ------------------------------------------------------------------ #


def chart_decode_tok_per_s():
    runs = load_runs("m4max", task="short-chat")
    cells: dict[tuple[str, str], list[float]] = {}
    for r in runs:
        lm = logical_model((r.get("model") or {}).get("id", ""))
        if lm is None:
            continue
        rt = r.get("runtime")
        if rt == "apple-fm":  # Apple FM has its own model, not a peer here
            continue
        cells.setdefault((lm, rt), []).append(r["metrics"]["decodeTokensPerSecond"])

    runtimes = ["mlx-swift", "llama.cpp", "coreml-llm"]
    x = list(range(len(LOGICAL_MODEL_ORDER)))
    width = 0.27

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for i, rt in enumerate(runtimes):
        ys = [median(cells.get((lm, rt), [])) or 0 for lm in LOGICAL_MODEL_ORDER]
        offsets = [xi + (i - 1) * width for xi in x]
        bars = ax.bar(offsets, ys, width, label=rt, color=PALETTE[rt], edgecolor="white", linewidth=0.5)
        for bar, y in zip(bars, ys):
            if y > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, y + 6, f"{y:.0f}",
                        ha="center", va="bottom", fontsize=8, color="#222")

    ax.set_xticks(x)
    ax.set_xticklabels(LOGICAL_MODEL_ORDER)
    ax.set_ylabel("Decode tok/s (median, n=3)")
    ax.set_title("Decode throughput — M4 Max, short-chat (128 tok)")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.text(0.5, -0.04,
             "MLX-Swift wins every cell (1.4×–1.8× over llama.cpp) after early-2026 mlx-swift-lm Qwen + Gemma kernels landed.",
             ha="center", fontsize=8.5, color="#555")
    plt.savefig(OUT / "decode_tok_per_s.png")
    plt.close(fig)
    print(f"wrote {OUT / 'decode_tok_per_s.png'}")


# ------------------------------------------------------------------ #
#  Chart 2 — J/token, 4 runtimes, Gemma 4 E2B sustained, M4 Max
# ------------------------------------------------------------------ #


def chart_energy_per_token():
    runs = load_runs("m4max")
    rows: dict[str, dict] = {}
    for r in runs:
        m = r.get("metrics") or {}
        if m.get("energySource") != "powermetrics":
            continue
        if r.get("task") != "sustained-generation":
            continue
        rt = r.get("runtime")
        rows[rt] = {
            "j_per_tok": m.get("energyJoulesPerToken") or 0,
            "j_total": m.get("energyJoules") or 0,
            "avg_w": m.get("averagePackagePowerW") or 0,
        }

    order = ["apple-fm", "mlx-swift", "llama.cpp", "coreml-llm"]
    labels = [r for r in order if r in rows]
    j_per_tok = [rows[r]["j_per_tok"] for r in labels]

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    colors = [PALETTE[r] for r in labels]
    bars = ax.barh(range(len(labels)), j_per_tok, color=colors, edgecolor="white", linewidth=0.6)
    for bar, v, r in zip(bars, j_per_tok, labels):
        ax.text(v + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f} J/tok  ·  {rows[r]['avg_w']:.1f} W avg",
                va="center", fontsize=9.5, color="#222")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Joules per generated token (lower = better)")
    ax.set_xlim(0, max(j_per_tok) * 1.45)
    ax.set_title("Energy per token — M4 Max, Gemma 4 E2B, sustained-512")
    fig.text(0.5, -0.06,
             "Apple FM ≈ 2× more efficient than the GPU backends; CoreML/ANE's low W is offset by slower decode → 4× the energy.",
             ha="center", fontsize=8.5, color="#555")
    plt.savefig(OUT / "energy_per_token.png")
    plt.close(fig)
    print(f"wrote {OUT / 'energy_per_token.png'}")


# ------------------------------------------------------------------ #
#  Chart 3 — ITL p99, 4 runtimes, Gemma 4 E2B short-chat, M4 Max
# ------------------------------------------------------------------ #


def chart_itl_jitter():
    runs = load_runs("m4max", task="short-chat")
    cells: dict[str, list[float]] = {}
    for r in runs:
        m = r.get("metrics") or {}
        p99 = m.get("interTokenLatencyP99MS")
        if p99 is None:
            continue
        rt = r.get("runtime")
        lm = logical_model((r.get("model") or {}).get("id", ""))
        if rt == "apple-fm" or lm == "Gemma 4 E2B":
            cells.setdefault(rt, []).append(p99)

    order = ["mlx-swift", "llama.cpp", "coreml-llm", "apple-fm"]
    labels = [r for r in order if r in cells]
    ys = [median(cells[r]) for r in labels]

    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    colors = [PALETTE[r] for r in labels]
    bars = ax.barh(range(len(labels)), ys, color=colors, edgecolor="white", linewidth=0.6)
    for bar, v in zip(bars, ys):
        ax.text(v + 3, bar.get_y() + bar.get_height() / 2,
                f"{v:.0f} ms", va="center", fontsize=9.5, color="#222")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Inter-token latency p99 (ms) — lower = smoother stream")
    ax.set_xlim(0, max(ys) * 1.2)
    ax.set_title("Streaming smoothness — M4 Max, Gemma 4 E2B (Apple FM uses own model)")
    fig.text(0.5, -0.05,
             "Apple FM streams in word-sized bursts (200 ms p99) vs MLX/llama.cpp's per-token cadence (5–10 ms). Same avg tok/s, very different chat UX.",
             ha="center", fontsize=8.5, color="#555")
    plt.savefig(OUT / "itl_jitter.png")
    plt.close(fig)
    print(f"wrote {OUT / 'itl_jitter.png'}")


# ------------------------------------------------------------------ #
#  Chart 4 — tok/s vs J/tok scatter (the Pareto picture)
# ------------------------------------------------------------------ #


def chart_tradeoff():
    """M4 Max Gemma-4-E2B: decode speed x decode-window J/token, one point per BUILD.

    2026-07-19 pass only (warm loads, contamination-checked) — older energy rows used a
    different window convention and session, and cross-session pooling is banned. Prefers
    energyJoulesPerTokenDecode; falls back to the full-window value (core-ai's --cmd row,
    where S=1 stepping makes the whole window decode-like — annotated).
    """
    runs = load_runs("m4max")
    CORE_AI = "#65a30d"
    # (id, label, color, label_offset_pts, ha) — offsets hand-placed so the upper-right
    # cluster (PTQ/OptiQ/litert/llama within 50 tok/s x 0.08 J/tok) doesn't collide.
    BUILDS = [
        ("mlx-community/gemma-4-e2b-it-4bit",            "MLX PTQ 4-bit",     PALETTE["mlx-swift"], (8, 10), "left"),
        ("mlx-community/gemma-4-e2b-it-qat-OptiQ-4bit",  "MLX QAT OptiQ",     PALETTE["mlx-swift"], (-10, 10), "right"),
        ("litert-community/gemma-4-E2B-it-litert-lm",    "LiteRT wNa8o8 (WebGPU path)", "#e11d48", (12, -2), "left"),
        ("unsloth/gemma-4-E2B-it-GGUF/Q4_K_M",           "llama.cpp Q4_K_M",  PALETTE["llama.cpp"], (-10, -24), "right"),
        ("core-ai/gemma4-e2b-gpu",                       "Core AI own int4\n(patched, S=1 window)", CORE_AI, (8, 10), "left"),
    ]
    points: dict[str, tuple[float, float, str, tuple, str]] = {}
    for r in runs:
        m = r.get("metrics") or {}
        if m.get("energySource") != "powermetrics":
            continue
        if not str(r.get("timestamp", "")).startswith("2026-07-19"):
            continue
        mid = (r.get("model") or {}).get("id")
        for bid, label, color, off, ha in BUILDS:
            if mid == bid:
                jt = m.get("energyJoulesPerTokenDecode") or m.get("energyJoulesPerToken")
                tok_s = m.get("decodeTokensPerSecond") or (
                    (m.get("generatedTokenCount") or 0)
                    / (m.get("energyMeasurementWindowSeconds") or 1))
                if jt:
                    points[label] = (tok_s, jt, color, off, ha)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for label, (tok_s, j_per_tok, color, off, ha) in points.items():
        ax.scatter(tok_s, j_per_tok, s=240, color=color, edgecolor="white",
                   linewidth=1.4, zorder=5)
        ax.annotate(label, (tok_s, j_per_tok), textcoords="offset points",
                    xytext=off, ha=ha, fontsize=10, fontweight="bold", color="#222")
        ax.annotate(f"{j_per_tok:.3f} J/tok · {tok_s:.0f} tok/s",
                    (tok_s, j_per_tok), textcoords="offset points",
                    xytext=(off[0], off[1] - 12 if off[1] <= 0 else -16), ha=ha,
                    fontsize=8.5, color="#555")

    ax.set_xlabel("Decode throughput (tok/s) — right = faster")
    ax.set_ylabel("Energy per token (J/tok, decode window) — down = more efficient")
    ax.set_title("Throughput × Energy — M4 Max, Gemma 4 E2B, best-available builds (2026-07-19)")
    ax.invert_yaxis()
    ax.set_xlim(0, max((pt[0] for pt in points.values()), default=1) * 1.28)
    ax.set_ylim(max((pt[1] for pt in points.values()), default=1) * 1.18, 0)
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    fig.text(0.5, -0.02,
             "MLX owns the Mac energy Pareto (fastest AND most efficient). Mac LiteRT runs the WebGPU→Metal path — "
             "a different efficiency class from the iPhone's native path; the int8-activation energy question needs the iPhone battery bench.",
             ha="center", fontsize=8.5, color="#555")
    plt.tight_layout()
    plt.savefig(OUT / "tradeoff.png")
    plt.close(fig)
    print(f"wrote {OUT / 'tradeoff.png'}")


# ------------------------------------------------------------------ #
#  Chart 0 — iPhone 17 Pro headline: decode tok/s + peak memory
# ------------------------------------------------------------------ #

def chart_iphone():
    """Gemma 4 E2B — every runtime at its best AVAILABLE build (2026-07-18 session).

    Three panels (one measure per axis — never dual-axis): decode, peak memory, GSM8K.
    Bar rows are fixed across panels, sorted by decode. Color follows the RUNTIME entity
    (both MLX builds share the MLX violet; the PTQ build carries a hatch as the secondary
    encoding). Palette adjacency validated (dataviz six checks): the sky-blue contrast
    WARN is relieved by direct value labels on every bar + the README table view.
    GSM8K constants come from the cross-runtime harness reports
    (hf-to-litertlm/reports/parity/, n=100, one identical protocol per row).
    """
    runs = load_runs("iphone17pro", task="short-chat")

    CORE_AI = "#65a30d"   # lime — distinct from every neighbour (validated)
    CACTUS = "#0f766e"    # deep teal — adjacency re-validated 2026-07-20 (worst pair vs lime ΔE 19.9)
    ROWS = [
        # (model.id, label, color, hatch, gsm8k, mem_note, j_per_tok, jtok_note)
        ("litert-community/gemma-4-E2B-it-litert-lm",
         "LiteRT-LM\nwNa8o8 QAT (official)", "#e11d48", None, 86.0, "", 0.122, ""),
        ("Cactus-Compute/gemma-4-E2B-it-cq4-uncalibrated",
         "Cactus ¶\nCQ4 uncalibrated (pre-07-09)", CACTUS, None, 87.0, "", 0.322, ""),
        ("mlx-community/gemma-4-e2b-it-4bit",
         "MLX-Swift\nPTQ 4-bit", PALETTE["mlx-swift"], "//", 84.0, "", 0.151, ""),
        ("unsloth/gemma-4-E2B-it-GGUF/Q4_K_M",
         "llama.cpp\nQ4_K_M (PTQ)", PALETTE["llama.cpp"], None, 76.0, "†", 0.483, ""),
        ("mlx-community/gemma-4-e2b-it-qat-OptiQ-4bit",
         "MLX-Swift\nQAT OptiQ int4", PALETTE["mlx-swift"], None, 91.0, "", 0.207, ""),
        ("core-ai/gemma4-e2b-gpu",
         "Core AI ‡\nown int4 (QAT q4_0)", CORE_AI, None, 88.0, "†", 0.352, "◊"),
    ]

    dec: dict = {}
    mem: dict = {}
    for r in runs:
        mid = r["model"]["id"]
        dec.setdefault(mid, []).append(r["metrics"]["decodeTokensPerSecond"])
        mem.setdefault(mid, []).append(r["metrics"]["memoryPeakDuringDecodeMB"])

    labels = [row[1] for row in ROWS]
    colors = [row[2] for row in ROWS]
    hatches = [row[3] for row in ROWS]
    dec_v = [median(dec.get(row[0], [])) or 0 for row in ROWS]
    mem_v = [median(mem.get(row[0], [])) or 0 for row in ROWS]
    gsm_v = [row[4] for row in ROWS]
    mem_note = [row[5] for row in ROWS]
    jt_v = [row[6] for row in ROWS]
    jt_note = [row[7] for row in ROWS]

    ys = list(range(len(ROWS)))[::-1]  # top row first
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15.5, 4.6),
                                             gridspec_kw={"width_ratios": [1.3, 1, 1, 1]})

    def hbars(ax, vals, fmt, title, notes=None):
        for y, v, c, h in zip(ys, vals, colors, hatches):
            ax.barh([y], [v], 0.62, color=c, hatch=h, edgecolor="white", linewidth=0.8)
            note = (notes[ys.index(y)] if notes else "")
            ax.text(v, y, " " + fmt.format(v) + note, va="center", ha="left",
                    fontsize=10, fontweight="bold", color="#222")
        ax.set_yticks(ys)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.grid(True, axis="x", alpha=0.25)
        ax.set_axisbelow(True)
        ax.margins(x=0.22)
        ax.spines[["top", "right"]].set_visible(False)

    hbars(ax1, dec_v, "{:.1f}", "Decode tok/s   ↑ better")
    ax1.set_yticklabels(labels, fontsize=9.5)
    # the unloadable row is a result, not an omission — keep it visible
    ax1.text(0.02, -0.72, "official QAT q4_0 GGUF: unloadable (llama.cpp vocab assert)",
             fontsize=8.5, color="#888", style="italic")
    ax1.set_ylim(-1.05, len(ROWS) - 0.4)

    hbars(ax2, mem_v, "{:.0f}", "Peak memory MB   ↓ better", notes=mem_note)
    ax2.set_yticklabels([])
    ax2.set_ylim(-1.05, len(ROWS) - 0.4)

    hbars(ax3, gsm_v, "{:.1f}", "GSM8K % (n=100)   ↑ better")
    ax3.set_yticklabels([])
    ax3.set_xlim(0, 100)
    ax3.set_ylim(-1.05, len(ROWS) - 0.4)

    hbars(ax4, jt_v, "{:.3f}", "Energy J/token   ↓ better", notes=jt_note)
    ax4.set_yticklabels([])
    ax4.set_ylim(-1.05, len(ROWS) - 0.4)

    fig.suptitle(
        "Gemma 4 E2B on iPhone 17 Pro (A19 Pro) — every runtime at its best available build"
        " · short-chat, median of 3 thermal-nominal cold runs · 2026-07-18 (Cactus 07-20, anchor-bridged)",
        fontsize=12, fontweight="bold", y=1.04)
    fig.text(0.5, -0.06,
             "† mmap'd weights (not comparable with wired-memory rows)   ·   ‡ patched engine (reference): Apple ships no Gemma-4 bundle"
             "   ·   ◊ shallow-rep lower bound (depth wall)   ·   GSM8K on M4 Max, one harness per row   ·   J/tok = battery-delta, 600 s sustained, unplugged",
             ha="center", fontsize=8.5, color="#666")
    plt.tight_layout()
    plt.savefig(OUT / "iphone_gemma4_bestavailable.png")
    plt.close(fig)
    print(f"wrote {OUT / 'iphone_gemma4_bestavailable.png'}")


# ------------------------------------------------------------------ #
#  Chart 5/6 — iPhone 17 Pro energy: J/token bar + speed×efficiency scatter
# ------------------------------------------------------------------ #

IPHONE_COLOR = {
    "litert-lm": "#e11d48", "mlx-swift": PALETTE["mlx-swift"],
    "llama.cpp": PALETTE["llama.cpp"], "coreml-llm": PALETTE["coreml-llm"],
}
IPHONE_RT_LABEL = {
    "litert-lm": "LiteRT-LM", "mlx-swift": "MLX-Swift",
    "llama.cpp": "llama.cpp", "coreml-llm": "CoreML/ANE",
}


def iphone_energy_points(model: str = "Gemma 4 E2B") -> dict[str, dict]:
    """Median energy stats per runtime from iphone17pro `energy`-task runs.

    Returns {} when no battery-delta energy rows exist yet, so the chart
    functions can skip cleanly until the measurements land."""
    runs = load_runs("iphone17pro", task="energy")
    agg: dict[str, dict[str, list[float]]] = {}
    for r in runs:
        m = r.get("metrics") or {}
        if m.get("energySource") != "battery-1pct":
            continue
        if m.get("energyJoulesPerToken") is None:
            continue
        if logical_model((r.get("model") or {}).get("id", "")) != model:
            continue
        rt = r.get("runtime")
        d = agg.setdefault(rt, {"jpt": [], "tok_s": [], "w": [], "tok_wh": []})
        d["jpt"].append(m["energyJoulesPerToken"])
        d["tok_s"].append(m.get("decodeTokensPerSecond") or 0)
        d["w"].append(m.get("averagePackagePowerW") or 0)
        j, gt = m.get("energyJoules") or 0, m.get("generatedTokenCount") or 0
        if j > 0 and gt > 0:
            d["tok_wh"].append(gt * 3600.0 / j)
    out: dict[str, dict] = {}
    for rt, d in agg.items():
        out[rt] = {
            "jpt": median(d["jpt"]), "tok_s": median(d["tok_s"]),
            "w": median(d["w"]),
            "tok_wh": median(d["tok_wh"]) if d["tok_wh"] else None,
        }
    return out


def chart_iphone_energy():
    pts = iphone_energy_points()
    if not pts:
        print("skip iphone_energy_per_token (no battery-delta energy rows yet)")
        return
    order = ["litert-lm", "mlx-swift", "llama.cpp", "coreml-llm"]
    labels = [r for r in order if r in pts]
    jpt = [pts[r]["jpt"] for r in labels]

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    colors = [IPHONE_COLOR[r] for r in labels]
    bars = ax.barh(range(len(labels)), jpt, color=colors, edgecolor="white", linewidth=0.6)
    for bar, v, r in zip(bars, jpt, labels):
        tw = pts[r]["tok_wh"]
        tail = f"  ·  {pts[r]['w']:.1f} W avg" + (f"  ·  {tw:.0f} tok/Wh" if tw else "")
        ax.text(v + max(jpt) * 0.015, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f} J/tok{tail}", va="center", fontsize=9.5, color="#222")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([IPHONE_RT_LABEL[r] for r in labels], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Joules per generated token (lower = better)")
    ax.set_xlim(0, max(jpt) * 1.6)
    ax.set_title("Energy per token — iPhone 17 Pro, Gemma 4 E2B, sustained (battery-delta)")
    fig.text(0.5, -0.06,
             "iPhone battery-delta (1% steps), unplugged. Whole-system incl. display — compare runtimes, not against the Mac powermetrics rows.",
             ha="center", fontsize=8.5, color="#555")
    plt.savefig(OUT / "iphone_energy_per_token.png")
    plt.close(fig)
    print(f"wrote {OUT / 'iphone_energy_per_token.png'}")


def chart_iphone_tradeoff():
    pts = iphone_energy_points()
    if not pts:
        print("skip iphone_tradeoff (no battery-delta energy rows yet)")
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for rt, d in pts.items():
        ax.scatter(d["tok_s"], d["jpt"], s=240, color=IPHONE_COLOR.get(rt, "#888"),
                   edgecolor="white", linewidth=1.4, zorder=5)
        ax.annotate(IPHONE_RT_LABEL.get(rt, rt), (d["tok_s"], d["jpt"]),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=11, fontweight="bold", color="#222")
        ax.annotate(f"{d['jpt']:.2f} J/tok · {d['tok_s']:.0f} tok/s · {d['w']:.1f} W",
                    (d["tok_s"], d["jpt"]), textcoords="offset points",
                    xytext=(8, -14), fontsize=8.5, color="#555")

    ax.set_xlabel("Sustained decode (tok/s) — right = faster")
    ax.set_ylabel("Energy per token (J/tok) — down = more efficient")
    ax.set_title("Throughput × Energy — iPhone 17 Pro, Gemma 4 E2B (battery-delta)")
    ax.invert_yaxis()
    ax.set_xlim(0, max(d["tok_s"] for d in pts.values()) * 1.25)
    ax.set_ylim(max(d["jpt"] for d in pts.values()) * 1.2, 0)
    fig.text(0.5, -0.02,
             "Does the low-power ANE win J/token despite slow decode, or does GPU speed pay for itself? The phone's answer.",
             ha="center", fontsize=8.5, color="#555")
    plt.savefig(OUT / "iphone_tradeoff.png")
    plt.close(fig)
    print(f"wrote {OUT / 'iphone_tradeoff.png'}")




# ------------------------------------------------------------------ #
#  Chart — thinking mode: the quality crown flips
# ------------------------------------------------------------------ #

def chart_thinking():
    """GSM8K thinking OFF vs ON per arm. The LiteRT-LM pair is the 2026-08-04 v0.15.0
    capture (toggle shipped in 0.15; same-build pair, same yardstick — the v0.13.1 OFF
    row was 86.0, disclosed in footnote ①). Time-to-answer annotated under the ON bars."""
    CORE_AI = "#65a30d"
    CACTUS = "#0f766e"
    arms = [
        ("Core AI\nown int4",  CORE_AI,             88.0, 92.0, "~75 s/answer ②"),
        ("MLX\nQAT OptiQ",     PALETTE["mlx-swift"], 91.0, 90.0, "~24 s/answer"),
        ("Cactus\nCQ4 uncalibrated", CACTUS,         87.0, 87.0, "~12 s/answer (est.)"),
        ("LiteRT-LM\nwNa8o8",  "#e11d48",            89.0, 92.0, "~43 s/answer measured ①"),
    ]
    x = list(range(len(arms)))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for i, (label, color, off, on, note) in enumerate(arms):
        ax.bar(i - w/2, off, w, color=color, alpha=0.45, edgecolor="white", linewidth=0.8)
        ax.text(i - w/2, off, f"{off:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#222")
        if on is not None:
            ax.bar(i + w/2, on, w, color=color, edgecolor="white", linewidth=0.8)
            ax.text(i + w/2, on, f"{on:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#222")
            delta = on - off
            ax.annotate(f"{'+' if delta >= 0 else ''}{delta:.0f}", (i + w/2, on - 7),
                        ha="center", fontsize=10.5, fontweight="bold", color="white")
        else:
            ax.bar(i + w/2, 92, w, fill=False, edgecolor="#bbb", linestyle="--", linewidth=1.2)
            ax.text(i + w/2, 46, "no thinking\ntoggle in the\npublic API", ha="center", va="center",
                    fontsize=8.5, color="#666", style="italic")
        ax.text(i, -20.5, note, ha="center", fontsize=8.5, color="#555")
    ax.set_xticks(x)
    ax.set_xticklabels([a[0] for a in arms], fontsize=10)
    ax.set_ylabel("GSM8K % (n=100)")
    ax.set_ylim(0, 104)
    ax.grid(True, axis="y", alpha=0.25); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#888", alpha=0.45, label="thinking OFF"),
                       Patch(facecolor="#888", label="thinking ON")],
              frameon=False, loc="upper center", ncol=2, fontsize=9,
              bbox_to_anchor=(0.5, 1.005))
    ax.set_title("Thinking flips the quality crown — Gemma 4 E2B, iPhone-relevant builds",
                 fontsize=12.5, fontweight="bold", pad=10)
    fig.text(0.5, -0.05,
             "① LiteRT-LM pair captured 8/4 on the v0.15.0 release (toggle shipped in 0.15; "
             "same build, same yardstick; the v0.13.1 OFF row read 86.0)",
             ha="center", fontsize=8.5, color="#666")
    fig.text(0.5, -0.115,
             "② decode collapses 34.2→11.7 tok/s at thinking depth on iPhone (only runtime "
             "that degrades with depth)   ·   Core AI and LiteRT-LM 92.0 equal the bf16 anchor",
             ha="center", fontsize=8.5, color="#666")
    plt.tight_layout()
    plt.savefig(OUT / "iphone_gemma4_thinking.png")
    plt.close(fig)
    print(f"wrote {OUT / 'iphone_gemma4_thinking.png'}")


# ------------------------------------------------------------------ #
#  Chart — deep context (p=1024): a three-orders memory gradient
# ------------------------------------------------------------------ #

def chart_deepcontext():
    """Peak memory to hold a 1,024-token context, log scale — the axis where the arms
    separate by orders of magnitude. Decode-at-depth stays flat on every surviving arm,
    so it rides along as a bar label, not a second axis."""
    CORE_AI = "#65a30d"
    rows = [
        ("LiteRT-LM  wNa8o8",      "#e11d48",            92,   "decode ~56 tok/s", ""),
        ("llama.cpp  Q4_K_M",      PALETTE["llama.cpp"], 300,  "decode 33.9 tok/s", " †"),
        ("Cactus  CQ4 uncalibrated", "#0f766e",          1068, "decode 40.3 tok/s", ""),
        ("MLX  PTQ 4-bit",         PALETTE["mlx-swift"], 3387, "decode 47.6 tok/s", "", "//"),
        ("MLX  QAT OptiQ",         PALETTE["mlx-swift"], 4999, "decode 35.1 tok/s", "", None),
    ]
    rows = [r if len(r) == 6 else (*r, None) for r in rows]
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    ys = list(range(len(rows) + 1))[::-1]
    for y, (label, color, mem, note, dag, hatch) in zip(ys[:len(rows)], rows):
        ax.barh([y], [mem], 0.6, color=color, hatch=hatch, edgecolor="white", linewidth=0.8)
        ax.text(mem * 1.12, y, f"{mem:,} MB{dag}   ·   {note}", va="center", fontsize=9.5, color="#222")
    y_dead = ys[len(rows)]
    ax.barh([y_dead], [6440], 0.6, fill=False, edgecolor=CORE_AI, linestyle="--", linewidth=1.4)
    ax.text(6440 * 0.96, y_dead, "jetsam @ depth 384 — cannot finish  ", va="center", ha="right",
            fontsize=9.5, color=CORE_AI, fontweight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows] + ["Core AI  own int4 ‡"], fontsize=10)
    ax.set_xscale("log")
    ax.set_xlim(50, 30000)
    ax.set_xlabel("Peak memory for a 1,024-token context (MB, log scale)   ↓ better")
    ax.grid(True, axis="x", alpha=0.25); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Deep context is a memory story, not a speed story — Gemma 4 E2B, iPhone 17 Pro (p=1024/g=256)",
                 fontsize=12, fontweight="bold", pad=10)
    fig.text(0.5, -0.06,
             "decode-at-depth stays flat on every surviving runtime (MLX 46.4→47.6, OptiQ 34.8→35.1 vs their short-chat rates)"
             "   ·   † mmap'd weights   ·   ‡ patched engine (reference); dashed bar = ~6.44 GB jetsam ceiling",
             ha="center", fontsize=8.5, color="#666")
    plt.tight_layout()
    plt.savefig(OUT / "iphone_gemma4_deepcontext.png")
    plt.close(fig)
    print(f"wrote {OUT / 'iphone_gemma4_deepcontext.png'}")


def main():
    chart_iphone()
    chart_thinking()
    chart_deepcontext()
    chart_iphone_energy()
    chart_iphone_tradeoff()
    chart_decode_tok_per_s()
    chart_energy_per_token()
    chart_itl_jitter()
    chart_tradeoff()
    print(f"\nAll charts in {OUT.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
