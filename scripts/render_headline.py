#!/usr/bin/env python3
"""Render the README headline decode table from stored records.

  python3 scripts/render_headline.py          # rewrite the generated block in README.md
  python3 scripts/render_headline.py --check  # CI: fail if stale

Why: the headline table used to be typed by hand and drifted from the body and
from results/ — on 2026-09-04 an iPhone row copied from a stale local README
re-published a retracted MLX number next to a Core AI number from a different
session. The top of the README is the part that gets quoted, so it is now a
projection of stored records, like LEADERBOARD.md. Hand edits inside the
markers are overwritten.

Sources (read-only):
  results/summary/device-runs.csv   harness short-chat cells. The per-cell numbers come
                                    from render_leaderboard's latest_session / arm_row,
                                    so a number shown in both places is the same number.
  results/hybrid/*.json             the CLI hybrid-model reports (llama-bench / mlx_lm).
  results/raw/**/*.json             Apple `llm-benchmark --output-json` records (detected by
                                    shape, not path) + the `mlx_lm benchmark` sweep logs
                                    beside them (mlx_sweep_<ver>.log): the Mac 512p/1024g
                                    protocol, its own table (budget-mode-rule).

Rules applied as code (slugs: methodology/fairness-rules.md):
  cold-warm-split      warm median where the session has warm runs, else the cold
                       number tagged "cold"; never pooled.
  thermal guard (§2)   only runs that STARTED thermal-nominal count — the filter
                       bench_common.started_nominal, shared with render_leaderboard
                       and regression_diff. A session whose runs all started hot is
                       skipped and the cell falls back to its newest nominal session
                       (results/raw/2026-08-26-iphone-coreai-pairs is THERMAL_FAIL on
                       every cell by its own gate file; it must not be the headline).
                       An arm with no nominal capture at all is listed under the
                       table, not in it (the leaderboard shows it flagged ⚠hot).
  iphone-session-variance
                       every cell carries its capture date; a row whose cells come
                       from different sessions gets an automatic note, and no ratio
                       is printed anywhere in the block.
  quant-per-arm-rule   the recipe (artifact, quantization, engine, n) of every cell
                       is listed right under the table.
  stored-report-rule   a number with no record under results/ does not appear. Known
                       gaps: the hybrid-model Core AI cells (HF cards only) and the
                       macOS-26-era Qwen3-0.6B artifact (archived hash, no record).
  no-cherry-pick       every artifact of a runtime is shown, fastest first; spread
                       over SPREAD_FLAG carries the same ⚠ as LEADERBOARD.md.

HEADLINE_MODELS selects WHICH harness rows appear (this is a headline, not the
standings — LEADERBOARD.md is that); it never selects numbers.
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common import DEVICE_ALIASES, DEVICE_DISPLAY, logical_model, started_nominal  # noqa: E402
from render_leaderboard import SPREAD_FLAG, arm_row, fmt, latest_session, load  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HYBRID = os.path.join(ROOT, "results", "hybrid")
RAW = os.path.join(ROOT, "results", "raw")
TARGET = os.path.join(ROOT, "README.md")
BEGIN = "<!-- BEGIN GENERATED: scripts/render_headline.py -->"
END = "<!-- END GENERATED: scripts/render_headline.py -->"
HEADLINE_TASK = "short-chat"

# Logical names from bench_common.LOGICAL_MODELS. Order = table order.
HEADLINE_MODELS = ["Qwen 3 0.6B", "Qwen 3 8B"]
PLATFORMS = ("ios", "mac")  # Android standings: LEADERBOARD.md
RUNTIME_LABEL = {
    "core-ai": "Apple Core AI",
    "mlx-swift": "MLX",
    "llama.cpp": "llama.cpp",
    "litert-lm": "LiteRT-LM",
    "coreml-llm": "Core ML",
}
RUNTIME_ORDER = list(RUNTIME_LABEL)
SUPERSCRIPTS = "¹²³⁴⁵⁶⁷⁸⁹"
LLM_BENCHMARK_KEYS = {"averages", "generation_tokens", "model", "num_trials", "prompt_tokens", "trials"}
# llm-benchmark campaign dirs whose name carries no device label. Factual basis,
# not guesswork: the ct041 README states "Mac Studio M4 Max, macOS 27.0 26A5378j".
CAMPAIGN_DEVICE = {"2026-07-13-mac-warm/coreai-ct041": "Mac16,9"}


def device_name(model_identifier):
    return DEVICE_DISPLAY.get(model_identifier, model_identifier)


def session_key(rows):
    """Campaign dir (one sitting) or the date for legacy flat rows — the same
    session notion render_leaderboard.latest_session uses."""
    r = rows[0]
    return r["campaign"] if r["campaign"] and r["campaign"] != "flat" else (r["timestamp"] or "")[:10]


# ----------------------------------------------------------------------------
# harness cells
# ----------------------------------------------------------------------------

def harness_rows():
    """(platform, device, model) -> runtime -> [(arm dict, model_id, session)]"""
    tree = {}
    hot_only = []  # arms with no nominal-thermal capture at all (failed-runs-stay)
    cells = {}
    for r in load("device-runs.csv"):
        if r["task"] != HEADLINE_TASK or not r["model_id"] or not r["device"]:
            continue
        if (r.get("platform") or "") not in PLATFORMS:
            continue
        model = logical_model(r["model_id"])
        if model not in HEADLINE_MODELS:
            continue
        cells.setdefault((r["platform"], r["device"], model, r["runtime"], r["model_id"]), []).append(r)
    for (plat, dev, model, rt, mid), rows in cells.items():
        ok = started_nominal(rows)  # fairness §2 guard, shared with render_leaderboard
        if not ok:
            hot_only.append((plat, dev, model, rt, mid, arm_row(rows)["date"]))
            continue
        a = arm_row(ok)
        sess, _ = latest_session(ok)  # the session arm_row used, for the note
        # hot-start captures newer than the session shown: named, so the reader
        # can see why LEADERBOARD.md (no thermal gate yet) may show a later date
        skipped = sorted({(r["timestamp"] or "")[:10] for r in rows
                          if r not in ok and (r["timestamp"] or "")[:10] > a["date"]})
        tree.setdefault((plat, dev, model), {}).setdefault(rt, []).append(
            (a, mid, session_key(sess), skipped))
    return tree, hot_only


def cell_text(arms):
    parts = []
    multi = len(arms) > 1
    for a, mid, _sess, _skipped in sorted(arms, key=lambda t: -(t[0]["warm"] or t[0]["cold"] or 0)):
        if a["warm"] is not None:
            txt = f"**{fmt(a['warm'])}**"
            if a["spread"] > SPREAD_FLAG:
                txt += f" ⚠spread {a['spread']:.0f}%"
        elif a["cold"] is not None:
            txt = f"{fmt(a['cold'])} cold"
        else:
            txt = "—"
        if multi:
            txt += f" `{mid.split('/')[-1]}`"
        txt += f" ({a['date']})"
        parts.append(txt)
    return "<br>".join(parts) if parts else "—"


def recipe_line(dev, model, rt, a, mid, sess, skipped):
    if a["warm"] is not None:
        basis = f"warm median of {a['warm_n']} in-process runs"
    else:
        n = a["n"]
        basis = (f"cold, {n} fresh-process launch{'es' if n != 1 else ''} in that session "
                 "(no warm runs), last one shown")
    where = f"session {a['date']}" + (f" (`{os.path.basename(sess)}`)" if sess != a["date"] else "")
    line = (f"- {dev} · {model} · {RUNTIME_LABEL.get(rt, rt)}: `{mid}` — {a['quant']}, "
            f"engine {a['engine']}, {basis}, {where}")
    if skipped:
        line += (f"; newer capture{'s' if len(skipped) > 1 else ''} on {', '.join(skipped)} "
                 "started hot and did not qualify (thermal guard)")
    return line


def session_note(model, dev_name, by_rt, n_sessions, marker):
    by = "; ".join(f"{label} {' and '.join(dates)}" for label, dates in by_rt.items())
    return (f"{marker} {model} · {dev_name}: the cells come from {n_sessions} capture "
            f"sessions ({by}). Same device, different sittings — device state moves between "
            "sessions (measured on the phone: `results/raw/2026-07-13-mlx-variance/`), so a "
            "ratio between two cells of this row is not a measurement; compare within one "
            "session (the dated tables below are per-session).")


def render_harness(lines):
    tree, hot_only = harness_rows()
    runtimes = [rt for rt in RUNTIME_ORDER
                if any(rt in arms for arms in tree.values())]
    runtimes += sorted({rt for arms in tree.values() for rt in arms} - set(runtimes))
    order = {p: i for i, p in enumerate(PLATFORMS)}
    keys = sorted(tree, key=lambda k: (order[k[0]], k[1], HEADLINE_MODELS.index(k[2])))

    lines.append(
        f"**Harness cells** — task `{HEADLINE_TASK}` (same prompt and 128-token budget "
        "for every arm, in-tree harness, per-run JSONL under `results/raw/`). "
        "**Bold** = warm, the median of one session's in-process runs (run 1 of every "
        "launch dropped as cold); \"cold\" = fresh-process first generation, shown where "
        "a session has no warm runs. Only runs that started thermal-nominal count "
        "([fairness-rules §2](methodology/fairness-rules.md)), and each cell is its newest "
        "qualifying session — the date in parentheses. Several artifacts of one runtime "
        "are listed fastest-first, never pooled.")
    lines.append("")
    lines.append("| Model | Device | " + " | ".join(RUNTIME_LABEL.get(rt, rt) for rt in runtimes) + " |")
    lines.append("|---|---|" + "|".join("---:" for _ in runtimes) + "|")
    notes, recipes = [], []
    latest = ""
    for key in keys:
        plat, dev, model = key
        arms = tree[key]
        sessions = {}   # session -> set(runtime label)
        by_rt = {}      # runtime label -> ["date (campaign)", ...]
        for rt in runtimes:
            for a, mid, sess, skipped in arms.get(rt, []):
                label = RUNTIME_LABEL.get(rt, rt)
                sessions.setdefault(sess, set()).add(label)
                shown = a["date"] + (f" (`{os.path.basename(sess)}`)" if sess != a["date"] else "")
                if shown not in by_rt.setdefault(label, []):
                    by_rt[label].append(shown)
                latest = max(latest, a["date"])
                recipes.append(recipe_line(device_name(dev), model, rt, a, mid, sess, skipped))
        marker = ""
        if len(sessions) > 1:
            marker = " " + SUPERSCRIPTS[len(notes) % len(SUPERSCRIPTS)]
            notes.append(session_note(model, device_name(dev), by_rt, len(sessions), marker.strip()))
        cells = [cell_text(arms[rt]) if rt in arms else "—" for rt in runtimes]
        lines.append(f"| {model}{marker} | {device_name(dev)} | " + " | ".join(cells) + " |")
    lines.append("")
    for n in notes:
        lines.append(n)
        lines.append("")
    if hot_only:
        lines.append("Arms with no thermal-nominal capture (failed-runs-stay; the hot captures "
                     "are in `results/raw/` and LEADERBOARD.md):")
        lines.append("")
        for plat, dev, model, rt, mid, date in sorted(hot_only):
            lines.append(f"- {device_name(dev)} · {model} · {RUNTIME_LABEL.get(rt, rt)}: "
                         f"`{mid}` — newest capture {date} started hot")
        lines.append("")
    lines.append("<details><summary>Recipes behind the harness cells (quant-per-arm-rule)</summary>")
    lines.append("")
    lines.extend(recipes)
    lines.append("")
    lines.append("</details>")
    lines.append("")
    return latest


# ----------------------------------------------------------------------------
# Apple llm-benchmark protocol (Mac): llm-benchmark JSON + mlx_lm benchmark sweep logs
# ----------------------------------------------------------------------------

def _campaign(path):
    """results/raw/<campaign...>/file -> (campaign rel dir, date, device identifier)."""
    rel = os.path.relpath(os.path.dirname(path), RAW)
    top = rel.split(os.sep)[0]
    date = top[:10] if re.match(r"\d{4}-\d{2}-\d{2}", top) else ""
    device = CAMPAIGN_DEVICE.get(rel)
    if device is None:
        for tok in top.split("-"):
            if tok in DEVICE_DISPLAY:
                device = DEVICE_ALIASES.get(tok, tok)  # label-space -> identifier, as build_summary does
                break
    return rel, date, device or rel


def llm_benchmark_cells():
    """(device, model) -> runtime -> [(value, date, campaign, recipe)], one entry per
    campaign; the caller keeps the newest campaign per runtime."""
    cells = {}
    for path in sorted(glob.glob(os.path.join(RAW, "**", "*.json"), recursive=True)):
        try:
            d = json.load(open(path))
        except (OSError, ValueError):
            continue
        if not isinstance(d, dict) or set(d) != LLM_BENCHMARK_KEYS:
            continue
        rel, date, device = _campaign(path)
        stem = os.path.basename(path)[:-5].split("_")[0]
        model = logical_model(stem)
        if model not in HEADLINE_MODELS:
            continue
        vals = [t["gen_tps"] for t in d["trials"] if t.get("gen_tps")]
        spread = (max(vals) - min(vals)) / d["averages"]["generation_tps"] * 100 if len(vals) > 1 else 0.0
        recipe = (f"`{d['model']}` — Apple `llm-benchmark`, {d['prompt_tokens']}p/"
                  f"{d['generation_tokens']}g, mean of {d['num_trials']} trials "
                  f"(trial spread {spread:.1f}%), `results/raw/{rel}/`")
        cells.setdefault((device, model), {}).setdefault("core-ai", []).append(
            (d["averages"]["generation_tps"], date, rel, recipe))
    for path in sorted(glob.glob(os.path.join(RAW, "**", "mlx_sweep_*.log"), recursive=True)):
        rel, date, device = _campaign(path)
        ver = re.search(r"mlx_sweep_([\d.]+)\.log$", path)
        ver = ver.group(1) if ver else "?"
        repo = None
        for line in open(path):
            m = re.match(r"=== (\S+) ", line)
            if m:
                repo = m.group(1)
                continue
            m = re.match(r"Averages: .*generation_tps=([\d.]+)", line)
            if m and repo:
                model = logical_model(repo)
                if model in HEADLINE_MODELS:
                    recipe = (f"`{repo}` — `mlx_lm benchmark` (mlx-lm {ver}), same 512p/1024g/5 "
                              f"arguments, mean of the trials, `results/raw/{rel}/{os.path.basename(path)}`")
                    cells.setdefault((device, model), {}).setdefault("mlx-swift", []).append(
                        (float(m.group(1)), date, rel, recipe))
                repo = None
    return cells


def render_llm_benchmark(lines):
    cells = llm_benchmark_cells()
    if not cells:
        return ""
    runtimes = [rt for rt in RUNTIME_ORDER if any(rt in arms for arms in cells.values())]
    lines.append(
        "**Apple `llm-benchmark` protocol, Mac** — 512-token prompt, 1024 generated, 5 trials, "
        "greedy; Apple Core AI = Apple's `llm-benchmark` release build, MLX = `mlx_lm benchmark` "
        "with the same arguments. A different budget from the harness rows above "
        "(budget-mode-rule): never compare a number here with one there. Each cell is its newest "
        "campaign — the date in parentheses.")
    lines.append("")
    lines.append("| Model | Device | " + " | ".join(RUNTIME_LABEL.get(rt, rt) for rt in runtimes) + " |")
    lines.append("|---|---|" + "|".join("---:" for _ in runtimes) + "|")
    notes, recipes = [], []
    latest = ""
    for (device, model) in sorted(cells, key=lambda k: (k[0], HEADLINE_MODELS.index(k[1]))):
        arms = cells[(device, model)]
        row, sessions, by_rt = [], set(), {}
        for rt in runtimes:
            if rt not in arms:
                row.append("—")
                continue
            val, date, rel, recipe = max(arms[rt], key=lambda t: t[1])
            sessions.add(rel)
            by_rt.setdefault(RUNTIME_LABEL.get(rt, rt), []).append(f"{date} (`{rel}`)")
            latest = max(latest, date)
            row.append(f"{val:.1f} ({date})")
            recipes.append(f"- {device_name(device)} · {model} · {RUNTIME_LABEL.get(rt, rt)}: {recipe}")
        marker = ""
        if len(sessions) > 1:
            marker = " " + SUPERSCRIPTS[len(notes) % len(SUPERSCRIPTS)]
            notes.append(session_note(model, device_name(device), by_rt, len(sessions), marker.strip()))
        lines.append(f"| {model}{marker} | {device_name(device)} | " + " | ".join(row) + " |")
    lines.append("")
    for n in notes:
        lines.append(n)
        lines.append("")
    lines.append("<details><summary>Recipes behind the llm-benchmark cells</summary>")
    lines.append("")
    lines.extend(recipes)
    lines.append("")
    lines.append("</details>")
    lines.append("")
    return latest


# ----------------------------------------------------------------------------
# hybrid CLI reports
# ----------------------------------------------------------------------------

def _num(entry, *keys):
    """Pull the decode number out of the two report shapes."""
    d = entry
    for k in keys:
        if d is None:
            return None
        d = d.get(k)
    return d


def hybrid_rows():
    """[(model, device, date, {runtime: (value, recipe, caveat)}, source)]"""
    out = []
    for path in sorted(glob.glob(os.path.join(HYBRID, "*.json"))):
        d = json.load(open(path))
        src = os.path.relpath(path, ROOT)
        if "results" in d:  # nemotron3-nano shape: one entry per (model, stack)
            device = d.get("hardware", {}).get("cpu", "?")
            date = d.get("date", "")
            rows = {}
            for e in d["results"] + d.get("granite_4_0_h_tiny", []):
                stack = e["stack"]
                rt = "llama.cpp" if stack.startswith("llama.cpp") else "mlx-swift" if stack.startswith("MLX") else stack
                dec = e["decode_tps"]
                val = dec.get("median", dec.get("mean"))
                basis = (f"{dec.get('test')}, median of {len(dec['runs'])} processes" if "runs" in dec
                         else f"{dec.get('test')} mean of 3 (±{dec.get('std'):.2f})")
                recipe = f"{e['repo']} {e['quant']} ({e.get('stack_version', stack)}; {basis})"
                rows.setdefault(e["model"], {})[rt] = (val, recipe, None)
            for model, arms in rows.items():
                out.append((model, device, date, arms, src))
        elif "rows" in d:  # hybrid-apple-silicon-bench-2 shape: one entry per model
            device = re.sub(r"\s+\d+\s*GiB$", "", d.get("machine", "?").split(",")[0])
            date = d.get("measured", "")
            lc = d.get("llama_cpp", {})
            mx = d.get("mlx", {})
            for e in d["rows"]:
                arms = {}
                l = e.get("llama_cpp")
                if l:
                    arms["llama.cpp"] = (
                        l["tg256"]["avg_ts"],
                        f"{l['weights']} (llama.cpp build {lc.get('build')} {lc.get('source', '')}; "
                        f"tg256 mean of 3 (±{l['tg256']['stddev_ts']:.2f}))",
                        None)
                m = e.get("mlx")
                if m:
                    arms["mlx-swift"] = (
                        m["median_gen_tps"],
                        f"{m['weights']} (mlx {mx.get('mlx')}, mlx-lm {mx.get('mlx_lm')}; "
                        f"256-token generation, median of {len(m['runs'])} processes)",
                        m.get("coherence"))
                out.append((e["model"], device, date, arms, src))
    return out


def render_hybrid(lines):
    rows = sorted(hybrid_rows(), key=lambda r: r[2])  # stable: report date, then file order
    if not rows:
        return ""
    runtimes = [rt for rt in RUNTIME_ORDER if any(rt in r[3] for r in rows)]
    lines.append(
        "**Hybrid Mamba-2 / Transformer models, CLI runs** — llama.cpp = `llama-bench` "
        "tg256 (pp512 prompt), MLX = `mlx_lm generate` 256 tokens; greedy, batch 1, mains "
        "power, one machine. Quantization is **not** equal across columns (Q4_K_M is not "
        "4-bit affine — see the recipes and the caveats in each report); ‡ = the report "
        "records a coherence problem with that output. Reports: " +
        ", ".join(f"`{s}`" for s in sorted({r[4] for r in rows})) + ".")
    lines.append("")
    lines.append("| Model | Device | " + " | ".join(RUNTIME_LABEL.get(rt, rt) for rt in runtimes) + " | Measured |")
    lines.append("|---|---|" + "|".join("---:" for _ in runtimes) + "|---|")
    recipes = []
    latest = ""
    for model, device, date, arms, src in rows:
        latest = max(latest, date)
        cells = []
        for rt in runtimes:
            if rt not in arms:
                cells.append("—")
                continue
            val, recipe, caveat = arms[rt]
            cells.append(f"{val:.1f}" + (" ‡" if caveat else ""))
            recipes.append(f"- {model} · {RUNTIME_LABEL.get(rt, rt)}: {recipe}"
                           + (f" — ‡ {caveat}" if caveat else ""))
        lines.append(f"| {model} | {device} | " + " | ".join(cells) + f" | {date} |")
    lines.append("")
    lines.append("<details><summary>Recipes behind the hybrid cells</summary>")
    lines.append("")
    lines.extend(recipes)
    lines.append("")
    lines.append("</details>")
    lines.append("")
    return latest


# ----------------------------------------------------------------------------

def generate():
    body = []
    latest = max(render_harness(body), render_llm_benchmark(body), render_hybrid(body))
    head = [
        BEGIN,
        "",
        "**Decode tok/s, batch 1, greedy.** Generated from stored records by "
        f"`scripts/render_headline.py` (newest record {latest or 'n/a'}) — do not edit "
        "inside the markers. Full standings with prefill, TTFT, memory and GSM8K: "
        "[LEADERBOARD.md](LEADERBOARD.md); every capture: [RESULTS.md](RESULTS.md) and "
        "`results/`. No cross-cell ratios are printed here on purpose: cells are "
        "compared only within one session, and the session is part of every cell.",
        "",
    ]
    return "\n".join(head + body + [END]) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    block = generate()
    current = open(TARGET).read()
    if BEGIN not in current or END not in current:
        print(f"README.md has no {BEGIN} / {END} markers — add them where the "
              "headline table goes", file=sys.stderr)
        return 2
    head, _, rest = current.partition(BEGIN)
    _, _, tail = rest.partition(END)
    new = head + block.rstrip("\n") + tail  # text after END is kept verbatim
    if args.check:
        if current != new:
            print("README.md headline block is stale — run scripts/render_headline.py",
                  file=sys.stderr)
            return 1
        print("README.md headline block is up to date.")
        return 0
    open(TARGET, "w").write(new)
    print(f"wrote {os.path.relpath(TARGET, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
