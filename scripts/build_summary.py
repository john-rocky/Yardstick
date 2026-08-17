#!/usr/bin/env python3
"""Normalize every structured result in the repo into results/summary/ (continuous-bench
condition 2: machine-readable accumulation, kept separate from the raw logs).

Inputs (read-only):
  results/quality/*.json                 six historical schema variants (see gap audit)
  results/raw/**/app-path*/*.json        device per-run records (BenchmarkResult shape)
  results/raw/**/device-jsonl/*.json     warm-matrix campaign pulls (same shape)
  results/raw/*.jsonl                    flat records behind RESULTS.md (same shape;
                                         single pretty JSON or one record per line)

Duplicate records (a campaign pull later imported to a flat file) are dropped by
record UUID, campaign shapes winning over flat.

Outputs (overwritten each run — derived data, raw stays canonical):
  results/summary/quality.csv
  results/summary/device-runs.csv
  results/summary/README.md

Stdlib only. Idempotent. New writers should emit schema/result.v1.json natively;
this script is the bridge for pre-v1 records.
"""
import csv, glob, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "summary")
os.makedirs(OUT, exist_ok=True)


def rel(p):
    return os.path.relpath(p, ROOT)


QUALITY_EXTRA = ["schema_version", "timestamp", "runtime", "thinking",
                 "engine_version", "engine_artifact",
                 "model_id", "model_quantization"]


def build_quality():
    rows = []
    # regression/*/ holds re-captures made by `./reproduce ... --regress` (condition 3);
    # they accumulate alongside the published reports, distinguished by `source`.
    files = (glob.glob(os.path.join(ROOT, "results", "quality", "*.json"))
             + glob.glob(os.path.join(ROOT, "results", "quality", "regression", "*", "*.json")))
    for f in sorted(files):
        d = json.load(open(f))
        if "mode" in d and "results" in d:  # fakequant harness variant
            rows.append({
                "source": rel(f), "tag": d.get("mode"), "n": d.get("n"),
                "correct": d.get("ok"),
                "acc": round(d.get("ok", 0) / d["n"], 4) if d.get("n") else None,
                "max_tokens": None, "runtime_build": None, "backend": None,
                "bundle": None, "has_correction_note": "rescored" in d,
                "gen_tokens_median": None, "decode_tps_median": None,
                **{k: None for k in QUALITY_EXTRA},
            })
            continue
        rows.append({
            "source": rel(f), "tag": d.get("tag"), "n": d.get("n"),
            "correct": d.get("correct"), "acc": d.get("acc"),
            "max_tokens": d.get("max_tokens") or d.get("max_output_tokens"),
            "runtime_build": d.get("runtime_build"), "backend": d.get("backend"),
            "bundle": d.get("bundle"),
            "has_correction_note": bool(d.get("correction") or d.get("note")),
            "gen_tokens_median": d.get("gen_tokens_median"),
            "decode_tps_median": d.get("decode_tps_median"),
            "schema_version": d.get("schemaVersion"),
            "timestamp": d.get("timestamp"),
            "runtime": d.get("runtime"),
            "thinking": d.get("conditions", {}).get("thinking"),
            "engine_version": d.get("engineVersion"),
            "engine_artifact": d.get("engineArtifact"),
            # v1 reports carry model identity (parity_gsm8k.py emits model.id /
            # model.quantization) — the quality<->speed join key is
            # (model_id, runtime, thinking). Pre-v1 rows stay empty: tag prose is
            # not decoded by guessing.
            "model_id": d.get("model", {}).get("id"),
            "model_quantization": d.get("model", {}).get("quantization"),
        })
    path = os.path.join(OUT, "quality.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return path, len(rows)


def platform_of(d):
    """ios / mac / android, from the record itself (never from file paths)."""
    dev = d.get("device", {})
    sysname = (dev.get("systemName") or "").lower()
    if sysname.startswith("ios"):
        return "ios"
    if sysname.startswith("mac"):
        return "mac"
    if sysname.startswith("android"):
        return "android"
    mi = dev.get("modelIdentifier") or ""
    if mi.startswith(("iPhone", "iPad")):
        return "ios"
    if mi.startswith("Mac"):
        return "mac"
    return d.get("platform") or ""


def iter_device_records():
    """Yield (path, record) across the three raw shapes, campaign shapes first
    so UUID dedup drops the flat re-imports, not the campaign originals."""
    seen_ids = set()
    campaign_files = sorted(
        glob.glob(os.path.join(ROOT, "results", "raw", "**", "app-path*", "*.json"),
                  recursive=True)
        + glob.glob(os.path.join(ROOT, "results", "raw", "**", "device-jsonl", "*.json"),
                    recursive=True))
    for f in campaign_files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        rid = d.get("id")
        if rid:
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
        yield f, d
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "raw", "*.jsonl"))):
        txt = open(f).read().strip()
        if not txt:
            continue
        try:
            records = [json.loads(txt)]           # single pretty-printed record
        except json.JSONDecodeError:
            try:
                records = [json.loads(line) for line in txt.splitlines() if line.strip()]
            except json.JSONDecodeError:
                continue
        for d in records:
            rid = d.get("id")
            if rid:
                if rid in seen_ids:
                    continue
                seen_ids.add(rid)
            # Early flat Mac records carry the degenerate modelIdentifier
            # "arm64" (all Macs would pool). The flat-filename convention
            # (<device>-<rest>, same rule render_results.py uses) carries the
            # real label — restore it from there, never by guessing.
            dev = d.setdefault("device", {})
            if dev.get("modelIdentifier") in ("arm64", "", None):
                label = os.path.basename(f).split("-", 1)[0]
                if label:
                    dev["modelIdentifier"] = label
            yield f, d


def build_device():
    rows = []
    for f, d in iter_device_records():
        m, dev, model = d.get("metrics", {}), d.get("device", {}), d.get("model", {})
        if not m:
            continue
        parent = os.path.dirname(f)
        campaign = ("flat" if os.path.basename(parent) == "raw"
                    else rel(os.path.dirname(parent)))
        rows.append({
            "source": rel(f),
            "campaign": campaign,
            "platform": platform_of(d),
            "timestamp": d.get("timestamp"),
            "runtime": d.get("runtime"),
            "schema_version": d.get("schemaVersion"),
            "engine_version": d.get("engineVersion"),
            "engine_artifact": d.get("engineArtifact"),
            "model_id": model.get("id"),
            "quantization": model.get("quantization"),
            "task": d.get("task"),
            "harness_stamp": m.get("harnessStamp"),
            "decode_tps": m.get("decodeTokensPerSecond"),
            "decode_tps_wall": m.get("decodeTokensPerSecondWallClock"),
            "prefill_tps": m.get("promptTokensPerSecond"),
            "prompt_tokens": m.get("promptTokenCount"),
            "gen_tokens": m.get("generatedTokenCount"),
            "ttft_ms": m.get("firstTokenLatencyMS"),
            "mem_footprint_median_mb": m.get("memoryMedianMB"),
            "mem_resident_median_mb": m.get("memoryMedianResidentMB"),
            "energy_j_per_tok": m.get("energyJoulesPerToken"),
            "thermal_initial": m.get("initialThermalState"),
            "thermal_final": m.get("finalThermalState"),
            "battery_state": dev.get("batteryState"),
            "cold_run": m.get("coldRun"),
            "device": dev.get("modelIdentifier"),
            "os_version": dev.get("systemVersion"),
        })
    rows.sort(key=lambda r: (r["campaign"], r["timestamp"] or ""))
    path = os.path.join(OUT, "device-runs.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return path, len(rows)


def build_history():
    """Flatten results/regression-reports/*/verdicts.json into history.csv —
    the per-cell verdict series behind release-over-release tracking. The
    committed verdicts.json files are the canon; this is derived."""
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "regression-reports",
                                           "*", "verdicts.json"))):
        report = os.path.basename(os.path.dirname(f))
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for v in d.get("verdicts", []):
            anchor = v.get("anchor") or {}
            rows.append({
                "report": report, "mode": v.get("mode"), "verdict": v.get("verdict"),
                "cell": v.get("cell"), "metric": v.get("metric"),
                "base_median": v.get("base_median"), "cand_median": v.get("cand_median"),
                "raw_delta_pct": v.get("raw_delta_pct"),
                "normalized_delta_pct": anchor.get("normalized_delta_pct"),
                "anchor_cell": anchor.get("cell"),
                "baseline": d.get("baseline"), "candidate": d.get("candidate"),
            })
    path = os.path.join(OUT, "history.csv")
    fields = ["report", "mode", "verdict", "cell", "metric", "base_median",
              "cand_median", "raw_delta_pct", "normalized_delta_pct",
              "anchor_cell", "baseline", "candidate"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    return path, len(rows)


def main():
    qp, qn = build_quality()
    dp, dn = build_device()
    hp, hn = build_history()
    readme = os.path.join(OUT, "README.md")
    with open(readme, "w") as fh:
        fh.write(
            "# results/summary — derived, machine-readable accumulation\n\n"
            "Generated by `scripts/build_summary.py` from `results/quality/` and the\n"
            "committed `app-path*/` device records. Raw logs stay canonical; this layer is\n"
            "for querying, regression tracking, and leaderboard export — regenerate any\n"
            f"time with `python3 scripts/build_summary.py`.\n\n"
            f"- `quality.csv` — {qn} GSM8K report rows (all historical schema variants normalized)\n"
            f"- `device-runs.csv` — {dn} per-run device records (speed / memory / energy cells)\n"
            f"- `history.csv` — {hn} regression verdicts flattened from "
            "`results/regression-reports/*/verdicts.json`\n\n"
            "Engine version is absent from pre-v1 rows (see the gap audit). Builds from\n"
            "2026-08-13 onward stamp `engineVersion`/`engineArtifact` into every device row\n"
            "(`stamp_engine_pins.sh` -> bundled engine-pins.json -> BenchmarkResult),\n"
            "surfaced here as `engine_version`/`engine_artifact`; new writers must emit\n"
            "`schema/result.v1.json`.\n\n"
            "Release-regression diffing over this layer: `scripts/regression_diff.py`\n"
            "(quality joins on tag; device cells join on device/runtime/model/task/cold-warm\n"
            "with budget-mode-rule/spread-rule/cross-session guardrails). The capture+diff loop is\n"
            "`./reproduce <platform> <table> --regress` (continuous-bench condition 3).\n"
        )
    print(f"wrote {qp} ({qn} rows)")
    print(f"wrote {dp} ({dn} rows)")
    print(f"wrote {hp} ({hn} rows)")
    print(f"wrote {readme}")


if __name__ == "__main__":
    main()
