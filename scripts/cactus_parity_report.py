#!/usr/bin/env python3
"""Build the Cactus-vs-LiteRT/MLX comparison table from the raw run artifacts.

    python3 scripts/cactus_parity_report.py            # prints markdown to stdout

Reads `results/raw/2026-07-10-cactus-parity/`:
  cactus_<device>_<backend>.jsonl   one JSON object per run, straight from Cactus's
                                    `cactus_benchmark_tokens` (index 0 = its warmup).
  parity_runs.log                   YARDSTICK_RUN_OK lines from the iPhone app runs.
  litert_native_benchmark.log       YARDSTICK_NATIVE_OK lines from LiteRT's own
                                    `benchmark(prefillTokens:decodeTokens:)`.

Metric definitions, chosen so both sides are measured the same way:

  prefill tok/s = promptTokens / TTFT.  Every engine here can produce it, and it is
    what a user waits through. Cactus emits it as `ttft_prompt_tps`; its headline
    `prefill_tps` divides by `cache_prime_ms - cache_state_copy_ms` instead, but the
    cache-copy term measures 0.01–0.03 ms, so the two agree to <0.01% (asserted below).

  decode tok/s = the engine's own steady-state rate over its decode window, i.e.
    excluding TTFT. Cactus reports (n-1)/(total-TTFT); this repo reports n/decodeTime.
    At n=100 the two definitions differ by ~1%.

Warm-only: run index 0 (Cactus) / `cold=1` (this repo) is dropped, matching Cactus's
"one warmup, mean of three".
"""
import json
import re
import statistics
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "results/raw/2026-07-10-cactus-parity"

RUN_OK = re.compile(
    r"YARDSTICK_RUN_OK run=(?P<run>\d+) cold=(?P<cold>\d) decode_tok_s=(?P<decode>[\d.]+) "
    r"ttft_ms=(?P<ttft>\d+) prefill_tok_s=(?P<prefill>[\d.]+) prompt_tokens=(?P<ptok>\d+) "
    r"peak_mb=(?P<peak>[\d.]+) tokens=(?P<gen>\d+)"
)
BEGIN = re.compile(r"YARDSTICK_BEGIN runtime=(?P<rt>\S+) model=(?P<model>\S+)")
NATIVE = re.compile(
    r"YARDSTICK_NATIVE_OK prefill_tokens=(?P<ptok>\d+) prefill_tok_s=(?P<prefill>[\d.]+) "
    r"decode_tokens=(?P<dtok>\d+) decode_tok_s=(?P<decode>[\d.]+) ttft_ms=(?P<ttft>[\d.]+)"
)


def mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def load_cactus(path):
    runs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    warm = runs[1:]  # index 0 is Cactus's own warmup pass
    # The headline prefill_tps subtracts the prefill→decode KV handoff. Confirm here
    # that the subtraction is numerically inert rather than taking it on trust.
    skew = max(abs(r["prefill_tps"] - r["ttft_prompt_tps"]) / r["ttft_prompt_tps"] for r in warm)
    return {
        "prefill": mean([r["ttft_prompt_tps"] for r in warm]),
        "decode": mean([r["decode_tps"] for r in warm]),
        "peak_mb": mean([r["peak_ram_usage_mb"] for r in warm]),
        "ttft_ms": mean([r["time_to_first_token_ms"] for r in warm]),
        "prompt_tokens": warm[0]["prompt_tokens"],
        "decode_tokens": warm[0]["completion_tokens"],
        "cache_copy_ms": mean([r["cache_state_copy_ms"] for r in warm]),
        "headline_prefill": mean([r["prefill_tps"] for r in warm]),
        "prefill_skew_pct": 100 * skew,
        "n_warm": len(warm),
    }


def load_app_runs(path):
    """Group YARDSTICK_RUN_OK lines under the preceding YARDSTICK_BEGIN."""
    if not path.exists():
        return {}
    out, current = {}, None
    for line in path.read_text().splitlines():
        if m := BEGIN.search(line):
            current = (m["rt"], m["model"])
            out.setdefault(current, [])
        elif (m := RUN_OK.search(line)) and current:
            out[current].append({
                "cold": m["cold"] == "1",
                "decode": float(m["decode"]),
                "ttft_ms": float(m["ttft"]),
                "prompt_tokens": int(m["ptok"]),
                "peak_mb": float(m["peak"]),
                "gen_tokens": int(m["gen"]),
                "engine_prefill": float(m["prefill"]),
            })
    return out


def summarise_app(runs):
    warm = [r for r in runs if not r["cold"]] or runs
    ptok = warm[0]["prompt_tokens"]
    gen = warm[0]["gen_tokens"]
    # prompt_tokens is 0 when a runtime cannot report it (LiteRT when capped before
    # EOS, CoreML always) and ttft_ms is 0 when nothing was generated — in both cases
    # the TTFT-derived prefill is undefined, not zero.
    ttfts = [r["ttft_ms"] for r in warm if r["ttft_ms"] > 0]
    prefill = mean([ptok / (t / 1000) for t in ttfts]) if (ptok and ttfts) else None
    # No TTFT (nothing decoded) → fall back to the engine's own prefill window, flagged `†`,
    # because prefill throughput is still meaningful when decode never started. `mean([])` is
    # nan, which is truthy — so test the list, not the mean.
    reported = [r["engine_prefill"] for r in warm if r["engine_prefill"] > 0]
    return {
        "engine_prefill": mean(reported) if (prefill is None and reported) else None,
        # A runtime that emitted zero tokens has no decode rate; printing 0.0 would read
        # as "measured, and slow" rather than "did not run" (llama.cpp on this prompt).
        "prefill": prefill,
        "decode": mean([r["decode"] for r in warm]) if gen > 0 else None,
        "peak_mb": mean([r["peak_mb"] for r in warm]),
        "ttft_ms": mean(ttfts) if ttfts else None,
        "prompt_tokens": ptok,
        "decode_tokens": gen,
        "n_warm": len(warm),
    }


def load_native(path):
    if not path.exists():
        return None
    rows = [m.groupdict() for line in path.read_text().splitlines() if (m := NATIVE.search(line))]
    if not rows:
        return None
    return {
        "prefill": mean([float(r["prefill"]) for r in rows]),
        "decode": mean([float(r["decode"]) for r in rows]),
        "ttft_ms": mean([float(r["ttft"]) for r in rows]),
        "prompt_tokens": int(rows[0]["ptok"]),
        "decode_tokens": int(rows[0]["dtok"]),
        "n": len(rows),
    }


def fmt(v, unit=""):
    return "—" if v is None else f"{v:,.1f}{unit}"


def main():
    print("## Gemma-4-E2B, 1K-token prefill / 100-token decode budget, greedy, warm mean\n")

    for device, label in (("iphone17pro", "iPhone 17 Pro (A19 Pro)"), ("m4max", "Mac Studio M4 Max")):
        rows = []
        for path in sorted(RAW.glob(f"cactus_{device}_*.jsonl")):
            backend = path.stem.split("_")[-1]
            c = load_cactus(path)
            rows.append((f"Cactus CQ4 ({backend})", c))

        if device == "iphone17pro":
            for (rt, model), runs in load_app_runs(RAW / "parity_runs.log").items():
                rows.append((f"{rt} — {model.split('/')[-1]}", summarise_app(runs)))
            if nat := load_native(RAW / "litert_native_benchmark.log"):
                rows.append(("litert-lm (native benchmark API)", nat))

        if not rows:
            continue
        print(f"### {label}\n")
        print("| engine | prefill tok/s | decode tok/s | TTFT ms | peak MB | prompt tok | decode tok |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        dagger = False
        for name, r in rows:
            prefill = fmt(r["prefill"])
            if r["prefill"] is None and r.get("engine_prefill"):
                prefill, dagger = f"{r['engine_prefill']:,.1f} †", True
            print(f"| {name} | {prefill} | {fmt(r['decode'])} | {fmt(r['ttft_ms'])} "
                  f"| {fmt(r.get('peak_mb'))} | {r['prompt_tokens']} | {r['decode_tokens']} |")
        if dagger:
            print("\n† engine-reported prefill window; no TTFT because the runtime decoded 0 tokens.")
        print()

    print("### Cactus's `prefill_tps` vs its own wall-clock\n")
    print("| run | headline prefill_tps | ttft_prompt_tps | cache_state_copy_ms | skew |")
    print("|---|---:|---:|---:|---:|")
    for path in sorted(RAW.glob("cactus_*.jsonl")):
        c = load_cactus(path)
        print(f"| {path.stem.removeprefix('cactus_')} | {c['headline_prefill']:,.1f} "
              f"| {c['prefill']:,.1f} | {c['cache_copy_ms']:.2f} | {c['prefill_skew_pct']:.3f}% |")

    gsm = [json.loads(p.read_text()) for p in sorted(RAW.glob("gsm8k_*.json")) if "smoke" not in p.stem]

    def qtable(rows, marker_label):
        # `no marker` is the fraction that never wrote the required marker. Without it, a model
        # that ran out of budget is indistinguishable from one that answered wrong.
        print(f"| model | GSM8K | never emitted `{marker_label}` | mean chars | max_tokens | n |")
        print("|---|---:|---:|---:|---:|---:|")
        for d in rows:
            no_marker = sum(1 for r in d["rows"] if not r.get("has_marker", True))
            chars = statistics.mean(r["chars"] for r in d["rows"])
            print(f"| {d['tag']} | {d['accuracy_pct']:.1f}% | {no_marker} / {d['n']} "
                  f"| {chars:,.0f} | {d['max_tokens']} | {d['n']} |")

    hashruns = [d for d in gsm if d.get("cot_style", "hash") == "hash"]
    if hashruns:
        print("\n### Quality gate — GSM8K (0-shot CoT, greedy, same prompt + extraction)\n")
        qtable(hashruns, "#### <n>")

    answeris = [d for d in gsm if d.get("cot_style") == "answer-is"]
    if answeris:
        print("\n### Marker robustness — same questions, a marker CQ4 can emit (`The answer is <n>.`)\n")
        print("Rules out the objection that CQ4 reasons fine but cannot type `####`.\n")
        qtable(answeris, "The answer is <n>.")


if __name__ == "__main__":
    main()
