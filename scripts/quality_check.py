#!/usr/bin/env python3
"""Score the `quality` task outputs for correctness + degeneracy → docs/litert-lm/QUALITY.md.

NOT a perplexity / MMLU eval. A guardrail that catches quantization-induced quality *collapse*
(wrong answers, degenerate repetition) so the decode-tok/s comparison across runtimes — each with
its own native 4-bit quant — is read at roughly equal output quality, not speed-at-any-quality.
Coarse by design (whole-output substring match); a collapsed quant misses many, not one.

    python3 scripts/quality_check.py
"""
import json, glob, re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "results" / "raw"
OUT = REPO / "docs" / "litert-lm" / "QUALITY.md"

# (label, regex) — greedy answers to the 8 fixed questions in QualityTask.swift.
CHECKS = [
    ("17+25=42",        r"\b42\b"),
    ("capital=Tokyo",   r"tokyo"),
    ("opp(hot)=cold",   r"\bcold\b"),
    ("days/week=7",     r"\bseven\b|\b7\b"),
    ("thanks(fr)=merci", r"merci"),
    ("8*7=56",          r"\b56\b"),
    ("0.9>0.11",        r"0\.9"),
    ("rhyme=blue",      r"\bblue\b"),
]


def correctness(text):
    t = text.lower()
    return [bool(re.search(p, t)) for _, p in CHECKS]


def degenerate(text):
    """True if the output loops or is special-token spam."""
    words = text.split()
    if len(words) >= 10:
        grams = [" ".join(words[i:i + 5]) for i in range(len(words) - 4)]
        if grams and Counter(grams).most_common(1)[0][1] >= 3:
            return True
        if len(set(words)) / len(words) < 0.30:
            return True
    # No-space loops (e.g. "<|fim_prefix|><|fim_prefix|>…"): tiny char diversity, or a
    # special token leaked + repeated. split()-by-space sees these as one "word", so check chars.
    if len(text) >= 40 and len(set(text)) < 15:
        return True
    if text.count("<|") >= 5 or text.count("<pad>") >= 5:
        return True
    return False


def empty_collapse(text, gen_tokens, streamed_chunks):
    """A run that reported decode tokens but streamed (almost) no text — the
    model emitted only special / non-decodable tokens. The substring checks
    can't see this (there is nothing to match), so it would otherwise score
    0/8 + "not degenerate", masking a collapse as a mere wrong-answer run.
    `streamed_chunks` is None on pre-2026-06 JSONL (field absent) — fall back
    to the output length in that case."""
    if streamed_chunks == 0 and gen_tokens and gen_tokens > 0:
        return True
    if gen_tokens and gen_tokens >= 8 and len(text.strip()) < 5:
        return True
    return False


def main():
    rows = []
    for f in sorted(RAW.glob("*-quality-run*.jsonl")):
        raw = open(f).read()
        try:
            d = json.loads(raw.split("\n")[0])  # JSONL: first record
        except json.JSONDecodeError:
            d = json.loads(raw)                 # pretty-printed single record (import_to_flat.py)
        dev = f.name.split("-quality-")[0]
        out = d.get("outputSample", "")
        m = d.get("metrics", {})
        gen = m.get("generatedTokenCount", 0)
        streamed = m.get("streamedChunkCount")  # None on pre-2026-06 rows
        hits = correctness(out)
        rows.append({
            "dev": dev,
            "runtime": d.get("runtime", "?"),
            "model": d.get("model", {}).get("displayName", d.get("model", {}).get("id", "?")),
            "quant": d.get("model", {}).get("quantization", "?"),
            "score": sum(hits), "hits": hits,
            "degen": degenerate(out) or empty_collapse(out, gen, streamed),
            "sample": " ".join(out.split())[:300] or f"(no text — {gen} tokens, 0 decoded)",
        })

    lines = [
        "# Quality parity — correctness + degeneracy guardrail",
        "",
        "> Scored by [`scripts/quality_check.py`](../../scripts/quality_check.py) from the `quality` task "
        "(8 fixed checkable questions, greedy). **Not** a perplexity/MMLU eval — a guardrail that catches "
        "quantization-induced quality *collapse* so the decode-tok/s tables compare runtimes at roughly "
        "equal quality, not speed-at-any-quality. Each runtime uses its native 4-bit quant (disclosed).",
        "",
        "| Device | Runtime | Model | Quant | Correct | Degenerate? |",
        "|---|---|---|---|---:|:--:|",
    ]
    for r in rows:
        flag = "⚠️ yes" if r["degen"] else "no"
        lines.append(f"| {r['dev']} | {r['runtime']} | {r['model']} | {r['quant']} | "
                     f"{r['score']}/8 | {flag} |")
    if not rows:
        lines.append("| _(no quality runs yet — run `--task quality`)_ | | | | | |")
    lines += ["", "## Per-question hits", "",
              "`" + "  ".join(c[0] for c in CHECKS) + "`", ""]
    for r in rows:
        marks = "".join("✓" if h else "·" for h in r["hits"])
        lines.append(f"- **{r['runtime']} {r['model']}** ({r['dev']}): `{marks}`  — “{r['sample']}”")
    OUT.write_text("\n".join(lines) + "\n")
    print("wrote", OUT.relative_to(REPO), f"({len(rows)} runs)")


if __name__ == "__main__":
    main()
