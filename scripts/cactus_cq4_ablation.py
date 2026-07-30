#!/usr/bin/env python3
"""Why Cactus's CQ4 Gemma-4-E2B scores ~3% on GSM8K: rule out the harness, then the format.

`gsm8k_cactus_vs_litert.py` scores CQ4 at 3/100 with **zero** answers ever reaching the
required `#### <number>` line. Before that can be read as a quantization result rather
than a measurement mistake, four alternative explanations have to die:

  1. wrong chat template   → compare `cactus_render_prompt` against the HF reference
  2. wrong stop sequences  → Gemma-4 ends turns with `<turn|>`, not `<end_of_turn>`
  3. thinking mode left on → Cactus's own benchmark prefixes its system prompt with `/no_think`
  4. output budget too low → maybe it just needs more tokens to reach the marker

and then the failure has to be located: is multi-step *reasoning* broken, or only
instruction-following on the answer format?

    python3 scripts/cactus_cq4_ablation.py --cactus-repo <clone> \
        --path <clone>/weights/gemma-4-e2b-it-cq4
"""
import argparse, json, os, sys
from pathlib import Path

COT = ("\n\nSolve this step by step. After your reasoning, write the final answer on its own "
       "line in the exact form:\n#### <number>")
NATALIA = ("Natalia sold clips to 48 of her friends in April, and then she sold half as many "
           "clips in May. How many clips did Natalia sell altogether in April and May?")  # 72
ONE_STEP = "A box has 48 apples. Half are red. How many are red?"                          # 24
ADD = "What is 17 + 25?"                                                                   # 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cactus-repo", required=True)
    ap.add_argument("--path", required=True, help="CQ4 bundle directory")
    ap.add_argument("--out", default="results/raw/2026-07-10-cactus-parity/cq4_ablation.json")
    args = ap.parse_args()

    os.environ.setdefault("CACTUS_NO_CLOUD_TELE", "1")
    sys.path.insert(0, str(Path(args.cactus_repo) / "python"))
    from cactus.bindings import cactus as C

    C.cactus_set_backend("metal")
    model = C.cactus_init(str(args.path))
    if not model:
        sys.exit("cactus_init failed")

    base = {"temperature": 0.0, "top_p": 1.0, "top_k": 1,
            "telemetry_enabled": False, "auto_handoff": False}

    def complete(prompt, max_tokens, system=None, **extra):
        C.cactus_reset(model)
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        r = C.cactus_complete(model, msgs, {**base, "max_tokens": max_tokens, **extra})
        return r["response"], r["decode_tokens"]

    out = {"bundle": str(args.path)}

    # (1) template — identical to the HF reference means Cactus renders Gemma-4 correctly.
    out["rendered_prompt"] = C.cactus_render_prompt(model, [{"role": "user", "content": "Hi"}])

    # (2)(3)(4) harness ablations on the exact scored prompt.
    ablations = []
    for label, kwargs in [
        ("baseline (as scored)",        dict(max_tokens=1024, stop_sequences=["<|im_end|>", "<end_of_turn>"])),
        ("gemma stop sequence",         dict(max_tokens=1024, stop_sequences=["<turn|>"])),
        ("no stop sequences",           dict(max_tokens=1024)),
        ("thinking disabled",           dict(max_tokens=1024, enable_thinking_if_supported=False)),
        ("cactus's own system prompt",  dict(max_tokens=1024, system="/no_think You are helpful.")),
        # 1536 is the ceiling here: cactus_complete segfaults at 4096 decode tokens, presumably
        # past the KV its 512-token default sliding window was sized for.
        ("1.5x the token budget",       dict(max_tokens=1536)),
    ]:
        txt, tokens = complete(NATALIA + COT, **kwargs)
        ablations.append({"variant": label, "decode_tokens": tokens,
                          "terminated": tokens < kwargs["max_tokens"],
                          "has_marker": "####" in txt, "chars": len(txt),
                          "correct": "72" in txt.replace(",", ""), "tail": txt[-140:]})
        print(f"{label:<28} tok={tokens:>5} terminated={ablations[-1]['terminated']!s:<5} "
              f"marker={ablations[-1]['has_marker']!s:<5} correct={ablations[-1]['correct']}")
    out["harness_ablations"] = ablations

    # Locate the failure: simple arithmetic and one-step word problems still work.
    probes = []
    for label, prompt, gold, budget in [
        ("bare arithmetic",           ADD,             "42", 768),
        ("one-step word problem",     ONE_STEP,        "24", 768),
        ("two-step, no CoT suffix",   NATALIA,         "72", 768),
        ("two-step, plain step-by-step", NATALIA + "\n\nLet's think step by step.", "72", 768),
    ]:
        txt, tokens = complete(prompt, budget)
        probes.append({"probe": label, "gold": gold, "decode_tokens": tokens,
                       "terminated": tokens < budget,
                       "correct": gold in txt.replace(",", ""), "chars": len(txt),
                       "head": txt[:200]})
        print(f"{label:<30} tok={tokens:>5} terminated={probes[-1]['terminated']!s:<5} "
              f"correct={probes[-1]['correct']}")
    out["failure_localisation"] = probes

    C.cactus_destroy(model)
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
