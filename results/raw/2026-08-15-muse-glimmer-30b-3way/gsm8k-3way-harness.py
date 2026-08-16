#!/usr/bin/env python3
"""GSM8K quality, Muse-Glimmer-30B, three arms — Core AI / MLX / ExecuTorch.

Imports the question set, the CoT suffix, the answer extractor and the scoring from
Yardstick's `scripts/parity_gsm8k.py` **without editing it**. That file is the shared
instrument for the Gemma-4 campaign; identical scoring is what makes these numbers
comparable, and not touching it is what keeps that campaign safe.

Two arms it does not already have:
  * MLX — its `run_mlx` uses `mlx_lm.load`, which does not know `muse_glimmer`
    (`ValueError: Model type muse_glimmer not supported`). This one goes through
    `mlx_vlm`, loading the model once instead of per question.
  * ExecuTorch — no arm existed; this drives Meta's own `solo_runner`
    (repo fairness rule 10: prefer the official runtime SDK).

**Two-pass budget.** Measured on this bundle: `llm-runner`'s wall time is
`5 s + 0.037 x max_tokens`, *independent of how many tokens are actually
generated* — it keeps stepping to the budget after the stop token halts output.
A budget wide enough for the longest answer therefore taxes all 100 questions.
So pass 1 runs at a low budget and pass 2 re-runs only the questions that hit it,
at a budget nothing has been seen to exceed. Every scored answer is un-truncated,
and the cost is set by the short questions instead of the longest one.
(ExecuTorch does not pay for unused budget; the same budget is still used on every
arm, because mixing budgets across arms is what fairness rule 3 forbids.)

Usage:
    _muse_gsm8k.py --arm coreai --n 2 --show     # probe first, always
    _muse_gsm8k.py --arm all --n 100
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

YARDSTICK = Path.home() / "Downloads" / "ios-llm-benchmark"
sys.path.insert(0, str(YARDSTICK / "scripts"))

from parity_gsm8k import COT, extract, load_q, norm  # noqa: E402

CA_RUNNER = os.path.expanduser("~/code/coreai/coreai-models/.build/release/llm-runner")
CA_BUNDLE = os.path.expanduser(
    "~/code/coreai/coreai-models/exports/muse_glimmer_30b_decode_int4hu_block32_sym")

ET_SOLO = os.path.expanduser(
    "~/code/et-pr/executorch/cmake-out/examples/models/muse-glimmer/solo_runner")
ET_PTE = next((Path(p) for p in Path(os.path.expanduser(
    "~/.cache/huggingface/hub/models--meta-models--Muse-Glimmer-30B-ExecuTorch-PTE/snapshots"
)).glob("*/muse-glimmer-k-quant-17G-128K-text-solo-metal/*.pte")), None)
ET_TOK = next((Path(p) for p in Path(os.path.expanduser(
    "~/.cache/huggingface/hub/models--meta-models--Muse-Glimmer-30B/snapshots"
)).glob("*/tokenizer.json")), None)

MLX_MODEL = "mlx-community/Muse-Glimmer-30B-4bit"


def arm_coreai(budget):
    def gen(prompt):
        p = subprocess.run(
            [CA_RUNNER, "--model", CA_BUNDLE, "--prompt", prompt,
             "--max-tokens", str(budget), "--sampling-strategy", "greedy",
             "--inference-engine-variant", "coreai-pipelined", "--warmup", "off"],
            capture_output=True, text=True, timeout=3600)
        # llm-runner frames the generation between "Generating..." and its perf summary;
        # feeding raw stdout to extract() would score tokens/sec as the answer.
        m = re.search(r"Generating\.\.\.\n(.*?)(?:\n\s*⏱|\n\s*=====|\Z)", p.stdout, re.S)
        n = re.search(r"Generation:\s*[\d.]+ms,\s*(\d+) tokens", p.stdout)
        return (m.group(1) if m else ""), (int(n.group(1)) if n else -1)

    return gen


def arm_executorch(budget):
    def gen(prompt):
        # --eos_id 200008 (<|eot|>): solo_runner defaults to 200001
        # (<|end_of_text|>), which this model does not emit at the end of a turn.
        # Matching the stop condition across arms is fairness rule 3.
        p = subprocess.run(
            [str(ET_SOLO), "--model_path", str(ET_PTE), "--tokenizer_path", str(ET_TOK),
             "--prompt", prompt, "--max_new_tokens", str(budget),
             "--temperature", "0", "--eos_id", "200008"],
            capture_output=True, text=True, timeout=3600)
        # solo_runner frames the generation BETWEEN its "Prefill:" and "Decode:" lines
        # and follows them with a PyTorchObserver JSON blob. Taking the wrong side of
        # that split hands extract() the blob, which scores `"prompt_tokens":91` as the
        # answer — caught on the first probe.
        m = re.search(r"\nPrefill:[^\n]*\n(.*?)(?:\nDecode:|\nPyTorchObserver|\Z)",
                      p.stdout, re.S)
        n = re.search(r"\nDecode:\s*(\d+) tokens", p.stdout)
        return (m.group(1) if m else ""), (int(n.group(1)) if n else -1)

    return gen


_MLX_CACHE = {}


def arm_mlx(budget):
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    if "m" not in _MLX_CACHE:  # load once across both passes
        _MLX_CACHE["m"], _MLX_CACHE["p"] = load(MLX_MODEL)
    model, processor = _MLX_CACHE["m"], _MLX_CACHE["p"]
    config = model.config

    def gen(prompt):
        formatted = apply_chat_template(processor, config, prompt, num_images=0)
        out = generate(model, processor, formatted, max_tokens=budget, verbose=False)
        txt = out if isinstance(out, str) else getattr(out, "text", str(out))
        tok = getattr(processor, "tokenizer", processor)
        try:
            n = len(tok.encode(txt))
        except Exception:  # noqa: BLE001 — the count is diagnostic, not the result
            n = -1
        return txt, n

    return gen


ARMS = {"coreai": arm_coreai, "executorch": arm_executorch, "mlx": arm_mlx}


def run_pass(arm, items, budget, show):
    """Score `items` (list of (index, question, gold)) at `budget`.

    Returns {index: record}. `capped` marks an answer that used the whole budget —
    it may be truncated, and a truncated answer is indistinguishable from a wrong
    one because the extractor falls back to "the last number in the text".
    """
    gen_fn = ARMS[arm](budget)
    out = {}
    for i, q, gold in items:
        t0 = time.time()
        n_tok = -1
        try:
            txt, n_tok = gen_fn(q + COT)
        except Exception as exc:  # noqa: BLE001 — one bad question must not kill the run
            print(f"  {arm} q{i+1} ERR {type(exc).__name__}: {str(exc)[:70]}", flush=True)
            txt = ""
        pred = norm(extract(txt))
        ok = pred == norm(gold)
        capped = n_tok >= budget
        out[i] = {"pred": pred, "gold": norm(gold), "ok": bool(ok),
                  "tokens": n_tok, "capped": bool(capped), "seconds": round(time.time() - t0, 1)}
        print(f"  {arm} q{i+1} {'OK' if ok else '..'} pred={pred} gold={norm(gold)} "
              f"tok={n_tok}{' CAPPED' if capped else ''} ({out[i]['seconds']:.0f}s)", flush=True)
        if show:
            print("  " + "-" * 60 + f"\n{txt[:1200]}\n  " + "-" * 60, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=[*ARMS, "all"], required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--budget", type=int, default=700, help="pass-1 generation budget")
    ap.add_argument("--rescue-budget", type=int, default=2048,
                    help="pass-2 budget, for questions that hit pass 1's")
    ap.add_argument("--show", action="store_true", help="print each generation (probe runs)")
    ap.add_argument("--out", default="/Users/majimadaisuke/code/coreai/_muse_gsm8k_results.json")
    args = ap.parse_args()

    for path, what in ((CA_RUNNER, "llm-runner"), (ET_SOLO, "solo_runner")):
        if not Path(path).exists():
            sys.exit(f"missing {what}: {path}")
    if ET_PTE is None or ET_TOK is None:
        sys.exit("missing ExecuTorch .pte or tokenizer.json in the HF cache")

    qs = load_q(args.n)
    items = [(i, q, gold) for i, (q, gold) in enumerate(qs)]
    print(f"{len(qs)} questions · greedy · budget {args.budget} "
          f"(rescue {args.rescue_budget}) · scoring from Yardstick parity_gsm8k", flush=True)

    results = {}
    for arm in ([args.arm] if args.arm != "all" else list(ARMS)):
        print(f"\n=== {arm} · pass 1 (budget {args.budget})", flush=True)
        t0 = time.time()
        recs = run_pass(arm, items, args.budget, args.show)

        rescue = [(i, q, gold) for i, q, gold in items if recs[i]["capped"]]
        if rescue:
            print(f"\n=== {arm} · pass 2 ({len(rescue)} capped, budget {args.rescue_budget})",
                  flush=True)
            for i, rec in run_pass(arm, rescue, args.rescue_budget, args.show).items():
                recs[i] = rec

        elapsed = time.time() - t0
        correct = sum(r["ok"] for r in recs.values())
        still_capped = sum(r["capped"] for r in recs.values())
        toks = sorted(r["tokens"] for r in recs.values() if r["tokens"] >= 0)
        results[arm] = {
            "correct": correct, "n": len(qs), "acc": correct / len(qs),
            "rescued": len(rescue), "still_capped": still_capped,
            "gen_tokens_median": toks[len(toks) // 2] if toks else None,
            "gen_tokens_max": toks[-1] if toks else None,
            "seconds": round(elapsed),
            "per_question": {str(i + 1): r for i, r in sorted(recs.items())},
        }
        print(f"== {arm}: {correct}/{len(qs)} = {100*correct/len(qs):.1f}%  "
              f"rescued {len(rescue)}  still-capped {still_capped}  ({elapsed:.0f}s)", flush=True)

    Path(args.out).write_text(json.dumps({
        "task": "gsm8k", "n": len(qs), "sampler": "greedy",
        "budget": args.budget, "rescue_budget": args.rescue_budget,
        "model": "meta-models/Muse-Glimmer-30B (text decoder)",
        "machine": "Mac Studio M4 Max (40-core GPU, 128 GB), macOS 27.0 26A5406e",
        "artifacts": {
            "coreai": "mlboydaisuke/Muse-Glimmer-30B-CoreAI int4hu (16.35 GB)",
            "mlx": f"{MLX_MODEL} (18 GB)",
            "executorch": "meta-models/...-ExecuTorch-PTE k-quant-17G text-solo-metal (17.9 GB)",
        },
        "scoring": "Yardstick scripts/parity_gsm8k.py — same questions, CoT suffix, extractor",
        "results": results,
    }, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
