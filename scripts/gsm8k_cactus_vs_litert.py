#!/usr/bin/env python3
"""GSM8K parity for the Cactus comparison: is CQ4 the same quality as int4-QAT?

Speed numbers across engines only mean something at equal output quality — each engine
ships its own 4-bit format (Cactus CQ4 = Hadamard rotation + per-group codebook;
LiteRT-LM = int4 QAT; MLX = affine 4-bit group quant). This scores all of them on the
same 0-shot-CoT GSM8K prompt with the same greedy decode and the same answer extraction,
so the only variable is the quantization + runtime.

The prompt, extraction and normalisation are imported from the canonical harness at
`~/code/litertlm-convert/scripts/parity_gsm8k.py` rather than re-implemented, so the
numbers stay comparable with the Falcon3 / Qwen3 parity runs recorded there.

    # 1. Cactus CQ4 — drives Cactus's own `cactus_complete` FFI (needs `cactus build --python`).
    #    Cloud handoff is forced off: a hybrid answer would score the cloud model, not CQ4.
    python3 scripts/gsm8k_cactus_vs_litert.py --backend cactus \
        --cactus-repo <clone> --path <clone>/weights/gemma-4-e2b-it-cq4 --n 150

    # 2. LiteRT-LM int4-QAT (macOS runs it on CPU; quality is backend-independent)
    python3 scripts/gsm8k_cactus_vs_litert.py --backend litertlm \
        --path ~/.cache/huggingface/hub/models--litert-community--gemma-4-E2B-it-litert-lm/snapshots/*/gemma-4-E2B-it.litertlm --n 150

    # 3. MLX 4-bit reference
    python3 scripts/gsm8k_cactus_vs_litert.py --backend mlx \
        --path mlx-community/gemma-4-e2b-it-4bit --n 150
"""
import argparse, importlib.util, json, os, re, subprocess, sys, time
from pathlib import Path

PARITY = Path.home() / "code/litertlm-convert/scripts/parity_gsm8k.py"
DATA = Path.home() / "code/litertlm-convert/evaldata/gsm8k_test.jsonl"
VERIFY = Path.home() / "code/litert-mac-verify/.build/release/litert-mac-verify"


def _load_parity():
    if not PARITY.exists():
        sys.exit(f"canonical harness not found: {PARITY}")
    spec = importlib.util.spec_from_file_location("parity_gsm8k", PARITY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_questions(n):
    out = []
    with open(DATA) as f:
        for line in f:
            d = json.loads(line)
            out.append((d["question"], d["answer"].split("####")[-1].strip().replace(",", "")))
            if len(out) >= n:
                break
    return out


def run_cactus(questions, repo, bundle, max_tokens, cot, backend="metal"):
    """Drives Cactus's own `cactus_complete` FFI, the same entry point its CLI uses."""
    os.environ.setdefault("CACTUS_NO_CLOUD_TELE", "1")
    sys.path.insert(0, str(Path(repo) / "python"))
    from cactus.bindings import cactus as C

    if C.cactus_set_backend(backend) != 0:
        print(f"  warning: backend '{backend}' unavailable; using engine default", flush=True)
    model = C.cactus_init(str(bundle))
    if not model:
        sys.exit("cactus_init failed")

    # top_k=1 + temperature=0 is Cactus's greedy path (see cactus_benchmark_tokens,
    # which decodes with exactly these values). auto_handoff off keeps it on-device.
    options = {
        "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0, "top_k": 1,
        "stop_sequences": ["<|im_end|>", "<end_of_turn>"],
        "telemetry_enabled": False, "auto_handoff": False,
    }
    outs = []
    try:
        for i, (q, _) in enumerate(questions):
            C.cactus_reset(model)
            try:
                resp = C.cactus_complete(model, [{"role": "user", "content": q + cot}], options)
                outs.append(resp.get("response", "") if isinstance(resp, dict) else "")
            except Exception as e:
                print(f"  cactus {i+1} ERR {type(e).__name__}: {str(e)[:70]}", flush=True)
                outs.append("")
    finally:
        C.cactus_destroy(model)
    return outs


def run_litertlm(questions, path, max_tokens, cot):
    outs = []
    for i, (q, _) in enumerate(questions):
        try:
            p = subprocess.run([str(VERIFY), str(path), q + cot, "--max-tokens", str(max_tokens), "--greedy"],
                               capture_output=True, text=True, timeout=600)
            m = re.search(r"^OUTPUT: \[(.*)\]$", p.stdout + "\n" + p.stderr, re.M)
            outs.append(m.group(1).replace("⏎", "\n") if m else "")
        except Exception as e:
            print(f"  litertlm {i+1} ERR {type(e).__name__}: {str(e)[:70]}", flush=True)
            outs.append("")
    return outs


def run_bf16(questions, path, max_tokens, cot):
    """Unquantized anchor: what the same weights score before any 4-bit format touches them."""
    import torch, transformers
    tok = transformers.AutoTokenizer.from_pretrained(path)
    # Gemma-4 is `Gemma4ForConditionalGeneration`; only the image-text-to-text auto class maps it.
    model = transformers.AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16)
    model = model.to(os.environ.get("PARITY_DEVICE", "mps")).eval()
    dev = next(model.parameters()).device
    outs = []
    for i, (q, _) in enumerate(questions):
        prompt = tok.apply_chat_template([{"role": "user", "content": q + cot}],
                                         tokenize=False, add_generation_prompt=True)
        try:
            ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(dev)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=max_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            outs.append(tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True))
        except Exception as e:
            print(f"  bf16 {i+1} ERR {type(e).__name__}: {str(e)[:70]}", flush=True)
            outs.append("")
    return outs


def run_mlx(questions, path, max_tokens, cot):
    from mlx_lm import load, generate
    import mlx.core as mx
    model, tok = load(path)
    outs = []
    for i, (q, _) in enumerate(questions):
        prompt = tok.apply_chat_template([{"role": "user", "content": q + cot}],
                                         tokenize=False, add_generation_prompt=True)
        try:
            outs.append(generate(model, tok, prompt=prompt, max_tokens=max_tokens, verbose=False))
        except Exception as e:
            print(f"  mlx {i+1} ERR {type(e).__name__}: {str(e)[:70]}", flush=True)
            outs.append("")
        try:
            mx.clear_cache()
        except Exception:
            pass
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["cactus", "litertlm", "mlx", "bf16"], required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--path", default=None, help="CQ4 bundle dir, litertlm file, or MLX HF repo id")
    ap.add_argument("--cactus-repo", default=None, help="cactus checkout (for its python bindings)")
    ap.add_argument("--cactus-backend", default="metal", choices=["metal", "cpu"])
    ap.add_argument("--cot-style", default="hash", choices=["hash", "answer-is"],
                    help="'hash' = the canonical '#### <n>' marker. 'answer-is' swaps in a plain "
                         "sentence, to separate a model that cannot reason from one that merely "
                         "cannot emit the '####' token (Cactus CQ4 writes '# 30' / 'abcd 42').")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default="results/raw/2026-07-10-cactus-parity")
    args = ap.parse_args()

    parity = _load_parity()
    cot, extract, norm = parity.COT, parity.extract, parity.norm
    if args.cot_style == "answer-is":
        cot = ("\n\nSolve this step by step. After your reasoning, write the final answer on its "
               "own line in the exact form:\nThe answer is <number>.")

    questions = load_questions(args.n)
    t0 = time.time()
    if args.backend == "cactus":
        if not args.cactus_repo:
            sys.exit("--cactus-repo is required for --backend cactus")
        outs = run_cactus(questions, args.cactus_repo, args.path, args.max_tokens, cot,
                          backend=args.cactus_backend)
    elif args.backend == "litertlm":
        outs = run_litertlm(questions, args.path, args.max_tokens, cot)
    elif args.backend == "bf16":
        outs = run_bf16(questions, args.path, args.max_tokens, cot)
    else:
        outs = run_mlx(questions, args.path, args.max_tokens, cot)
    elapsed = time.time() - t0

    correct, empties, rows = 0, 0, []
    for (q, gold), txt in zip(questions, outs):
        pred = norm(extract(txt))
        ok = pred == norm(gold)
        correct += ok
        empties += (not txt.strip())
        rows.append({
            "pred": pred, "gold": norm(gold), "ok": bool(ok), "chars": len(txt),
            # A model that runs out of budget before writing "#### <n>" is scored wrong
            # by extract()'s last-number fallback. Without these two flags a truncation
            # rate is indistinguishable from a wrong-answer rate.
            "has_marker": ("####" in txt) if args.cot_style == "hash"
                          else ("answer is" in txt.lower()),
            "tail": txt[-160:],
        })

    tag = args.tag or args.backend
    acc = 100.0 * correct / len(questions)
    no_marker = sum(1 for r in rows if not r["has_marker"])
    print(f"\n=== {tag} — GSM8K {correct}/{len(questions)} = {acc:.1f}%  "
          f"(empty: {empties}, no '####' marker: {no_marker}, {elapsed/60:.1f} min) ===")
    if no_marker:
        print(f"    {no_marker} answers never reached the '#### <n>' line — raise --max-tokens "
              f"before reading the accuracy as a quality signal.")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / f"gsm8k_{tag}.json"
    dest.write_text(json.dumps({
        "backend": args.backend, "tag": tag, "n": len(questions),
        "correct": correct, "accuracy_pct": acc, "empty_outputs": empties,
        "max_tokens": args.max_tokens, "cot_style": args.cot_style,
        "elapsed_seconds": round(elapsed, 1),
        "path": str(args.path),
        "cactus_backend": args.cactus_backend if args.backend == "cactus" else None,
        "rows": rows,
    }, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
