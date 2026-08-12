"""Canonical GSM8K quality harness — the generator of the README table's GSM8K column.

One protocol for every arm (fairness): chat template applied, thinking OFF unless
--thinking (then ON for EVERY arm compared), greedy, max_tokens 1024, same extractor,
zero-truncation verified. The pinned evaldata/gsm8k_test.jsonl is part of the protocol —
n=100 means the FIRST 100 of this exact file (an n=25-vs-n=100 mismatch once produced a
wrong bf16 anchor; do not substitute question sets).

Arms and their external prerequisites (env-overridable):
  --which mlx      : pip mlx_lm (Apple Silicon Mac)
  --which bf16     : pip torch+transformers
  --which coreai   : COREAI_RUNNER=<path to llm-runner> (Apple coreai-models build;
                     gemma-4 PLE bundles need the patched engine — see
                     methodology/core-ai-arm-provenance.md — and --raw-dir)
  --which int4     : litert-mac-verify at ~/code/litert-mac-verify (VERIFY constant)
  --which cactus   : a cactus checkout (see methodology/cactus-arm-session.md); FFI dylib
Reports land in results/quality/ (GSM8K_REPORTS to override).

Origin: developed across the 2026-07 Gemma-4-E2B campaign in hf-to-litertlm; this copy is
canonical for this repo as of 2026-07-20. See methodology/fairness-rules.md.
"""
import argparse, json, os, re, subprocess, sys, tempfile, time, types
from pathlib import Path

# scipy stub prelude — ONLY when scipy is actually broken. On a healthy scipy (e.g. the
# Accelerate-built scipy on Darwin 27, see darwin27-scipy-libomp-trap) stubbing is HARMFUL:
# the fake scipy.optimize lacks `milp`, which transformers' bf16 path pulls via
# sklearn → scipy.stats → `from scipy.optimize import milp`. So skip the stub when real
# scipy imports cleanly; only stub the broken-prebuilt case.
def _scipy_healthy():
    try:
        import scipy.optimize  # noqa: F401
        from scipy.optimize import milp  # noqa: F401
        import scipy.sparse.linalg._propack  # noqa: F401
        return True
    except Exception:
        return False

if not _scipy_healthy():
    _pp = types.ModuleType("scipy.sparse.linalg._propack")
    _pp.__file__ = "<stub:scipy._propack>"; _pp.__spec__ = None
    for _nm in ("_spropack", "_dpropack", "_cpropack", "_zpropack"):
        setattr(_pp, _nm, type("_S", (), {"__getattr__": lambda s, n: (lambda *a, **k: None), "__call__": lambda s, *a, **k: None})())
    sys.modules["scipy.sparse.linalg._propack"] = _pp
    _opt = types.ModuleType("scipy.optimize")
    _opt.__file__ = "<stub:scipy.optimize>"; _opt.__spec__ = None
    _opt.linear_sum_assignment = lambda *a, **k: None
    sys.modules["scipy.optimize"] = _opt

VERIFY = os.path.expanduser("~/code/litert-mac-verify/.build/release/litert-mac-verify")
DATA = os.environ.get("GSM8K_DATA", str(Path(__file__).resolve().parent.parent / "evaldata" / "gsm8k_test.jsonl"))
COT = ("\n\nSolve this step by step. After your reasoning, write the final answer on its own "
       "line in the exact form:\n#### <number>")

def load_q(n):
    out = []
    for line in open(DATA):
        d = json.loads(line)
        out.append((d["question"], d["answer"].split("####")[-1].strip().replace(",", "")))
        if len(out) >= n:
            break
    return out

def _trim_num(v):
    """'26.00'->'26', '10.0'->'10', '0.9'->'0.9'. The old inline `rstrip(".0")` stripped
    CHARACTERS, mangling trailing-zero integers ('20.00'->'2') — it scored LiteRT's correct
    "#### 20.00" as wrong (found 2026-07-20; LiteRT 85->86 after the fix)."""
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else str(f)
    except Exception:
        return v

def extract(text):
    """GSM8K-standard: prefer '#### N', then 'answer is/: N', then the last number."""
    if not text:
        return None
    t = text.replace(",", "")
    m = re.findall(r"####\s*\$?(-?\d+(?:\.\d+)?)", t)
    if m: return _trim_num(m[-1])
    m = re.findall(r"\\boxed\{\s*\$?(-?\d+(?:\.\d+)?)", t)  # OLMo-2 etc. mark the final answer with \boxed{}
    if m: return _trim_num(m[-1])
    m = re.findall(r"(?:answer|total|result)\s*(?:is|:|=)\s*\$?(-?\d+(?:\.\d+)?)", t, re.I)
    if m: return _trim_num(m[-1])
    m = re.findall(r"-?\d+(?:\.\d+)?", t)
    if not m: return None
    v = m[-1]
    return _trim_num(v)

def norm(x):
    if x is None: return None
    x = x.replace(",", "").lstrip("$")
    try:
        f = float(x); return str(int(f)) if f == int(f) else str(f)
    except: return x

def run_bf16(qs, path, max_tokens, bf16_template="chat"):
    import torch, transformers
    _DEV = os.environ.get("PARITY_DEVICE", "mps")  # set PARITY_DEVICE=cpu for 7B (MPS hits a hard Metal abort on this machine)
    tok = transformers.AutoTokenizer.from_pretrained(path)
    model = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(_DEV).eval()
    eot = tok.eos_token_id  # model-agnostic (Qwen3=<|im_end|>, Falcon=<|endoftext|>)
    pad = tok.pad_token_id if tok.pad_token_id is not None else eot
    c = 0
    for i, (q, gold) in enumerate(qs):
        if bf16_template == "chatml_simple":  # match the .litertlm's embedded simple template
            prompt = f"<|im_start|>user\n{q}{COT}<|im_end|>\n<|im_start|>assistant\n"
        elif bf16_template == "llama_simple":  # match the .litertlm's embedded Llama-3 simple template (BOS auto-added by tokenizer = runtime's start_token)
            prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{q}{COT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        elif bf16_template == "olmo_simple":  # match the .litertlm's embedded OLMo-2 simple template (BOS auto-added by tokenizer = runtime's start_token)
            prompt = f"<|user|>\n{q}{COT}\n<|assistant|>\n"
        elif bf16_template == "granite_simple":  # match the .litertlm's embedded Granite-3.x simple template (BOS auto-added by tokenizer = runtime's start_token)
            prompt = f"<|start_of_role|>user<|end_of_role|>{q}{COT}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
        elif bf16_template == "deepseek_r1":  # match the .litertlm's embedded DeepSeek-R1-Distill template (BOS auto-added by tokenizer = runtime's start_token)
            prompt = f"<｜User｜>{q}{COT}<｜Assistant｜>"
        elif bf16_template == "phi_simple":  # match the .litertlm's embedded Phi simple template (<|user|>…<|end|><|assistant|>)
            prompt = f"<|user|>{q}{COT}<|end|><|assistant|>"
        else:
            prompt = tok.apply_chat_template([{"role": "user", "content": q + COT}],
                                             tokenize=False, add_generation_prompt=True)
        try:
            ids = tok(prompt, return_tensors="pt").to(_DEV)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=max_tokens, do_sample=False, eos_token_id=eot, pad_token_id=pad)
            gen = [t for t in out[0][ids.input_ids.shape[1]:].tolist() if 0 <= t < len(tok)]  # guard out-of-range ids → tokenizer C++ OverflowError
            txt = tok.decode(gen, skip_special_tokens=True)
        except Exception as e:  # don't let one bad question kill a 100-q run
            print(f"  bf16 {i+1}/{len(qs)} ERR {type(e).__name__}: {str(e)[:60]}", flush=True)
            txt = ""
        ok = norm(extract(txt)) == norm(gold); c += ok
        print(f"  bf16 {i+1}/{len(qs)} {'OK' if ok else '..'} pred={norm(extract(txt))} gold={norm(gold)}", flush=True)
    return c

def run_mlx(qs, mlx_path, max_tokens, thinking=False):
    from mlx_lm import load, generate
    import mlx.core as mx
    model, tok = load(mlx_path)
    c = 0
    gen_tokens = []
    for i, (q, gold) in enumerate(qs):
        # Matched-mode rule: thinking OFF unless --thinking, in which case EVERY arm in the
        # comparison must run thinking ON (mlx via enable_thinking; coreai via --raw-tokens
        # of the same rendered prompt). Never mix modes across arms.
        try:
            prompt = tok.apply_chat_template([{"role": "user", "content": q + COT}],
                                             tokenize=False, add_generation_prompt=True,
                                             enable_thinking=thinking)
        except TypeError:
            prompt = tok.apply_chat_template([{"role": "user", "content": q + COT}],
                                             tokenize=False, add_generation_prompt=True)
        try:
            txt = generate(model, tok, prompt=prompt, max_tokens=max_tokens, verbose=False)
        except Exception as e:
            print(f"  mlx {i+1}/{len(qs)} ERR {type(e).__name__}: {str(e)[:60]}", flush=True); txt = ""
        try:
            mx.clear_cache()  # free Metal buffers between questions so they don't accumulate → OOM
        except Exception:
            pass
        n_gen = len(tok.encode(txt)) if txt else 0
        gen_tokens.append(n_gen)
        ok = norm(extract(txt)) == norm(gold); c += ok
        print(f"  mlx {i+1}/{len(qs)} {'OK' if ok else '..'} pred={norm(extract(txt))} gold={norm(gold)} gen={n_gen}", flush=True)
    if gen_tokens:
        import statistics as _st
        print(f"  mlx gen-tokens: median {_st.median(gen_tokens):.0f} mean {_st.mean(gen_tokens):.0f} max {max(gen_tokens)}", flush=True)
    return c


# Engine build under test. The default is the shared working checkout; override to compare
# engine versions (e.g. an isolated 0.2.0 + static-input-patch build). The engine version
# is part of the result — record it with the tag.
RUNNER = os.environ.get("COREAI_RUNNER",
                        os.path.expanduser("~/code/coreai/coreai-models/.build/release/llm-runner"))

def run_coreai(qs, bundle, max_tokens, raw_dir=None, thinking=False):
    """Core AI arm: the same COT prompt + greedy + extract() as the other arms, run through
    the official llm-runner (it applies the bundle's chat template, matching the bf16/mlx
    apply_chat_template path). gemma4 `tbl` bundles need --raw-dir for the PLE tables.
    COREAI_CHUNK_THRESHOLD=1: the gemma4 decode graph is S=1-static, so the prompt must be
    fed one token per step."""
    env = dict(os.environ, COREAI_CHUNK_THRESHOLD="1")
    c = 0
    tok = None
    if thinking:
        # llm-runner's internal swift-transformers templating renders thinking OFF and has
        # no flag; --raw-tokens bypasses it entirely. Render the thinking prompt with the
        # HF tokenizer (same gemma-4 vocab as the bundle) and hand the ids over.
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(os.path.expanduser(
            "~/code/litertlm-convert/src_models/gemma-4-E2B-it-qat-mobile-transformers"))
    gen_tokens = []
    for i, (q, gold) in enumerate(qs):
        try:
            # --warmup exact/1: the default warmup prefills shape [256], which the S=1-static
            # gemma4 decode graph rejects (fatal "not a valid substitution for source shape 1").
            cmd = [RUNNER, "--model", bundle,
                   "--max-tokens", str(max_tokens), "--sampling-strategy", "greedy",
                   "--warmup", "exact", "--warmup-length", "1"]
            if thinking:
                text = tok.apply_chat_template([{"role": "user", "content": q + COT}],
                                               tokenize=False, add_generation_prompt=True,
                                               enable_thinking=True)
                ids = tok(text, add_special_tokens=False)["input_ids"]
                assert ids[0] == 2 and 98 in ids, "thinking prompt must start with <bos> and contain <|think|>" 
                tf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
                json.dump({"tokens": ids}, tf); tf.close()
                cmd += ["--raw-tokens", tf.name]
            else:
                cmd += ["--prompt", q + COT]
            if raw_dir:
                cmd += ["--raw-dir", raw_dir]
            p_ = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
            # Isolate the generated text: llm-runner frames it between "Generating..." and
            # its perf summary. Feeding raw stdout to extract() would pick up the summary's
            # numbers (tokens/sec, ms) as the answer.
            m = re.search(r"Generating\.\.\.\n(.*?)(?:\n\s*⏱|\n\s*=====|\Z)", p_.stdout, re.S)
            txt = m.group(1) if m else ""
        except Exception as e:
            print(f"  coreai {i+1}/{len(qs)} ERR {type(e).__name__}: {str(e)[:60]}", flush=True)
            txt = ""
        n_gen = 0
        if txt and tok is not None:
            n_gen = len(tok.encode(txt, add_special_tokens=False))
        gen_tokens.append(n_gen)
        ok = norm(extract(txt)) == norm(gold); c += ok
        print(f"  coreai {i+1}/{len(qs)} {'OK' if ok else '..'} pred={norm(extract(txt))} gold={norm(gold)}"
              + (f" gen={n_gen}" if thinking else ""), flush=True)
    if thinking and gen_tokens:
        import statistics as _st
        print(f"  coreai gen-tokens: median {_st.median(gen_tokens):.0f} mean {_st.mean(gen_tokens):.0f} max {max(gen_tokens)}", flush=True)
    return c

def run_cactus(qs, bundle, max_tokens, cactus_repo=None, backend="metal", thinking=False):
    """Cactus arm: the same COT prompt + greedy + extract() through Cactus's own
    `cactus_complete` FFI (the entry point its CLI/benchmark use). Cloud handoff is forced
    off — a hybrid answer would score the cloud model, not the on-device build. top_k=1 +
    temperature=0 is Cactus's greedy path (mirrors cactus_benchmark_tokens). Needs the
    engine dylib: `bash <repo>/cactus-engine/build.sh` (or `cactus build --python`)."""
    os.environ.setdefault("CACTUS_NO_CLOUD_TELE", "1")
    repo = os.path.expanduser(cactus_repo or "~/code/cactus")
    sys.path.insert(0, os.path.join(repo, "python"))
    from cactus.bindings import cactus as C

    if C.cactus_set_backend(backend) != 0:
        print(f"  warning: backend '{backend}' unavailable; using engine default", flush=True)
    model = C.cactus_init(str(bundle))
    options = {
        "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0, "top_k": 1,
        "stop_sequences": ["<|im_end|>", "<end_of_turn>"],
        "telemetry_enabled": False, "auto_handoff": False,
        # Matched-mode rule: thinking ON only when every compared arm runs it ON.
        # For gemma-4, cactus renders a system turn containing <|think|> and returns
        # the trace in a separate "thinking" field; score the "response" field.
        "enable_thinking_if_supported": thinking,
    }
    c, no_marker = 0, 0
    think_tokens = []
    try:
        for i, (q, gold) in enumerate(qs):
            C.cactus_reset(model)
            try:
                resp = C.cactus_complete(model, [{"role": "user", "content": q + COT}], options)
                txt = resp.get("response", "") if isinstance(resp, dict) else ""
                if thinking and isinstance(resp, dict):
                    think_tokens.append(len(resp.get("thinking", "")) // 4)
            except Exception as e:
                print(f"  cactus {i+1}/{len(qs)} ERR {type(e).__name__}: {str(e)[:60]}", flush=True)
                txt = ""
            no_marker += ("####" not in txt)
            ok = norm(extract(txt)) == norm(gold); c += ok
            print(f"  cactus {i+1}/{len(qs)} {'OK' if ok else '..'} pred={norm(extract(txt))} gold={norm(gold)} chars={len(txt)}", flush=True)
    finally:
        C.cactus_destroy(model)
    print(f"  cactus: {no_marker}/{len(qs)} answers never emitted the '####' marker", flush=True)
    if think_tokens:
        import statistics as _st
        print(f"  cactus thinking (~tokens est. chars/4): median {_st.median(think_tokens):.0f} mean {_st.mean(think_tokens):.0f}", flush=True)
    return c


def run_litertlm(qs, litertlm, max_tokens, greedy=False, thinking=False):
    c = 0
    for i, (q, gold) in enumerate(qs):
        try:
            cmd = [VERIFY, litertlm, q + COT, "--max-tokens", str(max_tokens)]
            if greedy:
                cmd.append("--greedy")
            if thinking:
                cmd.append("--thinking")
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=400)
            m = re.search(r"^OUTPUT: \[(.*)\]$", p.stdout + "\n" + p.stderr, re.M)
            txt = m.group(1).replace("⏎", "\n") if m else ""
        except Exception:
            txt = ""
        ok = norm(extract(txt)) == norm(gold); c += ok
        print(f"  q {i+1}/{len(qs)} {'OK' if ok else '..'} pred={norm(extract(txt))} gold={norm(gold)}", flush=True)
    return c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--hf", default="src_models/falcon3-3b")
    ap.add_argument("--litertlm", default="out/falcon3-3b-mixed4/model.litertlm")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--which", choices=["bf16", "int4", "mlx", "coreai", "cactus"], required=True)
    ap.add_argument("--cactus-repo", default=None, help="cactus checkout with a built engine dylib (--which cactus)")
    ap.add_argument("--cactus-backend", default="metal", choices=["metal", "cpu"])
    ap.add_argument("--mlx-path", default=None)
    ap.add_argument("--bundle", default=None, help="Core AI bundle dir (--which coreai)")
    ap.add_argument("--raw-dir", default=None, help="PLE table dump dir for gemma4 tbl bundles")
    ap.add_argument("--bf16-template", choices=["chat", "chatml_simple", "llama_simple", "olmo_simple", "granite_simple", "deepseek_r1", "phi_simple"], default="chat")
    ap.add_argument("--greedy", action="store_true", help="run the .litertlm at temperature 0 (match bf16)")
    ap.add_argument("--thinking", action="store_true",
                    help="render the chat template with thinking ON (mlx: enable_thinking; "
                         "coreai: --raw-tokens bypass). Use the SAME mode on every arm you compare.")
    ap.add_argument("--tag", default=None, help="label for the result json")
    args = ap.parse_args()
    qs = load_q(args.n)
    t = time.time()
    if args.which == "bf16":
        c = run_bf16(qs, args.hf, args.max_tokens, args.bf16_template); tag = args.tag or "bf16"
    elif args.which == "mlx":
        c = run_mlx(qs, args.mlx_path, args.max_tokens, thinking=args.thinking); tag = args.tag or "mlx"
    elif args.which == "coreai":
        c = run_coreai(qs, args.bundle, args.max_tokens, args.raw_dir, thinking=args.thinking)
        tag = args.tag or "coreai"
    elif args.which == "cactus":
        c = run_cactus(qs, args.bundle, args.max_tokens, args.cactus_repo, args.cactus_backend,
                       thinking=args.thinking)
        tag = args.tag or "cactus"
    else:
        c = run_litertlm(qs, args.litertlm, args.max_tokens, args.greedy, thinking=args.thinking)
        tag = args.tag or os.path.basename(os.path.dirname(args.litertlm))
    acc = c / len(qs)
    print(f"== {tag}: {c}/{len(qs)} = {100*acc:.1f}%   ({time.time()-t:.0f}s)")
    os.makedirs(os.environ.get("GSM8K_REPORTS", "results/quality"), exist_ok=True)
    json.dump({"tag": tag, "n": len(qs), "correct": c, "acc": acc, "max_tokens": args.max_tokens},
              open(os.path.join(os.environ.get("GSM8K_REPORTS", "results/quality"), f"gsm8k_{tag}.json"), "w"), indent=2)
    outdir = os.environ.get("GSM8K_REPORTS", "results/quality")
    print(f"   wrote {outdir}/gsm8k_{tag}.json")

if __name__ == "__main__":
    main()
