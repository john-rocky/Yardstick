"""GSM8K n=100 for the *released* LiteRT-LM 0.15.0 pip arm (thinking OFF/ON pair).

Runtime: litert-lm-api==0.15.0 from PyPI (uploaded 2026-08-03) — the first released
LiteRT-LM artifact whose binary carries the thinking C symbols. Run with the repo
venv:  .venv-litert015/bin/python scripts/gsm8k_litert_pip_thinking.py

Protocol matches scripts/parity_gsm8k.py (same pinned questions, same COT suffix,
same extract/norm, greedy = top_k=1/top_p=1.0/temp=0, max_output_tokens 2048,
n=100). Backend: GPU by default (WebGPU/Dawn -> Metal — the same backend class as
the published v0.13.1 GPU table row; probe 2026-08-04: decode ~122 tok/s vs CPU
~45). Set LITERT_BACKEND=cpu to reproduce the July main-CPU reference surface.

Thinking traces arrive on the separate `channels` key of the response mapping;
only `content` text items are scored, so no output filtering is needed.

Disclosure: pip dylib, not the Swift xcframework the table row was captured
through (no v0.15.0 xcframework release assets exist yet). Same engine core and
backend class, different wrapper — reports record this.
"""
import importlib.util, json, os, statistics, sys, time

YARD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "parity_gsm8k", os.path.join(YARD, "scripts", "parity_gsm8k.py"))
pg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pg)

import litert_lm
from litert_lm import interfaces

# Env-overridable (continuous-bench gap 1-3): LITERTLM_BUNDLE points at the model;
# the default is the pinned snapshot (environment.lock.json -> models) in the standard
# HF cache, honoring HF_HOME on machines that relocate it.
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
BUNDLE = os.environ.get("LITERTLM_BUNDLE", os.path.join(
    HF_HOME, "hub", "models--litert-community--gemma-4-E2B-it-litert-lm",
    "snapshots", "9262660a1676eed6d0c477ab1a86344430854664",
    "gemma-4-E2B-it.litertlm"))
REPORTS = os.path.join(YARD, "results", "quality")
MAX_TOKENS = 2048
N = int(os.environ.get("GSM8K_N", "100"))
BACKEND = os.environ.get("LITERT_BACKEND", "gpu").lower()

GREEDY = interfaces.SamplerConfig(top_k=1, top_p=1.0, temperature=0.0)


def response_text(resp):
    return "\n".join(
        item["text"] for item in resp.get("content", [])
        if item.get("type") == "text")


def run_arm(engine, thinking):
    qs = pg.load_q(N)
    c = 0
    gen_tokens, decode_tps = [], []
    for i, (q, gold) in enumerate(qs):
        prompt = q + pg.COT
        conv = engine.create_conversation(
            sampler_config=GREEDY,
            thinking_config=interfaces.ThinkingConfig(
                enable_thinking=thinking, thinking_token_budget=-1),
            max_output_tokens=MAX_TOKENS,
        )
        try:
            t = time.time()
            resp = conv.send_message(prompt)
            dt = time.time() - t
            txt = response_text(resp)
            bi = conv.get_benchmark_info()
            gen_tokens.append(bi.last_decode_token_count)
            decode_tps.append(bi.last_decode_tokens_per_second)
        except Exception as e:
            print(f"  q {i+1}/{N} ERROR (counted wrong): {e!r:.120}", flush=True)
            txt, dt = "", 0.0
        finally:
            conv.close()
        ok = pg.norm(pg.extract(txt)) == pg.norm(gold)
        c += ok
        print(f"  q {i+1}/{N} {'OK' if ok else '..'} "
              f"pred={pg.norm(pg.extract(txt))} gold={pg.norm(gold)} "
              f"gen={gen_tokens[-1] if gen_tokens else '?'} ({dt:.0f}s)", flush=True)
    if gen_tokens:
        print(f"  gen-tokens: median {statistics.median(gen_tokens):.0f} "
              f"mean {statistics.mean(gen_tokens):.0f} max {max(gen_tokens)}  "
              f"decode median {statistics.median(decode_tps):.1f} tok/s", flush=True)
    return c, gen_tokens, decode_tps


def main():
    backend = (interfaces.Backend.GPU() if BACKEND == "gpu"
               else interfaces.Backend.CPU())
    engine = litert_lm.Engine(BUNDLE, backend=backend, enable_benchmark=True)
    print(f"engine loaded (backend={BACKEND})", flush=True)
    try:
        for thinking in (False, True):
            tag = (f"litert-gemma4-e2b-pip015{BACKEND}-"
                   + ("thinking" if thinking else "off"))
            print(f"=== arm: {tag} ===", flush=True)
            t = time.time()
            c, gen_tokens, decode_tps = run_arm(engine, thinking)
            acc = c / N
            print(f"== {tag}: {c}/{N} = {100*acc:.1f}%   ({time.time()-t:.0f}s)",
                  flush=True)
            os.makedirs(REPORTS, exist_ok=True)
            report = {
                "tag": tag, "n": N, "correct": c, "acc": acc,
                "max_output_tokens": MAX_TOKENS,
                "runtime_build": "litert-lm-api 0.15.0 (PyPI wheel "
                                 "macosx_12_0_arm64, uploaded 2026-08-03)",
                "backend": BACKEND,
                "bundle": "litert-community/gemma-4-E2B-it-litert-lm @ 9262660",
                "sampler": "greedy (top_k=1, top_p=1.0, temperature=0.0)",
                "note": "released-artifact thinking pair; pip dylib, not the "
                        "Swift xcframework surface of the v0.13.1 table row",
                "gen_tokens_median":
                    statistics.median(gen_tokens) if gen_tokens else None,
                "decode_tps_median":
                    statistics.median(decode_tps) if decode_tps else None,
            }
            path = os.path.join(REPORTS, f"gsm8k_{tag}.json")
            json.dump(report, open(path, "w"), indent=2)
            print(f"   wrote {path}", flush=True)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
