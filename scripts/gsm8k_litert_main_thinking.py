"""GSM8K n=100 for the LiteRT-LM *main-build* CLI arm (thinking OFF/ON pair).

Protocol matches scripts/parity_gsm8k.py (same pinned questions, same COT suffix,
same extract/norm, greedy, max_tokens 2048, n=100). Runtime: litert-lm main
d98e1872 + 6-line CLI patch (--enable_thinking/--greedy/--max_output_tokens),
CPU backend (the bazel CLI's macOS GPU executor does not initialize).
Disclosure: NOT the same build/backend as the table's v0.13.1 GPU rows — this is
a main-build controlled OFF/ON pair, reported as such.
"""
import importlib.util, json, os, re, statistics, subprocess, sys, time

YARD = os.path.expanduser("~/Downloads/ios-llm-benchmark")
spec = importlib.util.spec_from_file_location(
    "parity_gsm8k", os.path.join(YARD, "scripts", "parity_gsm8k.py"))
pg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pg)

CLI = os.path.expanduser(
    "~/code/litert-lm-main-wt/bazel-bin/runtime/engine/litert_lm_main")
BUNDLE = os.path.expanduser(
    "~/.cache/huggingface/hub/models--litert-community--gemma-4-E2B-it-litert-lm/"
    "snapshots/9262660a1676eed6d0c477ab1a86344430854664/gemma-4-E2B-it.litertlm")
REPORTS = os.path.join(YARD, "results", "quality")
MAX_TOKENS = 2048
N = int(os.environ.get("GSM8K_N", "100"))


def run_arm(thinking):
    qs = pg.load_q(N)
    c = 0
    gen_tokens = []
    errors = []
    for i, (q, gold) in enumerate(qs):
        prompt = q + pg.COT
        cmd = [CLI, "--backend=cpu", "--greedy",
               f"--max_output_tokens={MAX_TOKENS}",
               f"--model_path={BUNDLE}", f"--input_prompt={prompt}"]
        if thinking:
            cmd.append("--enable_thinking")
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=400)
            out = p.stdout
            # Response = stdout after the echoed prompt, before BenchmarkInfo.
            bench_at = out.find("BenchmarkInfo:")
            body = out[:bench_at] if bench_at >= 0 else out
            echo_end = body.find(prompt)
            txt = body[echo_end + len(prompt):].strip() if echo_end >= 0 else body.strip()
            m = re.search(r"Decode Turn 1: Processed (\d+) tokens", out)
            if m:
                gen_tokens.append(int(m.group(1)))
            if "Error:" in txt:
                errors.append(i)
                print(f"  q {i+1}/{N} CLI-ERROR: {txt[txt.find('Error:'):][:100]}",
                      flush=True)
                txt = ""
                if len(errors) > 5:
                    raise RuntimeError("too many CLI errors — aborting arm")
        except subprocess.TimeoutExpired:
            print(f"  q {i+1}/{N} TIMEOUT (counted wrong)", flush=True)
            txt = ""
        ok = pg.norm(pg.extract(txt)) == pg.norm(gold)
        c += ok
        print(f"  q {i+1}/{N} {'OK' if ok else '..'} "
              f"pred={pg.norm(pg.extract(txt))} gold={pg.norm(gold)}"
              + (f" gen={gen_tokens[-1]}" if gen_tokens and m else ""), flush=True)
    if gen_tokens:
        print(f"  gen-tokens: median {statistics.median(gen_tokens):.0f} "
              f"mean {statistics.mean(gen_tokens):.0f} max {max(gen_tokens)}", flush=True)
    return c, gen_tokens


def main():
    for thinking in (False, True):
        tag = ("litert-gemma4-e2b-maincpu-thinking" if thinking
               else "litert-gemma4-e2b-maincpu-off")
        print(f"=== arm: {tag} ===", flush=True)
        t = time.time()
        c, gen_tokens = run_arm(thinking)
        acc = c / N
        print(f"== {tag}: {c}/{N} = {100*acc:.1f}%   ({time.time()-t:.0f}s)", flush=True)
        os.makedirs(REPORTS, exist_ok=True)
        report = {"tag": tag, "n": N, "correct": c, "acc": acc,
                  "max_tokens": MAX_TOKENS,
                  "runtime_build": "litert-lm main d98e1872 + thinking-flag CLI patch",
                  "backend": "cpu",
                  "note": "controlled OFF/ON pair on one build; not comparable 1:1 "
                          "with the v0.13.1 GPU table rows",
                  "gen_tokens_median": statistics.median(gen_tokens) if gen_tokens else None}
        path = os.path.join(REPORTS, f"gsm8k_{tag}.json")
        json.dump(report, open(path, "w"), indent=2)
        print(f"   wrote {path}", flush=True)


if __name__ == "__main__":
    main()
