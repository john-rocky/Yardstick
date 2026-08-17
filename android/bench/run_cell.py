#!/usr/bin/env python3
"""Run one Android benchmark cell: N fresh-process runs -> schema-v1 JSON each.

  python3 android/bench/run_cell.py --runtime litert-lm --backend gpu \\
      --model-id litert-community/Qwen3-0.6B --task short-chat --runs 3 \\
      --out results/raw/<campaign>/app-path-android

Design decisions (methodology/android.md):
  - one engine process per run = COLD by the repo's definition; the very first
    run per (model, backend) builds engine caches and is labelled firstEver.
  - metrics use BenchmarkResult field names (what build_summary.py reads);
    absent metrics stay absent. llama-cli has no TTFT; litert has no sampler
    control (conditions.sampler = "engine-default", a disclosed same-budget
    deviation).
  - the recorded runtime is litert-lm-<backend> / llama.cpp — backend is part
    of arm identity (the join key has no backend column; same convention as
    core-ai's -ane/-gpu model ids).
  - taskset f0 pins to the big cores (upstream recommendation), recorded in
    conditions.
  - RSS is sampled from /proc/<pid>/status (VmRSS) by an on-device loop ->
    memoryMedianResidentMB; iOS phys_footprint has no Android equivalent and
    is never fabricated.
"""
import argparse
import datetime
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from device_probe import adb, battery, device_info, thermal_status  # noqa: E402
import parsers  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEV_DIR = "/data/local/tmp/llmbench"
HARNESS_STAMP = "2026-08-android-cli-v1"


def load_pins():
    p = os.path.join(ROOT, "android", "engine-pins.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(model_id, file_hint, runtime, serial):
    """HF-download (host cache) then adb-push once; returns (device_path, local_path).

    file= is REQUIRED when the repo holds more than one artifact: the quant
    variant is part of arm identity, and the Android arm must run the same
    file the iOS catalog pins (ModelCatalog primaryFile) — picking one by
    sort order would silently measure a different recipe.
    """
    from huggingface_hub import hf_hub_download, list_repo_files
    if file_hint:
        fname = file_hint
    else:
        files = list_repo_files(model_id)
        ext = ".litertlm" if runtime.startswith("litert-lm") else ".gguf"
        cands = [f for f in files if f.endswith(ext)]
        if len(cands) != 1:
            raise SystemExit(
                f"{model_id} holds {len(cands)} {ext} artifacts — pass file= in the "
                f"cell (match the iOS ModelCatalog primaryFile):\n  " + "\n  ".join(sorted(cands)))
        fname = cands[0]
    local = hf_hub_download(model_id, fname)
    dev_path = f"{DEV_DIR}/models/{model_id.replace('/', '_')}_{fname}"
    have = subprocess.run(["adb"] + (["-s", serial] if serial else []) +
                          ["shell", f"ls {dev_path}"], capture_output=True, text=True)
    if have.returncode != 0:
        adb(["shell", "mkdir", "-p", f"{DEV_DIR}/models"], serial)
        print(f"pushing {fname} …", file=sys.stderr)
        adb(["push", local, dev_path], serial, timeout=1800)
    return dev_path, local


def push_prompt(task, serial):
    local = os.path.join(ROOT, "prompts", "text", f"{task}.txt")
    if not os.path.exists(local):
        raise SystemExit(f"no canonical prompt for task {task!r} "
                         "(scripts/gen_task_prompts.py; same-budget rule)")
    dev_path = f"{DEV_DIR}/prompts/{task}.txt"
    adb(["shell", "mkdir", "-p", f"{DEV_DIR}/prompts"], serial)
    adb(["push", local, dev_path], serial)
    budget = None
    for line in open(os.path.join(ROOT, "prompts", "text", "budgets.tsv")):
        t, b = line.split("\t")
        if t == task:
            budget = int(b)
    return dev_path, budget


DEFAULT_LLAMA_CTX = 4096  # llama-cli otherwise defaults to the model's TRAINING
# context (Qwen3: 40960) — measured 4.8 GB RSS for a 0.6B Q4 before this pin.


def engine_command(runtime, backend, model_dev, task, prompt_dev, budget, max_tokens,
                   context_tokens):
    """The on-device command line for one run. Returns (cmd, binname, sampler, ctx_note)."""
    if runtime.startswith("litert-lm"):
        ctx = f" --max_num_tokens={context_tokens}" if context_tokens else ""
        ctx_note = context_tokens or "bundle-default"
        if task.startswith("native-benchmark-"):
            # ONLY advanced_main consumes the benchmark token counts (verified
            # in v0.16.0 sources AND on device: the plain main ran its default
            # ~20-token prompt regardless of the flags).
            p, d = task[len("native-benchmark-"):].split("x")
            core = (f"./litert_lm_advanced_main --backend={backend} --model_path={model_dev} "
                    f"--benchmark --benchmark_prefill_tokens={p} "
                    f"--benchmark_decode_tokens={d} --async=false{ctx}")
            return core, "litert_lm_advanced_main", "engine-default", ctx_note
        core = (f"./litert_lm_main --backend={backend} --model_path={model_dev} "
                f"--input_prompt_file={prompt_dev} "
                f"--max_output_tokens={max_tokens or budget} --async=false{ctx}")
        # no temperature/top-p flags exist -> engine-default sampling, disclosed
        return core, "litert_lm_main", "engine-default", ctx_note
    if runtime == "llama.cpp":
        # -t matches the taskset f0 mask (4 mid cores): llama-bench defaults to
        # 9 threads, which busy-poll against a 4-core mask and hang (measured:
        # >5 min without output; -t 4 completes in seconds).
        ctx = context_tokens or DEFAULT_LLAMA_CTX
        if task.startswith("native-benchmark-"):
            p, d = task[len("native-benchmark-"):].split("x")
            return (f"./llama-bench -m {model_dev} -t 4 -p {p} -n {d} -o json"), \
                "llama-bench", "n/a (llama-bench)", "llama-bench-managed"
        return (f"./llama-cli -m {model_dev} -t 4 -c {ctx} -f {prompt_dev} "
                f"-n {max_tokens or budget} --temp 0 --top-p 1 -no-cnv"), \
            "llama-cli", "greedy", ctx
    raise SystemExit(f"unknown android runtime {runtime!r}")


def run_once(cmd, binname, serial, timeout):
    """One engine process with an RSS sampler wrapped around it on-device.

    $! is the backgrounded subshell, not the engine (measured: sampling it
    reads ~1.7 MB forever) — resolve the real pid with pgrep -n on the binary
    name and sample that."""
    shell = (f"cd {DEV_DIR} && LD_LIBRARY_PATH=. taskset f0 {cmd} & pid=$!; "
             f"sleep 1; epid=$(pgrep -n -f {binname}); [ -z \"$epid\" ] && epid=$pid; "
             "while kill -0 $pid 2>/dev/null; do "
             "grep VmRSS /proc/$epid/status 2>/dev/null; sleep 0.5; done; wait $pid; "
             "echo EXIT_CODE=$?")
    out = adb(["shell", shell], serial, timeout=timeout)
    rss_kb = [int(x) for x in
              (line.split()[1] for line in out.splitlines() if line.startswith("VmRSS"))]
    exit_code = 1
    for line in out.splitlines():
        if line.startswith("EXIT_CODE="):
            exit_code = int(line.split("=", 1)[1])
    return out, exit_code, (statistics.median(rss_kb) / 1024 if rss_kb else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", required=True, choices=["litert-lm", "llama.cpp"])
    ap.add_argument("--backend", default=None, choices=["cpu", "gpu"])
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--file", default=None, help="artifact filename inside the HF repo")
    ap.add_argument("--task", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--context-tokens", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--serial", default=None)
    ap.add_argument("--first-ever", action="store_true",
                    help="mark run 1 as firstEver (engine cache build)")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    if args.runtime == "litert-lm" and not args.backend:
        ap.error("litert-lm needs --backend cpu|gpu (arm identity)")
    arm = f"litert-lm-{args.backend}" if args.runtime == "litert-lm" else "llama.cpp"

    pins = load_pins()
    if args.runtime == "litert-lm":
        tag, entry = sorted(pins.get("litert-lm", {}).items())[-1] if pins.get("litert-lm") else (None, {})
        engine_version, engine_artifact = tag, entry.get("litert_lm_main_sha256")
    else:
        tag, entry = sorted(pins.get("llama.cpp", {}).items())[-1] if pins.get("llama.cpp") else (None, {})
        engine_version, engine_artifact = tag, entry.get("llama_bench_sha256" if args.task.startswith("native-") else "llama_cli_sha256")

    model_dev, model_local = ensure_model(args.model_id, args.file, args.runtime, args.serial)
    prompt_dev = budget = None
    if not args.task.startswith("native-benchmark-"):
        prompt_dev, budget = push_prompt(args.task, args.serial)
    cmd, binname, sampler, ctx_note = engine_command(
        args.runtime, args.backend, model_dev, args.task,
        prompt_dev, budget, args.max_tokens, args.context_tokens)

    os.makedirs(args.out, exist_ok=True)
    dev = device_info(args.serial)
    model_sha = sha256_file(model_local)
    ok = 0
    for i in range(1, args.runs + 1):
        raw_status, thermal_name = thermal_status(args.serial)
        batt = battery(args.serial)
        t0 = time.time()
        console, exit_code, rss_mb = run_once(cmd, binname, args.serial, args.timeout)
        elapsed = time.time() - t0

        if args.runtime == "llama.cpp" and args.task.startswith("native-benchmark-"):
            tests = parsers.parse_llama_bench_json(console)
            metrics = {}
            for t in tests:
                if t["kind"] == "prefill":
                    metrics["promptTokensPerSecond"] = t["avg_ts"]
                    metrics["promptTokenCount"] = t["n_prompt"]
                else:
                    metrics["decodeTokensPerSecond"] = t["avg_ts"]
                    metrics["generatedTokenCount"] = t["n_gen"]
            cold = False  # llama-bench repeats in-process (warm-ish regime)
        elif args.runtime == "llama.cpp":
            metrics = parsers.parse_llama_cli(console)
            cold = True
        else:
            metrics = parsers.parse_litert(console)
            cold = True

        end_status, end_name = thermal_status(args.serial)
        metrics.update({
            "coldRun": cold,
            "harnessStamp": HARNESS_STAMP,
            "initialThermalState": thermal_name,
            "finalThermalState": end_name,
        })
        if rss_mb is not None:
            metrics["memoryMedianResidentMB"] = rss_mb
        if args.first_ever and i == 1:
            metrics["firstEver"] = True

        iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        stamp = iso.replace(":", "-")  # filename-safe form
        console_name = f"{arm}_{args.model_id.replace('/', '_')}_{args.task}_{stamp}_run{i}.log"
        with open(os.path.join(args.out, console_name), "w") as fh:
            fh.write(console)

        rec = {
            "schemaVersion": 1,
            "id": str(uuid.uuid4()),
            "runtime": arm,
            "engineVersion": engine_version,
            "engineArtifact": engine_artifact,
            "model": {"id": args.model_id, "quantization": guess_quant(model_dev),
                      "file": os.path.basename(model_dev), "sha256": model_sha},
            "task": args.task,
            "timestamp": iso,
            "device": {**dev, "batteryLevel": batt["batteryLevel"],
                       "batteryState": batt["batteryState"]},
            "conditions": {"sampler": sampler, "cpuAffinity": "taskset f0",
                           "contextTokens": ctx_note,
                           "thermalRawStatus": raw_status,
                           "thermalRawStatusFinal": end_status,
                           "screen": "on-usb", "elapsedSeconds": round(elapsed, 1),
                           "exitCode": exit_code},
            "metrics": metrics,
            "provenance": {"rawLog": console_name, "harness": "android/bench/run_cell.py"},
        }
        name = f"{arm}_{args.model_id.replace('/', '_')}_{args.task}_{stamp}_run{i}.json"
        json.dump(rec, open(os.path.join(args.out, name), "w"), indent=2)
        d = metrics.get("decodeTokensPerSecond")
        status = "OK" if exit_code == 0 and d else f"FAIL(exit={exit_code})"
        print(f"run {i}/{args.runs} {status} decode={d} thermal={thermal_name}->{end_name}")
        if exit_code == 0 and d:
            ok += 1
    return 0 if ok == args.runs else 1


# Exact labels for known artifacts — same strings as the iOS ModelCatalog, so
# the same artifact never carries two labels across platforms (quant-label-rule:
# a bare "int4" is not a spec). Adding a model to android cells => add its
# label here (docs/OPERATIONS.md, add-a-model).
ANDROID_QUANT_LABELS = {
    "qwen3_0_6b_mixed_int4.litertlm": "INT4 (mixed, blockwise gs32)",
    "gemma-4-E2B-it.litertlm": "wNa8o8 (int2/int4/int8 + int8 activations, QAT)",
}


def guess_quant(dev_path):
    """Exact label for known artifacts; otherwise only what the filename
    states (GGUF quant suffixes ARE specs), else 'unrecorded'."""
    base = os.path.basename(dev_path)
    for known, label in ANDROID_QUANT_LABELS.items():
        if base.endswith(known):
            return label
    low = base.lower()
    for pat in ("q4_k_m", "q8_0", "f16", "fp16"):
        if pat in low:
            return pat.upper() if pat.startswith("q") else pat
    return "unrecorded (artifact name carries no quant label)"


if __name__ == "__main__":
    sys.exit(main())
