#!/usr/bin/env python3
"""Steady-state greedy decode driver for LiteRT-LM via the `litert-lm-api` Python package.

Loads a .litertlm on the requested backend, runs ONE prompt through the model's own chat
template with greedy sampling (top_k=1, temperature=0), streams the output and timestamps
every streamed chunk (one decode step each), then writes a JSON record with the per-token
latency series, the steady-state decode rate (first --skip tokens excluded) and the
engine's own BenchmarkInfo numbers.

Prints `PID <pid>` at start and `DECODE_START` on the first streamed token so that
profile_cell.sh can attach `xctrace` mid-decode.

Usage:
  litert_decode_driver.py model.litertlm --prompt "..." --backend gpu \
      --max-num-tokens 1536 --max-output-tokens 1200 --out cell.json [--tag ...]

`--max-num-tokens` is the CONTEXT size (KV cache), not a generation budget — undersizing it
corrupts output on this runtime (see CLAUDE.md rule 3). `--max-output-tokens` is the budget.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time

import litert_lm
from litert_lm import interfaces


def _opt_version(pkg):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def sha256_head(path, nbytes=64 << 20):
    """sha256 of the first 64 MB + file size: identifies the artifact without hashing GBs."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read(nbytes))
    return f'head64MB:{h.hexdigest()[:16]}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--prompt', required=True)
    ap.add_argument('--backend', default='gpu', choices=['cpu', 'gpu'])
    ap.add_argument('--max-num-tokens', type=int, default=1024,
                    help='context size (KV cache); NOT a generation budget')
    ap.add_argument('--max-output-tokens', type=int, default=None)
    ap.add_argument('--skip', type=int, default=32,
                    help='tokens excluded at the start of the steady-state window')
    ap.add_argument('--cache-dir', default=':nocache',
                    help='":nocache" = CLI --cache no; "" = model default (disk)')
    ap.add_argument('--gpu-decode-steps-per-sync', type=int, default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default='')
    ap.add_argument('--hf-repo', default='')
    ap.add_argument('--hf-revision', default='')
    ap.add_argument('--quant', default='')
    ap.add_argument('--quiet', action='store_true', help='do not echo generated text')
    args = ap.parse_args()

    print(f'PID {os.getpid()}', flush=True)
    if args.backend == 'gpu':
        try:  # gpu_decode_steps_per_sync exists from litert-lm-api 0.15.0
            backend = interfaces.GPU(gpu_decode_steps_per_sync=args.gpu_decode_steps_per_sync)
        except TypeError:
            backend = interfaces.GPU()
    else:
        backend = interfaces.CPU()
    api_version = importlib.metadata.version('litert-lm-api')
    cache_dir = args.cache_dir
    if cache_dir == ':nocache' and tuple(int(x) for x in api_version.split('.')[:2]) < (0, 15):
        cache_dir = ''  # the ':nocache' sentinel does not exist before 0.15; use the model default

    t0 = time.perf_counter()
    with litert_lm.Engine(args.model, backend=backend, max_num_tokens=args.max_num_tokens,
                          cache_dir=cache_dir) as engine:
        init_s = time.perf_counter() - t0
        sampler = interfaces.SamplerConfig(top_k=1, top_p=1.0, temperature=0.0)
        with engine.create_session(sampler_config=sampler,
                                   max_output_tokens=args.max_output_tokens) as session:
            tp = time.perf_counter()
            session.run_prefill([args.prompt])
            prefill_s = time.perf_counter() - tp
            stamps, chunks = [], []
            for resp in session.run_decode_async():
                now = time.perf_counter()
                if not stamps:
                    print('DECODE_START', flush=True)
                stamps.append(now)
                chunks.append(resp.texts[0])
                if not args.quiet:
                    sys.stdout.write(resp.texts[0])
                    sys.stdout.flush()
            t_end = time.perf_counter()
            print('\nDECODE_END', flush=True)
            try:
                bi = session.get_benchmark_info()
                bench = {
                    'init_time_s': bi.init_time_in_second,
                    'ttft_s': bi.time_to_first_token_in_second,
                    'prefill_tokens': bi.last_prefill_token_count,
                    'prefill_tps': bi.last_prefill_tokens_per_second,
                    'decode_tokens': bi.last_decode_token_count,
                    'decode_tps': bi.last_decode_tokens_per_second,
                }
            except Exception as e:  # noqa: BLE001
                bench = {'error': str(e)}

    n = len(stamps)
    per_tok_ms = [(stamps[i] - stamps[i - 1]) * 1e3 for i in range(1, n)]
    skip = min(args.skip, max(n - 2, 0))
    steady_tps = (n - 1 - skip) / (stamps[-1] - stamps[skip]) if n - 1 - skip > 0 else None
    overall_tps = (n - 1) / (stamps[-1] - stamps[0]) if n > 1 else None
    st = sorted(per_tok_ms[skip:]) if n - 1 - skip > 0 else []
    rec = {
        'tag': args.tag,
        'runtime': 'litert-lm-api',
        'runtime_version': api_version,
        'cli_version': _opt_version('litert-lm'),
        'backend': args.backend,
        'gpu_decode_steps_per_sync': args.gpu_decode_steps_per_sync,
        'cache_dir': cache_dir,
        'model_path': os.path.abspath(args.model),
        'model_file': os.path.basename(args.model),
        'model_bytes': os.path.getsize(args.model),
        'model_id': sha256_head(args.model),
        'hfRepoId': args.hf_repo,
        'hfRevision': args.hf_revision,
        'quantization': args.quant,
        'prompt': args.prompt,
        'max_num_tokens': args.max_num_tokens,
        'max_output_tokens': args.max_output_tokens,
        'sampler': {'top_k': 1, 'top_p': 1.0, 'temperature': 0.0},
        'engine_init_s': round(init_s, 3),
        'prefill_wall_s': round(prefill_s, 4),
        'generated_chunks': n,
        'generated_text_chars': sum(len(c) for c in chunks),
        'generated_text_head': ''.join(chunks)[:200],
        'generated_text': ''.join(chunks),
        'decode_tps_overall': round(overall_tps, 2) if overall_tps else None,
        'decode_tps_steady': round(steady_tps, 2) if steady_tps else None,
        'steady_window_tokens': n - 1 - skip,
        'per_token_ms_p50': round(st[len(st) // 2], 3) if st else None,
        'per_token_ms_p90': round(st[int(len(st) * 0.9)], 3) if st else None,
        'per_token_ms_max': round(st[-1], 3) if st else None,
        'decode_wall_s': round(t_end - stamps[0], 3) if n else None,
        'benchmark_info': bench,
        'per_token_ms': [round(x, 3) for x in per_tok_ms],
        'epoch_decode_start': time.time() - (t_end - stamps[0]) if n else None,
        'epoch_decode_end': time.time(),
        'host': {
            'machine': platform.machine(), 'macos': platform.mac_ver()[0],
            'python': platform.python_version(),
            'model': subprocess.run(['sysctl', '-n', 'hw.model'], capture_output=True,
                                    text=True).stdout.strip(),
        },
    }
    with open(args.out, 'w') as f:
        json.dump(rec, f, separators=(',', ':'), sort_keys=True)
    print(f"RESULT steady_tps={rec['decode_tps_steady']} overall_tps={rec['decode_tps_overall']} "
          f"tokens={n} engine_decode_tps={bench.get('decode_tps')} out={args.out}", flush=True)


if __name__ == '__main__':
    main()
