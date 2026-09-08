#!/usr/bin/env python3
"""Steady-state greedy decode driver for MLX-LM (same record shape as litert_decode_driver.py).

Applies the model's chat template (as `python -m mlx_lm generate` does), greedy sampling
(temp=0), streams with `stream_generate`, timestamps every token, writes a JSON record.
Prints `PID <pid>` at start and `DECODE_START` on the first token for profile_cell.sh.
"""
import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import time

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--prompt', required=True)
    ap.add_argument('--max-tokens', type=int, default=1200)
    ap.add_argument('--skip', type=int, default=32)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default='')
    ap.add_argument('--quant', default='')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    print(f'PID {os.getpid()}', flush=True)
    t0 = time.perf_counter()
    model, tokenizer = load(args.model)
    init_s = time.perf_counter() - t0
    messages = [{'role': 'user', 'content': args.prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    sampler = make_sampler(temp=0.0)
    stamps, chunks, last = [], [], None
    for r in stream_generate(model, tokenizer, prompt_ids, max_tokens=args.max_tokens,
                             sampler=sampler):
        now = time.perf_counter()
        if not stamps:
            print('DECODE_START', flush=True)
        stamps.append(now)
        chunks.append(r.text)
        last = r
        if not args.quiet:
            print(r.text, end='', flush=True)
    t_end = time.perf_counter()
    print('\nDECODE_END', flush=True)
    n = len(stamps)
    per_tok_ms = [(stamps[i] - stamps[i - 1]) * 1e3 for i in range(1, n)]
    skip = min(args.skip, max(n - 2, 0))
    steady_tps = (n - 1 - skip) / (stamps[-1] - stamps[skip]) if n - 1 - skip > 0 else None
    overall_tps = (n - 1) / (stamps[-1] - stamps[0]) if n > 1 else None
    st = sorted(per_tok_ms[skip:]) if n - 1 - skip > 0 else []
    snap = ''
    try:
        # HF-cache snapshot commit of the resolved model dir, for provenance.
        p = os.path.realpath(os.path.join(os.path.expanduser('~/.cache/huggingface/hub'),
                                          'models--' + args.model.replace('/', '--'), 'refs', 'main'))
        snap = open(p).read().strip() if os.path.exists(p) else ''
    except OSError:
        pass
    rec = {
        'tag': args.tag, 'runtime': 'mlx-lm',
        'runtime_version': importlib.metadata.version('mlx-lm'),
        'mlx_version': mx.__version__,
        'backend': 'gpu', 'model_path': args.model, 'hfRepoId': args.model, 'hfRevision': snap,
        'quantization': args.quant, 'prompt': args.prompt, 'max_tokens': args.max_tokens,
        'sampler': {'temperature': 0.0}, 'prompt_tokens': len(prompt_ids),
        'engine_init_s': round(init_s, 3),
        'generated_chunks': n, 'generated_text_chars': sum(len(c) for c in chunks),
        'generated_text_head': ''.join(chunks)[:200],
        'decode_tps_overall': round(overall_tps, 2) if overall_tps else None,
        'decode_tps_steady': round(steady_tps, 2) if steady_tps else None,
        'steady_window_tokens': n - 1 - skip,
        'per_token_ms_p50': round(st[len(st) // 2], 3) if st else None,
        'per_token_ms_p90': round(st[int(len(st) * 0.9)], 3) if st else None,
        'per_token_ms_max': round(st[-1], 3) if st else None,
        'decode_wall_s': round(t_end - stamps[0], 3) if n else None,
        'mlx_reported': {
            'prompt_tps': getattr(last, 'prompt_tps', None),
            'generation_tps': getattr(last, 'generation_tps', None),
            'generation_tokens': getattr(last, 'generation_tokens', None),
            'peak_memory_gb': getattr(last, 'peak_memory', None),
        },
        'per_token_ms': [round(x, 3) for x in per_tok_ms],
        'epoch_decode_start': time.time() - (t_end - stamps[0]) if n else None,
        'epoch_decode_end': time.time(),
        'host': {'machine': platform.machine(), 'macos': platform.mac_ver()[0],
                 'python': platform.python_version(),
                 'model': subprocess.run(['sysctl', '-n', 'hw.model'], capture_output=True,
                                         text=True).stdout.strip()},
    }
    with open(args.out, 'w') as f:
        json.dump(rec, f, separators=(',', ':'), sort_keys=True)
    print(f"RESULT steady_tps={rec['decode_tps_steady']} overall_tps={rec['decode_tps_overall']} "
          f"tokens={n} mlx_generation_tps={rec['mlx_reported']['generation_tps']} out={args.out}",
          flush=True)


if __name__ == '__main__':
    main()
