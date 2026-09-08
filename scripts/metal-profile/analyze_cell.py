#!/usr/bin/env python3
"""Per-cell decode decomposition from a metal-gpu-intervals export.

Usage: analyze_cell.py <gpu.xml> <tok_per_s> <weight_gb> [--proc SUBSTR]
Reports: busy%, encoders/token, big-kernel cluster stats, per-token busy ms,
in-kernel achieved GB/s (weights-only lower bound), idle ms/token.
"""
import sys, statistics
from collections import Counter
from parse_intervals import load_rows

path, toks, wgb = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
proc = None
if '--proc' in sys.argv:
    proc = sys.argv[sys.argv.index('--proc') + 1]

cols, rows = load_rows(path)
byproc = Counter(r.get('process') for r in rows)
if proc is None:
    # dominant non-system process
    cand = [p for p, _ in byproc.most_common() if p and 'WindowServer' not in p and 'cmux' not in p]
    if not cand:
        sys.exit(f'no non-system process in the capture; processes seen: {dict(byproc)}')
    proc = cand[0]
else:
    proc = next((p for p in byproc if p and proc in p), proc)
R = [r for r in rows if r.get('process') == proc]
R.sort(key=lambda r: int(r['start']))
iv = [(int(r['start']), int(r['start']) + int(r['duration'])) for r in R]
span = (iv[-1][1] - iv[0][0]) / 1e9
mb = 0
cs, ce = iv[0]
gaps = []
for s, e in iv[1:]:
    if s > ce:
        mb += ce - cs
        gaps.append(s - ce)
        cs, ce = s, e
    else:
        ce = max(ce, e)
mb += ce - cs
busy = mb / 1e9
tok_time_ms = 1000.0 / toks
n_tok = span * toks
durs = sorted(int(r['duration']) for r in R)
big = [d for d in durs if d > 100_000]  # >100us
blits = [r for r in R if 'Blit' in (r.get('event-label') or '')]
print(f'process                {proc}')
print(f'span                   {span:.3f} s  (~{n_tok:.0f} tokens)')
print(f'GPU busy               {busy:.3f} s  = {busy/span*100:.1f}%')
print(f'idle                   {(span-busy)*1000/n_tok:.2f} ms/token  ({(span-busy)/span*100:.1f}%)')
print(f'encoders/token         {len(R)/n_tok:.1f}   (blits/token {len(blits)/n_tok:.1f})')
print(f'cmdbufs/token          {len(set(r.get("cmdbuffer-id") for r in R))/n_tok:.1f}')
print(f'busy/token             {busy*1000/n_tok:.2f} ms')
print(f'in-kernel GB/s (>=)    {wgb/(busy/n_tok):.0f}  (weights {wgb} GB / busy-per-token)')
print(f'end-to-end GB/s        {wgb*toks:.0f}  = {wgb*toks/546*100:.0f}% of 546 ceiling')
print(f'big encoders (>100us)  {len(big)/n_tok:.1f}/token, sum {sum(big)/1e6/n_tok:.2f} ms/token')
if big:
    bigms = [d / 1e3 for d in big]
    print(f'   big dur us: p10 {bigms[int(len(bigms)*.1)]:.0f} p50 {bigms[len(bigms)//2]:.0f} p90 {bigms[int(len(bigms)*.9)]:.0f} max {bigms[-1]:.0f}')
if gaps:
    g = sorted(gaps)
    print(f'gaps: n/token {len(gaps)/n_tok:.1f}  sum {sum(gaps)/1e6/n_tok:.2f} ms/token  p50 {g[len(g)//2]/1e3:.1f}us p99 {g[int(len(g)*.99)]/1e3:.0f}us')
