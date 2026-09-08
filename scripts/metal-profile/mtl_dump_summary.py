#!/usr/bin/env python3
"""Summarize a libmtl_dump capture: map pipelines to dumped MSL libraries, count dispatches per
kernel, show grid/threadgroup geometry and a rough per-kernel classification.

Usage: mtl_dump_summary.py <dump_dir> [--per-token N]  (N = dispatches per decode step, if known)
"""
import os, re, sys
from collections import Counter, defaultdict

d = sys.argv[1]
log = open(os.path.join(d, 'mtl_dump.log')).read().splitlines()
lib_of_pso = {}; last_lib = None; fn_of_pso = {}
for l in log:
    if l.startswith('LIB '):
        last_lib = int(l.split()[1])
    elif l.startswith('PSO '):
        m = re.match(r'PSO (0x[0-9a-f]+) fn=(\S+)', l)
        if m and last_lib is not None:
            lib_of_pso[m.group(1)] = last_lib; fn_of_pso[m.group(1)] = m.group(2)
disp = Counter(); geo = defaultdict(Counter); order = []
for l in log:
    if l.startswith('DISPATCH'):
        m = re.match(r'DISPATCH (tg|th) (0x[0-9a-f]+|0x0|\(nil\)) (grid|threads)=\(([\d,]+)\) tpg=\(([\d,]+)\)', l)
        if not m: continue
        pso = m.group(2); lib = lib_of_pso.get(pso, -1)
        disp[lib] += 1; geo[lib][(m.group(1), m.group(4), m.group(5))] += 1; order.append(lib)
print(f'libs {len(set(lib_of_pso.values()))}  psos {len(lib_of_pso)}  dispatches logged {len(order)}')


def classify(src):
    tags = []
    if re.search(r'& *0xfu?|>> *4u?|& *15u?', src): tags.append('4bit-unpack')
    if 'unpack_unorm' in src or 'as_type<half' in src: tags.append('half-pack')
    if re.search(r'\bhalf4\b|\bhalf\b', src): tags.append('half')
    if re.search(r'simd_sum|simd_shuffle|quad_sum', src): tags.append('simd-reduce')
    if 'threadgroup ' in src: tags.append('tg-mem')
    if re.search(r'atomic', src): tags.append('atomic')
    if 'exp(' in src or 'precise::exp' in src: tags.append('exp')
    if 'rsqrt' in src: tags.append('rsqrt')
    return ','.join(tags)

print(f'{"lib":>4} {"disp":>5} {"bytes":>6}  geometry (top)                                     tags')
for lib, n in sorted(disp.items(), key=lambda kv: -kv[1])[:40]:
    path = os.path.join(d, f'lib_{lib:04d}.metal')
    src = open(path).read() if os.path.exists(path) else ''
    g = geo[lib].most_common(2)
    gs = ' | '.join(f'{k[0]} {k[1]} tpg {k[2]} x{v}' for k, v in g)
    print(f'{lib:4d} {n:5d} {len(src):6d}  {gs:50s} {classify(src)}')
# per-step period: find the repeating cycle length in the dispatch order
if order:
    for period in range(20, 400):
        if len(order) >= 3 * period and order[:period] == order[period:2 * period] == order[2 * period:3 * period]:
            print(f'repeating dispatch period: {period} dispatches (first cycle libs: {order[:period]})'); break
    else:
        print('no clean repeating period found in the logged dispatches')
