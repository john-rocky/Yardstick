#!/usr/bin/env python3
"""Summarize decode-driver JSON records in a results directory into one table.

Usage: summarize_runs.py <results_dir> [--md]
Groups files <cell>_r<N>.json by <cell>; prints n, median / min / max of the steady-state decode
rate, generated tokens, runtime version, artifact size, and provenance (hfRepoId@revision).
"""
import glob, json, os, re, statistics, sys

d = sys.argv[1]; md = '--md' in sys.argv
cells = {}
for f in sorted(glob.glob(os.path.join(d, '*.json'))):
    m = re.match(r'(.+)_r(\d+)\.json$', os.path.basename(f))
    if not m: continue
    rec = json.load(open(f)); rec['_round'] = int(m.group(2)); rec['_file'] = os.path.basename(f)
    cells.setdefault(m.group(1), []).append(rec)
rows = []
for cell, recs in cells.items():
    tps = [r['decode_tps_steady'] for r in recs if r.get('decode_tps_steady')]
    r0 = recs[0]
    ver = f"{r0.get('runtime')} {r0.get('runtime_version')}" + (f" / mlx {r0['mlx_version']}" if r0.get('mlx_version') else '')
    prov = f"{r0.get('hfRepoId','')}@{(r0.get('hfRevision') or '')[:8]}"
    size = r0.get('model_bytes')
    rows.append((cell, len(tps), statistics.median(tps), min(tps), max(tps), (max(tps) - min(tps)) / statistics.median(tps) * 100,
                 r0.get('generated_chunks'), ver, prov, f"{size/1e9:.3f} GB" if size else ''))
hdr = ('cell', 'n', 'median tok/s', 'min', 'max', 'spread %', 'tokens/run', 'runtime', 'artifact', 'size')
if md:
    print('| ' + ' | '.join(hdr) + ' |'); print('|' + '---|' * len(hdr))
    for r in rows: print('| ' + ' | '.join(f'{x:.1f}' if isinstance(x, float) else str(x) for x in r) + ' |')
else:
    print(f'{hdr[0]:32s} {hdr[1]:>2s} {hdr[2]:>12s} {hdr[3]:>7s} {hdr[4]:>7s} {hdr[5]:>8s} {hdr[6]:>10s}  {hdr[7]:28s} {hdr[8]:52s} {hdr[9]}')
    for r in rows:
        print(f'{r[0]:32s} {r[1]:2d} {r[2]:12.1f} {r[3]:7.1f} {r[4]:7.1f} {r[5]:8.1f} {str(r[6]):>10s}  {r[7]:28s} {r[8]:52s} {r[9]}')
