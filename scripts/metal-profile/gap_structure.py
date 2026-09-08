#!/usr/bin/env python3
"""Where does the GPU idle inside one decode token? (companion to analyze_cell.py)

Usage: gap_structure.py <gpu.xml> <tok_per_s> [--proc SUBSTR] [--min-gap-us 50]
Lists the idle gaps > min-gap-us between consecutive GPU intervals of the process, keyed by
the label of the interval BEFORE the gap (what the GPU finished last) — so a two-bubble token
shows up as two dominant "after <label>" keys. Also prints per-token cadence and the
distribution of encoder durations for the biggest label families.
"""
import sys
from collections import Counter, defaultdict
from parse_intervals import load_rows

path, toks = sys.argv[1], float(sys.argv[2])
proc = sys.argv[sys.argv.index('--proc') + 1] if '--proc' in sys.argv else None
min_gap = float(sys.argv[sys.argv.index('--min-gap-us') + 1]) if '--min-gap-us' in sys.argv else 50.0
cols, rows = load_rows(path)
byproc = Counter(r.get('process') for r in rows)
if proc is None:
    cand = [p for p, _ in byproc.most_common() if p and 'WindowServer' not in p and 'cmux' not in p]
    proc = cand[0]
else:
    proc = next(p for p in byproc if p and proc in p)
R = sorted((r for r in rows if r.get('process') == proc), key=lambda r: int(r['start']))
span = (int(R[-1]['start']) + int(R[-1]['duration']) - int(R[0]['start'])) / 1e9
n_tok = span * toks
print(f'process {proc}   span {span:.3f}s  ~{n_tok:.0f} tokens   rows {len(R)}')


def fam(label):
    l = (label or '').strip()
    return l[:48]

# merged-busy gap analysis keyed by the label that precedes the gap
gaps = defaultdict(list)
ce = int(R[0]['start']) + int(R[0]['duration']); last_label = R[0].get('event-label')
for r in R[1:]:
    s, e = int(r['start']), int(r['start']) + int(r['duration'])
    if s > ce and (s - ce) / 1e3 >= min_gap:
        gaps[fam(last_label)].append((s - ce) / 1e3)
    if e >= ce:
        ce = e; last_label = r.get('event-label')
tot = sum(sum(v) for v in gaps.values())
print(f'gaps >= {min_gap:.0f} us: total {tot/1e3/n_tok:.3f} ms/token over {sum(len(v) for v in gaps.values())/n_tok:.2f} gaps/token')
for k, v in sorted(gaps.items(), key=lambda kv: -sum(kv[1]))[:8]:
    v = sorted(v)
    print(f'  after {k!r:52s} n/tok {len(v)/n_tok:4.2f}  sum {sum(v)/1e3/n_tok:.3f} ms/tok  p50 {v[len(v)//2]:6.0f} us  p90 {v[int(len(v)*.9)]:6.0f} us')

# encoder families by total time
fams = defaultdict(list)
for r in R:
    fams[fam(r.get('event-label'))].append(int(r['duration']) / 1e3)
print('encoder families (by GPU time):')
for k, v in sorted(fams.items(), key=lambda kv: -sum(kv[1]))[:10]:
    v = sorted(v)
    print(f'  {k!r:52s} n/tok {len(v)/n_tok:5.2f}  {sum(v)/1e3/n_tok:6.3f} ms/tok  p50 {v[len(v)//2]:7.1f} us  p90 {v[int(len(v)*.9)]:7.1f} us')
chan = Counter(r.get('channel-name') for r in R)
print('channels:', dict(chan))
