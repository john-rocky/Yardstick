#!/usr/bin/env python3
"""Parse xctrace-exported table XML (id/ref compressed) into per-row dicts.

Usage: parse_intervals.py <file.xml> [--pid PID] [--csv out.csv]
Prints summary stats for metal-gpu-intervals-like tables.
"""
import sys, xml.etree.ElementTree as ET
from collections import defaultdict


def load_rows(path):
    tree = ET.parse(path)
    root = tree.getroot()
    schema = root.find('.//schema')
    cols = [c.find('mnemonic').text for c in schema.findall('col')]
    # global id table for ref dereferencing
    byid = {}

    def register(el):
        i = el.get('id')
        if i is not None:
            byid[i] = el
        for ch in el:
            register(ch)

    def deref(el):
        r = el.get('ref')
        return byid[r] if r is not None else el

    rows = []
    for rowel in root.iter('row'):
        register(rowel)
    for rowel in root.iter('row'):
        vals = {}
        children = list(rowel)
        for i, ch in enumerate(children):
            ch = deref(ch)
            key = cols[i] if i < len(cols) else f'extra{i}'
            # value: numeric text if present else fmt
            txt = ch.text if ch.text and ch.text.strip() else None
            fmt = ch.get('fmt')
            # for formatted-label / process, dig for pid
            if ch.tag == 'process':
                pid = ch.find('pid')
                pid = deref(pid) if pid is not None else None
                vals[key + '_pid'] = int(pid.text) if pid is not None and pid.text else None
                vals[key] = fmt
            else:
                vals[key] = txt if txt is not None else fmt
        rows.append(vals)
    return cols, rows


if __name__ == '__main__':
    path = sys.argv[1]
    pid = None
    if '--pid' in sys.argv:
        pid = int(sys.argv[sys.argv.index('--pid') + 1])
    cols, rows = load_rows(path)
    print('cols:', cols)
    print('total rows:', len(rows))
    # filter by process pid if given
    if pid:
        rows = [r for r in rows if r.get('process_pid') == pid]
        print('rows for pid', pid, ':', len(rows))
    # channel breakdown
    chan = defaultdict(lambda: [0, 0])  # count, total dur ns
    for r in rows:
        c = r.get('channel-name') or '?'
        d = int(r.get('duration') or 0)
        chan[c][0] += 1
        chan[c][1] += d
    for c, (n, d) in sorted(chan.items(), key=lambda kv: -kv[1][1]):
        print(f'  {c:24s} n={n:6d} total={d/1e9:8.3f}s avg={d/n/1e3:9.1f}us')
