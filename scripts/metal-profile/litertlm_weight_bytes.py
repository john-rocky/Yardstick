#!/usr/bin/env python3
"""Streamed-vs-stored weight bytes of a .litertlm artifact (needs the `tflite` schema package).

Finds the embedded TFLite flatbuffer(s) (magic 'TFL3' at offset 4), lists the largest constant
tensors of the decode subgraph with their consumers, and reports the bytes that a decode step
streams (all constants) minus gather-only tables (EMBEDDING_LOOKUP / GATHER inputs), which is
the "streamed GB" convention of docs/metal-profile-m4max.md follow-up 4.

Usage: litertlm_weight_bytes.py model.litertlm [--top 12]
"""
import mmap, struct, sys
import tflite

path = sys.argv[1]; top = int(sys.argv[sys.argv.index('--top') + 1]) if '--top' in sys.argv else 12
fh = open(path, 'rb')
mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
# locate TFL3 magics in the container header region only (the flatbuffer table lives at the
# front; scanning the whole file would just page in gigabytes of weights)
cands = []
pos = 0
while True:
    i = mm.find(b'TFL3', pos, min(len(mm), 64 << 20))
    if i < 0: break
    if i >= 4: cands.append(i - 4)
    pos = i + 4
print(f'file {path}\n  size {len(mm)/1e9:.3f} GB  tflite candidates at {cands[:6]}')
best = None
for off in cands:
    try:
        m = tflite.Model.GetRootAsModel(mm, off)
        n = m.SubgraphsLength()
        if n == 0: continue
        best = (off, m); break
    except Exception:
        continue
if not best:
    sys.exit('no parsable TFLite flatbuffer found')
off, model = best
bufs = [model.Buffers(i) for i in range(model.BuffersLength())]
def buf_len(i):
    b = bufs[i]
    return b.DataLength() if b.DataLength() else (b.Size() if hasattr(b, 'Size') and b.Size() else 0)
# decode subgraph = the one that references the most constant bytes (it is the one that also
# streams the output head); a subgraph literally named 'decode' wins ties
names = [model.Subgraphs(i).Name() for i in range(model.SubgraphsLength())]
def referenced_bytes(i):
    sgi = model.Subgraphs(i)
    return sum(buf_len(sgi.Tensors(t).Buffer()) for t in range(sgi.TensorsLength()))
sel = max(range(model.SubgraphsLength()), key=lambda i: (referenced_bytes(i), names[i] == b'decode'))
sg = model.Subgraphs(sel)
print(f'  subgraphs {model.SubgraphsLength()} -> using #{sel} {names[sel]!r}: tensors {sg.TensorsLength()} ops {sg.OperatorsLength()}')
opcodes = [model.OperatorCodes(i) for i in range(model.OperatorCodesLength())]
def opname(oc):
    code = oc.BuiltinCode() if oc.BuiltinCode() else oc.DeprecatedBuiltinCode()
    for k, v in vars(tflite.BuiltinOperator).items():
        if v == code and not k.startswith('_'): return k if code != 32 else (oc.CustomCode() or b'CUSTOM').decode()
    return str(code)
consumers = {}
for i in range(sg.OperatorsLength()):
    op = sg.Operators(i); nm = opname(opcodes[op.OpcodeIndex()])
    for j in range(op.InputsLength()):
        consumers.setdefault(op.Inputs(j), []).append(nm)
rows = []
total = 0; gather_only = 0
for t in range(sg.TensorsLength()):
    ten = sg.Tensors(t); L = buf_len(ten.Buffer())
    if L == 0: continue
    total += L
    cons = consumers.get(t, [])
    g = cons and all(c in ('EMBEDDING_LOOKUP', 'GATHER') for c in cons)
    if g: gather_only += L
    rows.append((L, ten.Name().decode(errors='replace'), ten.Type(), sorted(set(cons)), g))
rows.sort(reverse=True)
allbuf = sum(buf_len(i) for i in range(len(bufs)))
print(f'  all buffers in the flatbuffer: {allbuf/1e9:.3f} GB')
print(f'  constant bytes in subgraph: {total/1e9:.3f} GB; gather-only tables: {gather_only/1e9:.3f} GB; streamed per decode step: {(total-gather_only)/1e9:.3f} GB')
for L, nm, ty, cons, g in rows[:top]:
    print(f'   {L/1e6:9.1f} MB  type {ty:2d}  {"GATHER-ONLY" if g else "streamed   "}  {nm[:70]}  <- {",".join(cons)[:60]}')
