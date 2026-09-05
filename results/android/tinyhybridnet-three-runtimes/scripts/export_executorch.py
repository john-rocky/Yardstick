"""torch.export -> to_edge_transform_and_lower(XnnpackPartitioner) -> .pte (plus an undelegated portable .pte)."""
import os
import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from importlib.metadata import version as _v
from common import PTE_PORTABLE, PTE_XNNPACK, golden_input, load_model


def export(model, example, partitioners):
    ep = torch.export.export(model, (example,), strict=True)
    edge = to_edge_transform_and_lower(ep, partitioner=partitioners,
                                       compile_config=EdgeCompileConfig(_check_ir_validity=False))
    return edge.to_executorch()


def summarize(et, title):
    gm = et.exported_program().graph_module
    ops, delegates = {}, 0
    for node in gm.graph.nodes:
        if node.op == "call_function":
            name = getattr(node.target, "__name__", str(node.target))
            if name.startswith("executorch_call_delegate"):
                delegates += 1
            else:
                ops[name] = ops.get(name, 0) + 1
    print(f"[{title}] delegate calls: {delegates}; non-delegated ops: {ops or '{}'}")


model = load_model()
example = golden_input()
et = export(model, example, [XnnpackPartitioner()])
summarize(et, "xnnpack")
open(PTE_XNNPACK, "wb").write(et.buffer)
et_p = export(model, example, [])
summarize(et_p, "portable")
open(PTE_PORTABLE, "wb").write(et_p.buffer)
print(f"executorch {_v("executorch")} torch {torch.__version__}: wrote {PTE_XNNPACK} ({os.path.getsize(PTE_XNNPACK):,} B), {PTE_PORTABLE} ({os.path.getsize(PTE_PORTABLE):,} B)")
