"""torch.onnx.export (TorchScript tracer, opset 18, static batch) -> .onnx; MHA fast path disabled so eval-mode attention lowers to primitive ops."""
import os
import onnx
import torch
from common import ONNX, golden_input, load_model

model = load_model()
torch.backends.mha.set_fastpath_enabled(False)
with torch.no_grad():
    torch.onnx.export(model, (golden_input(),), ONNX, input_names=["input"], output_names=["logits"],
                      opset_version=18, do_constant_folding=True, dynamo=False)
m = onnx.load(ONNX)
onnx.checker.check_model(m)
ops = {}
for n in m.graph.node:
    ops[n.op_type] = ops.get(n.op_type, 0) + 1
print(f"onnx {onnx.__version__} torch {torch.__version__}: wrote {ONNX} ({os.path.getsize(ONNX):,} B); {len(m.graph.node)} nodes: {dict(sorted(ops.items()))}")
