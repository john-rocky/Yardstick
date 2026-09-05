"""Host parity: ExecuTorch (python runtime, XNNPACK), LiteRT (ai_edge_litert Interpreter, XNNPACK CPU) and ONNX Runtime (CPU EP)
vs eager PyTorch on the golden input + 8 random inputs at three scales. Prints worst max|diff|, max rel, argmax agreement per runtime."""
import sys
import numpy as np
import torch
from common import GOLDEN_INPUT, GOLDEN_OUTPUT, INPUT_SHAPE, ONNX, PTE_XNNPACK, TFLITE, load_model

model = load_model()
x0 = torch.from_numpy(np.fromfile(GOLDEN_INPUT, dtype="<f4").reshape(INPUT_SHAPE))
golden = torch.from_numpy(np.fromfile(GOLDEN_OUTPUT, dtype="<f4").reshape(1, -1))
with torch.no_grad():
    assert torch.allclose(model(x0), golden, atol=1e-6), "golden stale"

torch.manual_seed(42)
inputs = [("golden", x0)] + [(f"random[{i}]", torch.randn(*INPUT_SHAPE) * [1.0, 3.0, 0.1][i % 3]) for i in range(8)]
refs = []
with torch.no_grad():
    refs = [model(x) for _, x in inputs]

runners = {}
try:
    from executorch.runtime import Runtime
    from importlib.metadata import version as _v
    m = Runtime.get().load_program(PTE_XNNPACK).load_method("forward")
    runners[f"ExecuTorch {_v('executorch')} (XNNPACK)"] = lambda x: m.execute([x])[0].numpy()
except Exception as e:
    print("executorch runner unavailable:", e)
try:
    from ai_edge_litert.interpreter import Interpreter
    import ai_edge_litert
    it = Interpreter(model_path=TFLITE)
    it.allocate_tensors()
    inp_det = it.get_input_details()[0]; out_det = it.get_output_details()[0]
    def run_tfl(x):
        it.set_tensor(inp_det["index"], x.numpy().astype(np.float32).reshape(inp_det["shape"]))
        it.invoke()
        return it.get_tensor(out_det["index"]).reshape(1, -1)
    runners[f"LiteRT {getattr(ai_edge_litert, '__version__', '?')} (Interpreter, CPU)"] = run_tfl
except Exception as e:
    print("litert runner unavailable:", e)
try:
    import onnxruntime as ort
    s = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    runners[f"ONNX Runtime {ort.__version__} (CPU EP)"] = lambda x: s.run(None, {"input": x.numpy()})[0]
except Exception as e:
    print("onnxruntime runner unavailable:", e)

print(f"{'runtime':45s} {'worst max|diff|':>16s} {'worst max rel':>14s} {'argmax agree':>12s}")
ok_all = True
for name, fn in runners.items():
    worst_abs = worst_rel = 0.0; agree = 0
    for (tag, x), ref in zip(inputs, refs):
        out = torch.from_numpy(np.asarray(fn(x), dtype=np.float32)).reshape(ref.shape)
        diff = (out - ref).abs()
        worst_abs = max(worst_abs, diff.max().item())
        worst_rel = max(worst_rel, (diff / ref.abs().clamp_min(1e-6)).max().item())
        agree += int(out.argmax(1).item() == ref.argmax(1).item())
    ok = worst_abs <= 1e-4 and agree == len(inputs)
    ok_all &= ok
    print(f"{name:45s} {worst_abs:16.3e} {worst_rel:14.3e} {agree:>9d}/{len(inputs)}  {'OK' if ok else 'FAIL'}")
print("PARITY OK (atol 1e-4)" if ok_all else "PARITY: at least one runtime failed atol 1e-4")
sys.exit(0 if ok_all else 1)
