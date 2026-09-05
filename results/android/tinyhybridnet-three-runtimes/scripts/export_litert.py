"""litert-torch (torch.export -> StableHLO -> LiteRT flatbuffer) -> .tflite, float32."""
import os
import torch
import litert_torch
from common import TFLITE, golden_input, load_model

model = load_model()
edge = litert_torch.convert(model, (golden_input(),))
edge.export(TFLITE)
print(f"litert_torch {getattr(litert_torch, '__version__', '?')} torch {torch.__version__}: wrote {TFLITE} ({os.path.getsize(TFLITE):,} B)")
