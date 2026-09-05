"""Shared paths/helpers for the B3 three-runtime export (TinyHybridNet from tasks/assets/T06/model.py)."""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model import TinyHybridNet  # noqa: E402

ARTIFACTS = os.path.join(ROOT, "artifacts")
WEIGHTS = os.path.join(ARTIFACTS, "tinyhybridnet_weights.pt")
PTE_XNNPACK = os.path.join(ARTIFACTS, "tinyhybridnet_xnnpack.pte")
PTE_PORTABLE = os.path.join(ARTIFACTS, "tinyhybridnet_portable.pte")
ONNX = os.path.join(ARTIFACTS, "tinyhybridnet.onnx")
TFLITE = os.path.join(ARTIFACTS, "tinyhybridnet.tflite")
GOLDEN_INPUT = os.path.join(ARTIFACTS, "golden_input_1x3x224x224_f32.bin")
GOLDEN_OUTPUT = os.path.join(ARTIFACTS, "golden_output_1x10_f32.bin")
INPUT_SHAPE = (1, 3, 224, 224)
NUM_CLASSES = 10


def load_model() -> TinyHybridNet:
    m = TinyHybridNet(num_classes=NUM_CLASSES)
    m.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    return m.eval()


def golden_input() -> torch.Tensor:
    torch.manual_seed(1234)
    return torch.randn(*INPUT_SHAPE)
