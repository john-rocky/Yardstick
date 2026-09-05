"""Freeze one random TinyHybridNet (seed 0; BN stats and pos embedding randomised so every op is exercised)."""
import os
import torch
from common import ARTIFACTS, NUM_CLASSES, WEIGHTS, TinyHybridNet, GOLDEN_INPUT, GOLDEN_OUTPUT, golden_input

torch.manual_seed(0)
m = TinyHybridNet(num_classes=NUM_CLASSES)
with torch.no_grad():
    m.pos.normal_(std=0.02)
    for bn in (m.stem.bn1, m.stem.bn2, m.stem.bn3):
        bn.weight.uniform_(0.5, 1.5)
        bn.bias.normal_(std=0.1)
        bn.running_mean.normal_(std=0.5)
        bn.running_var.uniform_(0.5, 2.0)
os.makedirs(ARTIFACTS, exist_ok=True)
torch.save(m.state_dict(), WEIGHTS)
m.eval()
x = golden_input()
with torch.no_grad():
    y = m(x)
x.numpy().astype("<f4").tofile(GOLDEN_INPUT)
y.numpy().astype("<f4").tofile(GOLDEN_OUTPUT)
print(f"saved {WEIGHTS}: {sum(p.numel() for p in m.parameters()):,} params; golden logits {y.numpy()[0].round(4).tolist()}")
