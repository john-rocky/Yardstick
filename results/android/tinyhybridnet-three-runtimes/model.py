import torch
import torch.nn as nn


class ConvStem(nn.Module):
    def __init__(self, dim: int = 96):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, dim, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        return x  # (B, dim, 28, 28)


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 96, heads: int = 4, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class TinyHybridNet(nn.Module):
    """Conv stem -> tokens -> transformer block -> mean pool -> classifier."""

    def __init__(self, num_classes: int = 10, dim: int = 96):
        super().__init__()
        self.stem = ConvStem(dim)
        self.pos = nn.Parameter(torch.zeros(1, 28 * 28, dim))
        self.block = EncoderBlock(dim)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):  # x: (B, 3, 224, 224)
        x = self.stem(x)
        x = x.flatten(2).transpose(1, 2)  # (B, 784, dim)
        x = x + self.pos
        x = self.block(x)
        x = self.norm(x).mean(dim=1)
        return self.head(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = TinyHybridNet().eval()
    with torch.no_grad():
        y = m(torch.randn(1, 3, 224, 224))
    print(y.shape)
