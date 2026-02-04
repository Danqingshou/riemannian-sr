import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, 1, 1),
        )
    def forward(self, x):
        return x + self.body(x)

class SRNet(nn.Module):
    def __init__(self, scale: int = 4, in_ch: int = 3, feat: int = 64, num_blocks: int = 8):
        super().__init__()
        self.head = nn.Conv2d(in_ch, feat, 3, 1, 1)
        self.body = nn.Sequential(*[ResidualBlock(feat) for _ in range(num_blocks)])
        # upsampling by pixel shuffle
        up_layers = []
        s = scale
        while s > 1:
            up_layers += [nn.Conv2d(feat, feat*4, 3, 1, 1), nn.PixelShuffle(2), nn.ReLU(inplace=True)]
            s //= 2
        self.upsample = nn.Sequential(*up_layers)
        self.tail = nn.Conv2d(feat, in_ch, 3, 1, 1)

    def forward(self, x):
        f = self.head(x)
        f = self.body(f)
        f = self.upsample(f)
        y = self.tail(f)
        return y.clamp(0,1)
