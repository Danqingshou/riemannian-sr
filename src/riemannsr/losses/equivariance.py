import torch
from torch import nn
import torchvision.transforms.functional as TF

_ROTATIONS = [0, 90, 180, 270]

def rotate_tensor(x: torch.Tensor, angle: int) -> torch.Tensor:
    # x: [B,C,H,W], use nearest to keep pixel grid
    imgs = [TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR) for img in x]
    return torch.stack(imgs, dim=0)

class EquivarianceLoss(nn.Module):
    def __init__(self, model: nn.Module, weight: float = 0.1, sample_one: bool = True):
        super().__init__()
        self.model = model
        self.weight = weight
        self.sample_one = sample_one
        self.criterion = nn.MSELoss(reduction="mean")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight <= 0:
            return x.new_tensor(0.0)
        angles = [_ROTATIONS[torch.randint(0, 4, ()).item()]] if self.sample_one else _ROTATIONS
        loss = 0.0
        with torch.no_grad():
            y0 = self.model(x)
        for a in angles:
            xg = rotate_tensor(x, a)
            yg = self.model(xg)
            y0g = rotate_tensor(y0, a)
            loss = loss + self.criterion(yg, y0g)
        loss = loss / len(angles)
        return self.weight * loss
