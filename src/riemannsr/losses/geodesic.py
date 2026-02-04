import torch
from torch import nn
from ..manifolds.base import Manifold

class GeodesicLoss(nn.Module):
    def __init__(self, manifold: Manifold, reduction: str = "mean"):
        super().__init__()
        self.manifold = manifold
        self.reduction = reduction

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # d_M(x,y) = ||log_x(y)||_x
        v = self.manifold.log(y_hat, y)
        d = self.manifold.norm(y_hat, v)  # shape: [B]
        loss = 0.5 * (d ** 2)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
