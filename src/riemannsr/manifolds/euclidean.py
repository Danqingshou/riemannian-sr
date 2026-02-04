import torch
from .base import Manifold

class Euclidean(Manifold):
    def __init__(self, clamp: bool = True):
        self.clamp = clamp

    def proj(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(0.0, 1.0) if self.clamp else x

    def inner(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return (u * v).flatten(start_dim=1).sum(dim=1)

    def log(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (self.proj(y) - self.proj(x))

    def exp(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.proj(x + v)
