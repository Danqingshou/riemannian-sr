from abc import ABC, abstractmethod
import torch

class Manifold(ABC):
    @abstractmethod
    def proj(self, x: torch.Tensor) -> torch.Tensor:
        """Project ambient point to manifold (if needed)."""
        raise NotImplementedError

    @abstractmethod
    def inner(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Inner product at point x between tangent vectors u, v.
        Returns tensor of shape compatible with reduction to batch scalar.
        """
        raise NotImplementedError

    @abstractmethod
    def log(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Logarithm map: tangent vector v at x such that exp_x(v) = y."""
        raise NotImplementedError

    @abstractmethod
    def exp(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exponential map: move from x along tangent vector v."""
        raise NotImplementedError

    def norm(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        ip = self.inner(x, v, v)
        return torch.sqrt(torch.clamp(ip, min=1e-12))

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        v = self.log(x, y)
        return self.norm(x, v)

    def retract(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Retraction approximating exp map (default: exp)."""
        return self.exp(x, v)
