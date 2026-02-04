import torch
import torchvision.transforms.functional as TF

def psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    mse = torch.mean((x - y) ** 2)
    if mse <= 1e-12:
        return torch.tensor(99.0, device=x.device)
    return 20 * torch.log10(torch.tensor(data_range, device=x.device)) - 10 * torch.log10(mse)

