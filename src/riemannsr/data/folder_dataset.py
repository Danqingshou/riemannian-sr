from pathlib import Path
from typing import Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class SRFolderDataset(Dataset):
    def __init__(self, root: str, split: str = "train", scale: int = 4, patch_size: int = 96):
        self.root = Path(root)
        self.lr_dir = self.root / split / "LR"
        self.hr_dir = self.root / split / "HR"
        self.scale = scale
        self.patch_size = patch_size
        self.lr_paths = sorted([p for p in self.lr_dir.glob("*.png")])
        assert len(self.lr_paths) > 0, f"No LR images found in {self.lr_dir}"
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.lr_paths)

    def _pair(self, lr_img: Image.Image) -> Image.Image:
        # Load corresponding HR by filename
        name = Path(lr_img.filename).name
        hr_path = self.hr_dir / name
        hr_img = Image.open(hr_path).convert("RGB")
        return hr_img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lr_path = self.lr_paths[idx]
        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = self._pair(lr_img)
        if self.patch_size > 0:
            # random crop paired
            import random
            w, h = lr_img.size
            ps = self.patch_size
            x = random.randint(0, max(0, w-ps))
            y = random.randint(0, max(0, h-ps))
            lr_crop = lr_img.crop((x,y,x+ps,y+ps))
            hr_crop = hr_img.crop((x*self.scale, y*self.scale, (x+ps)*self.scale, (y+ps)*self.scale))
            lr_img, hr_img = lr_crop, hr_crop
        lr = self.to_tensor(lr_img)
        hr = self.to_tensor(hr_img)
        return lr, hr
