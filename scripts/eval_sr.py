import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import sys
# add repository src to sys.path for local imports
repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from riemannsr.models.srnet import SRNet
from riemannsr.manifolds.euclidean import Euclidean
from riemannsr.data.folder_dataset import SRFolderDataset
from riemannsr.utils.image_ops import psnr


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--scale', type=int, default=4)
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--save_dir', type=str, default='outputs/eval')
    return p.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    val_set = SRFolderDataset(args.data_root, split='val', scale=args.scale, patch_size=0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)

    model = SRNet(scale=args.scale).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    psnr_sum = 0.0
    with torch.no_grad():
        for i, (lr, hr) in enumerate(tqdm(val_loader)):
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            psnr_sum += psnr(sr, hr).item()
            save_image(sr, save_dir / f"sr_{i:04d}.png")
    print(f"Val PSNR: {psnr_sum/len(val_loader):.2f}dB")

if __name__ == '__main__':
    main()
