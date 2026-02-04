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
from riemannsr.losses.geodesic import GeodesicLoss
from riemannsr.losses.equivariance import EquivarianceLoss
from riemannsr.data.folder_dataset import SRFolderDataset
from riemannsr.utils.image_ops import psnr


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--scale', type=int, default=4)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--patch', type=int, default=96)
    p.add_argument('--lambda_eq', type=float, default=0.1)
    p.add_argument('--save_dir', type=str, default='outputs/exp1')
    p.add_argument('--num_workers', type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_set = SRFolderDataset(args.data_root, split='train', scale=args.scale, patch_size=args.patch)
    val_set = SRFolderDataset(args.data_root, split='val', scale=args.scale, patch_size=0)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)

    model = SRNet(scale=args.scale).to(device)
    manifold = Euclidean()
    geo_loss = GeodesicLoss(manifold)
    eq_loss = EquivarianceLoss(model, weight=args.lambda_eq)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    save_dir = Path(args.save_dir)
    (save_dir / 'ckpts').mkdir(parents=True, exist_ok=True)

    best_psnr = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for lr, hr in pbar:
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            loss_geo = geo_loss(sr, hr)
            loss_eq = eq_loss(lr)
            loss = loss_geo + loss_eq
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix({'L_geo': f'{loss_geo.item():.4f}', 'L_eq': f'{loss_eq.item():.4f}'})

        # validation
        model.eval()
        psnr_sum = 0.0
        with torch.no_grad():
            for lr, hr in val_loader:
                lr, hr = lr.to(device), hr.to(device)
                sr = model(lr)
                psnr_sum += psnr(sr, hr).item()
        psnr_avg = psnr_sum / len(val_loader)
        print(f"Val PSNR: {psnr_avg:.2f}dB")
        if psnr_avg > best_psnr:
            best_psnr = psnr_avg
            torch.save({'epoch': epoch, 'model': model.state_dict()}, save_dir / 'ckpts' / 'best.pt')
        torch.save({'epoch': epoch, 'model': model.state_dict()}, save_dir / 'ckpts' / 'last.pt')

    print('Training done.')

if __name__ == '__main__':
    main()
