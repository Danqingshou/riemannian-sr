# Riemannian-SR: Super-Resolution with Manifold Geometry and Group Equivariance

This codebase implements the ideas from your article:
- Geodesic loss on a Riemannian manifold `M` instead of pixel-wise MSE.
- Group-equivariant regularization (rotations) to enforce geometric consistency.
- Modular manifold interfaces with exp/log maps and retractions.
- A minimal SR model and training/eval scripts.

## Features
- Manifold API: `exp_x(v)`, `log_x(y)`, `inner_x(u,v)`, `retract_x(v)`, `proj(x)`.
- Geodesic loss: `L_Geo = 1/2 * d_M(y, y_hat)^2`, where `d_M(x,y) = ||log_x(y)||_x`.
- Equivariance regularizer: `E_g || F(T_g x) - T_g F(x) ||^2` using 90° rotations by default.
- Pluggable manifolds: start with Euclidean (baseline) and extend to custom manifolds.

## Repo Structure
- `src/riemannsr/manifolds/`: base and example manifolds.
- `src/riemannsr/losses/`: geodesic and equivariance losses.
- `src/riemannsr/models/`: simple SR network.
- `src/riemannsr/data/`: folder dataset (paired LR/HR).
- `scripts/`: training and evaluation.

## Install
```bash
pip install -r requirements.txt
```

## Data
Organize dataset as:
```
DATA_ROOT/
  train/
    LR/*.png  # low-res images
    HR/*.png  # high-res images (same filenames)
  val/
    LR/*.png
    HR/*.png
```

## Train
```bash
python scripts/train_sr.py \
  --data_root PATH_TO_DATA \
  --scale 4 \
  --epochs 200 \
  --batch_size 16 \
  --lr 2e-4 \
  --manifold euclidean \
  --lambda_eq 0.1 \
  --save_dir outputs/exp1
```

## Evaluate
```bash
python scripts/eval_sr.py --data_root PATH_TO_DATA --scale 4 --ckpt outputs/exp1/last.pt
```

## Extending Manifolds
Add a new file in `src/riemannsr/manifolds/` and implement the `Manifold` interface. Then select it via `--manifold`.

## GitHub
Initialize and push:
```bash
git init
git add .
git commit -m "Init Riemannian-SR"
# create repo on GitHub named riemannian-sr, then
git remote add origin https://github.com/<yourname>/riemannian-sr.git
git branch -M main
git push -u origin main
```
