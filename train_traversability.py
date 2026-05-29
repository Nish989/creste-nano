"""
train_traversability.py

"""

import sys, os
_user_sp = os.path.expanduser('~/Library/Python/3.12/lib/python/site-packages')
if _user_sp not in sys.path:
    sys.path.insert(0, _user_sp)

import argparse, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ── Model ─────────────────────────────────────────────────────────────────────
class TraversabilityMLP(nn.Module):
    """Per-pixel scorer. Input: 384-dim DINOv2 feature. Output: scalar."""
    def __init__(self, feat_dim=384, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Dataset ───────────────────────────────────────────────────────────────────
class PixelPUDataset(Dataset):
    """
    Each item = one frame's BEV + steering.
    __getitem__ returns:
       pos_feats   (P, 384)  features at the trajectory pixels  (positives)
       unl_feats   (U, 384)  features at random pixels           (unlabeled)
    """
    def __init__(self, feature_dir, n_steps=12, n_unlabeled=64, augment=True):
        self.n_steps     = n_steps
        self.n_unlabeled = n_unlabeled
        self.augment     = augment
        self.files       = []
        self.steers      = []

        all_files = sorted(glob.glob(os.path.join(feature_dir, '*.npz')))
        print(f'Scanning {len(all_files)} files...')
        for f in all_files:
            try:
                d = np.load(f)
                if np.any(d['bev'] != 0, axis=2).sum() < 20:
                    continue
                self.files.append(f)
                self.steers.append(float(d['steering']))
            except Exception:
                continue
        print(f'Loaded {len(self.files)} valid frames')

    def _trajectory_pixels(self, steering):
        x, y = 32.0, 63.0
        pixels = []
        for _ in range(self.n_steps):
            y -= 1.5
            x += steering * 2.0
            pixels.append((int(np.clip(y, 0, 63)), int(np.clip(x, 0, 63))))
        return pixels

    def __len__(self):
        return len(self.files) * (2 if self.augment else 1)

    def __getitem__(self, idx):
        if self.augment:
            real_idx = idx // 2
            flip = (idx % 2 == 1)
        else:
            real_idx = idx
            flip = False

        d   = np.load(self.files[real_idx])
        bev = d['bev'].astype(np.float32)
        st  = float(self.steers[real_idx])

        if flip:
            bev = bev[:, ::-1, :].copy()
            st  = -st

        # Positive pixels: along the trajectory
        pos_yx  = self._trajectory_pixels(st)
        pos_feats = np.stack([bev[y, x, :] for (y, x) in pos_yx])

        # Unlabeled pixels: random sample from anywhere in the BEV that has features
        nz = np.any(bev != 0, axis=2)
        ys, xs = np.where(nz)
        if len(ys) > self.n_unlabeled:
            sel = np.random.choice(len(ys), self.n_unlabeled, replace=False)
            unl_feats = np.stack([bev[ys[i], xs[i], :] for i in sel])
        else:
            unl_feats = np.zeros((self.n_unlabeled, 384), dtype=np.float32)
            unl_feats[:len(ys)] = np.stack([bev[ys[i], xs[i], :] for i in range(len(ys))])

        return (torch.from_numpy(pos_feats.astype(np.float32)),
                torch.from_numpy(unl_feats.astype(np.float32)))


# ── nnPU loss ────────────────────────────────────────────────────────────────
def nnpu_loss(pos_scores, unl_scores, prior=0.1):
    """
    Non-negative Positive-Unlabeled loss (Kiryo et al. 2017).
    prior = estimated proportion of positives in the unlabeled set.
    """
    # ℓ = log(1 + exp(-z))   →   sigmoid logistic loss
    def lossfn(z, pos):
        # pos=True  → want z high → loss = log(1 + exp(-z))
        # pos=False → want z low  → loss = log(1 + exp( z))
        return F.softplus(-z) if pos else F.softplus(z)

    pos_pos_loss = lossfn(pos_scores,  True ).mean()
    pos_neg_loss = lossfn(pos_scores,  False).mean()
    unl_neg_loss = lossfn(unl_scores,  False).mean()

    neg_risk = unl_neg_loss - prior * pos_neg_loss
    if neg_risk < 0:                # clamp per nnPU
        return prior * pos_pos_loss - neg_risk * 0   # only positive part
    return prior * pos_pos_loss + neg_risk


def get_device():
    if torch.cuda.is_available(): return torch.device('cuda')
    if hasattr(torch.backends,'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def train(args):
    device = get_device()
    print(f'Device: {device}')

    dataset = PixelPUDataset(args.data_dir, augment=True)
    if len(dataset) == 0:
        print('No data'); return

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=0)
    model  = TraversabilityMLP().to(device)
    opt    = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'traversability_mlp.pth')

    for epoch in range(args.epochs):
        model.train()
        total, n = 0.0, 0
        margin_sum = 0.0
        for pos, unl in loader:
            pos = pos.to(device).view(-1, 384)   # (B*P, 384)
            unl = unl.to(device).view(-1, 384)   # (B*U, 384)
            ps = model(pos)
            us = model(unl)
            loss = nnpu_loss(ps, us, prior=args.prior)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); n += 1
            margin_sum += float(ps.mean() - us.mean())
        sched.step()
        avg_loss   = total / max(n,1)
        avg_margin = margin_sum / max(n,1)
        print(f'epoch {epoch+1:3d}/{args.epochs}  loss={avg_loss:+.4f}  '
              f'pos-unl margin={avg_margin:+.3f}  lr={opt.param_groups[0]["lr"]:.1e}')

        torch.save(model.state_dict(), out_path)

    print(f'\nSaved → {out_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',   default=os.path.expanduser('~/Desktop/mapless_nav_data/bev_features'))
    p.add_argument('--output_dir', default=os.path.expanduser('~/Desktop/JOYDEEP/models/reward_model'))
    p.add_argument('--epochs',     type=int,   default=25)
    p.add_argument('--batch_size', type=int,   default=64)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--prior',      type=float, default=0.1)
    train(p.parse_args())


if __name__ == '__main__':
    main()
