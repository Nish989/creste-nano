"""
train_steering.py
Direct steering regression — predicts human steering from BEV features.
Uses horizontal-flip augmentation to balance left/right turns.
"""

import sys, os
_user_sp = os.path.expanduser('~/Library/Python/3.12/lib/python/site-packages')
if _user_sp not in sys.path:
    sys.path.insert(0, _user_sp)

import argparse, glob, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ── Model: small CNN over BEV features ────────────────────────────────────────
class SteeringCNN(nn.Module):
    """
    Input:  BEV features  (B, 384, 64, 64)
    Output: steering      (B, 1)  in [-1, 1]
    """
    def __init__(self, feat_dim=384):
        super().__init__()
        self.conv1 = nn.Conv2d(feat_dim, 128, 3, stride=2, padding=1)  # 32x32
        self.bn1   = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64,  3, stride=2, padding=1)       # 16x16
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,  32,  3, stride=2, padding=1)       # 8x8
        self.bn3   = nn.BatchNorm2d(32)
        self.fc1   = nn.Linear(32 * 8 * 8, 128)
        self.fc2   = nn.Linear(128, 1)
        self.drop  = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return torch.tanh(self.fc2(x))


class DrivingDataset(Dataset):
    """
    Returns (BEV tensor (384, 64, 64), steering scalar).
    Each frame is included twice: original + horizontally flipped (with neg steer).
    """
    def __init__(self, feature_dir, augment=True):
        self.augment = augment
        self.files = []
        self.steers = []

        all_files = sorted(glob.glob(os.path.join(feature_dir, '*.npz')))
        print(f'Scanning {len(all_files)} files...')

        for f in all_files:
            try:
                d = np.load(f)
                bev = d['bev']
                # Skip empty BEVs (failed perception frames)
                if np.any(bev != 0, axis=2).sum() < 20:
                    continue
                self.files.append(f)
                self.steers.append(float(d['steering']))
            except Exception:
                continue

        print(f'Loaded {len(self.files)} valid frames')
        steers = np.array(self.steers)
        print(f'Steering distribution: min={steers.min():.3f} max={steers.max():.3f} '
              f'mean={steers.mean():.3f} std={steers.std():.3f}')
        print(f'Non-zero steering: {(steers != 0).sum()}/{len(steers)}')

    def __len__(self):
        # Double if augmenting (each frame twice: normal + flipped)
        return len(self.files) * (2 if self.augment else 1)

    def __getitem__(self, idx):
        if self.augment:
            real_idx = idx // 2
            flip = (idx % 2 == 1)
        else:
            real_idx = idx
            flip = False

        d   = np.load(self.files[real_idx])
        bev = d['bev'].astype(np.float32)             # (64, 64, 384)
        st  = float(self.steers[real_idx])

        if flip:
            bev = bev[:, ::-1, :].copy()              # flip left-right
            st  = -st                                  # invert steering

        # Channels-first for conv: (384, 64, 64)
        bev = np.transpose(bev, (2, 0, 1))
        return torch.from_numpy(bev), torch.tensor([st], dtype=torch.float32)


def get_device():
    if torch.cuda.is_available(): return torch.device('cuda')
    if hasattr(torch.backends,'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def train(args):
    device = get_device()
    print(f'Device: {device}')

    dataset = DrivingDataset(args.data_dir, augment=True)
    if len(dataset) == 0:
        print('No data'); return

    n_val   = max(int(0.1 * len(dataset)), 100)
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    model = SteeringCNN().to(device)
    opt   = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = float('inf')
    out_path = os.path.join(args.output_dir, 'steering_cnn.pth')
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        # ── Train ──
        model.train()
        tr_loss, n = 0.0, 0
        for bev, steer in train_loader:
            bev, steer = bev.to(device), steer.to(device)
            pred = model(bev)
            loss = F.mse_loss(pred, steer)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * bev.size(0); n += bev.size(0)
        tr_loss /= max(n, 1)

        # ── Validate ──
        model.eval()
        v_loss, vn = 0.0, 0
        preds, gts = [], []
        with torch.no_grad():
            for bev, steer in val_loader:
                bev, steer = bev.to(device), steer.to(device)
                pred = model(bev)
                v_loss += F.mse_loss(pred, steer, reduction='sum').item()
                vn += bev.size(0)
                preds.append(pred.cpu().numpy())
                gts.append(steer.cpu().numpy())
        v_loss /= max(vn, 1)
        preds = np.concatenate(preds).ravel()
        gts   = np.concatenate(gts).ravel()
        mask  = np.abs(gts) > 0.02
        corr_turns = (np.corrcoef(preds[mask], gts[mask])[0,1]
                      if mask.sum() > 5 else float('nan'))

        sched.step()
        print(f'epoch {epoch+1:3d}/{args.epochs}  train={tr_loss:.4f}  '
              f'val={v_loss:.4f}  corr_turns={corr_turns:+.3f}  lr={opt.param_groups[0]["lr"]:.1e}')

        if v_loss < best_val:
            best_val = v_loss
            torch.save(model.state_dict(), out_path)

    print(f'\nBest val loss: {best_val:.4f}')
    print(f'Saved best model → {out_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',   default=os.path.expanduser('~/Desktop/mapless_nav_data/bev_features'))
    p.add_argument('--output_dir', default=os.path.expanduser('~/Desktop/JOYDEEP/models/reward_model'))
    p.add_argument('--epochs',     type=int, default=40)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr',         type=float, default=1e-3)
    train(p.parse_args())


if __name__ == '__main__':
    main()
