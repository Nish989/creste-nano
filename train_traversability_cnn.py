"""
train_traversability_cnn.py
Spatial CNN (small UNet) over BEV features.

Treats the 64×64×384 BEV as a multi-channel 2D semantic scene.
Learns: where is the road region in this scene?

Supervision = dilated trajectory mask:
  - The car's recorded steering rolls out an 8-step trajectory through the BEV
  - Dilate that line by ~3 pixels → road slab mask (positive)
  - Pixels OUTSIDE the slab that still have DINOv2 features = candidate negatives
  - Empty pixels (no DINOv2 features) are ignored — no signal there

Loss: weighted BCE — positives + sampled hard negatives only.
Augmentation: horizontal flip with steering negation.
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
import cv2


# ── Small UNet over BEV ───────────────────────────────────────────────────────
class TraversabilityUNet(nn.Module):
    """
    In:  (B, 384, 64, 64)
    Out: (B, 1, 64, 64)   pre-sigmoid logits
    """
    def __init__(self, in_ch=384, base=64):
        super().__init__()
        # Encoder
        self.enc1 = self._block(in_ch, base)          # 64×64
        self.pool1 = nn.MaxPool2d(2)                  # → 32×32
        self.enc2 = self._block(base, base*2)         # 32×32
        self.pool2 = nn.MaxPool2d(2)                  # → 16×16
        self.bot  = self._block(base*2, base*4)       # 16×16
        # Decoder
        self.up2  = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)  # → 32
        self.dec2 = self._block(base*4, base*2)
        self.up1  = nn.ConvTranspose2d(base*2, base, 2, stride=2)    # → 64
        self.dec1 = self._block(base*2, base)
        self.out  = nn.Conv2d(base, 1, 1)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b  = self.bot(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)                # (B, 1, 64, 64) logits


# ── Build the supervision mask ────────────────────────────────────────────────
def build_corridor_mask(steering, dilate_px=3, n_steps=20):
    """Dilated trajectory mask for the recorded steering. 64×64."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    x, y = 32.0, 63.0
    for _ in range(n_steps):
        y -= 1.0
        x += steering * 1.0
        if 0 <= int(y) < 64 and 0 <= int(x) < 64:
            mask[int(y), int(x)] = 1
        if y <= 0: break
    if mask.sum() > 0:
        k = 2*dilate_px + 1
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8), iterations=1)
    return mask  # (64, 64), 1 = road, 0 = unlabeled


class BEVSpatialDataset(Dataset):
    def __init__(self, feature_dir, augment=True):
        self.augment = augment
        self.files, self.steers = [], []
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

    def __len__(self):
        return len(self.files) * (2 if self.augment else 1)

    def __getitem__(self, idx):
        if self.augment:
            ri, flip = idx // 2, (idx % 2 == 1)
        else:
            ri, flip = idx, False

        d   = np.load(self.files[ri])
        bev = d['bev'].astype(np.float32)
        st  = float(self.steers[ri])

        if flip:
            bev = bev[:, ::-1, :].copy()
            st  = -st

        # Corridor mask (positives)
        pos = build_corridor_mask(st, dilate_px=3, n_steps=22)
        # "Has features" mask — pixels with any DINOv2 signal at all
        has_feat = (np.any(bev != 0, axis=2)).astype(np.uint8)
        # Hard negatives: has features AND not in corridor
        neg = has_feat & (1 - pos)

        # Channels-first tensor
        bev = np.transpose(bev, (2, 0, 1))   # (384, 64, 64)
        return (torch.from_numpy(bev),
                torch.from_numpy(pos.astype(np.float32)),    # (64,64)
                torch.from_numpy(neg.astype(np.float32)))    # (64,64)


def get_device():
    if torch.cuda.is_available(): return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def train(args):
    device = get_device()
    print(f'Device: {device}')

    dataset = BEVSpatialDataset(args.data_dir, augment=True)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=True, num_workers=0)

    model = TraversabilityUNet().to(device)
    opt   = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out_path = os.path.join(args.output_dir, 'traversability_unet.pth')
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total, n = 0.0, 0
        pos_mean_running, neg_mean_running = 0.0, 0.0
        for bev, pos_mask, neg_mask in loader:
            bev      = bev.to(device)
            pos_mask = pos_mask.to(device)
            neg_mask = neg_mask.to(device)

            logits = model(bev).squeeze(1)            # (B, 64, 64)
            probs  = torch.sigmoid(logits)

            # Weighted BCE only on labeled pixels
            eps = 1e-7
            loss_pos = -torch.log(probs + eps) * pos_mask
            loss_neg = -torch.log(1 - probs + eps) * neg_mask

            n_pos = pos_mask.sum() + eps
            n_neg = neg_mask.sum() + eps
            loss = loss_pos.sum() / n_pos + loss_neg.sum() / n_neg

            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); n += 1

            # Track avg pred score on each mask
            pos_mean_running += float((probs * pos_mask).sum() / n_pos)
            neg_mean_running += float((probs * neg_mask).sum() / n_neg)

        sched.step()
        avg_loss = total / max(n, 1)
        pos_avg  = pos_mean_running / max(n, 1)
        neg_avg  = neg_mean_running / max(n, 1)
        margin   = pos_avg - neg_avg
        print(f'epoch {epoch+1:3d}/{args.epochs}  loss={avg_loss:.4f}  '
              f'pos_pred={pos_avg:.3f}  neg_pred={neg_avg:.3f}  '
              f'margin={margin:+.3f}  lr={opt.param_groups[0]["lr"]:.1e}')

        torch.save(model.state_dict(), out_path)

    print(f'\nSaved → {out_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',   default=os.path.expanduser('~/Desktop/mapless_nav_data/bev_features'))
    p.add_argument('--output_dir', default=os.path.expanduser('~/Desktop/JOYDEEP/models/reward_model'))
    p.add_argument('--epochs',     type=int,   default=15)
    p.add_argument('--batch_size', type=int,   default=8)
    p.add_argument('--lr',         type=float, default=5e-4)
    train(p.parse_args())


if __name__ == '__main__':
    main()
