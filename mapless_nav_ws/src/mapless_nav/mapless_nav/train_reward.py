import argparse
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class RewardHead(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class RewardModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = TrajectoryEncoder(input_dim)
        self.head = RewardHead()

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.head(self.encoder(x))


def info_nce(q, k_pos, k_neg, tau=0.07):
    B = q.shape[0]
    pos_sim = (q * k_pos).sum(-1) / tau
    neg_sim = torch.bmm(k_neg, q.unsqueeze(-1)).squeeze(-1) / tau
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    labels = torch.zeros(B, dtype=torch.long, device=q.device)
    return F.cross_entropy(logits, labels)


class DrivingDataset(Dataset):
    def __init__(self, feature_dir, n_steps=8, n_neg=8, window=5):
        self.n_steps = n_steps
        self.n_neg = n_neg
        self.window = window
        self.bevs = []
        self.steerings = []

        files = sorted(glob.glob(os.path.join(feature_dir, '*.npz')))
        print(f'loading {len(files)} files...')

        for f in files:
            d = np.load(f)
            bev = d['bev']
            if np.any(bev != 0, axis=2).sum() < 20:
                continue
            self.bevs.append(bev.astype(np.float32))
            self.steerings.append(float(d['steering']))

        print(f'loaded {len(self.bevs)} frames')

    def _traj(self, bev, steer):
        x, y = float(32), float(63)
        feats = []
        for _ in range(self.n_steps):
            y -= 1.5
            x += steer * 2.0
            feats.append(bev[int(np.clip(y, 0, 63)), int(np.clip(x, 0, 63))])
        v = np.concatenate(feats)
        return None if np.abs(v).sum() < 1e-6 else v

    def __len__(self):
        return len(self.bevs)

    def __getitem__(self, idx):
        bev = self.bevs[idx]
        s = self.steerings[idx]

        q = self._traj(bev, s)
        if q is None:
            q = np.zeros(self.n_steps * 384, dtype=np.float32)

        # positive: nearby frame
        pos_idx = idx
        for _ in range(10):
            c = np.random.randint(max(0, idx - self.window), min(len(self.bevs), idx + self.window + 1))
            if c != idx:
                pos_idx = c
                break
        k_pos = self._traj(self.bevs[pos_idx], self.steerings[pos_idx])
        if k_pos is None:
            k_pos = q.copy()

        # negatives
        neg_steers = list(np.random.choice([-1.0, -0.8, 0.8, 1.0], self.n_neg // 2))
        neg_steers += [-s + np.random.uniform(-0.2, 0.2), np.random.uniform(-1, 1)]
        while len(neg_steers) < self.n_neg:
            neg_steers.append(np.random.uniform(-1, 1))

        negs = []
        for ns in neg_steers[:self.n_neg]:
            n = self._traj(bev, float(np.clip(ns, -1, 1)))
            negs.append(n if n is not None else np.zeros(self.n_steps * 384, dtype=np.float32))

        return (
            torch.from_numpy(q.astype(np.float32)),
            torch.from_numpy(k_pos.astype(np.float32)),
            torch.from_numpy(np.stack(negs).astype(np.float32)),
        )


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def train(args):
    device = get_device()
    print(f'device: {device}')

    dataset = DrivingDataset(args.data_dir, n_neg=args.n_negatives)
    if len(dataset) == 0:
        print('no data found')
        return

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = RewardModel(8 * 384).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = float('inf')
    for epoch in range(args.epochs):
        total, n = 0.0, 0
        model.train()
        for q, kp, kn in loader:
            q, kp, kn = q.to(device), kp.to(device), kn.to(device)
            B, N, D = kn.shape
            qe = model.encode(q)
            pe = model.encode(kp)
            ne = model.encode(kn.view(B * N, D)).view(B, N, -1)
            loss = info_nce(qe, pe, ne, tau=args.temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        sched.step()
        avg = total / max(n, 1)
        if (epoch + 1) % 10 == 0:
            print(f'epoch {epoch+1}/{args.epochs} loss={avg:.4f}')
        if avg < best:
            best = avg

    print(f'best loss: {best:.4f}')
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, 'reward_mlp.pth')
    torch.save(model.state_dict(), path)
    print(f'saved to {path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--output_dir', default='./reward_model')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--temperature', type=float, default=0.07)
    p.add_argument('--n_negatives', type=int, default=8)
    train(p.parse_args())


if __name__ == '__main__':
    main()
