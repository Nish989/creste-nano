"""
Contrastive Reward Learning from GPS-supervised BEV features.

Replaces binary cross-entropy MLP with InfoNCE contrastive loss.
GPS track automatically mines positive (on-path) and negative (off-path) samples —
no manual labeling required.

Mathematical framework:
  InfoNCE loss:
    L = -log[ exp(z_i . z+ / tau) / (exp(z_i . z+ / tau) + sum_j exp(z_i . z-_j / tau)) ]

  Where:
    z_i  = query BEV trajectory feature
    z+   = positive (GPS-confirmed on-path) trajectory feature
    z-_j = negative (off-path) trajectory features
    tau  = temperature (0.07)

GPS-supervised mining:
  Positive: BEV cells that GPS track actually drove over
  Negative: BEV cells never visited + hard negatives from BEV edges

Usage:
  python3 -m mapless_nav.train_reward --data_dir ./bev_features --epochs 100
"""
import argparse
import os
import glob
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TrajectoryEncoder(nn.Module):
    """
    Encodes a BEV trajectory (sequence of patch features) into a unit-norm
    embedding vector. Used for contrastive similarity computation.

    Input:  [batch, n_steps * feature_dim]
    Output: [batch, embed_dim] (L2-normalized)
    """
    def __init__(self, input_dim, hidden_dim=256, embed_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return F.normalize(z, dim=-1)  # L2 normalize → unit sphere


class RewardMLP(nn.Module):
    """
    Reward head: maps trajectory embedding to scalar reward score [0, 1].
    Used at inference time by reward_node.py.
    """
    def __init__(self, embed_dim=128, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class ContrastiveRewardModel(nn.Module):
    """
    Full model: encoder + reward head.
    Trained with InfoNCE loss, evaluated with reward head.
    """
    def __init__(self, input_dim, hidden_dim=256, embed_dim=128):
        super().__init__()
        self.encoder = TrajectoryEncoder(input_dim, hidden_dim, embed_dim)
        self.reward_head = RewardMLP(embed_dim)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encoder(x)
        return self.reward_head(z)


# ---------------------------------------------------------------------------
# InfoNCE Loss
# ---------------------------------------------------------------------------

def info_nce_loss(queries, positives, negatives, temperature=0.07):
    """
    InfoNCE contrastive loss.

    L = -log[ exp(q . k+ / tau) / (exp(q . k+ / tau) + sum_j exp(q . k-_j / tau)) ]

    Args:
        queries:   [B, D] query embeddings (current trajectory)
        positives: [B, D] positive embeddings (GPS on-path trajectory)
        negatives: [B, N, D] negative embeddings (off-path trajectories)
        temperature: tau — controls sharpness of distribution

    Returns:
        scalar loss
    """
    B, D = queries.shape
    N = negatives.shape[1]

    # Positive similarities: [B]
    pos_sim = (queries * positives).sum(dim=-1) / temperature

    # Negative similarities: [B, N]
    neg_sim = torch.bmm(negatives, queries.unsqueeze(-1)).squeeze(-1) / temperature

    # Log-softmax over [positive, negatives]
    # Concatenate: [B, 1+N]
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)

    # Labels: positive is always index 0
    labels = torch.zeros(B, dtype=torch.long, device=queries.device)

    loss = F.cross_entropy(logits, labels)
    return loss


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ContrastiveDrivingDataset(Dataset):
    """
    GPS-supervised contrastive dataset from precomputed BEV features.

    For each frame:
      Query   = trajectory feature at human's actual steering (GPS positive)
      Positive = nearby frame with similar steering (temporal positive)
      Negatives = N hard negatives: edge trajectories + random + opposite
    """
    def __init__(self, feature_dir, n_steps=8, feature_dim=384,
                 bev_h=64, bev_w=64, n_negatives=8, window=5):
        self.n_steps = n_steps
        self.feature_dim = feature_dim
        self.n_negatives = n_negatives

        files = sorted(glob.glob(os.path.join(feature_dir, '*.npz')))
        print(f'Loading {len(files)} BEV feature files...')

        self.bevs = []
        self.steerings = []
        skipped = 0

        for f in files:
            data = np.load(f)
            bev = data['bev']
            steering = float(data['steering'])

            occupied = np.any(bev != 0, axis=2).sum()
            if occupied < 20:
                skipped += 1
                continue

            self.bevs.append(bev.astype(np.float32))
            self.steerings.append(steering)

        print(f'Loaded {len(self.bevs)} frames, skipped {skipped} empty BEVs')
        self.window = window  # temporal window for positive mining

    def _sample_traj(self, bev, steering, bev_h=64, bev_w=64):
        """Extract flattened BEV features along a trajectory."""
        cx = float(bev_w / 2)
        sy = float(bev_h - 1)
        x, y = cx, sy
        feats = []
        for _ in range(self.n_steps):
            y -= 1.5
            x += steering * 2.0
            by = int(np.clip(y, 0, bev_h - 1))
            bx = int(np.clip(x, 0, bev_w - 1))
            feats.append(bev[by, bx])
        vec = np.concatenate(feats)
        if np.abs(vec).sum() < 1e-6:
            return None
        return vec

    def __len__(self):
        return len(self.bevs)

    def __getitem__(self, idx):
        bev = self.bevs[idx]
        steering = self.steerings[idx]

        # Query: current frame, human steering
        query = self._sample_traj(bev, steering)
        if query is None:
            query = np.zeros(self.n_steps * self.feature_dim, dtype=np.float32)

        # Positive: temporally nearby frame with similar steering (GPS-supervised)
        pos_idx = idx
        for _ in range(10):
            candidate = np.random.randint(
                max(0, idx - self.window),
                min(len(self.bevs), idx + self.window + 1)
            )
            if candidate != idx:
                pos_idx = candidate
                break
        pos = self._sample_traj(self.bevs[pos_idx], self.steerings[pos_idx])
        if pos is None:
            pos = query.copy()

        # Negatives: hard negatives from off-path trajectories
        negatives = []
        neg_steerings = []

        # Edge negatives (far from human steering)
        for _ in range(self.n_negatives // 2):
            neg_s = np.random.choice([-1.0, -0.8, 0.8, 1.0])
            neg_steerings.append(neg_s)

        # Random + opposite negatives
        neg_steerings.append(-steering + np.random.uniform(-0.2, 0.2))
        neg_steerings.append(np.random.uniform(-1.0, 1.0))
        while len(neg_steerings) < self.n_negatives:
            neg_steerings.append(np.random.uniform(-1.0, 1.0))

        for ns in neg_steerings[:self.n_negatives]:
            ns = np.clip(ns, -1.0, 1.0)
            neg = self._sample_traj(bev, ns)
            if neg is None:
                neg = np.zeros(self.n_steps * self.feature_dim, dtype=np.float32)
            negatives.append(neg)

        negatives = np.stack(negatives, axis=0)  # [N, D]

        return (
            torch.from_numpy(query.astype(np.float32)),
            torch.from_numpy(pos.astype(np.float32)),
            torch.from_numpy(negatives.astype(np.float32)),
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def train(args):
    device = get_device()
    print(f'Training on {device}')

    dataset = ContrastiveDrivingDataset(
        args.data_dir,
        n_steps=8,
        feature_dim=384,
        n_negatives=args.n_negatives,
        window=args.temporal_window,
    )

    if len(dataset) == 0:
        print('No training data found!')
        return

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=0)

    input_dim = 8 * 384  # n_steps * feature_dim
    model = ContrastiveRewardModel(
        input_dim=input_dim,
        hidden_dim=256,
        embed_dim=128,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    best_loss = float('inf')

    for epoch in range(args.epochs):
        total_loss = 0.0
        n_batches = 0

        model.train()
        for queries, positives, negatives in loader:
            queries = queries.to(device)      # [B, D]
            positives = positives.to(device)  # [B, D]
            negatives = negatives.to(device)  # [B, N, D]

            # Encode all
            q_emb = model.encode(queries)      # [B, embed_dim]
            p_emb = model.encode(positives)    # [B, embed_dim]

            B, N, D_in = negatives.shape
            neg_flat = negatives.view(B * N, D_in)
            n_emb = model.encode(neg_flat).view(B, N, -1)  # [B, N, embed_dim]

            # InfoNCE loss
            loss = info_nce_loss(q_emb, p_emb, n_emb, temperature=args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}/{args.epochs}: InfoNCE loss={avg_loss:.4f} '
                  f'lr={scheduler.get_last_lr()[0]:.6f}')

        if avg_loss < best_loss:
            best_loss = avg_loss

    print(f'\nBest InfoNCE loss: {best_loss:.4f}')

    # Save full model (encoder + reward head)
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, 'reward_mlp.pth')
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')
    print(f'Copy to Jetson: ~/models/reward_model/reward_mlp.pth')


def main():
    parser = argparse.ArgumentParser(
        description='Train contrastive reward model from BEV features')
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', default='./reward_model')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--n_negatives', type=int, default=8)
    parser.add_argument('--temporal_window', type=int, default=5)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
