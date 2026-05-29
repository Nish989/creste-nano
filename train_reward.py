"""
Train Reward MLP from precomputed depth-projected BEV features.

Runs on any machine (Mac/Linux/Windows) — no camera or Jetson needed.
Just needs the bev_features/*.npz files from precompute_bev.py.

Usage:
  python3 train_reward.py --data_dir ./bev_features --epochs 100

Positive examples: BEV crops along the human's driven trajectory (center-forward).
Negative examples: off-path trajectories (hard negatives from BEV edges + random).
"""
import argparse
import os
import glob
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class RewardMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class DrivingDataset(Dataset):
    """
    Loads precomputed depth-projected BEV features.
    Generates positive (on-path) and negative (off-path) trajectory samples.
    """
    def __init__(self, feature_dir, n_steps=8, feature_dim=384,
                 bev_h=64, bev_w=64, n_negatives=5):
        self.samples = []
        self.labels = []
        self.n_steps = n_steps
        self.feature_dim = feature_dim

        files = sorted(glob.glob(os.path.join(feature_dir, '*.npz')))
        print(f'Loading {len(files)} BEV feature files...')

        empty_count = 0
        for feat_file in files:
            data = np.load(feat_file)
            bev = data['bev']  # [64, 64, 384] depth-projected
            steering = float(data['steering'])

            # Check if BEV has enough non-zero cells to be useful
            occupied = np.any(bev != 0, axis=2).sum()
            if occupied < 20:
                empty_count += 1
                continue

            # --- Positive sample: trajectory the human actually drove ---
            # Car is at bottom-center, drove forward with recorded steering
            pos = self._sample_trajectory(bev, steering, bev_h, bev_w)
            if pos is not None:
                self.samples.append(pos)
                self.labels.append(1.0)

            # --- Negative samples ---
            for _ in range(n_negatives):
                neg_type = np.random.choice(['edge', 'random', 'opposite'])

                if neg_type == 'edge':
                    # Sample from BEV edges (off-path areas)
                    neg_steer = np.random.choice([-1.0, 1.0]) * np.random.uniform(0.6, 1.0)
                    neg = self._sample_trajectory(bev, neg_steer, bev_h, bev_w)
                elif neg_type == 'opposite':
                    # Steer opposite to the human
                    neg_steer = -steering + np.random.uniform(-0.3, 0.3)
                    neg_steer = np.clip(neg_steer, -1.0, 1.0)
                    neg = self._sample_trajectory(bev, neg_steer, bev_h, bev_w)
                else:
                    # Random trajectory
                    neg_steer = np.random.uniform(-1.0, 1.0)
                    neg = self._sample_trajectory(bev, neg_steer, bev_h, bev_w)

                if neg is not None:
                    self.samples.append(neg)
                    self.labels.append(0.0)

        if empty_count > 0:
            print(f'Skipped {empty_count} frames with near-empty BEV')

        self.samples = np.array(self.samples, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)

        n_pos = int(self.labels.sum())
        n_neg = len(self.labels) - n_pos
        print(f'Dataset: {len(self.labels)} samples ({n_pos} positive, {n_neg} negative)')

    def _sample_trajectory(self, bev, steering, bev_h, bev_w):
        """Extract BEV features along a trajectory defined by steering angle."""
        center_x = bev_w / 2
        start_y = bev_h - 1

        features = []
        x = float(center_x)
        y = float(start_y)
        for step in range(self.n_steps):
            y -= 1.5
            x += steering * 2.0
            by = int(np.clip(y, 0, bev_h - 1))
            bx = int(np.clip(x, 0, bev_w - 1))
            features.append(bev[by, bx])

        feat_vec = np.concatenate(features)
        # Skip if all features are zero (trajectory went through empty BEV cells)
        if np.abs(feat_vec).sum() < 1e-6:
            return None
        return feat_vec

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.samples[idx]), torch.tensor(self.labels[idx])


def get_device():
    """Pick best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def train(args):
    device = get_device()
    print(f'Training on {device}')

    dataset = DrivingDataset(
        args.data_dir,
        n_steps=8,
        feature_dim=384,
        n_negatives=args.n_negatives,
    )

    if len(dataset) == 0:
        print('No training data found!')
        print('Run precompute_bev.py on Jetson first, then copy bev_features/ here.')
        return

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # MPS doesn't like multiprocess dataloading
    )

    input_dim = 8 * 384  # n_steps * feature_dim
    model = RewardMLP(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    best_acc = 0.0
    for epoch in range(args.epochs):
        total_loss = 0
        correct = 0
        total = 0

        model.train()
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            pred = model(features).squeeze(-1)
            loss = criterion(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            correct += ((pred > 0.5).float() == labels).sum().item()
            total += len(labels)

        acc = correct / total
        avg_loss = total_loss / total

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch + 1}/{args.epochs}: loss={avg_loss:.4f} acc={acc:.3f}')

        if acc > best_acc:
            best_acc = acc

    print(f'\nBest accuracy: {best_acc:.3f}')

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, 'reward_mlp.pth')
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')
    print(f'\nCopy {save_path} to Jetson at ~/models/reward_model/reward_mlp.pth')


def main():
    parser = argparse.ArgumentParser(description='Train reward model from BEV features')
    parser.add_argument('--data_dir', required=True,
                        help='Path to bev_features/ directory with .npz files')
    parser.add_argument('--output_dir', default='./reward_model',
                        help='Where to save trained model')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_negatives', type=int, default=5,
                        help='Negative samples per positive frame')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
