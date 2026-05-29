"""
build_prototype.py
Computes a "prototype good-driving embedding" from the training data.
Saves it as prototype.npy next to reward_mlp.pth.

Run once:
  python3 build_prototype.py
"""

import sys, os
_user_sp = os.path.expanduser('~/Library/Python/3.12/lib/python/site-packages')
if _user_sp not in sys.path:
    sys.path.insert(0, _user_sp)

import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Model (must match reward_node.py exactly) ─────────────────────────────────
class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

class RewardHead(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(),
                                  nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, z): return self.net(z)

class RewardModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = TrajectoryEncoder(input_dim)
        self.head = RewardHead()
    def encode(self, x): return self.encoder(x)
    def forward(self, x): return self.head(self.encoder(x))

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR  = os.path.expanduser('~/Desktop/mapless_nav_data/bev_features')
MODEL_PTH = os.path.expanduser('~/Desktop/JOYDEEP/models/reward_model/reward_mlp.pth')
OUT_PATH  = os.path.expanduser('~/Desktop/JOYDEEP/models/reward_model/prototype.npy')
N_STEPS   = 8
FEAT_DIM  = 384
BATCH     = 256

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f'Device: {device}')

model = RewardModel(N_STEPS * FEAT_DIM).to(device)
model.load_state_dict(torch.load(MODEL_PTH, map_location=device, weights_only=True))
model.eval()
print(f'Model loaded from {MODEL_PTH}')

files = sorted(glob.glob(os.path.join(DATA_DIR, '*.npz')))
print(f'Processing {len(files)} frames...')

embeddings = []
batch_vecs = []

def flush(batch_vecs, embeddings):
    arr = np.stack(batch_vecs).astype(np.float32)
    with torch.no_grad():
        t = torch.from_numpy(arr).to(device)
        e = model.encode(t).cpu().numpy()
    embeddings.append(e)
    batch_vecs.clear()

for idx, f in enumerate(files):
    d = np.load(f)
    bev     = d['bev'].astype(np.float32)
    steer   = float(d['steering'])

    # Build GT trajectory vector
    x, y = 32.0, 63.0
    feats = []
    for _ in range(N_STEPS):
        y -= 1.5
        x += steer * 2.0
        feats.append(bev[int(np.clip(y, 0, 63)), int(np.clip(x, 0, 63))])
    vec = np.concatenate(feats)
    if np.abs(vec).sum() < 1e-6:
        continue
    batch_vecs.append(vec)

    if len(batch_vecs) >= BATCH:
        flush(batch_vecs, embeddings)

    if (idx + 1) % 1000 == 0:
        print(f'  {idx+1}/{len(files)}')

if batch_vecs:
    flush(batch_vecs, embeddings)

all_embeds = np.concatenate(embeddings, axis=0)   # (N, 128)
print(f'Collected {len(all_embeds)} embeddings, shape {all_embeds.shape}')

# Prototype = mean of all embeddings, re-normalised
prototype = all_embeds.mean(axis=0)
prototype = prototype / (np.linalg.norm(prototype) + 1e-8)
print(f'Prototype norm: {np.linalg.norm(prototype):.4f}')

np.save(OUT_PATH, prototype)
print(f'Saved prototype to {OUT_PATH}')
print('Done. Copy prototype.npy to the Jetson alongside reward_mlp.pth.')
