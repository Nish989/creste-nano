import sys, os
# Ensure user site-packages is on path (needed when run via /Library/Frameworks Python)
_user_sp = os.path.expanduser('~/Library/Python/3.12/lib/python/site-packages')
if _user_sp not in sys.path:
    sys.path.insert(0, _user_sp)

"""
CREStE-Nano Software Simulator
Replays recorded BEV features through the MPPI planner + reward model
exactly as the real car would experience them — no hardware needed.

Usage:
  python3 simulate.py
  python3 simulate.py --data_dir ~/Desktop/mapless_nav_data/bev_features
  python3 simulate.py --speed 2.0   # run at 2x speed
"""

import argparse
import os
import glob
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import matplotlib
matplotlib.use('TkAgg')  # works on Mac
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# ── Models (must match reward_node.py exactly) ─────────────────────────────────

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
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
    def forward(self, z): return self.net(z)

class RewardModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = TrajectoryEncoder(input_dim)
        self.head = RewardHead()
    def encode(self, x): return self.encoder(x)
    def forward(self, x): return self.head(self.encoder(x))


# ── MPPI Planner (matches planner_node.py) ────────────────────────────────────

class MPPIPlanner:
    def __init__(self, K=1000, T=16, sigma=0.35, lam=0.1,
                 bev_w=128, bev_h=128, max_steer=1.0,
                 step_size=4.0, lateral_scale=4.0,
                 auto_throttle=0.2, wp_bias=0.3, momentum=0.8):
        self.K, self.T = K, T
        self.sigma, self.lam = sigma, lam
        self.bev_w, self.bev_h = bev_w, bev_h
        self.max_steer = max_steer
        self.step_size = step_size        # px forward per step
        self.lateral_scale = lateral_scale  # px lateral per steer unit per step
        self.auto_throttle = auto_throttle
        self.wp_bias = wp_bias
        self.momentum = momentum
        self.nominal_U = np.zeros(T)

    def sample(self):
        eps = np.random.normal(0, self.sigma, (self.K, self.T))
        U = np.clip(self.nominal_U[None, :] + eps, -self.max_steer, self.max_steer)
        cands = np.zeros((self.K, self.T, 2))
        x = np.full(self.K, float(self.bev_w / 2))
        y = np.full(self.K, float(self.bev_h - 1))
        for t in range(self.T):
            y = y - self.step_size
            x = x + U[:, t] * self.lateral_scale
            cands[:, t, 0] = y
            cands[:, t, 1] = x
        return U, eps, cands

    def update(self, scores, epsilons, waypoint_bearing=None):
        if waypoint_bearing is not None:
            bn = np.clip(waypoint_bearing / 90.0, -1.0, 1.0)
            first_steers = self.nominal_U[0] + epsilons[:, 0]
            scores = scores + self.wp_bias * (1.0 - np.abs(first_steers - bn))
        s = scores - scores.max()
        w = np.exp(s / self.lam)
        w /= (w.sum() + 1e-8)
        self.nominal_U = np.clip(
            self.momentum * self.nominal_U + np.einsum('k,kt->t', w, epsilons),
            -self.max_steer, self.max_steer
        )
        action = float(self.nominal_U[0])
        self.nominal_U = np.roll(self.nominal_U, -1)
        self.nominal_U[-1] = 0.0
        return action


# ── Score candidates using reward model ───────────────────────────────────────

def score_candidates(model, bev, candidates, device, gt_steer=None, prototype=None, n_steps=None, feat_dim=384):
    """
    Score MPPI candidates using the trained encoder.

    The RewardHead was never trained (only the encoder was, via InfoNCE).
    Instead we score by cosine similarity between each candidate's embedding
    and the ground-truth trajectory embedding for this frame.  The encoder was
    explicitly trained to put similar (nearby-steering) trajectories close
    together, so this gives a meaningful reward signal.

    Fallback: if gt_steer is None, score by average BEV feature magnitude
    along each path (road pixels have non-zero DINOv2 activations).
    """
    K = candidates.shape[0]
    T = candidates.shape[1]
    if n_steps is None:
        n_steps = T
    bev_max = bev.shape[0] - 1
    by = np.clip(candidates[:, :, 0].astype(int), 0, bev_max)
    bx = np.clip(candidates[:, :, 1].astype(int), 0, bev_max)
    feats = bev[by, bx, :].reshape(K, T * feat_dim)

    # Pure feature-magnitude scoring — no model needed, works with any T
    feat_norms = np.linalg.norm(bev[by, bx, :], axis=-1)  # (K, T)
    raw_score  = feat_norms.mean(axis=1)                   # (K,)

    with torch.no_grad():
        if prototype is not None and T == 8:
            # Blend in prototype cosine similarity when model input size matches
            t = torch.from_numpy(feats.astype(np.float32)).to(device)
            embeds = model.encode(t)
            proto_sim = (embeds * prototype).sum(dim=-1).cpu().numpy()
            scores = 0.8 * raw_score / (raw_score.max() + 1e-8) + 0.2 * (proto_sim + 1) / 2
        else:
            # T≠8 or no prototype: pure feature magnitude
            scores = raw_score / (raw_score.max() + 1e-8)

        if gt_steer is not None and T == 8:
            # Build the ground-truth trajectory from the recorded steering
            x, y = 32.0, 63.0
            gt_feat_list = []
            for _ in range(n_steps):
                y -= 1.5
                x += gt_steer * 2.0
                gt_feat_list.append(
                    bev[int(np.clip(y, 0, 63)), int(np.clip(x, 0, 63))]
                )
            gt_vec = np.concatenate(gt_feat_list).astype(np.float32)
            gt_t   = torch.from_numpy(gt_vec).unsqueeze(0).to(device)
            gt_emb = model.encode(gt_t)   # (1, 128) — L2-normalised
            scores = (embeds * gt_emb).sum(dim=-1).cpu().numpy()  # cosine sim
        else:
            # Fallback: average feature magnitude along path
            feat_norms = np.linalg.norm(bev[by, bx, :], axis=-1)  # (K, T)
            scores = feat_norms.mean(axis=1)

    return scores


# ── GPS helpers ───────────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def bearing_to(lat1, lon1, lat2, lon2):
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(math.radians(lat2))
    y = math.cos(math.radians(lat1))*math.sin(math.radians(lat2)) - \
        math.sin(math.radians(lat1))*math.cos(math.radians(lat2))*math.cos(dlon)
    b = math.degrees(math.atan2(x, y)) % 360
    return b


# ── Main simulation ───────────────────────────────────────────────────────────

def run(args):
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print(f'Device: {device}')

    # Load reward model
    model = RewardModel(8 * 384).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()
    print(f'Reward model loaded from {args.model}')

    # Load prototype embedding (pre-computed "good driving" reference)
    proto_path = os.path.join(os.path.dirname(args.model), 'prototype.npy')
    prototype = None
    if os.path.exists(proto_path):
        proto_np = np.load(proto_path).astype(np.float32)
        prototype = torch.from_numpy(proto_np).unsqueeze(0).to(device)  # (1, 128)
        print(f'Prototype loaded — using cosine-similarity scoring (real-car mode)')
    else:
        print(f'No prototype found — will use gt_steer scoring (validation mode)')

    # Load BEV features
    files = sorted(glob.glob(os.path.join(args.data_dir, '*.npz')))
    print(f'Loading {len(files)} frames...')
    frames = []
    for f in files:
        d = np.load(f)
        frames.append({
            'bev': d['bev'].astype(np.float32),
            'steering': float(d['steering']),
            'throttle': float(d['throttle']),
            'heading': float(d['heading']) if 'heading' in d else 0.0,
        })
    print(f'Loaded {len(frames)} frames')

    # Load waypoints
    waypoints = []
    if os.path.exists(args.route):
        with open(args.route) as f:
            data = yaml.safe_load(f)
            waypoints = [(w['lat'], w['lon']) for w in data.get('waypoints', [])]
    print(f'Waypoints: {len(waypoints)}')

    planner = MPPIPlanner(bev_w=128, bev_h=128, T=16, step_size=4.0, lateral_scale=4.0, sigma=0.35)

    # Simulated GPS position (start at first waypoint or default)
    if waypoints:
        sim_lat, sim_lon = waypoints[0]
    else:
        sim_lat, sim_lon = 30.5209, -97.7154

    sim_heading = frames[0]['heading'] if frames else 0.0
    # Start targeting waypoint 0 — don't skip it just because we start there.
    # Use a 1m arrival radius so we naturally advance as dead-reckoning moves us.
    wp_idx = 0
    WP_ARRIVAL_M = 1.0   # metres — tighter than the original 3 m

    # ── Matplotlib setup ──────────────────────────────────────────────────────
    plt.ion()
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle('CREStE-Nano Simulator', color='white', fontsize=14, fontweight='bold')

    ax_bev   = fig.add_subplot(2, 2, 1)
    ax_plan  = fig.add_subplot(2, 2, 2)
    ax_hist  = fig.add_subplot(2, 2, 3)
    ax_cmp   = fig.add_subplot(2, 2, 4)   # human vs planner comparison

    for ax in [ax_bev, ax_plan, ax_hist, ax_cmp]:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#aaa')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')

    steer_hist, thr_hist, score_hist = [], [], []
    human_steer_hist = []   # ground truth from recording
    plt.tight_layout()

    print('\nRunning simulation... Close window to stop.\n')

    for i, frame in enumerate(frames):
        if not plt.fignum_exists(fig.number):
            break

        # Upscale BEV from 64×64 to 128×128 for finer path sampling
        bev_raw = frame['bev']   # (64, 64, 384)
        bev = np.repeat(np.repeat(bev_raw, 2, axis=0), 2, axis=1)  # (128, 128, 384)

        # Waypoint bearing
        wp_bearing = None
        if waypoints and wp_idx < len(waypoints):
            wlat, wlon = waypoints[wp_idx]
            dist = haversine(sim_lat, sim_lon, wlat, wlon)
            if dist < WP_ARRIVAL_M and wp_idx + 1 < len(waypoints):
                wp_idx += 1
                print(f'  Waypoint {wp_idx}/{len(waypoints)} reached')
            if wp_idx < len(waypoints):
                wlat, wlon = waypoints[wp_idx]
                abs_bearing = bearing_to(sim_lat, sim_lon, wlat, wlon)
                wp_bearing = ((abs_bearing - sim_heading + 180) % 360) - 180

        # MPPI plan
        U, eps, cands = planner.sample()
        scores = score_candidates(model, bev, cands, device,
                                  gt_steer=None if prototype is not None else frame['steering'],
                                  prototype=prototype)
        best_k = np.argmax(scores)
        steer = planner.update(scores, eps, wp_bearing)
        throttle = planner.auto_throttle * max(0.5, 1.0 - abs(steer) * 0.5)
        throttle = max(-0.25, min(0.25, throttle))

        # Update simulated position (dead-reckoning)
        speed = throttle * 2.0
        sim_heading = (sim_heading + steer * 10) % 360
        dx = speed * math.sin(math.radians(sim_heading)) * 0.033
        dy = speed * math.cos(math.radians(sim_heading)) * 0.033
        sim_lat += dy / 111320
        sim_lon += dx / (111320 * math.cos(math.radians(sim_lat)))

        steer_hist.append(steer)
        thr_hist.append(throttle)
        # Normalise mean score to [0,1] for display (cosine sim is in [-1,1])
        score_hist.append(float((scores.mean() + 1.0) / 2.0))
        human_steer_hist.append(frame['steering'])

        # ── BEV visualization ─────────────────────────────────────────────────
        ax_bev.clear()
        ax_bev.set_facecolor('#161b22')
        bev_vis = np.linalg.norm(bev, axis=2)
        bev_vis = (bev_vis - bev_vis.min()) / (bev_vis.max() - bev_vis.min() + 1e-8)
        ax_bev.imshow(bev_vis, cmap='viridis', origin='upper')
        # Draw planned path
        best_path = cands[best_k]
        ax_bev.plot(best_path[:, 1], best_path[:, 0], 'r-', linewidth=2, label='planned path')
        ax_bev.set_xlim(0, 127); ax_bev.set_ylim(127, 0)
        ax_bev.set_title(f'BEV Features  frame {i+1}/{len(frames)}', color='white', fontsize=10)
        ax_bev.set_xlabel('x (BEV)', color='#aaa', fontsize=8)
        ax_bev.set_ylabel('y (BEV)', color='#aaa', fontsize=8)
        ax_bev.legend(fontsize=7, facecolor='#0d1117', labelcolor='white')

        # ── Score distribution ────────────────────────────────────────────────
        ax_plan.clear()
        ax_plan.set_facecolor('#161b22')
        norm = Normalize(vmin=scores.min(), vmax=scores.max())
        sm = ScalarMappable(cmap='RdYlGn', norm=norm)
        # Draw top 50 candidate paths
        top_k = np.argsort(scores)[-50:]
        for k in top_k:
            color = sm.to_rgba(scores[k])
            ax_plan.plot(cands[k, :, 1] - 64, -(cands[k, :, 0] - 127),
                        color=color, alpha=0.4, linewidth=0.8)
        # Best path
        ax_plan.plot(cands[best_k, :, 1] - 64, -(cands[best_k, :, 0] - 127),
                    'w-', linewidth=2.5, label=f'best  steer={steer:.2f}')
        ax_plan.axvline(0, color='#555', linestyle='--', linewidth=0.8)
        ax_plan.set_xlim(-70, 70)
        ax_plan.set_title(f'MPPI Candidates  wp_bearing={wp_bearing:.1f}°' if wp_bearing is not None
                          else 'MPPI Candidates  no waypoint', color='white', fontsize=10)
        ax_plan.set_xlabel('lateral (BEV pixels)', color='#aaa', fontsize=8)
        ax_plan.set_ylabel('forward', color='#aaa', fontsize=8)
        ax_plan.legend(fontsize=8, facecolor='#0d1117', labelcolor='white')

        # ── History plots ─────────────────────────────────────────────────────
        ax_hist.clear()
        ax_hist.set_facecolor('#161b22')
        w = min(200, len(steer_hist))
        ax_hist.plot(steer_hist[-w:], color='#76c7ff', linewidth=1.2, label='planner steer')
        ax_hist.plot(thr_hist[-w:], color='#4ade80', linewidth=1.2, label='throttle')
        ax_hist.plot(score_hist[-w:], color='#f59e0b', linewidth=1.0, alpha=0.7, label='avg score')
        ax_hist.axhline(0, color='#555', linewidth=0.8)
        ax_hist.set_ylim(-1.1, 1.1)
        ax_hist.set_title(f'Controls  wp {wp_idx+1}/{len(waypoints)}', color='white', fontsize=10)
        ax_hist.set_xlabel('frames', color='#aaa', fontsize=8)
        ax_hist.legend(fontsize=8, facecolor='#0d1117', labelcolor='white')

        # ── Human vs Planner comparison ───────────────────────────────────────
        ax_cmp.clear()
        ax_cmp.set_facecolor('#161b22')
        w2 = min(200, len(steer_hist))
        human_w  = human_steer_hist[-w2:]
        planner_w = steer_hist[-w2:]
        ax_cmp.plot(human_w,   color='#f97316', linewidth=1.4, label='you (recorded)')
        ax_cmp.plot(planner_w, color='#76c7ff', linewidth=1.4, label='planner', alpha=0.85)
        ax_cmp.fill_between(range(len(human_w)),
                            human_w, planner_w,
                            alpha=0.15, color='#ff4444')
        ax_cmp.axhline(0, color='#555', linewidth=0.8)
        ax_cmp.set_ylim(-1.1, 1.1)
        # Live correlation — only on frames where human actually steered
        if len(human_w) > 5:
            h = np.array(human_w)
            p = np.array(planner_w)
            mae = float(np.mean(np.abs(h - p)))
            # Only correlate on frames where human steered (avoid divide-by-zero on flat zeros)
            mask = np.abs(h) > 0.02
            if mask.sum() > 5:
                corr = float(np.corrcoef(h[mask], p[mask])[0, 1])
                title = f'Human vs Planner  corr={corr:.2f} (on turns)  MAE={mae:.3f}'
            else:
                title = f'Human vs Planner  all straight  MAE={mae:.3f}'
        else:
            title = 'Human vs Planner'
        ax_cmp.set_title(title, color='white', fontsize=10)
        ax_cmp.set_xlabel('frames', color='#aaa', fontsize=8)
        ax_cmp.set_ylabel('steering', color='#aaa', fontsize=8)
        ax_cmp.legend(fontsize=8, facecolor='#0d1117', labelcolor='white')

        err = abs(steer - frame['steering'])
        status = (f'Frame {i+1:4d}/{len(frames)} | '
                  f'planner={steer:+.3f} human={frame["steering"]:+.3f} err={err:.3f} | '
                  f'score={scores.mean():.3f} | '
                  f'wp={wp_idx+1}/{len(waypoints)}')
        print(f'\r{status}', end='', flush=True)

        plt.pause(0.001 / max(args.speed, 0.1))

    print('\nSimulation complete.')
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='/Users/nishanmani/Desktop/mapless_nav_data/bev_features')
    p.add_argument('--model',    default='/Users/nishanmani/Desktop/JOYDEEP/models/reward_model/reward_mlp.pth')
    p.add_argument('--route',    default='/Users/nishanmani/Desktop/mapless_nav_data/current_route.yaml')
    p.add_argument('--speed',    type=float, default=1.0, help='playback speed multiplier')
    args = p.parse_args()
    run(args)
