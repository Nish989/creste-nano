"""make_plots.py — produce evaluation plots from the trained UNet.

All four panels are computed *from the actual checkpoint* on the full 8,331
training-set frames, not from a stale log.  Output goes to docs/plots/ for
inclusion in the README and any slide deck.

Usage:
    python3 make_plots.py
    python3 make_plots.py --max_frames 2000        # quick run
"""

import argparse, glob, json, os, sys
_user_sp = os.path.expanduser('~/Library/Python/3.12/lib/python/site-packages')
if _user_sp not in sys.path:
    sys.path.insert(0, _user_sp)

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import gridspec

DATA_DIR  = os.path.expanduser('~/Desktop/mapless_nav_data')
MODEL_PTH = os.path.expanduser('~/Desktop/JOYDEEP/models/reward_model/traversability_unet.pth')
OUT_DIR   = os.path.expanduser('~/Desktop/JOYDEEP/docs/plots')

os.makedirs(OUT_DIR, exist_ok=True)


# ── Model (matches train_traversability_cnn.py) ───────────────────────────────
class TraversabilityUNet(nn.Module):
    def __init__(self, in_ch=384, base=64):
        super().__init__()
        self.enc1 = self._block(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._block(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bot  = self._block(base * 2, base * 4)
        self.up2  = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = self._block(base * 4, base * 2)
        self.up1  = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = self._block(base * 2, base)
        self.out  = nn.Conv2d(base, 1, 1)

    def _block(self, i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b  = self.bot(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


# ── Data loading helpers ──────────────────────────────────────────────────────
def build_frame_table(data_dir):
    entries = []
    for meta_path in sorted(glob.glob(os.path.join(data_dir, 'session_*/metadata.jsonl'))):
        session_dir = os.path.dirname(meta_path)
        with open(meta_path) as f:
            for line in f:
                m = json.loads(line)
                entries.append({
                    'img': os.path.join(session_dir, 'images', m['image']),
                    'steering': float(m['steering']),
                })
    return entries


def load_bev_files(data_dir):
    return sorted(glob.glob(os.path.join(data_dir, 'bev_features', '*.npz')))


# ── Forward pass + centre-of-mass extraction ──────────────────────────────────
@torch.no_grad()
def predict_heatmap(model, bev_raw, device):
    """bev_raw is (64, 64, 384) float32 → returns (64, 64) sigmoid heatmap."""
    t = torch.from_numpy(bev_raw.transpose(2, 0, 1)).unsqueeze(0).to(device)
    pix = torch.sigmoid(model(t)).squeeze().cpu().numpy()
    return pix


def heatmap_centre_of_mass(heatmap, row=None):
    """Mean column at one row (or argmax row if row=None) weighted by sigmoid."""
    H, W = heatmap.shape
    if row is None:
        row = H - 5
    weights = heatmap[row, :].astype(np.float64)
    if weights.sum() < 1e-6:
        return W / 2.0
    cols = np.arange(W)
    return float((cols * weights).sum() / weights.sum())


# ── Plotting style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'figure.facecolor': 'white',
})


# ── Plot 1: heatmap quality grid ──────────────────────────────────────────────
def plot_heatmap_grid(model, frames, bev_files, device, n_examples=6, out_path=None):
    """Pick frames spanning the steering distribution (sharp left, mild left,
    straight x2, mild right, sharp right) and show camera + heatmap pairs."""
    steerings = np.array([f['steering'] for f in frames[:len(bev_files)]])
    target_steers = np.linspace(steerings.min(), steerings.max(), n_examples)
    picks = [int(np.argmin(np.abs(steerings - s))) for s in target_steers]

    fig = plt.figure(figsize=(15, 6.5))
    gs = gridspec.GridSpec(2, n_examples, hspace=0.18, wspace=0.05,
                           left=0.03, right=0.99, top=0.92, bottom=0.04)

    for col, idx in enumerate(picks):
        img = cv2.imread(frames[idx]['img'])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bev_raw = np.load(bev_files[idx])['bev'].astype(np.float32)
        heatmap = predict_heatmap(model, bev_raw, device)

        ax_img = fig.add_subplot(gs[0, col])
        ax_img.imshow(img_rgb)
        ax_img.set_title(f"steering = {frames[idx]['steering']:+.2f}", fontsize=10)
        ax_img.set_xticks([]); ax_img.set_yticks([])

        ax_hm = fig.add_subplot(gs[1, col])
        ax_hm.imshow(heatmap, cmap='viridis', vmin=0, vmax=1)
        # mark centre-of-mass column
        com = heatmap_centre_of_mass(heatmap)
        ax_hm.axvline(com, color='red', linewidth=1.5, alpha=0.85)
        ax_hm.axvline(heatmap.shape[1] / 2 - 0.5, color='white',
                      linewidth=1, linestyle=':', alpha=0.6)
        ax_hm.set_xticks([]); ax_hm.set_yticks([])

    fig.suptitle('TraversabilityUNet predictions on representative frames',
                 fontsize=13, y=0.985)
    fig.text(0.01, 0.71, 'Camera', rotation=90, fontsize=10,
             va='center', color='#444')
    fig.text(0.01, 0.27, 'UNet sigmoid', rotation=90, fontsize=10,
             va='center', color='#444')
    fig.text(0.5, 0.005,
             'red line = predicted centre-of-mass column   '
             'white dotted = BEV centreline',
             ha='center', fontsize=9, color='#555')

    if out_path:
        fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ── Plot 2: corridor margin (the actual training-script metric) ──────────────
def build_corridor_mask(steering, dilate_px=3, n_steps=22):
    """Match train_traversability_cnn.py exactly."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    x, y = 32.0, 63.0
    for _ in range(n_steps):
        y -= 1.0
        x += steering * 1.0
        if 0 <= int(y) < 64 and 0 <= int(x) < 64:
            mask[int(y), int(x)] = 1
        if y <= 0:
            break
    if mask.sum() > 0:
        k = 2 * dilate_px + 1
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8), iterations=1)
    return mask


def plot_corridor_margin(model, frames, bev_files, device,
                         max_frames=None, out_path=None):
    """For each frame compute pos_pred (mean UNet sigmoid on the human's
    rollout corridor) and neg_pred (mean on the high-feature background).
    margin = pos_pred - neg_pred is what the training script tracked."""
    n = len(bev_files) if max_frames is None else min(max_frames, len(bev_files))
    pos_means = np.zeros(n, dtype=np.float32)
    neg_means = np.zeros(n, dtype=np.float32)

    print(f'computing corridor pos/neg margin on {n} frames ...')
    for i in range(n):
        bev_raw = np.load(bev_files[i])['bev'].astype(np.float32)
        heatmap = predict_heatmap(model, bev_raw, device)
        st = frames[i]['steering']
        pos = build_corridor_mask(st)
        has_feat = np.any(bev_raw != 0, axis=2).astype(np.uint8)
        neg = has_feat & (1 - pos)
        if pos.sum() > 0:
            pos_means[i] = float((heatmap * pos).sum() / pos.sum())
        else:
            pos_means[i] = np.nan
        if neg.sum() > 0:
            neg_means[i] = float((heatmap * neg).sum() / neg.sum())
        else:
            neg_means[i] = np.nan
        if (i + 1) % 500 == 0:
            print(f'  {i+1}/{n}')

    valid = (~np.isnan(pos_means)) & (~np.isnan(neg_means))
    pos_means = pos_means[valid]
    neg_means = neg_means[valid]
    margins = pos_means - neg_means

    pos_mean = float(pos_means.mean())
    neg_mean = float(neg_means.mean())
    margin_mean = float(margins.mean())
    margin_med = float(np.median(margins))
    frac_pos_margin = float((margins > 0).mean())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6),
                             gridspec_kw={'width_ratios': [1, 1]})

    # Left: distributions of pos_pred and neg_pred per frame (overlaid hist)
    ax = axes[0]
    ax.hist(pos_means, bins=50, color='#2ca02c', alpha=0.7,
            label=f'pos_pred (on corridor cells)   mean={pos_mean:.3f}',
            edgecolor='#15511c', linewidth=0.4)
    ax.hist(neg_means, bins=50, color='#d62728', alpha=0.65,
            label=f'neg_pred (on background)       mean={neg_mean:.3f}',
            edgecolor='#5f1010', linewidth=0.4)
    ax.set_xlabel('per-frame mean UNet sigmoid')
    ax.set_ylabel('frame count')
    ax.set_title(f'UNet sigmoid on labelled vs background cells\n'
                 f'(matches training-script tracking)')
    ax.legend(loc='upper right', frameon=True, framealpha=0.92, fontsize=9)
    ax.grid(alpha=0.3, linewidth=0.5)

    # Right: per-frame margin histogram
    ax = axes[1]
    ax.hist(margins, bins=60, color='#1f77b4', alpha=0.85,
            edgecolor='#0e3050', linewidth=0.4)
    ax.axvline(0, color='black', linewidth=1.2, linestyle='--', alpha=0.6,
               label='zero margin')
    ax.axvline(margin_mean, color='#d62728', linewidth=2,
               label=f'mean = {margin_mean:+.3f}')
    ax.axvline(margin_med, color='#2ca02c', linewidth=2, linestyle='--',
               label=f'median = {margin_med:+.3f}')
    ax.set_xlabel('margin = pos_pred − neg_pred')
    ax.set_ylabel('frame count')
    ax.set_title(f'Per-frame margin  ({100*frac_pos_margin:.1f}% of frames > 0)')
    ax.legend(loc='upper left', frameon=True, framealpha=0.92, fontsize=9)
    ax.grid(alpha=0.3, linewidth=0.5)

    fig.suptitle(f'Training objective measured across {len(pos_means)} frames',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)

    return {
        'pos_pred_mean': pos_mean,
        'neg_pred_mean': neg_mean,
        'margin_mean':   margin_mean,
        'margin_median': margin_med,
        'fraction_positive_margin': frac_pos_margin,
        'n_evaluated': int(len(pos_means)),
    }


# ── Plot 3: per-frame mean traversability distribution ───────────────────────
def plot_score_distribution(model, bev_files, device, max_frames=None, out_path=None):
    n = len(bev_files) if max_frames is None else min(max_frames, len(bev_files))
    means = np.zeros(n, dtype=np.float32)
    print(f'computing per-frame mean traversability on {n} frames ...')
    for i in range(n):
        bev_raw = np.load(bev_files[i])['bev'].astype(np.float32)
        heatmap = predict_heatmap(model, bev_raw, device)
        means[i] = float(heatmap.mean())
        if (i + 1) % 1000 == 0:
            print(f'  {i+1}/{n}')

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(means, bins=60, color='#1f77b4', alpha=0.85,
            edgecolor='#0e3050', linewidth=0.5)
    ax.axvline(means.mean(), color='#d62728', linewidth=2,
               label=f'mean = {means.mean():.3f}')
    ax.axvline(np.median(means), color='#2ca02c', linewidth=2, linestyle='--',
               label=f'median = {np.median(means):.3f}')
    ax.set_xlabel('per-frame mean UNet sigmoid traversability')
    ax.set_ylabel('frame count')
    ax.set_title(f'UNet output distribution across {n} dataset frames',
                 fontsize=11)
    ax.legend(loc='upper right', frameon=True, framealpha=0.92)
    ax.grid(alpha=0.3, linewidth=0.5)
    if out_path:
        fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ── Plot 4: MPPI rollout score distribution for a sample frame ───────────────
def plot_mppi_landscape(model, bev_files, device, frame_idx=4000, K=1000, T=8,
                       out_path=None):
    bev_raw = np.load(bev_files[frame_idx])['bev'].astype(np.float32)
    heatmap = predict_heatmap(model, bev_raw, device)
    H = W = 64
    bev_max = H - 1

    # Sample K rollouts the same way the planner does
    np.random.seed(0)
    sigma = 0.45
    eps = np.random.normal(0, sigma, (K, T))
    candidates = np.zeros((K, T, 2))
    x = np.full(K, float(W / 2))
    y = np.full(K, float(H - 1))
    step_size = 1.5
    lateral_scale = 2.0
    for t in range(T):
        y = y - step_size
        x = x + eps[:, t] * lateral_scale
        candidates[:, t, 0] = y
        candidates[:, t, 1] = x

    by = np.clip(candidates[:, :, 0].astype(int), 0, bev_max)
    bx = np.clip(candidates[:, :, 1].astype(int), 0, bev_max)
    scores = heatmap[by, bx].mean(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2),
                             gridspec_kw={'width_ratios': [1, 1.05]})

    # Left: BEV heatmap with rollouts overlaid coloured by score
    ax = axes[0]
    ax.imshow(heatmap, cmap='viridis', vmin=0, vmax=1, origin='upper')
    norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    cmap = plt.get_cmap('coolwarm')
    order = np.argsort(scores)
    for k in order[::5]:   # draw every 5th for legibility
        ax.plot(bx[k], by[k], color=cmap(1 - norm_scores[k]),
                linewidth=0.5, alpha=0.55)
    best_k = int(np.argmax(scores))
    ax.plot(bx[best_k], by[best_k], color='#ffe800', linewidth=2.5,
            label=f'best rollout  (score = {scores[best_k]:.3f})')
    ax.scatter([W / 2], [H - 1], s=70, color='red',
               edgecolors='white', linewidths=1.5, zorder=5, label='car')
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'frame {frame_idx}: UNet heatmap + K={K} sampled rollouts')
    ax.legend(loc='upper right', frameon=True, framealpha=0.92, fontsize=9)

    # Right: histogram of rollout scores
    ax = axes[1]
    ax.hist(scores, bins=40, color='#1f77b4', alpha=0.85,
            edgecolor='#0e3050', linewidth=0.5)
    ax.axvline(scores.mean(), color='#d62728', linewidth=2,
               label=f'mean = {scores.mean():.3f}')
    ax.axvline(scores.max(), color='#ffb000', linewidth=2,
               label=f'max  = {scores.max():.3f}')
    ax.set_xlabel('mean UNet sigmoid along rollout')
    ax.set_ylabel(f'rollouts (out of K={K})')
    ax.set_title('MPPI score landscape for this frame')
    ax.legend(loc='upper left', frameon=True, framealpha=0.92, fontsize=9)
    ax.grid(alpha=0.3, linewidth=0.5)

    fig.suptitle('What the planner sees: per-frame reward landscape',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    device = (torch.device('mps') if torch.backends.mps.is_available()
              else torch.device('cpu'))
    print(f'device: {device}')

    model = TraversabilityUNet().to(device)
    model.load_state_dict(torch.load(MODEL_PTH, map_location=device,
                                     weights_only=True))
    model.eval()
    print(f'loaded {MODEL_PTH}')

    frames = build_frame_table(DATA_DIR)
    bev_files = load_bev_files(DATA_DIR)
    print(f'{len(frames)} frames, {len(bev_files)} BEV feature files')

    # Plot 1 — heatmap grid (always cheap)
    plot_heatmap_grid(model, frames, bev_files, device,
                      n_examples=6,
                      out_path=os.path.join(OUT_DIR, 'heatmap_examples.png'))
    print('wrote heatmap_examples.png')

    # Plot 2 — corridor margin (the *actual* training-script metric)
    margin_stats = plot_corridor_margin(
        model, frames, bev_files, device,
        max_frames=args.max_frames,
        out_path=os.path.join(OUT_DIR, 'corridor_margin.png'))
    print(f'wrote corridor_margin.png  '
          f'(pos={margin_stats["pos_pred_mean"]:.3f} '
          f'neg={margin_stats["neg_pred_mean"]:.3f} '
          f'margin={margin_stats["margin_mean"]:+.3f})')

    # Plot 3 — score distribution
    plot_score_distribution(
        model, bev_files, device,
        max_frames=args.max_frames,
        out_path=os.path.join(OUT_DIR, 'score_distribution.png'))
    print('wrote score_distribution.png')

    # Plot 4 — MPPI landscape on a sample frame
    plot_mppi_landscape(model, bev_files, device,
                        frame_idx=min(args.sample_frame, len(bev_files) - 1),
                        K=1000, T=8,
                        out_path=os.path.join(OUT_DIR, 'mppi_landscape.png'))
    print('wrote mppi_landscape.png')

    # Persist real numbers so the README can reference them honestly
    with open(os.path.join(OUT_DIR, 'metrics.json'), 'w') as f:
        json.dump({
            **margin_stats,
            'model_path': MODEL_PTH,
            'description': (
                'Reproduced from the trained TraversabilityUNet checkpoint. '
                'pos_pred = mean sigmoid on the dilated human-rollout corridor '
                'mask; neg_pred = mean sigmoid on cells with features but '
                'outside the corridor. Margin = pos_pred - neg_pred is what '
                'train_traversability_cnn.py tracks per epoch.'
            ),
        }, f, indent=2)
    print('wrote metrics.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--max_frames', type=int, default=None,
                   help='cap frames evaluated for the scatter + distribution (default: all)')
    p.add_argument('--sample_frame', type=int, default=4000,
                   help='which dataset index to use for the MPPI landscape plot')
    main(p.parse_args())
