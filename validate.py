"""
validate.py — quantitative validation of the new architecture.
No GUI, no video.  Just runs the planner over the entire dataset
and prints honest metrics.

Tells us what to expect tomorrow on the real car.
"""

import sys, os
_user_sp = os.path.expanduser('~/Library/Python/3.12/lib/python/site-packages')
if _user_sp not in sys.path:
    sys.path.insert(0, _user_sp)

import glob, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TraversabilityMLP(nn.Module):
    def __init__(self, feat_dim=384, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)


class TraversabilityUNet(nn.Module):
    """Spatial CNN matches train_traversability_cnn.py."""
    def __init__(self, in_ch=384, base=64):
        super().__init__()
        self.enc1 = self._block(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._block(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        self.bot  = self._block(base*2, base*4)
        self.up2  = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = self._block(base*4, base*2)
        self.up1  = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = self._block(base*2, base)
        self.out  = nn.Conv2d(base, 1, 1)
    def _block(self, i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool1(e1)); b = self.bot(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


class MPPIPlanner:
    def __init__(self, K=1000, T=16, sigma=0.35, lam=0.1,
                 bev_w=128, bev_h=128, step_size=4.0, lateral_scale=4.0,
                 max_steer=1.0, momentum=0.8):
        self.K, self.T = K, T
        self.sigma, self.lam = sigma, lam
        self.bev_w, self.bev_h = bev_w, bev_h
        self.step_size = step_size
        self.lateral_scale = lateral_scale
        self.max_steer = max_steer
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

    def update(self, scores, epsilons):
        s = scores - scores.max()
        w = np.exp(s / self.lam); w /= (w.sum() + 1e-8)
        self.nominal_U = np.clip(
            self.momentum * self.nominal_U + np.einsum('k,kt->t', w, epsilons),
            -self.max_steer, self.max_steer)
        action = float(self.nominal_U[0])
        self.nominal_U = np.roll(self.nominal_U, -1)
        self.nominal_U[-1] = 0.0
        return action


def main():
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    # Prefer the UNet if it exists, fall back to MLP
    unet_path = os.path.expanduser('~/Desktop/JOYDEEP/models/reward_model/traversability_unet.pth')
    mlp_path  = os.path.expanduser('~/Desktop/JOYDEEP/models/reward_model/traversability_mlp.pth')
    USE_UNET = os.path.exists(unet_path)

    if USE_UNET:
        trav = TraversabilityUNet().to(device)
        trav.load_state_dict(torch.load(unet_path, map_location=device, weights_only=True))
        trav.eval()
        print(f'Loaded UNet from {unet_path}')
    elif os.path.exists(mlp_path):
        trav = TraversabilityMLP().to(device)
        trav.load_state_dict(torch.load(mlp_path, map_location=device, weights_only=True))
        trav.eval()
        print(f'Loaded MLP from {mlp_path}')
    else:
        print('No traversability model found. Train one first.')
        return

    # Load all frames
    files = sorted(glob.glob(os.path.expanduser('~/Desktop/mapless_nav_data/bev_features/*.npz')))
    print(f'Validating on {len(files)} frames...\n')

    planner = MPPIPlanner()

    human_steers   = []
    planner_steers = []
    pos_scores     = []   # traversability on actual path
    rand_scores    = []   # traversability on random alternative paths
    pixel_top_q    = []   # top 10% pixel score
    pixel_bot_q    = []   # bot 10% pixel score

    for i, f in enumerate(files):
        d = np.load(f)
        bev_raw = d['bev'].astype(np.float32)
        bev = np.repeat(np.repeat(bev_raw, 2, axis=0), 2, axis=1)  # 128×128
        human_steer = float(d['steering'])

        # Score every pixel
        with torch.no_grad():
            if USE_UNET:
                # UNet expects 64×64 with channels-first
                bev64 = bev_raw.transpose(2, 0, 1)
                t = torch.from_numpy(bev64).unsqueeze(0).to(device)
                pix64 = torch.sigmoid(trav(t)).squeeze().cpu().numpy()
                # Upsample to 128×128 to match path coords
                pix = np.kron(pix64, np.ones((2, 2)))
            else:
                flat = torch.from_numpy(bev.reshape(-1, 384)).to(device)
                pix = torch.sigmoid(trav(flat)).cpu().numpy().reshape(128, 128)

        # Pixel score distribution
        pixel_top_q.append(np.percentile(pix, 90))
        pixel_bot_q.append(np.percentile(pix, 10))

        # Score on the GT (human-driven) trajectory
        x, y = 64.0, 127.0
        gt_pixels = []
        for _ in range(16):
            y -= 4.0
            x += human_steer * 4.0
            gt_pixels.append((int(np.clip(y, 0, 127)), int(np.clip(x, 0, 127))))
        gt_score = np.mean([pix[r, c] for r, c in gt_pixels])
        pos_scores.append(gt_score)

        # Score on 5 random alternative paths (negative reference)
        rand_means = []
        for _ in range(5):
            rs = np.random.uniform(-0.6, 0.6)
            xr, yr = 64.0, 127.0
            rpx = []
            for _ in range(16):
                yr -= 4.0
                xr += rs * 4.0
                rpx.append((int(np.clip(yr, 0, 127)), int(np.clip(xr, 0, 127))))
            rand_means.append(np.mean([pix[r, c] for r, c in rpx]))
        rand_scores.append(np.mean(rand_means))

        # MPPI plan
        U, eps, cands = planner.sample()
        by = np.clip(cands[:, :, 0].astype(int), 0, 127)
        bx = np.clip(cands[:, :, 1].astype(int), 0, 127)
        cand_scores = pix[by, bx].mean(axis=1)
        steer = planner.update(cand_scores, eps)

        human_steers.append(human_steer)
        planner_steers.append(steer)

        if (i+1) % 1000 == 0:
            print(f'  {i+1}/{len(files)} frames processed')

    # ── Final metrics ─────────────────────────────────────────────────────
    h = np.array(human_steers)
    p = np.array(planner_steers)
    mask = np.abs(h) > 0.02
    corr_all   = np.corrcoef(h, p)[0, 1]
    corr_turns = np.corrcoef(h[mask], p[mask])[0, 1] if mask.sum() > 10 else float('nan')
    mae_all    = np.mean(np.abs(h - p))
    mae_turns  = np.mean(np.abs(h[mask] - p[mask]))

    pos_mean = np.mean(pos_scores)
    neg_mean = np.mean(rand_scores)
    margin   = pos_mean - neg_mean

    print('\n' + '='*60)
    print('VALIDATION RESULTS')
    print('='*60)
    print(f'Total frames evaluated:       {len(files)}')
    print(f'Turn frames (|steer| > 0.02): {mask.sum()}  ({100*mask.mean():.0f}%)')
    print()
    print('─── Steering agreement ─────────────────────────────────────')
    print(f'  Correlation (all frames):    {corr_all:+.3f}')
    print(f'  Correlation (turns only):    {corr_turns:+.3f}   ← KEY')
    print(f'  MAE (all frames):            {mae_all:.3f}')
    print(f'  MAE (turns only):            {mae_turns:.3f}')
    print()
    print('─── Traversability signal strength ────────────────────────')
    print(f'  Avg score on driven path:    {pos_mean:.3f}')
    print(f'  Avg score on random paths:   {neg_mean:.3f}')
    print(f'  Pos vs random margin:        {margin:+.3f}   ← KEY')
    print()
    print(f'  Pixel scores: 90th pct = {np.mean(pixel_top_q):.3f}, '
          f'10th pct = {np.mean(pixel_bot_q):.3f}')
    print(f'  Pixel contrast:              {np.mean(pixel_top_q) - np.mean(pixel_bot_q):+.3f}')
    print('='*60)

    # ── Interpretation ────────────────────────────────────────────────────
    print('\nWHAT THIS MEANS FOR TOMORROW:')
    if corr_turns > 0.5 and margin > 0.1:
        print('  ✓ STRONG signal. Real car should follow roads visibly.')
    elif corr_turns > 0.35 and margin > 0.05:
        print('  ~ MODERATE signal. Car will respond to roads, not perfectly.')
        print('    Expect some drift on sharp turns, decent on straights/gentle curves.')
    elif corr_turns > 0.2:
        print('  ⚠ WEAK signal. Car will mostly use waypoint bearing.')
        print('    Demo will work but won\'t look like true vision-driven nav.')
    else:
        print('  ✗ POOR signal. Approach not working — model didn\'t learn discriminative features.')

if __name__ == '__main__':
    main()
