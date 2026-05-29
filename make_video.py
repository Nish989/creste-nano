"""
make_video.py  —  CREStE-Nano demo video
Draws the MPPI planned path directly onto the camera image (like Tesla autopilot)
plus a small BEV inset in the corner.

Usage:
  python3 make_video.py --frames 500       # preview
  python3 make_video.py                    # full 8331 frames
"""

import sys, os
_user_sp = os.path.expanduser('~/Library/Python/3.12/lib/python/site-packages')
if _user_sp not in sys.path:
    sys.path.insert(0, _user_sp)

import argparse, glob, json, math
import cv2, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Traversability MLP (matches train_traversability.py) ─────────────────────
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

# ── MPPI Planner ──────────────────────────────────────────────────────────────
class MPPIPlanner:
    def __init__(self, K=1000, T=16, sigma=0.35, lam=0.1,
                 bev_w=128, bev_h=128, step_size=4.0, lateral_scale=4.0,
                 max_steer=1.0, wp_bias=0.3, momentum=0.8):
        self.K, self.T = K, T
        self.sigma, self.lam = sigma, lam
        self.bev_w, self.bev_h = bev_w, bev_h
        self.step_size = step_size
        self.lateral_scale = lateral_scale
        self.max_steer = max_steer
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

def score_candidates(bev, candidates):
    bev_max = bev.shape[0] - 1
    by = np.clip(candidates[:, :, 0].astype(int), 0, bev_max)
    bx = np.clip(candidates[:, :, 1].astype(int), 0, bev_max)
    feat_norms = np.linalg.norm(bev[by, bx, :], axis=-1)
    raw = feat_norms.mean(axis=1)
    return raw / (raw.max() + 1e-8), None


def score_candidates_pu(trav_model, bev, candidates, device):
    """Per-pixel PU traversability scoring.  Returns (path_scores, pixel_heatmap)."""
    bev_max = bev.shape[0] - 1
    H, W, D = bev.shape
    with torch.no_grad():
        flat = torch.from_numpy(bev.reshape(-1, D).astype(np.float32)).to(device)
        score_pix = torch.sigmoid(trav_model(flat)).cpu().numpy().reshape(H, W)
    by = np.clip(candidates[:, :, 0].astype(int), 0, bev_max)
    bx = np.clip(candidates[:, :, 1].astype(int), 0, bev_max)
    path_scores = score_pix[by, bx].mean(axis=1)
    return path_scores, score_pix

# ── Perspective projection: BEV pixel → camera image pixel ───────────────────
# BEV is 128×128. Car at (row=127, col=64).
# Physical scale: ~0.08 m per pixel (so 128px ≈ 10m forward)
# Camera: 1280×720, ~70° HFOV, mounted ~15cm above ground, ~5° pitch down

IMG_W, IMG_H = 1280, 720

# ── Homography: BEV (col, row) → image (x, y) ────────────────────────────────
# Calibrated from actual camera frames.
# BEV is 128×128, car at (row=127, col=64).
# 4 ground-plane correspondences picked from the sidewalk footage:
#   BEV near-left/right  ↔  sidewalk edges ~0.5m ahead in image
#   BEV far-left/right   ↔  sidewalk edges ~4m ahead near horizon
_bev_src = np.float32([
    [14,  124],   # near-left
    [114, 124],   # near-right
    [48,   20],   # far-left  (much further ahead)
    [80,   20],   # far-right (much further ahead)
])
_img_dst = np.float32([
    [190,  718],  # near-left  in image
    [1090, 718],  # near-right in image
    [510,  395],  # far-left   near horizon
    [770,  395],  # far-right  near horizon
])
_H, _ = cv2.findHomography(_bev_src, _img_dst)

def bev_to_img(bev_row, bev_col, bev_h=128, bev_w=128):
    if bev_row >= bev_h - 2:
        return None   # at car position, skip
    pt = np.array([[[float(bev_col), float(bev_row)]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(pt, _H)
    u, v = int(dst[0, 0, 0]), int(dst[0, 0, 1])
    if 0 <= u < IMG_W and 0 <= v < IMG_H:
        return u, v
    return None

# ── Build ordered frame list ──────────────────────────────────────────────────
def build_frame_list(data_dir):
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

# ── Draw path on image ────────────────────────────────────────────────────────
def draw_path_on_image(img, path_bev, color, thickness=3, alpha=0.85):
    overlay = img.copy()
    pts = []
    for pt in path_bev:
        uv = bev_to_img(pt[0], pt[1])
        if uv and 0 <= uv[0] < IMG_W and 0 <= uv[1] < IMG_H:
            pts.append(uv)
    for j in range(1, len(pts)):
        cv2.line(overlay, pts[j-1], pts[j], color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    return pts

def draw_filled_path(img, path_bev, color):
    """Draw a filled translucent corridor along the best path."""
    pts_l, pts_r = [], []
    for pt in path_bev:
        uv = bev_to_img(pt[0], pt[1])
        if uv and 0 <= uv[0] < IMG_W and 0 <= uv[1] < IMG_H:
            # Left/right offsets in image space (narrows with distance)
            offset = max(4, int(30 * (IMG_H - uv[1]) / IMG_H))
            pts_l.append((uv[0]-offset, uv[1]))
            pts_r.append((uv[0]+offset, uv[1]))
    if len(pts_l) < 2:
        return
    pts_all = np.array(pts_l + pts_r[::-1], dtype=np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts_all], color)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

# ── BEV inset ─────────────────────────────────────────────────────────────────
def make_bev_inset(bev, cands, scores, best_k, size=240, heatmap=None):
    """If heatmap is provided (per-pixel traversability), use it; else fallback to feature norm."""
    if heatmap is not None:
        bev_vis = heatmap
    else:
        bev_vis = np.linalg.norm(bev, axis=2)
        bev_vis = (bev_vis - bev_vis.min()) / (bev_vis.max() - bev_vis.min() + 1e-8)
    inset = cv2.applyColorMap((bev_vis*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    inset = cv2.resize(inset, (size, size))
    scale = size / 128.0
    # Just the best path — keep the inset readable
    pass
    # Best path
    bpts = [(int(cands[best_k,j,1]*scale), int(cands[best_k,j,0]*scale))
            for j in range(cands.shape[1])]
    for j in range(len(bpts)-1):
        cv2.line(inset, bpts[j], bpts[j+1], (255,255,255), 2, cv2.LINE_AA)
    # Car marker
    cv2.drawMarker(inset, (size//2, size-4), (100,200,255),
                   cv2.MARKER_TRIANGLE_UP, 10, 2)
    # Border
    cv2.rectangle(inset, (0,0), (size-1,size-1), (80,80,80), 1)
    return inset

# ── Main ──────────────────────────────────────────────────────────────────────
def run(args):
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print(f'Device: {device}')

    # Load traversability MLP
    trav_path = os.path.expanduser('~/Desktop/JOYDEEP/models/reward_model/traversability_mlp.pth')
    trav_model = None
    if os.path.exists(trav_path):
        trav_model = TraversabilityMLP().to(device)
        trav_model.load_state_dict(torch.load(trav_path, map_location=device, weights_only=True))
        trav_model.eval()
        print(f'Loaded TraversabilityMLP — PU mapless scoring active')
    else:
        print(f'No trav model found, using DINOv2 magnitude fallback')

    bev_files = sorted(glob.glob(os.path.join(args.data_dir, 'bev_features', '*.npz')))
    frames    = build_frame_list(args.data_dir)
    n = min(len(bev_files), len(frames), args.frames)
    print(f'Rendering {n} frames → {args.out}')

    planner = MPPIPlanner(T=24, step_size=4.5, lateral_scale=3.5)
    fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
    writer  = cv2.VideoWriter(args.out, fourcc, args.fps, (IMG_W, IMG_H))

    steer_h, planner_h = [], []

    for i in range(n):
        img = cv2.imread(frames[i]['img'])
        if img is None: continue
        img = cv2.resize(img, (IMG_W, IMG_H))

        d       = np.load(bev_files[i])
        bev_raw = d['bev'].astype(np.float32)
        bev     = np.repeat(np.repeat(bev_raw, 2, axis=0), 2, axis=1)  # 64→128
        human   = frames[i]['steering']

        U, eps, cands = planner.sample()
        if trav_model is not None:
            scores, heatmap = score_candidates_pu(trav_model, bev, cands, device)
        else:
            scores, heatmap = score_candidates(bev, cands)
        best_k  = np.argmax(scores)
        steer   = planner.update(scores, eps)

        steer_h.append(human)
        planner_h.append(steer)

        # ── Clean overlay: just the best path as a bright glowing line ────
        # Filled green corridor (subtle background)
        draw_filled_path(img, cands[best_k], (0, 180, 80))
        # Crisp bright green centerline on top
        draw_path_on_image(img, cands[best_k], (40, 255, 130), thickness=5, alpha=0.95)

        # ── BEV inset (top-right corner) ───────────────────────────────────
        inset = make_bev_inset(bev, cands, scores, best_k, size=220, heatmap=heatmap)
        ih, iw = inset.shape[:2]
        img[12:12+ih, IMG_W-iw-12:IMG_W-12] = inset

        # ── HUD text ───────────────────────────────────────────────────────
        cv2.putText(img, 'CREStE-Nano  |  mapless navigation',
                    (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(img, f'steer: {steer:+.2f}   frame: {i+1}/{n}',
                    (14, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,200), 1, cv2.LINE_AA)

        # Correlation
        if len(steer_h) > 10:
            h_a, p_a = np.array(steer_h), np.array(planner_h)
            mask = np.abs(h_a) > 0.02
            if mask.sum() > 5:
                corr = np.corrcoef(h_a[mask], p_a[mask])[0,1]
                cv2.putText(img, f'corr (turns): {corr:.2f}',
                            (14, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150,220,255), 1, cv2.LINE_AA)

        writer.write(img)

        if (i+1) % 200 == 0:
            print(f'  {i+1}/{n}', flush=True)

    writer.release()
    print(f'Saved → {args.out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='/Users/nishanmani/Desktop/mapless_nav_data')
    p.add_argument('--out',    default='/Users/nishanmani/Desktop/JOYDEEP/demo.mp4')
    p.add_argument('--fps',    type=int, default=15)
    p.add_argument('--frames', type=int, default=999999)
    run(p.parse_args())
