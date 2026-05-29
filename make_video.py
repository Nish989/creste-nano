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


# ── Traversability MLP (fallback) ────────────────────────────────────────────
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

# ── Traversability UNet (primary — matches train_traversability_cnn.py) ───────
class TraversabilityUNet(nn.Module):
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
        # Stash the *committed* sequence (pre-roll) so the visualizer can show
        # the actual MPPI output trajectory, not a single noisy sample.
        self.committed_U = self.nominal_U.copy()
        action = float(self.nominal_U[0])
        self.nominal_U = np.roll(self.nominal_U, -1)
        self.nominal_U[-1] = 0.0
        return action

    def nominal_trajectory(self):
        """Roll out committed_U from the car position to get the smooth MPPI
        output trajectory in BEV space (rows = forward, cols = lateral)."""
        U = self.committed_U if hasattr(self, 'committed_U') else self.nominal_U
        traj = np.zeros((self.T, 2))
        x = self.bev_w / 2.0
        y = self.bev_h - 1.0
        for t in range(self.T):
            y -= self.step_size
            x += U[t] * self.lateral_scale
            traj[t, 0] = y
            traj[t, 1] = x
        return traj

def score_candidates(bev, candidates):
    bev_max = bev.shape[0] - 1
    by = np.clip(candidates[:, :, 0].astype(int), 0, bev_max)
    bx = np.clip(candidates[:, :, 1].astype(int), 0, bev_max)
    feat_norms = np.linalg.norm(bev[by, bx, :], axis=-1)
    raw = feat_norms.mean(axis=1)
    return raw / (raw.max() + 1e-8), None


def score_candidates_pu(trav_model, bev, candidates, device):
    """MLP per-pixel traversability scoring.  Returns (path_scores, pixel_heatmap)."""
    bev_max = bev.shape[0] - 1
    H, W, D = bev.shape
    with torch.no_grad():
        flat = torch.from_numpy(bev.reshape(-1, D).astype(np.float32)).to(device)
        score_pix = torch.sigmoid(trav_model(flat)).cpu().numpy().reshape(H, W)
    by = np.clip(candidates[:, :, 0].astype(int), 0, bev_max)
    bx = np.clip(candidates[:, :, 1].astype(int), 0, bev_max)
    path_scores = score_pix[by, bx].mean(axis=1)
    return path_scores, score_pix

def score_candidates_unet(trav_model, bev_raw, candidates, device):
    """UNet spatial traversability scoring.  bev_raw is 64×64×384."""
    bev64 = bev_raw.transpose(2, 0, 1)   # (384,64,64)
    with torch.no_grad():
        t = torch.from_numpy(bev64).unsqueeze(0).to(device)
        pix64 = torch.sigmoid(trav_model(t)).squeeze().cpu().numpy()   # (64,64)
    # Upsample to 128×128 to match candidate coordinates
    pix = np.kron(pix64, np.ones((2, 2)))
    bev_max = pix.shape[0] - 1
    by = np.clip(candidates[:, :, 0].astype(int), 0, bev_max)
    bx = np.clip(candidates[:, :, 1].astype(int), 0, bev_max)
    path_scores = pix[by, bx].mean(axis=1)
    return path_scores, pix

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

# ── Path smoothing (chosen path comes out of MPPI noisy) ──────────────────────
def _smooth_path_bev(path_bev):
    """Fit a smooth quadratic curve x = f(y) through the BEV path so the
    rendered overlay is a clean arc rather than a jittery polyline."""
    p = np.asarray(path_bev, dtype=np.float32)
    if p.shape[0] < 4:
        return p
    y = p[:, 0]  # row (forward axis), decreases as we go forward
    x = p[:, 1]  # col (lateral)
    # Fit x as a smooth function of y (forward distance)
    coeffs = np.polyfit(y, x, 2)
    # Resample to twice as many points for a smoother polyline
    y_fine = np.linspace(y.min(), y.max(), p.shape[0] * 3)
    x_fine = np.polyval(coeffs, y_fine)
    return np.stack([y_fine, x_fine], axis=1)

def _project_path(path_bev):
    pts = []
    for pt in path_bev:
        uv = bev_to_img(pt[0], pt[1])
        if uv and 0 <= uv[0] < IMG_W and 0 <= uv[1] < IMG_H:
            pts.append(uv)
    return pts

# ── Draw path on image ────────────────────────────────────────────────────────
def draw_path_on_image(img, path_bev, color, thickness=3, alpha=0.85, smooth=True):
    if smooth:
        path_bev = _smooth_path_bev(path_bev)
    overlay = img.copy()
    pts = _project_path(path_bev)
    if len(pts) >= 2:
        arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [arr], False, color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    return pts

def draw_filled_path(img, path_bev, color, alpha=0.32, smooth=True):
    """Draw a filled translucent corridor along the (smoothed) best path."""
    if smooth:
        path_bev = _smooth_path_bev(path_bev)
    pts_l, pts_r = [], []
    for pt in path_bev:
        uv = bev_to_img(pt[0], pt[1])
        if uv and 0 <= uv[0] < IMG_W and 0 <= uv[1] < IMG_H:
            # Wider corridor that tapers cleanly with distance
            offset = max(6, int(46 * (IMG_H - uv[1]) / IMG_H))
            pts_l.append((uv[0] - offset, uv[1]))
            pts_r.append((uv[0] + offset, uv[1]))
    if len(pts_l) < 2:
        return
    pts_all = np.array(pts_l + pts_r[::-1], dtype=np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts_all], color, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

# ── Score → colour (red=bad, green=ok, cyan=best) ─────────────────────────────
def score_to_bgr(s):
    """Map normalized score in [0,1] to BGR (red→orange→yellow→green→cyan)."""
    s = float(np.clip(s, 0.0, 1.0))
    if s < 0.25:
        # red → orange
        t = s / 0.25
        b, g, r = 0, int(120 * t), 255
    elif s < 0.5:
        # orange → yellow
        t = (s - 0.25) / 0.25
        b, g, r = 0, int(120 + 135 * t), 255
    elif s < 0.75:
        # yellow → green
        t = (s - 0.5) / 0.25
        b, g, r = 0, 255, int(255 - 200 * t)
    else:
        # green → cyan
        t = (s - 0.75) / 0.25
        b, g, r = int(255 * t), 255, int(55 * (1 - t))
    return (b, g, r)

# ── Draw ALL candidate rollouts on the camera image ───────────────────────────
def draw_all_rollouts_camera(img, candidates, scores_norm, best_k,
                             max_draw=300, alpha=0.55):
    """Draw a sampled subset of MPPI rollouts on the camera image, coloured by score.
    Worst paths drawn first (under), best paths drawn last (over)."""
    overlay = img.copy()
    K = candidates.shape[0]
    # Sort by score ascending so high-score paths render on top
    order = np.argsort(scores_norm)
    # Subsample evenly across the score range for visual diversity
    if K > max_draw:
        idx = order[np.linspace(0, K-1, max_draw).astype(int)]
    else:
        idx = order
    for k in idx:
        if k == best_k:
            continue  # best path drawn separately on top
        pts = []
        for j in range(candidates.shape[1]):
            uv = bev_to_img(candidates[k, j, 0], candidates[k, j, 1])
            if uv:
                pts.append(uv)
        if len(pts) < 2:
            continue
        col = score_to_bgr(scores_norm[k])
        for j in range(1, len(pts)):
            cv2.line(overlay, pts[j-1], pts[j], col, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

# ── Viridis LUT (matches simulate.py cmap='viridis') ─────────────────────────
_VIRIDIS_LUT = None
def _viridis(arr_u8):
    """Apply matplotlib's viridis colormap via a precomputed LUT (no matplotlib import)."""
    global _VIRIDIS_LUT
    if _VIRIDIS_LUT is None:
        # 256-entry viridis in BGR order
        _v = np.array([
            [68,1,84],[68,2,86],[69,4,87],[69,5,89],[70,7,90],[70,8,92],[70,10,93],
            [70,11,94],[71,13,96],[71,14,97],[71,16,99],[71,17,100],[71,19,101],[72,20,103],
            [72,22,104],[72,23,105],[72,25,107],[72,26,108],[72,28,110],[72,29,111],
            [72,31,112],[72,32,113],[72,34,115],[72,35,116],[72,37,117],[72,38,118],
            [72,40,119],[72,41,120],[71,43,122],[71,44,123],[71,46,124],[71,47,125],
            [71,49,126],[71,50,127],[71,52,128],[70,53,129],[70,55,130],[70,56,131],
            [70,58,131],[70,59,132],[69,61,133],[69,62,134],[69,64,135],[68,65,136],
            [68,67,137],[68,68,137],[67,70,138],[67,71,139],[67,73,140],[66,74,141],
            [66,76,141],[65,77,142],[65,79,143],[64,80,144],[64,82,144],[63,83,145],
            [63,85,146],[62,86,146],[62,88,147],[61,89,148],[61,91,148],[60,93,149],
            [60,94,150],[59,96,150],[59,97,151],[58,99,151],[58,100,152],[57,102,153],
            [56,103,153],[56,105,154],[55,107,154],[55,108,155],[54,110,155],[54,111,156],
            [53,113,156],[53,114,157],[52,116,157],[52,118,158],[51,119,158],[51,121,159],
            [50,122,159],[50,124,160],[49,126,160],[49,127,161],[48,129,161],[48,130,162],
            [47,132,162],[47,134,163],[46,135,163],[46,137,163],[45,139,164],[45,140,164],
            [44,142,165],[44,143,165],[43,145,165],[43,147,166],[42,148,166],[42,150,167],
            [41,152,167],[41,153,167],[40,155,168],[40,157,168],[39,158,168],[39,160,169],
            [38,162,169],[38,163,169],[37,165,170],[37,167,170],[36,168,170],[36,170,171],
            [35,172,171],[35,173,171],[34,175,172],[34,177,172],[33,178,172],[33,180,172],
            [33,182,173],[32,183,173],[32,185,173],[31,187,173],[31,188,174],[31,190,174],
            [30,192,174],[30,193,174],[30,195,175],[29,197,175],[29,198,175],[29,200,175],
            [28,202,175],[28,203,176],[28,205,176],[27,207,176],[27,208,176],[27,210,176],
            [26,212,177],[26,213,177],[26,215,177],[25,217,177],[25,218,177],[25,220,177],
            [25,222,178],[24,223,178],[24,225,178],[24,227,178],[23,228,178],[23,230,178],
            [23,232,178],[22,233,179],[22,235,179],[22,237,179],[21,238,179],[21,240,179],
            [21,242,179],[21,243,179],[20,245,179],[20,247,180],[20,248,180],[20,250,180],
            [19,252,180],[40,253,168],[58,254,157],[76,254,145],[93,253,133],[110,252,121],
            [127,251,109],[143,249,97],[159,248,85],[175,246,72],[191,244,59],[206,242,46],
            [221,239,32],[236,237,19],[248,233,6],[253,231,37],[253,228,69],[252,225,100],
            [251,222,130],[249,219,160],[247,215,189],[245,212,216],[253,231,37],
        ], dtype=np.uint8)
        # Proper 256-entry viridis BGR LUT
        import struct
        lut = np.zeros((256, 3), dtype=np.uint8)
        viridis_rgb = [
            (68,1,84),(72,40,120),(62,88,147),(49,126,160),(38,162,169),(31,190,174),
            (25,220,177),(36,170,171),(94,201,98),(172,220,52),(253,231,37)
        ]
        for i in range(256):
            t = i / 255.0 * (len(viridis_rgb) - 1)
            lo_i = int(t); hi_i = min(lo_i + 1, len(viridis_rgb) - 1)
            f = t - lo_i
            r = int(viridis_rgb[lo_i][0] * (1-f) + viridis_rgb[hi_i][0] * f)
            g = int(viridis_rgb[lo_i][1] * (1-f) + viridis_rgb[hi_i][1] * f)
            b = int(viridis_rgb[lo_i][2] * (1-f) + viridis_rgb[hi_i][2] * f)
            lut[i] = [b, g, r]   # BGR for OpenCV
        _VIRIDIS_LUT = lut
    return _VIRIDIS_LUT[arr_u8]


# ── BEV inset (CREStE dotted-endpoint style) ──────────────────────────────────
def make_bev_inset(bev, cands, scores_norm, best_k, size=320, heatmap=None,
                   n_rollouts=60, committed_path=None):
    """CREStE-style BEV panel:
       • viridis (or muted dark) background showing the traversability heatmap
       • each rollout drawn as a series of small dots along the trajectory
       • each rollout terminates in a bigger filled-circle endpoint coloured by score
       • the chosen path's dots + endpoint glow bright cyan
       • a clean red car marker at the bottom-centre
       Looks like a particle-style data viz rather than a tangle of polylines."""

    if heatmap is not None:
        bg = heatmap.astype(np.float32)
    else:
        bg = np.linalg.norm(bev, axis=2).astype(np.float32)
    bg = (bg - bg.min()) / (bg.max() - bg.min() + 1e-8)
    bg_up = cv2.resize(bg, (size, size), interpolation=cv2.INTER_CUBIC)
    bg_u8 = (np.clip(bg_up, 0, 1) * 255).astype(np.uint8)
    inset = _viridis(bg_u8)
    # Mute the background so the dotted rollouts pop
    inset = (inset.astype(np.float32) * 0.55).clip(0, 255).astype(np.uint8)

    scale = size / 128.0
    K, T = cands.shape[0], cands.shape[1]

    # Stratified sample so the visible rollouts span the whole score range
    order = np.argsort(scores_norm)
    sample_idx = order[np.linspace(0, K-1, min(n_rollouts, K)).astype(int)]

    def _draw_rollout(target, k, base_radius=2, endpoint_radius=5, chosen=False):
        pts = [(int(cands[k, j, 1] * scale), int(cands[k, j, 0] * scale))
               for j in range(T)]
        col = (255, 240, 90) if chosen else score_to_bgr(scores_norm[k])
        # Path dots — small, every step
        for j, (x, y) in enumerate(pts[:-1]):
            r = base_radius + (1 if chosen else 0)
            cv2.circle(target, (x, y), r, col, -1, cv2.LINE_AA)
        # Endpoint — bigger filled circle with thin white ring
        ex, ey = pts[-1]
        cv2.circle(target, (ex, ey), endpoint_radius + (2 if chosen else 0),
                   col, -1, cv2.LINE_AA)
        cv2.circle(target, (ex, ey), endpoint_radius + (2 if chosen else 0),
                   (255, 255, 255) if chosen else (230, 230, 230), 1, cv2.LINE_AA)
        return pts[-1]

    # Draw non-chosen rollouts first
    for k in sample_idx:
        if k == best_k:
            continue
        _draw_rollout(inset, k, base_radius=2, endpoint_radius=4)

    # Chosen path on top with a halo. Prefer the committed (EMA-smoothed)
    # MPPI nominal trajectory if available so it matches the camera overlay.
    if committed_path is not None:
        cp = np.asarray(committed_path)
        bpts = [(int(cp[j, 1] * scale), int(cp[j, 0] * scale))
                for j in range(cp.shape[0])]
    else:
        bpts = [(int(cands[best_k, j, 1] * scale), int(cands[best_k, j, 0] * scale))
                for j in range(T)]

    # White halo behind the chosen path for glow
    halo = inset.copy()
    for j in range(len(bpts) - 1):
        cv2.line(halo, bpts[j], bpts[j+1], (255, 255, 255), 7, cv2.LINE_AA)
    cv2.addWeighted(halo, 0.28, inset, 0.72, 0, inset)
    # Bright cyan dotted committed path
    for x, y in bpts[:-1]:
        cv2.circle(inset, (x, y), 3, (255, 240, 90), -1, cv2.LINE_AA)
    ex, ey = bpts[-1]
    cv2.circle(inset, (ex, ey), 8, (255, 240, 90), -1, cv2.LINE_AA)
    cv2.circle(inset, (ex, ey), 8, (255, 255, 255), 1, cv2.LINE_AA)

    # Car marker (white-ringed red dot)
    cx, cy = size // 2, size - 10
    cv2.circle(inset, (cx, cy), 7, (255, 255, 255), -1)
    cv2.circle(inset, (cx, cy), 5, (60, 60, 220), -1)

    # Thin frame
    cv2.rectangle(inset, (0, 0), (size - 1, size - 1), (90, 90, 90), 1)

    return inset

# ── Main ──────────────────────────────────────────────────────────────────────
def run(args):
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print(f'Device: {device}')

    # Load traversability model — prefer UNet (corr_turns=0.914), fall back to MLP
    unet_path = os.path.expanduser('~/Desktop/JOYDEEP/models/reward_model/traversability_unet.pth')
    mlp_path  = os.path.expanduser('~/Desktop/JOYDEEP/models/reward_model/traversability_mlp.pth')
    trav_model = None
    use_unet   = False
    if os.path.exists(unet_path):
        trav_model = TraversabilityUNet().to(device)
        trav_model.load_state_dict(torch.load(unet_path, map_location=device, weights_only=True))
        trav_model.eval()
        use_unet = True
        print(f'Loaded TraversabilityUNet — spatial BEV scoring active')
    elif os.path.exists(mlp_path):
        trav_model = TraversabilityMLP().to(device)
        trav_model.load_state_dict(torch.load(mlp_path, map_location=device, weights_only=True))
        trav_model.eval()
        print(f'Loaded TraversabilityMLP fallback')
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
    BEV_PANEL = 340   # BEV inset edge length
    BEV_MARGIN = 18
    # EMA-smoothed display path so the cyan line on the camera reads as a
    # confident, decisive arc rather than a frame-to-frame jitter.
    ema_path = None
    EMA_ALPHA = 0.78   # higher = more smoothing

    for i in range(n):
        img = cv2.imread(frames[i]['img'])
        if img is None: continue
        img = cv2.resize(img, (IMG_W, IMG_H))

        d       = np.load(bev_files[i])
        bev_raw = d['bev'].astype(np.float32)
        bev     = np.repeat(np.repeat(bev_raw, 2, axis=0), 2, axis=1)  # 64→128
        human   = frames[i]['steering']

        U, eps, cands = planner.sample()
        if trav_model is not None and use_unet:
            scores, heatmap = score_candidates_unet(trav_model, bev_raw, cands, device)
        elif trav_model is not None:
            scores, heatmap = score_candidates_pu(trav_model, bev, cands, device)
        else:
            scores, heatmap = score_candidates(bev, cands)
        best_k  = np.argmax(scores)
        steer   = planner.update(scores, eps)

        # Normalize scores to [0,1] for colouring (so reds/cyans always span)
        s_min, s_max = scores.min(), scores.max()
        scores_norm = (scores - s_min) / (s_max - s_min + 1e-8)

        steer_h.append(human)
        planner_h.append(steer)

        # ── Compute the MPPI nominal trajectory (committed plan) ─────────────
        # This is the weighted average of all 1000 sampled rollouts — what the
        # planner is actually going to execute — not a single noisy sample.
        nom_path = planner.nominal_trajectory()
        # Temporal EMA across frames so the display path is decisive and stable
        if ema_path is None:
            ema_path = nom_path.copy()
        else:
            ema_path = EMA_ALPHA * ema_path + (1.0 - EMA_ALPHA) * nom_path

        # ── Camera overlay: clean Tesla-style corridor + smooth centreline ───
        # (rollouts live in the BEV panel — keep the main view uncluttered)
        # Wider soft-cyan corridor for the chosen path
        draw_filled_path(img, ema_path, (220, 200, 90), alpha=0.32)
        # Soft halo behind the centreline (white, very faint)
        draw_path_on_image(img, ema_path, (255, 255, 255),
                           thickness=16, alpha=0.18)
        # Bright cyan centreline on top, smooth and crisp
        draw_path_on_image(img, ema_path, (255, 240, 80),
                           thickness=5, alpha=0.95)

        # ── BEV inset (top-right corner) ─────────────────────────────────────
        inset = make_bev_inset(bev, cands, scores_norm, best_k,
                               size=BEV_PANEL, heatmap=heatmap, n_rollouts=60,
                               committed_path=ema_path)
        ih, iw = inset.shape[:2]
        x0 = IMG_W - iw - BEV_MARGIN
        y0 = BEV_MARGIN
        img[y0:y0+ih, x0:x0+iw] = inset

        # Tiny clean caption under the BEV
        cap_y = y0 + ih + 18
        cv2.putText(img, 'BEV  •  MPPI rollouts  •  UNet traversability',
                    (x0 + 2, cap_y), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    (210, 210, 210), 1, cv2.LINE_AA)

        # ── Minimal top-left HUD ─────────────────────────────────────────────
        # Single dark pill, just title + autonomy dot + steering bar.
        pill_w, pill_h = 360, 76
        pill = img.copy()
        cv2.rectangle(pill, (BEV_MARGIN, BEV_MARGIN),
                      (BEV_MARGIN + pill_w, BEV_MARGIN + pill_h), (15, 15, 15), -1)
        cv2.addWeighted(pill, 0.6, img, 0.4, 0, img)
        cv2.rectangle(img, (BEV_MARGIN, BEV_MARGIN),
                      (BEV_MARGIN + pill_w, BEV_MARGIN + pill_h),
                      (80, 80, 80), 1)

        # Title
        cv2.putText(img, 'CREStE-Nano',
                    (BEV_MARGIN + 14, BEV_MARGIN + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.78, (240, 240, 240), 2, cv2.LINE_AA)

        # Green status dot + label (very small, top-right of the pill)
        dot_x = BEV_MARGIN + pill_w - 130
        dot_y = BEV_MARGIN + 22
        cv2.circle(img, (dot_x, dot_y), 6, (90, 230, 110), -1)
        cv2.circle(img, (dot_x, dot_y), 7, (220, 220, 220), 1)
        cv2.putText(img, 'AUTONOMOUS',
                    (dot_x + 12, dot_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 240, 220), 1, cv2.LINE_AA)

        # Steering bar (thin, modern)
        bar_x = BEV_MARGIN + 14
        bar_y = BEV_MARGIN + 50
        bar_w = pill_w - 28
        bar_h = 8
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (45, 45, 45), -1)
        centre = bar_x + bar_w // 2
        fill_x = centre + int(steer * (bar_w // 2))
        fill_x = max(bar_x, min(bar_x + bar_w, fill_x))
        cv2.rectangle(img, (min(centre, fill_x), bar_y),
                      (max(centre, fill_x), bar_y + bar_h),
                      (255, 240, 90), -1)
        cv2.line(img, (centre, bar_y - 2), (centre, bar_y + bar_h + 2),
                 (180, 180, 180), 1)
        cv2.putText(img, f'steer {steer:+.2f}',
                    (bar_x, bar_y + bar_h + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(img, f'frame {i+1}/{n}',
                    (bar_x + bar_w - 110, bar_y + bar_h + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 160, 160), 1, cv2.LINE_AA)

        writer.write(img)

        if (i + 1) % 200 == 0:
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
