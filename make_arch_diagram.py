"""make_arch_diagram.py — clean vertical-flow architecture diagram.

Output: docs/architecture.png  +  docs/architecture.pdf
"""
import os, sys
_user_sp = os.path.expanduser('~/Library/Python/3.12/lib/python/site-packages')
if _user_sp not in sys.path:
    sys.path.insert(0, _user_sp)

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle


OUT_DIR = os.path.expanduser('~/Desktop/JOYDEEP/docs')
os.makedirs(OUT_DIR, exist_ok=True)


PALETTE = {
    'sensor':    {'face': '#e8f1ff', 'edge': '#3b6fb8', 'text': '#13315c'},
    'perception':{'face': '#e7f6e2', 'edge': '#2f8f3f', 'text': '#11471a'},
    'learning':  {'face': '#fff0c2', 'edge': '#c08a16', 'text': '#5a3e02'},
    'planning':  {'face': '#fbe2e2', 'edge': '#c33f3f', 'text': '#5a1414'},
    'control':   {'face': '#ece2fb', 'edge': '#7a4dc4', 'text': '#2c0e58'},
    'actuation': {'face': '#f1eedb', 'edge': '#8b7733', 'text': '#3a3209'},
    'side':      {'face': '#f4f4f4', 'edge': '#888', 'text': '#444'},
}


def block(ax, x, y, w, h, title, subtitle, kind,
          fontsize_title=10.5, fontsize_sub=8.5):
    s = PALETTE[kind]
    box = FancyBboxPatch((x, y), w, h,
        boxstyle='round,pad=0.02,rounding_size=0.05',
        facecolor=s['face'], edgecolor=s['edge'], linewidth=1.5, zorder=2)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h - 0.13, title,
            ha='center', va='top', fontsize=fontsize_title,
            color=s['text'], fontweight='bold')
    if subtitle:
        ax.text(x + w / 2, y + 0.13, subtitle,
                ha='center', va='bottom', fontsize=fontsize_sub,
                color=s['text'], linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, label=None,
          color='#222', dashed=False, lw=1.6,
          rad=0.0, label_dy=0.0, label_dx=0.22, label_color='#333',
          label_fontsize=8.5):
    ar = FancyArrowPatch((x0, y0), (x1, y1),
        arrowstyle='->', mutation_scale=16,
        linewidth=lw, color=color,
        linestyle='--' if dashed else '-',
        connectionstyle=f'arc3,rad={rad}', zorder=3)
    ax.add_patch(ar)
    if label:
        mx = (x0 + x1) / 2 + label_dx
        my = (y0 + y1) / 2 + label_dy
        ax.text(mx, my, label, ha='left', va='center',
                fontsize=label_fontsize, color=label_color, style='italic',
                bbox=dict(facecolor='white', edgecolor='none', pad=2.0))


def lane(ax, x, y, w, h, label, color, label_x=None):
    ax.add_patch(Rectangle((x, y), w, h,
        facecolor=color, edgecolor='none', alpha=0.20, zorder=1))
    lx = label_x if label_x is not None else x - 0.30
    ax.text(lx, y + h / 2, label, fontsize=9.5,
            color='#555', fontweight='bold', ha='right', va='center')


def build_figure():
    # Tall vertical canvas (extra horizontal room for sidecars)
    fig, ax = plt.subplots(figsize=(13.0, 13.5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 16)
    ax.set_aspect('equal'); ax.axis('off')

    # Central column where main pipeline lives
    CX = 4.2       # x of central blocks
    BW = 4.8       # block width
    BH = 1.05      # block height
    GAP = 0.55     # vertical gap

    # Lane bands (colored zones — purely visual)
    # We compute y-positions for the blocks first
    blocks = [
        # (kind, title, subtitle)
        ('sensor',     'EMEET 4K Webcam',
         '1280 × 720 @ 30 Hz  ·  manual 1/2000 s shutter\nMJPEG → V4L2 capture'),
        ('perception', 'DINOv2 ViT-S/14  +  Depth Anything V2',
         '384-dim patch features (14 × 14)   ·   dense monocular depth\n'
         'both TensorRT-optimised on Orin'),
        ('perception', 'BEV Projection',
         'back-project (u, v, d) → R³  ·  bin into 64 × 64 grid\n'
         '15 cm/cell  ·  9.6 m × 9.6 m footprint  →  F_BEV ∈ R^(64×64×384)'),
        ('learning',   'TraversabilityUNet  (~8 M params)',
         '384 → 1 spatial regressor  ·  weighted BCE\n'
         'labels = dilated human-rollout corridor masks'),
        ('learning',   'σ(f_ψ(F_BEV))     — 64 × 64 sigmoid heatmap',
         'pos_pred = 0.996   neg_pred = 0.004   margin = +0.992\n'
         '(reproduced over all 8 331 training-set frames)'),
        ('planning',   'MPPI Planner',
         'K = 1000 samples, T = 8 steps  ·  σ = 0.45, momentum = 0.4\n'
         'score = mean σ(f_ψ) along rollout  +  β · GPS-bearing bias'),
        ('control',    'Safety Watchdog  +  PWM Encoder',
         '500 ms /cmd_* timeout  ·  throttle clamp ±0.38  ·  E-stop short-circuit\n'
         'cmd ∈ [-1, 1]  →  1000–2000 µs  ·  8 µs deadband'),
        ('control',    'ESP8266 Serial Bridge',
         'USB-CDC 500 kbaud  ·  50 Hz PWM update  ·  arm-retry every 0.5 s'),
        ('actuation',  'Hardware:  Brushless ESC  +  Steering Servo  →  Arrma Typhon Mega',
         '1690 µs cruise (clears cogging)  ·  1000–2000 µs steer  ·  3 S LiPo, ~12 km/h'),
    ]

    # Place top-to-bottom, compute y for each
    top = 14.5
    ys = []
    y_cursor = top
    for _ in blocks:
        ys.append(y_cursor - BH)
        y_cursor -= (BH + GAP)

    # ── Lane bands (group blocks by colour) ──────────────────────────────────
    # Indices into the blocks list:
    lane_specs = [
        (range(0, 3), 'Perception  ·  5 FPS on Jetson Orin Nano',         '#b9d2f7'),
        (range(3, 5), 'Learned reward model',                             '#ffe3a3'),
        (range(5, 6), 'Planning',                                         '#f7c9c9'),
        (range(6, 8), 'Safety + low-level control',                       '#d4c4ee'),
        (range(8, 9), 'Hardware',                                         '#e2d39a'),
    ]
    for idxs, label, color in lane_specs:
        idxs = list(idxs)
        y_top = ys[idxs[0]] + BH + 0.18
        y_bot = ys[idxs[-1]] - 0.18
        h = y_top - y_bot
        lane(ax, CX - 0.20, y_bot, BW + 0.40, h, label, color)

    # ── Draw blocks ──────────────────────────────────────────────────────────
    for (kind, title, subtitle), y in zip(blocks, ys):
        block(ax, CX, y, BW, BH, title, subtitle, kind)

    # ── Vertical arrows between consecutive blocks ──────────────────────────
    DATA_LABELS = {
        (0, 1): None,                     # camera → perception
        (1, 2): None,
        (2, 3): 'F_BEV',
        (3, 4): None,                     # UNet → σ heatmap (same block in code)
        (4, 5): 'σ(f_ψ(F_BEV))',
        (5, 6): 'steering, throttle ∈ [-1, 1]',
        (6, 7): 'safe_cmd_steering / throttle',
        (7, 8): 'PWM µs',
    }
    for (i, j), lab in DATA_LABELS.items():
        y0 = ys[i]
        y1 = ys[j] + BH
        x = CX + BW / 2
        arrow(ax, x, y0, x, y1, label=lab, label_dx=0.30, label_dy=0.0,
              label_fontsize=8.4)

    # ── Side inputs (left of the column, with arrows landing cleanly on edge) ──
    GAP_FROM_COL = 0.55   # gap between sidecar block and central column

    def side_in(y_center, title, subtitle, label, color='#666', dashed=True):
        sw, sh = 2.6, BH - 0.32
        sx = CX - GAP_FROM_COL - sw
        sy = y_center - sh / 2
        block(ax, sx, sy, sw, sh, title, subtitle, 'side',
              fontsize_title=9, fontsize_sub=7.8)
        # Arrow stops just before column edge (no head intrusion)
        arrow(ax, sx + sw, y_center, CX - 0.05, y_center,
              dashed=dashed, color=color, lw=1.3,
              label=label, label_dx=-(GAP_FROM_COL - 0.05) / 2 - 0.05,
              label_dy=0.18, label_fontsize=7.8, label_color='#555')

    # GPS bearing feeds MPPI (bias on score)
    side_in(ys[5] + BH / 2, 'GPS  (u-blox @ 10 Hz)',
            'waypoint bearing ψ\nbiases MPPI score', '+β·alignment')
    # Dashboard feeds the autonomy flag (goes to MPPI mode_cb)
    side_in(ys[5] + BH + GAP / 2, 'Web Dashboard',
            'phone-friendly UI\nE-stop / waypoints / log', '/autonomous_mode',
            color='#888')

    # Future RLHF (right of MPPI/UNet, dashed, deliberately less prominent)
    fb_w, fb_h = 2.8, BH - 0.32
    fb_x = CX + BW + GAP_FROM_COL
    fb_y = ys[4] + (BH - fb_h) / 2
    block(ax, fb_x, fb_y, fb_w, fb_h,
          'Future: Online RLHF',
          'log interventions  ·  M = 500 FIFO\nonline BCE update on UNet',
          'side', fontsize_title=9, fontsize_sub=7.8)
    arrow(ax, fb_x, fb_y + fb_h / 2, CX + BW + 0.05, ys[3] + BH / 2,
          dashed=True, color='#888', rad=-0.12, lw=1.3,
          label='future feedback', label_dx=0.18, label_dy=0.35,
          label_fontsize=7.6, label_color='#777')

    # ── Title ────────────────────────────────────────────────────────────────
    fig.text(0.50, 0.965, 'CREStE-Nano — end-to-end architecture',
             ha='center', fontsize=15, fontweight='bold', color='#222')
    fig.text(0.50, 0.943,
             r'monocular RGB + GPS  $\rightarrow$  BEV traversability '
             r'$\rightarrow$  MPPI  $\rightarrow$  PWM  '
             r'$\;\cdot\;$  ~\$500 BOM  '
             r'$\;\cdot\;$  all 13 ROS 2 nodes on a single Jetson Orin Nano',
             ha='center', fontsize=10, color='#555')

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


if __name__ == '__main__':
    fig = build_figure()
    png = os.path.join(OUT_DIR, 'architecture.png')
    pdf = os.path.join(OUT_DIR, 'architecture.pdf')
    fig.savefig(png, dpi=170, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf, bbox_inches='tight', facecolor='white')
    print(f'wrote {png}')
    print(f'wrote {pdf}')
    plt.close(fig)
