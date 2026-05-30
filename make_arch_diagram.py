# Build the architecture diagram for the README and paper.
# Writes docs/architecture.{png,pdf}.
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
    # Canvas with room for sidecar blocks on the left
    fig, ax = plt.subplots(figsize=(13.0, 13.5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 16)
    ax.set_aspect('equal'); ax.axis('off')

    # Central column where main pipeline lives
    CX = 4.2       # x of central blocks
    BW = 4.8       # block width
    BH = 1.05      # block height
    GAP = 0.55     # vertical gap

    # Lane bands are visual grouping only. Compute block y-positions first.
    blocks = [
        # (kind, title, subtitle)
        ('sensor',     'Camera',
         '1280x720, manual exposure'),
        ('perception', 'DINOv2 + Depth Anything V2',
         '384-dim patch features\nmonocular dense depth'),
        ('perception', 'BEV Projection',
         '64x64 grid, 15 cm/cell\n9.6 m forward'),
        ('learning',   'TraversabilityUNet',
         'per-cell sigmoid\nweighted BCE on corridor masks'),
        ('learning',   'Traversability heatmap',
         '64x64'),
        ('planning',   'MPPI Planner',
         'K = 1000, T = 8\nscores from heatmap + GPS bearing'),
        ('control',    'Safety + PWM',
         '500 ms timeout\nthrottle clamp'),
        ('control',    'ESP8266 Serial Bridge',
         '50 Hz PWM'),
        ('actuation',  'ESC + Steering Servo',
         'Arrma Typhon Mega chassis'),
    ]

    # Place top-to-bottom, compute y for each
    top = 14.5
    ys = []
    y_cursor = top
    for _ in blocks:
        ys.append(y_cursor - BH)
        y_cursor -= (BH + GAP)

    # Lane bands (group blocks by colour)
    # Indices into the blocks list:
    lane_specs = [
        (range(0, 3), 'Perception',                  '#b9d2f7'),
        (range(3, 5), 'Reward model',                 '#ffe3a3'),
        (range(5, 6), 'Planning',                     '#f7c9c9'),
        (range(6, 8), 'Safety + control',             '#d4c4ee'),
        (range(8, 9), 'Hardware',                     '#e2d39a'),
    ]
    for idxs, label, color in lane_specs:
        idxs = list(idxs)
        y_top = ys[idxs[0]] + BH + 0.18
        y_bot = ys[idxs[-1]] - 0.18
        h = y_top - y_bot
        lane(ax, CX - 0.20, y_bot, BW + 0.40, h, label, color)

    # Draw blocks
    for (kind, title, subtitle), y in zip(blocks, ys):
        block(ax, CX, y, BW, BH, title, subtitle, kind)

    # Plain vertical arrows between consecutive blocks (no inline labels)
    for i in range(len(blocks) - 1):
        y0 = ys[i]
        y1 = ys[i + 1] + BH
        x = CX + BW / 2
        arrow(ax, x, y0, x, y1)

    # Sidecar inputs (left of the column)
    GAP_FROM_COL = 0.55

    def side_in(y_center, title, subtitle, color='#666', dashed=True):
        sw, sh = 2.4, BH - 0.40
        sx = CX - GAP_FROM_COL - sw
        sy = y_center - sh / 2
        block(ax, sx, sy, sw, sh, title, subtitle, 'side',
              fontsize_title=9, fontsize_sub=7.8)
        arrow(ax, sx + sw, y_center, CX - 0.05, y_center,
              dashed=dashed, color=color, lw=1.3)

    side_in(ys[5] + BH / 2, 'GPS', 'waypoint bearing')
    side_in(ys[5] + BH + GAP / 2, 'Dashboard', 'autonomous_mode flag',
            color='#888')

    # Future RLHF (right side, dashed, deliberately faint)
    fb_w, fb_h = 2.4, BH - 0.40
    fb_x = CX + BW + GAP_FROM_COL
    fb_y = ys[4] + (BH - fb_h) / 2
    block(ax, fb_x, fb_y, fb_w, fb_h,
          'Online RLHF',
          'future work',
          'side', fontsize_title=9, fontsize_sub=7.8)
    arrow(ax, fb_x, fb_y + fb_h / 2, CX + BW + 0.05, ys[3] + BH / 2,
          dashed=True, color='#888', rad=-0.12, lw=1.3)

    # Title
    fig.text(0.50, 0.965, 'CREStE-Nano end-to-end architecture',
             ha='center', fontsize=14, fontweight='bold', color='#222')

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
