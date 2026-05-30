# Build the architecture diagram for the README and paper.
# Writes docs/architecture.{png,pdf}.

import os, sys
_user_sp = os.path.expanduser('~/Library/Python/3.12/lib/python/site-packages')
if _user_sp not in sys.path:
    sys.path.insert(0, _user_sp)

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch


OUT_DIR = os.path.expanduser('~/Desktop/JOYDEEP/docs')
os.makedirs(OUT_DIR, exist_ok=True)


def box(ax, x, y, w, h, title, sub=None):
    ax.add_patch(Rectangle((x, y), w, h, facecolor='white',
                           edgecolor='black', linewidth=1.0, zorder=2))
    if sub:
        ax.text(x + w / 2, y + h * 0.66, title,
                ha='center', va='center', fontsize=10.5,
                color='black', fontweight='normal')
        ax.text(x + w / 2, y + h * 0.30, sub,
                ha='center', va='center', fontsize=8.5,
                color='#444')
    else:
        ax.text(x + w / 2, y + h / 2, title,
                ha='center', va='center', fontsize=10.5,
                color='black')


def arrow(ax, x0, y0, x1, y1, dashed=False, label=None):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle='->', mutation_scale=12,
        linewidth=0.9, color='black',
        linestyle='--' if dashed else '-', zorder=3))
    if label:
        ax.text((x0 + x1) / 2 + 0.18, (y0 + y1) / 2, label,
                ha='left', va='center', fontsize=8, color='#555',
                style='italic')


def build_figure():
    fig, ax = plt.subplots(figsize=(8.0, 11.0))
    ax.set_xlim(0, 8); ax.set_ylim(0, 14)
    ax.set_aspect('equal'); ax.axis('off')

    CX = 2.5
    BW = 3.0
    BH = 0.85
    GAP = 0.50

    blocks = [
        ('Camera',                   '1280x720, manual exposure'),
        ('DINOv2 + Depth Anything',  '384-dim patch features + depth'),
        ('BEV Projection',           '64x64 grid, 15 cm/cell'),
        ('TraversabilityUNet',       'per-cell sigmoid'),
        ('MPPI Planner',             'K=1000, T=8'),
        ('Safety + PWM',             None),
        ('ESP8266 Serial Bridge',    None),
        ('ESC + Steering Servo',     None),
    ]

    top = 13.2
    ys = []
    y = top
    for _ in blocks:
        ys.append(y - BH)
        y -= (BH + GAP)

    for (title, sub), yb in zip(blocks, ys):
        box(ax, CX, yb, BW, BH, title, sub)

    for i in range(len(blocks) - 1):
        arrow(ax, CX + BW / 2, ys[i], CX + BW / 2, ys[i + 1] + BH)

    # GPS sidecar -> MPPI
    gps_x, gps_w = 0.4, 1.6
    mppi_y = ys[4] + BH / 2
    box(ax, gps_x, mppi_y - 0.30, gps_w, 0.60, 'GPS')
    arrow(ax, gps_x + gps_w, mppi_y, CX, mppi_y, dashed=True)

    fig.text(0.5, 0.965,
             'CREStE-Nano architecture',
             ha='center', fontsize=12, fontweight='bold', color='black')

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


if __name__ == '__main__':
    fig = build_figure()
    png = os.path.join(OUT_DIR, 'architecture.png')
    pdf = os.path.join(OUT_DIR, 'architecture.pdf')
    fig.savefig(png, dpi=180, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf, bbox_inches='tight', facecolor='white')
    print(f'wrote {png}')
    print(f'wrote {pdf}')
    plt.close(fig)
