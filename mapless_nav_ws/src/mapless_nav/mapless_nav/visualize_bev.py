import argparse
import os
import sys
import json
import math
import glob
import numpy as np
import cv2

import torch

# BEV grid params (must match bev_projection_node.py)
BEV_W, BEV_H = 64, 64
BEV_RES = 0.15
CAM_HEIGHT = 0.35
CAM_PITCH = math.radians(15.0)
FOV_H = math.radians(75.0)
FOV_V = math.radians(50.0)
DINO_INPUT = 518
DEPTH_INPUT = 518
PATCH_SIZE = 14
N_PATCHES = DINO_INPUT // PATCH_SIZE  # 37

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def precompute_patch_rays():
    n = N_PATCHES
    angles_h = np.zeros(n * n)
    angles_v = np.zeros(n * n)
    for py in range(n):
        for px in range(n):
            idx = py * n + px
            u = (px + 0.5) / n - 0.5
            v = (py + 0.5) / n - 0.5
            angles_h[idx] = u * FOV_H
            angles_v[idx] = v * FOV_V + CAM_PITCH
    return angles_h, angles_v


def preprocess(frame, size, device):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size)).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0).float().to(device)


def load_models(model_dir, device):
    # DINOv2
    dino_path = os.path.join(model_dir, 'dinov2_small', 'dinov2_vits14.pth')
    if os.path.exists(dino_path):
        dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=False)
        dino.load_state_dict(torch.load(dino_path, map_location=device, weights_only=True))
    else:
        print('Downloading DINOv2-small...')
        dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        os.makedirs(os.path.dirname(dino_path), exist_ok=True)
        torch.save(dino.state_dict(), dino_path)
    dino = dino.to(device).eval()

    # Depth Anything V2
    depth_path = os.path.join(model_dir, 'depth_anything_v2_small', 'depth_anything_v2_vits.pth')
    da2_repo = os.path.join(model_dir, 'depth_anything_v2_small', 'Depth-Anything-V2')
    depth_model = None
    if os.path.exists(depth_path) and os.path.isdir(da2_repo):
        if da2_repo not in sys.path:
            sys.path.insert(0, da2_repo)
        from depth_anything_v2.dpt import DepthAnythingV2
        depth_model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
        depth_model.load_state_dict(torch.load(depth_path, map_location=device, weights_only=True))
        depth_model = depth_model.to(device).eval()
    else:
        print(f'WARNING: Depth model not found at {depth_path}')
        print('         Depth-projected BEV and point clouds will be skipped.')

    return dino, depth_model


def training_bev(patches):
    n = int(np.sqrt(patches.shape[0]))  # 37
    spatial = patches.reshape(n, n, -1)
    bev = np.zeros((BEV_H, BEV_W, 384), dtype=np.float32)
    for c in range(384):
        bev[:, :, c] = cv2.resize(spatial[:, :, c], (BEV_W, BEV_H))
    return bev


def depth_projected_bev(patches, depth_u8, angles_h, angles_v):
    n = N_PATCHES
    bev = np.zeros((BEV_H, BEV_W, 384), dtype=np.float32)
    counts = np.zeros((BEV_H, BEV_W), dtype=np.float32)

    dh, dw = depth_u8.shape
    depth_patches = np.zeros(n * n)
    for py in range(n):
        for px in range(n):
            dy = min(int((py + 0.5) / n * dh), dh - 1)
            dx = min(int((px + 0.5) / n * dw), dw - 1)
            depth_patches[py * n + px] = depth_u8[dy, dx]

    max_d = depth_patches.max()
    if max_d < 1:
        return bev, counts, np.zeros((0, 3))

    points_3d = []
    for i in range(n * n):
        ah = angles_h[i]
        av = angles_v[i]
        depth_m = (depth_patches[i] / max_d) * 10.0
        if depth_m < 0.3:
            continue

        x = depth_m * math.sin(ah)
        z = depth_m * math.cos(ah) * math.cos(av)
        y = depth_m * math.sin(av)
        points_3d.append([x, y, z])

        bev_x = int(x / BEV_RES + BEV_W / 2)
        bev_y = int(BEV_H - z / BEV_RES)
        if 0 <= bev_x < BEV_W and 0 <= bev_y < BEV_H:
            bev[bev_y, bev_x] += patches[i]
            counts[bev_y, bev_x] += 1

    mask = counts > 0
    bev[mask] /= counts[mask, np.newaxis]
    return bev, counts, np.array(points_3d) if points_3d else np.zeros((0, 3))


def pca_rgb(bev, mask=None):
    if mask is None:
        mask = np.linalg.norm(bev, axis=2) > 0

    if mask.sum() < 3:
        return np.full((BEV_H, BEV_W, 3), 20, dtype=np.uint8)

    feats = bev[mask]
    feats_c = feats - feats.mean(axis=0)
    _, _, Vt = np.linalg.svd(feats_c, full_matrices=False)
    rgb = feats_c @ Vt[:3].T

    for c in range(3):
        mn, mx = rgb[:, c].min(), rgb[:, c].max()
        if mx - mn > 1e-8:
            rgb[:, c] = (rgb[:, c] - mn) / (mx - mn) * 255
        else:
            rgb[:, c] = 128

    img = np.full((BEV_H, BEV_W, 3), 20, dtype=np.uint8)
    ys, xs = np.where(mask)
    img[ys, xs] = rgb.astype(np.uint8)
    return img


def render_pointcloud(points_3d):
    """Render point cloud from top-down view."""
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    if len(points_3d) == 0:
        return img

    x = points_3d[:, 0]
    z = points_3d[:, 2]
    y = points_3d[:, 1]

    coverage = BEV_W * BEV_RES / 2
    px = ((x / coverage + 1) * 256).astype(int)
    py = (512 - z / (BEV_H * BEV_RES) * 512).astype(int)

    y_range = y.max() - y.min()
    if y_range > 1e-8:
        y_norm = ((y - y.min()) / y_range * 255).astype(np.uint8)
    else:
        y_norm = np.full_like(y, 128, dtype=np.uint8)
    colors = cv2.applyColorMap(y_norm.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)

    for i in range(len(points_3d)):
        if 0 <= px[i] < 512 and 0 <= py[i] < 512:
            cv2.circle(img, (px[i], py[i]), 4, colors[i].tolist(), -1)

    return img


def upscale(img, size=384):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)


def add_label(img, text):
    h, w = img.shape[:2]
    bar = np.zeros((30, w, 3), dtype=np.uint8)
    cv2.putText(bar, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    return np.vstack([bar, img])


def main():
    parser = argparse.ArgumentParser(description='Visualize BEV maps and point clouds')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/mapless_nav_data'))
    parser.add_argument('--model_dir', default=os.path.expanduser('~/models'))
    parser.add_argument('--sample_every', type=int, default=20,
                        help='Process every Nth frame (default: 20)')
    parser.add_argument('--session', default=None,
                        help='Specific session directory name (default: all)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    dino, depth_model = load_models(args.model_dir, device)
    angles_h, angles_v = precompute_patch_rays()

    # Find sessions
    if args.session:
        sessions = [os.path.join(args.data_dir, args.session)]
    else:
        sessions = sorted(glob.glob(os.path.join(args.data_dir, 'session_*')))

    if not sessions:
        print(f'No sessions found in {args.data_dir}')
        return

    out_dir = os.path.join(args.data_dir, 'visualizations')
    os.makedirs(out_dir, exist_ok=True)

    total_frames = 0
    for session_dir in sessions:
        metadata_path = os.path.join(session_dir, 'metadata.jsonl')
        if not os.path.exists(metadata_path):
            print(f'Skipping {session_dir} (no metadata.jsonl)')
            continue

        session_name = os.path.basename(session_dir)
        session_out = os.path.join(out_dir, session_name)
        os.makedirs(session_out, exist_ok=True)

        with open(metadata_path) as f:
            entries = [json.loads(line) for line in f]

        sample_entries = entries[::args.sample_every]
        print(f'\n{session_name}: {len(sample_entries)} frames '
              f'(of {len(entries)}, every {args.sample_every})')

        for count, entry in enumerate(sample_entries):
            img_path = os.path.join(session_dir, 'images', entry['image'])
            if not os.path.exists(img_path):
                continue

            frame = cv2.imread(img_path)
            if frame is None:
                continue

            frame_id = entry.get('frame', count)
            print(f'  [{count+1}/{len(sample_entries)}] frame {frame_id}...', end=' ', flush=True)

            with torch.no_grad():
                # DINOv2 features
                dino_input = preprocess(frame, DINO_INPUT, device)
                feats = dino.forward_features(dino_input)
                patches = feats['x_norm_patchtokens'].cpu().numpy().squeeze(0)

                # Depth
                depth_u8 = None
                if depth_model is not None:
                    depth_input = preprocess(frame, DEPTH_INPUT, device)
                    depth_raw = depth_model(depth_input).cpu().numpy().squeeze()
                    depth_u8 = cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Training BEV (spatial resize, what train_reward.py uses)
            train_bev = training_bev(patches)
            train_bev_rgb = pca_rgb(train_bev)

            # Build visualization panels
            SZ = 384
            vis_orig = add_label(cv2.resize(frame, (SZ, SZ)), 'Original')

            vis_train_bev = add_label(upscale(train_bev_rgb, SZ), 'Training BEV (PCA)')

            if depth_u8 is not None:
                # Depth colorized
                vis_depth = add_label(
                    cv2.applyColorMap(cv2.resize(depth_u8, (SZ, SZ)), cv2.COLORMAP_INFERNO),
                    'Depth (Anything V2)')

                # Depth-projected BEV
                proj_bev, proj_counts, pts3d = depth_projected_bev(
                    patches, depth_u8, angles_h, angles_v)
                proj_bev_rgb = pca_rgb(proj_bev, proj_counts > 0)
                vis_proj_bev = add_label(upscale(proj_bev_rgb, SZ), 'Depth-Projected BEV (PCA)')

                # Point cloud
                vis_pc = add_label(
                    cv2.resize(render_pointcloud(pts3d), (SZ, SZ)),
                    'Point Cloud (top-down)')

                # 2x3 grid: orig | depth | pointcloud
                #            train_bev | proj_bev | info
                info = np.zeros((SZ, SZ, 3), dtype=np.uint8)
                steer = entry.get('steering', 0)
                throttle = entry.get('throttle', 0)
                lines = [
                    f'Frame: {frame_id}',
                    f'Steering: {steer:.3f}',
                    f'Throttle: {throttle:.3f}',
                    f'Points: {len(pts3d)}',
                    f'BEV cells: {int((proj_counts > 0).sum())}',
                ]
                gps = entry.get('gps', {})
                if isinstance(gps, dict) and gps.get('lat'):
                    lines.append(f'GPS: {gps["lat"]:.4f}, {gps["lon"]:.4f}')
                for i, line in enumerate(lines):
                    cv2.putText(info, line, (15, 40 + i * 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                info = add_label(info, 'Info')

                row1 = np.hstack([vis_orig, vis_depth, vis_pc])
                row2 = np.hstack([vis_train_bev, vis_proj_bev, info])
            else:
                # No depth model — just show original + training BEV
                info = np.zeros((SZ, SZ, 3), dtype=np.uint8)
                info = add_label(info, 'No Depth Model')
                row1 = np.hstack([vis_orig, vis_train_bev, info])
                row2 = None

            grid = np.vstack([row1, row2]) if row2 is not None else row1

            fname = f'{frame_id:06d}_grid.jpg'
            cv2.imwrite(os.path.join(session_out, fname), grid,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            total_frames += 1
            print('done')

    print(f'\nSaved {total_frames} visualizations to {out_dir}/')


if __name__ == '__main__':
    main()
