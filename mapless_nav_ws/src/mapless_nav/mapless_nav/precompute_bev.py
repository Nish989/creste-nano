import argparse
import os
import sys
import json
import glob
import math
import numpy as np
import cv2
import time

import torch


def precompute_patch_rays(n, fov_h, fov_v, cam_pitch):
    px = np.arange(n)
    py = np.arange(n)
    px_grid, py_grid = np.meshgrid(px, py)
    u = (px_grid.ravel() + 0.5) / n - 0.5
    v = (py_grid.ravel() + 0.5) / n - 0.5
    angles_h = u * fov_h
    angles_v = v * fov_v + cam_pitch
    return angles_h, angles_v, py_grid.ravel(), px_grid.ravel()


def project_to_bev(features, depth_u8, angles_h, angles_v, n,
                    bev_h=64, bev_w=64, bev_res=0.15, depth_py=None, depth_px=None):
    feat_dim = features.shape[1]
    depth_h, depth_w = depth_u8.shape

    # Generate patch indices if not provided (backward compat)
    if depth_py is None or depth_px is None:
        px = np.arange(n)
        py = np.arange(n)
        px_grid, py_grid = np.meshgrid(px, py)
        depth_py = py_grid.ravel()
        depth_px = px_grid.ravel()

    # Sample depth at patch centers — vectorized
    dy = np.minimum(((depth_py + 0.5) / n * depth_h).astype(int), depth_h - 1)
    dx = np.minimum(((depth_px + 0.5) / n * depth_w).astype(int), depth_w - 1)
    depth_patches = depth_u8[dy, dx].astype(np.float32)

    max_depth_val = depth_patches.max()
    if max_depth_val < 1:
        return np.zeros((bev_h, bev_w, feat_dim), dtype=np.float32)

    # Convert to meters and filter
    depth_m = (depth_patches / max_depth_val) * 10.0
    valid = depth_m >= 0.3

    # 3D projection — vectorized
    x = depth_m * np.sin(angles_h)
    z = depth_m * np.cos(angles_h) * np.cos(angles_v)

    bev_x = (x / bev_res + bev_w / 2).astype(int)
    bev_y = (bev_h - z / bev_res).astype(int)

    valid &= (bev_x >= 0) & (bev_x < bev_w) & (bev_y >= 0) & (bev_y < bev_h)

    bev_x_v = bev_x[valid]
    bev_y_v = bev_y[valid]
    feats_v = features[valid]

    bev = np.zeros((bev_h, bev_w, feat_dim), dtype=np.float32)
    counts = np.zeros((bev_h, bev_w), dtype=np.float32)
    np.add.at(bev, (bev_y_v, bev_x_v), feats_v)
    np.add.at(counts, (bev_y_v, bev_x_v), 1)

    mask = counts > 0
    bev[mask] /= counts[mask, np.newaxis]
    return bev


def main():
    parser = argparse.ArgumentParser(description='Precompute depth-projected BEV features')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/mapless_nav_data'))
    parser.add_argument('--output_dir', default=None,
                        help='Output dir (default: <data_dir>/bev_features)')
    parser.add_argument('--model_dir', default=os.path.expanduser('~/models'))
    parser.add_argument('--batch_size', type=int, default=1)
    # BEV params (must match params.yaml / bev_projection_node)
    parser.add_argument('--bev_width', type=int, default=64)
    parser.add_argument('--bev_height', type=int, default=64)
    parser.add_argument('--bev_res', type=float, default=0.15)
    parser.add_argument('--cam_height', type=float, default=0.35)
    parser.add_argument('--cam_pitch', type=float, default=15.0)
    parser.add_argument('--fov_h', type=float, default=90.0)  # EMEET Nova 4K
    parser.add_argument('--fov_v', type=float, default=58.0)  # EMEET Nova 4K
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.data_dir, 'bev_features')
    os.makedirs(output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Device: {device}')

    # --- Load DINOv2 ---
    dino_path = os.path.join(args.model_dir, 'dinov2_small', 'dinov2_vits14.pth')
    if os.path.exists(dino_path):
        dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=False)
        dino.load_state_dict(torch.load(dino_path, map_location=device, weights_only=True))
    else:
        print('Downloading DINOv2-small...')
        dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        os.makedirs(os.path.dirname(dino_path), exist_ok=True)
        torch.save(dino.state_dict(), dino_path)
    dino = dino.to(device).eval()

    # --- Load Depth Anything V2 ---
    depth_path = os.path.join(args.model_dir, 'depth_anything_v2_small', 'depth_anything_v2_vits.pth')
    da2_repo = os.path.join(args.model_dir, 'depth_anything_v2_small', 'Depth-Anything-V2')
    if not os.path.exists(depth_path) or not os.path.isdir(da2_repo):
        print(f'ERROR: Depth model not found at {depth_path}')
        print(f'       or repo missing at {da2_repo}')
        sys.exit(1)
    if da2_repo not in sys.path:
        sys.path.insert(0, da2_repo)
    from depth_anything_v2.dpt import DepthAnythingV2
    depth_model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
    depth_model.load_state_dict(torch.load(depth_path, map_location=device, weights_only=True))
    depth_model = depth_model.to(device).eval()

    # --- Precompute ray directions ---
    n = 518 // 14  # 37 patches per side
    fov_h = math.radians(args.fov_h)
    fov_v = math.radians(args.fov_v)
    cam_pitch = math.radians(args.cam_pitch)
    angles_h, angles_v, patch_py, patch_px = precompute_patch_rays(n, fov_h, fov_v, cam_pitch)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    def preprocess(img_bgr, size=518):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size))
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).unsqueeze(0).float().to(device)

    # --- Gather all frames across sessions ---
    sessions = sorted(glob.glob(os.path.join(args.data_dir, 'session_*')))
    all_entries = []
    for session_dir in sessions:
        metadata_path = os.path.join(session_dir, 'metadata.jsonl')
        if not os.path.exists(metadata_path):
            continue
        with open(metadata_path) as f:
            for line in f:
                entry = json.loads(line)
                entry['_session_dir'] = session_dir
                all_entries.append(entry)

    print(f'Found {len(all_entries)} frames across {len(sessions)} sessions')

    # --- Process ---
    global_idx = 0
    t0 = time.time()
    skipped = 0

    for i, entry in enumerate(all_entries):
        img_path = os.path.join(entry['_session_dir'], 'images', entry['image'])
        if not os.path.exists(img_path):
            skipped += 1
            continue

        out_path = os.path.join(output_dir, f'{global_idx:06d}.npz')
        # Skip if already computed
        if os.path.exists(out_path):
            global_idx += 1
            continue

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            skipped += 1
            continue

        with torch.no_grad():
            # DINOv2 features
            tensor = preprocess(img_bgr, 518)
            features = dino.forward_features(tensor)
            patches = features['x_norm_patchtokens'].cpu().numpy().squeeze(0)  # [1369, 384]

            # Depth
            depth_out = depth_model(tensor)  # [1, 1, H, W]
            depth_np = depth_out.cpu().numpy().squeeze()
            depth_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX)
            depth_u8 = depth_norm.astype(np.uint8)

        # Project to BEV
        bev = project_to_bev(
            patches, depth_u8, angles_h, angles_v, n,
            args.bev_height, args.bev_width, args.bev_res,
            depth_py=patch_py, depth_px=patch_px)

        np.savez_compressed(out_path,
            bev=bev,
            steering=entry.get('steering', 0.0),
            throttle=entry.get('throttle', 0.0),
            heading=entry.get('heading', 0.0),
        )
        global_idx += 1

        # Progress
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            fps = (i + 1 - skipped) / elapsed
            eta = (len(all_entries) - i - 1) / max(fps, 0.1)
            print(f'[{i+1}/{len(all_entries)}] {fps:.1f} frames/sec, '
                  f'ETA {eta/60:.1f} min, skipped {skipped}')

    elapsed = time.time() - t0
    print(f'\nDone: {global_idx} BEV features saved to {output_dir}')
    print(f'Time: {elapsed/60:.1f} min ({global_idx/max(elapsed,1):.1f} fps)')
    print(f'Skipped: {skipped} frames (missing images)')


if __name__ == '__main__':
    main()
