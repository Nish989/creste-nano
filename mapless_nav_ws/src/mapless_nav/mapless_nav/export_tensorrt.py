"""
Export DINOv2-small and Depth Anything V2 small to TensorRT engines.

Pipeline: PyTorch → ONNX → trtexec → .engine

Usage:
  python3 -m mapless_nav.export_tensorrt
  python3 -m mapless_nav.export_tensorrt --fp16   # recommended for Jetson

Requires: torch, onnx, tensorrt (via trtexec)
"""
import argparse
import os
import sys
import subprocess
import torch
import numpy as np

TRTEXEC = '/usr/src/tensorrt/bin/trtexec'
INPUT_SIZE = 518


def export_dino_onnx(model_dir, onnx_path, device):
    """Export DINOv2-small to ONNX."""
    print('=== Exporting DINOv2-small to ONNX ===')
    pth_path = os.path.join(model_dir, 'dinov2_small', 'dinov2_vits14.pth')

    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=False)
    dino.load_state_dict(torch.load(pth_path, map_location=device, weights_only=True))
    dino = dino.to(device).eval()

    # DINOv2 forward_features returns a dict; we need a wrapper that returns
    # just the patch tokens tensor for ONNX export
    class DINOv2Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            out = self.model.forward_features(x)
            return out['x_norm_patchtokens']  # [B, N, 384]

    wrapper = DINOv2Wrapper(dino).eval()
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)

    torch.onnx.export(
        wrapper, dummy, onnx_path,
        input_names=['input'],
        output_names=['patch_tokens'],
        opset_version=17,
        do_constant_folding=True,
    )
    print(f'  Saved ONNX: {onnx_path}')


def export_depth_onnx(model_dir, onnx_path, device):
    """Export Depth Anything V2 small to ONNX."""
    print('=== Exporting Depth Anything V2 small to ONNX ===')
    pth_path = os.path.join(model_dir, 'depth_anything_v2_small', 'depth_anything_v2_vits.pth')
    da2_repo = os.path.join(model_dir, 'depth_anything_v2_small', 'Depth-Anything-V2')

    if da2_repo not in sys.path:
        sys.path.insert(0, da2_repo)
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(pth_path, map_location=device, weights_only=True))
    model = model.to(device).eval()

    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)

    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=['input'],
        output_names=['depth'],
        opset_version=17,
        do_constant_folding=True,
    )
    print(f'  Saved ONNX: {onnx_path}')


def onnx_to_engine(onnx_path, engine_path, fp16=True):
    """Convert ONNX to TensorRT engine via trtexec."""
    print(f'=== Building TensorRT engine: {os.path.basename(engine_path)} ===')
    cmd = [
        TRTEXEC,
        f'--onnx={onnx_path}',
        f'--saveEngine={engine_path}',
        '--memPoolSize=workspace:2048MiB',
    ]
    if fp16:
        cmd.append('--fp16')
        print('  Using FP16 precision')

    print(f'  Running: {" ".join(cmd)}')
    print('  This may take several minutes on Jetson...')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'  ERROR: trtexec failed:\n{result.stderr[-2000:]}')
        return False
    print(f'  Saved engine: {engine_path}')
    return True


def main():
    parser = argparse.ArgumentParser(description='Export models to TensorRT')
    parser.add_argument('--model_dir', default=os.path.expanduser('~/models'))
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='Use FP16 precision (default, recommended for Jetson)')
    parser.add_argument('--fp32', action='store_true',
                        help='Use FP32 precision instead of FP16')
    parser.add_argument('--only', choices=['dino', 'depth'], default=None,
                        help='Only export one model')
    args = parser.parse_args()

    use_fp16 = not args.fp32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}, FP16: {use_fp16}')

    if not os.path.exists(TRTEXEC):
        print(f'ERROR: trtexec not found at {TRTEXEC}')
        sys.exit(1)

    dino_dir = os.path.join(args.model_dir, 'dinov2_small')
    depth_dir = os.path.join(args.model_dir, 'depth_anything_v2_small')

    if args.only != 'depth':
        onnx_path = os.path.join(dino_dir, 'dinov2_small.onnx')
        engine_path = os.path.join(dino_dir, 'dinov2_small.engine')
        export_dino_onnx(args.model_dir, onnx_path, device)
        if not onnx_to_engine(onnx_path, engine_path, fp16=use_fp16):
            print('DINOv2 TRT conversion failed')
            if args.only:
                sys.exit(1)

    if args.only != 'dino':
        onnx_path = os.path.join(depth_dir, 'depth_anything_v2_small.onnx')
        engine_path = os.path.join(depth_dir, 'depth_anything_v2_small.engine')
        export_depth_onnx(args.model_dir, onnx_path, device)
        if not onnx_to_engine(onnx_path, engine_path, fp16=use_fp16):
            print('Depth Anything V2 TRT conversion failed')
            if args.only:
                sys.exit(1)

    print('\nDone! To use TRT engines, set use_tensorrt:=true in your launch or params.yaml')


if __name__ == '__main__':
    main()
