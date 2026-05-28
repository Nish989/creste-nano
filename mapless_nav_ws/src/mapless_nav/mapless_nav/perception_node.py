import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import numpy as np
import cv2
import os
import time
import array


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        self.declare_parameter('model_dir', os.path.expanduser('~/models'))
        self.declare_parameter('input_size', 518)  # DINOv2 native size
        self.declare_parameter('depth_size', 518)
        self.declare_parameter('use_tensorrt', False)

        self.model_dir = self.get_parameter('model_dir').value
        self.input_size = self.get_parameter('input_size').value
        self.depth_size = self.get_parameter('depth_size').value
        self.use_trt = self.get_parameter('use_tensorrt').value

        os.makedirs(self.model_dir, exist_ok=True)

        # Load models
        self.dino_model = None
        self.depth_model = None
        self._load_models()

        # Publishers
        self.feat_pub = self.create_publisher(
            Float32MultiArray, '/perception/features', 5)
        self.depth_pub = self.create_publisher(
            Image, '/perception/depth', 5)

        # Subscriber
        self.create_subscription(CompressedImage, '/camera/image_raw/compressed', self.image_cb, 5)

        self.get_logger().info('Perception node started')
        self.frame_count = 0
        self.last_log_time = time.time()

    def _load_models(self):

        try:
            import torch
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.get_logger().info(f'Using device: {self.device}')
        except ImportError:
            self.get_logger().error('PyTorch not installed. Run perception in offline mode.')
            return

        import torch

        # Try TensorRT first
        trt_dino = os.path.join(self.model_dir, 'dinov2_small', 'dinov2_small.engine')
        trt_depth = os.path.join(self.model_dir, 'depth_anything_v2_small', 'depth_anything_v2_small.engine')

        if self.use_trt and os.path.exists(trt_dino) and os.path.exists(trt_depth):
            self._load_tensorrt(trt_dino, trt_depth)
        else:
            self._load_pytorch()

    def _load_pytorch(self):

        import torch
        import sys

        # DINOv2-small
        dino_path = os.path.join(self.model_dir, 'dinov2_small', 'dinov2_vits14.pth')
        if os.path.exists(dino_path):
            self.dino_model = torch.hub.load(
                'facebookresearch/dinov2', 'dinov2_vits14', pretrained=False)
            self.dino_model.load_state_dict(
                torch.load(dino_path, map_location=self.device, weights_only=True))
        else:
            self.get_logger().info('Downloading DINOv2-small...')
            self.dino_model = torch.hub.load(
                'facebookresearch/dinov2', 'dinov2_vits14')
            os.makedirs(os.path.dirname(dino_path), exist_ok=True)
            torch.save(self.dino_model.state_dict(), dino_path)

        self.dino_model = self.dino_model.to(self.device).eval()
        self.get_logger().info('DINOv2-small loaded (PyTorch)')

        # Depth Anything V2 small
        depth_path = os.path.join(
            self.model_dir, 'depth_anything_v2_small', 'depth_anything_v2_vits.pth')
        da2_repo = os.path.join(
            self.model_dir, 'depth_anything_v2_small', 'Depth-Anything-V2')

        if os.path.exists(depth_path) and os.path.isdir(da2_repo):
            if da2_repo not in sys.path:
                sys.path.insert(0, da2_repo)
            from depth_anything_v2.dpt import DepthAnythingV2
            self.depth_model = DepthAnythingV2(
                encoder='vits', features=64, out_channels=[48, 96, 192, 384])
            self.depth_model.load_state_dict(
                torch.load(depth_path, map_location=self.device, weights_only=True))
        else:
            self.get_logger().warn(
                f'Depth model not found at {depth_path}. '
                'Download weights and clone Depth-Anything-V2 repo.')
            return

        self.depth_model = self.depth_model.to(self.device).eval()
        self.get_logger().info('Depth Anything V2 small loaded (PyTorch)')

    def _load_tensorrt(self, dino_path, depth_path):

        try:
            import tensorrt as trt

            logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger)

            with open(dino_path, 'rb') as f:
                self.dino_engine = runtime.deserialize_cuda_engine(f.read())
            self.dino_context = self.dino_engine.create_execution_context()

            with open(depth_path, 'rb') as f:
                self.depth_engine = runtime.deserialize_cuda_engine(f.read())
            self.depth_context = self.depth_engine.create_execution_context()

            # Pre-allocate GPU buffers for TRT inference
            import torch
            self.device = torch.device('cuda')
            self.trt_dino_input = torch.zeros(1, 3, self.input_size, self.input_size,
                                              dtype=torch.float32, device=self.device)
            self.trt_depth_input = torch.zeros(1, 3, self.depth_size, self.depth_size,
                                               dtype=torch.float32, device=self.device)
            # Output shapes: DINOv2 patch tokens [1, N_patches, 384]
            n_patches = (self.input_size // 14) ** 2  # 37*37 = 1369
            self.trt_dino_output = torch.zeros(1, n_patches, 384,
                                               dtype=torch.float32, device=self.device)
            # Depth output [1, 1, H, W] — depth model outputs at input resolution
            self.trt_depth_output = torch.zeros(1, 1, self.depth_size, self.depth_size,
                                                dtype=torch.float32, device=self.device)

            self.get_logger().info('TensorRT engines loaded')
            self.use_trt = True
        except Exception as e:
            self.get_logger().warn(f'TensorRT load failed: {e}, falling back to PyTorch')
            self.use_trt = False
            self._load_pytorch()

    def _trt_infer_dino(self, input_tensor):

        self.trt_dino_input.copy_(input_tensor)
        self.dino_context.set_tensor_address('input', self.trt_dino_input.data_ptr())
        self.dino_context.set_tensor_address('patch_tokens', self.trt_dino_output.data_ptr())
        self.dino_context.execute_async_v3(0)
        import torch
        torch.cuda.synchronize()
        return self.trt_dino_output  # [1, N, 384]

    def _trt_infer_depth(self, input_tensor):

        self.trt_depth_input.copy_(input_tensor)
        self.depth_context.set_tensor_address('input', self.trt_depth_input.data_ptr())
        self.depth_context.set_tensor_address('depth', self.trt_depth_output.data_ptr())
        self.depth_context.execute_async_v3(0)
        import torch
        torch.cuda.synchronize()
        return self.trt_depth_output  # [1, 1, H, W]

    def _preprocess(self, frame, size):

        import torch
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size))
        img = img.astype(np.float32) / 255.0
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        tensor = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        return tensor

    def image_cb(self, msg):
        if self.dino_model is None and not self.use_trt:
            return

        import torch

        # Convert CompressedImage to numpy
        frame = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)

        with torch.no_grad():
            # DINOv2 features
            dino_input = self._preprocess(frame, self.input_size)
            if self.use_trt:
                patch_tokens = self._trt_infer_dino(dino_input)
            else:
                features = self.dino_model.forward_features(dino_input)
                patch_tokens = features['x_norm_patchtokens']  # [1, N, 384]

            # Publish features
            feat_msg = Float32MultiArray()
            feat_np = patch_tokens.cpu().numpy().squeeze(0)  # [N, 384]
            feat_flat = np.ascontiguousarray(feat_np, dtype=np.float32).ravel()
            feat_msg.layout.dim = [
                MultiArrayDimension(label='patches', size=feat_np.shape[0], stride=feat_np.shape[0] * feat_np.shape[1]),
                MultiArrayDimension(label='features', size=feat_np.shape[1], stride=feat_np.shape[1]),
            ]
            feat_msg.data = array.array('f', feat_flat.tobytes())
            self.feat_pub.publish(feat_msg)

            # Depth estimation
            depth_input = self._preprocess(frame, self.depth_size)
            if self.use_trt:
                depth = self._trt_infer_depth(depth_input)
            elif self.depth_model is not None:
                depth = self.depth_model(depth_input)  # [1, 1, H, W]
            else:
                depth = None

            if depth is not None:
                depth_np = depth.cpu().numpy().squeeze()  # [H, W]
                depth_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX)
                depth_u8 = depth_norm.astype(np.uint8)

                depth_msg = Image()
                depth_msg.header = msg.header
                depth_msg.height, depth_msg.width = depth_u8.shape
                depth_msg.encoding = 'mono8'
                depth_msg.step = depth_u8.shape[1]
                depth_msg.data = depth_u8.tobytes()
                self.depth_pub.publish(depth_msg)

        # FPS logging
        self.frame_count += 1
        now = time.time()
        if now - self.last_log_time >= 5.0:
            fps = self.frame_count / (now - self.last_log_time)
            self.get_logger().info(f'Perception: {fps:.1f} FPS')
            self.frame_count = 0
            self.last_log_time = now


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
