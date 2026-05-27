"""
BEV Projection Node - Projects DINOv2 patch features into a bird's eye view
grid using depth estimates, following the CREStE paradigm.

Input: /perception/features (DINOv2 patches) + /perception/depth
Output: /bev/features (BEV feature grid)
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import numpy as np
import array as _array


class BEVProjectionNode(Node):
    def __init__(self):
        super().__init__('bev_projection_node')

        # BEV grid parameters
        self.declare_parameter('bev_width', 64)       # grid cells wide
        self.declare_parameter('bev_height', 64)      # grid cells forward
        self.declare_parameter('bev_resolution', 0.15) # meters per cell
        self.declare_parameter('camera_height', 0.35)  # camera height above ground (meters)
        self.declare_parameter('camera_pitch', 15.0)   # degrees below horizontal
        self.declare_parameter('camera_fov_h', 90.0)   # horizontal FOV degrees (EMEET Nova 4K)
        self.declare_parameter('camera_fov_v', 58.0)   # vertical FOV degrees (EMEET Nova 4K)
        self.declare_parameter('dino_patch_size', 14)   # DINOv2 patch size
        self.declare_parameter('dino_input_size', 518)

        self.bev_w = self.get_parameter('bev_width').value
        self.bev_h = self.get_parameter('bev_height').value
        self.bev_res = self.get_parameter('bev_resolution').value
        self.cam_height = self.get_parameter('camera_height').value
        self.cam_pitch = np.radians(self.get_parameter('camera_pitch').value)
        self.fov_h = np.radians(self.get_parameter('camera_fov_h').value)
        self.fov_v = np.radians(self.get_parameter('camera_fov_v').value)
        self.patch_size = self.get_parameter('dino_patch_size').value
        self.input_size = self.get_parameter('dino_input_size').value

        self.patches_per_side = self.input_size // self.patch_size  # 37 for 518/14
        self.feature_dim = 384  # DINOv2-small

        # Precompute patch-to-ray mapping
        self._precompute_patch_rays()

        # State
        self.latest_features = None
        self.latest_depth = None

        # Subscribers
        self.create_subscription(
            Float32MultiArray, '/perception/features', self.feat_cb, 5)
        self.create_subscription(
            Image, '/perception/depth', self.depth_cb, 5)

        # Publisher
        self.bev_pub = self.create_publisher(
            Float32MultiArray, '/bev/features', 5)

        # Process at 10Hz
        self.create_timer(0.1, self.project)

        self.get_logger().info(
            f'BEV projection: {self.bev_w}x{self.bev_h} grid, '
            f'{self.bev_res}m/cell, {self.bev_w * self.bev_res:.1f}m x '
            f'{self.bev_h * self.bev_res:.1f}m coverage')

    def _precompute_patch_rays(self):
        """Compute the ray direction for each DINOv2 patch center."""
        n = self.patches_per_side
        px = np.arange(n)
        py = np.arange(n)
        px_grid, py_grid = np.meshgrid(px, py)  # both [n, n]
        u = (px_grid.ravel() + 0.5) / n - 0.5
        v = (py_grid.ravel() + 0.5) / n - 0.5
        self.patch_angles_h = u * self.fov_h
        self.patch_angles_v = v * self.fov_v + self.cam_pitch

        # Precompute depth sampling coordinates (used in project())
        self.depth_py = py_grid.ravel()
        self.depth_px = px_grid.ravel()

    def feat_cb(self, msg):
        n_patches = self.patches_per_side ** 2
        if len(msg.data) == n_patches * self.feature_dim:
            self.latest_features = np.array(msg.data).reshape(n_patches, self.feature_dim)

    def depth_cb(self, msg):
        self.latest_depth = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width)

    def project(self):
        if self.latest_features is None or self.latest_depth is None:
            return

        n = self.patches_per_side
        depth_h, depth_w = self.latest_depth.shape

        # Sample depth at each patch center — vectorized
        dy = np.minimum(((self.depth_py + 0.5) / n * depth_h).astype(int), depth_h - 1)
        dx = np.minimum(((self.depth_px + 0.5) / n * depth_w).astype(int), depth_w - 1)
        depth_patches = self.latest_depth[dy, dx].astype(np.float32)

        max_depth_val = depth_patches.max()
        if max_depth_val < 1:
            return

        # Convert to meters and filter — vectorized
        depth_m = (depth_patches / max_depth_val) * 10.0
        valid = depth_m >= 0.3

        # 3D projection — vectorized
        x = depth_m * np.sin(self.patch_angles_h)
        z = depth_m * np.cos(self.patch_angles_h) * np.cos(self.patch_angles_v)

        bev_x = (x / self.bev_res + self.bev_w / 2).astype(int)
        bev_y = (self.bev_h - z / self.bev_res).astype(int)

        # Bounds check — vectorized
        valid &= (bev_x >= 0) & (bev_x < self.bev_w) & (bev_y >= 0) & (bev_y < self.bev_h)

        bev_x_v = bev_x[valid]
        bev_y_v = bev_y[valid]
        feats_v = self.latest_features[valid]

        # Scatter-add features into BEV grid using np.add.at
        bev = np.zeros((self.bev_h, self.bev_w, self.feature_dim), dtype=np.float32)
        counts = np.zeros((self.bev_h, self.bev_w), dtype=np.float32)
        np.add.at(bev, (bev_y_v, bev_x_v), feats_v)
        np.add.at(counts, (bev_y_v, bev_x_v), 1)

        # Average features in cells with multiple patches
        mask = counts > 0
        bev[mask] /= counts[mask, np.newaxis]

        # Publish BEV features (assign numpy array directly, avoid .tolist())
        bev_flat = np.ascontiguousarray(bev, dtype=np.float32).ravel()
        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label='height', size=self.bev_h,
                                stride=self.bev_h * self.bev_w * self.feature_dim),
            MultiArrayDimension(label='width', size=self.bev_w,
                                stride=self.bev_w * self.feature_dim),
            MultiArrayDimension(label='features', size=self.feature_dim,
                                stride=self.feature_dim),
        ]
        msg.data = _array.array('f', bev_flat.tobytes())
        self.bev_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = BEVProjectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
