import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class RewardHead(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class RewardModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = TrajectoryEncoder(input_dim)
        self.head = RewardHead()

    def forward(self, x):
        return self.head(self.encoder(x))


class RewardNode(Node):
    def __init__(self):
        super().__init__('reward_node')

        self.declare_parameter('model_dir', os.path.expanduser('~/models/reward_model'))
        self.declare_parameter('bev_width', 64)
        self.declare_parameter('bev_height', 64)
        self.declare_parameter('feature_dim', 384)

        self.model_dir = self.get_parameter('model_dir').value
        self.bev_w = self.get_parameter('bev_width').value
        self.bev_h = self.get_parameter('bev_height').value
        self.feat_dim = self.get_parameter('feature_dim').value
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.latest_bev = None
        self._load_model()

        self.create_subscription(Float32MultiArray, '/bev/features', self.bev_cb, 5)
        self.create_subscription(Float32MultiArray, '/planner/candidates', self.candidates_cb, 5)
        self.scores_pub = self.create_publisher(Float32MultiArray, '/reward/scores', 5)

        self.get_logger().info(f'reward node ready ({self.device})')

    def _load_model(self):
        path = os.path.join(self.model_dir, 'reward_mlp.pth')
        if not os.path.exists(path):
            self.get_logger().warn(f'no model at {path}, train first')
            return
        try:
            self.model = RewardModel(8 * self.feat_dim)
            self.model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            self.model.eval().to(self.device)
            self.get_logger().info('reward model loaded')
        except Exception as e:
            self.get_logger().error(f'failed to load model: {e}')

    def bev_cb(self, msg):
        self.latest_bev = np.array(msg.data).reshape(self.bev_h, self.bev_w, self.feat_dim)

    def candidates_cb(self, msg):
        if self.latest_bev is None or self.model is None:
            return

        n_cand = msg.layout.dim[0].size
        n_steps = msg.layout.dim[1].size
        cands = np.array(msg.data).reshape(n_cand, n_steps, 2).astype(int)

        by = np.clip(cands[:, :, 0], 0, self.bev_h - 1)
        bx = np.clip(cands[:, :, 1], 0, self.bev_w - 1)
        batch = self.latest_bev[by, bx].reshape(n_cand, -1).astype(np.float32)

        with torch.no_grad():
            scores = self.model(torch.from_numpy(batch).to(self.device)).squeeze(-1).cpu().numpy()

        out = Float32MultiArray()
        out.layout.dim = [MultiArrayDimension(label='scores', size=n_cand, stride=n_cand)]
        out.data = scores.tolist()
        self.scores_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = RewardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
