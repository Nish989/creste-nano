"""
Reward Node - Scores trajectory candidates using contrastive reward model.
Loads the ContrastiveRewardModel trained with InfoNCE loss.

Input:  /bev/features, /planner/candidates
Output: /reward/scores
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model definition (must match train_reward.py)
# ---------------------------------------------------------------------------

class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, embed_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return F.normalize(self.encoder(x), dim=-1)


class RewardMLP(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class ContrastiveRewardModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, embed_dim=128):
        super().__init__()
        self.encoder = TrajectoryEncoder(input_dim, hidden_dim, embed_dim)
        self.reward_head = RewardMLP(embed_dim)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.reward_head(self.encoder(x))


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

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

        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()

        self.latest_bev = None

        self.create_subscription(Float32MultiArray, '/bev/features', self.bev_cb, 5)
        self.create_subscription(Float32MultiArray, '/planner/candidates', self.candidates_cb, 5)

        self.scores_pub = self.create_publisher(Float32MultiArray, '/reward/scores', 5)

        self.get_logger().info(f'Reward node started (device: {self.device})')

    def _load_model(self):
        model_path = os.path.join(self.model_dir, 'reward_mlp.pth')
        if not os.path.exists(model_path):
            self.get_logger().warn(
                f'Reward model not found at {model_path}. '
                'Train with train_reward.py first.')
            return

        try:
            input_dim = 8 * self.feat_dim
            self.model = ContrastiveRewardModel(input_dim=input_dim)
            state = torch.load(model_path, map_location='cpu', weights_only=True)
            self.model.load_state_dict(state)
            self.model.eval().to(self.device)
            self.get_logger().info('Contrastive reward model loaded')
        except Exception as e:
            self.get_logger().error(f'Failed to load reward model: {e}')

    def bev_cb(self, msg):
        self.latest_bev = np.array(msg.data).reshape(
            self.bev_h, self.bev_w, self.feat_dim)

    def candidates_cb(self, msg):
        if self.latest_bev is None or self.model is None:
            return

        data = np.array(msg.data)
        n_candidates = msg.layout.dim[0].size
        n_steps = msg.layout.dim[1].size
        candidates = data.reshape(n_candidates, n_steps, 2).astype(int)

        # Clip coordinates
        by = np.clip(candidates[:, :, 0], 0, self.bev_h - 1)
        bx = np.clip(candidates[:, :, 1], 0, self.bev_w - 1)

        # Gather BEV features: [n_candidates, n_steps, feat_dim]
        all_feats = self.latest_bev[by, bx]
        batch = all_feats.reshape(n_candidates, -1).astype(np.float32)

        tensor = torch.from_numpy(batch).to(self.device)

        with torch.no_grad():
            scores = self.model(tensor).squeeze(-1).cpu().numpy()

        scores_msg = Float32MultiArray()
        scores_msg.layout.dim = [
            MultiArrayDimension(label='scores', size=n_candidates,
                                stride=n_candidates),
        ]
        scores_msg.data = scores.tolist()
        self.scores_pub.publish(scores_msg)


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
