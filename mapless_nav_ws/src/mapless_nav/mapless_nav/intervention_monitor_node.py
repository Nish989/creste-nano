import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Bool, Float32MultiArray, String
import numpy as np
import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import threading


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


class InterventionMonitorNode(Node):
    def __init__(self):
        super().__init__('intervention_monitor_node')

        self.declare_parameter('model_dir', os.path.expanduser('~/models/reward_model'))
        self.declare_parameter('log_dir', os.path.expanduser('~/mapless_nav_data/interventions'))
        self.declare_parameter('online_lr', 1e-4)
        self.declare_parameter('update_batch_size', 16)
        self.declare_parameter('buffer_size', 500)
        self.declare_parameter('update_every_n', 10)
        self.declare_parameter('feature_dim', 384)
        self.declare_parameter('n_steps', 8)

        self.model_dir = self.get_parameter('model_dir').value
        self.log_dir = self.get_parameter('log_dir').value
        self.lr = self.get_parameter('online_lr').value
        self.batch_size = self.get_parameter('update_batch_size').value
        self.buffer_size = self.get_parameter('buffer_size').value
        self.update_every = self.get_parameter('update_every_n').value
        self.feat_dim = self.get_parameter('feature_dim').value
        self.n_steps = self.get_parameter('n_steps').value

        os.makedirs(self.log_dir, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.opt = None
        self._load_model()

        self.buf_feats = []
        self.buf_labels = []

        self.autonomous = False
        self.latest_bev = None
        self.latest_candidates = None
        self.last_teleop = 0.0
        self.last_auto = 0.0
        self.teleop_active = False

        self.n_interventions = 0
        self.auto_meters = 0.0
        self.n_updates = 0
        self.since_update = 0
        self.t0 = time.time()

        self.create_subscription(Bool, '/autonomous_mode', self.mode_cb, 10)
        self.create_subscription(Float64, '/cmd_steering', self.teleop_cb, 10)
        self.create_subscription(Float64, '/safe_cmd_steering', self.auto_cb, 10)
        self.create_subscription(Float32MultiArray, '/bev/features', self.bev_cb, 5)
        self.create_subscription(Float32MultiArray, '/planner/candidates', self.cands_cb, 5)
        self.create_subscription(Float64, '/gps/speed', self.speed_cb, 10)

        self.intervention_pub = self.create_publisher(Bool, '/intervention', 10)
        self.stats_pub = self.create_publisher(String, '/intervention_stats', 10)

        self.create_timer(5.0, self.log_stats)
        self.get_logger().info('intervention monitor running')

    def _load_model(self):
        path = os.path.join(self.model_dir, 'reward_mlp.pth')
        if not os.path.exists(path):
            self.get_logger().warn('no model found, online learning disabled')
            return
        try:
            self.model = RewardModel(self.n_steps * self.feat_dim).to(self.device)
            self.model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            self.model.train()
            self.opt = optim.Adam(self.model.parameters(), lr=self.lr)
            self.get_logger().info('model loaded for online updates')
        except Exception as e:
            self.get_logger().error(f'model load failed: {e}')

    def mode_cb(self, msg):
        self.autonomous = msg.data

    def teleop_cb(self, msg):
        self.last_teleop = msg.data

    def auto_cb(self, msg):
        self.last_auto = msg.data
        if not self.autonomous:
            return
        if abs(self.last_teleop - self.last_auto) > 0.15 and not self.teleop_active:
            self.teleop_active = True
            self._on_intervention()
        elif abs(self.last_teleop - self.last_auto) <= 0.15:
            self.teleop_active = False

    def bev_cb(self, msg):
        self.latest_bev = np.array(msg.data, dtype=np.float32).reshape(64, 64, self.feat_dim)

    def cands_cb(self, msg):
        if msg.layout.dim:
            nc = msg.layout.dim[0].size
            ns = msg.layout.dim[1].size
            self.latest_candidates = np.array(msg.data).reshape(nc, ns, 2)

    def speed_cb(self, msg):
        if self.autonomous:
            self.auto_meters += msg.data * 0.1

    def _on_intervention(self):
        self.n_interventions += 1
        self.since_update += 1
        self.get_logger().warn(f'intervention #{self.n_interventions}')
        self.intervention_pub.publish(Bool(data=True))

        if self.latest_bev is not None and self.latest_candidates is not None:
            ci = len(self.latest_candidates) // 2
            feat = self._get_feat(self.latest_bev, self.latest_candidates[ci])
            if feat is not None:
                self._add(feat, 0.0)

        with open(os.path.join(self.log_dir, 'interventions.jsonl'), 'a') as f:
            f.write(json.dumps({
                'time': time.time(),
                'n': self.n_interventions,
                'meters': self.auto_meters,
            }) + '\n')

        if self.since_update >= self.update_every:
            threading.Thread(target=self._update, daemon=True).start()
            self.since_update = 0

    def _get_feat(self, bev, cand):
        ns = min(self.n_steps, len(cand))
        by = np.clip(cand[:ns, 0].astype(int), 0, 63)
        bx = np.clip(cand[:ns, 1].astype(int), 0, 63)
        v = bev[by, bx].flatten().astype(np.float32)
        return None if np.abs(v).sum() < 1e-6 else v

    def _add(self, feat, label):
        self.buf_feats.append(feat)
        self.buf_labels.append(label)
        if len(self.buf_feats) > self.buffer_size:
            self.buf_feats.pop(0)
            self.buf_labels.pop(0)

    def _update(self):
        if self.model is None or len(self.buf_feats) < self.batch_size:
            return
        idx = np.random.choice(len(self.buf_feats), self.batch_size, replace=False)
        x = torch.from_numpy(np.stack([self.buf_feats[i] for i in idx])).to(self.device)
        y = torch.tensor([self.buf_labels[i] for i in idx], dtype=torch.float32).to(self.device)
        pred = self.model(x).squeeze(-1)
        loss = F.binary_cross_entropy(pred, y)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        self.n_updates += 1
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, 'reward_mlp.pth'))
        self.get_logger().info(f'online update #{self.n_updates} loss={loss.item():.4f}')

    def log_stats(self):
        nir = (self.n_interventions / max(self.auto_meters, 1)) * 100
        stats = {
            'interventions': self.n_interventions,
            'meters': round(self.auto_meters, 1),
            'nir': round(nir, 3),
            'updates': self.n_updates,
        }
        self.stats_pub.publish(String(data=json.dumps(stats)))
        self.get_logger().info(f'NIR={nir:.3f}/100m | dist={self.auto_meters:.0f}m | interventions={self.n_interventions}')


def main(args=None):
    rclpy.init(args=args)
    node = InterventionMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
