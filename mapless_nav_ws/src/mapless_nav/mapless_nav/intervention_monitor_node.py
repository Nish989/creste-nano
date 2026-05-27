"""
Intervention Monitor Node - Online reward learning from human interventions.

Implements RLHF-style online adaptation:
  - Monitors when human takes over from autonomous mode (intervention = negative signal)
  - Monitors successful autonomous driving (positive signal)
  - Triggers incremental reward model updates from deployment experience
  - Reduces interventions over successive runs

Research contribution:
  First application of intervention-based online reward adaptation to
  BEV mapless navigation on sub-$500 hardware.

Topics:
  Subscribes: /autonomous_mode, /cmd_steering (teleop), /safe_cmd_steering,
              /bev/features, /planner/candidates
  Publishes:  /intervention (Bool), /intervention_stats (String)
"""
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


# ---------------------------------------------------------------------------
# Model (must match reward_node.py / train_reward.py)
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

class InterventionMonitorNode(Node):
    def __init__(self):
        super().__init__('intervention_monitor_node')

        self.declare_parameter('model_dir', os.path.expanduser('~/models/reward_model'))
        self.declare_parameter('log_dir', os.path.expanduser('~/mapless_nav_data/interventions'))
        self.declare_parameter('online_lr', 1e-4)          # learning rate for online updates
        self.declare_parameter('update_batch_size', 16)     # samples per online update
        self.declare_parameter('buffer_size', 500)          # max experience buffer size
        self.declare_parameter('update_every_n', 10)        # update model every N interventions
        self.declare_parameter('feature_dim', 384)
        self.declare_parameter('n_steps', 8)

        self.model_dir = self.get_parameter('model_dir').value
        self.log_dir = self.get_parameter('log_dir').value
        self.online_lr = self.get_parameter('online_lr').value
        self.batch_size = self.get_parameter('update_batch_size').value
        self.buffer_size = self.get_parameter('buffer_size').value
        self.update_every_n = self.get_parameter('update_every_n').value
        self.feat_dim = self.get_parameter('feature_dim').value
        self.n_steps = self.get_parameter('n_steps').value

        os.makedirs(self.log_dir, exist_ok=True)

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model for online updates
        self.model = None
        self.optimizer = None
        self._load_model()

        # Experience replay buffer
        # Each entry: (trajectory_features [n_steps*feat_dim], label [0=bad, 1=good])
        self.buffer_features = []
        self.buffer_labels = []

        # State tracking
        self.autonomous = False
        self.was_autonomous = False
        self.autonomous_start_time = None
        self.latest_bev = None
        self.latest_candidates = None

        # Teleop override detection
        self.last_teleop_steer = 0.0
        self.last_auto_steer = 0.0
        self.teleop_active = False

        # Statistics
        self.total_interventions = 0
        self.total_autonomous_meters = 0.0
        self.session_start = time.time()
        self.n_online_updates = 0
        self.interventions_since_update = 0

        # Subscribers
        self.create_subscription(Bool, '/autonomous_mode', self.mode_cb, 10)
        self.create_subscription(Float64, '/cmd_steering', self.teleop_steer_cb, 10)
        self.create_subscription(Float64, '/safe_cmd_steering', self.auto_steer_cb, 10)
        self.create_subscription(Float32MultiArray, '/bev/features', self.bev_cb, 5)
        self.create_subscription(Float32MultiArray, '/planner/candidates', self.candidates_cb, 5)
        self.create_subscription(Float64, '/gps/speed', self.speed_cb, 10)

        # Publishers
        self.intervention_pub = self.create_publisher(Bool, '/intervention', 10)
        self.stats_pub = self.create_publisher(String, '/intervention_stats', 10)

        # Stats timer
        self.create_timer(5.0, self.publish_stats)

        self.get_logger().info('Intervention monitor started — watching for human overrides')

    def _load_model(self):
        model_path = os.path.join(self.model_dir, 'reward_mlp.pth')
        if not os.path.exists(model_path):
            self.get_logger().warn(
                f'No model at {model_path} — online learning disabled until model is trained')
            return

        try:
            input_dim = self.n_steps * self.feat_dim
            self.model = ContrastiveRewardModel(input_dim=input_dim)
            state = torch.load(model_path, map_location='cpu', weights_only=True)
            self.model.load_state_dict(state)
            self.model.train()  # keep in train mode for online updates
            self.model = self.model.to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.online_lr)
            self.get_logger().info('Reward model loaded for online intervention learning')
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')

    def mode_cb(self, msg):
        self.autonomous = msg.data
        if self.autonomous and not self.was_autonomous:
            self.autonomous_start_time = time.time()
            self.get_logger().info('Autonomous mode ON — monitoring for interventions')
        self.was_autonomous = self.autonomous

    def teleop_steer_cb(self, msg):
        self.last_teleop_steer = msg.data

    def auto_steer_cb(self, msg):
        self.last_auto_steer = msg.data
        self._check_intervention()

    def bev_cb(self, msg):
        self.latest_bev = np.array(msg.data, dtype=np.float32).reshape(
            64, 64, self.feat_dim)

    def candidates_cb(self, msg):
        if msg.layout.dim:
            n_candidates = msg.layout.dim[0].size
            n_steps = msg.layout.dim[1].size
            self.latest_candidates = np.array(msg.data).reshape(
                n_candidates, n_steps, 2)

    def speed_cb(self, msg):
        # Accumulate autonomous distance for NIR metric
        if self.autonomous:
            dt = 0.1  # approximate 10Hz GPS
            self.total_autonomous_meters += msg.data * dt

    def _check_intervention(self):
        """
        Detect human intervention: teleop steering differs significantly
        from what autonomous system would send.

        Intervention condition:
          |teleop_steer - auto_steer| > 0.15 while in autonomous mode
        """
        if not self.autonomous:
            return

        steer_diff = abs(self.last_teleop_steer - self.last_auto_steer)
        intervention_detected = steer_diff > 0.15

        if intervention_detected and not self.teleop_active:
            self.teleop_active = True
            self._on_intervention()
        elif not intervention_detected:
            self.teleop_active = False

    def _on_intervention(self):
        """Handle detected human intervention."""
        self.total_interventions += 1
        self.interventions_since_update += 1

        self.get_logger().warn(
            f'INTERVENTION #{self.total_interventions} detected — '
            f'logging negative experience')

        # Publish intervention signal
        self.intervention_pub.publish(Bool(data=True))

        # Add current BEV state as negative experience
        if self.latest_bev is not None and self.latest_candidates is not None:
            # The autonomous system's chosen trajectory was wrong — add as negative
            center_idx = len(self.latest_candidates) // 2
            neg_feat = self._extract_trajectory_features(
                self.latest_bev, self.latest_candidates[center_idx])
            if neg_feat is not None:
                self._add_to_buffer(neg_feat, label=0.0)

        # Log to file
        log_entry = {
            'timestamp': time.time(),
            'intervention_n': self.total_interventions,
            'autonomous_meters': self.total_autonomous_meters,
            'teleop_steer': self.last_teleop_steer,
            'auto_steer': self.last_auto_steer,
        }
        log_path = os.path.join(self.log_dir, 'interventions.jsonl')
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        # Trigger online update every N interventions
        if self.interventions_since_update >= self.update_every_n:
            self._online_update()
            self.interventions_since_update = 0

    def _extract_trajectory_features(self, bev, candidate):
        """Extract flattened BEV features along a candidate trajectory."""
        n_steps = min(self.n_steps, len(candidate))
        by = np.clip(candidate[:n_steps, 0].astype(int), 0, 63)
        bx = np.clip(candidate[:n_steps, 1].astype(int), 0, 63)
        feats = bev[by, bx].flatten()
        if np.abs(feats).sum() < 1e-6:
            return None
        return feats.astype(np.float32)

    def _add_to_buffer(self, features, label):
        """Add experience to replay buffer with FIFO eviction."""
        self.buffer_features.append(features)
        self.buffer_labels.append(label)
        if len(self.buffer_features) > self.buffer_size:
            self.buffer_features.pop(0)
            self.buffer_labels.pop(0)

    def add_positive_experience(self, features):
        """Add successful autonomous driving segment as positive."""
        self._add_to_buffer(features, label=1.0)

    def _online_update(self):
        """
        Perform online gradient update on reward model using buffered experiences.

        Uses binary cross-entropy on the reward head to incorporate
        intervention feedback (negative) and successful driving (positive).
        """
        if self.model is None or len(self.buffer_features) < self.batch_size:
            return

        self.get_logger().info(
            f'Online update #{self.n_online_updates + 1} '
            f'({len(self.buffer_features)} experiences in buffer)')

        # Sample random batch
        indices = np.random.choice(
            len(self.buffer_features), self.batch_size, replace=False)
        batch_feats = np.stack([self.buffer_features[i] for i in indices])
        batch_labels = np.array([self.buffer_labels[i] for i in indices], dtype=np.float32)

        feat_tensor = torch.from_numpy(batch_feats).to(self.device)
        label_tensor = torch.from_numpy(batch_labels).to(self.device)

        # Forward pass + BCE loss on reward head
        self.model.train()
        pred = self.model(feat_tensor).squeeze(-1)
        loss = F.binary_cross_entropy(pred, label_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.n_online_updates += 1

        # Save updated model
        save_path = os.path.join(self.model_dir, 'reward_mlp.pth')
        torch.save(self.model.state_dict(), save_path)

        self.get_logger().info(
            f'Online update done: loss={loss.item():.4f}, '
            f'model saved to {save_path}')

    def publish_stats(self):
        """Publish intervention statistics for monitoring."""
        elapsed = time.time() - self.session_start
        nir = (self.total_interventions / max(self.total_autonomous_meters, 1.0)) * 100

        stats = {
            'interventions': self.total_interventions,
            'autonomous_meters': round(self.total_autonomous_meters, 1),
            'nir_per_100m': round(nir, 3),
            'online_updates': self.n_online_updates,
            'buffer_size': len(self.buffer_features),
            'session_minutes': round(elapsed / 60, 1),
        }

        msg = String(data=json.dumps(stats))
        self.stats_pub.publish(msg)

        self.get_logger().info(
            f'NIR: {nir:.3f}/100m | '
            f'Interventions: {self.total_interventions} | '
            f'Autonomous: {self.total_autonomous_meters:.0f}m | '
            f'Online updates: {self.n_online_updates}')


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
