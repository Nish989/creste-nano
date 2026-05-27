"""
MPPI Planner Node - Model Predictive Path Integral controller.
Replaces greedy 21-candidate planner with probabilistic 1000-trajectory optimizer.

Mathematical framework:
  - Sample K trajectories with Gaussian noise around nominal control
  - Score each trajectory over T steps using reward model
  - Compute softmax weights: w_k = exp(S(tau_k) / lambda)
  - Update nominal control: u* = sum(w_k * u_k)
  - Execute first action, shift horizon, repeat

Input:  /bev/features, /reward/scores, /waypoint/bearing, /autonomous_mode
Output: /cmd_steering, /cmd_throttle, /planner/candidates
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Bool, Float32MultiArray, MultiArrayDimension
import numpy as np
import math


class MPPIPlannerNode(Node):
    def __init__(self):
        super().__init__('planner_node')

        # MPPI parameters
        self.declare_parameter('n_samples', 1000)       # K trajectories
        self.declare_parameter('n_steps', 8)            # T steps per trajectory
        self.declare_parameter('lambda_temp', 0.1)      # temperature for softmax
        self.declare_parameter('sigma_noise', 0.3)      # Gaussian noise std
        self.declare_parameter('bev_width', 64)
        self.declare_parameter('bev_height', 64)
        self.declare_parameter('max_steering', 1.0)
        self.declare_parameter('auto_throttle', 0.2)
        self.declare_parameter('waypoint_bias', 0.3)
        self.declare_parameter('plan_rate_hz', 10.0)
        self.declare_parameter('momentum', 0.8)         # nominal control momentum

        self.K = self.get_parameter('n_samples').value
        self.T = self.get_parameter('n_steps').value
        self.lam = self.get_parameter('lambda_temp').value
        self.sigma = self.get_parameter('sigma_noise').value
        self.bev_w = self.get_parameter('bev_width').value
        self.bev_h = self.get_parameter('bev_height').value
        self.max_steer = self.get_parameter('max_steering').value
        self.auto_throttle = self.get_parameter('auto_throttle').value
        self.wp_bias = self.get_parameter('waypoint_bias').value
        self.momentum = self.get_parameter('momentum').value
        rate = self.get_parameter('plan_rate_hz').value

        # State
        self.latest_scores = None
        self.waypoint_bearing = None
        self.wp_distance = float('inf')
        self.autonomous = False

        # MPPI nominal control sequence (warm-started between iterations)
        # Shape: [T] — one steering value per step
        self.nominal_U = np.zeros(self.T)

        # Current sampled trajectories (published for reward node to score)
        self.current_epsilons = None  # noise samples [K, T]

        # Subscribers
        self.create_subscription(Float32MultiArray, '/reward/scores', self.scores_cb, 5)
        self.create_subscription(Float64, '/waypoint/bearing', self.bearing_cb, 10)
        self.create_subscription(Float64, '/waypoint/distance', self.distance_cb, 10)
        self.create_subscription(Bool, '/autonomous_mode', self.mode_cb, 10)

        # Publishers
        self.steer_pub = self.create_publisher(Float64, '/cmd_steering', 10)
        self.thr_pub = self.create_publisher(Float64, '/cmd_throttle', 10)
        self.candidates_pub = self.create_publisher(Float32MultiArray, '/planner/candidates', 5)

        self.create_timer(1.0 / rate, self.plan)

        self.get_logger().info(
            f'MPPI Planner: K={self.K} samples, T={self.T} steps, '
            f'lambda={self.lam}, sigma={self.sigma}')

    def scores_cb(self, msg):
        if len(msg.data) == self.K:
            self.latest_scores = np.array(msg.data)

    def bearing_cb(self, msg):
        self.waypoint_bearing = msg.data

    def distance_cb(self, msg):
        self.wp_distance = msg.data

    def mode_cb(self, msg):
        self.autonomous = msg.data
        if self.autonomous:
            self.get_logger().info('MPPI Planner ACTIVE')
            self.nominal_U = np.zeros(self.T)  # reset on mode change
        else:
            self.get_logger().info('MPPI Planner STANDBY')

    def _sample_trajectories(self):
        """
        Sample K trajectories by adding Gaussian noise to nominal control.

        epsilon_k ~ N(0, sigma^2)
        u_k,t = clip(nominal_U[t] + epsilon_k,t, -max_steer, max_steer)

        Returns:
            U_samples: [K, T] steering sequences
            epsilons:  [K, T] noise samples (needed for MPPI update)
            candidates: [K, T, 2] BEV (y, x) cell sequences
        """
        epsilons = np.random.normal(0, self.sigma, size=(self.K, self.T))
        U_samples = np.clip(
            self.nominal_U[np.newaxis, :] + epsilons,
            -self.max_steer, self.max_steer
        )  # [K, T]

        # Roll out trajectories in BEV space
        candidates = np.zeros((self.K, self.T, 2))
        center_x = float(self.bev_w / 2)
        start_y = float(self.bev_h - 1)

        x = np.full(self.K, center_x)
        y = np.full(self.K, start_y)

        for t in range(self.T):
            y = y - 1.5                          # move forward
            x = x + U_samples[:, t] * 2.0       # lateral from steering
            candidates[:, t, 0] = y              # BEV row
            candidates[:, t, 1] = x              # BEV col

        return U_samples, epsilons, candidates

    def _mppi_update(self, scores, epsilons):
        """
        MPPI weighted update of nominal control.

        Weights: w_k = exp(S(tau_k) / lambda) / Z
        Update:  nominal_U += sum_k(w_k * epsilon_k)

        Args:
            scores:   [K] trajectory scores from reward model
            epsilons: [K, T] noise samples

        Returns:
            optimal first steering action (float)
        """
        # Add GPS waypoint bias to scores
        if self.waypoint_bearing is not None:
            bearing_norm = np.clip(self.waypoint_bearing / 90.0, -1.0, 1.0)
            # Estimate steering of each trajectory (mean of first step perturbations)
            first_steers = self.nominal_U[0] + epsilons[:, 0]
            alignment = 1.0 - np.abs(first_steers - bearing_norm)
            scores = scores + self.wp_bias * alignment

        # Compute softmax weights (numerically stable)
        score_shifted = scores - scores.max()
        weights = np.exp(score_shifted / self.lam)
        weights = weights / (weights.sum() + 1e-8)  # normalize

        # Weighted update of nominal control
        weighted_noise = np.einsum('k,kt->t', weights, epsilons)  # [T]
        self.nominal_U = np.clip(
            self.momentum * self.nominal_U + weighted_noise,
            -self.max_steer, self.max_steer
        )

        # Shift nominal control forward (receding horizon)
        optimal_action = self.nominal_U[0]
        self.nominal_U = np.roll(self.nominal_U, -1)
        self.nominal_U[-1] = 0.0  # zero-pad end

        return float(optimal_action)

    def plan(self):
        # Always sample and publish candidates for reward node to score
        U_samples, epsilons, candidates = self._sample_trajectories()
        self.current_epsilons = epsilons

        # Publish candidates [K, T, 2]
        cand_msg = Float32MultiArray()
        cand_msg.layout.dim = [
            MultiArrayDimension(label='candidates', size=self.K,
                                stride=self.K * self.T * 2),
            MultiArrayDimension(label='steps', size=self.T,
                                stride=self.T * 2),
            MultiArrayDimension(label='coords', size=2, stride=2),
        ]
        cand_msg.data = candidates.flatten().tolist()
        self.candidates_pub.publish(cand_msg)

        if not self.autonomous:
            return

        if self.latest_scores is None or len(self.latest_scores) != self.K:
            return

        # MPPI update — get optimal steering
        best_steering = self._mppi_update(
            self.latest_scores.copy(), self.current_epsilons)

        # Throttle: reduce for sharp turns
        throttle = self.auto_throttle * max(0.5, 1.0 - abs(best_steering) * 0.5)

        self.steer_pub.publish(Float64(data=best_steering))
        self.thr_pub.publish(Float64(data=float(throttle)))


def main(args=None):
    rclpy.init(args=args)
    node = MPPIPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
