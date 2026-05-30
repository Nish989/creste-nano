import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Bool, Float32MultiArray, MultiArrayDimension
import numpy as np


class MPPIPlannerNode(Node):
    def __init__(self):
        super().__init__('planner_node')

        self.declare_parameter('n_samples', 1000)
        self.declare_parameter('n_steps', 8)
        self.declare_parameter('lambda_temp', 0.1)
        self.declare_parameter('sigma_noise', 0.3)
        self.declare_parameter('bev_width', 64)
        self.declare_parameter('bev_height', 64)
        self.declare_parameter('max_steering', 1.0)
        self.declare_parameter('auto_throttle', 0.2)
        self.declare_parameter('waypoint_bias', 0.3)
        self.declare_parameter('plan_rate_hz', 10.0)
        self.declare_parameter('momentum', 0.8)

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

        self.latest_scores = None
        self.waypoint_bearing = None
        self.wp_distance = float('inf')
        self.autonomous = False
        self.nominal_U = np.zeros(self.T)
        self.current_epsilons = None
        self._scores_warmup_ticks = 0   # count ticks waiting for first scores

        self.create_subscription(Float32MultiArray, '/reward/scores', self.scores_cb, 5)
        self.create_subscription(Float64, '/waypoint/bearing', self.bearing_cb, 10)
        self.create_subscription(Float64, '/waypoint/distance', self.distance_cb, 10)
        self.create_subscription(Bool, '/autonomous_mode', self.mode_cb, 10)

        self.steer_pub = self.create_publisher(Float64, '/cmd_steering', 10)
        self.thr_pub = self.create_publisher(Float64, '/cmd_throttle', 10)
        self.candidates_pub = self.create_publisher(Float32MultiArray, '/planner/candidates', 5)

        self.create_timer(1.0 / rate, self.plan)
        self.get_logger().info(f'MPPI planner ready: K={self.K}, T={self.T}')

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
            self.nominal_U = np.zeros(self.T)
            self.get_logger().info('autonomous mode on')
        else:
            self.get_logger().info('standby')

    def _sample(self):
        epsilons = np.random.normal(0, self.sigma, size=(self.K, self.T))
        U = np.clip(self.nominal_U[np.newaxis, :] + epsilons, -self.max_steer, self.max_steer)

        candidates = np.zeros((self.K, self.T, 2))
        x = np.full(self.K, float(self.bev_w / 2))
        y = np.full(self.K, float(self.bev_h - 1))

        for t in range(self.T):
            y = y - 1.5
            x = x + U[:, t] * 2.0
            candidates[:, t, 0] = y
            candidates[:, t, 1] = x

        return U, epsilons, candidates

    def _update(self, scores, epsilons):
        if self.waypoint_bearing is not None:
            bn = np.clip(self.waypoint_bearing / 90.0, -1.0, 1.0)
            first_steers = self.nominal_U[0] + epsilons[:, 0]
            scores = scores + self.wp_bias * (1.0 - np.abs(first_steers - bn))

        s = scores - scores.max()
        w = np.exp(s / self.lam)
        w = w / (w.sum() + 1e-8)

        self.nominal_U = np.clip(
            self.momentum * self.nominal_U + np.einsum('k,kt->t', w, epsilons),
            -self.max_steer, self.max_steer
        )

        action = float(self.nominal_U[0])
        self.nominal_U = np.roll(self.nominal_U, -1)
        self.nominal_U[-1] = 0.0
        return action

    def plan(self):
        _, epsilons, candidates = self._sample()
        self.current_epsilons = epsilons

        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label='candidates', size=self.K, stride=self.K * self.T * 2),
            MultiArrayDimension(label='steps', size=self.T, stride=self.T * 2),
            MultiArrayDimension(label='coords', size=2, stride=2),
        ]
        msg.data = candidates.flatten().tolist()
        self.candidates_pub.publish(msg)

        if not self.autonomous:
            return

        if self.latest_scores is None or len(self.latest_scores) != self.K:
            # No scores yet. Drive straight at half throttle to keep the
            # safety watchdog fed while perception spins up.
            self._scores_warmup_ticks += 1
            if self._scores_warmup_ticks % 20 == 1:
                self.get_logger().warn(
                    f'Waiting for reward scores ({self._scores_warmup_ticks} ticks)...')
            self.steer_pub.publish(Float64(data=0.0))
            self.thr_pub.publish(Float64(data=self.auto_throttle * 0.5))
            return

        self._scores_warmup_ticks = 0
        steer = self._update(self.latest_scores.copy(), self.current_epsilons)
        throttle = self.auto_throttle * max(0.5, 1.0 - abs(steer) * 0.5)

        self.steer_pub.publish(Float64(data=steer))
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
