import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Bool


class SafetyNode(Node):
    def __init__(self):
        super().__init__('safety_node')

        self.declare_parameter('cmd_timeout_sec', 0.5)
        self.declare_parameter('max_speed', 0.3)

        self.timeout = self.get_parameter('cmd_timeout_sec').value
        self.max_speed = self.get_parameter('max_speed').value

        self.last_cmd_time = self.get_clock().now()
        self.estop = False
        self.last_steering = 0.0
        self.last_throttle = 0.0

        self.create_subscription(Float64, '/cmd_steering', self.steer_cb, 10)
        self.create_subscription(Float64, '/cmd_throttle', self.throttle_cb, 10)
        self.create_subscription(Bool, '/estop', self.estop_cb, 10)

        self.safe_steer_pub = self.create_publisher(Float64, '/safe_cmd_steering', 10)
        self.safe_thr_pub = self.create_publisher(Float64, '/safe_cmd_throttle', 10)

        self.create_timer(0.02, self.watchdog)

        self.get_logger().info('Safety node started')

    def steer_cb(self, msg):
        self.last_steering = msg.data
        self.last_cmd_time = self.get_clock().now()

    def throttle_cb(self, msg):
        self.last_throttle = msg.data
        self.last_cmd_time = self.get_clock().now()

    def estop_cb(self, msg):
        self.estop = msg.data
        if self.estop:
            self.get_logger().warn('Safety: E-STOP received, stopping motors')

    def watchdog(self):
        elapsed = (self.get_clock().now() - self.last_cmd_time).nanoseconds / 1e9
        timed_out = elapsed > self.timeout

        if self.estop or timed_out:
            self.safe_steer_pub.publish(Float64(data=0.0))
            self.safe_thr_pub.publish(Float64(data=0.0))
            return

        safe_throttle = max(-self.max_speed, min(self.max_speed, self.last_throttle))
        self.safe_steer_pub.publish(Float64(data=self.last_steering))
        self.safe_thr_pub.publish(Float64(data=safe_throttle))


def main(args=None):
    rclpy.init(args=args)
    node = SafetyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
