import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


class SpeedControllerNode(Node):
    def __init__(self):
        super().__init__('speed_controller_node')

        # PID gains
        self.declare_parameter('kp', 0.15)
        self.declare_parameter('ki', 0.05)
        self.declare_parameter('kd', 0.02)

        # Speed limits
        self.declare_parameter('max_target_speed', 3.0)  # m/s
        self.declare_parameter('min_throttle', 0.04)  # floor to avoid cogging (~1570us)
        self.declare_parameter('max_throttle', 0.3)  # max output (matches safety limit)

        # Control rate
        self.declare_parameter('rate_hz', 10.0)

        self.kp = self.get_parameter('kp').value
        self.ki = self.get_parameter('ki').value
        self.kd = self.get_parameter('kd').value
        self.max_target = self.get_parameter('max_target_speed').value
        self.min_thr = self.get_parameter('min_throttle').value
        self.max_thr = self.get_parameter('max_throttle').value
        rate = self.get_parameter('rate_hz').value

        # PID state
        self.integral = 0.0
        self.prev_error = 0.0
        self.dt = 1.0 / rate

        # Input state
        self.target_speed = 0.0  # m/s
        self.actual_speed = 0.0  # m/s

        # Subscribers
        self.create_subscription(Float64, '/cmd_speed', self.target_cb, 10)
        self.create_subscription(Float64, '/gps/speed', self.speed_cb, 10)

        # Publisher
        self.throttle_pub = self.create_publisher(Float64, '/cmd_throttle', 10)

        # Control loop
        self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(
            f'Speed controller started: kp={self.kp} ki={self.ki} kd={self.kd}, '
            f'throttle range [{self.min_thr}, {self.max_thr}]')

    def target_cb(self, msg):
        self.target_speed = max(0.0, min(self.max_target, msg.data))

    def speed_cb(self, msg):
        self.actual_speed = max(0.0, msg.data)

    def control_loop(self):
        # If target is zero, stop and reset PID
        if self.target_speed < 0.01:
            self.throttle_pub.publish(Float64(data=0.0))
            self.integral = 0.0
            self.prev_error = 0.0
            return

        error = self.target_speed - self.actual_speed

        # PID
        self.integral += error * self.dt
        # Anti-windup: clamp integral
        max_integral = self.max_thr / max(self.ki, 0.001)
        self.integral = max(-max_integral, min(max_integral, self.integral))

        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Clamp output, enforce minimum throttle to prevent cogging
        if output > 0:
            output = max(self.min_throttle, min(self.max_thr, output))
        else:
            output = max(-self.max_thr, min(-self.min_thr, output))

        self.throttle_pub.publish(Float64(data=output))


def main(args=None):
    rclpy.init(args=args)
    node = SpeedControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
