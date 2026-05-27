import signal
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Bool
import serial
import time


class PWMControlNode(Node):
    def __init__(self):
        super().__init__('pwm_control_node')

        # Parameters
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('serial_baud', 500000)

        port = self.get_parameter('serial_port').value
        baud = self.get_parameter('serial_baud').value

        # Track last sent values to avoid flooding serial
        self._last_steer = 1500
        self._last_thr = 1500

        # Open serial to ESP8266
        try:
            self.ser = serial.Serial(
                port, baud, timeout=0.1,
                write_timeout=0,  # non-blocking writes
            )
            time.sleep(2.0)  # ESP8266 resets on serial connect, wait for boot
            self.ser.reset_input_buffer()
            self.ser.write(b'P\n')
            self.ser.flush()
            time.sleep(0.1)
            resp = self.ser.readline().decode().strip()
            if resp == 'OK':
                self.get_logger().info(f'ESP8266 connected on {port}')
            else:
                self.get_logger().warn(f'ESP8266 on {port} - unexpected response: {resp!r}')
            self.ser.write(b'N\n')
            self.ser.flush()
            self.get_logger().info('--- ESP8266 PWM BRIDGE LIVE ---')
        except Exception as e:
            self.get_logger().error(f'Serial init error: {e}')
            self.ser = None

        self.armed = False
        self._arm_timer = self.create_timer(3.0, self._finish_arming)
        self.create_subscription(Float64, '/safe_cmd_steering', self.steering_cb, 10)
        self.create_subscription(Float64, '/safe_cmd_throttle', self.throttle_cb, 10)
        self.create_subscription(Bool, '/recording', self.recording_cb, 10)

    def _finish_arming(self):
        self.armed = True
        self.get_logger().info('--- ESC ARMED: READY FOR DATA COLLECTION ---')
        self._arm_timer.cancel()

    def _send(self, cmd):
        if self.ser and self.ser.is_open:
            self.ser.write(cmd)
            self.ser.flush()

    def steering_cb(self, msg):
        val = int(1500 + (msg.data * 500))
        val = max(1000, min(2000, val))
        if val != self._last_steer:
            self._last_steer = val
            self._send(f'S{val}\n'.encode())

    def throttle_cb(self, msg):
        if not self.armed:
            return
        limit = max(-0.12, min(0.12, msg.data))
        val = int(1500 + (limit * 500))
        val = max(1000, min(2000, val))
        if val != self._last_thr:
            self._last_thr = val
            self._send(f'T{val}\n'.encode())

    def recording_cb(self, msg):
        self._send(b'L1\n' if msg.data else b'L0\n')

    def stop(self):
        self._send(b'L0\n')
        self._send(b'N\n')

    def destroy_node(self):
        self.stop()
        if self.ser and self.ser.is_open:
            self.ser.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PWMControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
