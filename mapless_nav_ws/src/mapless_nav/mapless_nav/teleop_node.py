import os
import signal
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Bool
import evdev
from evdev import ecodes


class TeleopNode(Node):
    def __init__(self):
        super().__init__('teleop_node')

        self.declare_parameter('deadzone', 0.12)
        self.declare_parameter('steering_smoothing', 0.4)  # 0=no smoothing, 1=max smoothing
        self.deadzone = self.get_parameter('deadzone').value
        self.smoothing = self.get_parameter('steering_smoothing').value

        self.steer_pub = self.create_publisher(Float64, '/cmd_steering', 10)
        self.thr_pub = self.create_publisher(Float64, '/cmd_throttle', 10)
        self.estop_pub = self.create_publisher(Bool, '/estop', 10)
        self.record_pub = self.create_publisher(Bool, '/record_toggle', 10)
        self.auto_mode_pub = self.create_publisher(Bool, '/autonomous_mode', 10)

        # Dashboard can toggle autonomous mode without PS5 controller
        self.create_subscription(Bool, '/set_autonomous', self._set_auto_cb, 10)

        self.estop_active = False
        self.autonomous_mode = False
        self._estop_btn_prev = False
        self._record_btn_prev = False
        self._auto_btn_prev = False
        self._x_btn_prev = False
        self._x_last_press = 0.0
        self._options_held_since = None
        self.steering = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.device = None
        self.abs_info = {}
        self._last_event_time = time.monotonic()

        self._try_connect()
        if self.device is None:
            self.get_logger().info('No gamepad found, will retry every 2s...')
            self._retry_timer = self.create_timer(2.0, self._try_connect)

        self.create_timer(0.02, self._publish)

    def _set_auto_cb(self, msg):
        if msg.data != self.autonomous_mode:
            self.autonomous_mode = msg.data
            self.auto_mode_pub.publish(Bool(data=self.autonomous_mode))
            if self.autonomous_mode:
                self.get_logger().warn('AUTONOMOUS MODE ON (dashboard)')
            else:
                self.get_logger().info('Manual mode (dashboard)')

    def _try_connect(self):
        for path in evdev.list_devices():
            dev = evdev.InputDevice(path)
            if 'DualSense' in dev.name or 'Wireless Controller' in dev.name:
                if 'Touchpad' in dev.name or 'Motion' in dev.name:
                    continue
                caps = dev.capabilities(verbose=False)
                if ecodes.EV_KEY in caps and ecodes.EV_ABS in caps:
                    self.device = dev
                    for code, absinfo in dev.capabilities().get(ecodes.EV_ABS, []):
                        self.abs_info[code] = absinfo
                    self._last_event_time = time.monotonic()
                    self.get_logger().info(f'Gamepad connected: {dev.name} ({dev.path})')
                    if hasattr(self, '_retry_timer'):
                        self._retry_timer.cancel()
                    return

    def _normalize(self, code, value):
        info = self.abs_info.get(code)
        if info is None:
            return 0.0
        lo, hi = info.min, info.max
        if hi == lo:
            return 0.0
        return 2.0 * (value - lo) / (hi - lo) - 1.0

    def _drain_events(self):
        if self.device is None:
            return
        try:
            while True:
                event = self.device.read_one()
                if event is None:
                    break
                if event.type == ecodes.EV_ABS:
                    self._last_event_time = time.monotonic()
                    val = self._normalize(event.code, event.value)
                    if event.code == ecodes.ABS_X:
                        self.steering = val
                    elif event.code == ecodes.ABS_RZ:
                        self.throttle = (val + 1.0) / 2.0
                    elif event.code == ecodes.ABS_Z:
                        self.brake = (val + 1.0) / 2.0
                elif event.type == ecodes.EV_KEY:
                    self._last_event_time = time.monotonic()
                    if event.code == ecodes.BTN_WEST:
                        btn_now = bool(event.value)
                        if btn_now and not self._estop_btn_prev:
                            self.estop_active = not self.estop_active
                            self.estop_pub.publish(Bool(data=self.estop_active))
                            if self.estop_active:
                                self.get_logger().warn('E-STOP ACTIVATED')
                            else:
                                self.get_logger().info('E-stop released')
                        self._estop_btn_prev = btn_now
                    elif event.code == ecodes.BTN_NORTH:
                        btn_now = bool(event.value)
                        if btn_now and not self._record_btn_prev:
                            self.record_pub.publish(Bool(data=True))
                            self.get_logger().info('Record toggle pressed')
                        self._record_btn_prev = btn_now
                    elif event.code == ecodes.BTN_EAST:
                        btn_now = bool(event.value)
                        if btn_now and not self._auto_btn_prev:
                            self.autonomous_mode = not self.autonomous_mode
                            self.auto_mode_pub.publish(Bool(data=self.autonomous_mode))
                            if self.autonomous_mode:
                                self.get_logger().warn('AUTONOMOUS MODE ON — Circle to take back control')
                            else:
                                self.get_logger().info('Manual mode — you have control')
                        self._auto_btn_prev = btn_now
                    elif event.code == ecodes.BTN_SOUTH:
                        btn_now = bool(event.value)
                        if btn_now and not self._x_btn_prev:
                            now = time.monotonic()
                            if now - self._x_last_press < 0.5:
                                self.get_logger().info('X double-tap — shutting down!')
                                os.kill(0, signal.SIGINT)
                                return
                            self._x_last_press = now
                        self._x_btn_prev = btn_now
                    elif event.code == ecodes.BTN_START:
                        if event.value:
                            self._options_held_since = time.monotonic()
                        else:
                            self._options_held_since = None
        except OSError:
            self.get_logger().warn('Gamepad disconnected')
            self.device = None
            self._retry_timer = self.create_timer(2.0, self._try_connect)

    def _publish(self):
        self._drain_events()

        if self.device is None:
            # No PS5. Autonomous mode is still driven by /set_autonomous.
            # In estop, broadcast zero so the watchdog doesn't keep stale values.
            if self.estop_active:
                self.steer_pub.publish(Float64(data=0.0))
                self.thr_pub.publish(Float64(data=0.0))
            return

        if self._options_held_since is not None:
            if time.monotonic() - self._options_held_since >= 3.0:
                self.get_logger().info('OPTIONS held 3s — shutting down!')
                os.kill(0, signal.SIGINT)
                return

        if self.estop_active:
            self.steer_pub.publish(Float64(data=0.0))
            self.thr_pub.publish(Float64(data=0.0))
            return

        steer = self.steering
        if abs(steer) < self.deadzone:
            steer = 0.0
        # Smooth steering to reduce jitter
        self._smooth_steer = getattr(self, '_smooth_steer', 0.0)
        self._smooth_steer = self.smoothing * self._smooth_steer + (1 - self.smoothing) * steer
        steer = self._smooth_steer

        thr = self.throttle if self.throttle >= self.deadzone else 0.0
        brk = self.brake if self.brake >= self.deadzone else 0.0

        if thr > brk:
            net_throttle = thr
        elif brk > thr:
            net_throttle = -brk
        else:
            net_throttle = 0.0

        if self.autonomous_mode:
            if abs(net_throttle) > 0.5:
                self.autonomous_mode = False
                self.auto_mode_pub.publish(Bool(data=False))
                self.get_logger().warn('Joystick override — back to MANUAL')
            else:
                return

        self.steer_pub.publish(Float64(data=-steer))
        self.thr_pub.publish(Float64(data=net_throttle))

    def destroy_node(self):
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
