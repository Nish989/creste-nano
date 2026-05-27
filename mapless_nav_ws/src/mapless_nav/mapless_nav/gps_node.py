"""
GPS Node - Reads UBX binary from HGLRC M100 Pro (u-blox) over UART.
Publishes NavSatFix to /gps/fix and speed/course to /gps/speed, /gps/course.
"""
import struct

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import Float64
import serial


class GPSNode(Node):
    def __init__(self):
        super().__init__('gps_node')

        self.declare_parameter('port', '/dev/ttyTHS1')
        self.declare_parameter('baud', 115200)
        self.declare_parameter('min_satellites', 10)

        port = self.get_parameter('port').value
        baud = self.get_parameter('baud').value
        self.min_sats = self.get_parameter('min_satellites').value

        try:
            self.ser = serial.Serial(port, baud, timeout=0.1)
            self.get_logger().info(f'GPS opened on {port} @ {baud} (UBX mode)')
        except serial.SerialException as e:
            self.get_logger().error(f'Failed to open GPS: {e}')
            self.ser = None
            return

        self.fix_pub = self.create_publisher(NavSatFix, '/gps/fix', 10)
        self.speed_pub = self.create_publisher(Float64, '/gps/speed', 10)
        self.course_pub = self.create_publisher(Float64, '/gps/course', 10)

        self.has_fix = False
        self.num_sats = 0
        self.buf = bytearray()

        # Poll at 50Hz to keep up with 10Hz GPS output
        self.create_timer(0.02, self.read_serial)

    def read_serial(self):
        if self.ser is None or not self.ser.is_open:
            return

        try:
            waiting = self.ser.in_waiting or 1
            raw = self.ser.read(waiting)
            if not raw:
                return
            self.buf.extend(raw)
        except serial.SerialException:
            return

        # Parse all complete UBX messages in the buffer
        while True:
            msg = self._parse_ubx()
            if msg is None:
                break
            cls, msg_id, payload = msg
            # NAV-PVT (0x01, 0x07)
            if cls == 0x01 and msg_id == 0x07:
                self._handle_nav_pvt(payload)

        # Prevent buffer from growing unbounded if no sync found
        if len(self.buf) > 2048:
            self.buf = self.buf[-512:]

    def _parse_ubx(self):
        """Extract one UBX message from self.buf. Returns (cls, id, payload) or None."""
        # Find sync header 0xB5 0x62
        while len(self.buf) >= 2:
            if self.buf[0] == 0xB5 and self.buf[1] == 0x62:
                break
            # Discard non-sync bytes
            del self.buf[0]

        # Need at least header (2) + class (1) + id (1) + length (2) = 6
        if len(self.buf) < 6:
            return None

        length = struct.unpack_from('<H', self.buf, 4)[0]
        total = 6 + length + 2  # header + payload + checksum

        if len(self.buf) < total:
            return None  # Incomplete message

        cls = self.buf[2]
        msg_id = self.buf[3]
        payload = bytes(self.buf[6:6 + length])

        # Verify checksum
        ck_a, ck_b = 0, 0
        for b in self.buf[2:6 + length]:
            ck_a = (ck_a + b) & 0xFF
            ck_b = (ck_b + ck_a) & 0xFF

        if ck_a != self.buf[6 + length] or ck_b != self.buf[6 + length + 1]:
            # Bad checksum - skip sync bytes and try again
            del self.buf[:2]
            return self._parse_ubx()

        # Consume the message
        del self.buf[:total]
        return cls, msg_id, payload

    def _handle_nav_pvt(self, payload):
        """Parse UBX-NAV-PVT and publish fix + velocity."""
        if len(payload) < 92:
            return

        fix_type = payload[20]
        num_sv = payload[23]
        lon = struct.unpack_from('<i', payload, 24)[0] * 1e-7  # degrees
        lat = struct.unpack_from('<i', payload, 28)[0] * 1e-7  # degrees
        h_msl = struct.unpack_from('<i', payload, 36)[0] / 1000.0  # mm -> m
        h_acc = struct.unpack_from('<I', payload, 40)[0] / 1000.0  # mm -> m
        v_acc = struct.unpack_from('<I', payload, 44)[0] / 1000.0  # mm -> m
        g_speed = struct.unpack_from('<i', payload, 60)[0] / 1000.0  # mm/s -> m/s
        head_mot = struct.unpack_from('<i', payload, 64)[0] * 1e-5  # degrees

        self.num_sats = num_sv

        # Check fix quality (2=2D, 3=3D, 4=GNSS+DR, 5=time-only)
        valid_fix = fix_type in (2, 3, 4)

        if not self.has_fix:
            if valid_fix and num_sv >= self.min_sats:
                self.has_fix = True
                self.get_logger().info(
                    f'GPS fix acquired: {num_sv} satellites, fix type {fix_type}')
            else:
                self.get_logger().info(
                    f'Waiting for GPS fix: {num_sv}/{self.min_sats} sats, '
                    f'fix type {fix_type}',
                    throttle_duration_sec=5.0)
                return

        if not valid_fix:
            return

        # Publish NavSatFix
        fix = NavSatFix()
        fix.header.stamp = self.get_clock().now().to_msg()
        fix.header.frame_id = 'gps'
        fix.latitude = lat
        fix.longitude = lon
        fix.altitude = h_msl

        fix.status.status = NavSatStatus.STATUS_FIX
        fix.status.service = NavSatStatus.SERVICE_GPS

        # Use reported accuracy for covariance
        h_cov = h_acc ** 2
        v_cov = v_acc ** 2
        fix.position_covariance = [h_cov, 0.0, 0.0,
                                   0.0, h_cov, 0.0,
                                   0.0, 0.0, v_cov]
        fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN

        self.fix_pub.publish(fix)

        # Publish speed and course
        self.speed_pub.publish(Float64(data=g_speed))
        self.course_pub.publish(Float64(data=head_mot))

    def destroy_node(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GPSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
