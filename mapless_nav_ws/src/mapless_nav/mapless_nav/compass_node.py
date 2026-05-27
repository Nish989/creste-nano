"""
Compass Node - Reads heading from QMC5883L magnetometer over I2C.
Publishes heading in degrees (0-360, 0=North) to /compass/heading.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import smbus2
import math
import struct


# QMC5883L registers
QMC5883L_ADDR = 0x0D
REG_DATA = 0x00
REG_STATUS = 0x06
REG_CONTROL1 = 0x09
REG_CONTROL2 = 0x0A
REG_SET_RESET = 0x0B


class CompassNode(Node):
    def __init__(self):
        super().__init__('compass_node')

        self.declare_parameter('i2c_bus', 1)
        self.declare_parameter('rate_hz', 20.0)
        self.declare_parameter('declination_deg', 0.0)  # magnetic declination for your area

        bus_num = self.get_parameter('i2c_bus').value
        rate = self.get_parameter('rate_hz').value
        self.declination = self.get_parameter('declination_deg').value

        try:
            self.bus = smbus2.SMBus(bus_num)
            self._init_sensor()
            self.get_logger().info(f'Compass (QMC5883L) initialized on I2C bus {bus_num}')
        except Exception as e:
            self.get_logger().error(f'Failed to init compass: {e}')
            self.bus = None
            return

        self.heading_pub = self.create_publisher(Float64, '/compass/heading', 10)
        self.create_timer(1.0 / rate, self.read_heading)

        # Simple calibration offsets (set during calibration)
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.scale_x = 1.0
        self.scale_y = 1.0

    def _init_sensor(self):
        # Reset
        self.bus.write_byte_data(QMC5883L_ADDR, REG_CONTROL2, 0x80)
        import time
        time.sleep(0.01)
        # Set/Reset period
        self.bus.write_byte_data(QMC5883L_ADDR, REG_SET_RESET, 0x01)
        # Continuous mode, 200Hz ODR, 8G range, 512 oversampling
        self.bus.write_byte_data(QMC5883L_ADDR, REG_CONTROL1, 0x1D)

    def read_heading(self):
        if self.bus is None:
            return

        try:
            # Check data ready
            status = self.bus.read_byte_data(QMC5883L_ADDR, REG_STATUS)
            if not (status & 0x01):
                return

            # Read 6 bytes: XL, XH, YL, YH, ZL, ZH
            data = self.bus.read_i2c_block_data(QMC5883L_ADDR, REG_DATA, 6)
            x, y, z = struct.unpack('<hhh', bytes(data))

            # Apply calibration
            x_cal = (x - self.offset_x) * self.scale_x
            y_cal = (y - self.offset_y) * self.scale_y

            # Calculate heading
            heading = math.degrees(math.atan2(y_cal, x_cal))
            heading += self.declination
            heading = (heading + 360) % 360

            self.heading_pub.publish(Float64(data=heading))

        except Exception as e:
            self.get_logger().warn(f'Compass read error: {e}', throttle_duration_sec=5.0)

    def destroy_node(self):
        if self.bus:
            self.bus.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CompassNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
