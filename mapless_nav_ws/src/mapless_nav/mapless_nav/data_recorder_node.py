import queue
import threading
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, NavSatFix
from std_msgs.msg import Float64, Bool
import cv2
import numpy as np
import json
import os
from datetime import datetime


class DataRecorderNode(Node):
    def __init__(self):
        super().__init__('data_recorder_node')

        self.declare_parameter('output_dir', os.path.expanduser('~/mapless_nav_data'))
        self.output_dir = self.get_parameter('output_dir').value
        os.makedirs(self.output_dir, exist_ok=True)

        # State
        self.recording = False
        self.session_dir = None
        self.frame_count = 0
        self.metadata_file = None

        # Latest sensor data
        self.latest_gps = None
        self.latest_heading = None
        self.latest_steering = 0.0
        self.latest_throttle = 0.0

        # Background write queue — keeps image_cb from blocking on disk I/O
        self._write_queue = queue.Queue(maxsize=60)
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

        # Publishers
        self.recording_pub = self.create_publisher(Bool, '/recording', 10)
        # Publish sweep directly to safe topic so teleop's 0.0 doesn't overwrite it
        self.steer_pub = self.create_publisher(Float64, '/safe_cmd_steering', 10)

        # Subscribers
        self.create_subscription(CompressedImage, '/camera/image_raw/compressed', self.image_cb, 10)
        self.create_subscription(NavSatFix, '/gps/fix', self.gps_cb, 10)
        self.create_subscription(Float64, '/gps/course', self.heading_cb, 10)
        self.create_subscription(Float64, '/cmd_steering', self.steer_cb, 10)
        self.create_subscription(Float64, '/cmd_throttle', self.throttle_cb, 10)
        self.create_subscription(Bool, '/record_toggle', self.record_cb, 10)

        # Publish recording state at 2Hz
        self.create_timer(0.5, self._publish_state)

        self._wiggle_thread = None

        self.get_logger().info(f'Data recorder ready. Press Triangle to start recording. Output: {self.output_dir}')

    def _writer_loop(self):

        while True:
            item = self._write_queue.get()
            if item is None:
                break
            img_path, frame, meta_file, meta_line = item
            cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            meta_file.write(meta_line + '\n')

    def gps_cb(self, msg):
        self.latest_gps = {
            'lat': msg.latitude,
            'lon': msg.longitude,
            'alt': msg.altitude,
            'fix': msg.status.status,
        }

    def heading_cb(self, msg):
        self.latest_heading = msg.data

    def steer_cb(self, msg):
        self.latest_steering = msg.data

    def throttle_cb(self, msg):
        self.latest_throttle = msg.data

    def record_cb(self, msg):
        if msg.data:
            self.toggle_recording()

    def toggle_recording(self):
        if self.recording:
            self.stop_recording()
        else:
            self._start_wiggle()

    def _start_wiggle(self):

        if self._wiggle_thread is not None and self._wiggle_thread.is_alive():
            return
        self._wiggle_thread = threading.Thread(target=self._wiggle_sequence, daemon=True)
        self._wiggle_thread.start()

    def _wiggle_sequence(self):
        def pub_for(value, duration):
            end = time.monotonic() + duration
            while time.monotonic() < end:
                self.steer_pub.publish(Float64(data=value))
                time.sleep(0.02)  # 50Hz — matches safety node rate so wiggle wins reliably
        pub_for(1.0, 1.0)    # full right 1s
        pub_for(-1.0, 1.0)   # full left 1s
        pub_for(0.0, 0.5)    # center 0.5s
        self.start_recording()

    def start_recording(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = os.path.join(self.output_dir, f'session_{timestamp}')
        os.makedirs(os.path.join(self.session_dir, 'images'), exist_ok=True)

        self.metadata_file = open(
            os.path.join(self.session_dir, 'metadata.jsonl'), 'w')
        self.frame_count = 0
        self.recording = True
        self.get_logger().info(f'RECORDING STARTED: {self.session_dir}')

    def stop_recording(self):
        self.recording = False
        # Flush the write queue before closing the file
        self._write_queue.join()
        if self.metadata_file:
            self.metadata_file.close()
            self.metadata_file = None
        self.get_logger().info(
            f'RECORDING STOPPED: {self.frame_count} frames saved')

    def _publish_state(self):
        self.recording_pub.publish(Bool(data=self.recording))

    def image_cb(self, msg):
        if not self.recording:
            return

        frame = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)

        fname = f'{self.frame_count:06d}.jpg'
        img_path = os.path.join(self.session_dir, 'images', fname)

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        entry = {
            'frame': self.frame_count,
            'timestamp': stamp,
            'image': fname,
            'gps': self.latest_gps,
            'heading': self.latest_heading,
            'steering': self.latest_steering,
            'throttle': self.latest_throttle,
        }
        meta_line = json.dumps(entry)

        try:
            self._write_queue.put_nowait((img_path, frame.copy(), self.metadata_file, meta_line))
        except queue.Full:
            self.get_logger().warn('Write queue full — dropping frame', throttle_duration_sec=1.0)
            return

        self.frame_count += 1

    def destroy_node(self):
        if self.recording:
            self.stop_recording()
        self._write_queue.put(None)  # stop writer thread
        self._writer_thread.join()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DataRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
