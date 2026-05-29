import subprocess
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        self.declare_parameter('device', '/dev/video0')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('jpeg_quality', 80)
        # exposure_time: 0 = auto (default). Set to a positive int to use manual
        # shutter speed (v4l2 exposure_time_absolute units = 100µs).
        self.declare_parameter('exposure_time', 0)

        device = self.get_parameter('device').value
        width = self.get_parameter('width').value
        height = self.get_parameter('height').value
        fps = self.get_parameter('fps').value
        self.jpeg_quality = self.get_parameter('jpeg_quality').value
        exposure_time = self.get_parameter('exposure_time').value

        self.cap = None
        self._device = device
        self._exposure_time = exposure_time if exposure_time > 0 else 10

        # Try GStreamer with hardware JPEG decoder first.
        # Use manual exposure in GStreamer so auto_exposure doesn't fight us outdoors
        # Always use manual exposure — auto_exposure blows out outdoors.
        # exposure_time param overrides default of 30 (×100µs = 1/333s shutter).
        _exp = exposure_time if exposure_time > 0 else 40
        _gst_exp = f"auto_exposure=1,exposure_time_absolute={_exp}"
        gst_pipeline = (
            f"v4l2src device={device} extra-controls=\"c,{_gst_exp}\" ! "
            f"image/jpeg,width={width},height={height},framerate=30/1 ! "
            f"nvjpegdec ! nvvidconv ! "
            f"video/x-raw,format=BGRx ! videoconvert ! "
            f"video/x-raw,format=BGR ! "
            f"appsink drop=1 max-buffers=1 sync=false"
        )
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                self.cap = cap
                self.get_logger().info(
                    f'Camera opened via GStreamer (hw decode): {width}x{height} @ {fps}fps')
            else:
                cap.release()

        # Fallback: OpenCV V4L2 software decode
        if self.cap is None:
            self.get_logger().warn('GStreamer hw decode failed, falling back to V4L2 sw decode')
            cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                self.get_logger().fatal(f'Failed to open camera at {device}')
                raise RuntimeError(f'Failed to open camera at {device}')
            self.cap = cap
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            self.get_logger().info(
                f'Camera opened via V4L2 (sw decode): {actual_w}x{actual_h} @ {actual_fps:.0f}fps')

        self.pub = self.create_publisher(CompressedImage, '/camera/image_raw/compressed', 10)
        self.create_timer(1.0 / fps, self.capture)

        # Set exposure AFTER pipeline is open — camera resets during GStreamer init
        self._set_exposure(self._device, self._exposure_time)
        # Reapply every 5s — some cameras drift back to auto
        self.create_timer(5.0, lambda: self._set_exposure(self._device, self._exposure_time))

    def _set_exposure(self, device, exposure_time):
        if exposure_time <= 0:
            exposure_time = 40  # default manual shutter for outdoors
        # Manual exposure: disable auto, then set absolute shutter speed
        r1 = subprocess.run(
            ['v4l2-ctl', f'--device={device}', '-c', 'auto_exposure=1'],
            capture_output=True)
        r2 = subprocess.run(
            ['v4l2-ctl', f'--device={device}', '-c', f'exposure_time_absolute={exposure_time}'],
            capture_output=True)
        if r1.returncode == 0 and r2.returncode == 0:
            self.get_logger().info(
                f'Exposure: manual, shutter={exposure_time} (×100µs = 1/{10000//exposure_time}s)')
        else:
            self.get_logger().warn(
                f'Failed to set manual exposure — check v4l2-ctl. Staying on auto.')

    @staticmethod
    def _build_gamma_lut(gamma: float):
        """Build a lookup table for gamma correction (gamma>1 darkens highlights)."""
        import numpy as np
        table = (np.arange(256, dtype=np.float32) / 255.0) ** gamma
        return (table * 255).astype('uint8')

    # CLAHE for adaptive local contrast — handles bright outdoor + dark shadow in same frame
    _clahe = None

    def capture(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return

        # Build CLAHE once
        if CameraNode._clahe is None:
            # clipLimit=2.5: moderate contrast boost without noise amplification
            # tileGridSize=(8,8): local 8×8 grid so bright ground + dark trees both normalize
            CameraNode._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

        # Apply CLAHE on L channel in LAB space — preserves colours, fixes exposure locally
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = CameraNode._clahe.apply(l)
        lab = cv2.merge([l, a, b])
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        msg.format = 'jpeg'
        msg.data = buf.tobytes()
        self.pub.publish(msg)

    def destroy_node(self):
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
