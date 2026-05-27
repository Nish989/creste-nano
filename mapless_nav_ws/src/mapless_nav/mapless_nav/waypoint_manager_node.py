"""
Waypoint Manager Node - Loads a route YAML file, tracks progress through
waypoints, and publishes bearing/distance to current target waypoint.

Input: /gps/fix, /gps/course (GPS motion heading)
Output: /waypoint/bearing, /waypoint/distance, /waypoint/index
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float64, Int32, Bool, String
import yaml
import math
import os


def haversine(lat1, lon1, lat2, lon2):
    """Distance in meters between two GPS coordinates."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing_to(lat1, lon1, lat2, lon2):
    """Bearing in degrees (0=North, 90=East) from point 1 to point 2."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


class WaypointManagerNode(Node):
    def __init__(self):
        super().__init__('waypoint_manager_node')

        self.declare_parameter('route_file', '')
        self.declare_parameter('waypoint_radius', 3.0)  # meters to consider "reached"
        self.declare_parameter('save_waypoint_button_topic', '/save_waypoint')

        route_file = self.get_parameter('route_file').value
        self.wp_radius = self.get_parameter('waypoint_radius').value

        # Load waypoints
        self.waypoints = []
        self.current_idx = 0
        if route_file and os.path.exists(route_file):
            self._load_route(route_file)
        else:
            self.get_logger().info('No route file loaded. Use /save_waypoint to record waypoints.')

        # State
        self.current_lat = None
        self.current_lon = None
        self.current_heading = None

        # Subscribers
        self.create_subscription(NavSatFix, '/gps/fix', self.gps_cb, 10)
        self.create_subscription(Float64, '/gps/course', self.heading_cb, 10)
        self.create_subscription(Bool, '/save_waypoint', self.save_waypoint_cb, 10)

        # Publishers
        self.bearing_pub = self.create_publisher(Float64, '/waypoint/bearing', 10)
        self.distance_pub = self.create_publisher(Float64, '/waypoint/distance', 10)
        self.index_pub = self.create_publisher(Int32, '/waypoint/index', 10)
        self.status_pub = self.create_publisher(String, '/waypoint/status', 10)

        # Update at 10Hz
        self.create_timer(0.1, self.update)

        self.get_logger().info(
            f'Waypoint manager: {len(self.waypoints)} waypoints loaded, '
            f'reach radius: {self.wp_radius}m')

    def _load_route(self, path):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        self.waypoints = [(wp['lat'], wp['lon']) for wp in data.get('waypoints', [])]
        self.get_logger().info(f'Loaded {len(self.waypoints)} waypoints from {path}')

    def save_route(self, path):
        data = {
            'waypoints': [
                {'lat': lat, 'lon': lon} for lat, lon in self.waypoints
            ]
        }
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        self.get_logger().info(f'Saved {len(self.waypoints)} waypoints to {path}')

    def save_waypoint_cb(self, msg):
        """Save current GPS position as a waypoint."""
        if self.current_lat is None:
            self.get_logger().warn('No GPS fix, cannot save waypoint')
            return
        self.waypoints.append((self.current_lat, self.current_lon))
        self.get_logger().info(
            f'Waypoint {len(self.waypoints)} saved: '
            f'{self.current_lat:.7f}, {self.current_lon:.7f}')

        # Auto-save to file
        save_path = os.path.expanduser('~/mapless_nav_data/current_route.yaml')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.save_route(save_path)

    def gps_cb(self, msg):
        if msg.status.status >= 0:  # has fix
            self.current_lat = msg.latitude
            self.current_lon = msg.longitude

    def heading_cb(self, msg):
        self.current_heading = msg.data

    def update(self):
        if self.current_lat is None or not self.waypoints:
            return

        if self.current_idx >= len(self.waypoints):
            self.status_pub.publish(String(data='MISSION_COMPLETE'))
            self.bearing_pub.publish(Float64(data=0.0))
            self.distance_pub.publish(Float64(data=0.0))
            return

        target_lat, target_lon = self.waypoints[self.current_idx]

        # Distance to target
        dist = haversine(self.current_lat, self.current_lon, target_lat, target_lon)
        self.distance_pub.publish(Float64(data=dist))

        # Absolute bearing to target
        abs_bearing = bearing_to(self.current_lat, self.current_lon, target_lat, target_lon)

        # Relative bearing (how much to turn from current heading)
        if self.current_heading is not None:
            rel_bearing = abs_bearing - self.current_heading
            # Normalize to [-180, 180]
            rel_bearing = (rel_bearing + 180) % 360 - 180
        else:
            rel_bearing = 0.0

        self.bearing_pub.publish(Float64(data=rel_bearing))
        self.index_pub.publish(Int32(data=self.current_idx))

        # Check if waypoint reached
        if dist < self.wp_radius:
            self.get_logger().info(
                f'Waypoint {self.current_idx} reached! '
                f'({dist:.1f}m) Advancing to next.')
            self.current_idx += 1
            if self.current_idx >= len(self.waypoints):
                self.get_logger().info('ALL WAYPOINTS REACHED - MISSION COMPLETE')
                self.status_pub.publish(String(data='MISSION_COMPLETE'))

        self.status_pub.publish(String(
            data=f'WP {self.current_idx}/{len(self.waypoints)} '
                 f'dist={dist:.1f}m bearing={rel_bearing:.0f}deg'))


def main(args=None):
    rclpy.init(args=args)
    node = WaypointManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Save route on exit if waypoints were recorded
        if node.waypoints:
            save_path = os.path.expanduser('~/mapless_nav_data/current_route.yaml')
            node.save_route(save_path)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
