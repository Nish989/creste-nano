"""Launch file for Phase 2-3: Data collection (teleop + sensors + recording)."""
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('mapless_nav'),
        'config', 'params.yaml'
    )

    return LaunchDescription([
        # Teleop stack
        Node(
            package='mapless_nav',
            executable='teleop_node',
            name='teleop_node',
            parameters=[config],
            output='screen',
        ),
        Node(
            package='mapless_nav',
            executable='safety_node',
            name='safety_node',
            parameters=[config],
            output='screen',
        ),
        Node(
            package='mapless_nav',
            executable='pwm_control_node',
            name='pwm_control_node',
            parameters=[config],
            output='screen',
        ),
        # Sensors
        Node(
            package='mapless_nav',
            executable='camera_node',
            name='camera_node',
            parameters=[config],
            output='screen',
        ),
        Node(
            package='mapless_nav',
            executable='gps_node',
            name='gps_node',
            parameters=[config],
            output='screen',
        ),
        # Recorder
        Node(
            package='mapless_nav',
            executable='data_recorder_node',
            name='data_recorder_node',
            parameters=[config],
            output='screen',
        ),
        # Waypoint manager (for recording waypoints during driving)
        Node(
            package='mapless_nav',
            executable='waypoint_manager_node',
            name='waypoint_manager_node',
            parameters=[config],
            output='screen',
        ),
    ])
