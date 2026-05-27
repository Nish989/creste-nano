"""Launch file for Phase 1: Teleop driving."""
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
    ])
