"""Minimal test launch: camera → perception → BEV → reward → planner.
Skips teleop, GPS, safety, PWM, waypoint nodes for indoor stand testing."""
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
            executable='camera_node',
            name='camera_node',
            parameters=[config],
            output='screen',
        ),
        Node(
            package='mapless_nav',
            executable='perception_node',
            name='perception_node',
            parameters=[config],
            output='screen',
        ),
        Node(
            package='mapless_nav',
            executable='bev_projection_node',
            name='bev_projection_node',
            parameters=[config],
            output='screen',
        ),
        Node(
            package='mapless_nav',
            executable='reward_node',
            name='reward_node',
            parameters=[config],
            output='screen',
        ),
        Node(
            package='mapless_nav',
            executable='planner_node',
            name='planner_node',
            parameters=[config],
            output='screen',
        ),
    ])
