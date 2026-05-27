"""Launch file for Phase 6: Full autonomous navigation."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('mapless_nav'),
        'config', 'params.yaml'
    )

    route_file_arg = DeclareLaunchArgument(
        'route_file',
        default_value=os.path.expanduser('~/mapless_nav_data/current_route.yaml'),
        description='Path to route YAML file'
    )

    return LaunchDescription([
        route_file_arg,

        # Safety (always first)
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
        # Teleop for manual override / e-stop
        Node(
            package='mapless_nav',
            executable='teleop_node',
            name='teleop_node',
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
        # Perception
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
        # Planning
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
        Node(
            package='mapless_nav',
            executable='speed_controller_node',
            name='speed_controller_node',
            parameters=[config],
            output='screen',
        ),
        Node(
            package='mapless_nav',
            executable='waypoint_manager_node',
            name='waypoint_manager_node',
            parameters=[config, {
                'route_file': LaunchConfiguration('route_file'),
            }],
            output='screen',
        ),
        Node(
            package='mapless_nav',
            executable='intervention_monitor_node',
            name='intervention_monitor_node',
            parameters=[config],
            output='screen',
        ),
    ])
