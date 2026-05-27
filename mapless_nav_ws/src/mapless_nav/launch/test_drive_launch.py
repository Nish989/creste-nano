"""Test launch for first autonomous driving test.
Camera + perception + BEV + reward + planner + safety + PWM + teleop.
No GPS, no waypoints, no speed controller.

Controls:
  Left stick    = steering (manual mode)
  R2 / L2       = throttle / brake (manual mode)
  Square        = E-STOP toggle (always works)
  Circle        = Toggle autonomous mode
  X (double-tap)= Quit
  Options (3s)  = Shutdown

Starts in MANUAL mode. Press Circle to engage autonomous.
Touching joystick or pressing Circle again returns to manual.
E-stop always overrides everything.
"""
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
        # Safety + PWM (always first)
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
        # Teleop for manual control + mode switching + e-stop
        Node(
            package='mapless_nav',
            executable='teleop_node',
            name='teleop_node',
            parameters=[config],
            output='screen',
        ),
        # Camera
        Node(
            package='mapless_nav',
            executable='camera_node',
            name='camera_node',
            parameters=[config],
            output='screen',
        ),
        # Perception pipeline
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
        # Reward + planner (only publishes when autonomous mode toggled)
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
