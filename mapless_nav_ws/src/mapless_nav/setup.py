from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'mapless_nav'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jetson',
    maintainer_email='jetson@todo.todo',
    description='Mapless autonomous navigation using CREStE paradigm on Jetson Orin Nano',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'teleop_node = mapless_nav.teleop_node:main',
            'pwm_control_node = mapless_nav.pwm_control_node:main',
            'safety_node = mapless_nav.safety_node:main',
            'camera_node = mapless_nav.camera_node:main',
            'gps_node = mapless_nav.gps_node:main',
            'compass_node = mapless_nav.compass_node:main',
            'data_recorder_node = mapless_nav.data_recorder_node:main',
            'perception_node = mapless_nav.perception_node:main',
            'bev_projection_node = mapless_nav.bev_projection_node:main',
            'reward_node = mapless_nav.reward_node:main',
            'planner_node = mapless_nav.planner_node:main',
            'speed_controller_node = mapless_nav.speed_controller_node:main',
            'waypoint_manager_node = mapless_nav.waypoint_manager_node:main',
            'intervention_monitor_node = mapless_nav.intervention_monitor_node:main',
        ],
    },
)
