#!/bin/bash
# Run data collection - launch from terminal or SSH before heading outside.
# Triangle = start/stop recording (ESP8266 LED on = recording)
# Double-tap X = shutdown everything
# Options held 3s = shutdown everything

source /opt/ros/humble/setup.bash
source ~/mapless_nav_ws/install/setup.bash

ros2 launch mapless_nav data_collection_launch.py
