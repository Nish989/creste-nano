#!/bin/bash
# Launch the autonomous stack and set /autonomous_mode true. Ctrl+C to stop.
# Use this if the dashboard hangs.

set -e

cd "$HOME/mapless_nav_ws"

echo "killing stale nodes..."
pkill -f 'ros2 launch mapless_nav' 2>/dev/null || true
pkill -f 'mapless_nav/(camera|gps|teleop|pwm|safety|waypoint|perception|bev|reward|planner|speed|intervention)' 2>/dev/null || true
sleep 2

source /opt/ros/humble/setup.bash
source "$HOME/mapless_nav_ws/install/setup.bash"

echo "launching (log: /tmp/drive.log)..."
nohup ros2 launch mapless_nav autonomous_launch.py > /tmp/drive.log 2>&1 &
LAUNCH_PID=$!

# ESC arm-retry takes ~5 s, give the stack 12 s total
sleep 12
ros2 node list
echo
tail -10 /tmp/drive.log
echo

ros2 topic pub -1 /autonomous_mode std_msgs/msg/Bool '{data: true}'
echo
echo "autonomous mode active. tail -f /tmp/drive.log to watch. Ctrl+C to stop."

trap 'echo; ros2 topic pub -1 /autonomous_mode std_msgs/msg/Bool "{data: false}" 2>/dev/null; kill $LAUNCH_PID 2>/dev/null; sleep 1; pkill -f "ros2 launch mapless_nav" 2>/dev/null; pkill -f "mapless_nav/" 2>/dev/null; exit 0' INT

wait $LAUNCH_PID
