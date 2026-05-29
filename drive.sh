#!/bin/bash
# drive.sh — launches the full autonomous stack and arms it for driving.
# Use this when the dashboard is being unreliable.
# Run on the Jetson:    ~/drive.sh
# Stop with Ctrl+C — that kills everything cleanly.

set -e

cd "$HOME/mapless_nav_ws"

echo "=== killing any existing nodes ==="
pkill -f 'ros2 launch mapless_nav' 2>/dev/null || true
pkill -f 'mapless_nav/(camera|gps|teleop|pwm|safety|waypoint|perception|bev|reward|planner|speed|intervention)' 2>/dev/null || true
sleep 2

echo "=== sourcing ROS2 ==="
source /opt/ros/humble/setup.bash
source "$HOME/mapless_nav_ws/install/setup.bash"

echo "=== launching autonomous stack (logs -> /tmp/drive.log) ==="
nohup ros2 launch mapless_nav autonomous_launch.py > /tmp/drive.log 2>&1 &
LAUNCH_PID=$!
echo "launch PID = $LAUNCH_PID"

# Wait for nodes to come up (ESC arming takes ~5s now)
echo "=== waiting 12s for nodes + ESC arming ==="
sleep 12

echo "=== node list ==="
ros2 node list

echo ""
echo "=== last 10 launch log lines ==="
tail -10 /tmp/drive.log

echo ""
echo "=== triggering AUTONOMOUS mode ==="
ros2 topic pub -1 /autonomous_mode std_msgs/msg/Bool '{data: true}'

echo ""
echo "=========================================="
echo "  AUTONOMOUS MODE ACTIVE"
echo "  Ctrl+C here to stop everything cleanly"
echo "  Watch logs: tail -f /tmp/drive.log"
echo "=========================================="

# Cleanup on Ctrl+C
trap 'echo ""; echo "=== stopping ==="; ros2 topic pub -1 /autonomous_mode std_msgs/msg/Bool "{data: false}" 2>/dev/null; kill $LAUNCH_PID 2>/dev/null; sleep 1; pkill -f "ros2 launch mapless_nav" 2>/dev/null; pkill -f "mapless_nav/" 2>/dev/null; echo "stopped."; exit 0' INT

# Hold until user Ctrl+Cs
wait $LAUNCH_PID
