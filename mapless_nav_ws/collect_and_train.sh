#!/bin/bash
# =============================================================
# Mapless Nav: Collect Data & Train Reward Model (one-shot)
#
# What this does:
#   1. Launches all nodes (teleop + sensors + recorder)
#   2. You drive around with the PS5 controller
#      - Press SQUARE to start/stop recording
#      - Press PS button for emergency stop
#      - Hold R1 (deadman) + R2 (throttle) + left stick (steer)
#   3. When you're done driving, press Ctrl+C
#   4. It automatically extracts features and trains the reward model
# =============================================================

set -e

DATA_DIR="$HOME/mapless_nav_data"
MODEL_DIR="$HOME/models/reward_model"
WS_DIR="$HOME/mapless_nav_ws"

echo "========================================"
echo " MAPLESS NAV - Data Collection & Training"
echo "========================================"
echo ""
echo "Controls:"
echo "  R1 (hold)     = deadman switch (must hold to drive)"
echo "  R2             = throttle"
echo "  L2             = brake/reverse"
echo "  Left stick     = steering"
echo "  SQUARE       = start/stop RECORDING"
echo "  PS button      = emergency stop"
echo ""
echo "Press SQUARE to begin recording before you start driving."
echo "Press SQUARE again when you finish a run."
echo "You can do multiple recording sessions."
echo "Press Ctrl+C when you're done collecting data."
echo ""
echo "========================================"
echo ""

# Source ROS2
source /opt/ros/humble/setup.bash
source "$WS_DIR/install/setup.bash"

# Step 1: Data collection
echo "[1/3] Launching data collection..."
echo "      Drive around and record with SQUARE. Ctrl+C when done."
echo ""

ros2 launch mapless_nav data_collection_launch.py || true

echo ""
echo "========================================"
echo "[2/4] Checking recorded data..."
echo "========================================"
echo ""

# Count recorded frames
FRAME_COUNT=$(find "$DATA_DIR" -name "*.jpg" 2>/dev/null | wc -l)
if [ "$FRAME_COUNT" -eq 0 ]; then
    echo "ERROR: No recorded frames found in $DATA_DIR"
    echo "Did you press SQUARE to start recording?"
    exit 1
fi
echo "Found $FRAME_COUNT recorded frames."
echo ""

# Step 2: Visualize BEV to sanity-check pipeline
echo "========================================"
echo "[3/4] Generating BEV visualizations..."
echo "      Check output in $DATA_DIR/visualizations/"
echo "========================================"
echo ""

cd "$WS_DIR"
python3 -m mapless_nav.visualize_bev \
    --data_dir "$DATA_DIR" \
    --sample_every 20

echo ""
echo "Visualizations saved to $DATA_DIR/visualizations/"
echo "Review them before training. Press Ctrl+C to abort, or Enter to continue."
read -r

# Step 3: Extract features and train
echo "========================================"
echo "[4/4] Training reward model..."
echo "========================================"
echo ""

python3 -m mapless_nav.train_reward \
    --data_dir "$DATA_DIR" \
    --model_dir "$MODEL_DIR" \
    --epochs 50

echo ""
echo "========================================"
echo " DONE!"
echo ""
echo " Reward model saved to: $MODEL_DIR/reward_mlp.pth"
echo ""
echo " To run autonomous mode:"
echo "   cd ~/mapless_nav_ws"
echo "   source install/setup.bash"
echo "   ros2 launch mapless_nav autonomous_launch.py"
echo "========================================"
