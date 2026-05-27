# CREStE-Nano

A $500 autonomous RC car that drives to GPS waypoints without a map. Built as independent research to extend [CREStE (RSS 2025)](https://amrl.cs.utexas.edu/creste/) from Prof. Joydeep Biswas's lab at UT Austin onto ultra-low-cost hardware.

The robot learns what terrain looks like from 20-30 minutes of human driving, then navigates on its own using only a webcam and GPS.

---

## The Research Question

CREStE achieved 2km mapless navigation on a $10,000 Clearpath Jackal with LiDAR. This project asks: **does the same paradigm work on $500 hardware with monocular depth instead of LiDAR?** Nobody has studied this tradeoff.

---

## Hardware

| Part | Cost |
|------|------|
| Jetson Orin Nano Super | ~$250 |
| Arrma Typhon Mega (RC car) | ~$150 |
| EMEET SmartCam Nova 4K | ~$50 |
| HGLRC M10 GPS | ~$25 |
| ESP8266 NodeMCU | ~$5 |
| Power bank + misc | ~$20 |
| **Total** | **~$500** |

---

## How It Works

```
Camera → DINOv2 + Depth Anything V2 → BEV grid → reward model → MPPI planner → ESP8266 → car
```

1. **Perception** — DINOv2-small extracts 384-dim semantic features per image patch. Depth Anything V2 estimates depth from a single camera.

2. **BEV Projection** — Features get projected from camera view onto a 64×64 top-down grid (9.6m × 9.6m, 15cm/cell) using the depth map. No LiDAR needed.

3. **Reward Learning** — A contrastive model (InfoNCE loss) trained on GPS-supervised demonstrations learns what driveable terrain looks like in BEV feature space.

4. **MPPI Planning** — 1000 random trajectories sampled, scored by the reward model over 8 steps, combined via softmax weighted average. GPS bearing biases toward the destination.

5. **Online Adaptation** — Every time a human takes over (intervention), the reward model updates online. NIR (interventions/100m) logged automatically.

---

## Key Contributions

- Monocular depth as LiDAR substitute for BEV projection — first systematic study
- GPS-supervised InfoNCE contrastive reward learning (no manual labeling)
- MPPI trajectory optimization on edge hardware
- RLHF-style online reward adaptation from human interventions

---

## Results

| | CREStE (RSS 2025) | Ours |
|--|---|---|
| Cost | ~$10,000 | ~$500 |
| Depth sensor | LiDAR | Monocular |
| NIR (interventions/100m) | 0.05 | TBD |

*Results pending outdoor evaluation*

---

## Setup

```bash
# SSH into Jetson
ssh nishan@192.168.1.125

# Build
cd ~/mapless_nav_ws && colcon build
source install/setup.bash

# Teleop (test hardware)
ros2 launch mapless_nav teleop_launch.py

# Collect training data
ros2 launch mapless_nav data_collection_launch.py

# Precompute BEV features (on Jetson)
python3 -m mapless_nav.precompute_bev --data_dir ~/mapless_nav_data

# Train reward model (on Mac/GPU)
python3 train_reward.py --data_dir ./bev_features

# Autonomous mode
ros2 launch mapless_nav autonomous_launch.py

# Web dashboard (open phone at http://192.168.1.125:8080)
python3 ~/dashboard/app.py
```

---

## Wiring

```
Jetson USB         → ESP8266 USB (/dev/ttyUSB0, 500000 baud)
ESP8266 D1         → ESC (throttle PWM)
ESP8266 D2         → Servo (steering PWM)
Jetson Pin 2 (5V)  → GPS VCC
Jetson Pin 14 GND  → GPS GND
Jetson Pin 10 RX   → GPS TX
Jetson Pin 8  TX   → GPS RX
EMEET camera       → Jetson USB
```

---

## Math

BEV projection:
$$x = d\sin\theta_h, \quad z = d\cos\theta_h\cos\theta_v, \quad b_x = \lfloor x/r + W/2 \rfloor$$

InfoNCE loss:
$$\mathcal{L} = -\log\frac{\exp(\mathbf{z} \cdot \mathbf{z}^+ / \tau)}{\exp(\mathbf{z} \cdot \mathbf{z}^+ / \tau) + \sum_j \exp(\mathbf{z} \cdot \mathbf{z}^-_j / \tau)}$$

MPPI update:
$$w_k = \frac{\exp(S_k/\lambda)}{\sum_j \exp(S_j/\lambda)}, \quad u^* = \sum_k w_k u_k$$

---

## References

- Zhang et al., CREStE, RSS 2025
- Oquab et al., DINOv2, TMLR 2023
- Yang et al., Depth Anything V2, NeurIPS 2024
- Williams et al., MPPI, ICRA 2017
