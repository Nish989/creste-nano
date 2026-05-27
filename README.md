# CREStE-Nano: Mapless Autonomous Navigation on Sub-$500 Hardware

> *Extending the CREStE paradigm (Zhang et al., RSS 2025) to ultra-low-cost edge hardware using monocular depth as a LiDAR substitute.*

---

## Overview

A $500 autonomous RC car that navigates to GPS waypoints on sidewalks and outdoor paths **without any pre-built map**. The robot learns terrain traversability from 20-30 minutes of human driving demonstrations and navigates in real time using only a webcam and GPS.

**Research Question:**
> Can monocular depth substitute for LiDAR in BEV-based learned reward mapless navigation? What is the performance cost of reducing hardware from $10,000 to $500?

This project directly extends [CREStE (RSS 2025)](https://amrl.cs.utexas.edu/creste/) by Prof. Joydeep Biswas's AMRL lab at UT Austin, which achieved 2km mapless navigation with 1 intervention on a $10,000 Clearpath Jackal. We test whether the paradigm holds on a $500 RC car with monocular vision only.

---

## Key Contributions

1. **Monocular Depth BEV Projection** — replaces LiDAR with Depth Anything V2 for bird's eye view feature projection. First systematic study of this substitution in learned reward navigation.

2. **MPPI Trajectory Optimization** — replaces greedy candidate selection with Model Predictive Path Integral control (1000 trajectories, 8-step horizon).

3. **GPS-Supervised Contrastive Reward Learning** — replaces binary MLP with InfoNCE contrastive loss, automatically mining positive/negative samples from GPS tracks.

4. **Online Intervention Adaptation** — RLHF-style online reward model updates from human takeovers during deployment. Measures NIR (interventions/100m) automatically.

---

## Hardware — $500 Total

| Component | Cost | Role |
|-----------|------|------|
| Jetson Orin Nano Super | ~$250 | Main compute (67 TOPS, 1024 CUDA cores) |
| Arrma Typhon Mega | ~$150 | 1/8 scale 4WD chassis |
| EMEET SmartCam Nova 4K | ~$50 | 90° HFOV, USB camera |
| HGLRC M10 GPS | ~$25 | 10Hz, 2m accuracy |
| ESP8266 NodeMCU | ~$5 | PWM bridge to servo/ESC |
| Power Bank (USB-C PD) | ~$20 | 12V to Jetson |

---

## Software Pipeline

```
EMEET Camera (90° HFOV)
        ↓
DINOv2-small + Depth Anything V2 small
        ↓
BEV Projection (64×64 grid, 0.15m/cell, monocular depth)
        ↓
Contrastive Reward Model (InfoNCE, GPS-supervised)
        ↓
MPPI Planner (K=1000 trajectories, T=8 steps)
        ↓
GPS Waypoint Bias
        ↓
ESP8266 → Servo + ESC → Car moves
        ↓
Intervention Monitor (NIR metric, online adaptation)
```

---

## Results

*[To be filled after outdoor evaluation]*

| Metric | CREStE (RSS 2025) | Ours |
|--------|-------------------|------|
| Hardware cost | ~$10,000 | ~$500 |
| Depth sensor | LiDAR | Monocular |
| NIR (interventions/100m) | 0.05 | TBD |
| Max continuous distance | 2km | TBD |
| Training data | Large scale | ~30 min |

---

## Repository Structure

```
├── mapless_nav_ws/               # ROS2 Humble workspace
│   └── src/mapless_nav/
│       ├── mapless_nav/
│       │   ├── camera_node.py          # EMEET camera → /camera/image_raw
│       │   ├── gps_node.py             # UART GPS → /gps/fix
│       │   ├── perception_node.py      # DINOv2 + Depth Anything V2
│       │   ├── bev_projection_node.py  # Camera features → BEV grid
│       │   ├── reward_node.py          # Contrastive reward model inference
│       │   ├── planner_node.py         # MPPI trajectory optimization
│       │   ├── pwm_control_node.py     # ROS2 → ESP8266 serial
│       │   ├── safety_node.py          # Watchdog + e-stop
│       │   ├── waypoint_manager_node.py # GPS route following
│       │   ├── intervention_monitor_node.py # RLHF online adaptation
│       │   ├── precompute_bev.py       # Offline: images → BEV .npz
│       │   ├── train_reward.py         # InfoNCE contrastive training
│       │   └── export_tensorrt.py      # TensorRT optimization
│       ├── launch/
│       │   ├── teleop_launch.py
│       │   ├── data_collection_launch.py
│       │   └── autonomous_launch.py
│       └── config/
│           └── params.yaml
├── dashboard/
│   └── app.py                    # Web control panel (phone-accessible)
├── esp8266_pwm_bridge/
│   └── esp8266_pwm_bridge.ino    # Arduino firmware
└── PROJECT_OVERVIEW.md
```

---

## Quick Start

```bash
# SSH into Jetson
ssh nishan@192.168.1.125

# Source ROS2
source ~/mapless_nav_ws/install/setup.bash

# Teleop (verify hardware works)
ros2 launch mapless_nav teleop_launch.py

# Collect training data
ros2 launch mapless_nav data_collection_launch.py

# Precompute BEV features (run on Jetson after data collection)
python3 -m mapless_nav.precompute_bev --data_dir ~/mapless_nav_data

# Train reward model (run on Mac/GPU machine)
python3 train_reward.py --data_dir ./bev_features --epochs 100

# Full autonomous mode
ros2 launch mapless_nav autonomous_launch.py

# Web dashboard (access at http://192.168.1.125:8080)
python3 ~/dashboard/app.py
```

---

## Wiring

```
Jetson USB          ──→ ESP8266 USB (/dev/ttyUSB0, 500000 baud)
ESP8266 D1 (GPIO5)  ──→ ESC signal (throttle)
ESP8266 D2 (GPIO4)  ──→ Servo signal (steering)
Jetson Pin 2  (5V)  ──→ GPS VCC
Jetson Pin 14 (GND) ──→ GPS GND
Jetson Pin 10 (RX)  ──→ GPS TX
Jetson Pin 8  (TX)  ──→ GPS RX
EMEET Camera USB    ──→ Jetson USB
```

---

## Math

**BEV Projection:**
$$b_x = \lfloor x/r + W/2 \rfloor, \quad b_y = \lfloor H - z/r \rfloor$$
$$x = d\sin\theta_h, \quad z = d\cos\theta_h\cos\theta_v$$

**InfoNCE Contrastive Loss:**
$$\mathcal{L} = -\log\frac{\exp(\mathbf{z}_i \cdot \mathbf{z}^+ / \tau)}{\exp(\mathbf{z}_i \cdot \mathbf{z}^+ / \tau) + \sum_j \exp(\mathbf{z}_i \cdot \mathbf{z}^-_j / \tau)}$$

**MPPI Update:**
$$\bar{u}_t \leftarrow \sum_{k=1}^{K} w_k u_{k,t}, \quad w_k = \frac{\exp(S(\tau_k)/\lambda)}{\sum_j \exp(S(\tau_j)/\lambda)}$$

---

## References

- Zhang et al. (2025). *CREStE: Scalable Mapless Navigation with Internet Scale Priors and Counterfactual Guidance.* RSS 2025. [Paper](https://amrl.cs.utexas.edu/creste/)
- Oquab et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision.* TMLR.
- Yang et al. (2024). *Depth Anything V2.* NeurIPS 2024.
- Williams et al. (2017). *Information Theoretic MPC for Model-Based Reinforcement Learning.* ICRA 2017.

---

## Author

Built by a high school student in Austin, TX as independent research.
Inspired by and extending the work of Prof. Joydeep Biswas's AMRL lab at UT Austin.
