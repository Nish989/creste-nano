# Mapless Autonomous Navigation on Ultra-Low-Cost Hardware
## Project Overview for Claude Code

---

## 1. What This Project Is

A sub-$500 autonomous RC car that navigates outdoor environments (sidewalks, campus paths, neighborhoods) **without any map**. It uses a webcam + GPS + vision foundation models to understand terrain in real time and drive toward GPS waypoints — staying on sidewalks, avoiding grass, obstacles, and people.

This is inspired by CREStE (RSS 2025) from UT Austin's AMRL lab (Prof. Joydeep Biswas), which achieved 2km mapless urban navigation with 1 intervention. Our contribution: testing whether this paradigm works on ultra-low-cost edge hardware with monocular vision only.

**This is NOT SLAM. There is NO map building. The robot reacts to what it sees RIGHT NOW.**

### End Goal
A car that you give a GPS waypoint to and it drives there autonomously — staying on sidewalk, avoiding obstacles, without any pre-built map. Evaluated by number of human interventions per 100m driven (lower = better). CREStE achieved ~0.05 interventions/100m; we aim to show the paradigm works even on $500 hardware.

---

## 2. Hardware

| Component | Specs | Interface |
|-----------|-------|-----------|
| Jetson Orin Nano Super | 67 TOPS, 8GB, 6-core ARM A78AE @ 1.7GHz, 1024 CUDA cores | Main compute |
| Arrma Typhon Mega | 1/8 scale 4WD buggy, brushless converted, lowered suspension | Chassis |
| EMEET SmartCam Nova 4K | 4K/1080p @ 30-60fps, 90° HFOV, 58° VFOV, USB, fixed focus | USB to Jetson (/dev/video0) |
| HGLRC M10 GPS | u-blox M10, 72ch, 10Hz, 2m accuracy, UBX protocol | UART (TX→Pin10 RX, RX→Pin8 TX, VCC→Pin2 5V, GND→Pin14) |
| ESP8266 (NodeMCU) | PWM bridge: receives serial from Jetson, outputs 50Hz RC PWM | USB serial to Jetson (/dev/ttyUSB0, 500000 baud) |
| Steering Servo | Standard RC servo, powered by BEC | ESP8266 D2 (GPIO4) |
| ESC (brushless) | Stock ESC, signal + ground only to ESP8266 | ESP8266 D1 (GPIO5) |
| Power Bank | USB-C PD, powers Jetson via 12V PD trigger cable | DC barrel jack on Jetson |
| BEC | Steps battery voltage to 5-6V | Servo power + ESP8266 VIN |

### Wiring Summary
```
Jetson USB          ──→ ESP8266 USB (serial bridge, /dev/ttyUSB0)
ESP8266 D1 (GPIO5)  ──→ ESC signal (throttle)
ESP8266 D2 (GPIO4)  ──→ Servo signal (steering)
ESP8266 GND         ──→ Common ground with servo/ESC
ESP8266 VIN         ──→ BEC 5V

Jetson Pin 2 (5V)   ──→ GPS VCC
Jetson Pin 14 (GND) ──→ GPS GND
Jetson Pin 10 (RX)  ──→ GPS TX
Jetson Pin 8 (TX)   ──→ GPS RX
Webcam USB          ──→ Jetson USB
Power Bank USB-C    ──→ PD trigger cable ──→ Jetson DC barrel jack (12V)
```

### ESP8266 Serial Protocol (500000 baud, /dev/ttyUSB0)
**ALL motion commands go through the ESP8266 — no direct GPIO PWM from Jetson.**

| Command | Example | Effect |
|---------|---------|--------|
| `S<us>\n` | `S1500\n` | Set steering pulse width (1000–2000 μs) |
| `T<us>\n` | `T1600\n` | Set throttle pulse width (1000–2000 μs) |
| `N\n` | — | Both channels to neutral (1500 μs) |
| `P\n` | — | Ping — ESP responds `OK\n` |
| `L1\n` / `L0\n` | — | Onboard LED on/off (recording indicator) |

**Wiring matches labels:** Servo on D2 (S command = steering), ESC on D1 (T command = throttle).

### Important Notes
- RC PWM: 50Hz, 1000μs (full left/reverse) → 1500μs (neutral) → 2000μs (full right/forward).
- Jetson has no direct PWM to servo/ESC — the ESP8266 is the only control path.
- Camera uses V4L2 backend with MJPG codec (`CAP_V4L2` + `FOURCC('M','J','P','G')`).

---

## 3. Software Architecture

### OS & Framework
- JetPack (Ubuntu-based) on Jetson Orin Nano Super
- ROS2 Humble
- Python 3 primarily

### Software Pipeline (~10-15fps target)

```
┌──────────────────────────────────────┐
│              SENSORS                  │
│  /dev/video0 (EMEET 4K, 90° HFOV)   │
│  UART GPS (10Hz)                      │
└────────────┬────────────┬─────────────┘
             │            │
             ▼            │
┌────────────────────┐    │
│    PERCEPTION      │    │
│  DINOv2-small      │    │
│  Depth Anything V2 │    │
│  → /perception/*  │    │
└────────────┬───────┘    │
             │            │
             ▼            │
┌────────────────────┐    │
│  BEV PROJECTION    │    │
│  features + depth  │    │
│  → /bev/features   │    │
└────────────┬───────┘    │
             │            │
             ▼            │
┌────────────────────┐    │
│  REWARD PREDICTION │    │
│  BEV → cost/cell   │    │
│  → /cost_map       │    │
└────────────┬───────┘    │
             │            ▼
             └──→ PLANNER (cost map + GPS waypoint)
                     │
                     ▼
              pwm_control_node
                     │  serial ASCII (500kbaud)
                     ▼
               ESP8266 NodeMCU
                  D1   D2
                  │     │
                 ESC  Servo
```

### ROS2 Node Structure

```
mapless_nav_ws/src/mapless_nav/
├── launch/
│   ├── teleop_launch.py           # Gamepad → ESP8266 → car
│   ├── data_collection_launch.py  # Teleop + recording
│   ├── test_drive_launch.py
│   ├── test_pipeline_launch.py
│   └── autonomous_launch.py       # Full autonomous mode
├── mapless_nav/
│   ├── camera_node.py             # /dev/video0 → /camera/image_raw
│   ├── gps_node.py                # UART → /gps/fix
│   ├── compass_node.py
│   ├── teleop_node.py             # Gamepad → steering/throttle topics
│   ├── pwm_control_node.py        # Topics → serial → ESP8266
│   ├── perception_node.py         # DINOv2 + Depth Anything V2
│   ├── bev_projection_node.py     # Camera features + depth → BEV grid
│   ├── reward_node.py             # BEV → cost map
│   ├── planner_node.py            # cost map + GPS → commands
│   ├── safety_node.py             # Emergency stop
│   ├── speed_controller_node.py
│   ├── waypoint_manager_node.py
│   ├── data_recorder_node.py      # Saves images + metadata.jsonl
│   ├── precompute_bev.py          # Offline: images → BEV .npz files
│   ├── train_reward.py            # Train reward MLP
│   ├── visualize_bev.py
│   └── export_tensorrt.py
├── config/
│   ├── camera_params.yaml         # FOV: 90° H, 58° V
│   ├── pwm_params.yaml            # port: /dev/ttyUSB0, baud: 500000
│   ├── gps_params.yaml
│   └── nav_params.yaml
└── esp8266_pwm_bridge/
    └── esp8266_pwm_bridge.ino     # Flash via Arduino IDE (500000 baud)
```

---

## 4. Training Data

### Current data (as of 2026-04-08)
- **8 sessions, ~6,150 frames ≈ ~10 minutes** of human driving
- Collected with old **Lenovo 500 FHD (75° HFOV)** — BEV projections will be slightly off with new 90° camera. Re-collect before final evaluation.

### How much data is needed

| Amount | What you get |
|--------|-------------|
| ~10 min / ~6k frames | Barely enough — trains but generalizes poorly |
| **~20-30 min / ~12-18k frames** | **Sweet spot for sidewalk-vs-grass discrimination** |
| ~1-2 hours / ~36-72k frames | CREStE-scale robustness |

**Next step:** Collect 2-3 more sessions (~20 min total) with the EMEET camera before re-training.

### Data format
```
~/mapless_nav_data/
├── session_<timestamp>/
│   ├── images/           # JPEG frames
│   └── metadata.jsonl    # {image, steering, throttle, heading, gps_lat, gps_lon, timestamp}
└── bev_features/         # Preprocessed .npz files (run precompute_bev.py to generate)
```

---

## 5. Development Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 — Teleoperation | ✅ Done | Gamepad → ESP8266 → car drives |
| 2 — Sensors + Recording | ✅ Done | Camera + GPS streaming and recording |
| 3 — Data Collection | 🔄 In progress | ~10 min collected, need ~20-30 min more with new camera |
| 4 — Perception Pipeline | ✅ Done | DINOv2 + Depth Anything V2 on Jetson GPU |
| 5 — BEV + Reward Learning | 🔄 In progress | Nodes written, needs new camera data + training |
| 6 — Planning + Autonomy | 🔜 Next | Planner node written, end-to-end test pending |
| 7 — Evaluation + Paper | 🔜 Later | GARSEF target, NIR/subgoal metrics |

---

## 6. Quick Start Commands

```bash
# SSH into Jetson
ssh jetson@192.168.1.125

# Source ROS2
source ~/mapless_nav_ws/install/setup.bash

# Manual teleop (gamepad)
ros2 launch mapless_nav teleop_launch.py

# Collect training data (teleop + auto-record)
ros2 launch mapless_nav data_collection_launch.py

# Precompute BEV features from all sessions (re-run after collecting new data)
cd ~/mapless_nav_ws
python3 -m mapless_nav.precompute_bev --data_dir ~/mapless_nav_data

# Train reward model
python3 -m mapless_nav.train_reward --data_dir ~/mapless_nav_data/bev_features

# Full autonomous mode
ros2 launch mapless_nav autonomous_launch.py

# Camera test + depth map + point cloud
python3 ~/camera_depth_test.py

# SCP files to laptop
scp jetson@192.168.1.125:/home/jetson/camera_test.jpg .
scp jetson@192.168.1.125:/home/jetson/depth_comparison.jpg .
scp jetson@192.168.1.125:/home/jetson/pointcloud.ply .
```

---

## 7. Key Models & Libraries

| Model/Library | Purpose | FPS on Orin Nano |
|---------------|---------|-----------------|
| DINOv2-small | Semantic feature extraction | ~15-20 |
| Depth Anything V2 small | Monocular depth | ~10-15 |
| Custom reward MLP | BEV → cost | ~100+ |
| ROS2 Humble | Robot middleware | — |
| PyTorch + TensorRT | Inference | — |
| pyserial | ESP8266 serial bridge | — |
| OpenCV (CAP_V4L2) | Camera capture | — |

---

## 8. Research Context

### CREStE (the paper we're replicating cheaply)
- RSS 2025, UT Austin AMRL (Prof. Joydeep Biswas)
- DINOv2 + SAM2 features → BEV → learned reward → mapless navigation
- 2km Austin streets, 1 intervention, ~$10k Clearpath Jackal platform

### Our contribution
- Same paradigm on sub-$500 hardware with monocular depth instead of LiDAR
- Quantify failure modes and minimum viable data requirements at this cost point

### Target venues
- GARSEF → TXSEF → ISEF
- Potential workshop paper

### References
- Zhang et al. (2025). "CREStE: Scalable Mapless Navigation with Internet Scale Priors and Counterfactual Guidance." RSS 2025.
- AMRL Lab: https://amrl.cs.utexas.edu/
- CREStE: https://amrl.cs.utexas.edu/creste/
