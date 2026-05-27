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

### 1. Semantic Feature Extraction (DINOv2)

An image $I \in \mathbb{R}^{H \times W \times 3}$ is divided into $N = \frac{H}{14} \times \frac{W}{14}$ non-overlapping patches. DINOv2-small maps each patch to a token:

$$\mathbf{f}_i = \text{DINOv2}(p_i) \in \mathbb{R}^{384}, \quad i = 1, \ldots, N$$

The patch tokens form a feature map $F \in \mathbb{R}^{h \times w \times 384}$ where $h = H/14$, $w = W/14$.

---

### 2. Monocular Depth Estimation

Depth Anything V2 predicts a dense depth map $D \in \mathbb{R}^{H \times W}$ from a single RGB image:

$$D = f_\theta(I)$$

Since the model outputs relative (affine-invariant) depth, we normalize per frame:

$$\hat{D}(u,v) = d_{\min} + \frac{D(u,v) - D_{\min}}{D_{\max} - D_{\min}}(d_{\max} - d_{\min})$$

where $d_{\min} = 0.5\text{m}$, $d_{\max} = 10\text{m}$ are physical range limits.

---

### 3. BEV Projection

Each pixel $(u, v)$ with depth $d = \hat{D}(u,v)$ is back-projected to 3D using camera intrinsics $(f_x, f_y, c_x, c_y)$:

$$X = \frac{(u - c_x) \cdot d}{f_x}, \quad Y = \frac{(v - c_y) \cdot d}{f_y}, \quad Z = d$$

Camera-to-robot transform (fixed mount with height $h_c$, pitch $\alpha$):

$$\begin{bmatrix} x_r \\ y_r \\ z_r \end{bmatrix} = R_\alpha \begin{bmatrix} X \\ Y \\ Z \end{bmatrix} + \begin{bmatrix} 0 \\ h_c \\ 0 \end{bmatrix}$$

Grid index in BEV (resolution $r = 0.15$ m/cell, grid size $W = H = 64$):

$$b_x = \left\lfloor \frac{x_r}{r} + \frac{W}{2} \right\rfloor, \quad b_z = \left\lfloor \frac{z_r}{r} \right\rfloor$$

The semantic feature at each pixel is splatted into the corresponding BEV cell. When multiple pixels map to the same cell, features are averaged:

$$\mathbf{g}_{b_x, b_z} = \frac{1}{|P_{b_x,b_z}|} \sum_{(u,v) \in P_{b_x,b_z}} \mathbf{f}(u, v)$$

giving BEV grid $G \in \mathbb{R}^{64 \times 64 \times 384}$.

---

### 4. Trajectory Feature Encoding

A candidate trajectory is a sequence of $T=8$ BEV grid coordinates $\{(b_x^t, b_z^t)\}_{t=1}^T$. The trajectory feature vector is formed by concatenating the grid features along all steps:

$$\phi(\tau) = \left[\mathbf{g}_{b_x^1, b_z^1} \,\|\, \mathbf{g}_{b_x^2, b_z^2} \,\|\, \cdots \,\|\, \mathbf{g}_{b_x^T, b_z^T}\right] \in \mathbb{R}^{T \cdot 384}$$

This is passed through the encoder MLP:

$$\mathbf{z} = \frac{E_\theta(\phi(\tau))}{\|E_\theta(\phi(\tau))\|_2} \in \mathbb{R}^{128}$$

where $E_\theta$ is a 3-layer MLP (input → 256 → 256 → 128) with ReLU activations and L2 normalization on the output.

---

### 5. Contrastive Reward Learning (InfoNCE)

Training uses GPS-supervised contrastive pairs. For each anchor frame with demonstrated steering $s$:

- **Anchor** $\mathbf{q}$: trajectory feature from the demonstrated steering action
- **Positive** $\mathbf{k}^+$: trajectory feature from a temporally nearby frame ($|i - j| \leq 5$), same terrain context
- **Negatives** $\{\mathbf{k}^-_j\}_{j=1}^{N}$: trajectories from hard negative steerings (sharp turns $s \in \{-1, -0.8, 0.8, 1.0\}$) and random steerings

The InfoNCE loss:

$$\mathcal{L}_{\text{NCE}} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(\mathbf{q}_i \cdot \mathbf{k}_i^+ / \tau)}{\exp(\mathbf{q}_i \cdot \mathbf{k}_i^+ / \tau) + \sum_{j=1}^{N} \exp(\mathbf{q}_i \cdot \mathbf{k}_{ij}^- / \tau)}$$

with temperature $\tau = 0.07$, batch size $B = 64$, $N = 8$ negatives per anchor.

At inference, the reward head $h_\psi : \mathbb{R}^{128} \to [0,1]$ scores each trajectory:

$$r(\tau) = \sigma\!\left(h_\psi(\mathbf{z})\right), \quad h_\psi : \mathbb{R}^{128} \xrightarrow{} \mathbb{R}^{64} \xrightarrow{} \mathbb{R}^1$$

---

### 6. MPPI Trajectory Optimization

At each planning step, $K=1000$ control sequences $\{U_k\}_{k=1}^K$ are sampled around the nominal sequence $\bar{U} \in \mathbb{R}^T$:

$$U_k = \bar{U} + \epsilon_k, \quad \epsilon_k \sim \mathcal{N}(0, \sigma^2 I), \quad \sigma = 0.3$$

Each $U_k$ is rolled out in BEV space:

$$x^{t+1} = x^t + u_k^t \cdot \delta_x, \quad z^{t+1} = z^t - \delta_z$$

with $\delta_x = 2.0$, $\delta_z = 1.5$ cells/step (15cm grid → ~22.5cm/step forward). Scores from the reward model are augmented with a GPS bearing bias:

$$S_k = r(\tau_k) + \beta \cdot \left(1 - \left|\bar{u}_k^0 - \hat{b}\right|\right)$$

where $\hat{b} = \text{clip}(\psi_{\text{bearing}} / 90°, -1, 1)$ is the normalized GPS bearing and $\beta = 0.3$.

MPPI softmax weights:

$$w_k = \frac{\exp\!\left(\frac{S_k - \max_j S_j}{\lambda}\right)}{\sum_{j} \exp\!\left(\frac{S_j - \max_j S_j}{\lambda}\right)}, \quad \lambda = 0.1$$

Nominal sequence update (receding horizon with momentum $\mu = 0.8$):

$$\bar{U} \leftarrow \text{clip}\!\left(\mu \bar{U} + \sum_{k=1}^K w_k \epsilon_k,\; -1, 1\right)$$

Action executed: $u^* = \bar{u}^0$, then $\bar{U}$ is shifted left by one step.

---

### 7. Online Reward Adaptation (RLHF)

An intervention is detected when:

$$\left|s_{\text{teleop}}(t) - s_{\text{auto}}(t)\right| > 0.15$$

On each intervention, the trajectory that was being executed gets added to a replay buffer as a negative example (label $y=0$). The buffer stores up to $M=500$ samples with FIFO replacement.

Every $n=10$ interventions, a mini-batch of size 16 is sampled uniformly from the buffer and the model is updated with binary cross-entropy:

$$\mathcal{L}_{\text{online}} = -\frac{1}{B}\sum_{i=1}^{B} \left[y_i \log r_i + (1 - y_i)\log(1 - r_i)\right]$$

using Adam with $\eta = 10^{-4}$ and gradient clipping at norm 1.0. The updated weights are saved to disk and hot-reloaded by the reward node.

---

### 8. Evaluation Metric

NIR (Normalized Intervention Rate) — primary metric from CREStE:

$$\text{NIR} = \frac{N_{\text{interventions}}}{d_{\text{autonomous}}} \times 100 \quad \left[\frac{\text{interventions}}{100\text{m}}\right]$$

Lower is better. CREStE on a $10,000 Jackal with LiDAR achieves NIR = 0.05. We target NIR < 1.0 on $500 hardware with monocular depth.

---

## References

- Zhang et al., CREStE, RSS 2025
- Oquab et al., DINOv2, TMLR 2023
- Yang et al., Depth Anything V2, NeurIPS 2024
- Williams et al., MPPI, ICRA 2017
