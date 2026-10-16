# AMR-Nav2-Suite: Autonomous Mobile Robot Navigation Engine

An industrial-grade navigation engine for Autonomous Mobile Robots (AMRs) featuring dual-layer path planning with **$A^*$ Global Planner**, **Dynamic Window Approach (DWA) Local Planner**, and **Differential Drive Kinematics**.

---

## Core Features

* **Grid Occupancy Mapping:** Spatial discretization for wall obstacles and dynamic obstacles.
* **Global $A^*$ Path Search:** Optimal shortest-path generation with Euclidean distance heuristic.
* **DWA Local Controller:** Dynamic obstacle avoidance and smooth trajectory rollout.
* **Differential Drive Odometry:** Forward kinematics integration tracking pose $(x, y, \theta)$.
* **Telemetry Data Logger:** Real-time logging of velocity setpoints and pose state to `amr_navigation_log.csv`.

---

## Kinematics & Control Laws

$$\begin{bmatrix} \dot{x} \\ \dot{y} \\ \dot{\theta} \end{bmatrix} = \begin{bmatrix} \cos\theta & 0 \\ \sin\theta & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} v \\ \omega \end{bmatrix}$$

---

## Execution Guide

```bash
# Install dependencies
pip install -r requirements.txt

# Run AMR Navigator
python amr_navigation_engine.py