# AMR-Nav2-Suite (v2.0 Industrial Edition)
## Autonomous Mobile Robot Navigation Engine with AMCL, TEB Planner & Multi-Layer Costmap

An industrial-grade ROS 2-style Navigation Suite for Autonomous Mobile Robots (AMRs) featuring **Adaptive Monte Carlo Localization (AMCL)**, **Timed Elastic Band (TEB) Local Planner**, and **Exponential Inflation Multi-Layered Costmaps**.

---

## Key Features in v2.0

* **AMCL Particle Filter Localization:** Probabilistic cloud estimation with motion noise propagation, costmap measurement likelihood, and low-variance resampling.
* **TEB Local Planner:** Kinematically constrained trajectory optimization respecting $v_{\text{max}}$, $\omega_{\text{max}}$, acceleration bounds, and dynamic hazard repulsion.
* **Multi-Layered Costmap:** Integrated static obstacle layer and dynamic exponential inflation decay layer ($C_{\text{cost}} = 254 \cdot \exp(-\alpha d)$).
* **Tactical Visualizer:** Real-time dashboard showing particle clouds, inflation zones, TEB local band ribbons, and velocity setpoints.
* **Telemetry Data Logger:** Live export of pose estimates and velocity setpoints to `amr_nav2_v2_telemetry.csv`.

---

## Mathematical Mechanics

AMCL particle likelihood update based on costmap $C(r,c)$:

$$w_k^{[m]} \propto \exp\left(-\frac{C(r,c)}{50.0}\right)$$

Inflation cost decay formula:

$$C(d) = 254 \cdot \exp\left(-\alpha \cdot (d - r_{\text{inscribed}})\right)$$

---

## Execution Guide

```bash
# Install dependencies
pip install -r requirements.txt

# Run AMR Nav2 Engine v2.0
python amr_navigation_engine.py