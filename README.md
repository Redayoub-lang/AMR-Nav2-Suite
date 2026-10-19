# AMR-Nav2-Suite (v3.0 Enterprise Edition)
## ROS 2 Nav2 Behavior Tree Architecture with Multi-AMR Fleet Traffic Manager

An enterprise-grade Autonomous Mobile Robot (AMR) Navigation Engine featuring **Nav2 Behavior Tree Control Flow**, **Multi-Robot Intersection Lock Allocation**, and **3D Depth Camera Perception Layering**.

---

## Enterprise Features in v3.0

* **ROS 2 Nav2 Behavior Tree Engine:** Reactive Sequence and Fallback tree execution handling recovery actions (Spin, Costmap Clearance, Replanning).
* **Multi-AMR Fleet Traffic Manager:** Priority-based temporal reservation grid locks preventing warehouse intersection deadlocks.
* **3D Perception Layer:** Depth camera 3D cloud projection detecting overhanging hazards ($0.3\text{m} \le z \le 2.0\text{m}$).
* **Fleet Tactical Dashboard:** Live monitoring of BT state node execution, intersection locks, and battery state-of-charge.
* **Telemetry Logger:** Exports high-frequency navigation states to `amr_nav2_v3_telemetry.csv`.

---

## Mathematical Mechanics

Behavior Tree Fallback Node Evaluation:

$$\mathcal{S}_{\text{Fallback}}(N_1, \dots, N_k) = \begin{cases} \text{SUCCESS} & \text{if } \exists i : \mathcal{S}(N_i) = \text{SUCCESS} \\ \text{FAILURE} & \text{otherwise} \end{cases}$$

---

## Quick Execution

```bash
# Install dependencies
pip install -r requirements.txt

# Run v3.0 Engine
python amr_navigation_engine.py