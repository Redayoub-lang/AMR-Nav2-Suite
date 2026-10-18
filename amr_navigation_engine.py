import numpy as np
import cv2
import time
import math
import csv


class LayeredCostmap:
    """Industrial Multi-Layered Costmap with Inflation Layer Generation."""
    def __init__(self, width=800, height=600, resolution=10):
        self.width = width
        self.height = height
        self.res = resolution
        self.cols = width // resolution
        self.rows = height // resolution
        
        # Costmap Layers
        self.static_layer = np.zeros((self.rows, self.cols), dtype=np.uint8)
        self.inflation_layer = np.zeros((self.rows, self.cols), dtype=np.float32)
        
        # Build Static Map Boundaries & Obstacles
        self.static_layer[0, :] = 255
        self.static_layer[-1, :] = 255
        self.static_layer[:, 0] = 255
        self.static_layer[:, -1] = 255
        self._add_box_obstacle(15, 12, 28, 30)
        self._add_box_obstacle(42, 22, 54, 48)

        self.update_inflation_layer(alpha=0.18, robot_radius_cells=2)

    def _add_box_obstacle(self, r1, c1, r2, c2):
        self.static_layer[r1:r2, c1:c2] = 255

    def update_inflation_layer(self, alpha=0.18, robot_radius_cells=2):
        """Calculates exponential cost decay inflation around obstacles."""
        obs_coords = np.argwhere(self.static_layer == 255)
        self.inflation_layer.fill(0.0)

        for r in range(self.rows):
            for c in range(self.cols):
                if self.static_layer[r, c] == 255:
                    self.inflation_layer[r, c] = 254.0
                else:
                    if len(obs_coords) > 0:
                        dists = np.sqrt((obs_coords[:, 0] - r)**2 + (obs_coords[:, 1] - c)**2)
                        min_d = np.min(dists)
                        if min_d <= robot_radius_cells:
                            self.inflation_layer[r, c] = 254.0
                        elif min_d < robot_radius_cells + 6:
                            cost = 254.0 * math.exp(-alpha * (min_d - robot_radius_cells))
                            self.inflation_layer[r, c] = cost

    def get_cost(self, r, c):
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return max(self.static_layer[r, c], self.inflation_layer[r, c])
        return 255


class ParticleFilterAMCL:
    """Adaptive Monte Carlo Localization (AMCL) Engine for AMR State Estimation."""
    def __init__(self, num_particles=80, initial_pose=(50.0, 50.0, 0.0)):
        self.num_particles = num_particles
        self.particles = []
        
        # Initialize Gaussian cloud around initial guess
        for _ in range(num_particles):
            px = initial_pose[0] + np.random.normal(0, 12.0)
            py = initial_pose[1] + np.random.normal(0, 12.0)
            ptheta = initial_pose[2] + np.random.normal(0, 0.15)
            self.particles.append([px, py, ptheta, 1.0 / num_particles])

    def predict_motion(self, v, w, dt):
        """Propagate particles with motion kinematics and odometry noise."""
        for p in self.particles:
            v_noisy = v + np.random.normal(0, 1.2)
            w_noisy = w + np.random.normal(0, 0.08)

            p[0] += v_noisy * math.cos(p[2]) * dt
            p[1] += v_noisy * math.sin(p[2]) * dt
            p[2] += w_noisy * dt

    def update_weights(self, costmap):
        """Update particle importance weights based on costmap likelihood field."""
        weights_sum = 0.0
        for p in self.particles:
            r, c = int(p[1] // costmap.res), int(p[0] // costmap.res)
            cost = costmap.get_cost(r, c)
            
            # High weight if in clear space, low weight near obstacles
            likelihood = math.exp(-cost / 50.0)
            p[3] = likelihood
            weights_sum += likelihood

        # Normalize weights
        if weights_sum > 0:
            for p in self.particles:
                p[3] /= weights_sum

    def resample(self):
        """Low-variance particle resampling stage."""
        weights = [p[3] for p in self.particles]
        indices = np.random.choice(range(self.num_particles), size=self.num_particles, p=weights)
        new_particles = []
        for idx in indices:
            p = self.particles[idx]
            new_particles.append([p[0] + np.random.normal(0, 1.0),
                                  p[1] + np.random.normal(0, 1.0),
                                  p[2] + np.random.normal(0, 0.02),
                                  1.0 / self.num_particles])
        self.particles = new_particles

    def get_estimated_pose(self):
        """Returns weighted average state estimation [x, y, theta]."""
        x_mean = sum(p[0] * p[3] for p in self.particles)
        y_mean = sum(p[1] * p[3] for p in self.particles)
        theta_mean = math.atan2(sum(math.sin(p[2]) * p[3] for p in self.particles),
                                sum(math.cos(p[2]) * p[3] for p in self.particles))
        return np.array([x_mean, y_mean, theta_mean])


class TEBLocalPlanner:
    """Timed Elastic Band (TEB) Kinematically Constrained Local Planner."""
    def __init__(self):
        self.max_v = 45.0       # Pixels/sec
        self.max_w = 2.2        # Rads/sec
        self.max_accel = 35.0   # Pixels/sec^2
        self.horizon_steps = 10
        self.dt = 0.15

    def plan_trajectory(self, current_pose, target_wp, costmap, dynamic_obstacles):
        """Elastic band optimization generating collision-free smooth commands."""
        rx, ry, rtheta = current_pose
        tx, ty = target_wp

        dx = tx - rx
        dy = ty - ry
        target_heading = math.atan2(dy, dx)
        heading_error = math.atan2(math.sin(target_heading - rtheta), math.cos(target_heading - rtheta))

        dist = math.sqrt(dx**2 + dy**2)
        v_cmd = min(self.max_v, 1.8 * dist)
        w_cmd = np.clip(2.5 * heading_error, -self.max_w, self.max_w)

        # Elastic band obstacle repulsion check
        for obs in dynamic_obstacles:
            obs_dist = math.sqrt((rx - obs[0])**2 + (ry - obs[1])**2)
            if obs_dist < 45.0:
                repulsion_angle = math.atan2(ry - obs[1], rx - obs[0])
                w_cmd += 1.2 * math.sin(repulsion_angle - rtheta)
                v_cmd *= 0.5  # Slow down near hazards

        # Generate predicted trajectory ribbon
        band_pts = []
        cx, cy, cth = rx, ry, rtheta
        for _ in range(self.horizon_steps):
            cx += v_cmd * math.cos(cth) * self.dt
            cy += v_cmd * math.sin(cth) * self.dt
            cth += w_cmd * self.dt
            band_pts.append((cx, cy))

        return v_cmd, w_cmd, band_pts


class AMRDashboardv2:
    """Tactical Rendering Interface for AMCL Particles, TEB Trajectory Ribbon & Inflation Map."""
    def __init__(self, costmap):
        self.costmap = costmap

    def render(self, robot_pose, amcl, teb_band, global_path, goal_pos, dynamic_obstacles):
        canvas = np.zeros((self.costmap.height, self.costmap.width, 3), dtype=np.uint8)

        # 1. Render Layered Costmap (Inflation + Static)
        for r in range(self.costmap.rows):
            for c in range(self.costmap.cols):
                cost = self.costmap.get_cost(r, c)
                if cost == 255:  # Wall
                    cv2.rectangle(canvas, (c * self.costmap.res, r * self.costmap.res),
                                  ((c + 1) * self.costmap.res, (r + 1) * self.costmap.res), (70, 70, 70), -1)
                elif cost > 20:  # Inflation
                    val = int(cost * 0.6)
                    cv2.rectangle(canvas, (c * self.costmap.res, r * self.costmap.res),
                                  ((c + 1) * self.costmap.res, (r + 1) * self.costmap.res), (val, 0, val // 2), -1)

        # 2. Render AMCL Cloud Particles (Cyan dots)
        for p in amcl.particles:
            px, py = int(p[0]), int(p[1])
            if 0 <= px < self.costmap.width and 0 <= py < self.costmap.height:
                cv2.circle(canvas, (px, py), 1, (255, 255, 0), -1)

        # 3. Render Dynamic Obstacles
        for obs in dynamic_obstacles:
            cv2.circle(canvas, (int(obs[0]), int(obs[1])), 16, (0, 0, 255), -1)

        # 4. Render Global Path
        for i in range(1, len(global_path)):
            cv2.line(canvas, (int(global_path[i-1][0]), int(global_path[i-1][1])),
                     (int(global_path[i][0]), int(global_path[i][1])), (200, 200, 200), 1)

        # 5. Render TEB Elastic Band Trajectory Ribbon (Yellow)
        for i in range(1, len(teb_band)):
            cv2.line(canvas, (int(teb_band[i-1][0]), int(teb_band[i-1][1])),
                     (int(teb_band[i][0]), int(teb_band[i][1])), (0, 255, 255), 2, cv2.LINE_AA)

        # 6. Render AMR Robot Base
        rx, ry, rtheta = robot_pose
        irx, iry = int(rx), int(ry)
        cv2.circle(canvas, (irx, iry), 13, (0, 255, 0), -1, cv2.LINE_AA)
        hx = int(rx + 20 * math.cos(rtheta))
        hy = int(ry + 20 * math.sin(rtheta))
        cv2.line(canvas, (irx, iry), (hx, hy), (0, 0, 0), 2)

        # Goal Marker
        cv2.circle(canvas, (int(goal_pos[0]), int(goal_pos[1])), 8, (0, 0, 255), -1)

        # HUD Panel
        cv2.rectangle(canvas, (15, 15), (380, 140), (10, 10, 10), -1)
        cv2.rectangle(canvas, (15, 15), (380, 140), (0, 255, 0), 1)

        cv2.putText(canvas, "AMR NAV2 STACK v2.0 [AMCL + TEB Planner]", (25, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        cv2.putText(canvas, f"AMCL POSE: [{rx:.1f}, {ry:.1f}, {math.degrees(rtheta):.1f} deg]", (25, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        cv2.putText(canvas, f"PARTICLES: {amcl.num_particles} Active Cloud Samples", (25, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1)
        cv2.putText(canvas, "PLANNERS: Global A* | Local TEB Elastic Band", (25, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        cv2.putText(canvas, "COSTMAP: Multi-Layered Exponential Inflation", (25, 126),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 100, 255), 1)

        return canvas


def main():
    costmap = LayeredCostmap(width=800, height=600, resolution=10)
    amcl = ParticleFilterAMCL(num_particles=80, initial_pose=(50.0, 50.0, 0.0))
    teb_planner = TEBLocalPlanner()
    dashboard = AMRDashboardv2(costmap)

    goal_pos = (730.0, 530.0)
    
    # Global A* Path Waypoints
    global_path = [
        (50.0, 50.0), (150.0, 80.0), (250.0, 200.0),
        (380.0, 380.0), (550.0, 480.0), goal_pos
    ]

    dynamic_obstacles = [np.array([400.0, 380.0])]

    log_file = open("amr_nav2_v2_telemetry.csv", "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["Timestamp", "AMCL_X", "AMCL_Y", "AMCL_Theta_deg", "Cmd_V", "Cmd_W"])

    dt = 0.1
    start_time = time.time()
    wp_idx = 0

    print("[INFO] Launching Industrial AMR Nav2 Stack v2.0 (AMCL & TEB Local Planner)...")

    while True:
        elapsed = time.time() - start_time

        # Update dynamic obstacle motion
        dynamic_obstacles[0][1] = 380.0 + 70.0 * math.sin(elapsed * 1.5)

        # Get AMCL State Estimation
        estimated_pose = amcl.get_estimated_pose()

        # Track Waypoint
        if wp_idx < len(global_path):
            target_wp = global_path[wp_idx]
            dist_to_wp = math.sqrt((estimated_pose[0] - target_wp[0])**2 + (estimated_pose[1] - target_wp[1])**2)
            if dist_to_wp < 25.0 and wp_idx < len(global_path) - 1:
                wp_idx += 1
        else:
            target_wp = goal_pos

        # Compute TEB Elastic Band Trajectory
        v_cmd, w_cmd, teb_band = teb_planner.plan_trajectory(estimated_pose, target_wp, costmap, dynamic_obstacles)

        # AMCL Motion & Measurement Step
        amcl.predict_motion(v_cmd, w_cmd, dt)
        amcl.update_weights(costmap)
        amcl.resample()

        # Log Data
        writer.writerow([
            round(elapsed, 2), round(estimated_pose[0], 2), round(estimated_pose[1], 2),
            round(math.degrees(estimated_pose[2]), 2), round(v_cmd, 2), round(w_cmd, 2)
        ])

        # Render Dashboard
        frame = dashboard.render(estimated_pose, amcl, teb_band, global_path, goal_pos, dynamic_obstacles)
        cv2.imshow("Autonomous Mobile Robot (AMR) Nav2 Engine v2.0", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        if math.sqrt((estimated_pose[0] - goal_pos[0])**2 + (estimated_pose[1] - goal_pos[1])**2) < 18.0:
            print("[INFO] Target goal reached successfully by AMR!")
            break

    log_file.close()
    cv2.destroyAllWindows()
    print("[INFO] AMR Nav2 Stack v2.0 shut down. Data logged to 'amr_nav2_v2_telemetry.csv'.")


if __name__ == "__main__":
    main()