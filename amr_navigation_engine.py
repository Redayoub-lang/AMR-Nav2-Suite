import numpy as np
import cv2
import time
import math
import csv
from heapq import heappush, heappop


class GridOccupancyMap:
    """Manages 2D Spatial Occupancy Grid and Obstacle Layer."""
    def __init__(self, width=800, height=600, resolution=10):
        self.width = width
        self.height = height
        self.res = resolution  # Pixels per cell
        self.cols = width // resolution
        self.rows = height // resolution
        self.grid = np.zeros((self.rows, self.cols), dtype=np.uint8)

        # Static Obstacles (Walls and Boundaries)
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1
        
        # Internal Obstacle Blocks
        self.add_obstacle_box(15, 10, 25, 30)
        self.add_obstacle_box(45, 20, 55, 45)

    def add_obstacle_box(self, r1, c1, r2, c2):
        """Add a rectangular block obstacle."""
        self.grid[r1:r2, c1:c2] = 1

    def is_blocked(self, row, col):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.grid[row, col] == 1
        return True


class GlobalAStarPlanner:
    """Global Path Planner utilizing A* Heuristic Graph Search Algorithm."""
    def __init__(self, grid_map):
        self.map = grid_map

    def heuristic(self, a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def plan_path(self, start_pos, goal_pos):
        """Compute shortest collision-free grid path from Start to Goal."""
        start = (int(start_pos[1] // self.map.res), int(start_pos[0] // self.map.res))
        goal = (int(goal_pos[1] // self.map.res), int(goal_pos[0] // self.map.res))

        open_set = []
        heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        # 8-connected grid movement
        neighbors = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]

        while open_set:
            _, current = heappop(open_set)

            if current == goal:
                # Reconstruct Path
                path = []
                while current in came_from:
                    path.append((current[1] * self.map.res + self.map.res // 2,
                                 current[0] * self.map.res + self.map.res // 2))
                    current = came_from[current]
                path.append(start_pos)
                return path[::-1]

            for dr, dc in neighbors:
                neighbor = (current[0] + dr, current[1] + dc)
                if self.map.is_blocked(neighbor[0], neighbor[1]):
                    continue

                move_cost = 1.414 if dr != 0 and dc != 0 else 1.0
                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))

        return []  # No path found


class DWALocalPlanner:
    """Dynamic Window Approach (DWA) for Dynamic Obstacle Avoidance and Trajectory Rollout."""
    def __init__(self):
        self.max_speed = 40.0       # Pixels/sec
        self.max_yaw_rate = 2.0     # Rads/sec
        self.max_accel = 30.0      # Pixels/sec^2
        self.max_dyaw = 3.0        # Rads/sec^2
        self.dt = 0.1
        self.predict_time = 1.5

    def compute_velocity_commands(self, robot_pose, current_vel, target_waypoint, obstacles):
        """Evaluates dynamic window and selects optimal (v, omega) command."""
        rx, ry, rtheta = robot_pose
        cv, cw = current_vel

        # 1. Calculate Dynamic Window Limits
        vs = [0, self.max_speed, -self.max_yaw_rate, self.max_yaw_rate]
        vd = [cv - self.max_accel * self.dt, cv + self.max_accel * self.dt,
              cw - self.max_dyaw * self.dt, cw + self.max_dyaw * self.dt]

        dw = [max(vs[0], vd[0]), min(vs[1], vd[1]), max(vs[2], vd[2]), min(vs[3], vd[3])]

        best_v, best_w = 0.0, 0.0
        best_score = -float('inf')

        # 2. Evaluate Trajectory Rollouts
        for v in np.linspace(dw[0], dw[1], 8):
            for w in np.linspace(dw[2], dw[3], 12):
                # Predict trajectory
                traj = []
                x, y, theta = rx, ry, rtheta
                for _ in range(int(self.predict_time / self.dt)):
                    x += v * math.cos(theta) * self.dt
                    y += v * math.sin(theta) * self.dt
                    theta += w * self.dt
                    traj.append((x, y))

                # Objective Function Evaluation
                heading = -math.atan2(target_waypoint[1] - traj[-1][1], target_waypoint[0] - traj[-1][0])
                heading_score = math.pi - abs(heading - traj[-1][1]) if len(traj) > 0 else 0

                min_dist = float('inf')
                for obs in obstacles:
                    for tx, ty in traj:
                        dist = math.sqrt((tx - obs[0])**2 + (ty - obs[1])**2)
                        if dist < min_dist:
                            min_dist = dist

                if min_dist < 15.0:  # Collision threshold
                    continue

                score = 0.6 * heading_score + 0.3 * min_dist + 0.1 * v
                if score > best_score:
                    best_score = score
                    best_v, best_w = v, w

        return best_v, best_w


class AutonomousRobot:
    """Differential Drive Autonomous Mobile Robot (AMR)."""
    def __init__(self, start_pos=(50.0, 50.0, 0.0)):
        self.x, self.y, self.theta = start_pos
        self.v = 0.0
        self.w = 0.0
        self.radius = 12.0

    def update_odometry(self, v_cmd, w_cmd, dt):
        """Update robot state via differential drive kinematics."""
        self.v = v_cmd
        self.w = w_cmd
        self.x += self.v * math.cos(self.theta) * dt
        self.y += self.v * math.sin(self.theta) * dt
        self.theta += self.w * dt


class AMRDashboard:
    """Renders Real-time Navigation Map, Occupancy Grid, and Control HUD."""
    def __init__(self, grid_map):
        self.map = grid_map

    def render(self, robot, global_path, target_wp, dynamic_obstacles):
        canvas = np.zeros((self.map.height, self.map.width, 3), dtype=np.uint8)

        # 1. Render Grid Occupancy Map
        for r in range(self.map.rows):
            for c in range(self.map.cols):
                if self.map.grid[r, c] == 1:
                    cv2.rectangle(canvas, (c * self.map.res, r * self.map.res),
                                  ((c + 1) * self.map.res, (r + 1) * self.map.res), (60, 60, 60), -1)

        # 2. Render Dynamic Obstacles
        for obs in dynamic_obstacles:
            cv2.circle(canvas, (int(obs[0]), int(obs[1])), 15, (0, 0, 255), -1)

        # 3. Render Global Planned Path
        for i in range(1, len(global_path)):
            cv2.line(canvas, (int(global_path[i-1][0]), int(global_path[i-1][1])),
                     (int(global_path[i][0]), int(global_path[i][1])), (255, 255, 0), 2)

        # 4. Render Active Waypoint
        if target_wp is not None:
            cv2.circle(canvas, (int(target_wp[0]), int(target_wp[1])), 6, (0, 255, 255), -1)

        # 5. Render AMR Chassis and Orientation Line
        rx, ry = int(robot.x), int(robot.y)
        cv2.circle(canvas, (rx, ry), int(robot.radius), (0, 255, 0), -1)
        hx = int(rx + (robot.radius + 10) * math.cos(robot.theta))
        hy = int(ry + (robot.radius + 10) * math.sin(robot.theta))
        cv2.line(canvas, (rx, ry), (hx, hy), (0, 0, 0), 2)

        # 6. Render Control HUD Overlay
        cv2.rectangle(canvas, (15, 15), (340, 115), (10, 10, 10), -1)
        cv2.rectangle(canvas, (15, 15), (340, 115), (0, 255, 0), 1)

        cv2.putText(canvas, "AMR NAVIGATION: Nav2 Stack Engine", (25, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        cv2.putText(canvas, f"POSE (X,Y,Theta): [{robot.x:.1f}, {robot.y:.1f}, {math.degrees(robot.theta):.1f}deg]",
                    (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        cv2.putText(canvas, f"CMD VEL (Linear, Angular): [{robot.v:.1f} px/s, {robot.w:.2f} rad/s]",
                    (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        cv2.putText(canvas, "PLANNERS: Global A* | Local DWA", (25, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)

        return canvas


def main():
    grid_map = GridOccupancyMap(width=800, height=600, resolution=10)
    global_planner = GlobalAStarPlanner(grid_map)
    local_planner = DWALocalPlanner()
    robot = AutonomousRobot(start_pos=(50.0, 50.0, 0.0))
    dashboard = AMRDashboard(grid_map)

    goal_pos = (730.0, 530.0)
    global_path = global_planner.plan_path((robot.x, robot.y), goal_pos)

    # Dynamic Obstacle Position
    dynamic_obstacles = [np.array([400.0, 300.0])]

    log_file = open("amr_navigation_log.csv", "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["Timestamp", "Robot_X", "Robot_Y", "Theta_deg", "Cmd_Linear_V", "Cmd_Angular_W"])

    dt = 0.1
    start_time = time.time()
    wp_index = 0

    print("[INFO] Starting AMR Navigation Engine (A* Global + DWA Local)...")

    while True:
        elapsed = time.time() - start_time

        # Update dynamic obstacle motion
        dynamic_obstacles[0][1] = 300.0 + 80.0 * math.sin(elapsed * 1.2)

        # Waypoint tracking selection
        if wp_index < len(global_path):
            target_wp = global_path[wp_index]
            dist_to_wp = math.sqrt((robot.x - target_wp[0])**2 + (robot.y - target_wp[1])**2)
            if dist_to_wp < 20.0 and wp_index < len(global_path) - 1:
                wp_index += 1
        else:
            target_wp = goal_pos

        # Compute Local DWA Velocity Setpoints
        v_cmd, w_cmd = local_planner.compute_velocity_commands(
            (robot.x, robot.y, robot.theta), (robot.v, robot.w), target_wp, dynamic_obstacles
        )

        # Kinematic Motion Step
        robot.update_odometry(v_cmd, w_cmd, dt)

        # Log Data
        writer.writerow([
            round(elapsed, 2), round(robot.x, 2), round(robot.y, 2),
            round(math.degrees(robot.theta), 2), round(v_cmd, 2), round(w_cmd, 2)
        ])

        # Render Interface
        frame = dashboard.render(robot, global_path, target_wp, dynamic_obstacles)
        cv2.imshow("Autonomous Mobile Robot (AMR) Nav2 Engine", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        # Check goal reach
        if math.sqrt((robot.x - goal_pos[0])**2 + (robot.y - goal_pos[1])**2) < 15.0:
            print("[INFO] Goal reached successfully!")
            break

    log_file.close()
    cv2.destroyAllWindows()
    print("[INFO] AMR Navigation System shut down. Log saved to 'amr_navigation_log.csv'.")


if __name__ == "__main__":
    main()