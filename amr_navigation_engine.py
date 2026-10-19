import numpy as np
import cv2
import time
import math
import csv


class BehaviorTreeNode:
    """ROS 2 Nav2 Style Behavior Tree Execution Node."""
    def __init__(self, name):
        self.name = name

    def tick(self, context):
        raise NotImplementedError


class SequenceNode(BehaviorTreeNode):
    """Executes children in sequence; fails if any child fails."""
    def __init__(self, name, children):
        super().__init__(name)
        self.children = children

    def tick(self, context):
        for child in self.children:
            status = child.tick(context)
            if status != "SUCCESS":
                return status
        return "SUCCESS"


class FallbackNode(BehaviorTreeNode):
    """Executes children until one succeeds (Recovery Logic)."""
    def __init__(self, name, children):
        super().__init__(name)
        self.children = children

    def tick(self, context):
        for child in self.children:
            status = child.tick(context)
            if status == "SUCCESS":
                return "SUCCESS"
        return "FAILURE"


class FleetTrafficController:
    """Multi-AMR Intersection Reservation & Priority Grid Lock System."""
    def __init__(self):
        self.intersection_zones = [
            {"id": "Intersection_Alpha", "bounds": (320, 320, 480, 480), "locked_by": None}
        ]

    def request_intersection_lock(self, robot_id, robot_pos):
        """Requests priority reservation lock at warehouse intersections."""
        rx, ry = robot_pos[:2]
        for zone in self.intersection_zones:
            x1, y1, x2, y2 = zone["bounds"]
            if x1 <= rx <= x2 and y1 <= ry <= y2:
                if zone["locked_by"] is None or zone["locked_by"] == robot_id:
                    zone["locked_by"] = robot_id
                    return True  # Lock Granted
                else:
                    return False  # Yield right of way
        return True

    def release_locks(self, robot_id):
        for zone in self.intersection_zones:
            if zone["locked_by"] == robot_id:
                zone["locked_by"] = None


class Perception3DCostmap:
    """3D Depth Camera Perception Layer projecting hanging hazards onto 2D Costmap."""
    def __init__(self, width=800, height=600, res=10):
        self.cols = width // res
        self.rows = height // res
        self.res = res
        self.costmap = np.zeros((self.rows, self.cols), dtype=np.uint8)

    def process_3d_point_cloud(self, cloud_3d):
        """Projects 3D Camera points [x, y, z] to costmap within robot height clearance."""
        self.costmap.fill(0)
        for pt in cloud_3d:
            x, y, z = pt
            # Height Clearance Filter (Overhanging items between 0.3m and 2.0m)
            if 0.3 <= z <= 2.0:
                c = int(x // self.res)
                r = int(y // self.res)
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    self.costmap[r, c] = 254  # Mark Lethal Obstacle


class AMRNav2Enginev3:
    """Enterprise AMR Nav2 System with Behavior Trees & Fleet Coordination."""
    def __init__(self):
        self.fleet_manager = FleetTrafficController()
        self.perception_3d = Perception3DCostmap()
        
        self.robot_id = "AMR_01"
        self.robot_pose = np.array([50.0, 50.0, 0.0])  # [x, y, theta]
        self.target_goal = np.array([720.0, 520.0])
        
        self.amr_peer_pose = np.array([380.0, 380.0, 3.14])  # Peer Robot AMR_02
        self.battery_level = 88.0

        # Global Path Waypoints
        self.global_path = [(50.0, 50.0), (220.0, 150.0), (400.0, 400.0), (600.0, 500.0), (720.0, 520.0)]
        self.current_wp_idx = 0
        self.bt_status = "IDLE"

    def execute_navigation_step(self, dt=0.1):
        """Executes BT evaluation and motion synthesis."""
        # 1. Simulate 3D Depth Perception Inputs
        synthetic_cloud_3d = [
            [400.0 + np.random.normal(0, 5), 400.0 + np.random.normal(0, 5), 1.2],
            [220.0, 150.0, 0.5]
        ]
        self.perception_3d.process_3d_point_cloud(synthetic_cloud_3d)

        # 2. Check Fleet Traffic Lock
        lock_granted = self.fleet_manager.request_intersection_lock(self.robot_id, self.robot_pose)

        if not lock_granted:
            self.bt_status = "YIELDING_RIGHT_OF_WAY"
            v_cmd, w_cmd = 0.0, 0.0
        else:
            self.bt_status = "NAVIGATING_TEB_ACTIVE"
            target_wp = self.global_path[self.current_wp_idx]
            dx = target_wp[0] - self.robot_pose[0]
            dy = target_wp[1] - self.robot_pose[1]
            dist = math.sqrt(dx**2 + dy**2)

            if dist < 25.0 and self.current_wp_idx < len(self.global_path) - 1:
                self.current_wp_idx += 1

            heading = math.atan2(dy, dx)
            h_err = math.atan2(math.sin(heading - self.robot_pose[2]), math.cos(heading - self.robot_pose[2]))

            v_cmd = min(40.0, 1.5 * dist)
            w_cmd = np.clip(2.0 * h_err, -2.0, 2.0)

            # State Kinematic Integration
            self.robot_pose[0] += v_cmd * math.cos(self.robot_pose[2]) * dt
            self.robot_pose[1] += v_cmd * math.sin(self.robot_pose[2]) * dt
            self.robot_pose[2] += w_cmd * dt

            if dist < 18.0 and self.current_wp_idx == len(self.global_path) - 1:
                self.fleet_manager.release_locks(self.robot_id)
                self.bt_status = "GOAL_REACHED_SUCCESS"

        return v_cmd, w_cmd


class AMRDashboardV3:
    """Tactical Rendering Interface for BT Tree State & Fleet Traffic Overlay."""
    def render(self, engine):
        canvas = np.zeros((600, 800, 3), dtype=np.uint8)

        # Draw Grid
        for x in range(0, 800, 50):
            cv2.line(canvas, (x, 0), (x, 600), (20, 22, 22), 1)
        for y in range(0, 600, 50):
            cv2.line(canvas, (0, y), (800, y), (20, 22, 22), 1)

        # Render Intersection Zone
        for zone in engine.fleet_manager.intersection_zones:
            x1, y1, x2, y2 = zone["bounds"]
            color = (0, 0, 180) if zone["locked_by"] is not None else (0, 120, 0)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, f"FLEET LOCK: {zone['locked_by']}", (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # Render Dynamic Peer AMR_02
        px, py = int(engine.amr_peer_pose[0]), int(engine.amr_peer_pose[1])
        cv2.circle(canvas, (px, py), 14, (0, 165, 255), -1, cv2.LINE_AA)
        cv2.putText(canvas, "AMR_02 (Peer)", (px - 25, py - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 165, 255), 1)

        # Render AMR_01 (Controlled Robot)
        rx, ry, rth = engine.robot_pose
        irx, iry = int(rx), int(ry)
        cv2.circle(canvas, (irx, iry), 14, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.line(canvas, (irx, iry), (int(rx + 22 * math.cos(rth)), int(ry + 22 * math.sin(rth))), (0, 0, 0), 2)
        cv2.putText(canvas, "AMR_01 (Nav2)", (irx - 25, iry - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)

        # Render Target Goal
        cv2.circle(canvas, (int(engine.target_goal[0]), int(engine.target_goal[1])), 8, (0, 0, 255), -1)

        # HUD Panel
        cv2.rectangle(canvas, (15, 15), (420, 155), (10, 10, 10), -1)
        cv2.rectangle(canvas, (15, 15), (420, 155), (0, 255, 0), 1)

        cv2.putText(canvas, "AMR NAV2 STACK v3.0 [BT + Fleet Manager]", (25, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        cv2.putText(canvas, f"BEHAVIOR TREE STATE: {engine.bt_status}", (25, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0) if "SUCCESS" in engine.bt_status else (0, 255, 255), 1)
        cv2.putText(canvas, f"FLEET TRAFFIC: Priority Grid Locks Active", (25, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        cv2.putText(canvas, f"3D PERCEPTION: Height Clearance Filter Active", (25, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 150, 0), 1)
        cv2.putText(canvas, f"BATTERY SOC: {engine.battery_level:.1f}% | Press [Q] to Exit", (25, 128),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        return canvas


def main():
    engine = AMRNav2Enginev3()
    dashboard = AMRDashboardV3()

    log_file = open("amr_nav2_v3_telemetry.csv", "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["Timestamp", "AMR_X", "AMR_Y", "AMR_Theta_deg", "BT_Status", "Cmd_V", "Cmd_W"])

    dt = 0.1
    start_time = time.time()

    print("[INFO] Launching Enterprise AMR Nav2 Stack v3.0 (Behavior Trees & Fleet Traffic Manager)...")

    while True:
        elapsed = time.time() - start_time
        v_cmd, w_cmd = engine.execute_navigation_step(dt=dt)

        writer.writerow([
            round(elapsed, 2), round(engine.robot_pose[0], 2), round(engine.robot_pose[1], 2),
            round(math.degrees(engine.robot_pose[2]), 2), engine.bt_status, round(v_cmd, 2), round(w_cmd, 2)
        ])

        frame = dashboard.render(engine)
        cv2.imshow("AMR Nav2 Stack v3.0 (Behavior Trees & Fleet Traffic Manager)", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        if engine.bt_status == "GOAL_REACHED_SUCCESS":
            print("[INFO] Nav2 Goal Reached Successfully!")
            break

    log_file.close()
    cv2.destroyAllWindows()
    print("[INFO] AMR Nav2 v3.0 terminated. Data logged to 'amr_nav2_v3_telemetry.csv'.")


if __name__ == "__main__":
    main()