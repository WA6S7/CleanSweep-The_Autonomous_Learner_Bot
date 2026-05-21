#!/usr/bin/env python3
 
import math
import sys
import time
from pathlib import Path
 
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


 
ROOT = Path.home() / "thesis_ws" / "src" / "clean_sweep_rl_v2"
sys.path.insert(0, str(ROOT))
 
from coverage_env.room         import load_room_grid, world_to_cell, cell_to_world
from coverage_env.coverage_env import CoverageEnv
from q_learning.q_agent        import QLearningAgent
 
CELL_SIZE_M       = 0.20
LINEAR_SPEED      = 0.15
LINEAR_SPEED_MAX  = 0.25
ANGULAR_SPEED     = 1.00
MIN_TURN_SPEED    = 0.8
OBSTACLE_DIST_M   = 0.25
POSITION_TOL_M    = 0.06
HEADING_TOL_RAD   = 0.10
CMD_HZ            = 20
ARRIVAL_TIMEOUT_S = 65.0
 
 
_ODOM_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=10)
 

_SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10)
 
 
def yaw_from_quat(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)
 
 
def angle_diff(a, b):
    d = a - b
    while d >  math.pi: d -= 2 * math.pi
    while d < -math.pi: d += 2 * math.pi
    return d

 
class CoverageExecutorGazebo(Node):
 
    def __init__(self):
        super().__init__("coverage_executor")
 
        self.declare_parameter("qtable_path",
            str(ROOT / "results_20cm_cells" / "q_table_final.pkl"))
        self.declare_parameter("cell_size", CELL_SIZE_M)
        self.declare_parameter("sdf_path",  str(ROOT / "maps" / "env1_bedroom.sdf"))
        self.declare_parameter("odom_topic", "/ground_truth_pose")

        # Hardware needs ArUco for absolute fix
        # Gazebo publishes ground-truth pose directly, so no wait needed.
        self.declare_parameter("require_aruco", False)

        self.qtable_path   = self.get_parameter("qtable_path").value
        self.cell_size     = self.get_parameter("cell_size").value
        sdf_path           = self.get_parameter("sdf_path").value
        odom_topic         = self.get_parameter("odom_topic").value
        self.require_aruco = self.get_parameter("require_aruco").value

        self.x = 0.0; self.y = 0.0; self.yaw = 0.0
        self.front_dist    = 999.0
        self.rear_dist     = 999.0
        self.odom_received = False
        self.aruco_received = False

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", _ODOM_QOS)

        self.create_subscription(Odometry, odom_topic, self._odom_cb, _SENSOR_QOS)
        self.get_logger().info(f"Subscribing to pose on: {odom_topic}")
        self.create_subscription(LaserScan, "/ultrasonic_front",  self._front_cb,  _SENSOR_QOS)
        self.create_subscription(LaserScan, "/ultrasonic_rear",   self._rear_cb,   _SENSOR_QOS)
        self.create_subscription(PoseWithCovarianceStamped, "/aruco/pose",
                                 self._aruco_cb, _SENSOR_QOS)
 
        # Load room grid
        self.wall_grid, self.meta = load_room_grid(
            Path(sdf_path), cell_size=self.cell_size)
        
        # Create cell env
        self.env = CoverageEnv(self.wall_grid, cell_size_m=self.cell_size,
                               n_objects=0)
        
        # Load Q-Learning policy
        self.agent = QLearningAgent(n_actions=self.env.N_ACTIONS)
        self.agent.load(self.qtable_path)
        self.agent.epsilon = 0.0
 
        self.get_logger().info(
            f"CoverageExecutorGazebo ready  "
            f"grid={self.env.rows}×{self.env.cols}  "
            f"strips={self.env.n_strips}  cell={self.cell_size*100:.0f}cm")
 
    # --- Callbacks ---
 
    def _odom_cb(self, msg):
        self.x   = msg.pose.pose.position.x
        self.y   = msg.pose.pose.position.y
        self.yaw = yaw_from_quat(msg.pose.pose.orientation)
        self.odom_received = True
 
    def _front_cb(self, msg):
        valid = [r for r in msg.ranges
                 if msg.range_min <= r <= msg.range_max and math.isfinite(r)]
        self.front_dist = min(valid) if valid else 999.0
 
    def _rear_cb(self, msg):
        valid = [r for r in msg.ranges
                 if msg.range_min <= r <= msg.range_max and math.isfinite(r)]
        self.rear_dist = min(valid) if valid else 999.0

    def _aruco_cb(self, msg):
        self.aruco_received = True

    # --- Motion helpers ---
 
    def _publish_vel(self, linear, angular):
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.cmd_pub.publish(msg)

    def _stop(self):
        self._publish_vel(0.0, 0.0)
 
    def _spin_once(self):
        rclpy.spin_once(self, timeout_sec=0.0)
 
    def _current_cell(self):
        return world_to_cell(self.x, self.y, self.meta)
 
    def _spin_for(self, duration_s):
        """Spin callbacks for duration_s without blocking — replaces time.sleep."""
        end = time.time() + duration_s
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.005)
 
    def _turn_to_heading(self, target_yaw):
        t0 = time.time()
        settled_since = None
        yaw_start = self.yaw
        last_log = t0
        while time.time() - t0 < 10.0:
            rclpy.spin_once(self, timeout_sec=0.01)
            err = angle_diff(target_yaw, self.yaw)
            now = time.time()
            if now - last_log > 0.5:
                self.get_logger().info(
                    f"  turn: target={math.degrees(target_yaw):+6.1f}°  "
                    f"yaw={math.degrees(self.yaw):+6.1f}°  "
                    f"err={math.degrees(err):+6.1f}°  "
                    f"Δyaw={math.degrees(angle_diff(self.yaw, yaw_start)):+6.1f}°")
                last_log = now
            if abs(err) < HEADING_TOL_RAD:
                if settled_since is None:
                    settled_since = now
                elif now - settled_since > 0.15:
                    self._stop()
                    self._spin_for(0.1)
                    return True
            else:
                settled_since = None
            scale    = min(1.0, abs(err) / (math.pi / 6))
            w_mag    = max(MIN_TURN_SPEED, ANGULAR_SPEED * scale)
            w        = math.copysign(w_mag, err)
            self._publish_vel(0.0, w)
        self._stop()
        self.get_logger().warn(
            f"  TURN TIMEOUT after 10s  "
            f"target={math.degrees(target_yaw):+6.1f}°  "
            f"final_yaw={math.degrees(self.yaw):+6.1f}°  "
            f"Δyaw={math.degrees(angle_diff(self.yaw, yaw_start)):+6.1f}° "
            f"(if Δyaw≈0 → IMU is not updating!)")
        return False
 
    def _drive_straight(self, dist_m):
        locked_yaw = self.yaw
        x0, y0 = self.x, self.y
        while math.hypot(self.x - x0, self.y - y0) < dist_m:
            rclpy.spin_once(self, timeout_sec=0.01)
            if self.front_dist < OBSTACLE_DIST_M:
                self.get_logger().warn(
                    f"Front obstacle at {self.front_dist:.2f}m — stopping")
                self._stop()
                return False
            # Slow down when approaching obstacles
            speed = LINEAR_SPEED + (LINEAR_SPEED_MAX - LINEAR_SPEED) * min(
                1.0, max(0.0, (self.front_dist - 0.40) / 0.60))
            heading_err = angle_diff(locked_yaw, self.yaw)
            w = max(-ANGULAR_SPEED * 0.3,
                    min(ANGULAR_SPEED * 0.3, heading_err * 2.0))
            self._publish_vel(speed, w)
        self._stop()
        self._spin_for(0.05)
        return True
 
    def drive_to_cell(self, row, col):
        tx, ty  = cell_to_world(row, col, self.meta)
        t_start = time.time()
        while True:
            self._spin_once()
            if time.time() - t_start > ARRIVAL_TIMEOUT_S:
                self._stop()
                self.get_logger().warn(f"Timeout -> ({row},{col})")
                return False
            dx, dy = tx - self.x, ty - self.y
            dist   = math.hypot(dx, dy)
            if dist < POSITION_TOL_M:
                self._stop(); return True
            if self.front_dist < OBSTACLE_DIST_M:
                self._stop()
                self.get_logger().warn(
                    f"Obstacle/wall {self.front_dist:.2f}m before ({row},{col})")
                return False
            if not self._turn_to_heading(math.atan2(dy, dx)):
                self.get_logger().warn(
                    f"Turn failed -> ({row},{col}) — aborting cell")
                return False
            self._drive_straight(min(dist, CELL_SIZE_M))
 
    def backup_one_cell(self):
        self.get_logger().info(f"Backing up (rear={self.rear_dist:.2f}m)...")
        t0 = time.time()
        while time.time() - t0 < CELL_SIZE_M / LINEAR_SPEED:
            rclpy.spin_once(self, timeout_sec=0.01)
            if self.rear_dist < OBSTACLE_DIST_M:
                self.get_logger().warn(
                    f"Rear obstacle at {self.rear_dist:.2f}m — stopping backup")
                break
            self._publish_vel(-LINEAR_SPEED, 0.0)
            time.sleep(1.0 / CMD_HZ)
        self._stop()
        self._spin_for(0.1)
    
    def _next_cell_is_wall(self, row, col, target_row, target_col):
        dr = target_row - row
        dc = target_col - col

        # Normalise to unit step
        if dr != 0: dr = dr // abs(dr)
        if dc != 0: dc = dc // abs(dc)
        nr, nc = row + dr, col + dc
        if 0 <= nr < self.env.rows and 0 <= nc < self.env.cols:
            return bool(self.wall_grid[nr, nc] == 1)
        return True  # If it is out of bounds, it should be treated as a wall
     

    def _body_sweep(self, heading):
        offsets = [
            ("L45", heading + math.radians(45)),
            ("center", heading),
            ("R45", heading - math.radians(45)),
        ]
        readings = []
        for label, yaw in offsets:
            self._turn_to_heading(yaw)
            self.front_dist = 999.0  # clear stale mid-turn reading
            self._spin_for(0.20)
            readings.append((label, self.front_dist))
        self._turn_to_heading(heading)
        self.get_logger().info(
            f"  Sweep: " + ", ".join(f"{l}={d:.2f}m" for l, d in readings))
        return readings

    def _quick_check(self, heading):
        # Face target and read center ultrasonic only, without sweeping
        self._turn_to_heading(heading)
        self.front_dist = 999.0  # clear stale mid-turn reading
        self._spin_for(0.20)
        return self.front_dist

    def _preflight_check(self, target_row, target_col, full_sweep=False):
        tx, ty = cell_to_world(target_row, target_col, self.meta)
        dx, dy = tx - self.x, ty - self.y
        heading = math.atan2(dy, dx)

        cr, cc = self._current_cell()
        if self._next_cell_is_wall(cr, cc, target_row, target_col):
            self._turn_to_heading(heading)
            return True

        if full_sweep:
            readings = self._body_sweep(heading)
            min_dist = min(r[1] for r in readings)
            if min_dist < self.cell_size * 1.2:
                close = [f"{l}={d:.2f}m" for l, d in readings
                         if d < self.cell_size * 1.2]
                self.get_logger().info(
                    f"  Preflight: obstacle at ({target_row},{target_col}) "
                    f"[{', '.join(close)}]")
                self.env.live_grid[target_row, target_col] = 1
                self.env.discovered_grid[target_row, target_col] = 1
                self.env.n_free = int((self.env.live_grid == 0).sum())
                return False
        else:
            dist = self._quick_check(heading)
            if dist < self.cell_size * 1.2:
                self.get_logger().info(
                    f"  Quick check: obstacle {dist:.2f}m at ({target_row},{target_col})")
                self.env.live_grid[target_row, target_col] = 1
                self.env.discovered_grid[target_row, target_col] = 1
                self.env.n_free = int((self.env.live_grid == 0).sum())
                return False
        return True

    # --- Coverage ---
 
    def run_coverage(self):
        self.get_logger().info("Using native Gazebo odometry (no world offset needed).")
 
        # Wait for EKF output
        self.get_logger().info("Waiting for /odom...")
        t_wait = time.time()
        next_log = 5.0
        while not self.odom_received:
            rclpy.spin_once(self, timeout_sec=0.5)
            elapsed = time.time() - t_wait
            if elapsed > 30.0:
                self.get_logger().error(
                    "/odometry/filtered not received after 30s. "
                    "Is ekf_filter_node running? "
                    "Check: ros2 topic hz /odometry/filtered")
                return
            if elapsed >= next_log:
                self.get_logger().info(f"Still waiting... ({elapsed:.0f}s)")
                next_log += 5.0

        # Wait for first ArUco fix (EKF starts at 0,0)
        # This is only required on the hardware as Gazebo ground-truth pose is absolute
        if self.require_aruco:
            self.get_logger().info("Waiting for first ArUco fix...")
            t_wait = time.time()
            next_log = 5.0
            while not self.aruco_received:
                rclpy.spin_once(self, timeout_sec=0.5)
                elapsed = time.time() - t_wait
                if elapsed > 30.0:
                    self.get_logger().warn(
                        "No ArUco fix after 30s — starting with dead-reckoning only.")
                    break
                if elapsed >= next_log:
                    self.get_logger().info(
                        f"Still waiting for ArUco... ({elapsed:.0f}s)")
                    next_log += 5.0
        else:
            self.get_logger().info(
                "require_aruco=False — skipping ArUco wait (simulation mode).")
            
        # Let EKF settle after ArUco correction
        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.05)

        self.get_logger().info(
            f"Ready. World pos: ({self.x:.3f}, {self.y:.3f})")

        # Initialise environment
        self.env.live_grid[:]       = self.wall_grid.copy()
        self.env.visited[:]         = 0
        self.env.visit_count[:]     = 0
        self.env.discovered_grid[:] = self.wall_grid.copy()
        self.env.nb_steps           = 0
        self.env._strip_visited     = [False] * self.env.n_strips
        self.env.free_cells         = [(r,c) for r in range(self.env.rows)
                                              for c in range(self.env.cols)
                                              if self.wall_grid[r,c] == 0]
        self.env.n_free = len(self.env.free_cells)
        self.env.path   = []
 
        sr, sc = self._current_cell()
        self.env.pos            = (sr, sc)
        self.env._current_strip = self.env._strip_for_pos((sr, sc))
        self.env._entry_side    = 0
        state = self.env.get_state_key()
 
        self.get_logger().info(
            f"Starting from cell ({sr},{sc})  "
            f"strips={self.env.n_strips}  floor={self.env.n_free}")
 
        total_covered = 0
 
        while True:
            valid = self.env.unvisited_strips()
            if not valid: break
 
            action = self.agent.select_action(state, valid_actions=valid, greedy=True)
            strip  = self.env.strips[action]
            self.get_logger().info(
                f"→ Strip {action}  rows {strip.rows[0]}–{strip.rows[-1]}")
 
            # Navigate to nearest free cell in strip
            cr, cc = self._current_cell()
            br, bc, bd = strip.rows[0], strip.col_min, float("inf")
            for srow in strip.rows:
                for scol in range(self.env.cols):
                    if self.wall_grid[srow, scol] == 0:
                        d = abs(srow-cr) + abs(scol-cc)
                        if d < bd:
                            bd, br, bc = d, srow, scol
            if not self._preflight_check(br, bc, full_sweep=True):
                self.get_logger().warn(
                    f"Strip entry ({br},{bc}) blocked — skipping strip {action}")
                self.env._strip_visited[action] = True
                state = self.env.get_state_key()
                continue
            self.drive_to_cell(br, bc)
 
            # Boustrophedon sweep
            go_right = True
            strip_covered = 0
            for row in strip.rows:
                cols = sorted([c for c in range(self.env.cols)
                               if self.wall_grid[row, c] == 0],
                              reverse=not go_right)
                for ci, col in enumerate(cols):
                    if self.env.discovered_grid[row, col] == 1:
                        continue
                    # Full sweep at start of each row, quick check mid-row
                    sweep = (ci == 0)
                    if not self._preflight_check(row, col, full_sweep=sweep):
                        continue
                    ok = self.drive_to_cell(row, col)

                    if ok:
                        self.env.visited[row, col]      = 1
                        self.env.visit_count[row, col] += 1
                        strip_covered += 1
                        total_covered += 1
                        self.env.path.append((row, col))
                    else:
                        self.env.live_grid[row, col]       = 1
                        self.env.discovered_grid[row, col] = 1
                        self.env.n_free = int((self.env.live_grid == 0).sum())
                        self.backup_one_cell()
                go_right = not go_right
 
            self.env._strip_visited[action] = True
            self.env._current_strip         = action
            state = self.env.get_state_key()
            cov   = total_covered / max(self.env.n_free, 1)
            self.get_logger().info(
                f"  Strip {action} done  covered={strip_covered}  "
                f"total={cov*100:.1f}%")
 
        self._stop()
        final_cov = total_covered / max(self.env.n_free, 1)
        self.get_logger().info(
            f"Done! {total_covered}/{self.env.n_free} = {final_cov*100:.1f}%")
 
 
def main(args=None):
    rclpy.init(args=args)
    node = CoverageExecutorGazebo()
    try:
        node.run_coverage()
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted")
    finally:
        node._stop()
        node.destroy_node()
        rclpy.shutdown()
 
 
if __name__ == "__main__":
    main()