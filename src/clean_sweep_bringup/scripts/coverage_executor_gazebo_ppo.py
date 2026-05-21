#!/usr/bin/env python3

import math
import sys
import time
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

ROOT = Path.home() / "thesis_ws" / "src" / "clean_sweep_rl_v2"
sys.path.insert(0, str(ROOT))

from coverage_env.room import load_room_grid, world_to_cell, cell_to_world
from ppo_training.cell_env import CellCoverageEnv, DELTAS

CELL_SIZE_M       = 0.20
LINEAR_SPEED      = 0.15
LINEAR_SPEED_MAX  = 0.25
ANGULAR_SPEED     = 1.00
MIN_TURN_SPEED    = 0.8
OBSTACLE_DIST_M   = 0.25
WALL_SAFE_DIST_M  = 0.12
POSITION_TOL_M    = 0.06
HEADING_TOL_RAD   = 0.10
CMD_HZ            = 20
ARRIVAL_TIMEOUT_S = 65.0

# Loop detection
LOOP_WINDOW       = 10
LOOP_REVISIT_MAX  = 3   # trigger BFS if current cell was visited this many times or more

ACTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]

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


# PyTorch Policy Network (mirrors sb3 MaskableActorCriticPolicy)

class GridCNN(nn.Module):
    """Same architecture as ppo_training/feature_extractor.py GridCNN."""
    def __init__(self, n_channels=4, features_dim=128, grid_h=17, grid_w=15):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat_size = self.cnn(torch.zeros(1, n_channels, grid_h, grid_w)).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(flat_size, features_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.linear(self.cnn(x))


class PPOPolicy(nn.Module):
    """Minimal policy for inference only (action_net head)."""
    def __init__(self, n_channels=4, grid_h=17, grid_w=15,
                 features_dim=128, n_actions=4):
        super().__init__()
        self.pi_features_extractor = GridCNN(n_channels, features_dim, grid_h, grid_w)
        self.mlp_extractor_policy_net = nn.Sequential(
            nn.Linear(features_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.action_net = nn.Linear(64, n_actions)

    def forward(self, obs, action_masks=None):
        features = self.pi_features_extractor(obs)
        latent = self.mlp_extractor_policy_net(features)
        logits = self.action_net(latent)
        if action_masks is not None:
            # Masking invalid actions with large negative value
            logits = logits + (action_masks.float().log() + 1e-8)
        return logits

    def predict(self, obs_np, action_masks_np):
        """Numpy-in, int-out convenience wrapper."""
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_np).float().unsqueeze(0)
            mask_t = torch.from_numpy(action_masks_np).float().unsqueeze(0)
            logits = self.forward(obs_t, mask_t)
            probs = torch.softmax(logits, dim=1)
            return int(torch.multinomial(probs, 1).item())


def load_ppo_policy(weights_path, n_channels, grid_h, grid_w, features_dim=128):
    """Build policy network and load weights from policy_weights.pt."""
    policy = PPOPolicy(n_channels, grid_h, grid_w, features_dim, n_actions=4)

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)

    # Map sb3 keys 
    new_sd = {}
    for k, v in state_dict.items():
        nk = k
        # pi_features_extractor stays the same
        nk = nk.replace("mlp_extractor.policy_net.", "mlp_extractor_policy_net.")
        new_sd[nk] = v

    # Only load the keys the model has (skip value network weights)
    model_keys = set(policy.state_dict().keys())
    filtered = {k: v for k, v in new_sd.items() if k in model_keys}
    policy.load_state_dict(filtered, strict=True)
    policy.eval()
    return policy


# --- Helpers ---

def yaw_from_quat(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def angle_diff(a, b):
    d = a - b
    while d >  math.pi: d -= 2 * math.pi
    while d < -math.pi: d += 2 * math.pi
    return d


class CoverageExecutorPPO(Node):

    def __init__(self):
        super().__init__("coverage_executor_ppo")

        self.declare_parameter("model_path",
            str(ROOT / "results_ppo_3" / "policy_weights.pt"))
        self.declare_parameter("cell_size", CELL_SIZE_M)
        self.declare_parameter("sdf_path", str(ROOT / "maps" / "env1_bedroom.sdf"))
        self.declare_parameter("odom_topic", "/ground_truth_pose")
        
        # Hardware needs ArUco for absolute fix
        # Gazebo publishes ground-truth pose directly, so no wait needed.
        self.declare_parameter("require_aruco", False)

        model_path         = self.get_parameter("model_path").value
        self.cell_size     = self.get_parameter("cell_size").value
        sdf_path           = self.get_parameter("sdf_path").value
        odom_topic         = self.get_parameter("odom_topic").value
        self.require_aruco = self.get_parameter("require_aruco").value

        self.x = 0.0; self.y = 0.0; self.yaw = 0.0
        self.front_dist = 999.0
        self.rear_dist = 999.0
        self.odom_received = False
        self.aruco_received = False

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", _ODOM_QOS)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, _SENSOR_QOS)
        self.get_logger().info(f"Subscribing to pose on: {odom_topic}")
        self.create_subscription(LaserScan, "/ultrasonic_front", self._front_cb, _SENSOR_QOS)
        self.create_subscription(LaserScan, "/ultrasonic_rear", self._rear_cb, _SENSOR_QOS)
        self.create_subscription(PoseWithCovarianceStamped, "/aruco/pose",
                                 self._aruco_cb, _SENSOR_QOS)

        # Load room grid
        self.wall_grid, self.meta = load_room_grid(
            Path(sdf_path), cell_size=self.cell_size)

        # Create cell env 
        self.env = CellCoverageEnv(
            self.wall_grid, cell_size_m=self.cell_size, n_objects=0)

        # Load PPO policy
        obs_shape = self.env.observation_space.shape  # (4, 17, 15)
        self.policy = load_ppo_policy(
            model_path,
            n_channels=obs_shape[0],
            grid_h=obs_shape[1],
            grid_w=obs_shape[2])
        self.get_logger().info(f"Loaded PPO policy weights: {model_path}")

        self.get_logger().info(
            f"CoverageExecutorPPO ready  "
            f"grid={self.env.rows}x{self.env.cols}  cell={self.cell_size*100:.0f}cm")

    # --- Callbacks ---

    def _odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
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

    def _spin_for(self, duration_s):
        end = time.time() + duration_s
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.005)

    def _current_cell(self):
        return world_to_cell(self.x, self.y, self.meta)

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
            scale = min(1.0, abs(err) / (math.pi / 6))
            w_mag = max(MIN_TURN_SPEED, ANGULAR_SPEED * scale)
            w = math.copysign(w_mag, err)
            self._publish_vel(0.0, w)
        self._stop()
        self.get_logger().warn(
            f"  TURN TIMEOUT after 10s  "
            f"target={math.degrees(target_yaw):+6.1f}°  "
            f"final_yaw={math.degrees(self.yaw):+6.1f}°  "
            f"Δyaw={math.degrees(angle_diff(self.yaw, yaw_start)):+6.1f}° "
            f"(if Δyaw≈0 → IMU is not updating!)")
        return False

    def _drive_straight(self, dist_m, obstacle_thresh=OBSTACLE_DIST_M):
        locked_yaw = self.yaw
        x0, y0 = self.x, self.y
        while math.hypot(self.x - x0, self.y - y0) < dist_m:
            rclpy.spin_once(self, timeout_sec=0.01)
            if self.front_dist < obstacle_thresh:
                self.get_logger().warn(
                    f"Front obstacle at {self.front_dist:.2f}m — stopping")
                self._stop()
                return False
            speed = LINEAR_SPEED + (LINEAR_SPEED_MAX - LINEAR_SPEED) * min(
                1.0, max(0.0, (self.front_dist - 0.40) / 0.60))
            heading_err = angle_diff(locked_yaw, self.yaw)
            w = max(-ANGULAR_SPEED * 0.3,
                    min(ANGULAR_SPEED * 0.3, heading_err * 2.0))
            self._publish_vel(speed, w)
        self._stop()
        self._spin_for(0.05)
        return True

    def _is_near_wall(self, row, col):
        for dr, dc in DELTAS:
            nr, nc = row + dr, col + dc
            if not (0 <= nr < self.env.rows and 0 <= nc < self.env.cols):
                return True
            if self.wall_grid[nr, nc] == 1:
                return True
        return False

    def drive_to_cell(self, row, col):
        near_wall = self._is_near_wall(row, col)
        thresh = WALL_SAFE_DIST_M if near_wall else OBSTACLE_DIST_M

        tx, ty = cell_to_world(row, col, self.meta)
        t_start = time.time()
        while True:
            self._spin_once()
            if time.time() - t_start > ARRIVAL_TIMEOUT_S:
                self._stop()
                self.get_logger().warn(f"Timeout -> ({row},{col})")
                return False
            dx, dy = tx - self.x, ty - self.y
            dist = math.hypot(dx, dy)
            if dist < POSITION_TOL_M:
                self._stop()
                return True
            if self.front_dist < thresh:
                self._stop()
                return False
            if not self._turn_to_heading(math.atan2(dy, dx)):
                self.get_logger().warn(
                    f"Turn failed -> ({row},{col}) — aborting cell")
                return False
            self._drive_straight(min(dist, CELL_SIZE_M), obstacle_thresh=thresh)

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

    # --- Grid state sync ---

    def _sync_env_to_odom(self):
        r, c = self._current_cell()
        r = max(0, min(r, self.env.rows - 1))
        c = max(0, min(c, self.env.cols - 1))
        self.env.pos = (r, c)
        return r, c

    def _mark_cell_visited(self, r, c):
        if self.env.visited[r, c] == 0:
            self.env.visited[r, c] = 1
            self.env.n_visited += 1
        self.env.visit_count[r, c] += 1
        self.env.path.append((r, c))

    def _discover_obstacle_at(self, r, c):
        if 0 <= r < self.env.rows and 0 <= c < self.env.cols:
            if self.env.discovered_grid[r, c] == 0:
                self.env.live_grid[r, c] = 1
                self.env.discovered_grid[r, c] = 1
                self.env.n_free = int((self.env.live_grid == 0).sum())
                self.env._frontier_dirty = True

    def _preflight_check(self, action, tr, tc):
        tx, ty = cell_to_world(tr, tc, self.meta)
        dx, dy = tx - self.x, ty - self.y
        heading = math.atan2(dy, dx)
        self._turn_to_heading(heading)

        # After turn completes, take fresh readings
        self.front_dist = 999.0  # clear stale mid-turn reading
        readings = []
        for _ in range(3):
            self._spin_for(0.20)
            readings.append(self.front_dist)
        dist = min(readings)
        self.get_logger().info(
            f"  Preflight ({tr},{tc}): readings={[f'{r:.2f}' for r in readings]}")
        if dist < self.cell_size * 1.2:
            self.get_logger().info(
                f"  Preflight: BLOCKED dist={dist:.2f}m")
            self._discover_obstacle_at(tr, tc)
            return False
        return True

    # --- Coverage run ---

    def run_coverage(self):
        self.get_logger().info("Waiting for odometry...")
        t_wait = time.time()
        next_log = 5.0
        while not self.odom_received:
            rclpy.spin_once(self, timeout_sec=0.5)
            elapsed = time.time() - t_wait
            if elapsed > 30.0:
                self.get_logger().error("No odometry after 30s. Aborting.")
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

        self.get_logger().info(f"Ready. World pos: ({self.x:.3f}, {self.y:.3f})")

        # Reset env
        self.env.reset()
        self.env.live_grid[:] = self.wall_grid.copy()
        self.env.discovered_grid[:] = self.wall_grid.copy()
        self.env.visited[:] = 0
        self.env.visit_count[:] = 0
        self.env.n_free = int((self.env.live_grid == 0).sum())
        self.env.n_visited = 0
        self.env.nb_steps = 0
        self.env.path = []
        self.env._frontier_dirty = True

        sr, sc = self._sync_env_to_odom()
        self._mark_cell_visited(sr, sc)

        self.get_logger().info(
            f"Starting from cell ({sr},{sc})  free={self.env.n_free}")

        obs = self.env._obs()

        recent_positions = []
        ppo_actions = 0
        bfs_actions = 0
        skipped = 0
        step = 0
        max_steps = 4 * self.env.n_free

        while self.env.n_visited < self.env.n_free and step < max_steps:
            step += 1

            # Get action from PPO policy
            action_masks = self.env.action_masks()
            action = self.policy.predict(obs, action_masks)

            # Detect if there is a loop
            recent_positions.append(self.env.pos)
            if len(recent_positions) > LOOP_WINDOW:
                recent_positions.pop(0)

            used_bfs = False
            cur_visits = self.env.visit_count[self.env.pos[0], self.env.pos[1]]
            too_many_revisits = cur_visits >= LOOP_REVISIT_MAX
            too_few_unique = (len(recent_positions) == LOOP_WINDOW
                              and len(set(recent_positions)) <= LOOP_WINDOW // 2)
            if too_many_revisits or too_few_unique:
                bfs_action = self.env.bfs_to_nearest_unvisited()
                if bfs_action is not None:
                    action = bfs_action
                    used_bfs = True

            if used_bfs:
                bfs_actions += 1
            else:
                ppo_actions += 1

            # Compute target cell
            dr, dc = DELTAS[action]
            cr, cc = self.env.pos
            tr, tc = cr + dr, cc + dc

            # If PPO picks a visited cell but an unvisited neighbour exists,
            # override with BFS to avoid wasted moves.
            if (not used_bfs
                    and 0 <= tr < self.env.rows and 0 <= tc < self.env.cols
                    and self.env.visited[tr, tc] == 1):
                has_unvisited_neighbor = any(
                    0 <= cr + ddr < self.env.rows
                    and 0 <= cc + ddc < self.env.cols
                    and self.env.live_grid[cr + ddr, cc + ddc] == 0
                    and self.env.visited[cr + ddr, cc + ddc] == 0
                    for ddr, ddc in DELTAS)
                if has_unvisited_neighbor:
                    bfs_a = self.env.bfs_to_nearest_unvisited()
                    if bfs_a is not None:
                        action = bfs_a
                        dr, dc = DELTAS[action]
                        tr, tc = cr + dr, cc + dc
                        used_bfs = True

            src = "BFS" if used_bfs else "PPO"
            self.get_logger().info(
                f"Step {step}: {src} → {ACTION_NAMES[action]} "
                f"from ({cr},{cc}) to ({tr},{tc})")

            # Check bounds
            if not (0 <= tr < self.env.rows and 0 <= tc < self.env.cols):
                self.get_logger().warn(f"  Out of bounds — skipping")
                obs = self.env._obs()
                continue

            # Check if the obstacle is known and update env so action mask blocks it
            if self.env.discovered_grid[tr, tc] == 1:
                skipped += 1
                if skipped > 3:
                    # if completely stuck, try BFS from scratch
                    bfs_a = self.env.bfs_to_nearest_unvisited()
                    if bfs_a is not None:
                        dr2, dc2 = DELTAS[bfs_a]
                        tr2, tc2 = cr + dr2, cc + dc2
                        if (0 <= tr2 < self.env.rows and
                            0 <= tc2 < self.env.cols and
                            self.env.discovered_grid[tr2, tc2] == 0):
                            self.get_logger().info(
                                f"  Force BFS fallback → {ACTION_NAMES[bfs_a]} to ({tr2},{tc2})")
                            action = bfs_a
                            tr, tc = tr2, tc2
                            skipped = 0
                        else:
                            self.get_logger().warn(f"  BFS also blocked — skipping")
                            obs = self.env._obs()
                            continue
                    else:
                        self.get_logger().warn(f"  No BFS path found")
                        obs = self.env._obs()
                        continue
                else:
                    self.get_logger().warn(
                        f"  Known obstacle at ({tr},{tc}) — skipping")
                    obs = self.env._obs()
                    continue

            skipped = 0

            # Preflight check
            if not self._preflight_check(action, tr, tc):
                obs = self.env._obs()
                continue

            # Drive to target cell
            ok = self.drive_to_cell(tr, tc)

            if ok:
                self._sync_env_to_odom()
                self.env.pos = (tr, tc)
                self._mark_cell_visited(tr, tc)
                self.env._frontier_dirty = True
            else:
                self.get_logger().info(
                    f"Obstacle discovered at ({tr},{tc}) via collision")
                self._discover_obstacle_at(tr, tc)
                self.backup_one_cell()
                self._sync_env_to_odom()

            obs = self.env._obs()

            if step % 20 == 0:
                cov = self.env.coverage_fraction()
                self.get_logger().info(
                    f"  Step {step}  cell=({self.env.pos[0]},{self.env.pos[1]})  "
                    f"coverage={cov*100:.1f}%  "
                    f"PPO={ppo_actions} BFS={bfs_actions}")

        self._stop()
        cov = self.env.coverage_fraction()
        total = max(ppo_actions + bfs_actions, 1)
        self.get_logger().info(
            f"{'='*50}\n"
            f"  Coverage complete!\n"
            f"  Visited: {self.env.n_visited}/{self.env.n_free} "
            f"({cov*100:.1f}%)\n"
            f"  Steps: {step}\n"
            f"  Actions: PPO={ppo_actions} "
            f"({100*ppo_actions/total:.0f}%)  "
            f"BFS={bfs_actions} ({100*bfs_actions/total:.0f}%)\n"
            f"{'='*50}")


def main(args=None):
    rclpy.init(args=args)
    node = CoverageExecutorPPO()
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
