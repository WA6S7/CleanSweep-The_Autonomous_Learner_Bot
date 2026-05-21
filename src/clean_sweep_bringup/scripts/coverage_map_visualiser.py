#!/usr/bin/env python3
 
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
 
ROOT = Path("/home/user/thesis_ws/src/clean_sweep_rl_v2")
sys.path.insert(0, str(ROOT))
 
from coverage_env.room import load_room_grid, world_to_cell
from coverage_env.coverage_env import CoverageEnv
 
CELL_SIZE_M     = 0.20
OBSTACLE_DIST_M = 0.20
 
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
 
 
class CoverageMapVisualiser(Node):
 
    def __init__(self):
        super().__init__("coverage_map_visualiser")
 
        self.declare_parameter("sdf_path",  str(ROOT / "maps" / "env1_bedroom.sdf"))
        self.declare_parameter("cell_size", CELL_SIZE_M)
        self.declare_parameter("odom_topic", "/ground_truth_pose")

        sdf_path       = self.get_parameter("sdf_path").value
        self.cell_size = self.get_parameter("cell_size").value
        odom_topic     = self.get_parameter("odom_topic").value
 
        self.wall_grid, self.meta = load_room_grid(
            Path(sdf_path), cell_size=self.cell_size)
        self.env = CoverageEnv(self.wall_grid, cell_size_m=self.cell_size,
                               n_objects=0)
        self.env.live_grid[:]   = self.wall_grid.copy()
        self.env.visited[:]     = 0
        self.env.visit_count[:] = 0
        self.env.n_free         = int((self.wall_grid == 0).sum())
 
        self.rows = self.env.rows
        self.cols = self.env.cols
        self.robot_row  = None
        self.robot_col  = None
        self.robot_yaw  = 0.0
        self.steps      = 0
        self.front_dist = 999.0
        self.visit_order = np.zeros((self.rows, self.cols), dtype=int)
        self.visit_seq   = 0
        
        # set on first odom msg
        self.start_time  = None
 
        self.create_subscription(Odometry, odom_topic, self._odom_cb, _SENSOR_QOS)
        self.get_logger().info(f"Subscribing to pose on: {odom_topic}")
        self.create_subscription(LaserScan, "/ultrasonic_front",  self._front_cb,  _SENSOR_QOS)
        self.create_subscription(LaserScan, "/ultrasonic_rear",   self._rear_cb,   _SENSOR_QOS)
 
        self.get_logger().info(
            f"Visualiser ready  grid={self.rows}×{self.cols}  "
            f"cell={self.cell_size*100:.0f}cm")
 
    def _odom_cb(self, msg):
        if self.start_time is None:
            self.start_time = time.monotonic()
        x   = msg.pose.pose.position.x
        y   = msg.pose.pose.position.y
        yaw = yaw_from_quat(msg.pose.pose.orientation)

        row, col = world_to_cell(x, y, self.meta)
        row = max(0, min(self.rows-1, row))
        col = max(0, min(self.cols-1, col))
        self.robot_yaw = yaw
        self._n = getattr(self, '_n', 0) + 1
        if self._n % 50 == 1:
            self.get_logger().info(
                f"world=({x:.3f},{y:.3f}) -> grid({row},{col})")
        cell_entered = (row, col) != (self.robot_row, self.robot_col)
        if self.wall_grid[row, col] == 0 and self.env.live_grid[row, col] == 0:
            if self.env.visit_count[row, col] == 0:
                self.steps += 1
                self.visit_seq += 1
                self.visit_order[row, col] = self.visit_seq
            if cell_entered:
                self.env.visit_count[row, col] += 1
            self.env.visited[row, col]      = 1
        self.robot_row, self.robot_col = row, col
 
    def _front_cb(self, msg):
        valid = [r for r in msg.ranges
                 if msg.range_min <= r <= msg.range_max and math.isfinite(r)]
        self.front_dist = min(valid) if valid else 999.0
        if self.front_dist < OBSTACLE_DIST_M and self.robot_row is not None:
            dr, dc = self._heading_delta()
            or_, oc = self.robot_row + dr, self.robot_col + dc
            if 0 <= or_ < self.rows and 0 <= oc < self.cols \
                    and self.wall_grid[or_, oc] == 0:
                self.env.live_grid[or_, oc] = 1
 
    def _rear_cb(self, msg):
        pass
 
    def _heading_delta(self):
        a = (self.robot_yaw + math.pi) % (2*math.pi) - math.pi
        if   -math.pi/4 <= a <  math.pi/4:  return  0,  1
        elif  math.pi/4 <= a < 3*math.pi/4: return -1,  0
        elif a >= 3*math.pi/4 or a < -3*math.pi/4: return 0, -1
        else:                                return  1,  0
 
    def get_display_grid(self):
        display = np.ones((self.rows, self.cols, 3), dtype=np.float32)
        max_seq = max(self.visit_seq, 1)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.wall_grid[r, c]:
                    display[r, c] = [0.15, 0.15, 0.15]
                elif self.env.live_grid[r, c]:
                    display[r, c] = [0.85, 0.50, 0.15]
                elif self.visit_order[r, c] > 0:
                    # Colour gradient: early=blue, late=red
                    t = (self.visit_order[r, c] - 1) / max_seq
                    display[r, c] = [t, 0.2 + 0.3*(1-t), 1.0 - t]
        return display
 
    def coverage_pct(self):
        return 100.0 * int(self.env.visited.sum()) / max(1, self.env.n_free)

    def overlap_stats(self):
        vc = self.env.visit_count
        total_visits   = int(vc.sum())

        # cells visited at least once
        unique_visits  = int((vc > 0).sum())        
        
        # extra / redundant visits
        overlap_visits = total_visits - unique_visits 
        overlap_pct    = 100.0 * overlap_visits / max(1, total_visits)
        return overlap_visits, total_visits, overlap_pct

    def elapsed_seconds(self):
        if self.start_time is None:
            return 0.0
        return time.monotonic() - self.start_time

    def print_summary(self):
        elapsed = self.elapsed_seconds()
        mins, secs = divmod(elapsed, 60)
        ov, tv, op = self.overlap_stats()
        cov = self.coverage_pct()
        self.get_logger().info("=" * 50)
        self.get_logger().info("         RUN SUMMARY")
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"  Coverage        : {cov:.1f}%")
        self.get_logger().info(f"  Unique steps    : {self.steps}")
        self.get_logger().info(f"  Total visits    : {tv}")
        self.get_logger().info(f"  Overlap visits  : {ov}")
        self.get_logger().info(f"  Overlap         : {op:.1f}%")
        self.get_logger().info(f"  Elapsed time    : {int(mins)}m {secs:.1f}s")
        self.get_logger().info("=" * 50)
 
 
def main(args=None):
    rclpy.init(args=args)
    node = CoverageMapVisualiser()

    # Spinning rclpy in a background thread so odom/laser callbacks keep firing
    stop_spin = threading.Event()
    def _spin():
        while not stop_spin.is_set() and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    spin_thread = threading.Thread(target=_spin, daemon=True)
    spin_thread.start()

    fig, ax = plt.subplots(figsize=(8, 9))
    fig.patch.set_facecolor("#1a1a2e")
 
    im = ax.imshow(np.ones((node.rows, node.cols, 3), dtype=np.float32),
                   interpolation="nearest", origin="upper",
                   extent=[-0.5, node.cols-0.5, node.rows-0.5, -0.5])
 
    for s in node.env.strips:
        ax.axhline(s.rows[0]-0.5, color="#E07820", lw=0.8, ls="--", alpha=0.5)
 
    # text for visit order numbers
    order_texts = {}
    for r in range(node.rows):
        for c in range(node.cols):
            if not node.wall_grid[r, c]:
                txt = ax.text(c, r, "", ha="center", va="center",
                              fontsize=6, color="white", fontweight="bold", zorder=5)
                order_texts[(r, c)] = txt

    robot_dot = ax.plot([], [], "o", color="#00BFFF", markersize=10,
                        markeredgecolor="white", markeredgewidth=1.5, zorder=11)[0]
    robot_arrow = ax.annotate("", xy=(0,0), xytext=(0,0), zorder=10,
        arrowprops=dict(arrowstyle="->, head_width=0.4, head_length=0.4",
                        color="#00BFFF", lw=2.0))
 
    ax.legend(handles=[
        mpatches.Patch(color=[0.0,0.5,1.0],  label="Visited early"),
        mpatches.Patch(color=[1.0,0.2,0.0],  label="Visited late"),
        mpatches.Patch(color=[0.85,0.50,0.15], label="Obstacle"),
        mpatches.Patch(color=[0.15,0.15,0.15], label="Wall"),
        mpatches.Patch(color=[0.00,0.75,1.00], label="Robot"),
    ], loc="lower right", fontsize=7, framealpha=0.85, facecolor="#f0f0f0")
 
    ax.set_xlim(-0.5, node.cols-0.5)
    ax.set_ylim(node.rows-0.5, -0.5)
    ax.axis("off")
    plt.tight_layout()
 
    def update(_):
        # Reading the current state and refreshing the display.
        im.set_data(node.get_display_grid())
        for (r, c), txt in order_texts.items():
            seq = node.visit_order[r, c]
            txt.set_text(str(seq) if seq > 0 else "")
        if node.robot_row is not None:
            r, c = node.robot_row, node.robot_col
            dc =  0.8 * math.cos(node.robot_yaw)
            dr = -0.8 * math.sin(node.robot_yaw)
            robot_dot.set_data([c], [r])
            robot_arrow.xy     = (c+dc, r+dr)
            robot_arrow.xytext = (c, r)
        _, _, ov_pct = node.overlap_stats()
        elapsed = node.elapsed_seconds()
        e_min, e_sec = divmod(elapsed, 60)
        ax.set_title(
            f"Coverage {node.coverage_pct():.1f}%  |  "
            f"steps={node.steps}  |  overlap={ov_pct:.1f}%  |  "
            f"{int(e_min)}m{e_sec:04.1f}s",
            fontsize=10, color="white", pad=8)
        fig.patch.set_facecolor("#1a1a2e")
        return [im, robot_dot, robot_arrow]
 
    ani = FuncAnimation(fig, update, interval=200, blit=False, cache_frame_data=False)
    plt.show()
    node.print_summary()
    stop_spin.set()
    spin_thread.join(timeout=1.0)
    node.destroy_node()
    rclpy.shutdown()
 
 
if __name__ == "__main__":
    main()