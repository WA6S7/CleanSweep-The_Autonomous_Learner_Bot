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

# Discrete heat tiers indexed by visit count 
HEAT_COLORS = [
    (0.20, 0.55, 0.95),   # 1× — blue
    (0.20, 0.85, 0.65),   # 2× — teal/green
    (0.95, 0.90, 0.20),   # 3× — yellow
    (1.00, 0.55, 0.10),   # 4× — orange
    (0.95, 0.20, 0.15),   # 5× — red
    (0.55, 0.05, 0.55),   # 6+× — magenta
]


def heat_color(count):
    # Mapping a visit count to a colour tier
    idx = min(max(int(count) - 1, 0), len(HEAT_COLORS) - 1)
    return HEAT_COLORS[idx]


def yaw_from_quat(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class CoverageHeatmapVisualiser(Node):

    def __init__(self):
        super().__init__("coverage_heatmap_visualiser")

        self.declare_parameter("sdf_path",  str(ROOT / "maps" / "env1_bedroom.sdf"))
        self.declare_parameter("cell_size", CELL_SIZE_M)
        self.declare_parameter("odom_topic", "/ground_truth_pose")
        self.declare_parameter("show_counts", True)

        sdf_path        = self.get_parameter("sdf_path").value
        self.cell_size  = self.get_parameter("cell_size").value
        odom_topic      = self.get_parameter("odom_topic").value
        self.show_counts = bool(self.get_parameter("show_counts").value)

        # Building the grid/env from the SDF, and starting with a clean map
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
        self.start_time = None

        self.create_subscription(Odometry, odom_topic, self._odom_cb, _SENSOR_QOS)
        self.get_logger().info(f"Subscribing to pose on: {odom_topic}")
        self.create_subscription(LaserScan, "/ultrasonic_front",  self._front_cb,  _SENSOR_QOS)
        self.create_subscription(LaserScan, "/ultrasonic_rear",   self._rear_cb,   _SENSOR_QOS)

        self.get_logger().info(
            f"Heatmap visualiser ready  grid={self.rows}×{self.cols}  "
            f"cell={self.cell_size*100:.0f}cm")

    def _odom_cb(self, msg):
        if self.start_time is None:
            # clock starts on first pose
            self.start_time = time.monotonic()  
        x   = msg.pose.pose.position.x
        y   = msg.pose.pose.position.y
        yaw = yaw_from_quat(msg.pose.pose.orientation)

        # World pose to grid cell
        row, col = world_to_cell(x, y, self.meta)
        row = max(0, min(self.rows-1, row))
        col = max(0, min(self.cols-1, col))
        self.robot_yaw = yaw
        self._n = getattr(self, '_n', 0) + 1
        # logging to every 50th pose
        if self._n % 50 == 1:  
            self.get_logger().info(
                f"world=({x:.3f},{y:.3f}) -> grid({row},{col})")
        cell_entered = (row, col) != (self.robot_row, self.robot_col)
        # Count a visit only on free, non-obstacle cells
        if self.wall_grid[row, col] == 0 and self.env.live_grid[row, col] == 0:
            if self.env.visit_count[row, col] == 0:
                self.steps += 1               # first time in a new unique cell
            if cell_entered:
                self.env.visit_count[row, col] += 1  # bump only on cell change, not per message
            self.env.visited[row, col] = 1
        self.robot_row, self.robot_col = row, col

    def _front_cb(self, msg):
        # Taking the closest valid beam as the front distance
        valid = [r for r in msg.ranges
                 if msg.range_min <= r <= msg.range_max and math.isfinite(r)]
        self.front_dist = min(valid) if valid else 999.0
        
        if self.front_dist < OBSTACLE_DIST_M and self.robot_row is not None:
            # Marking the cell directly ahead as a discovered obstacle
            dr, dc = self._heading_delta()
            or_, oc = self.robot_row + dr, self.robot_col + dc
            if 0 <= or_ < self.rows and 0 <= oc < self.cols \
                    and self.wall_grid[or_, oc] == 0:
                self.env.live_grid[or_, oc] = 1

    def _rear_cb(self, msg):
        pass  # rear sensor is unused here, kept for topic symmetry

    def _heading_delta(self):
        # Snapping yaw to the nearest cardinal direction -> (drow, dcol) step
        a = (self.robot_yaw + math.pi) % (2*math.pi) - math.pi
        if   -math.pi/4 <= a <  math.pi/4:  return  0,  1
        elif  math.pi/4 <= a < 3*math.pi/4: return -1,  0
        elif a >= 3*math.pi/4 or a < -3*math.pi/4: return 0, -1
        else:                                return  1,  0

    def get_display_grid(self):
        # Building the RGB image: walls grey, obstacles orange, visited cells heat-coloured
        display = np.ones((self.rows, self.cols, 3), dtype=np.float32)
        vc = self.env.visit_count
        for r in range(self.rows):
            for c in range(self.cols):
                if self.wall_grid[r, c]:
                    display[r, c] = [0.15, 0.15, 0.15]   # wall
                elif self.env.live_grid[r, c]:
                    display[r, c] = [0.85, 0.50, 0.15]   # discovered obstacle
                elif vc[r, c] > 0:
                    display[r, c] = heat_color(vc[r, c])  # visit heat
        return display

    def coverage_pct(self):
        return 100.0 * int(self.env.visited.sum()) / max(1, self.env.n_free)

    def overlap_stats(self):
        # Overlap = repeat visits (every visit beyond the first to a cell)
        vc = self.env.visit_count
        total_visits   = int(vc.sum())
        unique_visits  = int((vc > 0).sum())
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
        vc = self.env.visit_count
        max_v = int(vc.max())
        revisited = int((vc > 1).sum())
        self.get_logger().info("=" * 50)
        self.get_logger().info("       HEATMAP RUN SUMMARY")
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"  Coverage         : {cov:.1f}%")
        self.get_logger().info(f"  Unique cells     : {self.steps}")
        self.get_logger().info(f"  Revisited cells  : {revisited}")
        self.get_logger().info(f"  Max visits/cell  : {max_v}")
        self.get_logger().info(f"  Total visits     : {tv}")
        self.get_logger().info(f"  Overlap visits   : {ov}")
        self.get_logger().info(f"  Overlap          : {op:.1f}%")
        self.get_logger().info(f"  Elapsed time     : {int(mins)}m {secs:.1f}s")
        self.get_logger().info("=" * 50)


def main(args=None):
    rclpy.init(args=args)
    node = CoverageHeatmapVisualiser()

    # Spinning ROS in a background thread
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

    # Dashed lines marking the boustrophedon strip boundaries
    for s in node.env.strips:
        ax.axhline(s.rows[0]-0.5, color="#E07820", lw=0.8, ls="--", alpha=0.5)

    count_texts = {}
    if node.show_counts:
        for r in range(node.rows):
            for c in range(node.cols):
                if not node.wall_grid[r, c]:
                    txt = ax.text(c, r, "", ha="center", va="center",
                                  fontsize=6, color="white", fontweight="bold",
                                  zorder=5)
                    count_texts[(r, c)] = txt

    robot_dot = ax.plot([], [], "o", color="#00BFFF", markersize=10,
                        markeredgecolor="white", markeredgewidth=1.5, zorder=11)[0]
    robot_arrow = ax.annotate("", xy=(0,0), xytext=(0,0), zorder=10,
        arrowprops=dict(arrowstyle="->, head_width=0.4, head_length=0.4",
                        color="#00BFFF", lw=2.0))

    heat_handles = [
        mpatches.Patch(color=HEAT_COLORS[i],
                       label=f"{i+1}×" if i < len(HEAT_COLORS)-1
                                       else f"{i+1}+×")
        for i in range(len(HEAT_COLORS))
    ]
    ax.legend(handles=heat_handles + [
        mpatches.Patch(color=[0.85,0.50,0.15], label="Obstacle"),
        mpatches.Patch(color=[0.15,0.15,0.15], label="Wall"),
        mpatches.Patch(color=[0.00,0.75,1.00], label="Robot"),
    ], loc="lower right", fontsize=7, framealpha=0.85, facecolor="#f0f0f0",
       title="Visits", title_fontsize=7)

    ax.set_xlim(-0.5, node.cols-0.5)
    ax.set_ylim(node.rows-0.5, -0.5)
    ax.axis("off")
    plt.tight_layout()

    def update(_):
        im.set_data(node.get_display_grid())
        if node.show_counts:
            vc = node.env.visit_count
            for (r, c), txt in count_texts.items():
                v = int(vc[r, c])
                txt.set_text(str(v) if v > 0 else "")
        if node.robot_row is not None:
            r, c = node.robot_row, node.robot_col
            dc =  0.8 * math.cos(node.robot_yaw)
            dr = -0.8 * math.sin(node.robot_yaw)
            robot_dot.set_data([c], [r])
            robot_arrow.xy     = (c+dc, r+dr)
            robot_arrow.xytext = (c, r)
        _, _, ov_pct = node.overlap_stats()
        max_v = int(node.env.visit_count.max())
        elapsed = node.elapsed_seconds()
        e_min, e_sec = divmod(elapsed, 60)
        ax.set_title(
            f"Coverage {node.coverage_pct():.1f}%  |  "
            f"steps={node.steps}  |  overlap={ov_pct:.1f}%  |  "
            f"max={max_v}×  |  "
            f"{int(e_min)}m{e_sec:04.1f}s",
            fontsize=10, color="white", pad=8)
        fig.patch.set_facecolor("#1a1a2e")
        return [im, robot_dot, robot_arrow]

    ani = FuncAnimation(fig, update, interval=200, blit=False, cache_frame_data=False)
    plt.show()           # blocks until the window is closed
    node.print_summary() # print run stats on exit
    stop_spin.set()
    spin_thread.join(timeout=1.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
