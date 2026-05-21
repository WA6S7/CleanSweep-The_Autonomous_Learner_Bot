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

ROOT = Path("/home/user/thesis_ws/src/clean_sweep_rl_v2")
sys.path.insert(0, str(ROOT))

from coverage_env.room import load_room_grid

CELL_SIZE_M = 0.20

_SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10)


def yaw_from_quat(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def world_to_cell_f(wx, wy, meta):
    # Continuous (row, col) — no floor, so particles plot smoothly
    col = (wx - meta["origin_x"]) / meta["cell_size"]
    row = (meta["origin_y"] - wy)  / meta["cell_size"]
    return row, col


class EKFParticleVisualiser(Node):

    def __init__(self):
        super().__init__("ekf_particle_visualiser")

        self.declare_parameter("sdf_path",  str(ROOT / "maps" / "env1_bedroom.sdf"))
        self.declare_parameter("cell_size", CELL_SIZE_M)
        self.declare_parameter("odom_topic", "/odometry/filtered")
        self.declare_parameter("n_particles", 600)

        self.declare_parameter("min_sigma_m", 0.0)

        sdf_path        = self.get_parameter("sdf_path").value
        self.cell_size  = self.get_parameter("cell_size").value
        odom_topic      = self.get_parameter("odom_topic").value
        self.n_particles = int(self.get_parameter("n_particles").value)
        self.min_sigma   = float(self.get_parameter("min_sigma_m").value)

        self.wall_grid, self.meta = load_room_grid(
            Path(sdf_path), cell_size=self.cell_size)
        self.rows, self.cols = self.wall_grid.shape

        self.mean_xy   = None       # (x, y) in metres
        self.mean_yaw  = 0.0
        self.cov_xy    = np.zeros((2, 2))    # 2×2 position covariance
        self.cov_yaw   = 0.0
        self.particles = None       # (N, 2) array of (row, col) for plotting
        self.last_msg_time = None

        self._rng = np.random.default_rng()

        self.create_subscription(Odometry, odom_topic, self._odom_cb, _SENSOR_QOS)
        self.get_logger().info(f"Subscribing to: {odom_topic}")
        self.get_logger().info(
            f"Grid={self.rows}×{self.cols}  cell={self.cell_size*100:.0f}cm  "
            f"N={self.n_particles}  min_sigma={self.min_sigma:.3f}m")

    def _odom_cb(self, msg):
        x   = msg.pose.pose.position.x
        y   = msg.pose.pose.position.y
        yaw = yaw_from_quat(msg.pose.pose.orientation)

        # Taking the xy and yaw blocks out of the 6x6 pose covariance
        cov = np.array(msg.pose.covariance, dtype=float).reshape(6, 6)
        cov_xy  = cov[0:2, 0:2]
        cov_yaw = float(cov[5, 5])

        # Inflating xy covariance up to the configured minimum spread
        if self.min_sigma > 0.0:
            floor = self.min_sigma ** 2
            cov_xy = cov_xy + np.eye(2) * max(0.0, floor - min(cov_xy[0,0], cov_xy[1,1]))

        self.mean_xy  = (x, y)
        self.mean_yaw = yaw
        self.cov_xy   = cov_xy
        self.cov_yaw  = cov_yaw
        self.last_msg_time = time.monotonic()

        # Drawing a Gaussian cloud representing the EKF position belief
        try:
            samples = self._rng.multivariate_normal(
                mean=[x, y], cov=cov_xy, size=self.n_particles,
                check_valid="ignore")
        except np.linalg.LinAlgError:
            samples = np.tile([x, y], (self.n_particles, 1))  

        # Cache as continuous grid coords for smooth plotting
        rows, cols = world_to_cell_f(samples[:, 0], samples[:, 1], self.meta)
        self.particles = np.column_stack([rows, cols])

    def stddev_xy(self):
        return float(np.sqrt(self.cov_xy[0, 0])), float(np.sqrt(self.cov_xy[1, 1]))

    def stddev_yaw_deg(self):
        return math.degrees(math.sqrt(max(self.cov_yaw, 0.0)))


def render_walls(ax, wall_grid):
    # Draw the static map once as a background (white floor, grey walls)
    rows, cols = wall_grid.shape
    img = np.ones((rows, cols, 3), dtype=np.float32)
    img[wall_grid > 0] = [0.15, 0.15, 0.15]
    ax.imshow(img, interpolation="nearest", origin="upper",
              extent=[-0.5, cols-0.5, rows-0.5, -0.5])


def main(args=None):
    rclpy.init(args=args)
    node = EKFParticleVisualiser()

    # Spinning ROS in a background thread
    stop_spin = threading.Event()
    def _spin():
        while not stop_spin.is_set() and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    spin_thread = threading.Thread(target=_spin, daemon=True)
    spin_thread.start()

    fig, ax = plt.subplots(figsize=(8, 9))
    fig.patch.set_facecolor("#1a1a2e")

    render_walls(ax, node.wall_grid)

    particle_scatter = ax.scatter([], [], s=4, c="#FF3030", alpha=0.55,
                                  edgecolors="none", zorder=3)
    mean_dot = ax.plot([], [], "o", color="#000000", markersize=9,
                       markeredgecolor="white", markeredgewidth=1.5, zorder=11)[0]
    mean_arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0), zorder=10,
        arrowprops=dict(arrowstyle="->, head_width=0.4, head_length=0.4",
                        color="#FFD700", lw=2.0))

    ax.legend(handles=[
        mpatches.Patch(color="#FF3030", label=f"Particles (N={node.n_particles})"),
        mpatches.Patch(color="#000000", label="EKF mean"),
        mpatches.Patch(color="#FFD700", label="Heading"),
        mpatches.Patch(color=[0.15, 0.15, 0.15], label="Wall"),
    ], loc="lower right", fontsize=8, framealpha=0.85, facecolor="#f0f0f0",
       title="EKF belief", title_fontsize=8)

    ax.set_xlim(-0.5, node.cols - 0.5)
    ax.set_ylim(node.rows - 0.5, -0.5)
    ax.axis("off")
    plt.tight_layout()

    def update(_):
        # Animation tick which moves the particle cloud, mean marker, and heading arrow
        if node.particles is not None:
            particle_scatter.set_offsets(
                np.column_stack([node.particles[:, 1], node.particles[:, 0]]))
        if node.mean_xy is not None:
            r, c = world_to_cell_f(node.mean_xy[0], node.mean_xy[1], node.meta)
            dc =  0.8 * math.cos(node.mean_yaw)
            dr = -0.8 * math.sin(node.mean_yaw)
            mean_dot.set_data([c], [r])
            mean_arrow.xy     = (c + dc, r + dr)
            mean_arrow.xytext = (c, r)

            sx, sy = node.stddev_xy()
            syaw_deg = node.stddev_yaw_deg()
            
            stale = ""
            if node.last_msg_time is not None \
                    and (time.monotonic() - node.last_msg_time) > 1.0:
                stale = "  [STALE]"
            ax.set_title(
                f"EKF particle cloud  |  "
                f"μ=({node.mean_xy[0]:+.2f},{node.mean_xy[1]:+.2f})  "
                f"σ=({sx*100:.1f},{sy*100:.1f})cm  "
                f"σ_yaw={syaw_deg:.1f}°{stale}",
                fontsize=10, color="white", pad=8)
        else:
            ax.set_title("Waiting for EKF pose…", fontsize=10, color="white", pad=8)
        fig.patch.set_facecolor("#1a1a2e")
        return [particle_scatter, mean_dot, mean_arrow]

    ani = FuncAnimation(fig, update, interval=200, blit=False, cache_frame_data=False)
    plt.show()
    stop_spin.set()
    spin_thread.join(timeout=1.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
