#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.time import Time

from std_msgs.msg import Float32, String
from std_srvs.srv import Empty, Trigger
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped

try:
    from tf2_ros import Buffer, TransformListener
    from tf2_ros import TransformException
except Exception: 
    Buffer = None
    TransformListener = None
    TransformException = Exception


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def stamp_to_sec(stamp) -> float:
    # Converting builtin_interfaces/Time to float seconds
    try:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9
    except Exception:
        return float(time.time())


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


@dataclass
class GridSpec:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    frame_id: str


class CoverageMonitor(Node):
    def __init__(self) -> None:
        super().__init__("coverage_monitor")

        # --- Parameters ---
        self.declare_parameter("mode", "map") 
        self.declare_parameter("pose_source", "odom")  
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("grid_frame", "map")
        self.declare_parameter("base_frame", "base_link")

        self.declare_parameter("robot_radius", 0.18)  # meters 
        self.declare_parameter("resolution", 0.05)    # meters/cell 
        self.declare_parameter("min_pose_delta", 0.02)  # meters; ignoring tiny moves

        # Hardware robustness
        self.declare_parameter("max_speed_mps", 1.0)  # reject paints above this speed, probably teleports
        self.declare_parameter("covariance_source", "none")  # none, odom, or amcl_pose
        self.declare_parameter("amcl_pose_topic", "/amcl_pose")
        self.declare_parameter("cov_timeout_sec", 1.0)  # if the covariance older than 1, ignore it
        self.declare_parameter("max_xy_sigma", 0.0)  # meters; if >0 reject paints when sigma > threshold
        self.declare_parameter("radius_mode", "nominal")  # nominal, conservative, or optimistic
        self.declare_parameter("sigma_multiplier", 2.0)  # k used in radius adjustment
        self.declare_parameter("radius_min_m", 0.02)  # lower bound for effective radius in conservative mode

        # map mode thresholds
        self.declare_parameter("free_threshold", 50)      # less than or equal to 50 is free
        self.declare_parameter("include_unknown", False)  # unknown is treated as free

        # room_manifest mode
        self.declare_parameter("room_x", 2.77)  # meters
        self.declare_parameter("room_y", 3.295)  # meters
        self.declare_parameter("room_centered", True)  # room centered at (0,0)
        self.declare_parameter("manifest_path", os.path.expanduser("~/.cache/rand_obstacles_manifest.json"))
        self.declare_parameter("manifest_watch_period", 0.5)  # seconds
        self.declare_parameter("reset_on_manifest_change", True)

        # publishing
        self.declare_parameter("publish_rate_hz", 2.0)
        self.declare_parameter("publish_grid", True)

        self.mode: str = str(self.get_parameter("mode").value).strip().lower()
        self.pose_source: str = str(self.get_parameter("pose_source").value).strip().lower()
        self.odom_topic: str = str(self.get_parameter("odom_topic").value)
        self.map_topic: str = str(self.get_parameter("map_topic").value)
        self.grid_frame: str = str(self.get_parameter("grid_frame").value)
        self.base_frame: str = str(self.get_parameter("base_frame").value)

        self.robot_radius: float = float(self.get_parameter("robot_radius").value)
        self.room_resolution: float = float(self.get_parameter("resolution").value)
        self.min_pose_delta: float = float(self.get_parameter("min_pose_delta").value)

        self.max_speed_mps: float = float(self.get_parameter("max_speed_mps").value)
        self.covariance_source: str = str(self.get_parameter("covariance_source").value).strip().lower()
        self.amcl_pose_topic: str = str(self.get_parameter("amcl_pose_topic").value)
        self.cov_timeout_sec: float = float(self.get_parameter("cov_timeout_sec").value)
        self.max_xy_sigma: float = float(self.get_parameter("max_xy_sigma").value)
        self.radius_mode: str = str(self.get_parameter("radius_mode").value).strip().lower()
        self.sigma_multiplier: float = float(self.get_parameter("sigma_multiplier").value)
        self.radius_min_m: float = float(self.get_parameter("radius_min_m").value)

        self.free_threshold: int = int(self.get_parameter("free_threshold").value)
        self.include_unknown: bool = bool(self.get_parameter("include_unknown").value)

        self.room_x: float = float(self.get_parameter("room_x").value)
        self.room_y: float = float(self.get_parameter("room_y").value)
        self.room_centered: bool = bool(self.get_parameter("room_centered").value)
        self.manifest_path: str = str(self.get_parameter("manifest_path").value)
        self.manifest_watch_period: float = float(self.get_parameter("manifest_watch_period").value)
        self.reset_on_manifest_change: bool = bool(self.get_parameter("reset_on_manifest_change").value)

        self.publish_rate_hz: float = float(self.get_parameter("publish_rate_hz").value)
        self.publish_grid: bool = bool(self.get_parameter("publish_grid").value)

        # --- Internal state ---
        self.grid: Optional[GridSpec] = None
        self.free_mask: Optional[np.ndarray] = None
        self.covered_mask: Optional[np.ndarray] = None

        self.free_count: int = 0
        self.covered_free_count: int = 0

        self._circle_offsets: Optional[np.ndarray] = None
        self._circle_radius_cells: Optional[int] = None

        # pose tracking for throttling and outlier rejection
        self._last_pose_xy: Optional[Tuple[float, float]] = None
        self._last_pose_t: Optional[float] = None

        # latest covariance estimate (sigma in xy)
        self._last_sigma_xy: Optional[float] = None
        self._last_sigma_t: Optional[float] = None

        # manifest watcher
        self._manifest_mtime: Optional[float] = None

        # --- QoS ---
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # --- Subscribers ---
        self.sub_odom = None
        self.sub_map = None
        self.sub_amcl = None

        if self.mode == "map":
            self.sub_map = self.create_subscription(OccupancyGrid, self.map_topic, self._on_map, qos)
        elif self.mode == "room_manifest":
            self._init_room_from_manifest_or_params()
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        if self.pose_source == "odom":
            self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self._on_odom, qos)
        elif self.pose_source == "tf":
            if Buffer is None:
                raise RuntimeError("tf2_ros is not available, cannot use pose_source=tf")
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)
            self._tf_timer = self.create_timer(0.05, self._poll_tf_pose)  # 20 Hz poll
        else:
            raise ValueError(f"Unsupported pose_source: {self.pose_source}")

        if self.covariance_source == "amcl_pose":
            # Subscribing regardless of pose_source
            self.sub_amcl = self.create_subscription(PoseWithCovarianceStamped, self.amcl_pose_topic, self._on_amcl_pose, qos)

        # --- Publishers ---
        self.pub_percent = self.create_publisher(Float32, "/coverage/percent", 10)
        self.pub_stats = self.create_publisher(String, "/coverage/stats", 10)
        self.pub_grid = self.create_publisher(OccupancyGrid, "/coverage/grid", 10)

        # --- Services ---
        self.srv_reset = self.create_service(Empty, "/coverage/reset", self._on_reset)
        self.srv_report = self.create_service(Trigger, "/coverage/report", self._on_report)

        # --- Timers ---
        if self.publish_rate_hz <= 0:
            self.publish_rate_hz = 2.0
        self._pub_timer = self.create_timer(1.0 / self.publish_rate_hz, self._publish)

        if self.mode == "room_manifest" and self.manifest_path:
            self._manifest_timer = self.create_timer(self.manifest_watch_period, self._poll_manifest)

        self.get_logger().info(
            f"coverage_monitor started: mode={self.mode}, pose_source={self.pose_source}, "
            f"covariance_source={self.covariance_source}, grid_frame={self.grid_frame}, base_frame={self.base_frame}"
        )

    # --- Covariance ---
    @staticmethod
    def _sigma_xy_from_cov(cov: List[float]) -> Optional[float]:
        if not cov or len(cov) < 8:
            return None
        try:
            var_x = float(cov[0])
            var_y = float(cov[7])
            if var_x < 0 or var_y < 0:
                return None
            var = max(var_x, var_y)
            if var <= 0:
                return None
            return float(math.sqrt(var))
        except Exception:
            return None

    def _update_sigma(self, sigma_xy: Optional[float], t: float) -> None:
        if sigma_xy is None:
            return
        self._last_sigma_xy = float(sigma_xy)
        self._last_sigma_t = float(t)

    def _get_sigma(self) -> Optional[float]:
        if self._last_sigma_xy is None or self._last_sigma_t is None:
            return None
        if self.cov_timeout_sec > 0 and (time.time() - self._last_sigma_t) > self.cov_timeout_sec:
            return None
        return self._last_sigma_xy

    # --- Geometry helpers ---
    def _ensure_circle_offsets(self, radius_m: float) -> None:
        if self.grid is None:
            return
        radius_m = max(0.0, float(radius_m))
        r_cells = int(math.ceil(radius_m / self.grid.resolution))
        if r_cells <= 0:
            r_cells = 1
        if self._circle_offsets is not None and self._circle_radius_cells == r_cells:
            return

        pts: List[Tuple[int, int]] = []
        r2 = r_cells * r_cells
        for dy in range(-r_cells, r_cells + 1):
            for dx in range(-r_cells, r_cells + 1):
                if dx * dx + dy * dy <= r2:
                    pts.append((dx, dy))
        self._circle_offsets = np.array(pts, dtype=np.int16)
        self._circle_radius_cells = r_cells

    def _world_to_grid(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        if self.grid is None:
            return None
        gx = int(math.floor((x - self.grid.origin_x) / self.grid.resolution))
        gy = int(math.floor((y - self.grid.origin_y) / self.grid.resolution))
        if gx < 0 or gy < 0 or gx >= self.grid.width or gy >= self.grid.height:
            return None
        return gx, gy

    # --- Acceptance / robustness ---
    def _effective_radius(self) -> float:
        # Return the radius used for painting 
        base = float(self.robot_radius)
        sigma = self._get_sigma() if self.covariance_source in ("odom", "amcl_pose") else None
        if sigma is None or self.radius_mode == "nominal":
            return base

        k = float(self.sigma_multiplier)
        if self.radius_mode == "optimistic":
            return base + k * sigma
        if self.radius_mode == "conservative":
            return max(float(self.radius_min_m), base - k * sigma)

        # fallback
        return base

    def _should_paint_pose(self, x: float, y: float, t: float) -> bool:
        # tiny move throttle
        if self._last_pose_xy is not None:
            lx, ly = self._last_pose_xy
            dx = x - lx
            dy = y - ly
            dist2 = dx * dx + dy * dy
            if dist2 < self.min_pose_delta * self.min_pose_delta:
                return False
            dist = math.sqrt(dist2)
        else:
            self._last_pose_xy = (x, y)
            self._last_pose_t = t
            return True

        # dt
        dt = None
        if self._last_pose_t is not None:
            dt = t - self._last_pose_t
            if dt is not None and dt <= 1e-6:
                dt = None

        # speed gate 
        if dt is not None and self.max_speed_mps > 0:
            speed = dist / dt
            if speed > self.max_speed_mps:

                # update last pose/time baseline but skip painting
                self._last_pose_xy = (x, y)
                self._last_pose_t = t
                return False

        # covariance gate
        if self.max_xy_sigma > 0 and self.covariance_source in ("odom", "amcl_pose"):
            sigma = self._get_sigma()
            if sigma is not None and sigma > self.max_xy_sigma:

                # update last pose/time baseline but skip painting
                self._last_pose_xy = (x, y)
                self._last_pose_t = t
                return False

        # accept
        self._last_pose_xy = (x, y)
        self._last_pose_t = t
        return True

    # --- Painting ---
    def _paint_coverage_at(self, x: float, y: float) -> None:
        # Paint robot footprint onto covered_mask at world position x,y
        if self.grid is None or self.free_mask is None or self.covered_mask is None:
            return

        radius = self._effective_radius()
        self._ensure_circle_offsets(radius)
        if self._circle_offsets is None:
            return

        ij = self._world_to_grid(x, y)
        if ij is None:
            return
        cx, cy = ij

        offsets = self._circle_offsets
        xs = offsets[:, 0].astype(np.int32) + cx
        ys = offsets[:, 1].astype(np.int32) + cy

        inb = (xs >= 0) & (xs < self.grid.width) & (ys >= 0) & (ys < self.grid.height)
        xs = xs[inb]
        ys = ys[inb]

        free = self.free_mask[ys, xs]
        if not np.any(free):
            return

        xs = xs[free]
        ys = ys[free]

        newly = ~self.covered_mask[ys, xs]
        if np.any(newly):
            self.covered_mask[ys[newly], xs[newly]] = True
            self.covered_free_count += int(np.count_nonzero(newly))

    # --- Pose callbacks ---
    def _on_amcl_pose(self, msg: PoseWithCovarianceStamped) -> None:
        sigma = self._sigma_xy_from_cov(list(msg.pose.covariance))
        self._update_sigma(sigma, stamp_to_sec(msg.header.stamp))

    def _on_odom(self, msg: Odometry) -> None:
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        t = stamp_to_sec(msg.header.stamp)

        # update covariance if requested
        if self.covariance_source == "odom":
            sigma = self._sigma_xy_from_cov(list(msg.pose.covariance))
            self._update_sigma(sigma, t)

        # yaw isn't required for disk painting but retained for debugging
        q = msg.pose.pose.orientation
        _yaw = quat_to_yaw(q.x, q.y, q.z, q.w) 

        if self._should_paint_pose(x, y, t):
            self._paint_coverage_at(x, y)

    def _poll_tf_pose(self) -> None:
        if getattr(self, "_tf_buffer", None) is None or self.grid is None:
            return
        try:
            tf = self._tf_buffer.lookup_transform(
                self.grid.frame_id if self.grid.frame_id else self.grid_frame,
                self.base_frame,
                Time()  # latest
            )
            x = float(tf.transform.translation.x)
            y = float(tf.transform.translation.y)
            t = stamp_to_sec(tf.header.stamp) if hasattr(tf, "header") else float(time.time())

            if self._should_paint_pose(x, y, t):
                self._paint_coverage_at(x, y)
        except TransformException:
            return

    # --- Map mode ---
    def _on_map(self, msg: OccupancyGrid) -> None:
        w = int(msg.info.width)
        h = int(msg.info.height)
        res = float(msg.info.resolution)
        ox = float(msg.info.origin.position.x)
        oy = float(msg.info.origin.position.y)
        frame_id = msg.header.frame_id if msg.header.frame_id else self.grid_frame

        data = np.array(msg.data, dtype=np.int16).reshape((h, w))

        if self.include_unknown:
            free = (data == -1) | (data <= self.free_threshold)
        else:
            free = (data != -1) & (data <= self.free_threshold)

        rebuild = (
            self.grid is None
            or self.grid.width != w
            or self.grid.height != h
            or abs(self.grid.resolution - res) > 1e-9
            or abs(self.grid.origin_x - ox) > 1e-9
            or abs(self.grid.origin_y - oy) > 1e-9
            or self.grid.frame_id != frame_id
        )

        self.grid = GridSpec(width=w, height=h, resolution=res, origin_x=ox, origin_y=oy, frame_id=frame_id)
        self.free_mask = free.astype(bool)
        self.free_count = int(np.count_nonzero(self.free_mask))

        if rebuild or self.covered_mask is None or self.covered_mask.shape != (h, w):
            self.covered_mask = np.zeros((h, w), dtype=bool)
            self.covered_free_count = 0
            self._last_pose_xy = None
            self._last_pose_t = None
            self.get_logger().info(f"Map received: {w}x{h} @ {res:.3f} m/cell, free_cells={self.free_count}")
        else:
            # keeping the covered mask but recomputing the covered-free count against the updated free_mask
            self.covered_free_count = int(np.count_nonzero(self.free_mask & self.covered_mask))

        
        self._circle_offsets = None
        self._circle_radius_cells = None

    # --- Room+manifest mode ---
    def _init_room_from_manifest_or_params(self) -> None:
        room_x = self.room_x
        room_y = self.room_y
        res = self.room_resolution
        frame_id = self.grid_frame

        obstacles = []
        manifest = None

        if self.manifest_path and os.path.exists(self.manifest_path):
            try:
                manifest = json.loads(open(self.manifest_path, "r").read())
            except Exception as e:
                self.get_logger().warn(f"Failed to parse manifest '{self.manifest_path}': {e}")

        if isinstance(manifest, dict):
            room_x = float(manifest.get("room_x", room_x))
            room_y = float(manifest.get("room_y", room_y))
            res = float(manifest.get("resolution", res))
            frame_id = str(manifest.get("grid_frame", frame_id))
            obstacles = manifest.get("obstacles", []) if isinstance(manifest.get("obstacles", []), list) else []

        # Building grid spec
        w = int(math.ceil(room_x / res))
        h = int(math.ceil(room_y / res))
        if w <= 1:
            w = 2
        if h <= 1:
            h = 2

        if self.room_centered:
            ox = -0.5 * room_x
            oy = -0.5 * room_y
        else:
            ox = 0.0
            oy = 0.0

        self.grid = GridSpec(width=w, height=h, resolution=res, origin_x=ox, origin_y=oy, frame_id=frame_id)

        self.free_mask = np.ones((h, w), dtype=bool)

        # remove obstacles
        carved = 0
        if isinstance(obstacles, list):
            carved = self._carve_obstacles_from_manifest(obstacles)

        self.free_count = int(np.count_nonzero(self.free_mask))
        self.covered_mask = np.zeros((h, w), dtype=bool)
        self.covered_free_count = 0
        self._last_pose_xy = None
        self._last_pose_t = None

        self._circle_offsets = None
        self._circle_radius_cells = None

        self.get_logger().info(
            f"Room grid init: {w}x{h} @ {res:.3f} m/cell, origin=({ox:.2f},{oy:.2f}), "
            f"free_cells={self.free_count}, obstacles_carved={carved}"
        )

    def _carve_obstacles_from_manifest(self, obstacles: List[dict]) -> int:
        # Removing obstacle footprints from grid, to get actual coverage percentage 
        if self.grid is None or self.free_mask is None:
            return 0

        carved = 0
        for ob in obstacles:
            if not isinstance(ob, dict):
                continue
            shape = str(ob.get("shape", "")).lower()
            pose = ob.get("pose", {})
            if not isinstance(pose, dict):
                continue
            x = float(pose.get("x", 0.0))
            y = float(pose.get("y", 0.0))

            if shape == "box":
                sx = float(ob.get("size_x", ob.get("sx", 0.0)))
                sy = float(ob.get("size_y", ob.get("sy", 0.0)))
                if sx <= 0 or sy <= 0:
                    continue
                self._carve_box(x, y, sx, sy)
                carved += 1
            elif shape == "cylinder":
                r = float(ob.get("radius", ob.get("r", 0.0)))
                if r <= 0:
                    continue
                self._carve_circle(x, y, r)
                carved += 1

        return carved

    def _carve_box(self, cx: float, cy: float, sx: float, sy: float) -> None:
        if self.grid is None or self.free_mask is None:
            return
        
        # axis-aligned carve
        x0 = cx - 0.5 * sx
        x1 = cx + 0.5 * sx
        y0 = cy - 0.5 * sy
        y1 = cy + 0.5 * sy

        gx0 = int(math.floor((x0 - self.grid.origin_x) / self.grid.resolution))
        gx1 = int(math.ceil((x1 - self.grid.origin_x) / self.grid.resolution))
        gy0 = int(math.floor((y0 - self.grid.origin_y) / self.grid.resolution))
        gy1 = int(math.ceil((y1 - self.grid.origin_y) / self.grid.resolution))

        gx0 = int(clamp(gx0, 0, self.grid.width - 1))
        gx1 = int(clamp(gx1, 0, self.grid.width))
        gy0 = int(clamp(gy0, 0, self.grid.height - 1))
        gy1 = int(clamp(gy1, 0, self.grid.height))

        self.free_mask[gy0:gy1, gx0:gx1] = False

    def _carve_circle(self, cx: float, cy: float, radius: float) -> None:
        if self.grid is None or self.free_mask is None:
            return
        ij = self._world_to_grid(cx, cy)
        if ij is None:
            return
        gx, gy = ij
        r_cells = int(math.ceil(radius / self.grid.resolution))
        r2 = r_cells * r_cells

        y0 = max(0, gy - r_cells)
        y1 = min(self.grid.height - 1, gy + r_cells)
        x0 = max(0, gx - r_cells)
        x1 = min(self.grid.width - 1, gx + r_cells)

        for yy in range(y0, y1 + 1):
            dy = yy - gy
            for xx in range(x0, x1 + 1):
                dx = xx - gx
                if dx * dx + dy * dy <= r2:
                    self.free_mask[yy, xx] = False

    def _poll_manifest(self) -> None:
        if not self.manifest_path:
            return
        try:
            if not os.path.exists(self.manifest_path):
                return
            mtime = os.path.getmtime(self.manifest_path)
            if self._manifest_mtime is None:
                self._manifest_mtime = mtime
                return
            if mtime != self._manifest_mtime:
                self._manifest_mtime = mtime
                if self.reset_on_manifest_change:
                    self.get_logger().info("Manifest changed, resetting coverage grid.")
                    self._init_room_from_manifest_or_params()
        except Exception:
            return

    # --- Publish / services ---
    def _compute_stats(self) -> Dict[str, object]:
        if self.grid is None or self.free_mask is None:
            return {
                "coverage_percent": 0.0,
                "free_cells": 0,
                "covered_free_cells": 0,
                "resolution": None,
                "free_area_m2": 0.0,
                "covered_free_area_m2": 0.0,
                "mode": self.mode,
            }
        free = int(self.free_count)
        covered = int(self.covered_free_count)
        res = float(self.grid.resolution)
        cell_area = res * res
        cov = 0.0 if free <= 0 else 100.0 * (covered / free)

        sigma = self._get_sigma() if self.covariance_source in ("odom", "amcl_pose") else None
        return {
            "coverage_percent": float(cov),
            "free_cells": free,
            "covered_free_cells": covered,
            "resolution": res,
            "free_area_m2": float(free * cell_area),
            "covered_free_area_m2": float(covered * cell_area),
            "frame_id": self.grid.frame_id,
            "mode": self.mode,
            "pose_source": self.pose_source,
            "robot_radius": float(self.robot_radius),
            "effective_radius": float(self._effective_radius()),
            "sigma_xy_m": float(sigma) if sigma is not None else None,
            "timestamp_unix": time.time(),
        }

    def _build_coverage_grid_msg(self) -> Optional[OccupancyGrid]:
        if self.grid is None or self.free_mask is None or self.covered_mask is None:
            return None
        h, w = self.free_mask.shape

        out = np.full((h, w), 100, dtype=np.int8)
        out[self.free_mask] = 0
        out[self.free_mask & self.covered_mask] = 50

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.grid.frame_id if self.grid.frame_id else self.grid_frame

        msg.info.resolution = float(self.grid.resolution)
        msg.info.width = int(self.grid.width)
        msg.info.height = int(self.grid.height)
        msg.info.origin.position.x = float(self.grid.origin_x)
        msg.info.origin.position.y = float(self.grid.origin_y)
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        msg.data = out.reshape(-1).tolist()
        return msg

    def _publish(self) -> None:
        stats = self._compute_stats()

        msg = Float32()
        msg.data = float(stats["coverage_percent"])
        self.pub_percent.publish(msg)

        s = String()
        s.data = json.dumps(stats, separators=(",", ":"), sort_keys=True)
        self.pub_stats.publish(s)

        if self.publish_grid:
            grid_msg = self._build_coverage_grid_msg()
            if grid_msg is not None:
                self.pub_grid.publish(grid_msg)

    def _on_reset(self, _req, _res):
        self._reset_coverage()
        return _res

    def _on_report(self, _req, res: Trigger.Response):
        stats = self._compute_stats()
        res.success = True
        res.message = json.dumps(stats, separators=(",", ":"), sort_keys=True)
        return res

    def _reset_coverage(self) -> None:
        if self.grid is None:
            return
        if self.covered_mask is not None:
            self.covered_mask[:, :] = False
        self.covered_free_count = 0
        self._last_pose_xy = None
        self._last_pose_t = None
        self.get_logger().info("Coverage reset.")

def main(args=None) -> None:
    rclpy.init(args=args)
    node = CoverageMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
