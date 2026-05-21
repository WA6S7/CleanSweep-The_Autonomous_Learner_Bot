#!/usr/bin/env python3
"""
aruco_localizer_node.py  —  Layer 3: ArUco marker-based localisation
═════════════════════════════════════════════════════════════════════
Detects ArUco 4×4_1000 markers in the camera image, computes the
camera-to-marker transform via solvePnP, then converts that into
a robot world pose using the known marker map.

Subscribes:
  /camera/image_raw       sensor_msgs/Image
  /camera/camera_info     sensor_msgs/CameraInfo

Publishes:
  /aruco/pose             geometry_msgs/PoseWithCovarianceStamped
  /aruco/markers          sensor_msgs/Image  (debug: annotated frame)
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseWithCovarianceStamped

try:
    import cv2
    from cv2 import aruco
except ImportError:
    cv2 = None
    aruco = None

try:
    from cv_bridge import CvBridge
except ImportError:
    CvBridge = None



MARKER_SIZE_M = 0.15    # default side length (15 cm)
MARKER_SIZE_SMALL = 0.142  # IDs 4–9 were printed slightly smaller as they were printed later

DEFAULT_MARKER_MAP = {
    0: {"x": -1.385, "y": -0.748, "z": 0.165, "yaw":  0.0},
    1: {"x":  1.085, "y": -1.65,  "z": 0.165, "yaw":  math.pi / 2.0},
    2: {"x":  1.385, "y":  1.277, "z": 0.165, "yaw":  math.pi},
    3: {"x":  1.044, "y":  1.65,  "z": 0.165, "yaw": -math.pi / 2.0},
    4: {"x":  1.385, "y": -1.037, "z": 0.165, "yaw":  math.pi,        "size": MARKER_SIZE_SMALL},
    5: {"x":  1.03,  "y": -0.36,  "z": 0.165, "yaw": -math.pi / 2.0,  "size": MARKER_SIZE_SMALL},
    6: {"x": -0.419, "y": -0.22,  "z": 0.165, "yaw":  0.0,        "size": MARKER_SIZE_SMALL},
    7: {"x":  0.799, "y": -0.181, "z": 0.165, "yaw":  math.pi,            "size": MARKER_SIZE_SMALL},
    8: {"x": -0.165, "y": -1.06,  "z": 0.165, "yaw":  math.pi / 2.0,  "size": MARKER_SIZE_SMALL},
    9: {"x":  0.425, "y":  1.203, "z": 0.165, "yaw": -math.pi / 2.0,  "size": MARKER_SIZE_SMALL},
}


def _rotation_matrix_z(yaw):
    """3×3 rotation about Z axis."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def _build_marker_poses(marker_map, default_marker_size):
    """
    Pre-compute, for every marker ID:
      • T_world_marker  (4×4 homogeneous: world ← marker-centre)
      • obj_points      (4×3 float32: marker corner coords in marker frame)

    Each marker entry may carry its own "size" key (side length in metres);
    if absent, default_marker_size is used. The marker frame has Z pointing
    out of the wall (into the room), X to the right along the wall, Y upward.
    """
    
    R_base = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    poses = {}
    for mid, m in marker_map.items():
        size = m.get("size", default_marker_size)
        half = size / 2.0
        
        obj_pts = np.array([
            [-half,  half, 0.0],
            [ half,  half, 0.0],
            [ half, -half, 0.0],
            [-half, -half, 0.0],
        ], dtype=np.float32)

        R = _rotation_matrix_z(m["yaw"]) @ R_base
        t = np.array([m["x"], m["y"], m["z"]])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        poses[mid] = {"T_world_marker": T, "obj_points": obj_pts, "size": size}
    return poses


class ArucoLocalizerNode(Node):

    def __init__(self):
        super().__init__("aruco_localizer_node")

        if cv2 is None:
            self.get_logger().error("OpenCV (cv2) not found — cannot run ArUco detection")
            return
        if CvBridge is None:
            self.get_logger().error("cv_bridge not found — install ros-<distro>-cv-bridge")
            return

        # --- Parameters ---
        self.declare_parameter("process_rate_hz", 5.0)
        self.declare_parameter("marker_size_m", MARKER_SIZE_M)
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("map_frame", "odom")

        self.declare_parameter("cam_x_on_robot", 0.08)   # 8 cm forward of base_link centre
        self.declare_parameter("cam_z_on_robot", 0.15)    # 15 cm above ground
        self.declare_parameter("cam_pitch_rad", -0.349)   # −20° (tilted down)

        self.process_hz = self.get_parameter("process_rate_hz").value
        self.marker_size = self.get_parameter("marker_size_m").value
        self.debug_img = self.get_parameter("publish_debug_image").value
        self.base_frame = self.get_parameter("base_frame").value
        self.map_frame = self.get_parameter("map_frame").value
        self.cam_x = self.get_parameter("cam_x_on_robot").value
        self.cam_z = self.get_parameter("cam_z_on_robot").value
        self.cam_pitch = self.get_parameter("cam_pitch_rad").value

        # --- ArUco setup ---
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
        self.aruco_params = aruco.DetectorParameters()

        # --- Marker map ---
        self.marker_poses = _build_marker_poses(DEFAULT_MARKER_MAP, self.marker_size)

        # --- Camera-to-robot transform ---
        self._T_base_cam = self._build_T_base_cam()

        # --- Camera intrinsics ---
        self.camera_matrix = None
        self.dist_coeffs = None

        # --- CV bridge ---
        self.bridge = CvBridge()

        # --- Rate limiting ---
        self._min_period_ns = int(1e9 / self.process_hz)
        self._last_process_ns = 0

        # --- QoS ---
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # --- Subscribers ---
        self.create_subscription(Image, "/camera/image_raw",
                                 self._image_cb, sensor_qos)
        self.create_subscription(CameraInfo, "/camera/camera_info",
                                 self._camera_info_cb, sensor_qos)

        # --- Publishers ---
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/aruco/pose", sensor_qos)

        self.debug_pub = None
        if self.debug_img:
            self.debug_pub = self.create_publisher(Image, "/aruco/markers", 10)

        self.get_logger().info(
            f"aruco_localizer_node started  "
            f"rate={self.process_hz}Hz  markers={list(DEFAULT_MARKER_MAP.keys())}  "
            f"size={self.marker_size}m"
        )

    # --- Camera-to-robot transform ---

    def _build_T_base_cam(self):
       
        #   T_base_camBody (camera body frame on the robot)
        #   camera body: X right, Y down, Z forward (before optical transform)

        #   Rotation: base_footprint ← camera_optical
        #   camera_optical: Z forward, X right, Y down
        #   base_footprint: X forward, Y left, Z up

        R_base_opt_no_tilt = np.array([
            [0.0,  0.0,  1.0],   # base X = cam Z
            [-1.0, 0.0,  0.0],   # base Y = −cam X
            [0.0, -1.0,  0.0],   # base Z = −cam Y
        ])

        # Apply camera pitch as rotation about the
        # camera-optical X axis.  In the optical frame X = right,
        # Y = down, Z = forward, so rotating about X tips Z toward/away
        # from Y — i.e. tilts the view up/down.  cam_pitch < 0 → down.
        cp = math.cos(self.cam_pitch)
        sp = math.sin(self.cam_pitch)
        R_pitch = np.array([
            [1.0,  0.0,  0.0],
            [0.0,  cp,  -sp],
            [0.0,  sp,   cp],
        ])

        R = R_base_opt_no_tilt @ R_pitch

        t = np.array([self.cam_x, 0.0, self.cam_z])

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    # --- Callbacks ---

    def _camera_info_cb(self, msg: CameraInfo):
        if self.camera_matrix is not None:
            return  # already have it
        K = np.array(msg.k).reshape(3, 3)
        if K[0, 0] == 0.0:
            self.get_logger().warn(
                "CameraInfo has zero intrinsics (no calibration file loaded)")
            return
        self.camera_matrix = K
        self.dist_coeffs = np.array(msg.d) if len(msg.d) > 0 else np.zeros(5)
        self.get_logger().info(
            f"Camera intrinsics received: fx={K[0,0]:.1f} fy={K[1,1]:.1f}")

    def _image_cb(self, msg: Image):
        # Rate-limit
        now_ns = self.get_clock().now().nanoseconds
        if (now_ns - self._last_process_ns) < self._min_period_ns:
            return
        self._last_process_ns = now_ns

        # Fallback: estimate intrinsics from image size if CameraInfo unavailable
        if self.camera_matrix is None:
            if msg.width > 0 and msg.height > 0 and not hasattr(self, '_intrinsics_warned'):
                self._intrinsics_warned = True
                
                fx = float(msg.width) * 1.03
                fy = fx * 1.16  # slight rectangular pixel correction
                cx = float(msg.width) / 2.0
                cy = float(msg.height) / 2.0
                self.camera_matrix = np.array([
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0]])
                self.dist_coeffs = np.zeros(5)
                self.get_logger().warn(
                    f"No calibration file — using estimated intrinsics: "
                    f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}. "
                    f"Pose accuracy will be reduced. Recalibrate for best results.")

        # Convert to OpenCV
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge error: {e}")
            return

        h, w = frame.shape[:2]

        # Convert to grayscale for detection 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, rejected = aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)

        n_det = 0 if ids is None else len(ids)
        n_rej = 0 if rejected is None else len(rejected)
        self.get_logger().info(
            f"Frame {w}x{h}  detected={n_det}  rejected={n_rej}",
            throttle_duration_sec=2.0)

        if self.debug_pub is not None:
            debug_frame = frame.copy()
            if ids is not None and len(ids) > 0:
                aruco.drawDetectedMarkers(debug_frame, corners, ids)
            if rejected:
                aruco.drawDetectedMarkers(debug_frame, rejected,
                                          borderColor=(100, 0, 255))
            debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding="bgr8")
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)

        if ids is None or len(ids) == 0:
            return

        # Process each detected marker
        best_pose = None
        best_dist = float("inf")
        best_id = -1

        for i, marker_id in enumerate(ids.flatten()):
            if marker_id not in self.marker_poses:
                continue

            mp = self.marker_poses[marker_id]
            obj_pts = mp["obj_points"]
            img_pts = corners[i].reshape(4, 2)

            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts,
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not ok:
                continue

            # Distance to marker 
            dist = np.linalg.norm(tvec)

            # Build T_cam_marker (4×4)
            R_cam_marker, _ = cv2.Rodrigues(rvec)
            T_cam_marker = np.eye(4)
            T_cam_marker[:3, :3] = R_cam_marker
            T_cam_marker[:3, 3] = tvec.flatten()


            T_world_marker = mp["T_world_marker"]
            T_marker_cam = np.linalg.inv(T_cam_marker)
            T_cam_base = np.linalg.inv(self._T_base_cam)

            T_world_base = T_world_marker @ T_marker_cam @ T_cam_base

            # log computed pose for every marker detection
            _x = T_world_base[0, 3]
            _y = T_world_base[1, 3]
            _yaw = math.atan2(T_world_base[1, 0], T_world_base[0, 0])
            self.get_logger().info(
                f"Marker {marker_id}: dist={dist:.2f}m  "
                f"→ robot ({_x:.3f}, {_y:.3f}) yaw={math.degrees(_yaw):.1f}°")

            if dist < best_dist:
                best_dist = dist
                best_pose = T_world_base
                best_id = marker_id

        if best_pose is None:
            return

        # Extract x, y, yaw from T_world_base
        x = best_pose[0, 3]
        y = best_pose[1, 3]
        yaw = math.atan2(best_pose[1, 0], best_pose[0, 0])

        self.get_logger().info(
            f"PUBLISH m{best_id} d={best_dist:.2f}m  "
            f"pose=({x:.3f}, {y:.3f}) yaw={math.degrees(yaw):.1f}°")

        # Publish PoseWithCovarianceStamped
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = msg.header.stamp
        pose_msg.header.frame_id = self.map_frame

        pose_msg.pose.pose.position.x = x
        pose_msg.pose.pose.position.y = y
        pose_msg.pose.pose.position.z = 0.0

        qx, qy, qz, qw = quat_from_yaw(yaw)
        pose_msg.pose.pose.orientation.x = qx
        pose_msg.pose.pose.orientation.y = qy
        pose_msg.pose.pose.orientation.z = qz
        pose_msg.pose.pose.orientation.w = qw

        # Covariance: scale with distance
        # At 1m: ~3cm position uncertainty, ~5° heading uncertainty
        d2 = max(0.25, best_dist * best_dist)
        xy_var = 0.001 * d2      # ~3cm at 1m, ~6cm at 2m
        yaw_var = 0.008 * d2     # ~5° at 1m

        cov = pose_msg.pose.covariance
        cov[0] = xy_var       # x
        cov[7] = xy_var       # y
        cov[35] = yaw_var     # yaw

        self.pose_pub.publish(pose_msg)


def quat_from_yaw(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def main(args=None):
    rclpy.init(args=args)
    node = ArucoLocalizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
