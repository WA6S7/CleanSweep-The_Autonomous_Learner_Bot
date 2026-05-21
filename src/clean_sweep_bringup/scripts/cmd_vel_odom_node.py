#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.time import Time

from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster

#   cmd_vel_odom_node.py  —  Layer 1: Dead-reckoning odometry

#   Produces /odom by fusing:
#    /cmd_vel  (commanded linear & angular velocity)
#    /imu      (MPU-6050 gyroscope)

def yaw_from_quat(q):
    """Extract yaw from a quaternion (geometry_msgs style)."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def quat_from_yaw(yaw):
    """Return (x, y, z, w) quaternion for a pure-yaw rotation."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class CmdVelOdomNode(Node):

    def __init__(self):
        super().__init__("cmd_vel_odom_node")

        # --- Parameters ---
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("use_imu_heading", True)
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("publish_tf", True)

        # For linear speed scale, cmd_vel does not perfectly match real speed.
        self.declare_parameter("linear_scale", 1.0)
        self.declare_parameter("initial_x", 0.0)
        self.declare_parameter("initial_y", 0.0)
        self.declare_parameter("initial_yaw", 0.0)
        
        self.declare_parameter("gyro_sign", 1.0)

        self.rate_hz = self.get_parameter("publish_rate_hz").value
        self.use_imu = self.get_parameter("use_imu_heading").value
        self.odom_frame = self.get_parameter("odom_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.do_tf = self.get_parameter("publish_tf").value
        self.linear_scale = self.get_parameter("linear_scale").value
        self.initial_yaw = self.get_parameter("initial_yaw").value
        self.gyro_sign = float(self.get_parameter("gyro_sign").value)

        # --- State ---
        self.x = self.get_parameter("initial_x").value
        self.y = self.get_parameter("initial_y").value
        self.yaw = self.initial_yaw

        self.cmd_vx = 0.0       # latest commanded linear.x
        self.cmd_wz = 0.0       # latest commanded angular.z

        self.imu_yaw = None     # latest IMU-derived yaw 
        self.imu_wz = 0.0       # latest IMU gyro z rate
        self._imu_yaw_offset = None  # offset so IMU yaw starts at 0

        self._last_stamp = None

        # --- QoS ---
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # --- Subscribers ---
        self.create_subscription(Twist, "/cmd_vel", self._cmd_vel_cb, reliable_qos)
        self.create_subscription(Imu, "/imu", self._imu_cb, sensor_qos)

        # --- Publishers ---
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.tf_broadcaster = TransformBroadcaster(self) if self.do_tf else None

        # --- Timer ---
        self.create_timer(1.0 / self.rate_hz, self._tick)

        self.get_logger().info(
            f"cmd_vel_odom_node started  "
            f"rate={self.rate_hz}Hz  imu_heading={self.use_imu}  "
            f"linear_scale={self.linear_scale}"
        )

    # --- Callbacks ---

    def _cmd_vel_cb(self, msg: Twist):
        self.cmd_vx = msg.linear.x
        self.cmd_wz = msg.angular.z

    def _imu_cb(self, msg: Imu):
        # Gyroscope z-rate (rad/s)
        self.imu_wz = msg.angular_velocity.z * self.gyro_sign

        # Absolute orientation 
        q = msg.orientation
        if q.w == 0.0 and q.x == 0.0 and q.y == 0.0 and q.z == 0.0:
            # Some IMU drivers leave orientation zeroed out; skip
            return

        raw_yaw = yaw_from_quat(q) * self.gyro_sign

        # Initialise offset so IMU delta starts at 0 — our absolute heading
        # then becomes initial_yaw + delta.
        if self._imu_yaw_offset is None:
            self._imu_yaw_offset = raw_yaw

        self.imu_yaw = raw_yaw - self._imu_yaw_offset

    # --- Integration tick ---

    def _tick(self):
        now = self.get_clock().now()
        stamp = now.to_msg()

        if self._last_stamp is None:
            self._last_stamp = now
            return

        dt = (now - self._last_stamp).nanoseconds * 1e-9
        self._last_stamp = now

        if dt <= 0.0 or dt > 1.0:
            return  # skip bad intervals

        # --- Heading update ---
        if self.use_imu and self.imu_yaw is not None:
            # IMU delta is added to launched initial yaw
            self.yaw = self.initial_yaw + self.imu_yaw
        else:
            # Fallback: integrate gyro rate (already sign-corrected) or cmd_vel
            wz = self.imu_wz if self.use_imu else self.cmd_wz
            self.yaw += wz * dt

        # Normalise yaw to [-pi, pi]
        self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))

        # --- Position update ---
        vx = self.cmd_vx * self.linear_scale
        self.x += vx * math.cos(self.yaw) * dt
        self.y += vx * math.sin(self.yaw) * dt

        # --- Publish odometry ---
        qx, qy, qz, qw = quat_from_yaw(self.yaw)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        # Covariance — cmd_vel dead reckoning is low-confidence
        # [x, y, z, roll, pitch, yaw] — 6x6 diagonal
        pc = odom.pose.covariance
        pc[0] = 0.05    # x variance  (drifts ~22 cm/m)
        pc[7] = 0.05    # y variance
        pc[35] = 0.02   # yaw variance

        odom.twist.twist.linear.x = vx
        odom.twist.twist.angular.z = self.imu_wz if self.use_imu else self.cmd_wz
        tc = odom.twist.covariance
        tc[0] = 0.01
        tc[35] = 0.01

        self.odom_pub.publish(odom)

        # --- Broadcast TF ---
        if self.tf_broadcaster is not None:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self.odom_frame
            t.child_frame_id = self.base_frame
            t.transform.translation.x = self.x
            t.transform.translation.y = self.y
            t.transform.rotation.x = qx
            t.transform.rotation.y = qy
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
            self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelOdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
