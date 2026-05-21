#!/usr/bin/env python3

import subprocess, threading, math, rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

ROBOT_NAME = "my_robot"
WORLD_NAME = "empty"

def yaw_from_quat(x, y, z, w):
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)

class GroundTruthPublisher(Node):

    def __init__(self):
        super().__init__("ground_truth_publisher")
        self.pub = self.create_publisher(Odometry, "/ground_truth_pose", 10)
        
        # Publishing /odometry/filtered as well so the sim matches the hardware topic
        # layout (where EKF fuses odom+IMU+ArUco into /odometry/filtered).
        self.pub_filtered = self.create_publisher(Odometry, "/odometry/filtered", 10)

        self._x = 0.0; self._y = 0.0
        self._qx = 0.0; self._qy = 0.0
        self._qz = 0.0; self._qw = 1.0
        self._ready = False
        self._lock = threading.Lock()  # guards the latest pose shared with the timer

        # Single persistent `gz topic` process streaming Gazebo's dynamic pose info
        self._proc = subprocess.Popen(
            ["gz", "topic", "-e", "-t",
             f"/world/{WORLD_NAME}/dynamic_pose/info"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

        # Parsing that stream off-thread so the ROS timer stays responsive
        self._thread = threading.Thread(
            target=self._parse_loop, daemon=True)
        self._thread.start()

        self.create_timer(0.02, self._publish)  # 50 Hz publish
        self.get_logger().info("Ground truth publisher ready")

    def _parse_loop(self):
        in_robot = in_pos = in_ori = False
        x = y = qx = qy = qz = qw = None

        for line in self._proc.stdout:
            s = line.strip()

            # Start of the robot's pose entry
            if f'name: "{ROBOT_NAME}"' in s:
                in_robot = True
                in_pos = in_ori = False
                x = y = qx = qy = qz = qw = None
                continue

            if not in_robot:
                continue

            if s == "position {":
                in_pos = True; continue
            if s == "orientation {":
                in_ori = True; continue

            if in_pos:
                if   s.startswith("x:"): x = float(s.split(":")[1])
                elif s.startswith("y:"): y = float(s.split(":")[1])
                elif s == "}": in_pos = False

            elif in_ori:
                if   s.startswith("x:"): qx = float(s.split(":")[1])
                elif s.startswith("y:"): qy = float(s.split(":")[1])
                elif s.startswith("z:"): qz = float(s.split(":")[1])
                elif s.startswith("w:"): qw = float(s.split(":")[1])
                elif s == "}":
                    # End of orientation
                    in_ori = False
                    in_robot = False
                    if None not in (x, y, qx, qy, qz, qw):
                        with self._lock:
                            self._x=x; self._y=y
                            self._qx=qx; self._qy=qy
                            self._qz=qz; self._qw=qw
                            self._ready = True

    def _publish(self):
        # copying the latest pose under lock, then publishing it
        with self._lock:
            if not self._ready:
                return 
            x=self._x; y=self._y
            qx=self._qx; qy=self._qy
            qz=self._qz; qw=self._qw

        msg = Odometry()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.child_frame_id  = "base_footprint"
        msg.pose.pose.position.x    = x
        msg.pose.pose.position.y    = y
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        self.pub.publish(msg)
        self.pub_filtered.publish(msg)

    def destroy_node(self):
        self._proc.terminate()  # ending the gz topic subprocess on shutdown
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GroundTruthPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()