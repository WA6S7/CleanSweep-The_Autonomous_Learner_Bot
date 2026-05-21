#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Imu

import os
import math
import time
import termios
import threading
import subprocess
import std_msgs.msg

BAUD_RATE    = 115200

# Setting the valid ultrasonic range
RANGE_MIN_M  = 0.02
RANGE_MAX_M  = 3.40


class SerialBridgeNode(Node):

    def __init__(self):
        super().__init__('serial_bridge_node')

        self.declare_parameter('serial_port', '/dev/ttyACM0')
        serial_port = self.get_parameter('serial_port').value

        try:
            # Configuring the port with stty 
            subprocess.run(
                ['stty', '-F', serial_port, str(BAUD_RATE), 'raw', '-echo', '-hupcl'],
                check=True, capture_output=True)

            # Open with raw file I/O
            self.fd = os.open(serial_port, os.O_RDWR | os.O_NOCTTY)

            attrs = termios.tcgetattr(self.fd)
            attrs[2] = attrs[2] & ~termios.HUPCL
            termios.tcsetattr(self.fd, termios.TCSANOW, attrs)

            time.sleep(3.0)  

            self.get_logger().info(f'Connected to Arduino on {serial_port}')
        except Exception as e:
            self.get_logger().error(f'Failed to open serial port: {e}')
            raise

        self.cmd_sub   = self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)
        self.servo_sub = self.create_subscription(
            std_msgs.msg.Int16, '/servo_angle', self.servo_callback, 10)
        self.front_pub = self.create_publisher(LaserScan, '/ultrasonic_front', 10)
        self.rear_pub  = self.create_publisher(LaserScan, '/ultrasonic_rear',  10)
        self.imu_pub   = self.create_publisher(Imu,       '/imu',              10)

        self.running = True
        self.read_thread = threading.Thread(target=self.read_loop, daemon=True)
        self.read_thread.start()

        self.get_logger().info('Serial bridge node started.')

    def cmd_callback(self, msg: Twist):
        # Sending the velocity command to Arduino 
        line = f'V{msg.linear.x:.3f},{msg.angular.z:.3f}\n'
        try:
            os.write(self.fd, line.encode('utf-8'))
        except Exception as e:
            self.get_logger().warn(f'Serial write error: {e}')

    def servo_callback(self, msg: std_msgs.msg.Int16):
        angle = max(30, min(150, msg.data))
        try:
            os.write(self.fd, f'A{angle}\n'.encode('utf-8'))
        except Exception as e:
            self.get_logger().warn(f'Servo write error: {e}')

    def read_loop(self):
        # Reading raw serial bytes and splitting them into newline and terminated lines
        buffer = b''
        count = 0
        self.get_logger().info('Read thread started, waiting for data...')
        while self.running:
            try:
                chunk = os.read(self.fd, 512)
                if not chunk:
                    time.sleep(0.01)
                    continue
                if count == 0:
                    self.get_logger().info(f'First bytes received: {chunk[:40]}')
                buffer += chunk 
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    text = line.decode('utf-8', errors='ignore').strip()
                    count += 1
                    if count <= 5:
                        self.get_logger().info(f'Serial line [{count}]: {text[:80]}')
                    self.parse_state(text)
            except OSError as e:
                self.get_logger().warn(f'Serial read error: {e}')
                time.sleep(0.1)

    def parse_state(self, line: str):
        # State lines look like "S<front>,<rear>,<ax>,<ay>,<az>,<gx>,<gy>,<gz>"
        if not line.startswith('S'):
            return
        try:
            parts = line[1:].split(',')
            if len(parts) != 8:
                return
            dist_front = float(parts[0])
            dist_rear  = float(parts[1])
            ax, ay, az = float(parts[2]), float(parts[3]), float(parts[4])
            gx, gy, gz = float(parts[5]), float(parts[6]), float(parts[7])
        except ValueError:
            return  # incorrect line — drop it

        now = self.get_clock().now()

        # Ultrasonics arrive in cm
        front_m = dist_front / 100.0 if dist_front < 998.0 else float('inf')
        self.front_pub.publish(self._make_scan(now, front_m))

        rear_m = dist_rear / 100.0 if dist_rear < 998.0 else float('inf')
        self.rear_pub.publish(self._make_scan(now, rear_m))

        # Raw MPU-6050 accel + gyro
        imu = Imu()
        imu.header.stamp    = now.to_msg()
        imu.header.frame_id = 'base_footprint'
        imu.linear_acceleration.x = ax
        imu.linear_acceleration.y = ay
        imu.linear_acceleration.z = az
        imu.angular_velocity.x = gx
        imu.angular_velocity.y = gy
        imu.angular_velocity.z = gz
        imu.orientation_covariance[0] = -1.0
        self.imu_pub.publish(imu)

    def _make_scan(self, now, range_m):
        # Wrapping a single ultrasonic reading as a 1-beam LaserScan
        scan = LaserScan()
        scan.header.stamp    = now.to_msg()
        scan.header.frame_id = 'base_footprint'
        scan.angle_min       = 0.0
        scan.angle_max       = 0.0
        scan.angle_increment = 0.0
        scan.time_increment  = 0.0
        scan.scan_time       = 0.05
        scan.range_min       = RANGE_MIN_M
        scan.range_max       = RANGE_MAX_M
        scan.ranges          = [range_m]
        return scan

    def destroy_node(self):
        # Stopping the motors and releasing the port on shutdown
        self.running = False
        try:
            os.write(self.fd, b'V0.0,0.0\n')
            os.close(self.fd)
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SerialBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
