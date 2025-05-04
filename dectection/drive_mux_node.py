#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from ackermann_msgs.msg import AckermannDriveStamped

class DriveMux(Node):
    def __init__(self):
        super().__init__('drive_mux')

        self.controller_mode = "FOLLOWING"  # FOLLOWING or AVOIDING

        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/vesc/low_level/input/navigation", 10)

        self.follower_sub = self.create_subscription(AckermannDriveStamped, "/drive_follower", self.follower_cb, 10)
        self.wall_sub = self.create_subscription(AckermannDriveStamped, "/drive_wall", self.wall_cb, 10)

        self.obstacle_sub = self.create_subscription(Bool, "/obstacle_detected", self.obstacle_cb, 10)

    def obstacle_cb(self, msg):
        if msg.data and self.controller_mode != "AVOIDING":
            self.get_logger().info("Obstacle detected → switching to WALL FOLLOWER")
            self.controller_mode = "AVOIDING"
        elif not msg.data and self.controller_mode != "FOLLOWING":
            self.get_logger().info("Obstacle cleared → switching to TRAJECTORY FOLLOWER")
            self.controller_mode = "FOLLOWING"

    def follower_cb(self, msg):
        if self.controller_mode == "FOLLOWING":
            self.get_logger().info("NAVIGATING")
            self.drive_pub.publish(msg)

    def wall_cb(self, msg):
        if self.controller_mode == "AVOIDING":
            self.get_logger().info("WALL FOLLOWING")
            self.drive_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    mux = DriveMux()
    rclpy.spin(mux)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
