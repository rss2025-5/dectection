#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from enum import Enum
import math
import time

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

class HeistState(Enum):
    WAITING_FOR_LOCATIONS = 0
    NAVIGATING_TO_1 = 1
    AT_LOCATION_1 = 2
    NAVIGATING_TO_2 = 3
    AT_LOCATION_2 = 4
    ESCAPING = 5
    FINISHED = 6

class StateMachine(Node):
    def __init__(self):
        super().__init__('state_machine')

        # Publisher to send goals to planner
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # Subscriber to receive odometry
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Subscriber to clicked points (reuse goal_pose topic as clicked input)
        self.clicked_sub = self.create_subscription(PoseStamped, '/goal_pose', self.clicked_callback, 10)

        self.state = HeistState.WAITING_FOR_LOCATIONS
        self.location1 = None
        self.location2 = None
        self.start_location = None
        self.got_goals = 0

        self.current_pos = None

        self.timer = self.create_timer(0.1, self.state_machine_step)  # 10Hz

    def clicked_callback(self, msg):
        if self.got_goals == 0:
            self.location1 = msg
            self.get_logger().info(f"Location 1 saved: ({msg.pose.position.x}, {msg.pose.position.y})")
            self.got_goals += 1
        elif self.got_goals == 1:
            self.location2 = msg
            self.get_logger().info(f"Location 2 saved: ({msg.pose.position.x}, {msg.pose.position.y})")
            self.got_goals += 1

            # Save starting pose (first odom received after goals)
            if self.current_pos is not None:
                self.start_location = self.create_pose_stamped(self.current_pos[0], self.current_pos[1])
                self.get_logger().info(f"Start location saved at ({self.current_pos[0]}, {self.current_pos[1]})")

            # Send first goal
            self.publish_goal(self.location1)
            self.state = HeistState.NAVIGATING_TO_1

    def odom_callback(self, msg):
        self.current_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def state_machine_step(self):
        if self.current_pos is None:
            return  # no odometry yet

        if self.state == HeistState.NAVIGATING_TO_1:
            if self.is_close(self.current_pos, self.location1.pose.position):
                self.get_logger().info("Arrived at location 1 → pickup starting")
                self.stop_robot()
                time.sleep(5)  # simulate 5 sec pickup
                self.publish_goal(self.location2)
                self.state = HeistState.NAVIGATING_TO_2

        elif self.state == HeistState.NAVIGATING_TO_2:
            if self.is_close(self.current_pos, self.location2.pose.position):
                self.get_logger().info("Arrived at location 2 → pickup starting")
                self.stop_robot()
                time.sleep(5)  # simulate 5 sec pickup
                self.publish_goal(self.start_location)
                self.state = HeistState.ESCAPING

        elif self.state == HeistState.ESCAPING:
            if self.is_close(self.current_pos, self.start_location.pose.position):
                self.get_logger().info("Escaped back to start! Heist complete 🎉")
                self.stop_robot()
                self.state = HeistState.FINISHED

    def publish_goal(self, pose_msg):
        self.goal_pub.publish(pose_msg)
        self.get_logger().info(f"Published goal: ({pose_msg.pose.position.x}, {pose_msg.pose.position.y})")

    def stop_robot(self):
        stop_goal = PoseStamped()
        stop_goal.header.frame_id = 'map'
        stop_goal.pose.position.x = self.current_pos[0]
        stop_goal.pose.position.y = self.current_pos[1]
        self.goal_pub.publish(stop_goal)  # optional: sends dummy goal to hold position

    def is_close(self, current, target, threshold=0.4):
        dx = current[0] - target.x
        dy = current[1] - target.y
        return math.hypot(dx, dy) < threshold

    def create_pose_stamped(self, x, y):
        ps = PoseStamped()
        ps.header.frame_id = 'map'
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.orientation.w = 1.0
        return ps

def main(args=None):
    rclpy.init(args=args)
    sm = StateMachine()
    rclpy.spin(sm)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
