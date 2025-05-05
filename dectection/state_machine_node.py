#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from enum import Enum
import math
import time

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool

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
        self.safety_stop = self.create_publisher(AckermannDriveStamped, '/vesc/low_level/input/safety', 10)
        self.ready_to_save = self.create_publisher(Bool, '/ready_save', 10)
        self.end_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # Subscriber to receive odometry
        self.odom_sub = self.create_subscription(Odometry, '/pf/pose/odom', self.odom_callback, 10)
        self.init_sub = self.create_subscription(PoseStamped, '/initialpose', self.init_cb, 10)
        # Subscriber to clicked points (reuse goal_pose topic as clicked input)
        self.clicked_sub = self.create_subscription(PoseStamped, '/goal_pose', self.clicked_callback, 10)

        self.state = HeistState.WAITING_FOR_LOCATIONS
        self.location1 = None
        self.location2 = None
        self.start_location = None


        self.current_pos = None
        self.rotating = True
        self.turn_stage = 1
        self.rotate_start_time = time.time()
        self.timer = self.create_timer(0.1, self.state_machine_step)  # 10Hz

    def init_cb(self, msg):
        self.start_location = msg

    def clicked_callback(self, msg):
        self.state = HeistState.NAVIGATING_TO_1
        if self.location1 is None:
            self.location1 = msg
            return
        if self.location2 is None:
            self.location2 = msg
            return

    def odom_callback(self, msg):
        self.current_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def state_machine_step(self):
        if self.current_pos is None:
            self.get_logger().info('No current pose')
            return  # no odometry yet

        if self.state == HeistState.NAVIGATING_TO_1:
            self.get_logger().info('nav 1')
            if self.is_close(self.current_pos, self.location1.pose.position):
                self.get_logger().info("Arrived at location 1. Pickup starting")
                self.stop_robot()
                self.state = HeistState.NAVIGATING_TO_2

        elif self.state == HeistState.NAVIGATING_TO_2:
            self.get_logger().info('nav 2')
            if self.is_close(self.current_pos, self.location2.pose.position):
                self.get_logger().info("Arrived at location 2. Pickup starting")
                self.stop_robot()
                self.state = HeistState.ESCAPING

                # make a U turn for the end
                self.rotate_start_time = time.time()
                while self.rotating:
                    drive_msg = AckermannDriveStamped()

                    if self.turn_stage == 1:
                        drive_msg.drive.speed = 0.5  # forward
                        drive_msg.drive.steering_angle = 0.34  # left turn (adjustable)
                        self.safety_stop.publish(drive_msg)

                        if time.time() - self.rotate_start_time >= 2:  # forward turn duration
                            self.rotate_start_time = time.time()
                            self.turn_stage = 2

                    elif self.turn_stage == 2:
                        drive_msg.drive.speed = -0.5  # reverse
                        drive_msg.drive.steering_angle = -0.34  # right turn
                        self.safety_stop.publish(drive_msg)

                        if time.time() - self.rotate_start_time >= 1.5:  # reverse duration
                            self.rotate_start_time = time.time()
                            self.turn_stage = 3

                    elif self.turn_stage == 3:
                        drive_msg.drive.speed = 0.5  # forward again
                        drive_msg.drive.steering_angle = 0.34  # left turn
                        self.safety_stop.publish(drive_msg)

                        if time.time() - self.rotate_start_time >= 2:  # final forward duration
                            self.rotating = False
                            self.get_logger().info("3-point turn complete")
                            self.end_pub.publish(self.start_location)
                            # self.state = HeistState.ESCAPING

                    # return  # don’t process other states while rotating

        elif self.state == HeistState.ESCAPING:
            if self.is_close(self.current_pos, self.start_location.pose.position):
                self.get_logger().info("Escaped back to start! Heist complete.")
                self.state = HeistState.FINISHED

    # def publish_goal(self, pose_msg):
    #     self.goal_pub.publish(pose_msg)
    #     self.get_logger().info(f"Published goal: ({pose_msg.pose.position.x}, {pose_msg.pose.position.y})")

    def stop_robot(self):
        # stop car for 5 seconds and save image of banana
        start_time = time.time()
        ready_msg = Bool()
        ready_msg.data = True
        self.ready_to_save.publish(ready_msg)
        while time.time() - start_time <= 5:
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = 0.0
            self.safety_stop.publish(drive_msg)

    def is_close(self, current, target, threshold=0.4):
        self.get_logger().info('checking')
        dx = current[0] - target.x
        dy = current[1] - target.y

        within_thres = math.hypot(dx, dy) < threshold
        self.get_logger().info(f'within_tres:{within_thres}')
        return within_thres

    # def create_pose_stamped(self, x, y):
    #     ps = PoseStamped()
    #     ps.header.frame_id = 'map'
    #     ps.pose.position.x = x
    #     ps.pose.position.y = y
    #     ps.pose.orientation.w = 1.0
    #     return ps

def main(args=None):
    rclpy.init(args=args)
    sm = StateMachine()
    rclpy.spin(sm)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
