#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from enum import Enum
import math
import time

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool
from vs_msgs.msg import ConeLocationPixel

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
        self.gen_return = self.create_publisher(Bool, '/can_return', 10)
        self.start_new = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.red_sub = self.create_subscription(Bool, '/is_red', self.red_light, 10)
        # Subscriber to receive odometry
        self.odom_sub = self.create_subscription(Odometry, '/pf/pose/odom', self.odom_callback, 10)
        self.init_sub = self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.init_cb, 10)
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

        self.current_pos_msg = None

    def init_cb(self, msg):
        if self.state != HeistState.ESCAPING:
            self.get_logger().info('START AND END POINT SETTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT')
            end_pose = PoseStamped()
            end_pose.pose = msg.pose.pose
            self.start_location = end_pose

    def red_light(self, msg):
        self.get_logger().info(f'{msg.data}')
        # while red light detected, stop it for 2 secs
        if msg.data:
            self.stop_robot(True)
        else:
            return

    def clicked_callback(self, msg):
        if self.state != HeistState.ESCAPING:
            self.state = HeistState.NAVIGATING_TO_1
            if self.location1 is None:
                self.location1 = msg
                return
            if self.location2 is None:
                self.location2 = msg
                return

    def odom_callback(self, msg):
        self.current_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        pos_msg = PoseWithCovarianceStamped()
        pos_msg.pose = msg.pose
        pos_msg.header = msg.header
        self.current_pos_msg = pos_msg

    def state_machine_step(self):
        if self.current_pos is None:
            self.get_logger().info('No current pose')
            return  # no odometry yet

        if self.state == HeistState.NAVIGATING_TO_1:

            if self.is_close(self.current_pos, self.location1.pose.position):
                self.get_logger().info("Arrived at location 1. Pickup starting")
                self.stop_robot()
                self.state = HeistState.NAVIGATING_TO_2

        elif self.state == HeistState.NAVIGATING_TO_2:

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
                        drive_msg.drive.steering_angle = -0.50  # right turn
                        self.safety_stop.publish(drive_msg)

                        if time.time() - self.rotate_start_time >= 1.5:  # reverse duration
                            self.rotate_start_time = time.time()
                            self.turn_stage = 3

                    elif self.turn_stage == 3:
                        drive_msg.drive.speed = 0.5  # forward again
                        drive_msg.drive.steering_angle = 0.50  # left turn
                        self.safety_stop.publish(drive_msg)

                        if time.time() - self.rotate_start_time >= 2:  # final forward duration
                            self.rotating = False
                            self.get_logger().info("3-point turn complete")
                            # tell the planner to generate a new path
                            # reset the planner
                            condition = Bool()
                            condition.data = True
                            self.gen_return.publish(condition)
                            # buffer
                            time.sleep(3)

                            #   NEWWWW
                            # Get current yaw
                            current_yaw = self.get_yaw_from_quaternion(self.current_pos_msg.pose.pose.orientation)

                            # Add 180 degrees (π radians) to flip the orientation
                            new_yaw = current_yaw + math.pi

                            # Normalize to [-π, π]
                            while new_yaw > math.pi:
                                new_yaw -= 2.0 * math.pi
                            while new_yaw < -math.pi:
                                new_yaw += 2.0 * math.pi

                            # Create new quaternion
                            q = self.convert_to_quaternion(new_yaw)

                            # Set it in the message
                            self.current_pos_msg.pose.pose.orientation.x = q['x']
                            self.current_pos_msg.pose.pose.orientation.y = q['y']
                            self.current_pos_msg.pose.pose.orientation.z = q['z']
                            self.current_pos_msg.pose.pose.orientation.w = q['w']

                            # give planner a new initial pose with current pose
                            self.start_new.publish(self.current_pos_msg)

                            # give planner the end goal
                            self.end_pub.publish(self.start_location)
                            # self.state = HeistState.ESCAPING

                    # return  # don’t process other states while rotating

        elif self.state == HeistState.ESCAPING:
            if self.is_close(self.current_pos, self.start_location.pose.position):
                self.get_logger().info("Escaped back to start! Heist complete.")
                self.state = HeistState.FINISHED

    def convert_to_quaternion(self, yaw):
        """Convert a yaw angle to quaternion"""
        return {
            'x': 0.0,
            'y': 0.0,
            'z': math.sin(yaw/2.0),
            'w': math.cos(yaw/2.0)
        }

    def get_yaw_from_quaternion(self, q):
        """Extract yaw angle from quaternion"""
        # Uses the same formula as in your PathPlan class
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def stop_robot(self, traffic_light = False):
        if not traffic_light:
            # stop car for 5 seconds and save image of banana
            start_time = time.time()
            ready_msg = Bool()
            ready_msg.data = True
            self.ready_to_save.publish(ready_msg)
            while time.time() - start_time <= 5:
                drive_msg = AckermannDriveStamped()
                drive_msg.drive.speed = 0.0
                self.safety_stop.publish(drive_msg)
        else:
            start_time = time.time()
            while time.time() - start_time <= 2:
                drive_msg = AckermannDriveStamped()
                drive_msg.drive.speed = 0.0
                self.safety_stop.publish(drive_msg)


    def is_close(self, current, target, threshold=0.4):
        dx = current[0] - target.x
        dy = current[1] - target.y

        within_thres = math.hypot(dx, dy) < threshold
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
