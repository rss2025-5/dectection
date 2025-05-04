import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped

# img: height = 360
# img: width = 640

class LanePurePursuit(Node):
    def __init__(self):
        super().__init__('lane_pure_pursuit')
        self.bridge = CvBridge()

        # Parameters
        self.declare_parameters(namespace='',
            parameters=[
                ('camera_topic', '/zed/zed_node/rgb/image_rect_color'),
                ('max_speed', 2.0),
                ('lookahead_distance', 0.5), # 0.8 og
                ('hough_threshold', 50),
                ('min_line_length', 50),
                ('max_line_gap', 30)
            ])

        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value

        self.lookahead_distance = self.get_parameter('lookahead_distance').get_parameter_value().double_value
        self.max_speed = self.get_parameter('max_speed').get_parameter_value().double_value
        self.hough_threshold = self.get_parameter('hough_threshold').get_parameter_value().integer_value
        self.min_line_length = self.get_parameter('min_line_length').get_parameter_value().integer_value
        self.max_line_gap = self.get_parameter('max_line_gap').get_parameter_value().integer_value

        self.wheel_base = 0.33
        self.max_steering_angle = np.pi/6

        # Initialize visualization variables
        self.target_point = None
        self.left_fit = None
        self.right_fit = None

        self.prev_left_fit = None
        self.prev_right_fit = None

        self.initial_left_fit = None
        self.initial_right_fit = None
        self.set_init = False

        # Subscribers and Publishers
        self.subscription = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            10)
        self.cmd_pub = self.create_publisher(AckermannDriveStamped, '/vesc/low_level/input/navigation', 10)
        self.img_pub = self.create_publisher(Image, '/pred_lines', 10)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            mask = self.preprocess_image(cv_image)
            lines = self.detect_hough_lines(mask)

            if lines is not None and len(lines) > 0:
                left_line, right_line = self.find_closest_lines(lines, cv_image.shape)

                # Create visualization
                vis_lines = []
                if left_line is not None:
                    vis_lines.append(left_line)
                if right_line is not None:
                    vis_lines.append(right_line)

                # Only process if we have lane data
                if left_line is not None or right_line is not None:
                    self.pure_pursuit_control(left_line, right_line, cv_image.shape[1], cv_image.shape[0])

                # Publish visualization regardless of control success
                self.pub_image(msg, vis_lines, left_line, right_line)
            else:
                # No lines detected, but still publish the image
                self.pub_image(msg, None, None, None)

        except Exception as e:
            self.get_logger().error(f"Processing error: {str(e)}")

    def preprocess_image(self, image):
        # Convert to HSV and threshold for white lines
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([255, 30, 255])
        mask = cv.inRange(hsv, lower_white, upper_white)

        # Apply morphological operations
        kernel = np.ones((5,5), np.uint8)
        return cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    def detect_hough_lines(self, image):
        # Edge detection and Hough transform
        edges = cv.Canny(image, 50, 150)
        lines = cv.HoughLinesP(
            edges,
            1,
            np.pi/180,
            self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        # Safety check for None
        if lines is None:
            return None

        filtered_lines = []
        for line in lines:
            imgh = image.shape[0] // 3
            x1, y1, x2, y2 = line[0]
            if y1 > imgh and y2 > imgh:
                filtered_lines.append(line)

        return filtered_lines if filtered_lines else None


    def find_closest_lines(self, lines, img_shape):
        left_candidates = []
        right_candidates = []
        img_height, img_width = img_shape[:2]
        center_x = img_width // 2

        # Define maximum allowed distance from center (as a percentage of image width)
        # Adjust this value based on your needs
        max_center_distance_percentage = 0.3
        max_center_distance = img_width * max_center_distance_percentage

        # Calculate horizon y-coordinate (lookahead line)
        horizon_y = int(img_height * (1 - self.lookahead_distance))

        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate line properties
            slope = (y2 - y1) / (x2 - x1 + 1e-5)
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

            # Filter horizontal lines and short lines
            if abs(slope) < 0.3 or length < 70:
                continue

            # Calculate intersection with horizon line
            if slope != 0:
                # y = mx + b => x = (y - b) / m
                # Find x where y = horizon_y
                b = y1 - slope * x1
                intersection_x = (horizon_y - b) / slope

                # Skip lines that intersect too far from center
                if abs(intersection_x - center_x) > max_center_distance:
                    continue

                # Calculate bottom intercept for classification
                bottom_x = (img_height - b) / slope
            else:
                # Skip vertical lines
                continue

            # Classify left/right lines
            if slope < 0 and bottom_x < img_width/2:
                left_candidates.append((intersection_x, line))
            elif slope > 0 and bottom_x > img_width/2:
                right_candidates.append((intersection_x, line))

        # Select closest lines to center, safely handle empty lists
        left_line = None
        right_line = None

        if left_candidates:
            # Sort by distance to center
            left_candidates.sort(key=lambda x: abs(center_x - x[0]))
            left_line = left_candidates[0][1]

            # Apply additional filtering with previous/initial fit if needed
            if self.set_init and self.initial_left_fit is not None:
                ml, bl = self.line_equation(left_line)
                init_ml, init_bl = self.line_equation(self.initial_left_fit)
                # Check if line is consistent with initial line
                if abs(ml - init_ml) <= 0.5 or abs(bl - init_bl) <= 25:
                    pass  # Keep current line
                elif len(left_candidates) > 1:
                    # Try next candidate
                    left_line = left_candidates[1][1]
            else:
                self.initial_left_fit = left_line
                self.set_init = True

        if right_candidates:
            # Sort by distance to center
            right_candidates.sort(key=lambda x: abs(center_x - x[0]))
            right_line = right_candidates[0][1]

            # Apply additional filtering with previous/initial fit if needed
            if self.set_init and self.initial_right_fit is not None:
                mr, br = self.line_equation(right_line)
                init_mr, init_br = self.line_equation(self.initial_right_fit)
                # Check if line is consistent with initial line
                if abs(mr - init_mr) <= 0.5 or abs(br - init_br) <= 25:
                    pass  # Keep current line
                elif len(right_candidates) > 1:
                    # Try next candidate
                    right_line = right_candidates[1][1]
            else:
                self.initial_right_fit = right_line
                self.set_init = True

        return left_line, right_line

    # Calculate line equations
    def line_equation(self, line):
        x1, y1, x2, y2 = line[0]
        m = (y2 - y1) / (x2 - x1 + 1e-5)
        b = y1 - m * x1
        return m, b

    def pure_pursuit_control(self, left_line, right_line, img_width, img_height):

        # Get line equations
        if left_line is not None:
            m_left, b_left = self.line_equation(left_line)
        else:
            m_left, b_left = None, None

        if right_line is not None:
            m_right, b_right = self.line_equation(right_line)
        else:
            m_right, b_right = None, None

        if m_left is None and m_right is None:
            return

        if m_left is None:
            m_left = m_right
            b_left = b_right - 100.0
        if m_right is None:
            m_right = m_left
            b_right = b_left + 100.0

        # Store line equations for visualization
        self.left_fit = (m_left, b_left)
        self.right_fit = (m_right, b_right)

        # Calculate lookahead point in image coordinates
        lookahead_y = int(img_height * (1 - self.lookahead_distance))

        # Calculate x-coordinates where the lines cross the lookahead horizon
        try:
            left_x = int((lookahead_y - b_left) / m_left) if m_left != 0 else 0
            right_x = int((lookahead_y - b_right) / m_right) if m_right != 0 else img_width
            target_x = int((left_x + right_x) / 2)

            # Store target point for visualization
            self.target_point = (target_x, lookahead_y)
        except Exception as e:
            self.get_logger().error(f"Target point calculation error: {str(e)}")
            return

        # Pure Pursuit calculations
        L = self.lookahead_distance
        yt = (img_width/2 - target_x) * 0.001  # Convert pixels to meters
        curvature = 2 * yt / (L ** 2)
        steering_angle = np.arctan(self.wheel_base * curvature)
        steering_angle = np.clip(steering_angle,
                                -self.max_steering_angle,
                                self.max_steering_angle)

        # Calculate speed based on steering angle
        speed = self.max_speed * (1 - 0.4 * abs(np.sin(steering_angle)))

        # Publish command
        cmd = AckermannDriveStamped()
        cmd.drive.steering_angle = steering_angle
        cmd.drive.speed = speed
        self.cmd_pub.publish(cmd)

    # Enhanced visualization function to show the center distance filter
    def pub_image(self, img_msg, lines=None, left_line=None, right_line=None):
        try:
            # Process image with CV Bridge
            src = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")

            # Check if image is loaded fine
            if src is None:
                self.get_logger().info("Error with vision!")
                return

            h, w = src.shape[:2]
            center_x = w // 2
            # Calculate horizon y-coordinate (lookahead line)
            y_horizon = int(h * (1 - self.lookahead_distance))

            # Define maximum allowed distance from center
            max_center_distance_percentage = 0.2
            max_center_distance = w * max_center_distance_percentage

            # Draw center region boundaries
            left_boundary = int(center_x - max_center_distance)
            right_boundary = int(center_x + max_center_distance)

            # Draw lookahead horizon line (yellow)
            cv.line(src, (0, y_horizon), (w, y_horizon), (255, 255, 0), 1, cv.LINE_AA)

            # Draw center region boundaries (green vertical lines)
            cv.line(src, (left_boundary, y_horizon - 20), (left_boundary, y_horizon + 20), (0, 255, 0), 2, cv.LINE_AA)
            cv.line(src, (right_boundary, y_horizon - 20), (right_boundary, y_horizon + 20), (0, 255, 0), 2, cv.LINE_AA)

            # Draw center point on horizon
            cv.circle(src, (center_x, y_horizon), 5, (255, 0, 255), -1)

            # Draw all detected lines if provided
            if lines is not None:
                for line in lines:
                    if line is not None:
                        l = line[0]
                        cv.line(src, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 2, cv.LINE_AA)

            # Draw left and right lane line projections
            if self.left_fit is not None and self.right_fit is not None:
                m_left, b_left = self.left_fit
                m_right, b_right = self.right_fit

                # Draw projected lane lines
                y_bottom = h

                try:
                    if m_left != 0:
                        x_left_bottom = int((y_bottom - b_left) / m_left)
                        x_left_horizon = int((y_horizon - b_left) / m_left)
                        cv.line(src, (x_left_bottom, y_bottom), (x_left_horizon, y_horizon), (0, 255, 255), 2, cv.LINE_AA)

                        # Draw intersection point
                        cv.circle(src, (x_left_horizon, y_horizon), 7, (0, 255, 0), -1)
                except Exception as e:
                    self.get_logger().debug(f"Left projection issue: {str(e)}")

                try:
                    if m_right != 0:
                        x_right_bottom = int((y_bottom - b_right) / m_right)
                        x_right_horizon = int((y_horizon - b_right) / m_right)
                        cv.line(src, (x_right_bottom, y_bottom), (x_right_horizon, y_horizon), (0, 255, 255), 2, cv.LINE_AA)

                        # Draw intersection point
                        cv.circle(src, (x_right_horizon, y_horizon), 7, (0, 255, 0), -1)
                except Exception as e:
                    self.get_logger().debug(f"Right projection issue: {str(e)}")

            # Draw target point and vehicle center with a connecting line
            if self.target_point is not None:
                vehicle_center = (w//2, h)

                # Draw vehicle center point
                cv.circle(src, vehicle_center, 5, (128, 0, 128), -1)

                # Draw target point (large red circle)
                cv.circle(src, self.target_point, 10, (0, 0, 255), -1)

                # Draw line connecting vehicle center to target point
                cv.line(src, vehicle_center, self.target_point, (255, 0, 255), 2, cv.LINE_AA)

                # Calculate and display steering angle
                dx = self.target_point[0] - vehicle_center[0]
                dy = vehicle_center[1] - self.target_point[1]  # Invert y axis
                angle_rad = np.arctan2(dx, dy)
                angle_deg = np.degrees(angle_rad)

                # Add steering info
                cv.putText(src, f"Steering: {angle_deg:.1f} deg",
                        (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Add lookahead distance info
                cv.putText(src, f"Lookahead: {self.lookahead_distance:.2f}",
                        (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Add allowed center distance info
                cv.putText(src, f"Max center dist: {max_center_distance_percentage*100:.0f}%",
                        (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Convert OpenCV image back to ROS Image message
            out_msg = self.bridge.cv2_to_imgmsg(src, encoding="bgr8")
            out_msg.header = img_msg.header  # Preserve the timestamp and frame_id

            # Publish the processed image
            self.img_pub.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f"Visualization error: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    lane_node = LanePurePursuit()
    rclpy.spin(lane_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
