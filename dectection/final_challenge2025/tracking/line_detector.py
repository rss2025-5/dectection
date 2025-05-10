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
                ('max_speed', 4.0),
                ('lookahead_distance', 0.51), # 0.8 og
                ('hough_threshold', 30),
                ('min_line_length', 50),
                ('max_line_gap', 30),
                ('lane_width_tolerance', 0.2)  # 30% tolerance for lane width variation
            ])

        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.lookahead_distance = self.get_parameter('lookahead_distance').get_parameter_value().double_value
        self.max_speed = self.get_parameter('max_speed').get_parameter_value().double_value
        self.hough_threshold = self.get_parameter('hough_threshold').get_parameter_value().integer_value
        self.min_line_length = self.get_parameter('min_line_length').get_parameter_value().integer_value
        self.max_line_gap = self.get_parameter('max_line_gap').get_parameter_value().integer_value
        self.lane_width_tolerance = self.get_parameter('lane_width_tolerance').get_parameter_value().double_value

        self.wheel_base = 0.33
        self.max_steering_angle = 3.1415/10.0 #10 degrees 

        # Initialize visualization variables
        self.target_point = None
        self.left_fit = None
        self.right_fit = None

        self.prev_left_fit = None
        self.prev_right_fit = None

        self.initial_left_fit = None
        self.initial_right_fit = None
        self.set_init = False

        # Store the expected lane width (horizontal distance between lines at horizon)
        self.expected_lane_width = None

        # Subscribers and Publishers
        self.subscription = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            10)
        self.cmd_pub = self.create_publisher(AckermannDriveStamped, '/vesc/high_level/input/nav_0', 10)
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

                # Calculate bottom intercept for classification
                bottom_x = (img_height - b) / slope
            else:
                # Skip vertical lines
                continue

            # Classify left/right lines
            if slope < 0 and bottom_x < img_width/2:
                left_candidates.append((intersection_x, slope, b, line))
            elif slope > 0 and bottom_x > img_width/2:
                right_candidates.append((intersection_x, slope, b, line))

        # Select the best lines
        left_line = None
        right_line = None

        # If we have both initial lane lines, calculate the expected lane width
        if self.set_init and self.initial_left_fit is not None and self.initial_right_fit is not None:
            ml, bl = self.line_equation(self.initial_left_fit)
            mr, br = self.line_equation(self.initial_right_fit)

            left_x_at_horizon = (horizon_y - bl) / ml if ml != 0 else 0
            right_x_at_horizon = (horizon_y - br) / mr if mr != 0 else img_width

            # Store the initial lane width if not already set
            if self.expected_lane_width is None:
                self.expected_lane_width = right_x_at_horizon - left_x_at_horizon
                self.get_logger().info(f"Initial lane width set to: {self.expected_lane_width} pixels")

        # If we have the expected lane width, filter by consistent lane width
        if self.expected_lane_width is not None and self.expected_lane_width > 0:
            valid_left_right_pairs = []

            # Try all combinations of left and right candidates
            for left_data in left_candidates:
                left_x, left_m, left_b, left_line_data = left_data

                for right_data in right_candidates:
                    right_x, right_m, right_b, right_line_data = right_data

                    # Calculate the lane width at horizon
                    current_width = right_x - left_x

                    # Check if width is within tolerance
                    width_ratio = current_width / self.expected_lane_width
                    if 1 - self.lane_width_tolerance <= width_ratio <= 1 + self.lane_width_tolerance:
                        # This pair maintains the expected lane width
                        lane_center = (left_x + right_x) / 2
                        center_distance = abs(lane_center - center_x)
                        valid_left_right_pairs.append((center_distance, left_data, right_data))

            # Select the best pair (closest to center)
            if valid_left_right_pairs:
                valid_left_right_pairs.sort(key=lambda x: x[0])  # Sort by center distance
                best_pair = valid_left_right_pairs[0]
                left_line = best_pair[1][3]
                right_line = best_pair[2][3]

                #self.get_logger().info(f"Selected lane pair with width: {best_pair[2][0] - best_pair[1][0]:.1f} pixels")

                # Update initial fits if needed
                if not self.set_init:
                    self.initial_left_fit = left_line
                    self.initial_right_fit = right_line
                    self.set_init = True

                max_x_jump = img_shape[1] * 0.03125  # 20% of image width

                # Reject left line if it jumps too much from previous
                if self.prev_left_fit is not None and left_line is not None:
                    m_prev, b_prev = self.line_equation(self.prev_left_fit)
                    m_curr, b_curr = self.line_equation(left_line)

                    x_prev = (horizon_y - b_prev) / m_prev if m_prev != 0 else 0
                    x_curr = (horizon_y - b_curr) / m_curr if m_curr != 0 else 0

                    if abs(x_curr - x_prev) > max_x_jump:
                        #self.get_logger().warn("Left line jumped too far, reverting to previous")
                        left_line = self.prev_left_fit

                # Reject right line if it jumps too much from previous
                if self.prev_right_fit is not None and right_line is not None:
                    m_prev, b_prev = self.line_equation(self.prev_right_fit)
                    m_curr, b_curr = self.line_equation(right_line)

                    x_prev = (horizon_y - b_prev) / m_prev if m_prev != 0 else img_shape[1]
                    x_curr = (horizon_y - b_curr) / m_curr if m_curr != 0 else img_shape[1]

                    if abs(x_curr - x_prev) > max_x_jump:
                        #self.get_logger().warn("Right line jumped too far, reverting to previous")
                        right_line = self.prev_right_fit


                return left_line, right_line

        # If we couldn't find a valid pair or don't have expected width yet, fall back to individual selection
        if not left_line or not right_line:
            #self.get_logger().info("Falling back to individual line selection")

            # Select left line (closest to center at horizon)
            if left_candidates:
                left_candidates.sort(key=lambda x: abs(center_x - x[0]))
                left_line = left_candidates[0][3]

                # Apply additional filtering with previous/initial fit if needed
                if self.initial_left_fit is not None:
                    ml, bl = self.line_equation(left_line)
                    init_ml, init_bl = self.line_equation(self.initial_left_fit)

                    # Calculate horizons points for both current and initial
                    if ml != 0 and init_ml != 0:
                        left_x_horizon = (horizon_y - bl) / ml
                        init_left_x_horizon = (horizon_y - init_bl) / init_ml

                        # If horizontal position is very different, try next candidate
                        if abs(left_x_horizon - init_left_x_horizon) > img_width * 0.2:  # 20% of width
                            if len(left_candidates) > 1:
                                left_line = left_candidates[1][3]

            # Select right line (closest to center at horizon)
            if right_candidates:
                right_candidates.sort(key=lambda x: abs(center_x - x[0]))
                right_line = right_candidates[0][3]

                # Apply additional filtering with previous/initial fit
                if self.initial_right_fit is not None:
                    mr, br = self.line_equation(right_line)
                    init_mr, init_br = self.line_equation(self.initial_right_fit)

                    # Calculate horizons points for both current and initial
                    if mr != 0 and init_mr != 0:
                        right_x_horizon = (horizon_y - br) / mr
                        init_right_x_horizon = (horizon_y - init_br) / init_mr

                        # If horizontal position is very different, try next candidate
                        if abs(right_x_horizon - init_right_x_horizon) > img_width * 0.2:  # 20% of width
                            if len(right_candidates) > 1:
                                right_line = right_candidates[1][3]

            # Set initial fits if this is the first detection
            if not self.set_init and left_line is not None and right_line is not None:
                self.initial_left_fit = left_line
                self.initial_right_fit = right_line
                self.set_init = True

        # Update previous fits
        if left_line is not None:
            self.prev_left_fit = left_line
        if right_line is not None:
            self.prev_right_fit = right_line

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
            b_left = b_right - 120.0
        if m_right is None:
            m_right = m_left
            b_right = b_left + 120.0

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
        yt = (img_width/2 - target_x)*0.0005
        curvature = 2 * yt / (L ** 2)
        steering_angle = np.arctan(self.wheel_base * curvature)
        steering_angle = np.clip(steering_angle,
                                -self.max_steering_angle,
                                self.max_steering_angle)

        # Calculate speed based on steering angle
        speed = self.max_speed * (1 - abs(np.sin(steering_angle)))
        self.get_logger().info(f"speed: {speed}")
        # Publish command
        cmd = AckermannDriveStamped()
        cmd.drive.steering_angle = steering_angle
        cmd.drive.speed = speed
        self.cmd_pub.publish(cmd)


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

            # Draw lookahead horizon line (yellow)
            cv.line(src, (0, y_horizon), (w, y_horizon), (255, 255, 0), 1, cv.LINE_AA)

            # Draw all detected lines if provided
            if lines is not None:
                for line in lines:
                    if line is not None:
                        l = line[0]
                        cv.line(src, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 2, cv.LINE_AA)

            # Draw initial expected lane width visual guide
            if self.initial_left_fit is not None and self.initial_right_fit is not None:
                init_ml, init_bl = self.line_equation(self.initial_left_fit)
                init_mr, init_br = self.line_equation(self.initial_right_fit)

                try:
                    if init_ml != 0 and init_mr != 0:
                        init_left_x = int((y_horizon - init_bl) / init_ml)
                        init_right_x = int((y_horizon - init_br) / init_mr)

                        # Draw initial lane width with faint dashed line
                        cv.line(src, (init_left_x, y_horizon), (init_right_x, y_horizon), (50, 100, 50), 1, cv.LINE_AA)

                        # Draw expected lane width tolerance boundaries
                        if self.expected_lane_width is not None:
                            lane_width = init_right_x - init_left_x
                            lane_center = (init_left_x + init_right_x) // 2

                            min_width = lane_width * (1 - self.lane_width_tolerance)
                            max_width = lane_width * (1 + self.lane_width_tolerance)

                            min_left_x = int(lane_center - max_width/2)
                            max_left_x = int(lane_center - min_width/2)
                            min_right_x = int(lane_center + min_width/2)
                            max_right_x = int(lane_center + max_width/2)

                            # Draw min/max left position
                            cv.line(src, (min_left_x, y_horizon-10), (min_left_x, y_horizon+10), (0, 128, 255), 2, cv.LINE_AA)
                            cv.line(src, (max_left_x, y_horizon-10), (max_left_x, y_horizon+10), (0, 128, 255), 2, cv.LINE_AA)

                            # Draw min/max right position
                            cv.line(src, (min_right_x, y_horizon-10), (min_right_x, y_horizon+10), (0, 128, 255), 2, cv.LINE_AA)
                            cv.line(src, (max_right_x, y_horizon-10), (max_right_x, y_horizon+10), (0, 128, 255), 2, cv.LINE_AA)

                            # Add lane width info
                            cv.putText(src, f"Lane width: {self.expected_lane_width:.1f}px",
                                    (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            cv.putText(src, f"Tolerance: {self.lane_width_tolerance*100:.0f}%",
                                    (10, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                except Exception as e:
                    self.get_logger().debug(f"Initial lane width visualization error: {str(e)}")

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

                        # Draw current lane width
                        if m_left != 0:
                            x_left_horizon = int((y_horizon - b_left) / m_left)
                            current_width = x_right_horizon - x_left_horizon
                            cv.putText(src, f"Current width: {current_width:.1f}px",
                                    (10, 180), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
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

            # Convert OpenCV image back to ROS Image message
            out_msg = self.bridge.cv2_to_imgmsg(src, encoding="bgr8")
            out_msg.header = img_msg.header  # Preserve the timestamp and frame_id

            # Publish the processed image
            self.img_pub.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f"Visualization error: {str(e)}")

    def find_line_intersection(self, m1, b1, m2, b2):
        """
        Find the intersection point of two lines given in slope-intercept form:
        y = m1*x + b1 and y = m2*x + b2

        Returns:
            tuple: (x, y) coordinates of intersection point, or None if parallel
        """
        if m1 == m2:  # Parallel lines
            return None

        # Calculate intersection point
        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1

        return (int(x), int(y))

def main(args=None):
    rclpy.init(args=args)
    lane_node = LanePurePursuit()
    rclpy.spin(lane_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
