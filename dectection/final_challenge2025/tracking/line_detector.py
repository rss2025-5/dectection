import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv as cv
import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped


class LanePurePursuit(Node):
    def __init__(self):
        super().__init__('lane_pure_pursuit')
        self.bridge = CvBridge()
        
        # Parameters
        self.declare_parameters(namespace='',
            parameters=[
                ('camera_topic', '/zed/zed_node/rgb/image_rect_color'),
                ('max_speed', 1.0),
                ('lookahead_distance', 0.6),
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
        self.max_steering_angle = np.pi/2
        
        # Subscribers and Publishers
        self.subscription = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv(msg, "bgr8")
            mask = self.preprocess_image(cv_image)
            lines = self.detect_hough_lines(mask)
            
            if lines is not None:
                self.pub_image(msg, lines)
                left_line, right_line = self.find_closest_lines(lines, cv_image.shape)
                if left_line and right_line:
                    self.pure_pursuit_control(left_line, right_line, cv_image.shape[1], cv_image.shape[0])
                
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
        return cv.HoughLinesP(
            edges, 
            1, 
            np.pi/180,
            self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

    def find_closest_lines(self, lines, img_shape):
        left_candidates = []
        right_candidates = []
        img_height, img_width = img_shape[:2]
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate line properties
            slope = (y2 - y1) / (x2 - x1 + 1e-5)
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            # Filter horizontal lines and short lines
            if abs(slope) < 0.5 or length < 50:
                continue
                
            # Calculate bottom intercept
            if slope != 0:
                intercept = x1 - y1/slope
                bottom_x = intercept + img_height/slope
            else:
                bottom_x = x1
                
            # Classify left/right lines
            if slope < 0 and bottom_x < img_width/2:
                left_candidates.append((bottom_x, line))
            elif slope > 0 and bottom_x > img_width/2:
                right_candidates.append((bottom_x, line))
        
        # Select closest lines
        left_line = max(left_candidates, key=lambda x: x[0])[1] if left_candidates else None
        right_line = min(right_candidates, key=lambda x: x[0])[1] if right_candidates else None
        
        return left_line, right_line

    def pure_pursuit_control(self, left_line, right_line, img_width, img_height):
        # Calculate line equations
        def line_equation(line):
            x1, y1, x2, y2 = line[0]
            m = (y2 - y1) / (x2 - x1 + 1e-5)
            b = y1 - m * x1
            return m, b
        
        # Get line equations
        m_left, b_left = line_equation(left_line)
        m_right, b_right = line_equation(right_line)
        
        # Calculate lookahead point 
        lookahead_y = img_height * self.lookahead_distance
        left_x = (lookahead_y - b_left) / m_left
        right_x = (lookahead_y - b_right) / m_right
        target_x = (left_x + right_x) / 2
        
        # Pure Pursuit calculations
        L = self.lookahead_distance
        yt = (img_width/2 - target_x) * 0.001  # Convert pixels to meters
        curvature = 2 * yt / (L ** 2)
        steering_angle = np.arctan(self.wheel_base*curvature)
        steering_angle = np.clip(steering_angle, 
                                -self.max_steering_angle,
                                self.max_steering_angle)
        
        # Publish command

        cmd = AckermannDriveStamped()
        cmd.drive.steering_angle = steering_angle
        cmd.drive.speed = self.max_speed*(1 - 0.4*abs(np.sin(steering_angle)))
        self.cmd_pub.publish(cmd)


    def pub_image(self, img_msg, linesP):
        # Process image with CV Bridge

        src = self.bridge.imgmsg_to_cv(img_msg, "bgr8")
        # Check if image is loaded fine
        if src is None:
            self.get_logger().info("Error with vision!")
        
        
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv.line(src, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)

        out = np.array(src)

        # Convert OpenCV image back to ROS Image message
        out_msg = self.bridge.cv_to_imgmsg(out, encoding="bgr8")
        out_msg.header = img_msg.header  # Optionally preserve the timestamp and frame_id

        # Publish the processed image
        self.publisher.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    lane_node = LanePurePursuit()
    rclpy.spin(lane_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
