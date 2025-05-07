import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
import cv2

from sensor_msgs.msg import Image
from .detector import Detector
from std_msgs.msg import Bool
import numpy as np

class DetectorNode(Node):
    def __init__(self):
        super().__init__("detector")
        self.detector = Detector()
        self.publisher = self.create_publisher(Image, "/predicted/image", 10)
        self.traffic = self.create_publisher(Image, "/traffic", 10)
        self.subscriber = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.callback, 1)
        self.subscriber = self.create_subscription(Bool, "/ready_save", self.save_image, 1)
        self.bridge = CvBridge()

        self.get_logger().info("Detector Initialized")

        self.detected_image = None
        self.banana_detected = False

        self.predictions = None
        self.image_count = 1

    def callback(self, img_msg):
        # Process image with CV Bridge
        image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")

        self.detector.set_threshold(0.5)

        results = self.detector.predict(image)

        self.predictions = results["predictions"]
        original_image = results["original_image"]

        out = self.detector.draw_box(original_image, self.predictions, draw_all=True)

        out = np.array(out)
        self.detected_image = out

        # Convert OpenCV image back to ROS Image message
        out_msg = self.bridge.cv2_to_imgmsg(out, encoding="bgr8")
        out_msg.header = img_msg.header  # Optionally preserve the timestamp and frame_id

        if any(label == "traffic light" for _, label in self.predictions):
            self.traffic.publish(img_msg)
        # Publish the processed image
        self.publisher.publish(out_msg)

    def save_image(self, ready_to_save):
        if self.predictions:
            self.banana_detected = any(label == "banana" for _, label in self.predictions)
            if self.banana_detected and ready_to_save.data:
                self.get_logger().info("Banana detected! Saving image.")
                cv2.imwrite("detected_banana" + str(self.image_count)+ ".png", self.detected_image)
                self.image_count+=1


def main(args=None):
    rclpy.init(args=args)
    detector = DetectorNode()
    rclpy.spin(detector)
    rclpy.shutdown()

if __name__=="__main__":
    main()
