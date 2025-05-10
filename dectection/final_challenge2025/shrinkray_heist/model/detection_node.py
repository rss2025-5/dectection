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

        #if any(label == "traffic light" for _, label in self.predictions):
        #    self.traffic.publish(img_msg)
        # Publish the processed image
        self.publisher.publish(out_msg)
        h, w = image.shape[:2]
        for (x1, y1, x2, y2), label in self.predictions:
            if label == "traffic light":
                #self.get_logger().info("TRAFFIC LIGHT")
                iout = np.array(image)
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(w, int(x2))
                y2 = min(h,int( y1 + (y2-y1)/3))
                iout = image[y1:y2, x1:x2]
                t_msg = self.bridge.cv2_to_imgmsg(iout,encoding="bgr8")
                #t_msg.header = img_msg.header
                #t_msg = img_msg
                self.traffic.publish(t_msg)
                break
            else:
                black_img = np.zeros((360, 640, 3), dtype=np.uint8)
                black_img = self.bridge.cv2_to_imgmsg(black_img, encoding="bgr8")

                self.traffic.publish(black_img)
                self.get_logger().info('Published black image.')

        #self.publisher.publish(out_msg)

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
