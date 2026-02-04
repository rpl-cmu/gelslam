import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class SavedVideoStreamerNode(Node):
    def __init__(self):
        super().__init__("saved_video_streamer_node")
        self.bridge = CvBridge()

        # Get the parent directory of the saved data
        self.declare_parameter("data_dir", "")
        self.data_dir = (
            self.get_parameter("data_dir").get_parameter_value().string_value
        )
        # Create the directory for other nodes to save information
        self.save_dir = os.path.join(self.data_dir, "gelslam_online")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        # Read the saved video from file
        self.cap = cv2.VideoCapture(os.path.join(self.data_dir, "gelsight.avi"))
        self.image_pub = self.create_publisher(Image, "image", 10)
        # Create the timer to publish images
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.timer = self.create_timer(1.0 / self.fps, self.publish_image)

    def publish_image(self):
        # Send the saved video frames
        ret, image = self.cap.read()
        if ret:
            image_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            self.image_pub.publish(image_msg)
        else:
            self.get_logger().warning("Video stream ended")
            self.cap.release()
            raise SystemExit


def main(args=None):
    rclpy.init(args=args)
    node = SavedVideoStreamerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        node.destroy_node()


if __name__ == "__main__":
    main()
