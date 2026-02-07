import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time

import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from gelslam.core.tracker import Tracker
from gelslam.utils import Logger
from gelslam_msgs.msg import FrameMsg, KeyFrameMsg

"""
Single thread:
1. Track thread:
    - Image callback: Tracks the frame and publishes keyframes/frames.
"""


class TrackerNode(Node):
    """
    ROS2 node for real-time tracking.
    """

    def __init__(self):
        super().__init__("tracker_node")
        self.bridge = CvBridge()
        # Load configuration path
        self.declare_parameter("config_path", "")
        config_path = (
            self.get_parameter("config_path").get_parameter_value().string_value
        )
        config_path = os.path.abspath(os.path.expanduser(config_path))
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        # Get the calibration model path
        calib_model_path = config["device_config"]["calibration_model_path"]
        if not os.path.isabs(calib_model_path):
            calib_model_path = os.path.join(
                os.path.dirname(config_path), calib_model_path
            )
        # Get the directory to save the tracker states
        self.declare_parameter("data_dir", "")
        data_dir = self.get_parameter("data_dir").get_parameter_value().string_value
        data_dir = os.path.abspath(os.path.expanduser(data_dir))
        self.save_dir = os.path.join(data_dir, "gelslam_online")

        # The logger
        self.logger = Logger(ros_logger=self.get_logger())

        # Initialize tracker
        self.tracker = Tracker(calib_model_path, config, logger=self.logger)

        # Initialize subscribers
        self.create_subscription(Image, "image", self.image_callback, 1)

        # Initialize publishers
        self.keyframe_pub = self.create_publisher(KeyFrameMsg, "keyframe", 10)
        self.frame_pub = self.create_publisher(FrameMsg, "frame", 10)

    def image_callback(self, msg):
        """
        Callback function for the image subscriber.
        Tracks the frame and publishes keyframes/frames.

        :param msg: Image; The image message.
        """
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        ret, track_result = self.tracker.track(image, log_prefix="Track Thread")
        if ret:
            frame_msg, keyframe_msgs = track_result
            for keyframe_msg in keyframe_msgs:
                self.keyframe_pub.publish(keyframe_msg)
                time.sleep(0.001)
            self.frame_pub.publish(frame_msg)
            time.sleep(0.001)

    def destroy_node(self):
        """
        Clean up node and save tracker state.
        """
        # Clean up node
        super().destroy_node()
        # Save tracker state
        self.tracker.save(self.save_dir)


def main(args=None):
    """
    Main function to initialize and run the node.
    """
    rclpy.init(args=args)
    node = TrackerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        node.destroy_node()


if __name__ == "__main__":
    main()
