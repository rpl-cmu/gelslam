import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time

from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import yaml

from gelslam_msgs.msg import KeyFrameMsg, FrameMsg
from gelslam.core.tracker import Tracker


class TrackerNode(Node):
    def __init__(self):
        super().__init__("tracker_node")
        self.bridge = CvBridge()
        # Get the configuration path and load the configuration
        self.declare_parameter("config_path", "")
        config_path = (
            self.get_parameter("config_path").get_parameter_value().string_value
        )
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        # Get the calibration model path
        calib_model_path = config["device_config"]["calibration_model_path"]
        if not os.path.isabs(calib_model_path):
            calib_model_path = os.path.join(
                os.path.dirname(config_path), calib_model_path
            )
        # Get the save data directory
        self.declare_parameter("data_dir", "")
        data_dir = self.get_parameter("data_dir").get_parameter_value().string_value
        self.save_dir = os.path.join(data_dir, "gelslam_online")

        # The tracker object
        self.tracker = Tracker(calib_model_path, config, logger=self.get_logger())

        # Create the subscriber
        self.create_subscription(Image, "image", self.image_callback, 1)

        # Create the publishers
        self.keyframe_pub = self.create_publisher(KeyFrameMsg, "keyframe", 10)
        self.frame_pub = self.create_publisher(FrameMsg, "frame", 10)

    def image_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        ret, track_result = self.tracker.track(image, log_prefix="Track")
        if ret:
            frame_msg, keyframe_msgs = track_result
            for keyframe_msg in keyframe_msgs:
                self.keyframe_pub.publish(keyframe_msg)
                time.sleep(0.001)
            self.frame_pub.publish(frame_msg)
            time.sleep(0.001)

    def destroy_node(self):
        # Destroy the node
        super().destroy_node()
        # Save the tracker results
        self.tracker.save(self.save_dir)


def main(args=None):
    rclpy.init(args=args)
    node = TrackerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        node.destroy_node()


if __name__ == "__main__":
    main()
