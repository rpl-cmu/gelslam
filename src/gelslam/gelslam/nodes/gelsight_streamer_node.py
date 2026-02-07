import os
import threading

import cv2
import rclpy
import yaml
from cv_bridge import CvBridge
from gs_sdk.gs_device import FastCamera
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image


class GelSightStreamerNode(Node):
    """
    ROS2 node for streaming images from a GelSight sensor.
    """

    def __init__(self):
        super().__init__("gelsight_streamer_node")
        self.bridge = CvBridge()
        # Get the configuration path and load the configuration
        self.declare_parameter("config_path", "")
        config_path = (
            self.get_parameter("config_path").get_parameter_value().string_value
        )
        config_path = os.path.abspath(os.path.expanduser(config_path))
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            device_config = config["device_config"]
            device_name = device_config["device_name"]
            imgh = device_config["imgh"]
            imgw = device_config["imgw"]
            raw_imgh = device_config["raw_imgh"]
            raw_imgw = device_config["raw_imgw"]
            framerate = device_config["framerate"]
        # Get the data directory to save the video file
        self.declare_parameter("data_dir", "")
        self.data_dir = (
            self.get_parameter("data_dir").get_parameter_value().string_value
        )
        self.data_dir = os.path.abspath(os.path.expanduser(self.data_dir))
        # Create the directory for other nodes to save information
        self.save_dir = os.path.join(self.data_dir, "gelslam_online")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        # Initialize and connect to the device
        self.device = FastCamera(
            device_name, imgh, imgw, raw_imgh, raw_imgw, framerate, verbose=False
        )
        self.device.connect(verbose=False)
        # The video writer
        self.video_writer = cv2.VideoWriter(
            os.path.join(self.data_dir, "gelsight.avi"),
            cv2.VideoWriter_fourcc(*"FFV1"),
            framerate,
            (imgw, imgh),
        )
        # Flag to control the streaming thread
        self.running = True
        # Start the image streaming thread
        self.stream_thread = threading.Thread(target=self.stream_images)
        self.stream_thread.start()
        # Initialize publishers
        self.image_pub = self.create_publisher(Image, "image", 10)

    def stream_images(self):
        """
        Continuously captures, saves, and publishes images.
        """
        while self.running:
            image = self.device.get_image()
            self.video_writer.write(image)
            image_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            if self.context.ok():
                self.image_pub.publish(image_msg)

    def destroy_node(self):
        # Cleanup
        self.running = False
        super().destroy_node()
        # Stop the streaming thread
        self.stream_thread.join()
        self.device.release()
        # Save the tactile video
        self.video_writer.release()


def main(args=None):
    """
    Main function to initialize and run the node.
    """
    rclpy.init(args=args)
    node = GelSightStreamerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.destroy_node()


if __name__ == "__main__":
    main()
