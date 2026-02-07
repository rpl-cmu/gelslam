import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import copy
import threading
import time

import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from gelslam.core.frame import center_of_frame_msg, frame_msg2mesh, pose_of_frame_msg
from gelslam.core.keyframe import KeyFrameDB
from gelslam.utils import Logger
from gelslam.visualization.visible_coverage_meshes import (
    VisibleCoverageMeshes,
    compute_visible_coverage_meshes,
)
from gelslam.visualization.visualizer import Visualizer
from gelslam_msgs.msg import FrameMsg, KeyFrameMsg, VisibleCoverageGraphMsg

"""
Four threads:
1. K thread (Keyframe callback):
    - Reads keyframe messages and adds them to the KeyFrameDB.
2. F thread (Frame callback):
    - Updates the current frame information.
3. M thread (Mesh update):
    - Updates the VisibleCoverageMeshes based on VisibleCoverageGraphMsg from loop closure.
4. V thread (Visualization):
    - Visualizes the current frame and the visible coverage meshes.
"""


class VisualizerNode(Node):
    """
    ROS2 node for real-time rendering of the GelSLAM process.
    """

    def __init__(self):
        super().__init__("pose_graph_node")
        self.bridge = CvBridge()
        # Load the configuration
        self.declare_parameter("config_path", "")
        config_path = (
            self.get_parameter("config_path").get_parameter_value().string_value
        )
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            self.ppmm = config["device_config"]["ppmm"]
        # Get the save data directory
        self.declare_parameter("data_dir", "")
        data_dir = self.get_parameter("data_dir").get_parameter_value().string_value
        self.save_dir = os.path.join(data_dir, "gelslam_online")

        # The overall running flag
        self.running = True

        # The logger (We don't log rendering node)
        self.logger = Logger(ros_logger=self.get_logger())

        # KeyFrameDB (keyframe database) is never modified except appending.
        # K thread writes (appending), V thread reads
        self.add_keyframe_lock = threading.Lock()
        self.keyframedb = KeyFrameDB(self.ppmm, Logger())

        # Visible coverage meshes are updated only in M thread (triggered by loop closure).
        self.visible_coverage_meshes_update_lock = threading.Lock()
        self.visible_coverage_meshes = VisibleCoverageMeshes()
        self.visible_coverage_meshes_update_count = 0

        # Frame information is updated only in F thread.
        self.update_frame_lock = threading.Lock()
        self.curr_frame_msg = None

        # Viz states: to check if new frame is updated, and if visible coverage meshes is updated
        self.prev_vis_fid = None
        self.prev_visible_coverage_meshes_update_count = 0

        # Initialize KeyFrame subscriber (K thread)
        self.keyframe_callback_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(
            KeyFrameMsg,
            "keyframe",
            self.keyframe_callback,
            10,
            callback_group=self.keyframe_callback_group,
        )

        # Initialize Frame subscriber (F thread)
        self.frame_callback_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(
            FrameMsg,
            "frame",
            self.frame_callback,
            1,
            callback_group=self.frame_callback_group,
        )

        # Initialize Visible Coverage Graph Subscriber (M thread)
        self.visible_coverage_graph_callback_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(
            VisibleCoverageGraphMsg,
            "visible_coverage_graph",
            self.visible_coverage_graph_callback,
            1,
            callback_group=self.visible_coverage_graph_callback_group,
        )

        # Start the V thread: visualization
        self.visualization_thread = threading.Thread(target=self.visualization)
        self.visualization_thread.start()

    def frame_callback(self, msg):
        """
        Updates the current frame message. (F thread)

        :param msg: FrameMsg; The frame message.
        """
        with self.update_frame_lock:
            self.curr_frame_msg = msg

    def keyframe_callback(self, keyframe_msg):
        """
        Adds the received keyframe to the KeyFrameDB.

        :param keyframe_msg: KeyFrameMsg; The received keyframe message.
        """
        with self.add_keyframe_lock:
            self.keyframedb.insert(keyframe_msg)

    def visible_coverage_graph_callback(self, msg):
        """
        Updates visible coverage meshes based on received VisibleCoverageGraph message.

        :param msg: VisibleCoverageGraphMsg; The visible coverage graph message.
        """
        # Get mapping from kid to kidx
        with self.add_keyframe_lock:
            kid2kidx = self.keyframedb.get_kid2kidx()
        # New loop closure detected, compute the visible coverage meshes
        visible_coverage_meshes = compute_visible_coverage_meshes(
            msg, self.keyframedb, kid2kidx
        )
        # Update the visible coverage meshes
        with self.visible_coverage_meshes_update_lock:
            self.visible_coverage_meshes = visible_coverage_meshes
            self.visible_coverage_meshes_update_count += 1
        self.logger.info(
            "Rendering Thread -- Global Mesh (Visible Coverage Meshes) Updated: render with %.3fx scale"
            % (1.0 / (2**visible_coverage_meshes.viz_level))
        )

    def visualization(self):
        """
        Visualization loop (V thread).
        Renders based on visible coverage meshes and current frame.
        """
        # Initialize visualizer
        self.vis = Visualizer()
        self.vis.create_window()

        while self.running:
            # Copy and get the information from frame and keyframe callbacks
            with self.update_frame_lock:
                curr_frame_msg = copy.deepcopy(self.curr_frame_msg)
            with self.add_keyframe_lock:
                targeted_size = self.keyframedb.size()
            # Skip if new frame not ready yet
            if curr_frame_msg is None:
                self.vis.visualize_nothing()
                time.sleep(0.01)
                continue
            elif curr_frame_msg.fid == self.prev_vis_fid:
                self.vis.visualize_nothing()
                time.sleep(0.01)
                continue
            # Find the frame's reference keyframe's kidx
            ref_kidx = self.keyframedb.find_kidx_from_kid(
                curr_frame_msg.ref_kid, targeted_size
            )
            # Skip if the reference keyframe is not in the KeyframeDB
            if ref_kidx is None:
                self.vis.visualize_nothing()
                time.sleep(0.01)
                continue

            # If visible coverage mesh is updated, visualize it accordingly
            with self.visible_coverage_meshes_update_lock:
                new_trial_flag, start_T_ref = (
                    self.visible_coverage_meshes.get_keyframe_pose(
                        ref_kidx, self.keyframedb
                    )
                )
                if (
                    self.visible_coverage_meshes_update_count
                    != self.prev_visible_coverage_meshes_update_count
                ):
                    self.prev_visible_coverage_meshes_update_count = (
                        self.visible_coverage_meshes_update_count
                    )
                    self.vis.clear_geometries()
                    self.vis.add_visible_coverage_meshes(self.visible_coverage_meshes)
            if new_trial_flag:
                self.vis.clear_geometries()
                self.curr_frame_msg = None

            # Update current frame mesh
            curr_frame_mesh = frame_msg2mesh(
                self.bridge,
                curr_frame_msg,
                start_T_ref,
                self.ppmm,
            )
            self.vis.update_curr_frame_mesh(curr_frame_mesh)
            self.prev_vis_fid = curr_frame_msg.fid
            # Compute the viewing pose
            start_T_frame = start_T_ref @ pose_of_frame_msg(curr_frame_msg)
            frame_T_center = np.eye(4, dtype=np.float32)
            frame_T_center[:3, 3] = center_of_frame_msg(
                self.bridge, curr_frame_msg, self.ppmm
            )
            start_T_vis = start_T_frame @ frame_T_center
            # Visualize
            self.vis.visualize(start_T_vis)

    def destroy_node(self):
        """
        Cleanup and stop the visualization thread.
        """
        self.running = False
        super().destroy_node()
        self.visualization_thread.join()


def main(args=None):
    """
    Main function to initialize and run the node.
    """
    rclpy.init(args=args)
    node = VisualizerNode()
    executor = MultiThreadedExecutor(num_threads=5)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, SystemExit):
        node.destroy_node()


if __name__ == "__main__":
    main()
