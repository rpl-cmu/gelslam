import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
import copy
import threading
import yaml

from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node

from gelslam_msgs.msg import KeyFrameMsg, FrameMsg, VisibleCoverageGraphMsg
from gelslam.core.frame import (
    center_of_frame_msg,
    pose_of_frame_msg,
    frame_msg2mesh,
)
from gelslam.visualization.visualizer import Visualizer
from gelslam.visualization.visible_coverage_meshes import (
    compute_visible_coverage_meshes,
    VisibleCoverageMeshes,
)
from gelslam.core.keyframe import KeyFrameDB

"""
Four threads:
1. K thread, Keyframe callback thread:
    - Add keyframe to the keyframe DB.
2. F thread, Frame callback thread:
    - Update the current frame.
3. M thread, Mesh update thread:
    - Update the VisibleCoverageMeshes based on the received VisibleCoverageGraphMsg from loop closure.
4. V thread, Visualization thread:
    - Visualize the current frame and the visible coverage meshes.
"""


class VisualizerNode(Node):
    def __init__(self):
        super().__init__("pose_graph_node")
        self.bridge = CvBridge()
        # Get the configuration path and load the configuration
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

        # The keyframe Database is never modified except appending.
        # It will only be appended in the K thread.
        self.add_keyframe_lock = threading.Lock()
        self.keyframedb = KeyFrameDB(
            ppmm=self.ppmm, logger=self.get_logger()
        )  # keyframe DB: K thread write, V thread read

        # The visible coverage meshes is updated only in the M thread, which is triggered by the loop closure.
        self.visible_coverage_meshes_update_lock = threading.Lock()
        self.visible_coverage_meshes = VisibleCoverageMeshes()
        self.visible_coverage_meshes_update_count = 0

        # The frame information is only modified in the F thread
        self.update_frame_lock = threading.Lock()
        self.curr_frame_msg = None

        # For viz purpose, we record these to check if new frame is updated, and if visible coverage meshes is updated
        self.prev_vis_fid = None
        self.prev_visible_coverage_meshes_update_count = 0

        # Create the KeyFrame Subscriber
        self.keyframe_callback_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(
            KeyFrameMsg,
            "keyframe",
            self.keyframe_callback,
            10,
            callback_group=self.keyframe_callback_group,
        )

        # Create the Frame Subscriber
        self.frame_callback_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(
            FrameMsg,
            "frame",
            self.frame_callback,
            1,
            callback_group=self.frame_callback_group,
        )

        # Create the visible coverage graph Subscriber that updates the visible coverage meshes
        self.visible_coverage_graph_callback_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(
            VisibleCoverageGraphMsg,
            "visible_coverage_graph",
            self.visible_coverage_graph_callback,
            1,
            callback_group=self.visible_coverage_graph_callback_group,
        )

        # Start the visualization thread
        self.visualization_thread = threading.Thread(target=self.visualization)
        self.visualization_thread.start()

    def frame_callback(self, msg):
        """
        This callback will visualize the mesh if could be updated.
        Includes two steps, update keyframe mesh and render the frame.
        """
        with self.update_frame_lock:
            self.curr_frame_msg = msg

    def keyframe_callback(self, keyframe_msg):
        """
        The keyframe callback wills add the keyframe to the keyframe DB
        """
        # Put the keyframe in the keyframe DB
        with self.add_keyframe_lock:
            self.keyframedb.insert(
                keyframe_msg, log_prefix="Visualizer KeyFrame Adding"
            )

    def visible_coverage_graph_callback(self, msg):
        """
        The callback that updates the visible coverage meshes based on the received VisibleCoverageGraphMsg.
        """
        # Get the mapping from kid to kidx, since the message sticks to kid not kidx
        with self.add_keyframe_lock:
            kid2kidx = self.keyframedb.get_kid2kidx()
        # New loop closure detected, compute the visible coverage meshes
        tic = time.time()
        visible_coverage_meshes = compute_visible_coverage_meshes(
            msg, self.keyframedb, kid2kidx
        )
        toc = time.time()
        # Update the visible coverage meshes
        with self.visible_coverage_meshes_update_lock:
            self.visible_coverage_meshes = visible_coverage_meshes
            self.visible_coverage_meshes_update_count += 1
        self.get_logger().info(
            "Visualization -- Visible Coverage Meshes Updated! Take time: %.3f, Viz Level: %d"
            % (toc - tic, visible_coverage_meshes.viz_level)
        )

    def visualization(self):
        """
        Visualization has its own thread.
        Visualization is only based on the visible coverage meshes and the current frame.
        """
        # Create the visualizer
        self.vis = Visualizer(logger=self.get_logger())
        self.vis.create_window()

        while self.running:
            # Copy and get the information from frame callback
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
            # Skip if the reference keyframe is not in the keyframe DB
            if ref_kidx is None:
                self.get_logger().info(
                    "Visualization -- Skip since reference keyframe not found."
                )
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
                    self.get_logger().info(
                        "Visualization -- Visible Coverage Meshes Added!"
                    )

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
            # Visualize
            start_T_frame = start_T_ref @ pose_of_frame_msg(curr_frame_msg)
            frame_T_center = np.eye(4, dtype=np.float32)
            frame_T_center[:3, 3] = center_of_frame_msg(
                self.bridge, curr_frame_msg, self.ppmm
            )
            start_T_vis = start_T_frame @ frame_T_center
            self.vis.visualize(start_T_vis)

    def destroy_node(self):
        # Destroy the node
        self.running = False
        super().destroy_node()
        # Stop the visualization thread
        self.visualization_thread.join()


def main(args=None):
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
