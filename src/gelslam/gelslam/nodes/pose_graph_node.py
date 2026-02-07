import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import threading
import time

import numpy as np
import open3d as o3d
import rclpy
import yaml
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from gelslam.core.coverage_graph import CoverageGraph, create_visible_coverage_graph_msg
from gelslam.core.keyframe import KeyFrameDB, compute_adjusted_pointcloud
from gelslam.core.parent_groups_info import ParentGroupsInfo
from gelslam.core.pose_graph import PoseGraph, PoseGraphSolutions
from gelslam.utils import Logger, pointcloud2mesh
from gelslam_msgs.msg import KeyFrameMsg, VisibleCoverageGraphMsg

"""
Two threads:
1. K thread (Keyframe callback):
    - Adds keyframe to the KeyFrameDB.
2. L thread (Loop closure):
    - Detects loops.
    - Performs Loop Closure: updates parent groups, solves poses, updates coverage graph.
    - Publishes the visible coverage graph message.
"""


class PoseGraphNode(Node):
    """
    ROS2 node for backend pose graph optimization.
    """

    def __init__(self):
        super().__init__("pose_graph_node")
        # Load the configuration
        self.declare_parameter("config_path", "")
        config_path = (
            self.get_parameter("config_path").get_parameter_value().string_value
        )
        config_path = os.path.abspath(os.path.expanduser(config_path))
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        # Get the directory to save the pose graph states
        self.declare_parameter("data_dir", "")
        data_dir = self.get_parameter("data_dir").get_parameter_value().string_value
        data_dir = os.path.abspath(os.path.expanduser(data_dir))
        self.save_dir = os.path.join(data_dir, "gelslam_online")
        # Get the online rendering flag
        self.declare_parameter("online_rendering", True)
        self.online_rendering = (
            self.get_parameter("online_rendering").get_parameter_value().bool_value
        )

        # The overall running flag
        self.running = True

        # The logger
        self.logger = Logger(ros_logger=self.get_logger())

        # KeyframeDB (keyframe database) is never modified except appending.
        # K thread writes (appending), L thread reads
        self.add_keyframe_lock = threading.Lock()
        self.keyframedb = KeyFrameDB(
            ppmm=config["device_config"]["ppmm"], logger=self.logger
        )

        # Parent groups, pose graph, pose graph solutions, and coverage graph are updated in L thread.
        self.parent_groups_info = ParentGroupsInfo()
        self.pose_graph = PoseGraph(config)
        self.pose_graph_solutions = PoseGraphSolutions()
        self.coverage_graph = CoverageGraph(config)
        if self.online_rendering:
            self.visible_coverage_graph_pub = self.create_publisher(
                VisibleCoverageGraphMsg, "visible_coverage_graph", 1
            )

        # Initialize KeyFrame subscriber (K thread)
        self.create_subscription(
            KeyFrameMsg,
            "keyframe",
            self.keyframe_callback,
            10,
        )

        # Start the L thread: loop detection and pose graph optimization thread
        self.loop_closure_thread = threading.Thread(target=self.loop_closure)
        self.loop_closure_thread.start()

    def keyframe_callback(self, keyframe_msg):
        """
        Adds received keyframe to KeyFrameDB (K thread).

        :param keyframe_msg: KeyFrameMsg; The received keyframe message.
        """
        with self.add_keyframe_lock:
            self.keyframedb.insert(keyframe_msg, log_prefix="Keyframe Insert Thread")

    def loop_closure(self):
        """
        Loop closure thread (L thread).
        Detects loops, merges parent groups, solves poses, and updates coverage graph.

        Steps:
        1. Detect loops, add loop closing factors to the pose graph. (Expensive)
        2. Perform loop closure updates. (Faster)
        """
        while self.running:
            # Get the targeted number of keyframes
            with self.add_keyframe_lock:
                targeted_size = self.keyframedb.size()
            # Skip if no new keyframes since last loop closure
            updated_size = self.pose_graph_solutions.size()
            if targeted_size == 0 or targeted_size == updated_size:
                time.sleep(0.01)
                continue

            self.logger.info(
                "Loop Closure Thread -- Begin: current keyframe database size: %d"
                % targeted_size
            )
            # Add the odometry factors to the graph
            self.pose_graph.add_odometry_factors(
                self.keyframedb, updated_size, targeted_size
            )

            # Detect loops
            tar_kidx = targeted_size - 1
            matched_kidxs = self.pose_graph.detect_and_add_loops(
                self.keyframedb, tar_kidx, self.coverage_graph
            )
            self.logger.info(
                "Loop Closure Thread -- Loop detected: keyframe index %d matched with keyframe indices %s"
                % (tar_kidx, str(matched_kidxs))
            )

            # Update parent groups due to newly introduced keyframes since last loop closure
            self.parent_groups_info.update_wrt_new_keyframes(
                self.keyframedb, updated_size, targeted_size
            )
            # Update parent groups due to newly detected loops
            (
                original_member_kidxs,
                new_member_kidxs,
                other_kidxs,
                removed_trial_groups,
            ) = self.parent_groups_info.update_wrt_loop_closure(
                self.keyframedb,
                tar_kidx,
                matched_kidxs,
                updated_size,
            )
            # Remove prior factors from pose graph due to trial fusion
            self.pose_graph.remove_prior_factors(removed_trial_groups)
            # Pose graph optimization
            self.pose_graph_solutions = self.pose_graph.solve()

            # Add new coverage node to coverage graph that represents new keyframes
            self.coverage_graph.add_new_coverage_nodes(updated_size, targeted_size)
            # Update the coverage graph based on the new keyframes from other trials
            self.coverage_graph.update_wrt_new_keyframes(
                other_kidxs,
                self.keyframedb,
                self.pose_graph_solutions,
                self.parent_groups_info,
            )
            # Update the coverage graph based on keyframes in the merged trials
            self.coverage_graph.update_wrt_loop_closure(
                original_member_kidxs,
                new_member_kidxs,
                self.keyframedb,
                self.pose_graph_solutions,
            )
            # Publish the visible coverage graph message for rendering
            if self.online_rendering:
                self.visible_coverage_graph_pub.publish(
                    create_visible_coverage_graph_msg(
                        tar_kidx,
                        self.keyframedb,
                        self.pose_graph_solutions,
                        self.parent_groups_info,
                        self.coverage_graph,
                    )
                )

            # Log the connectability information
            n_disjoint_trials = np.unique(self.parent_groups_info.parent_groups).size

            self.logger.info(
                "Loop Closure Thread -- End: disjoint trials: %d, largest trial keyframe portions: %d/%d"
                % (
                    np.unique(self.parent_groups_info.parent_groups).size,
                    np.max(self.parent_groups_info.parent_group_sizes),
                    np.sum(self.parent_groups_info.parent_group_sizes),
                )
            )

    def merge_mesh(self):
        """
        Find the keyframes from the largest connected group of keyframes.
        Merging the meshes and save the reconstructed mesh.
        """
        updated_size = self.pose_graph_solutions.size()
        active_parent_group = self.parent_groups_info.get_largest_parent_group()
        active_kidxs = []
        for kidx in range(updated_size):
            parent_group = self.parent_groups_info.get_parent_group(
                kidx, self.keyframedb
            )
            if parent_group == active_parent_group:
                active_kidxs.append(kidx)
        # Merging the meshes
        merged_mesh = o3d.geometry.TriangleMesh()
        for kidx in active_kidxs:
            keyframe = self.keyframedb[kidx]
            if not self.coverage_graph[kidx].is_active:
                continue
            adjusted_pointcloud = compute_adjusted_pointcloud(
                kidx,
                self.coverage_graph[kidx].neighbor_kidxs,
                self.keyframedb,
                self.pose_graph_solutions,
            )
            keyframe_mesh = pointcloud2mesh(adjusted_pointcloud, keyframe.C)
            merged_mesh += keyframe_mesh
        save_path = os.path.join(self.save_dir, "reconstructed_mesh.ply")
        o3d.io.write_triangle_mesh(
            save_path,
            merged_mesh,
            write_ascii=False,
            compressed=False,
        )
        self.logger.info("Merging and saving mesh...")
        self.logger.info("Reconstructed mesh saved in %s" % (save_path))

    def destroy_node(self):
        """
        Cleanup node and stop the loop closure thread.
        Merge the meshes for reconstruction.
        Save pose graph states.
        """
        self.running = False
        super().destroy_node()
        self.loop_closure_thread.join()

        # Merge the meshes for reconstruction
        self.merge_mesh()

        # Save pose graph states
        self.pose_graph.save(self.save_dir)
        self.keyframedb.save(self.save_dir)
        self.parent_groups_info.save(self.save_dir)
        self.pose_graph_solutions.save(self.save_dir)


def main(args=None):
    """
    Main function to initialize and run the node.
    """
    rclpy.init(args=args)
    node = PoseGraphNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, SystemExit):
        node.destroy_node()


if __name__ == "__main__":
    main()
