import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import open3d as o3d
import pickle
import time
import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
import yaml

from gelslam_msgs.msg import KeyFrameMsg, VisibleCoverageGraphMsg
from gelslam.core.keyframe import KeyFrame, KeyFrameDB, compute_adjusted_pointcloud
from gelslam.core.pose_graph import PoseGraph, PoseGraphSolutions
from gelslam.core.parent_groups_info import ParentGroupsInfo
from gelslam.core.coverage_graph import (
    create_visible_coverage_graph_msg,
    CoverageGraph,
)
from gelslam.utils import pointcloud2mesh


"""
Two threads:
1. K thread, Keyframe callback thread:
    - Add keyframe to the keyframe DB.
2. L thread, Loop closure thread:
    - Detect loops.
    - Loop Closure: update parent groups, solve poses, update coverage graph.
    - Publish the visible coverage graph message.
"""


class PoseGraphNode(Node):
    def __init__(self):
        super().__init__("pose_graph_node")
        # Get the configuration path and load the configuration
        self.declare_parameter("config_path", "")
        config_path = (
            self.get_parameter("config_path").get_parameter_value().string_value
        )
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        # Get the save data directory
        self.declare_parameter("data_dir", "")
        data_dir = self.get_parameter("data_dir").get_parameter_value().string_value
        self.save_dir = os.path.join(data_dir, "gelslam_online")
        # Get the online rendering flag
        self.declare_parameter("online_rendering", True)
        self.online_rendering = (
            self.get_parameter("online_rendering").get_parameter_value().bool_value
        )

        # The overall running flag
        self.running = True

        # The keyframe Database is never modified except appending.
        # It will only be appended in the K thread.
        self.add_keyframe_lock = threading.Lock()
        self.keyframedb = KeyFrameDB(
            ppmm=config["device_config"]["ppmm"], logger=self.get_logger()
        )  # keyframe DB: K thread write, L thread read

        # The parent_groups, pose_graph_solutions, and coverage_graph are updated together in the L thread.
        self.parent_groups_info = ParentGroupsInfo(logger=self.get_logger())
        self.pose_graph = PoseGraph(config, logger=self.get_logger())
        self.pose_graph_solutions = PoseGraphSolutions()
        self.coverage_graph = CoverageGraph(config, logger=self.get_logger())
        if self.online_rendering:
            self.visible_coverage_graph_pub = self.create_publisher(
                VisibleCoverageGraphMsg, "visible_coverage_graph", 1
            )

        # Create the KeyFrame Subscriber
        self.create_subscription(
            KeyFrameMsg,
            "keyframe",
            self.keyframe_callback,
            10,
        )

        # Start the loop detection and solve thread
        self.loop_closure_thread = threading.Thread(target=self.loop_closure)
        self.loop_closure_thread.start()

    def keyframe_callback(self, keyframe_msg):
        """
        The keyframe callback wills add the keyframe to the keyframe DB
        """
        # Put the keyframe in the keyframe DB
        with self.add_keyframe_lock:
            self.keyframedb.insert(
                keyframe_msg, log_prefix="Pose Graph KeyFrame Adding"
            )

    def loop_closure(self):
        """
        This thread detect loops, merge parent groups, solve poses, and update coverage graph.
        It includes two steps:
        1. Detect loops, add loop closing factors to the pose graph. (Time Consuming)
        2. Loop Closure: merge parent groups, solve poses, update coverage graph. (Less time consuming)
        """
        while self.running:
            # Get the target number of keyframes to be fully updated (loop closed)
            with self.add_keyframe_lock:
                targeted_size = self.keyframedb.size()
            # Skip loop closure if no keyframe added to DB since last loop closure
            updated_size = self.pose_graph_solutions.size()
            if targeted_size == 0 or targeted_size == updated_size:
                time.sleep(0.01)
                continue

            self.get_logger().info(
                "Loop Closure -- Initiated! DB size: %d, Updated pose size: %d"
                % (targeted_size, updated_size)
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
            self.get_logger().info(
                "Loop Closure -- matched indices to idx %d: %s"
                % (tar_kidx, str(matched_kidxs))
            )

            # Loop Closure: Solve poses, update parent groups, and update coverage graph
            # Update parent groups due to new keyframes since the previous loop closure
            self.parent_groups_info.update_wrt_new_keyframes(
                self.keyframedb, updated_size, targeted_size
            )
            # Update parent groups due to detected loops
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
            # Remove prior factors from pose graph
            self.pose_graph.remove_prior_factors(removed_trial_groups)
            # Solve the pose graph
            self.pose_graph_solutions = self.pose_graph.solve()

            # Add new coverage node to coverage graph that represents new keyframes
            self.coverage_graph.add_new_coverage_nodes(updated_size, targeted_size)
            # Update the coverage graph based on the new keyframes that is not merged
            self.coverage_graph.update_wrt_new_keyframes(
                other_kidxs,
                self.keyframedb,
                self.pose_graph_solutions,
                self.parent_groups_info,
                log_prefix="Loop Closure",
            )
            # Update the coverage graph based on the loop closure
            self.coverage_graph.update_wrt_loop_closure(
                original_member_kidxs,
                new_member_kidxs,
                self.keyframedb,
                self.pose_graph_solutions,
                log_prefix="Loop Closure",
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

            # Log to check if everything is correctly updates
            self.get_logger().info(
                "Loop Closure -- Finished! Updated pose size: %d, parent_groups: %s, parent_group_sizes: %s"
                % (
                    self.pose_graph_solutions.size(),
                    str(self.parent_groups_info.parent_groups),
                    str(self.parent_groups_info.parent_group_sizes),
                )
            )

    def merge_mesh(self):
        # Find the active kidxs
        updated_size = self.pose_graph_solutions.size()
        active_parent_group = self.parent_groups_info.get_largest_parent_group()
        active_kidxs = []
        for kidx in range(updated_size):
            parent_group = self.parent_groups_info.get_parent_group(
                kidx, self.keyframedb
            )
            if parent_group == active_parent_group:
                active_kidxs.append(kidx)
        # Get the merged mesh
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
        o3d.io.write_triangle_mesh(
            os.path.join(self.save_dir, "reconstructed_mesh.ply"),
            merged_mesh,
            write_ascii=False,
            compressed=False,
        )

    def destroy_node(self):
        # Destroy the node
        self.running = False
        super().destroy_node()
        # Stop the loop closure thread
        self.loop_closure_thread.join()

        # Reconstruct the mesh
        self.merge_mesh()

        # Save the pose graph optimization result
        self.pose_graph.save(self.save_dir)
        self.keyframedb.save(self.save_dir)
        self.parent_groups_info.save(self.save_dir)
        self.pose_graph_solutions.save(self.save_dir)


def main(args=None):
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
