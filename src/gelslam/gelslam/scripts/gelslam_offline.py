import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse

import cv2
import numpy as np
import open3d as o3d
import yaml
from cv_bridge import CvBridge

from gelslam.core.coverage_graph import CoverageGraph, create_visible_coverage_graph_msg
from gelslam.core.frame import center_of_frame_msg, frame_msg2mesh, pose_of_frame_msg
from gelslam.core.keyframe import KeyFrameDB, compute_adjusted_pointcloud
from gelslam.core.parent_groups_info import ParentGroupsInfo
from gelslam.core.pose_graph import PoseGraph, PoseGraphSolutions
from gelslam.core.tracker import Tracker
from gelslam.utils import Logger, pointcloud2mesh
from gelslam.visualization.visible_coverage_meshes import (
    VisibleCoverageMeshes,
    compute_visible_coverage_meshes,
)
from gelslam.visualization.visualizer import Visualizer

"""
Offline GelSLAM reconstruction.
"""


def main():
    """
    Main function for offline GelSLAM reconstruction.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Offline GelSLAM reconstruction.")
    parser.add_argument(
        "-d",
        "--data_dir",
        required=True,
        type=str,
        help="path to save data",
    )
    parser.add_argument(
        "-c",
        "--config_path",
        required=True,
        type=str,
        help="path to the sensor configuration file",
    )
    parser.add_argument(
        "-r",
        "--rendering",
        action="store_true",
        help="Render the reconstruction process",
    )
    parser.add_argument(
        "-s",
        "--save_gelslam_state",
        action="store_true",
        help="Save the gelslam state",
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
        ppmm = config["device_config"]["ppmm"]

    # Get the calibration model path
    calib_model_path = config["device_config"]["calibration_model_path"]
    if not os.path.isabs(calib_model_path):
        calib_model_path = os.path.join(
            os.path.dirname(args.config_path), calib_model_path
        )

    # Load tactile video
    data_dir = args.data_dir
    cap = cv2.VideoCapture(os.path.join(data_dir, "gelsight.avi"))
    images = []
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break
        images.append(image)
    cap.release()

    # Create save directory
    save_dir = os.path.join(data_dir, "gelslam_offline")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Initialize objects
    logger = Logger(print_output=True)
    tracker = Tracker(
        calib_model_path, config, skip_background_check=True, logger=logger
    )
    keyframedb = KeyFrameDB(ppmm, logger)
    pose_graph = PoseGraph(config)
    pose_graph_solutions = PoseGraphSolutions()
    parent_groups_info = ParentGroupsInfo()
    coverage_graph = CoverageGraph(config)
    if args.rendering:
        bridge = CvBridge()
        visible_coverage_meshes = VisibleCoverageMeshes()
        vis = Visualizer()
        vis.create_window()

    # Loop through all tactile images
    for image in images:
        # Track the new frame
        ret, track_result = tracker.track(image, log_prefix="Track")
        if not ret:
            continue
        frame_msg, keyframe_msgs = track_result
        # For all keyframes, do loop closures
        for keyframe_msg in keyframe_msgs:
            keyframedb.insert(keyframe_msg, log_prefix="KeyFrame Insert")
            targeted_size = keyframedb.size()
            updated_size = pose_graph_solutions.size()
            logger.info(
                "Loop Closure -- Begin: current keyframe database size: %d"
                % (targeted_size)
            )
            # Add odometry factors
            pose_graph.add_odometry_factors(keyframedb, updated_size, targeted_size)
            tar_kidx = targeted_size - 1
            # Detect and add loops
            matched_kidxs = pose_graph.detect_and_add_loops(
                keyframedb, tar_kidx, coverage_graph
            )
            logger.info(
                "Loop Closure -- Loop detected: keyframe index %d matched with keyframe indices %s"
                % (tar_kidx, str(matched_kidxs))
            )
            # Update parent groups due to new keyframes since last loop closure
            parent_groups_info.update_wrt_new_keyframes(
                keyframedb, updated_size, targeted_size
            )
            # Update parent groups based on detected loops
            (
                original_member_kidxs,
                new_member_kidxs,
                other_kidxs,
                removed_trial_groups,
            ) = parent_groups_info.update_wrt_loop_closure(
                keyframedb,
                tar_kidx,
                matched_kidxs,
                updated_size,
            )
            # Remove prior factors
            pose_graph.remove_prior_factors(removed_trial_groups)
            # Pose graph optimization
            pose_graph_solutions = pose_graph.solve()
            # Add new coverage node to coverage graph that represents new keyframes
            coverage_graph.add_new_coverage_nodes(updated_size, targeted_size)
            # Update the coverage graph based on the new keyframes from other trials
            coverage_graph.update_wrt_new_keyframes(
                other_kidxs,
                keyframedb,
                pose_graph_solutions,
                parent_groups_info,
            )
            # Update the coverage graph based on keyframes in the merged trials
            coverage_graph.update_wrt_loop_closure(
                original_member_kidxs,
                new_member_kidxs,
                keyframedb,
                pose_graph_solutions,
            )
            logger.info(
                "Loop Closure -- End: disjoint trials: %d, largest trial keyframe portions: %d/%d"
                % (
                    np.unique(parent_groups_info.parent_groups).size,
                    np.max(parent_groups_info.parent_group_sizes),
                    np.sum(parent_groups_info.parent_group_sizes),
                )
            )
            # Visualize the coverage meshes
            if args.rendering:
                visible_coverage_graph_msg = create_visible_coverage_graph_msg(
                    tar_kidx,
                    keyframedb,
                    pose_graph_solutions,
                    parent_groups_info,
                    coverage_graph,
                )
                # Update active meshes
                kid2kidx = keyframedb.get_kid2kidx()
                visible_coverage_meshes = compute_visible_coverage_meshes(
                    visible_coverage_graph_msg, keyframedb, kid2kidx
                )
                logger.info(
                    "Rendering -- Global Mesh (Visible Coverage Meshes) Updated: render with %.3fx scale"
                    % (1.0 / (2**visible_coverage_meshes.viz_level))
                )
                # Update keyframe's meshes in the visualizer
                targeted_size = keyframedb.size()
                ref_kidx = keyframedb.find_kidx_from_kid(
                    frame_msg.ref_kid, targeted_size
                )
                if ref_kidx is not None:
                    # "None" happens when failed to track, where two keyframe_msgs are created
                    new_trial_flag, start_T_ref = (
                        visible_coverage_meshes.get_keyframe_pose(ref_kidx, keyframedb)
                    )
                    vis.clear_geometries()
                    vis.add_visible_coverage_meshes(visible_coverage_meshes)
        # Visualize current frame and update viewing pose
        if args.rendering:
            # Update keyframe meshes
            if len(keyframe_msgs) > 0:
                if new_trial_flag:
                    vis.clear_geometries()
            # Update current frame mesh
            frame_mesh = frame_msg2mesh(
                bridge,
                frame_msg,
                start_T_ref,
                ppmm,
            )
            vis.update_curr_frame_mesh(frame_mesh)
            # Visualize
            start_T_frame = start_T_ref @ pose_of_frame_msg(frame_msg)
            frame_T_center = np.eye(4, dtype=np.float32)
            frame_T_center[:3, 3] = center_of_frame_msg(bridge, frame_msg, ppmm)
            start_T_vis = start_T_frame @ frame_T_center
            vis.visualize(start_T_vis)

    # Destroy visualizer
    if args.rendering:
        vis.destroy_window()

    # Merge and save reconstructed meshes
    updated_size = pose_graph_solutions.size()
    active_parent_group = parent_groups_info.get_largest_parent_group()
    active_kidxs = []
    for kidx in range(updated_size):
        parent_group = parent_groups_info.get_parent_group(kidx, keyframedb)
        if parent_group == active_parent_group:
            active_kidxs.append(kidx)
    # Construct a new and clean coverage graph
    coverage_graph = CoverageGraph(config)
    coverage_graph.add_new_coverage_nodes(0, updated_size)
    coverage_graph.update_wrt_new_keyframes(
        active_kidxs,
        keyframedb,
        pose_graph_solutions,
        parent_groups_info,
    )
    # Generate merged mesh
    merged_mesh = o3d.geometry.TriangleMesh()
    for kidx in active_kidxs:
        keyframe = keyframedb[kidx]
        if not coverage_graph[kidx].is_active:
            continue
        adjusted_pointcloud = compute_adjusted_pointcloud(
            kidx,
            coverage_graph[kidx].neighbor_kidxs,
            keyframedb,
            pose_graph_solutions,
        )
        keyframe_mesh = pointcloud2mesh(adjusted_pointcloud, keyframe.C)
        merged_mesh += keyframe_mesh
    save_path = os.path.join(save_dir, "reconstructed_mesh.ply")
    o3d.io.write_triangle_mesh(
        save_path,
        merged_mesh,
        write_ascii=False,
        compressed=False,
    )
    logger.info("Merging and saving mesh...")
    logger.info("Reconstructed mesh saved in %s" % (save_path))

    # Save states
    if args.save_gelslam_state:
        tracker.save(save_dir)
        keyframedb.save(save_dir)
        pose_graph.save(save_dir)
        pose_graph_solutions.save(save_dir)
        parent_groups_info.save(save_dir)
    logger.info("Done reconstructing!")


if __name__ == "__main__":
    main()
