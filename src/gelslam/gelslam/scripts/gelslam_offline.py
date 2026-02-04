import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse

import open3d as o3d
import cv2
import numpy as np
import yaml
from cv_bridge import CvBridge

from gelslam.core.tracker import Tracker
from gelslam.core.keyframe import KeyFrameDB, compute_adjusted_pointcloud
from gelslam.core.pose_graph import PoseGraph, PoseGraphSolutions
from gelslam.core.parent_groups_info import ParentGroupsInfo
from gelslam.core.coverage_graph import (
    create_visible_coverage_graph_msg,
    CoverageGraph,
)
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
from gelslam.utils import pointcloud2mesh

"""
Offline reconstruction (the full pipeline) without visualization.
If you want detailed debugging or final reconstruction, this is the script to run.
"""


def main():
    # Argument Parser
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

    args = parser.parse_args()

    # Read the configuration
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
        ppmm = config["device_config"]["ppmm"]

    # Get the calibration model path
    calib_model_path = config["device_config"]["calibration_model_path"]
    if not os.path.isabs(calib_model_path):
        calib_model_path = os.path.join(
            os.path.dirname(args.config_path), calib_model_path
        )

    # Load the tactile video and the background image
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

    # Prepare the objects
    tracker = Tracker(calib_model_path, config)
    keyframedb = KeyFrameDB(ppmm=ppmm)
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
        ret, track_result = tracker.track(image, log_prefix="Track")
        if not ret:
            continue
        frame_msg, keyframe_msgs = track_result
        # For all keyframe, do loop closures
        for keyframe_msg in keyframe_msgs:
            keyframedb.insert(keyframe_msg, log_prefix="KeyFrame Adding")
            # Loop Closure
            targeted_size = keyframedb.size()
            updated_size = pose_graph_solutions.size()
            print(
                "Loop Closure -- Initiated! DB size: %d, Updated pose size: %d"
                % (targeted_size, updated_size)
            )
            # Add the odometry factors to the graph
            pose_graph.add_odometry_factors(keyframedb, updated_size, targeted_size)
            tar_kidx = targeted_size - 1
            matched_kidxs = pose_graph.detect_and_add_loops(
                keyframedb, tar_kidx, coverage_graph
            )
            print(
                "Loop Closure -- matched indices to idx %d: %s"
                % (tar_kidx, str(matched_kidxs))
            )
            # Update parent groups due to new keyframes since the previous loop closure
            parent_groups_info.update_wrt_new_keyframes(
                keyframedb, updated_size, targeted_size
            )
            # Update parent groups due to detected loops
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
            # Remove prior factors from pose graph
            pose_graph.remove_prior_factors(removed_trial_groups)
            # Solve the pose graph
            pose_graph_solutions = pose_graph.solve()
            # Add new coverage node to coverage graph that represents new keyframes
            coverage_graph.add_new_coverage_nodes(updated_size, targeted_size)
            # Update the coverage graph based on the new keyframes that is not merged
            coverage_graph.update_wrt_new_keyframes(
                other_kidxs,
                keyframedb,
                pose_graph_solutions,
                parent_groups_info,
                log_prefix="Loop Closure",
            )
            # Update the coverage graph based on the loop closure
            coverage_graph.update_wrt_loop_closure(
                original_member_kidxs,
                new_member_kidxs,
                keyframedb,
                pose_graph_solutions,
                log_prefix="Loop Closure",
            )
            # coverage_graph.log_coverage_graph(log_prefix="Loop Closure")
            # Log to check if everything is correctly updates
            print(
                "Loop Closure -- Finished! Updated pose size: %d, parent_groups: %s, parent_group_sizes: %s"
                % (
                    pose_graph_solutions.size(),
                    str(parent_groups_info.parent_groups),
                    str(parent_groups_info.parent_group_sizes),
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
                print("Rendering -- Visible Coverage Meshes Updated!")
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
        # Visualize the current frame's mesh and adjust the camera pose
        if args.rendering:
            # Update the keyframe's meshes
            if len(keyframe_msgs) > 0:
                if new_trial_flag:
                    vis.clear_geometries()
            # Update current frame's mesh
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
            print("Rendering -- Visualize Frame ID: %d" % (frame_msg.fid))

    # Destroy the visualizer
    if args.rendering:
        vis.destroy_window()

    # Merge and save the recosntructed meshes
    updated_size = pose_graph_solutions.size()
    active_parent_group = parent_groups_info.get_largest_parent_group()
    active_kidxs = []
    for kidx in range(updated_size):
        parent_group = parent_groups_info.get_parent_group(kidx, keyframedb)
        if parent_group == active_parent_group:
            active_kidxs.append(kidx)
    # Get the merged mesh
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
    o3d.io.write_triangle_mesh(
        os.path.join(save_dir, "reconstructed_mesh.ply"),
        merged_mesh,
        write_ascii=False,
        compressed=False,
    )

    # Save the results
    tracker.save(save_dir)
    keyframedb.save(save_dir)
    pose_graph.save(save_dir)
    pose_graph_solutions.save(save_dir)
    parent_groups_info.save(save_dir)
    coverage_graph.save(save_dir)
    print("Bye!")


if __name__ == "__main__":
    main()
