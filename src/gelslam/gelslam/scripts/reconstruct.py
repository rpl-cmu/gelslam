import argparse
import os

import open3d as o3d
import yaml

from gelslam.core.coverage_graph import CoverageGraph
from gelslam.core.keyframe import KeyFrameDB, compute_adjusted_pointcloud
from gelslam.core.parent_groups_info import ParentGroupsInfo
from gelslam.core.pose_graph import PoseGraphSolutions
from gelslam.utils import Logger, pointcloud2mesh

config_path = os.path.join(os.path.dirname(__file__), "../../config/config.yaml")


def reconstruct():
    """
    Main function for offline reconstruction.
    Loads saved data and generates a merged mesh.
    """
    # Argument Parser
    parser = argparse.ArgumentParser(description="Offline reconstruct.")
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        help="path to save data",
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        default=config_path,
        help="path to the sensor configuration file",
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="gelslam_online",
        choices=["gelslam_online", "gelslam_offline"],
        help="method of running the reconstruction",
    )
    args = parser.parse_args()

    # Read the configuration
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
        ppmm = config["device_config"]["ppmm"]

    # Load the reconstruction data
    data_dir = args.data_dir
    save_dir = os.path.join(data_dir, args.method)
    load_dir = save_dir
    logger = Logger()
    keyframedb = KeyFrameDB.load(load_dir, logger)
    parent_groups_info = ParentGroupsInfo.load(load_dir)
    pose_graph_solutions = PoseGraphSolutions.load(load_dir)
    updated_size = pose_graph_solutions.size()
    # Find the active kidxs
    active_parent_group = parent_groups_info.get_largest_parent_group()
    active_kidxs = []
    for kidx in range(updated_size):
        parent_group = parent_groups_info.get_parent_group(kidx, keyframedb)
        if parent_group == active_parent_group:
            active_kidxs.append(kidx)
    # Construct the coverage graph
    coverage_graph = CoverageGraph(config)
    coverage_graph.add_new_coverage_nodes(0, updated_size)
    coverage_graph.update_wrt_new_keyframes(
        active_kidxs,
        keyframedb,
        pose_graph_solutions,
        parent_groups_info,
    )
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
        os.path.join(save_dir, "merged_mesh.ply"),
        merged_mesh,
        write_ascii=False,
        compressed=False,
    )


if __name__ == "__main__":
    reconstruct()
