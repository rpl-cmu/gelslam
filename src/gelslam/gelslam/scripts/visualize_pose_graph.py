import argparse
import os

import numpy as np
import open3d as o3d
import yaml

from gelslam.core.coverage_graph import CoverageGraph
from gelslam.core.keyframe import KeyFrameDB
from gelslam.core.parent_groups_info import ParentGroupsInfo
from gelslam.core.pose_graph import PoseGraph, PoseGraphSolutions
from gelslam.utils import Logger, matrix_in_m


def create_edge_mesh(point1, point2, color=[0, 0, 0], radius=0.00005):
    """
    Creates a cylinder mesh representing an edge between two points.

    :param point1: np.ndarray (3,); The first point position (in meters).
    :param point2: np.ndarray (3,); The second point position (in meters).
    :param color: list of float; The color of the edge.
    :param radius: float; The radius of the edge (in meters).
    :return: open3d.geometry.TriangleMesh; The edge mesh.
    """
    height = np.linalg.norm(point2 - point1) + 1e-5
    edge_mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
    edge_mesh.paint_uniform_color(color)
    midpoint = (point1 + point2) / 2
    direction = (point2 - point1) / height
    default_direction = np.array([0, 0, 1])
    # Compute rotation matrix to align the cylinder
    if not np.allclose(direction, default_direction):
        axis = np.cross(default_direction, direction)
        axis_length = np.linalg.norm(axis)
        if axis_length > 1e-6:
            axis = axis / axis_length
            angle = np.arccos(np.dot(default_direction, direction))
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
            edge_mesh.rotate(R, center=(0, 0, 0))
    # Translate to the midpoint
    edge_mesh.translate(midpoint)
    return edge_mesh


def main():
    """
    Main function to visualize the pose graph.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Plot the pose graph.")
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
        "-m",
        "--method",
        type=str,
        required=True,
        choices=["gelslam_online", "gelslam_offline"],
        help="method of running the reconstruction",
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
        ppmm = config["device_config"]["ppmm"]
    # Load states
    data_dir = args.data_dir
    load_dir = os.path.join(data_dir, args.method)
    logger = Logger()
    pose_graph = PoseGraph.load(load_dir)
    keyframedb = KeyFrameDB.load(load_dir, logger)
    parent_groups_info = ParentGroupsInfo.load(load_dir)
    pose_graph_solutions = PoseGraphSolutions.load(load_dir)
    updated_size = pose_graph_solutions.size()
    # Find the keyframes from the largest connected group of keyframes
    active_parent_group = parent_groups_info.get_largest_parent_group()
    active_kidxs = []
    for kidx in range(updated_size):
        parent_group = parent_groups_info.get_parent_group(kidx, keyframedb)
        if parent_group == active_parent_group:
            active_kidxs.append(kidx)
    # Construct coverage graph
    coverage_graph = CoverageGraph(config)
    coverage_graph.add_new_coverage_nodes(0, updated_size)
    coverage_graph.update_wrt_new_keyframes(
        active_kidxs,
        keyframedb,
        pose_graph_solutions,
        parent_groups_info,
    )

    # Calculate the pose of each keyframe's patch center
    keyframe_centers = []
    for kidx in range(updated_size):
        keyframe = keyframedb[kidx]
        C = keyframe.C
        H = keyframe.H
        start_T_curr = matrix_in_m(
            pose_graph.graph_result.atPose3(kidx).matrix()
        ).astype(np.float32)
        cy, cx = np.mean(np.column_stack(np.where(C)), axis=0)
        cz = H[int(cy), int(cx)] - 8
        center = (
            np.array([cx - C.shape[1] / 2.0, cy - C.shape[0] / 2.0, cz]) * ppmm / 1000
        )
        center = np.dot(start_T_curr[:3, :3], center) + start_T_curr[:3, 3]
        keyframe_centers.append(center)

    vis_objects = []
    # Create node meshes
    for kidx in range(updated_size):
        parent_group = parent_groups_info.get_parent_group(kidx, keyframedb)
        if parent_group != active_parent_group:
            continue
        if coverage_graph[kidx].is_active:
            node_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.0004)
            node_mesh.paint_uniform_color([0, 1, 0])
        else:
            node_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.00025)
            node_mesh.paint_uniform_color([0, 0, 0])
        node_mesh.translate(keyframe_centers[kidx])
        vis_objects.append(node_mesh)
    # Create edge meshes
    for i in range(pose_graph.graph.size()):
        factor = pose_graph.graph.at(i)
        if len(factor.keys()) == 1:
            continue
        kidx1, kidx2 = factor.keys()
        parent_group1 = parent_groups_info.get_parent_group(kidx1, keyframedb)
        if parent_group1 != active_parent_group:
            continue
        parent_group2 = parent_groups_info.get_parent_group(kidx2, keyframedb)
        if parent_group2 != active_parent_group:
            continue
        edge_mesh = create_edge_mesh(
            keyframe_centers[kidx1], keyframe_centers[kidx2], color=[1, 0, 0]
        )
        vis_objects.append(edge_mesh)

    # Load the reconstructed mesh
    mesh = o3d.io.read_triangle_mesh(os.path.join(load_dir, "reconstructed_mesh.ply"))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.3, 0.3, 0.3])
    vis_objects.append(mesh)
    o3d.visualization.draw_geometries(vis_objects)


if __name__ == "__main__":
    main()
