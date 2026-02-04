import time
import numpy as np
import open3d as o3d
from gelslam.utils import Logger


class Visualizer:
    """
    This is the visualizer class for real-time rendering using open3d.
    """

    def __init__(self, logger=None):
        self.logger = Logger(logger)
        self.vis = o3d.visualization.Visualizer()
        # The current frame mesh
        self.curr_frame_mesh = None

    def create_window(self):
        self.vis.create_window()

    def destroy_window(self):
        self.vis.destroy_window()

    def add_geometry(self, mesh):
        self.vis.add_geometry(mesh)

    def remove_geometry(self, mesh):
        self.vis.remove_geometry(mesh)

    def clear_geometries(self):
        self.vis.clear_geometries()

    def add_visible_coverage_meshes(self, visible_coverage_meshes):
        for mesh in visible_coverage_meshes.visible_coverage_meshes:
            self.add_geometry(mesh)

    def update_curr_frame_mesh(self, curr_frame_mesh):
        """Update the current frame mesh based on new current frame mesh."""
        if self.curr_frame_mesh is not None:
            self.remove_geometry(self.curr_frame_mesh)
        self.curr_frame_mesh = curr_frame_mesh
        self.add_geometry(self.curr_frame_mesh)

    def visualize(self, start_T_vis):
        """
        Visualize the scene with regards to the center point of the scene.
        (binds with the current frame's center).
        """
        # Compute the extrinsic matrix
        extrinsic_matrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.03],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        extrinsic_matrix = extrinsic_matrix @ np.linalg.inv(start_T_vis)
        ctr = self.vis.get_view_control()
        params = ctr.convert_to_pinhole_camera_parameters()
        params.extrinsic = extrinsic_matrix
        ctr.convert_from_pinhole_camera_parameters(params, True)
        self.vis.poll_events()
        self.vis.update_renderer()
        time.sleep(0.0001)

    def visualize_nothing(self):
        """Visualize the scene without switching the view."""
        self.vis.poll_events()
        self.vis.update_renderer()
        time.sleep(0.0001)
