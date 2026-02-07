import time

import numpy as np
import open3d as o3d


class Visualizer:
    """
    Handles real-time 3D rendering using Open3D.
    """

    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        # The current frame mesh
        self.curr_frame_mesh = None

    def create_window(self):
        """
        Creates the visualization window.
        """
        self.vis.create_window()

    def destroy_window(self):
        """
        Destroys the visualization window.
        """
        self.vis.destroy_window()

    def add_geometry(self, mesh):
        """
        Adds a geometry to the visualizer.

        :param mesh: open3d.geometry; The geometry to add.
        """
        self.vis.add_geometry(mesh)

    def remove_geometry(self, mesh):
        """
        Removes a geometry from the visualizer.

        :param mesh: open3d.geometry; The geometry to remove.
        """
        self.vis.remove_geometry(mesh)

    def clear_geometries(self):
        """
        Clears all geometries from the visualizer.
        """
        self.vis.clear_geometries()

    def add_visible_coverage_meshes(self, visible_coverage_meshes):
        """
        Adds visible coverage meshes to the visualizer.

        :param visible_coverage_meshes: VisibleCoverageMeshes; The visible coverage meshes.
        """
        for mesh in visible_coverage_meshes.visible_coverage_meshes:
            self.add_geometry(mesh)

    def update_curr_frame_mesh(self, curr_frame_mesh):
        """
        Updates the currently visualized frame mesh.

        :param curr_frame_mesh: open3d.geometry.TriangleMesh; The current frame mesh.
        """
        if self.curr_frame_mesh is not None:
            self.remove_geometry(self.curr_frame_mesh)
        self.curr_frame_mesh = curr_frame_mesh
        self.add_geometry(self.curr_frame_mesh)

    def visualize(self, start_T_vis):
        """
        Renders the scene.
        Updates the camera view based on the current frame pose (follow cam).

        :param start_T_vis: np.ndarray; The camera pose.
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
        """
        Updates the renderer without changing the view or scene content.
        """
        self.vis.poll_events()
        self.vis.update_renderer()
        time.sleep(0.0001)
