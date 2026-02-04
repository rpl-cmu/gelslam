import numpy as np
from gelslam.core.pose_graph import PoseGraphSolutions
from gelslam.core.keyframe import compute_adjusted_pointcloud
from gelslam.utils import pointcloud2mesh


def compute_visible_coverage_meshes(visible_coverage_graph_msg, keyframedb, kid2kidx):
    """
    Compute the visible coverage meshes.
    """
    kidxs = []
    for visible_coverage_node_msg in visible_coverage_graph_msg.visible_coverage_nodes:
        try:
            kidxs.append(kid2kidx[visible_coverage_node_msg.kid])
        except KeyError:
            continue
    # Compute visualization level
    n_points = 0
    for kidx in kidxs:
        n_points += keyframedb[kidx].pointcloud.shape[0]
    if n_points > 1000000:
        viz_level = 3
    elif n_points > 350000:
        viz_level = 2
    elif n_points > 150000:
        viz_level = 1
    else:
        viz_level = 0
    # construct pose graph solutions until lc_kidx, only visible coverage keyframes have values
    if len(kidxs) == 0:
        start_T_currs = np.zeros((0, 4, 4), dtype=np.float32)
    else:
        start_T_currs = np.zeros((kidxs[-1] + 1, 4, 4), dtype=np.float32)
    for visible_coverage_node_msg in visible_coverage_graph_msg.visible_coverage_nodes:
        try:
            kidx = kid2kidx[visible_coverage_node_msg.kid]
            start_T_curr = np.array(visible_coverage_node_msg.ref_t_curr).reshape(4, 4)
            start_T_currs[kidx] = start_T_curr
        except KeyError:
            continue
    pose_graph_solutions = PoseGraphSolutions(start_T_currs)
    # Create the meshes
    meshes = []
    for visible_coverage_node_msg in visible_coverage_graph_msg.visible_coverage_nodes:
        try:
            kidx = kid2kidx[visible_coverage_node_msg.kid]
            neighbor_kidxs = []
            for neighbor_kid in visible_coverage_node_msg.neighbor_kids:
                try:
                    neighbor_kidxs.append(kid2kidx[neighbor_kid])
                except KeyError:
                    continue
            adjusted_pointcloud = compute_adjusted_pointcloud(
                kidx,
                neighbor_kidxs,
                keyframedb,
                pose_graph_solutions,
                viz_level,
            )
            adjusted_mesh = pointcloud2mesh(
                adjusted_pointcloud, keyframedb[kidx].get_C(viz_level)
            )
            meshes.append(adjusted_mesh)
        except KeyError:
            pass
    # Create the visible coverage meshes
    if len(kidxs) == 0:
        visible_coverage_meshes = VisibleCoverageMeshes(meshes)
    else:
        visible_coverage_meshes = VisibleCoverageMeshes(
            meshes, kidxs[-1], pose_graph_solutions[kidxs[-1]], viz_level
        )
    return visible_coverage_meshes


class VisibleCoverageMeshes:
    """
    The VisibleCoverageGraphMsg is transformed into VisibleCoverageMeshes.
    It contains all coverage meshes in the same parent group as the loop closing keyframe.
    It also saves the kidx and the pose of the loop closing keyframe.
    """

    def __init__(
        self,
        visible_coverage_meshes=[],
        lc_kidx=None,
        start_T_lc=None,
        viz_level=0,
    ):
        self.visible_coverage_meshes = visible_coverage_meshes
        # The loop closing keyframe's kidx
        self.lc_kidx = lc_kidx
        self.start_T_lc = start_T_lc
        self.viz_level = viz_level

    def get_keyframe_pose(self, tar_kidx, keyframedb):
        """
        Get the pose of the keyframe.
        :param tar_kidx: int; the keyframe index.
        :param keyframedb: KeyFrameDB; the keyframe database.
        :return:
            - bool; if the keyframe is not in the same parent group with the visible coverage meshes.
            - np.ndarray; the pose of the keyframe reference to the visible coverage meshes.
        """
        if len(self.visible_coverage_meshes) == 0:
            return True, np.eye(4, dtype=np.float32)
        start_T_prev = self.start_T_lc
        for kidx in range(self.lc_kidx + 1, tar_kidx + 1):
            keyframe = keyframedb[kidx]
            if keyframe.is_new_trial():
                return True, np.eye(4, dtype=np.float32)
            else:
                start_T_prev = start_T_prev @ keyframe.ref_T_curr
        return False, start_T_prev
