from gelslam.core.keyframe import compute_overlap_weight, compute_revealed_weight
from gelslam_msgs.msg import VisibleCoverageGraphMsg, VisibleCoverageNodeMsg


def create_visible_coverage_graph_msg(
    tar_kidx, keyframedb, pose_graph_solutions, parent_groups_info, coverage_graph
):
    """
    Creates a VisibleCoverageGraphMsg from the current state.

    :param tar_kidx: int; The target keyframe index (currently visible).
    :param keyframedb: KeyFrameDB; The keyframe database.
    :param pose_graph_solutions: PoseGraphSolutions; Solved poses.
    :param parent_groups_info: ParentGroupsInfo; Parent group information.
    :param coverage_graph: CoverageGraph; The coverage graph.
    :return: VisibleCoverageGraphMsg; The visible coverage graph message.
    """
    visible_coverage_graph_msg = VisibleCoverageGraphMsg()
    tar_parent_group = parent_groups_info.get_parent_group(tar_kidx, keyframedb)
    for kidx in range(tar_kidx + 1):
        parent_group = parent_groups_info.get_parent_group(kidx, keyframedb)
        if parent_group == tar_parent_group and coverage_graph[kidx].is_active:
            visible_coverage_node_msg = VisibleCoverageNodeMsg()
            visible_coverage_node_msg.kid = keyframedb[kidx].kid
            for neighbor_kidx in coverage_graph[kidx].neighbor_kidxs:
                visible_coverage_node_msg.neighbor_kids.append(
                    keyframedb[neighbor_kidx].kid
                )
            visible_coverage_node_msg.ref_t_curr = (
                pose_graph_solutions[kidx].flatten().tolist()
            )
            visible_coverage_graph_msg.visible_coverage_nodes.append(
                visible_coverage_node_msg
            )
    return visible_coverage_graph_msg


class CoverageNode:
    """
    Represents a node in the coverage graph.
    """

    def __init__(self):
        self.is_active = True
        self.neighbor_kidxs = []

    def add_neighbor_kidx(self, neighbor_kidx):
        """
        Add a neighbor keyframe index.

        :param neighbor_kidx: int; The neighbor keyframe index.
        """
        self.neighbor_kidxs.append(neighbor_kidx)

    def remove_neighbor_kidx(self, neighbor_kidx):
        """
        Remove a neighbor keyframe index.

        :param neighbor_kidx: int; The neighbor keyframe index.
        """
        self.neighbor_kidxs.remove(neighbor_kidx)

    def set_not_active(self):
        """
        Set the node as not active.
        """
        self.clear_neighbor_kidxs()
        self.is_active = False

    def clear_neighbor_kidxs(self):
        """
        Clear all neighbor keyframe indices.
        """
        self.neighbor_kidxs = []


class CoverageGraph:
    """
    Handles the coverage graph.
    """

    def __init__(self, config):
        """
        Initialize the CoverageGraph.

        :param config: dict; The configuration.
        """
        self.overlap_weight_threshold = config["gelslam_config"]["pose_graph_config"][
            "coverage_graph_overlap_threshold"
        ]
        self.coverage_graph = []

    def __getitem__(self, kidx):
        """
        Get the coverage node by keyframe index.

        :param kidx: int; The keyframe index.
        :return: CoverageNode; The coverage node.
        """
        return self.coverage_graph[kidx]

    def size(self):
        """
        Get the size of the coverage graph.

        :return: int; The size of the coverage graph.
        """
        return len(self.coverage_graph)

    def add_new_coverage_nodes(self, updated_size, targeted_size):
        """
        Adds new nodes to the coverage graph for newly inserted keyframes.

        :param updated_size: int; The size of the KeyframeDB before update.
        :param targeted_size: int; The size of the KeyframeDB after update.
        """
        for kidx in range(updated_size, targeted_size):
            self.coverage_graph.append(CoverageNode())

    def update_wrt_new_keyframes(
        self,
        tar_kidxs,
        keyframedb,
        pose_graph_solutions,
        parent_groups_info,
    ):
        """
        Updates the coverage graph based on newly introduced keyframes (e.g., after loop closure).
        Checks for overlap with existing nodes in the same parent group and prunes redundant nodes.

        :param tar_kidxs: list of int; The target keyframe indices.
        :param keyframedb: KeyFrameDB; The keyframe database.
        :param pose_graph_solutions: PoseGraphSolutions; Solved poses.
        :param parent_groups_info: ParentGroupsInfo; Parent group information.
        """
        # Update the coverage graph by combining with all nodes with same parent group and active
        for tar_kidx in tar_kidxs:
            tar_parent_group = parent_groups_info.get_parent_group(tar_kidx, keyframedb)
            # Find neighboring nodes
            for ref_kidx in range(tar_kidx):
                # Skip nodes with different parent group or not active
                ref_parent_group = parent_groups_info.get_parent_group(
                    ref_kidx, keyframedb
                )
                if ref_parent_group != tar_parent_group:
                    continue
                if not self.coverage_graph[ref_kidx].is_active:
                    continue
                # Check for overlap
                overlap_weight = compute_overlap_weight(
                    tar_kidx,
                    ref_kidx,
                    keyframedb,
                    pose_graph_solutions,
                )
                if overlap_weight > self.overlap_weight_threshold:
                    self.coverage_graph[ref_kidx].add_neighbor_kidx(tar_kidx)
                    self.coverage_graph[tar_kidx].add_neighbor_kidx(ref_kidx)
            # Prune if not revealed enough weight
            revealed_weight = compute_revealed_weight(
                tar_kidx,
                self.coverage_graph[tar_kidx].neighbor_kidxs,
                keyframedb,
                pose_graph_solutions,
            )
            if revealed_weight < self.overlap_weight_threshold:
                for neighbor_kidx in self.coverage_graph[tar_kidx].neighbor_kidxs:
                    self.coverage_graph[neighbor_kidx].remove_neighbor_kidx(tar_kidx)
                self.coverage_graph[tar_kidx].set_not_active()
                continue
            # Prune the coverage graph
            for neighbor_kidx in self.coverage_graph[tar_kidx].neighbor_kidxs:
                revealed_weight = compute_revealed_weight(
                    neighbor_kidx,
                    self.coverage_graph[neighbor_kidx].neighbor_kidxs,
                    keyframedb,
                    pose_graph_solutions,
                )
                if revealed_weight < self.overlap_weight_threshold:
                    for neighbor_kidx_ in self.coverage_graph[
                        neighbor_kidx
                    ].neighbor_kidxs:
                        self.coverage_graph[neighbor_kidx_].remove_neighbor_kidx(
                            neighbor_kidx
                        )
                    self.coverage_graph[neighbor_kidx].set_not_active()

    def update_wrt_loop_closure(
        self,
        original_member_kidxs,
        new_member_kidxs,
        keyframedb,
        pose_graph_solutions,
    ):
        """
        Updates the coverage graph after a loop closure event.
        Merges new members into the coverage graph and prunes redundant nodes.

        :param original_member_kidxs: list of int; The original member keyframe indices.
        :param new_member_kidxs: list of int; The new member keyframe indices.
        :param keyframedb: KeyFrameDB; The keyframe database.
        :param pose_graph_solutions: PoseGraphSolutions; Solved poses.
        """
        for new_kidx in new_member_kidxs:
            new_keyframe = keyframedb[new_kidx]
            # Skip if the node is not active
            if not self.coverage_graph[new_kidx].is_active:
                continue
            # Clear the neighbors
            self.coverage_graph[new_kidx].clear_neighbor_kidxs()
            # Check for keyframes with enough overlap
            for member_kidx in original_member_kidxs:
                # Skip if the node is not active
                if not self.coverage_graph[member_kidx].is_active:
                    continue
                # Check for overlap
                overlap_weight = compute_overlap_weight(
                    new_kidx,
                    member_kidx,
                    keyframedb,
                    pose_graph_solutions,
                )
                if overlap_weight > self.overlap_weight_threshold:
                    self.coverage_graph[member_kidx].add_neighbor_kidx(new_kidx)
                    self.coverage_graph[new_kidx].add_neighbor_kidx(member_kidx)
            # Prune if not revealed enough weight
            revealed_weight = compute_revealed_weight(
                new_kidx,
                self.coverage_graph[new_kidx].neighbor_kidxs,
                keyframedb,
                pose_graph_solutions,
            )
            if revealed_weight < self.overlap_weight_threshold:
                for neighbor_kidx in self.coverage_graph[new_kidx].neighbor_kidxs:
                    self.coverage_graph[neighbor_kidx].remove_neighbor_kidx(new_kidx)
                self.coverage_graph[new_kidx].set_not_active()
                continue
            # Prune the coverage graph
            for neighbor_kidx in self.coverage_graph[new_kidx].neighbor_kidxs:
                revealed_weight = compute_revealed_weight(
                    neighbor_kidx,
                    self.coverage_graph[neighbor_kidx].neighbor_kidxs,
                    keyframedb,
                    pose_graph_solutions,
                )
                if revealed_weight < self.overlap_weight_threshold:
                    for neighbor_kidx_ in self.coverage_graph[
                        neighbor_kidx
                    ].neighbor_kidxs:
                        self.coverage_graph[neighbor_kidx_].remove_neighbor_kidx(
                            neighbor_kidx
                        )
                    self.coverage_graph[neighbor_kidx].set_not_active()
