import os
import pickle
from gelslam.utils import Logger
from gelslam.core.keyframe import (
    compute_overlap_weight,
    compute_revealed_weight,
)
from gelslam_msgs.msg import VisibleCoverageNodeMsg, VisibleCoverageGraphMsg


def create_visible_coverage_graph_msg(
    tar_kidx, keyframedb, pose_graph_solutions, parent_groups_info, coverage_graph
):
    """
    Create the visible coverage graph message.
    :param tar_kidx; the target keyframe index that is visible. All keyframes in the same parent group that is active is included.
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
    def __init__(self):
        self.is_active = True
        self.neighbor_kidxs = []

    def add_neighbor_kidx(self, neighbor_kidx):
        self.neighbor_kidxs.append(neighbor_kidx)

    def remove_neighbor_kidx(self, neighbor_kidx):
        self.neighbor_kidxs.remove(neighbor_kidx)

    def set_not_active(self):
        self.clear_neighbor_kidxs()
        self.is_active = False

    def clear_neighbor_kidxs(self):
        self.neighbor_kidxs = []


class CoverageGraph:
    def __init__(self, config, logger=None):
        self.overlap_weight_threshold = config["gelslam_config"]["pose_graph_config"][
            "coverage_graph_overlap_threshold"
        ]
        self.logger = Logger(logger)
        self.coverage_graph = []

    def __getitem__(self, kidx):
        return self.coverage_graph[kidx]

    def save(self, save_dir):
        # Remove the not-pickable states
        self.logger = None
        # Pickle the rest
        with open(os.path.join(save_dir, "coverage_graph.pkl"), "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, load_dir, logger=None):
        # Load the pickled file
        with open(os.path.join(load_dir, "coverage_graph.pkl"), "rb") as f:
            instance = pickle.load(f)
        # Construct the not-pickable states
        instance.logger = Logger(logger)
        return instance

    def size(self):
        return len(self.coverage_graph)

    def add_new_coverage_nodes(self, updated_size, targeted_size):
        """Add new coverage nodes to the coverage graph."""
        for kidx in range(updated_size, targeted_size):
            self.coverage_graph.append(CoverageNode())

    def update_wrt_new_keyframes(
        self,
        tar_kidxs,
        keyframedb,
        pose_graph_solutions,
        parent_groups_info,
        log_prefix="Loop Closure",
    ):
        """
        This happens after loop closure detected.
        It updates the coverage graph based on the newly introduced keyframes not in the merged group.
        """
        self.logger.info(
            "%s -- Update coverage graph wrt new keyframes: %s"
            % (log_prefix, tar_kidxs)
        )
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
        log_prefix="Loop Closure",
    ):
        """
        This happens after loop closure detected.
        It updates the coverage graph by merging the keyframes that are new members of the merged group
        and the keyframes that are originally already in the merged group.
        """
        self.logger.info(
            "%s -- Update coverage graph wrt loop closure: new_members = %s"
            % (log_prefix, new_member_kidxs)
        )
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

    def log_coverage_graph(self, log_prefix="Loop Closure"):
        for kidx, coverage_node in enumerate(self.coverage_graph):
            if coverage_node.is_active:
                self.logger.info(
                    f"{log_prefix} -- Coverage Graph Node {kidx}: {coverage_node.neighbor_kidxs}"
                )
            else:
                self.logger.info(
                    f"{log_prefix} -- Coverage Graph Node {kidx}: Not active"
                )
