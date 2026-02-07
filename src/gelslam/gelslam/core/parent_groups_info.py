import os
import pickle

import numpy as np


class ParentGroupsInfo:
    """
    Tracks and manages parent group assignments for keyframes. Same parent group means same trial.
    Used for detecting and merging groups during loop closure.
    """

    def __init__(self):
        self.parent_groups = []
        self.parent_group_sizes = []

    def save(self, save_dir):
        # Pickle the rest
        with open(os.path.join(save_dir, "parent_groups_info.pkl"), "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, load_dir):
        # Load the pickled file
        with open(os.path.join(load_dir, "parent_groups_info.pkl"), "rb") as f:
            instance = pickle.load(f)
        return instance

    def get_largest_parent_group(self):
        """
        Get the largest parent group.

        :return: int; The largest parent group.
        """
        return np.argmax(self.parent_group_sizes)

    def get_parent_group(self, kidx, keyframedb):
        """
        Get the parent group of a keyframe.

        :param kidx: int; The keyframe index.
        :param keyframedb: KeyFrameDB; The keyframe database.
        :return: int; The parent group.
        """
        return self.parent_groups[keyframedb[kidx].trial_group]

    def update_wrt_new_keyframes(self, keyframedb, updated_size, targeted_size):
        """
        Update the parent groups and sizes based on the newly introduced keyframes.
        Newly introduced keyframes are kidxs from updated_size to targeted_size.

        :param keyframedb: KeyFrameDB; The keyframe database.
        :param updated_size: int; The size of the database before update.
        :param targeted_size: int; The size of the database after update.
        """
        for kidx in range(updated_size, targeted_size):
            keyframe = keyframedb[kidx]
            if keyframe.is_new_trial():
                self.parent_groups.append(keyframe.trial_group)
                self.parent_group_sizes.append(1)
            else:
                parent_group = self.parent_groups[keyframe.trial_group]
                self.parent_group_sizes[parent_group] += 1

    def update_wrt_loop_closure(
        self, keyframedb, tar_kidx, matched_kidxs, updated_size
    ):
        """
        Updates parent groups and sizes based on detected loop closures.
        Merges groups if necessary.

        :param keyframedb: KeyFrameDB; The keyframe database.
        :param tar_kidx: int; The target keyframe index.
        :param matched_kidxs: list of int; The matched keyframe indices.
        :param updated_size: int; The original number of keyframes before loop closure.
        :return: Tuple containing:
            - original_member_kidxs: Keyframes indices already in the merged group.
            - new_member_kidxs: Keyframes indices newly added to the merged group.
            - other_kidxs: Keyframes indices belonging to other groups.
            - removed_trial_groups: List of trial groups that were removed (merged).
        """
        # Check what parent groups to be merged
        tar_parent_group = self.get_parent_group(tar_kidx, keyframedb)
        merging_parent_groups = [tar_parent_group]
        for ref_kidx in matched_kidxs:
            ref_parent_group = self.get_parent_group(ref_kidx, keyframedb)
            if ref_parent_group != tar_parent_group:
                if ref_parent_group not in merging_parent_groups:
                    merging_parent_groups.append(ref_parent_group)

        # Merge the parent group and returns how the indices are merged
        # The keyframes that are members of the original parent group
        original_member_kidxs = []
        # The newly added keyframes of the parent group
        new_member_kidxs = []
        # Newly added keyframes that belongs to other parent group
        other_kidxs = []
        # Parent groups that were removed (merged)
        removed_trial_groups = []
        if len(merging_parent_groups) > 1:
            # Identify new parent group
            merging_parent_group_sizes = []
            for parent_group in merging_parent_groups:
                merging_parent_group_sizes.append(self.parent_group_sizes[parent_group])
            new_parent_group = merging_parent_groups[
                np.argmax(merging_parent_group_sizes)
            ]
            # Identify how parent groups are updated
            for kidx in range(updated_size):
                parent_group = self.get_parent_group(kidx, keyframedb)
                if parent_group == new_parent_group:
                    original_member_kidxs.append(kidx)
                elif parent_group in merging_parent_groups:
                    new_member_kidxs.append(kidx)
            for kidx in range(updated_size, tar_kidx + 1):
                parent_group = self.get_parent_group(kidx, keyframedb)
                if parent_group in merging_parent_groups:
                    new_member_kidxs.append(kidx)
                else:
                    other_kidxs.append(kidx)
            # Set new parent groups to all groups to be merged
            for i in range(len(self.parent_groups)):
                if self.parent_groups[i] in merging_parent_groups:
                    self.parent_groups[i] = new_parent_group
            # Update the parent group sizes and remove prior factors
            self.parent_group_sizes[new_parent_group] = np.sum(
                merging_parent_group_sizes
            )
            for parent_group in merging_parent_groups:
                if parent_group != new_parent_group:
                    self.parent_group_sizes[parent_group] = 0
                    removed_trial_groups.append(parent_group)
        else:
            # Identify how parent groups are updated
            for kidx in range(updated_size):
                parent_group = self.get_parent_group(kidx, keyframedb)
                if parent_group == tar_parent_group:
                    original_member_kidxs.append(kidx)
            for kidx in range(updated_size, tar_kidx + 1):
                parent_group = self.get_parent_group(kidx, keyframedb)
                if parent_group == tar_parent_group:
                    new_member_kidxs.append(kidx)
                else:
                    other_kidxs.append(kidx)
        return (
            original_member_kidxs,
            new_member_kidxs,
            other_kidxs,
            removed_trial_groups,
        )
