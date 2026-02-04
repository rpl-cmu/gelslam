import pickle
import os
import cv2
import gtsam
import numpy as np
from normalflow.registration import normalflow, LoseTrackError
from gelslam.utils import (
    matrix_in_mm,
    matrix_in_m,
    M2T,
    Logger,
)

noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1]))


class PoseGraph:
    """
    The pose graph of the reconstruction.
    It handles the optimization of the poses.
    """

    def __init__(self, config, logger=None):
        self.pose_graph_config = config["gelslam_config"]["pose_graph_config"]
        self.logger = Logger(logger)
        self.graph = gtsam.NonlinearFactorGraph()
        self.graph_init = gtsam.Values()
        self.graph_result = None
        # The prior factor indexes, each reflects a trial
        self.prior_factor_idxs = []

    def add_odometry_factors(self, keyframedb, updated_size, targeted_size):
        """
        Add odometry factors and reset the initialization to the pose graph.
        """
        if self.graph_result is not None:
            self.graph_init = gtsam.Values(self.graph_result)
            start_T_ref = matrix_in_m(
                self.graph_result.atPose3(updated_size - 1).matrix()
            ).astype(np.float32)
        for kidx in range(updated_size, targeted_size):
            keyframe = keyframedb[kidx]
            if keyframe.is_new_trial():
                # New trial introduced
                prior_mean = gtsam.Pose3(np.eye(4))
                self.graph.add(gtsam.PriorFactorPose3(kidx, prior_mean, noise))
                self.prior_factor_idxs.append(self.graph.size() - 1)
                self.graph_init.insert(kidx, prior_mean)
                start_T_ref = np.eye(4, dtype=np.float32)
            else:
                # Stay on the old trial
                odometry = gtsam.Pose3(matrix_in_mm(keyframe.ref_T_curr))
                self.graph.add(
                    gtsam.BetweenFactorPose3(kidx - 1, kidx, odometry, noise)
                )
                # Pose graph initialization
                start_T_curr = start_T_ref @ keyframe.ref_T_curr
                self.graph_init.insert(kidx, gtsam.Pose3(matrix_in_mm(start_T_curr)))

    def detect_and_add_loops(self, keyframedb, tar_kidx, coverage_graph):
        """
        Detect loops between the target kidx to its previous keyframes.
        Add factors accordingly when loops are detected.
        """
        # Get the keyframe to detect loop on
        tar_keyframe = keyframedb[tar_kidx]
        tar_group = tar_keyframe.trial_group
        matched_kidxs = []
        for ref_kidx in range(tar_kidx):
            ref_keyframe = keyframedb[ref_kidx]
            ref_group = ref_keyframe.trial_group
            # Skip the nearby matches within the same group
            if ref_group == tar_group and ref_kidx == tar_kidx - 1:
                continue
            # Skip if either frames have not enough features
            if len(ref_keyframe.kp) == 0 or len(tar_keyframe.kp) == 0:
                continue
            # Skip if not active in coverage graph
            if ref_kidx < coverage_graph.size():
                if not coverage_graph[ref_kidx].is_active:
                    continue
            # SIFT Matching
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(ref_keyframe.des, tar_keyframe.des)
            M, inliers = cv2.estimateAffinePartial2D(
                np.float32([ref_keyframe.kp[m.queryIdx].pt for m in matches]).reshape(
                    -1, 1, 2
                ),
                np.float32([tar_keyframe.kp[m.trainIdx].pt for m in matches]).reshape(
                    -1, 1, 2
                ),
                maxIters=100,
                ransacReprojThreshold=3,
            )
            if inliers is None or M is None:
                continue
            lcsift_tar_T_ref = M2T(
                M,
                tar_keyframe.L.shape[1],
                tar_keyframe.L.shape[0],
                tar_keyframe.ppmm,
            )
            lcsift_score = np.sum(inliers)
            # Threshold for SIFT Matching
            if lcsift_score < self.pose_graph_config["min_sift_inliers"]:
                continue
            # NormalFlow Matching
            try:
                lcnf_tar_T_ref = normalflow(
                    ref_keyframe.N,
                    ref_keyframe.C,
                    ref_keyframe.H,
                    ref_keyframe.L,
                    tar_keyframe.N,
                    tar_keyframe.C,
                    tar_keyframe.H,
                    tar_keyframe.L,
                    lcsift_tar_T_ref,
                    tar_keyframe.ppmm,
                    scr_threshold=self.pose_graph_config["scr_threshold"],
                    ccs_threshold=self.pose_graph_config["ccs_threshold"],
                )
            except LoseTrackError:
                continue
            # Add loop closing factor
            matched_kidxs.append(ref_kidx)
            odometry = gtsam.Pose3(matrix_in_mm(np.linalg.inv(lcnf_tar_T_ref)))
            self.graph.add(
                gtsam.BetweenFactorPose3(ref_kidx, tar_kidx, odometry, noise)
            )

        return matched_kidxs

    def remove_prior_factors(self, removed_trial_groups):
        for trial_group in removed_trial_groups:
            self.graph.remove(self.prior_factor_idxs[trial_group])
            self.prior_factor_idxs[trial_group] = -1

    def solve(self):
        """
        Solve the pose graph optimization.
        """
        # Solve the pose graph
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.graph_init)
        self.graph_result = optimizer.optimize()
        # Construct the pose graph solutions
        solved_start_T_currs = []
        for kidx in range(self.graph_result.size()):
            start_T_curr = matrix_in_m(self.graph_result.atPose3(kidx).matrix()).astype(
                np.float32
            )
            solved_start_T_currs.append(start_T_curr)
        pose_graph_solutions = PoseGraphSolutions(solved_start_T_currs)
        return pose_graph_solutions

    def save(self, save_dir):
        # Turn prior_factor_idxs into prior_keys for saving due to g2o will rearrange factors
        self.prior_keys = []
        for prior_factor_idx in self.prior_factor_idxs:
            if prior_factor_idx != -1:
                prior_factor = self.graph.at(prior_factor_idx)
                self.prior_keys.append(prior_factor.keys())
            else:
                # Prior factor deleted, then use a dummy key
                self.prior_keys.append([-1])
        self.prior_factor_idxs = []
        # Write the pose graph
        gtsam.writeG2o(
            self.graph, self.graph_result, os.path.join(save_dir, "pose_graph.g2o")
        )
        # Remove the not-picklable states
        self.logger = None
        self.graph = None
        self.graph_init = None
        self.graph_result = None
        # Pickle the rest
        with open(os.path.join(save_dir, "pose_graph.pkl"), "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, load_dir, logger=None):
        # Load the pickled file
        with open(os.path.join(load_dir, "pose_graph.pkl"), "rb") as f:
            instance = pickle.load(f)
        # Construct the not-pickable states
        instance.logger = Logger(logger)
        # Load the graph itinstance
        instance.graph, instance.graph_result = gtsam.readG2o(
            os.path.join(load_dir, "pose_graph.g2o"), is3D=True
        )
        instance.graph_init = gtsam.Values(instance.graph_result)
        # Trun prior_keys into prior_factor_idxs
        instance.prior_factor_idxs = []
        for prior_keys in instance.prior_keys:
            if prior_keys[0] != -1:
                instance.graph.add(
                    gtsam.PriorFactorPose3(prior_keys[0], gtsam.Pose3(np.eye(4)), noise)
                )
                instance.prior_factor_idxs.append(instance.graph.size() - 1)
            else:
                # Dummy prior factor, use a dummy index
                instance.prior_factor_idxs.append(-1)
        return instance


class PoseGraphSolutions:
    """
    The data structure that holds the solved result of pose graph optimization.
    """

    def __init__(self, solved_start_T_currs=[]):
        self.solved_start_T_currs = solved_start_T_currs

    def __getitem__(self, idx):
        return self.solved_start_T_currs[idx]

    def save(self, save_dir):
        with open(os.path.join(save_dir, "pose_graph_solutions.pkl"), "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, load_dir):
        with open(os.path.join(load_dir, "pose_graph_solutions.pkl"), "rb") as f:
            instance = pickle.load(f)
        return instance

    def size(self):
        return len(self.solved_start_T_currs)
