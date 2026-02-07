import os
import pickle
import shutil

import cv2
import numpy as np
from cv_bridge import CvBridge
from normalflow.utils import height2pointcloud

from gelslam.utils import (
    get_backproj_weights,
    get_backproj_weights_and_pointcloud,
    get_sift_features,
    mask2weight,
)


def keyframe_msg2keyframe(bridge, keyframe_msg, trial_group, new_trial_flag, ppmm):
    """
    Converts a KeyFrameMsg to a KeyFrame object.

    :param bridge: CvBridge; To convert images.
    :param bridge: CvBridge; The cv bridge.
    :param keyframe_msg: KeyFrameMsg; The keyframe message.
    :param trial_group: int; The trial group of the keyframe.
    :param new_trial_flag: bool; The flag of new trial.
    :param ppmm: float; Pixels size in milimeters.
    :return: KeyFrame; The keyframe object.
    """
    # Convert the message images to numpy arrays
    H = bridge.imgmsg_to_cv2(keyframe_msg.height_map)
    C = bridge.imgmsg_to_cv2(keyframe_msg.contact_mask) > 0
    N = bridge.imgmsg_to_cv2(keyframe_msg.normal_map)
    L = bridge.imgmsg_to_cv2(keyframe_msg.laplacian_map)
    # Get the transformation to the reference frame
    ref_T_curr = np.array(keyframe_msg.ref_t_curr).reshape(4, 4)
    if new_trial_flag:
        ref_T_curr = np.eye(4, dtype=np.float32)
    # Construct the keyframe
    keyframe = KeyFrame(
        H,
        C,
        N,
        L,
        keyframe_msg.kid,
        keyframe_msg.fid,
        trial_group,
        new_trial_flag,
        ref_T_curr,
        ppmm,
    )
    return keyframe


def compute_overlap_weight(tar_kidx, ref_kidx, keyframedb, pose_graph_solutions):
    """
    Compute the overlapping weight between two keyframes by projected reference pointcloud to the
    target keyframe.

    :param tar_kidx: int; The target keyframe index.
    :param ref_kidx: int; The reference keyframe index.
    :param keyframedb: KeyFrameDB; The keyframe database.
    :param pose_graph_solutions: PoseGraphSolutions; Solved poses.
    :return: float; The overlap weight.
    """
    start_T_tar = pose_graph_solutions[tar_kidx]
    tar_keyframe = keyframedb[tar_kidx]
    start_T_ref = pose_graph_solutions[ref_kidx]
    ref_keyframe = keyframedb[ref_kidx]
    # If touch direction not aligned, not overlapped
    cos_angle = start_T_ref[:3, 2] @ start_T_tar[:3, 2]
    if cos_angle < 0:
        return 0.0
    # Estimate the overlap weight
    tar_T_ref = np.linalg.inv(start_T_tar) @ start_T_ref
    W_backproj = get_backproj_weights(
        tar_keyframe.W, ref_keyframe.pointcloud, tar_T_ref, tar_keyframe.ppmm
    )
    return np.sum(W_backproj)


def compute_revealed_weight(tar_kidx, neighbor_kidxs, keyframedb, pose_graph_solutions):
    """
    Computes the weight of the target keyframe that is NOT covered by its neighbors (revealed weight).

    :param tar_kidx: int; The target keyframe index.
    :param neighbor_kidxs: list of int; The indices of neighbor keyframes.
    :param keyframedb: KeyFrameDB; The keyframe database.
    :param pose_graph_solutions: PoseGraphSolutions; Solved poses.
    :return: float; The revealed weight.
    """
    tar_keyframe = keyframedb[tar_kidx]
    start_T_tar = pose_graph_solutions[tar_kidx]
    if len(neighbor_kidxs) == 0:
        return np.sum(tar_keyframe.W[tar_keyframe.C])
    W_tar_backprojs = []
    for neighbor_kidx in neighbor_kidxs:
        neighbor_keyframe = keyframedb[neighbor_kidx]
        start_T_neighbor = pose_graph_solutions[neighbor_kidx]
        neighbor_T_tar = np.linalg.inv(start_T_neighbor) @ start_T_tar
        W_tar_backprojs.append(
            get_backproj_weights(
                neighbor_keyframe.W,
                tar_keyframe.pointcloud,
                neighbor_T_tar,
                tar_keyframe.ppmm,
            )
        )
    W_tar_coverage = np.max(W_tar_backprojs, axis=0)
    revealed_weight = np.sum(
        tar_keyframe.W[tar_keyframe.C]
        - np.min([tar_keyframe.W[tar_keyframe.C], W_tar_coverage], axis=0)
    )
    return revealed_weight


def compute_adjusted_pointcloud(
    tar_kidx, neighbor_kidxs, keyframedb, pose_graph_solutions, viz_level=0
):
    """
    Computes the weighted average pointcloud for the target keyframe, incorporating data from neighbors.
    Supports downsampling via viz_level.

    :param tar_kidx: int; The target keyframe index.
    :param neighbor_kidxs: list of int; The indices of neighbor keyframes.
    :param keyframedb: KeyFrameDB; The keyframe database.
    :param pose_graph_solutions: PoseGraphSolutions; Solved poses.
    :param viz_level: int; The visualization level (0 for full resolution).
    :return: np.ndarray (N, 3); The adjusted pointcloud.
    """
    tar_keyframe = keyframedb[tar_kidx]
    start_T_tar = pose_graph_solutions[tar_kidx]
    # Initialize the weighted pointcloud
    weights_total = np.ones(
        (tar_keyframe.get_pointcloud(viz_level).shape[0], 1), dtype=np.float32
    ) * 1e-8 + tar_keyframe.get_W(viz_level)[tar_keyframe.get_C(viz_level)].reshape(
        -1, 1
    )
    weighted_pointcloud_total = tar_keyframe.get_pointcloud(viz_level) * weights_total
    # Update the weighted pointcloud based on neighbors
    for neighbor_kidx in neighbor_kidxs:
        neighbor_keyframe = keyframedb[neighbor_kidx]
        start_T_neighbor = pose_graph_solutions[neighbor_kidx]
        neighbor_T_tar = np.linalg.inv(start_T_neighbor) @ start_T_tar
        weights_backproj, pointcloud_backproj = get_backproj_weights_and_pointcloud(
            neighbor_keyframe.get_H(viz_level),
            neighbor_keyframe.get_W(viz_level),
            tar_keyframe.get_pointcloud(viz_level),
            neighbor_T_tar,
            tar_keyframe.get_ppmm(viz_level),
        )
        weights_total += weights_backproj
        weighted_pointcloud_total += weights_backproj * pointcloud_backproj
    # Transform the averaged pointcloud
    pointcloud = weighted_pointcloud_total / weights_total
    pointcloud = np.dot(start_T_tar[:3, :3], pointcloud.T).T + start_T_tar[:3, 3]
    return pointcloud


class KeyFrame:
    """
    Represents a keyframe containing geometry (H, C, N, L) and features.
    Includes logic for downsampling and visualization.
    """

    def __init__(
        self, H, C, N, L, kid, fid, trial_group, new_trial_flag, ref_T_curr, ppmm
    ):
        # Set the frame information
        self.H = H
        self.C = C
        self.N = N
        self.L = L
        self.W = mask2weight(self.C)
        self.pointcloud = height2pointcloud(self.H, self.C, ppmm)
        self.ppmm = ppmm
        # The keyframe's ID
        self.kid = kid
        # The keyframe's frame ID
        self.fid = fid
        # The trial group of the keyframe
        self.trial_group = trial_group
        # The flag of new trial
        self.new_trial_flag = new_trial_flag
        # The estimated trnasformation to the previous keyframe, eye if head of the new trial
        self.ref_T_curr = ref_T_curr
        # Visualization Only: Compute the downsampling of the geometry, with rate=2 and 4
        self.downsampled_Cs = [self.C[::2, ::2], self.C[::4, ::4], self.C[::8, ::8]]
        self.downsampled_Ws = [self.W[::2, ::2], self.W[::4, ::4], self.W[::8, ::8]]
        self.downsampled_Hs = [
            self.H[::2, ::2] / 2.0,
            self.H[::4, ::4] / 4.0,
            self.H[::8, ::8] / 8.0,
        ]
        self.downsampled_ppmms = [self.ppmm * 2, self.ppmm * 4, self.ppmm * 8]
        self.downsampled_pointclouds = [
            height2pointcloud(
                self.downsampled_Hs[0],
                self.downsampled_Cs[0],
                self.downsampled_ppmms[0],
            ),
            height2pointcloud(
                self.downsampled_Hs[1],
                self.downsampled_Cs[1],
                self.downsampled_ppmms[1],
            ),
            height2pointcloud(
                self.downsampled_Hs[2],
                self.downsampled_Cs[2],
                self.downsampled_ppmms[2],
            ),
        ]

        # Get the SIFT features
        self.kp, self.des = get_sift_features(
            self.L, self.C, cv2.SIFT_create(contrastThreshold=0.02, nOctaveLayers=3)
        )

    def compute_sift_features(self):
        self.kp, self.des = get_sift_features(
            self.L, self.C, cv2.SIFT_create(contrastThreshold=0.02, nOctaveLayers=3)
        )

    def save(self, save_path):
        self.kp = None
        self.des = None
        with open(save_path, "wb") as f:
            pickle.dump(self, f)

    def is_new_trial(self):
        return self.new_trial_flag

    def get_C(self, level=0):
        """Get the contact mask supporting downsampling."""
        if level == 0:
            return self.C
        elif level == 1 or level == 2 or level == 3:
            return self.downsampled_Cs[level - 1]
        else:
            raise ValueError("Invalid level")

    def get_W(self, level=0):
        """Get the weights supporting downsampling."""
        if level == 0:
            return self.W
        elif level == 1 or level == 2 or level == 3:
            return self.downsampled_Ws[level - 1]
        else:
            raise ValueError("Invalid level")

    def get_H(self, level=0):
        """Get the height map supporting downsampling."""
        if level == 0:
            return self.H
        elif level == 1 or level == 2 or level == 3:
            return self.downsampled_Hs[level - 1]
        else:
            raise ValueError("Invalid level")

    def get_pointcloud(self, level=0):
        """Get the pointcloud supporting downsampling."""
        if level == 0:
            return self.pointcloud
        elif level == 1 or level == 2 or level == 3:
            return self.downsampled_pointclouds[level - 1]
        else:
            raise ValueError("Invalid level")

    def get_ppmm(self, level=0):
        """Get the ppmm supporting downsampling."""
        if level == 0:
            return self.ppmm
        elif level == 1 or level == 2 or level == 3:
            return self.downsampled_ppmms[level - 1]
        else:
            raise ValueError("Invalid level")


class KeyFrameDB:
    """
    Manages a database of KeyFrames.
    Supports saving/loading and retrieving keyframes.
    Note: kidx is not the same as kid. kid is the unique identifier of a keyframe,
    while kidx is the index of a keyframe in the database. They can be different in multi-threaded scenarios.
    """

    def __init__(self, ppmm, logger):
        self.logger = logger
        self.bridge = CvBridge()
        self.ppmm = ppmm
        # The keyframe database variables
        self.curr_trial_group = -1
        self.keyframes = []

    def __getitem__(self, idx):
        return self.keyframes[idx]

    def save(self, save_dir):
        # Save the keyframes
        keyframe_save_dir = os.path.join(save_dir, "keyframedb")
        if os.path.exists(keyframe_save_dir):
            shutil.rmtree(keyframe_save_dir)
        os.makedirs(keyframe_save_dir)
        for kidx, keyframe in enumerate(self.keyframes):
            save_path = os.path.join(keyframe_save_dir, f"keyframe_{kidx:05d}.pkl")
            keyframe.save(save_path)
        self.keyframes = []
        # Remove the not-pickable states
        self.logger = None
        self.bridge = None
        # Pickle the rest
        with open(os.path.join(save_dir, "keyframedb.pkl"), "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, load_dir, logger):
        # Load the pickled file
        with open(os.path.join(load_dir, "keyframedb.pkl"), "rb") as f:
            instance = pickle.load(f)
        # Construct the not-pickable states
        instance.logger = logger
        instance.bridge = CvBridge()
        # Load keyframes
        instance.keyframes = []
        keyframe_load_dir = os.path.join(load_dir, "keyframedb")
        keyframe_filenames = os.listdir(keyframe_load_dir)
        keyframe_filenames.sort()
        for keyframe_filename in keyframe_filenames:
            with open(os.path.join(keyframe_load_dir, keyframe_filename), "rb") as f:
                keyframe = pickle.load(f)
                keyframe.compute_sift_features()
                instance.keyframes.append(keyframe)
        return instance

    def size(self):
        return len(self.keyframes)

    def get_kid2kidx(self):
        """Request the dictionary mapping kid to kidx."""
        kid2kidx = {}
        for kidx, keyframe in enumerate(self.keyframes):
            kid2kidx[keyframe.kid] = kidx
        return kid2kidx

    def find_kidx_from_kid(self, tar_kid, targeted_size):
        """
        Find the kidx of the keyframe with the given ID but within the targeted size.
        If not found, return None.

        :param tar_kid: int; The target keyframe ID.
        :param targeted_size: int; The size of the database to search within.
        :return: int or None; The keyframe index.
        """
        tar_kidx = None
        if targeted_size != 0:
            for kidx in range(targeted_size - 1, -1, -1):
                if self.keyframes[kidx].kid == tar_kid:
                    tar_kidx = kidx
                    break
                elif self.keyframes[kidx].kid < tar_kid:
                    break
        return tar_kidx

    def insert(self, keyframe_msg, log_prefix="Keyframe Insert Thread"):
        """
        Turn the keyframe message to keyframe and insert to the database.

        :param keyframe_msg: KeyFrameMsg; The keyframe message.
        :param log_prefix: str; The prefix for logging.
        """
        # Detect keyframe drop
        new_trial_flag = keyframe_msg.new_trial_flag
        if keyframe_msg.kid != 0:
            if len(self.keyframes) == 0:
                self.logger.warning("%s -- Keyframe Drop Detected" % log_prefix)
                new_trial_flag = True
            elif self.keyframes[-1].kid != keyframe_msg.kid - 1:
                self.logger.warning("%s -- Keyframe Drop Detected" % log_prefix)
                new_trial_flag = True
        # When New Trial Detected, update the trial group
        if new_trial_flag:
            self.curr_trial_group += 1
        # Get the keyframe
        keyframe = keyframe_msg2keyframe(
            self.bridge,
            keyframe_msg,
            trial_group=self.curr_trial_group,
            new_trial_flag=new_trial_flag,
            ppmm=self.ppmm,
        )
        self.logger.info(
            "%s -- keyframe ID = %d, trial group = %d"
            % (log_prefix, keyframe_msg.kid, self.curr_trial_group)
        )
        # Append the keyframe to the database
        self.keyframes.append(keyframe)
