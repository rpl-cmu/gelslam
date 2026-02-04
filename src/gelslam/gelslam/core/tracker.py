import os
import pickle
import json
import shutil
import cv2
from cv_bridge import CvBridge
import numpy as np
from normalflow.registration import normalflow, LoseTrackError
from normalflow.utils import Frame
from gs_sdk.gs_reconstruct import Reconstructor
from gelslam.utils import Logger
from gelslam.core.frame import create_keyframe_msg, create_frame_msg
from gelslam.visualization.viz_utils import (
    plot_laplacian_comparison,
    label_text_and_pad,
)

# TODO: In tracker, in pose estimation mode, we remove threshold if tracking wrt the previous frame.
# TODO: If lose track (previous frame is in track), set the previous pose as prior
# TODO: There is a bug in the tracking, check out pose_estimation/utils


class Tracker:
    """
    The tracker class handles the tracking of the frames.
    """

    def __init__(self, calib_model_path, config, logger=None):
        self.logger = Logger(logger)
        self.ppmm = config["device_config"]["ppmm"]
        self.contact_mask_config = config["contact_mask_config"]
        self.track_config = config["gelslam_config"]["track_config"]
        self.bridge = CvBridge()
        # Construct the reconstructor
        self.recon = Reconstructor(calib_model_path)

        # Save the first few images as background images
        self.bg_images = []
        # Tracking states
        self.prev_in_contact_flag = False
        self.prev_is_keyframe_flag = False
        self.ref_T_prev = np.eye(4, dtype=np.float32)
        self.prev_frame = None
        self.curr_kid = -1
        self.ref_frame = None
        self.curr_fid = -1

    def save(self, save_dir):
        # Remove the not-pickable states
        self.logger = None
        self.bridge = None
        self.recon = None
        # Pickle the rest
        with open(os.path.join(save_dir, "tracker.pkl"), "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, load_dir, calib_model_path, logger=None):
        # Load the pickled file
        with open(os.path.join(load_dir, "tracker.pkl"), "rb") as f:
            instance = pickle.load(f)
        # Construct the not-pickable states
        instance.logger = Logger(logger)
        instance.bridge = CvBridge()
        instance.recon = Reconstructor(calib_model_path)
        instance.recon.load_bg(instance.bg_image)
        return instance

    def track(self, image, log_prefix="Track"):
        """
        Track the tactile image. It returns a flag and the actual return values.
        If still collecting background images or not in contact,
        the flag is False and return value is None.
        :return:
            ret: bool; the validity of the return result.
            (frame_msg, keyframe_msgs):
                frame_msg: FrameMsg; the current frame information.
                keyframe_msgs: list of KeyFrameMsg; the list of new keyframes.
        """
        if len(self.bg_images) < 10:
            # Collect the background images
            self.bg_images.append(image)
            if len(self.bg_images) == 10:
                self.logger.info("%s -- Background images collected" % log_prefix)
                self.bg_image = np.mean(self.bg_images, axis=0)
                self.recon.load_bg(self.bg_image)
            return False, None
        else:
            # Obtain the frame to be tracked
            G, H, C = self.recon.get_surface_info(
                image,
                self.ppmm,
                color_dist_threshold=self.contact_mask_config["color_dist_threshold"],
                height_threshold=self.contact_mask_config["height_threshold"],
            )
            frame = Frame(
                G,
                H,
                C,
                contact_threshold=self.track_config["contact_threshold"],
            )
            # Skip if not in contact
            if not frame.is_contacted:
                self.logger.info("%s -- Skip Frame: not in contact" % log_prefix)
                self.prev_in_contact_flag = False
                return False, None
            else:
                self.curr_fid += 1
                keyframe_msgs = []
                if not self.prev_in_contact_flag:
                    self.logger.info(
                        "%s -- New Trial: No previous contact" % log_prefix
                    )
                    # If the object is newly contacted, create new trial
                    self.curr_kid += 1
                    keyframe_msg = create_keyframe_msg(
                        self.bridge,
                        frame,
                        kid=self.curr_kid,
                        fid=self.curr_fid,
                        is_new_trial=True,
                        ref_T_curr=np.eye(4, dtype=np.float32),
                    )
                    keyframe_msgs.append(keyframe_msg)
                    # Set the tracker information
                    self.prev_in_contact_flag = True
                    self.prev_is_keyframe_flag = True
                    self.ref_T_prev = np.eye(4, dtype=np.float32)
                    self.prev_frame = frame
                    self.ref_frame = frame
                else:
                    try:
                        curr_T_ref = normalflow(
                            self.ref_frame.N,
                            self.ref_frame.C,
                            self.ref_frame.H,
                            self.ref_frame.L,
                            frame.N,
                            frame.C,
                            frame.H,
                            frame.L,
                            np.linalg.inv(self.ref_T_prev),
                            self.ppmm,
                            scr_threshold=self.track_config["scr_threshold"],
                            ccs_threshold=self.track_config["ccs_threshold"],
                        )
                        # The current frame is not a keyframe
                        self.prev_in_contact_flag = True
                        self.prev_is_keyframe_flag = False
                        self.ref_T_prev = np.linalg.inv(curr_T_ref)
                        self.prev_frame = frame
                        self.logger.info("%s -- Regular Frame" % log_prefix)

                    except LoseTrackError:
                        if self.prev_is_keyframe_flag:
                            self.logger.info(
                                "%s -- New Trial: Failed to track" % log_prefix
                            )
                            # Set the current frame as keyframe and new trial
                            self.curr_kid += 1
                            keyframe_msg = create_keyframe_msg(
                                self.bridge,
                                frame,
                                kid=self.curr_kid,
                                fid=self.curr_fid,
                                is_new_trial=True,
                                ref_T_curr=np.eye(4, dtype=np.float32),
                            )
                            keyframe_msgs.append(keyframe_msg)
                            # Set the tracker information
                            self.prev_in_contact_flag = True
                            self.prev_is_keyframe_flag = True
                            self.ref_T_prev = np.eye(4, dtype=np.float32)
                            self.prev_frame = frame
                            self.ref_frame = frame
                        else:
                            self.logger.info(
                                "%s -- New Keyframe: Set previous frame as keyframe"
                                % log_prefix
                            )
                            # Set the previous frame as keyframe
                            self.curr_kid += 1
                            keyframe_msg = create_keyframe_msg(
                                self.bridge,
                                self.prev_frame,
                                kid=self.curr_kid,
                                fid=self.curr_fid - 1,
                                is_new_trial=False,
                                ref_T_curr=self.ref_T_prev,
                            )
                            keyframe_msgs.append(keyframe_msg)
                            # Set the tracker information
                            self.prev_in_contact_flag = True
                            self.prev_is_keyframe_flag = True
                            self.ref_T_prev = np.eye(4, dtype=np.float32)
                            self.ref_frame = self.prev_frame
                            # Check if the current frame is a keyframe reletive to the new keyframe
                            try:
                                curr_T_ref = normalflow(
                                    self.ref_frame.N,
                                    self.ref_frame.C,
                                    self.ref_frame.H,
                                    self.ref_frame.L,
                                    frame.N,
                                    frame.C,
                                    frame.H,
                                    frame.L,
                                    np.linalg.inv(self.ref_T_prev),
                                    self.ppmm,
                                    scr_threshold=self.track_config["scr_threshold"],
                                    ccs_threshold=self.track_config["ccs_threshold"],
                                )
                                # The current frame is not a keyframe
                                self.prev_in_contact_flag = True
                                self.prev_is_keyframe_flag = False
                                self.ref_T_prev = np.linalg.inv(curr_T_ref)
                                self.prev_frame = frame
                                self.logger.info("%s -- Regular Frame" % log_prefix)

                            except LoseTrackError:
                                # NormalFlow failed for two consecutive frames
                                self.logger.info(
                                    "%s -- New Trial: Failed to track" % log_prefix
                                )
                                # Set the current frame as keyframe and new trial
                                self.curr_kid += 1
                                keyframe_msg = create_keyframe_msg(
                                    self.bridge,
                                    frame,
                                    kid=self.curr_kid,
                                    fid=self.curr_fid,
                                    is_new_trial=True,
                                    ref_T_curr=np.eye(4, dtype=np.float32),
                                )
                                keyframe_msgs.append(keyframe_msg)
                                # Set the tracker information
                                self.prev_in_contact_flag = True
                                self.prev_is_keyframe_flag = True
                                self.ref_T_prev = np.eye(4, dtype=np.float32)
                                self.prev_frame = frame
                                self.ref_frame = frame

                # Construct the current frame message
                frame_msg = create_frame_msg(
                    self.bridge,
                    frame,
                    fid=self.curr_fid,
                    ref_kid=self.curr_kid,
                    ref_T_curr=self.ref_T_prev,
                )

                return True, (frame_msg, keyframe_msgs)
