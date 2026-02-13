import os
import pickle

import cv2
import numpy as np
from cv_bridge import CvBridge
from gs_sdk.gs_reconstruct import Reconstructor
from normalflow.registration import LoseTrackError, normalflow
from normalflow.utils import Frame

from gelslam.core.frame import create_frame_msg, create_keyframe_msg


class Tracker:
    """
    Handles frame tracking.
    """

    def __init__(self, calib_model_path, config, skip_background_check, logger):
        """
        Initialize the Tracker.

        :param calib_model_path: str; The path to the calibration model.
        :param config: dict; The configuration.
        :param skip_background_check: bool; Whether to skip the background check step.
        :param logger: Logger; The logger object.
        """
        self.logger = logger
        self.ppmm = config["device_config"]["ppmm"]
        self.contact_mask_config = config["contact_mask_config"]
        self.track_config = config["gelslam_config"]["track_config"]
        self.bridge = CvBridge()
        # Initialize reconstructor
        self.recon = Reconstructor(calib_model_path)
        # Skip background check flag
        self.skip_background_check = skip_background_check

        # Store initial images as background
        self.bg_images = []
        # Tracking state variables
        self.prev_in_contact_flag = False
        self.prev_is_keyframe_flag = False
        self.ref_T_prev = np.eye(4, dtype=np.float32)
        self.prev_frame = None
        self.curr_kid = -1
        self.ref_frame = None
        self.curr_fid = -1

    def save(self, save_dir):
        """
        Save the tracker state to a directory.

        :param save_dir: str; The directory to save to.
        """
        # Exclude non-picklable objects
        self.logger = None
        self.bridge = None
        self.recon = None
        # Pickle remaining state
        with open(os.path.join(save_dir, "tracker.pkl"), "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, load_dir, calib_model_path, logger):
        """
        Load the tracker state from a directory.

        :param load_dir: str; The directory to load from.
        :param calib_model_path: str; The path to the calibration model.
        :param logger: Logger; The logger object.
        :return: Tracker; The loaded tracker.
        """
        # Load pickled state
        with open(os.path.join(load_dir, "tracker.pkl"), "rb") as f:
            instance = pickle.load(f)
        # Reconstruct non-picklable objects
        instance.logger = logger
        instance.bridge = CvBridge()
        instance.recon = Reconstructor(calib_model_path)
        instance.recon.load_bg(instance.bg_image)
        return instance

    def track(self, image, log_prefix="Track"):
        """
        Tracks the tactile image.

        :param image: np.ndarray; The tactile image.
        :param log_prefix: str; The prefix for logging.
        :return: (bool, tuple or None);
            - bool: Validity of the result. Returns False if collecting background or not in contact.
            - tuple or None: (frame_msg, keyframe_msgs) if valid, else None.
                - frame_msg (FrameMsg): Current frame information.
                - keyframe_msgs (list[KeyFrameMsg]): List of new keyframes.
        """
        if len(self.bg_images) < 10:
            # Collect background images
            self.bg_images.append(image)
            if len(self.bg_images) == 10:
                self.logger.info("%s -- Background images collected" % log_prefix)

                self.bg_image = np.mean(self.bg_images, axis=0)
                # Background check by user
                if not self.skip_background_check:
                    display_image = self.bg_image.copy().astype(np.uint8)
                    display_image = cv2.resize(
                        display_image,
                        (display_image.shape[1] * 3, display_image.shape[0] * 3),
                    )
                    cv2.putText(
                        display_image,
                        "Ensure no contact. Press 'y' to confirm, 'n' to reject.",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("Background image verification", display_image)
                    while True:
                        key = cv2.waitKey(0) & 0xFF
                        if key == ord("n"):
                            raise SystemExit(
                                "User aborted: contact detected in background image."
                            )
                        elif key == ord("y"):
                            break
                    cv2.destroyWindow("Background image verification")
                # Load the confirmed background
                self.recon.load_bg(self.bg_image)
            return False, None
        else:
            # Get surface information
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
                self.logger.info("%s -- Skip Frame: Not in contact" % log_prefix)
                self.prev_in_contact_flag = False
                return False, None
            else:
                self.curr_fid += 1
                keyframe_msgs = []
                if not self.prev_in_contact_flag:
                    self.logger.info(
                        "%s -- New Trial: New contact initiated, keyframe ID: %d"
                        % (log_prefix, self.curr_kid + 1)
                    )
                    # New contact detected, start a new trial
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
                    # Update tracker state
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
                        # Current frame is tracked successfully (not a keyframe)
                        self.prev_in_contact_flag = True
                        self.prev_is_keyframe_flag = False
                        self.ref_T_prev = np.linalg.inv(curr_T_ref)
                        self.prev_frame = frame
                        self.logger.info(
                            "%s -- Regular Frame: frame ID: %d"
                            % (log_prefix, self.curr_fid)
                        )

                    except LoseTrackError:
                        if self.prev_is_keyframe_flag:
                            self.logger.info(
                                "%s -- New Trial: Failed to track, keyframe ID: %d"
                                % (log_prefix, self.curr_kid + 1)
                            )
                            # Tracking failed. Set current frame as keyframe and start a new trial.
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
                            # Update tracker state
                            self.prev_in_contact_flag = True
                            self.prev_is_keyframe_flag = True
                            self.ref_T_prev = np.eye(4, dtype=np.float32)
                            self.prev_frame = frame
                            self.ref_frame = frame
                        else:
                            self.logger.info(
                                "%s -- New Keyframe, keyframe ID: %d"
                                % (log_prefix, self.curr_kid + 1)
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
                            # Update tracker state
                            self.prev_in_contact_flag = True
                            self.prev_is_keyframe_flag = True
                            self.ref_T_prev = np.eye(4, dtype=np.float32)
                            self.ref_frame = self.prev_frame
                            # Check if the current frame can be tracked relative to the new keyframe
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
                                # Current frame is tracked successfully (not a keyframe)
                                self.prev_in_contact_flag = True
                                self.prev_is_keyframe_flag = False
                                self.ref_T_prev = np.linalg.inv(curr_T_ref)
                                self.prev_frame = frame
                                self.logger.info(
                                    "%s -- Regular Frame: frame ID: %d"
                                    % (log_prefix, self.curr_fid)
                                )

                            except LoseTrackError:
                                # Tracking failed. Set current frame as a keyframe and start a new trial.
                                self.logger.info(
                                    "%s -- New Trial: Failed to track, keyframe ID: %d"
                                    % (log_prefix, self.curr_kid + 1)
                                )
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
                                # Update tracker state
                                self.prev_in_contact_flag = True
                                self.prev_is_keyframe_flag = True
                                self.ref_T_prev = np.eye(4, dtype=np.float32)
                                self.prev_frame = frame
                                self.ref_frame = frame

                # Create current frame message
                frame_msg = create_frame_msg(
                    self.bridge,
                    frame,
                    fid=self.curr_fid,
                    ref_kid=self.curr_kid,
                    ref_T_curr=self.ref_T_prev,
                )

                return True, (frame_msg, keyframe_msgs)
