import numpy as np
from normalflow.utils import height2pointcloud

from gelslam.utils import pointcloud2mesh
from gelslam_msgs.msg import FrameMsg, KeyFrameMsg


def create_keyframe_msg(bridge, frame, kid, fid, is_new_trial, ref_T_curr):
    """
    Creates a KeyFrameMsg from a Frame object.

    :param bridge: CvBridge; To convert images.
    :param frame: Frame; Object containing data.
    :param kid: int; Keyframe ID.
    :param fid: int; Frame ID.
    :param is_new_trial: bool; Flag if this is a new trial.
    :param ref_T_curr: np.ndarray; Transformation from current to reference.
    :return: KeyFrameMsg; The keyframe message.
    """
    keyframe_msg = KeyFrameMsg()
    keyframe_msg.height_map = bridge.cv2_to_imgmsg(frame.H, encoding="32FC1")
    keyframe_msg.contact_mask = bridge.cv2_to_imgmsg(
        frame.C.astype(np.uint8), encoding="8UC1"
    )
    keyframe_msg.normal_map = bridge.cv2_to_imgmsg(frame.N, encoding="32FC3")
    keyframe_msg.laplacian_map = bridge.cv2_to_imgmsg(frame.L, encoding="32FC1")
    keyframe_msg.kid = kid
    keyframe_msg.fid = fid
    keyframe_msg.new_trial_flag = is_new_trial
    keyframe_msg.ref_t_curr = ref_T_curr.flatten().tolist()
    return keyframe_msg


def create_frame_msg(bridge, frame, fid, ref_kid, ref_T_curr):
    """
    Creates a FrameMsg from a Frame object.

    :param bridge: CvBridge; To convert images.
    :param frame: Frame; Object containing data.
    :param fid: int; Frame ID.
    :param ref_kid: int; Reference Keyframe ID.
    :param ref_T_curr: np.ndarray; Transformation from current to reference.
    :return: FrameMsg; The frame message.
    """
    frame_msg = FrameMsg()
    frame_msg.height_map = bridge.cv2_to_imgmsg(frame.H, encoding="32FC1")
    frame_msg.contact_mask = bridge.cv2_to_imgmsg(
        frame.C.astype(np.uint8), encoding="8UC1"
    )
    frame_msg.normal_map = bridge.cv2_to_imgmsg(frame.N, encoding="32FC3")
    frame_msg.laplacian_map = bridge.cv2_to_imgmsg(frame.L, encoding="32FC1")
    frame_msg.fid = fid
    frame_msg.ref_kid = ref_kid
    frame_msg.ref_t_curr = ref_T_curr.flatten().tolist()
    return frame_msg


def pose_of_frame_msg(frame_msg):
    """
    Extracts the pose of the frame message relative to its reference keyframe.

    :param frame_msg: FrameMsg; The frame message.
    :return: np.ndarray (4, 4); The pose matrix.
    """
    return np.array(frame_msg.ref_t_curr).reshape(4, 4).astype(np.float32)


def center_of_frame_msg(bridge, frame_msg, ppmm):
    """
    Compute the center of the frame.

    :param bridge: CvBridge; To convert images.
    :param frame_msg: FrameMsg; The frame message.
    :param ppmm: float; Pixels size in milimeters.
    :return: np.ndarray (3,); The center point in meters.
    """
    C = bridge.imgmsg_to_cv2(frame_msg.contact_mask) > 0
    H = bridge.imgmsg_to_cv2(frame_msg.height_map)
    xys = np.column_stack(np.where(C))
    cy, cx = np.mean(xys, axis=0)
    cz = H[int(cy), int(cx)]
    point = np.array([cx - C.shape[1] / 2.0, cy - C.shape[0] / 2.0, cz]) * ppmm / 1000
    return point


def frame_msg2mesh(bridge, frame_msg, start_T_ref, ppmm, raise_dist=0.0002):
    """
    Convert the frame message to a mesh.
    Transform the mesh to the correct position based on the transformation.

    :param bridge: CvBridge; To convert images.
    :param frame_msg: FrameMsg; The frame message.
    :param start_T_ref: np.ndarray; Transformation.
    :param ppmm: float; Pixels size in milimeters.
    :param raise_dist: float; Distance to raise the mesh.
    :return: open3d.geometry.TriangleMesh; The mesh.
    """
    start_T_frame = start_T_ref @ pose_of_frame_msg(frame_msg)
    C = bridge.imgmsg_to_cv2(frame_msg.contact_mask) > 0
    H = bridge.imgmsg_to_cv2(frame_msg.height_map)
    pointcloud = height2pointcloud(H, C, ppmm)
    if raise_dist != 0.0:
        pointcloud[:, 2] -= raise_dist
    pointcloud = np.dot(start_T_frame[:3, :3], pointcloud.T).T + start_T_frame[:3, 3]
    mesh = pointcloud2mesh(pointcloud, C)
    mesh.paint_uniform_color([1.0, 0.0, 0.0])
    return mesh
