import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import binary_erosion, distance_transform_edt
from normalflow.utils import wide_remap, height2pointcloud


class Logger:
    """
    A class that can be used as logger in ROS or printer outside of ROS.
    """

    def __init__(self, logger=None):
        self.logger = logger

    def info(self, msg):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    def warning(self, msg):
        if self.logger is not None:
            self.logger.warning(msg)
        else:
            print(msg)


def mask2weight(C):
    """
    Given the contact mask, convert it into a weight map.

    :param C: np.ndarray (H, W); the contact mask.
    :return W: np.ndarray (H, W); the weight map.
    """
    l = C.shape[0] / 60.0
    W = (np.tanh(distance_transform_edt(C) / l - 3) + 1) / 2.0 * C
    return W.astype(np.float32)


def quick_check_overlap(H_ref, C_ref, pointcloud_tar, ref_T_tar, ppmm=0.0634):
    """
    Quickly check if two height maps overlap.
    :param H_ref: np.ndarray (H, W); the height map of the reference frame.
    :param C_ref: np.ndarray (H, W); the contact mask of the reference frame.
    :param pointcloud_tar: np.ndarray (N, 3); the pointcloud of the target frame.
    :param ref_T_tar: np.ndarray (4, 4); the homogeneous transformation matrix.
    :param ppmm: float; pixel per millimeter.
    :return bool; whether the two height maps overlap.
    """
    # TODO: If too slow, can subsample the pointcloud
    # Find the bounding cuboid of the reference frame
    ys, xs = np.where(C_ref)
    zs = H_ref[C_ref]
    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)
    z_min, z_max = np.min(zs), np.max(zs)
    width = (x_max - x_min + 1) * ppmm / 1000.0
    height = (y_max - y_min + 1) * ppmm / 1000.0
    depth = (z_max - z_min) * ppmm / 1000.0
    center = np.array(
        [
            ((x_min + x_max) / 2.0 - H_ref.shape[1] / 2 + 0.5) * ppmm / 1000.0,
            ((y_min + y_max) / 2.0 - H_ref.shape[0] / 2 + 0.5) * ppmm / 1000.0,
            ((z_min + z_max) / 2.0) * ppmm / 1000.0,
        ]
    )
    # Project pointcloud_tar to the cuboid frame
    pointcloud_tar_backproj = (
        np.dot(ref_T_tar[:3, :3], pointcloud_tar.T).T + ref_T_tar[:3, 3] - center
    )
    is_overlapped = np.any(
        np.logical_and.reduce(
            [
                pointcloud_tar_backproj[:, 0] > -width / 2.0,
                pointcloud_tar_backproj[:, 0] < width / 2.0,
                pointcloud_tar_backproj[:, 1] > -height / 2.0,
                pointcloud_tar_backproj[:, 1] < height / 2.0,
                pointcloud_tar_backproj[:, 2] > -depth / 2.0,
                pointcloud_tar_backproj[:, 2] < depth / 2.0,
            ]
        )
    )
    return is_overlapped


def get_backproj_weights(
    W_tar, pointcloud_ref, tar_T_ref, dist_threshold=1e-3, ppmm=0.0634
):
    """
    Given the height map and weight map of the target frame, return the backprojected weight map
    map to the reference frame.
    This function assumed that the two frames are aligned and overlapped.

    :param W_tar: np.ndarray (H, W); the weight map of the target frame.
    :param pointcloud_ref: np.ndarray (N, 3); the pointcloud of the reference frame.
    :param tar_T_ref: np.ndarray (4, 4); the homogeneous transformation matrix.
    :param ppmm: float; pixel per millimeter.
    :param dist_threshold: float; the distance threshold that sets the weight too far to zero.
    :return: np.ndarray (N, 1); the backprojected weight map.
    """
    remapped_pointcloud_ref = (
        np.dot(tar_T_ref[:3, :3], pointcloud_ref.T).T + tar_T_ref[:3, 3]
    )
    remapped_xx_ref = (
        remapped_pointcloud_ref[:, 0] * 1000.0 / ppmm + W_tar.shape[1] / 2 - 0.5
    )
    remapped_yy_ref = (
        remapped_pointcloud_ref[:, 1] * 1000.0 / ppmm + W_tar.shape[0] / 2 - 0.5
    )
    W_tar_backproj = wide_remap(
        W_tar, remapped_xx_ref, remapped_yy_ref, mode=cv2.INTER_NEAREST
    )[:, 0]
    return W_tar_backproj


def get_backproj_weights_and_pointcloud(
    H_tar, W_tar, pointcloud_ref, tar_T_ref, ppmm=0.0634, dist_threshold=3e-3
):
    """
    Given the height map and weight map of the target frame, return the backprojected weight map and pointcloud
    map to the reference frame.

    :param H_tar: np.ndarray (H, W); the height map of the target frame.
    :param W_tar: np.ndarray (H, W); the weight map of the target frame.
    :param pointcloud_ref: np.ndarray (N, 3); the pointcloud of the reference frame.
    :param tar_T_ref: np.ndarray (4, 4); the homogeneous transformation matrix.
    :param ppmm: float; pixel per millimeter.
    :param dist_threshold: float; the distance threshold that sets the weight too far to zero.
    :return:
        weights_tar_backproj: np.ndarray (N, 1); the backprojected weight map.
        pointcloud_tar_backproj: np.ndarray (N, 3); the backprojected pointcloud.
    """
    ref_T_tar = np.linalg.inv(tar_T_ref)
    remapped_pointcloud_ref = (
        np.dot(tar_T_ref[:3, :3], pointcloud_ref.T).T + tar_T_ref[:3, 3]
    )
    remapped_xx_ref = (
        remapped_pointcloud_ref[:, 0] * 1000.0 / ppmm + H_tar.shape[1] / 2 - 0.5
    )
    remapped_yy_ref = (
        remapped_pointcloud_ref[:, 1] * 1000.0 / ppmm + H_tar.shape[0] / 2 - 0.5
    )
    H_tar_backproj = wide_remap(H_tar, remapped_xx_ref, remapped_yy_ref)[:, 0]
    weights_tar_backproj = wide_remap(W_tar, remapped_xx_ref, remapped_yy_ref)[
        :, 0
    ].reshape(-1, 1)
    pointcloud_tar_backproj = np.stack(
        (
            remapped_pointcloud_ref[:, 0],
            remapped_pointcloud_ref[:, 1],
            H_tar_backproj * ppmm / 1000.0,
        ),
        axis=-1,
    )
    pointcloud_tar_backproj = (
        np.dot(ref_T_tar[:3, :3], pointcloud_tar_backproj.T).T + ref_T_tar[:3, 3]
    )
    # Mask out the points too far away
    dist = np.abs(pointcloud_ref[:, 2] - pointcloud_tar_backproj[:, 2])
    dist = (
        dist / dist[0] * np.linalg.norm(pointcloud_ref[0] - pointcloud_tar_backproj[0])
    )
    weights_tar_backproj[dist > dist_threshold, 0] = 0.0
    return weights_tar_backproj, pointcloud_tar_backproj

def get_backproj_mask(C_tar, H_ref, tar_T_ref, ppmm=0.0634):
    """
    Given the contact mask of the target frame, return the backprojected contact mask to the reference frame.

    :param C_tar: np.ndarray (H, W); the contact map of the target frame.
    :param H_ref: np.ndarray (H, W); the height map of the reference frame.
    :param tar_T_ref: np.ndarray (4, 4); the homogeneous transformation matrix from the reference frame to the target frame.
    :param ppmm: float; pixel per millimeter.
    :return:
        C_tar_backproj: np.ndarray (H, W); the backprojected contact mask.
    """
    # Use float32 instead of float64
    H_ref = H_ref.astype(np.float32)
    tar_T_ref = tar_T_ref.astype(np.float32)
    # Get the remapped pixels
    pointcloud_ref = height2pointcloud(H_ref, np.ones_like(H_ref), ppmm)
    remapped_pointcloud_ref = (
        np.dot(tar_T_ref[:3, :3], pointcloud_ref.T).T + tar_T_ref[:3, 3]
    )
    remapped_xx_ref = (
        remapped_pointcloud_ref[:, 0] * 1000.0 / ppmm + H_ref.shape[1] / 2 - 0.5
    )
    remapped_yy_ref = (
        remapped_pointcloud_ref[:, 1] * 1000.0 / ppmm + H_ref.shape[0] / 2 - 0.5
    )
    # Get the backprojected laplacians and contact mask
    C_tar_backproj = (
        wide_remap(C_tar.astype(np.float32), remapped_xx_ref, remapped_yy_ref)[:, 0]
        > 0.5
    )
    C_tar_backproj = C_tar_backproj.reshape(H_ref.shape)
    xx_region = np.logical_and(remapped_xx_ref >= 0, remapped_xx_ref < C_tar.shape[1])
    yy_region = np.logical_and(remapped_yy_ref >= 0, remapped_yy_ref < C_tar.shape[0])
    xy_region = np.logical_and(xx_region, yy_region).reshape(C_tar.shape)
    erode_size = max(C_tar.shape[0] // 48, 2)
    C_tar_backproj = np.logical_and(C_tar_backproj, xy_region)
    C_tar_backproj = binary_erosion(
        C_tar_backproj, structure=np.ones((erode_size, erode_size))
    )
    return C_tar_backproj

def pointcloud2mesh(pointcloud, C):
    """
    Given a pointcloud and the contact mask, create a triangle mesh.
    Note that the pointcloud need to be in the flattened image ordering or masked pointcloud.

    :param pointcloud: np.2darray (N, 3); the pointcloud.
    :param C: np.2darray (H, W); the contact mask.
    :return mesh: o3d.geometry.TriangleMesh; the triangle mesh.
    """
    if len(pointcloud) == C.shape[0] * C.shape[1]:
        pointcloud = pointcloud[C.flatten()]
    # Construct the pointcloud faces
    O = np.zeros_like(C, dtype=np.int32)
    O[np.where(C)] = np.arange(np.sum(C))
    O_padded = np.pad(O, ((0, 1), (0, 1)), mode="constant", constant_values=-1)
    C_padded = np.pad(C, ((0, 1), (0, 1)), mode="constant", constant_values=False)
    index1 = np.where(C)
    index2 = (index1[0] + 1, index1[1])
    index3 = (index1[0], index1[1] + 1)
    index4 = (index1[0] + 1, index1[1] + 1)
    upper_faces = np.stack(
        [O_padded[index1], O_padded[index2], O_padded[index3]], axis=-1
    )
    mask_upper_faces = np.logical_and(C_padded[index2], C_padded[index3])
    lower_faces = np.stack(
        [O_padded[index2], O_padded[index4], O_padded[index3]], axis=-1
    )
    mask_lower_faces = np.logical_and(mask_upper_faces, C_padded[index4])
    faces = np.concatenate(
        [upper_faces[mask_upper_faces], lower_faces[mask_lower_faces]], axis=0
    )
    # Create a triangle mesh from the coordinates
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(pointcloud)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    # Assign the normals
    mesh.compute_vertex_normals()
    return mesh


def matrix_in_mm(T_m):
    """Matrix in meters to matrix in millimeters."""
    T_mm = T_m.copy()
    T_mm[:3, 3] *= 1000
    return T_mm


def matrix_in_m(T_mm):
    """Matrix in millimeters to matrix in meters."""
    T_m = T_mm.copy()
    T_m[:3, 3] /= 1000
    return T_m


def laplacian2gray(L):
    """
    Get the gray image from the laplacian map.

    :param L: np.ndarray (H, W); the laplacian map.
    :return: np.ndarray (H, W); the gray image.
    """
    L = ((L + 0.7) / 1.4) * 255
    return L.astype(np.uint8)


def get_sift_features(L, C, sift):
    """
    Calculate the SIFT feature from the laplacian of the normal map.

    :param L: np.2darray (H, W); the laplacian map.
    :param C: np.2darray (H, W); the contact mask.
    :param sift: cv2.SIFT_create(); the SIFT object.
    :return kp: list; the keypoints.
            des: np.2darray (N, 128); the descriptors.
    """
    # Speed up feature extraction by only extracting features within the contact mask
    ys, xs = np.where(C)
    min_x, max_x = np.min(xs), np.max(xs)
    min_y, max_y = np.min(ys), np.max(ys)
    gray_L = laplacian2gray(L)
    kp, des = sift.detectAndCompute(gray_L[min_y : max_y + 1, min_x : max_x + 1], None)
    if des is None:
        return [], None
    else:
        # Filter out features too close to mask boundaries
        erode_size = max(C.shape[0] // 48, 1)
        eroded_C = binary_erosion(C, structure=np.ones((erode_size, erode_size)))
        filtered_kp = []
        filtered_des = []
        for k, d in zip(kp, des):
            k.pt = (k.pt[0] + min_x, k.pt[1] + min_y)
            if eroded_C[int(k.pt[1]), int(k.pt[0])]:
                filtered_kp.append(k)
                filtered_des.append(d)
        filtered_des = np.array(filtered_des)
        if len(filtered_des) == 0:
            filtered_kp = []
            filtered_des = None
        return filtered_kp, filtered_des


def M2T(M, w, h, ppmm):
    """
    Convert a 2D affine transformation matrix to a 3D transformation matrix,
    :param M: (2, 3) numpy.ndarray, 2D affine transformation matrix.
    :param w: int, width of the image.
    :param h: int, height of the image.
    :param ppmm: float, pixels per millimeter.
    :return T: (4, 4) numpy.ndarray, 3D transformation matrix.
    """
    tx = (M[0, 2] + (M[0, 0] * w / 2 + M[0, 1] * h / 2) - w / 2) * ppmm / 1000.0
    ty = (M[1, 2] + (M[1, 0] * w / 2 + M[1, 1] * h / 2) - h / 2) * ppmm / 1000.0
    scale = np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)
    T = np.array(
        [
            [M[0, 0] / scale, M[0, 1] / scale, 0, tx],
            [M[1, 0] / scale, M[1, 1] / scale, 0, ty],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    return T
