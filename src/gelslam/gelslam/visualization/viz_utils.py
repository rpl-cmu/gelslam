import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.ndimage import binary_erosion

from normalflow.utils import height2pointcloud
from gelslam.utils import laplacian2gray, wide_remap


def arrange_images_in_grid(frames, n_rows=None, n_cols=None):
    """
    Arrange a list of frames into the closest n_rows x n_cols grid,
    filling any empty space with black.
    :param frames: list of np.ndarray (H, W, C); list of images with the same shape.
    :return: np.ndarray (H', W', C); grid image containing all input images.
    """
    if n_rows is None and n_cols is None:
        n_rows = int(np.ceil(np.sqrt(len(frames) - 1e-8)))
        n_cols = int(np.ceil(len(frames) / n_rows))
    h, w, c = frames[0].shape
    grid_frame = np.zeros((n_rows * h, n_cols * w, c), dtype=np.uint8)
    for idx, frame in enumerate(frames):
        row, col = divmod(idx, n_cols)
        grid_frame[row * h : (row + 1) * h, col * w : (col + 1) * w] = frame
    return grid_frame


def label_text_and_pad(frame, text, pad_width=10):
    """
    An utility function that takes an image, label text in the lower right, and pad the border.
    :param frame: np.ndarray(H, W, 3); the image.
    :param text: string; the text to be labeled.
    :param pad_width: int; the number of pixels to be padded.
    :return frame: np.ndarray (H+2p, W+2p, 3); the labeled and padded image
    """
    cv2.putText(
        frame,
        text,
        (
            frame.shape[1]
            - 10
            - cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0],
            frame.shape[0] - 10,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    frame = cv2.copyMakeBorder(
        frame,
        pad_width,
        pad_width,
        pad_width,
        pad_width,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )
    return frame


def plot_sift_matches(L_ref, ref_kp, C_shared_ref, L_tar, tar_kp, matches):
    """
    Plot the SIFT matches between two frames.

    :param L_ref: np.ndarray (H, W); the laplacian map of the reference frame.
    :param ref_kp: list of cv2.KeyPoint; the keypoints of the reference frame.
    :param C_shared_ref: np.ndarray (H, W); the shared contact map of the reference frame.
    :param L_tar: np.ndarray (H, W); the laplacian map of the target frame.
    :param tar_kp: list of cv2.KeyPoint; the keypoints of the target frame.
    :param matches: list of tuple; the matches between the keypoints.
    :return: np.ndarray (H, 2W+10, 3); the image with matches.
    """
    frame_ref = cv2.cvtColor(laplacian2gray(L_ref), cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(
        C_shared_ref.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(frame_ref, contours, -1, (0, 255, 0), 1)
    cv2.putText(
        frame_ref,
        "Reference Frame",
        (10, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )
    frame_tar = cv2.cvtColor(laplacian2gray(L_tar), cv2.COLOR_GRAY2BGR)
    cv2.putText(
        frame_tar,
        "Target Frame",
        (10, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )
    spacer = np.zeros((frame_ref.shape[0], 10, 3), dtype=np.uint8)
    frame_matches = np.hstack((frame_ref, spacer, frame_tar))
    # Draw unmatched keypoints in blue for both frames
    ref_matched_indices = {match[0] for match in matches}
    for k, kp in enumerate(ref_kp):
        if k not in ref_matched_indices:
            ref_pt = tuple(map(int, kp.pt))
            cv2.circle(frame_matches, ref_pt, 2, (255, 0, 0), -1)
    tar_matched_indices = {match[1] for match in matches}
    for k, kp in enumerate(tar_kp):
        if k not in tar_matched_indices:
            tar_pt = (
                int(kp.pt[0] + frame_ref.shape[1] + spacer.shape[1]),
                int(kp.pt[1]),
            )
            cv2.circle(frame_matches, tar_pt, 2, (255, 0, 0), -1)
    # Draw keypoints within shared contact mask in green for the reference frame
    for kp in ref_kp:
        if C_shared_ref[int(kp.pt[1]), int(kp.pt[0])]:
            ref_pt = tuple(map(int, kp.pt))
            cv2.circle(frame_matches, ref_pt, 2, (0, 255, 0), -1)
    # Draw matches with lines for tracking
    for match in matches:
        ref_pt = tuple(map(int, ref_kp[match[0]].pt))
        tar_pt = (
            int(tar_kp[match[1]].pt[0] + frame_ref.shape[1] + spacer.shape[1]),
            int(tar_kp[match[1]].pt[1]),
        )
        cv2.line(frame_matches, ref_pt, tar_pt, (0, 0, 255), 1)
        cv2.circle(frame_matches, ref_pt, 2, (0, 0, 255), -1)
        cv2.circle(frame_matches, tar_pt, 2, (0, 0, 255), -1)
    return frame_matches


def get_backproj_laplacian(L_tar, C_tar, H_ref, tar_T_ref, ppmm=0.0634):
    """
    Given the laplacian and contact map of the target frame, return the backprojected laplacian
    map to the reference frame.

    :param L_tar: np.ndarray (H, W); the laplacian map of the target frame.
    :param C_tar: np.ndarray (H, W); the contact map of the target frame.
    :param H_ref: np.ndarray (H, W); the height map of the reference frame.
    :param tar_T_ref: np.ndarray (4, 4); the homogeneous transformation matrix from the reference frame to the target frame.
    :param ppmm: float; pixel per millimeter.
    :return:
        L_tar_backproj: np.ndarray (H, W); the backprojected laplacian map.
        C_tar_backproj: np.ndarray (H, W); the backprojected contact mask.
    """
    # Use float32 instead of float64
    L_tar = L_tar.astype(np.float32)
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
    L_tar_backproj = wide_remap(L_tar, remapped_xx_ref, remapped_yy_ref)[:, 0]
    L_tar_backproj = L_tar_backproj.reshape((H_ref.shape[0], H_ref.shape[1]))
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
    return L_tar_backproj, C_tar_backproj


def plot_laplacian_comparison(
    L_ref, C_ref, H_ref, L_tar, C_tar, tar_T_ref, ppmm=0.0634
):
    """
    Plot the laplacian comparison plot.

    :param L_ref: np.ndarray (H, W); the laplacian map of the reference frame.
    :param C_ref: np.ndarray (H, W); the contact map of the reference frame.
    :param H_ref: np.ndarray (H, W); the height map of the reference frame.
    :param L_tar: np.ndarray (H, W); the laplacian map of the target frame.
    :param C_tar: np.ndarray (H, W); the contact map of the target frame.
    :param tar_T_ref: np.ndarray (4, 4); the homogeneous transformation matrix.
    :param ppmm: float; pixel per millimeter.
    :return: np.ndarray (H, 3W+20, 3); the laplacian comparison plot.
    """
    # Back project laplacian to compute laplacian difference and shared contact mask
    L_tar_backproj, C_tar_backproj = get_backproj_laplacian(
        L_tar, C_tar, H_ref, tar_T_ref, ppmm
    )
    L_diff = np.clip(L_ref - L_tar_backproj, -0.7, 0.7)
    C_shared = np.logical_and(C_ref, C_tar_backproj)
    # Plot the Laplacian comparison plot
    contours, _ = cv2.findContours(
        C_shared.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    frame_ref = cv2.cvtColor(laplacian2gray(L_ref), cv2.COLOR_GRAY2BGR)
    cv2.putText(
        frame_ref,
        "Reference Frame",
        (10, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )
    cv2.drawContours(frame_ref, contours, -1, (0, 255, 0), 1)
    frame_tar_backproj = cv2.cvtColor(
        laplacian2gray(L_tar_backproj), cv2.COLOR_GRAY2BGR
    )
    cv2.putText(
        frame_tar_backproj,
        "Projected Target Frame",
        (10, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )
    cv2.drawContours(frame_tar_backproj, contours, -1, (0, 255, 0), 1)
    L_diff[np.logical_not(C_shared)] = 0.0
    frame_diff = cv2.cvtColor(laplacian2gray(L_diff), cv2.COLOR_GRAY2BGR)
    cv2.putText(
        frame_diff,
        "Difference Map",
        (10, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )
    spacer = np.zeros((frame_ref.shape[0], 10, 3), dtype=np.uint8)
    frame_laplacian_comparison = np.hstack(
        (frame_ref, spacer, frame_tar_backproj, spacer, frame_diff)
    )
    return frame_laplacian_comparison


def plot_laplacian_dot(L_ref, C_ref, H_ref, L_tar, C_tar, tar_T_ref, ppmm=0.0634):
    """
    Plot the dot product of two laplacian maps when transformed into same coordinate.

    :param L_ref: np.ndarray (H, W); the laplacian map of the reference frame.
    :param C_ref: np.ndarray (H, W); the contact map of the reference frame.
    :param H_ref: np.ndarray (H, W); the height map of the reference frame.
    :param L_tar: np.ndarray (H, W); the laplacian map of the target frame.
    :param C_tar: np.ndarray (H, W); the contact map of the target frame.
    :param tar_T_ref: np.ndarray (4, 4); the homogeneous transformation matrix.
    :param ppmm: float; pixel per millimeter.
    :return: np.ndarray (H, 3W+20, 3); the laplacian dot plot.
    """
    # Back project laplacian to compute laplacian difference and shared contact mask
    L_tar_backproj, C_tar_backproj = get_backproj_laplacian(
        L_tar, C_tar, H_ref, tar_T_ref, ppmm
    )
    L_dot = np.clip(L_ref * L_tar_backproj, -0.05, 0.05)
    L_norm = np.clip(L_ref * L_ref, -0.05, 0.05)
    C_shared = np.logical_and(C_ref, C_tar_backproj)
    # Plot the Laplacian comparison plot
    contours, _ = cv2.findContours(
        C_shared.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    frame_ref = cv2.cvtColor(laplacian2gray(L_ref), cv2.COLOR_GRAY2BGR)
    cv2.putText(
        frame_ref,
        "Reference Frame",
        (10, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )
    cv2.drawContours(frame_ref, contours, -1, (0, 255, 0), 1)
    frame_tar_backproj = cv2.cvtColor(
        laplacian2gray(L_tar_backproj), cv2.COLOR_GRAY2BGR
    )
    cv2.putText(
        frame_tar_backproj,
        "Projected Target Frame",
        (10, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )
    cv2.drawContours(frame_tar_backproj, contours, -1, (0, 255, 0), 1)
    L_dot[np.logical_not(C_shared)] = 0.0
    L_dot_image = ((L_dot + 0.05) / 0.1 * 255).astype(np.uint8)
    frame_diff = cv2.cvtColor(L_dot_image, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        frame_diff,
        "Ref dot Tar Map",
        (10, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )
    L_norm[np.logical_not(C_shared)] = 0.0
    L_norm_image = ((L_norm + 0.05) / 0.1 * 255).astype(np.uint8)
    frame_norm = cv2.cvtColor(L_norm_image, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        frame_norm,
        "Ref dot Ref Map",
        (10, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )
    spacer = np.zeros((frame_ref.shape[0], 10, 3), dtype=np.uint8)
    frame_laplacian_comparison = np.hstack(
        (frame_ref, spacer, frame_tar_backproj, spacer, frame_norm, spacer, frame_diff)
    )
    return frame_laplacian_comparison


def plot_match_histogram(
    fig, ax, matched_scores, unmatched_scores, matched_bins, unmatched_bins
):
    """
    Plot the matching histogram.

    :param fig: plt.Figure; the figure.
    :param ax: plt.Axes; the axes.
    :param matched_scores: list of float; the matched scores.
    :param unmatched_scores: list of float; the unmatched scores.
    :param matched_bins: np.ndarray; the bins for matched scores.
    :param unmatched_bins: np.ndarray; the bins for unmatched scores.
    """
    ax.hist(
        np.clip(matched_scores, matched_bins[0], matched_bins[-1]),
        bins=matched_bins,
        alpha=1.0,
        edgecolor="red",
        facecolor="none",
        label="Matched",
        histtype="stepfilled",
        linewidth=3.0,
    )
    ax.hist(
        np.clip(unmatched_scores, unmatched_bins[0], unmatched_bins[-1]),
        bins=unmatched_bins,
        alpha=1.0,
        edgecolor="blue",
        facecolor="none",
        label="Unmatched",
        histtype="stepfilled",
        linewidth=3.0,
    )
    ax.legend()


def create_edge_mesh(point1, point2, color=[0, 0, 0], radius=0.00005):
    """
    Create the edge mesh with the cylinder mesh.
    :param point1: np.ndarray (3,); the first point position (in meters).
    :param point2: np.ndarray (3,); the second point position (in meters).
    :param color: list of float; the color of the edge.
    :param radius: float; the radius of the edge (in meters).
    """
    height = np.linalg.norm(point2 - point1)
    edge_mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
    edge_mesh.paint_uniform_color(color)
    midpoint = (point1 + point2) / 2
    direction = (point2 - point1) / height
    default_direction = np.array([0, 0, 1])
    # Compute the rotation matrix to align the cylinder
    if not np.allclose(direction, default_direction):
        axis = np.cross(default_direction, direction)
        axis_length = np.linalg.norm(axis)
        if axis_length > 1e-6:
            axis = axis / axis_length
            angle = np.arccos(np.dot(default_direction, direction))
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
            edge_mesh.rotate(R, center=(0, 0, 0))
    # Translate to the midpoint
    edge_mesh.translate(midpoint)
    return edge_mesh


# def get_backproj_gxy(G_tar, C_tar, H_ref, tar_T_ref, ppmm=0.0634):
#     """
#     Given the gradient map and contact map of the target frame, return the backprojected normal
#     map to the reference frame.
#     :param G_tar: np.3darray (H, W, 2); the gradient map (gxy) of the target frame.
#     :param C_tar: np.2darray (H, W); the contact map of the target frame.
#     :param H_ref: np.2darray (H, W); the height map of the reference frame.
#     :param tar_T_ref: np.2darray (4, 4); the homogeneous transformation matrix from the reference frame to the target frame.
#     :param ppmm: float; pixel per millimeter.
#     :return:
#         G_tar_backproj: np.3darray (H, W, 2); the backprojected gradient map.
#         C_tar_backproj: np.2darray (H, W); the backprojected contact mask.
#     """
#     # Use float32 instead of float64
#     G_tar = G_tar.astype(np.float32)
#     H_ref = H_ref.astype(np.float32)
#     tar_T_ref = tar_T_ref.astype(np.float32)
#     ref_T_tar = np.linalg.inv(tar_T_ref)
#     # Get the rotated normal map
#     N_tar = gxy2normal(G_tar)
#     warped_N_tar = np.dot(ref_T_tar[:3, :3], N_tar.reshape(-1, 3).T).T.reshape(
#         N_tar.shape
#     )
#     # Get the warped pixels
#     pointcloud_ref = height2pointcloud(H_ref, ppmm).astype(np.float32)
#     warped_pointcloud_ref = (
#         np.dot(tar_T_ref[:3, :3], pointcloud_ref.T).T + tar_T_ref[:3, 3]
#     )
#     warped_xx_ref = (
#         warped_pointcloud_ref[:, 0] * 1000.0 / ppmm + H_ref.shape[1] / 2 - 0.5
#     )
#     warped_yy_ref = (
#         warped_pointcloud_ref[:, 1] * 1000.0 / ppmm + H_ref.shape[0] / 2 - 0.5
#     )
#     # Get the backprojected normals and contact mask
#     N_tar_backproj = wide_remap(warped_N_tar, warped_xx_ref, warped_yy_ref)[:, 0, :]
#     N_tar_backproj = N_tar_backproj.reshape((H_ref.shape[0], H_ref.shape[1], 3))
#     C_tar_backproj = (
#         wide_remap(C_tar.astype(np.float32), warped_xx_ref, warped_yy_ref)[:, 0] > 0.5
#     )
#     C_tar_backproj = C_tar_backproj.reshape(H_ref.shape)
#     xx_region = np.logical_and(warped_xx_ref >= 0, warped_xx_ref < C_tar.shape[1])
#     yy_region = np.logical_and(warped_yy_ref >= 0, warped_yy_ref < C_tar.shape[0])
#     xy_region = np.logical_and(xx_region, yy_region).reshape(C_tar.shape)
#     erode_size = max(C_tar.shape[0] // 48, 2)
#     C_tar_backproj = np.logical_and(C_tar_backproj, xy_region)
#     C_tar_backproj = binary_erosion(
#         C_tar_backproj, structure=np.ones((erode_size, erode_size))
#     )
#     G_tar_backproj = np.stack(
#         [
#             -N_tar_backproj[:, :, 0] / (N_tar_backproj[:, :, 2] + 1e-8),
#             -N_tar_backproj[:, :, 1] / (N_tar_backproj[:, :, 2] + 1e-8),
#         ],
#         axis=-1,
#     )
#     return G_tar_backproj, C_tar_backproj


# def gxy2image(G):
#     """
#     Get the BGR image from gxy map
#     :param G: np.ndarray (H, W, 2); the gxy map.
#     :return: np.ndarray (H, W, 3); the image.
#     """
#     N = gxy2normal(G)
#     image = ((N + 1) * 127.5).astype(np.uint8)
#     image = np.stack((image[:, :, 2], image[:, :, 1], image[:, :, 0]), axis=-1)
#     return image
