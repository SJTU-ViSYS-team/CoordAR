import ipdb
import numpy as np
import math
import pytorch3d.transforms
import torch
from detectron2.layers import cat

from scipy.linalg import logm
import numpy.linalg as LA
from math import pi
from transforms3d.euler import euler2quat, mat2euler, quat2euler, euler2mat
from transforms3d.quaternions import (
    axangle2quat,
    mat2quat,
    qinverse,
    qmult,
    quat2mat,
)
from transforms3d.axangles import axangle2mat
from src.utils.pose_utils import quat2mat_torch
from .se3 import se3_inverse, se3_mul


def calc_RT_delta(
    pose_src, pose_tgt, T_means, T_stds, rot_coord="MODEL", rot_type="MATRIX"
):
    """project the points in source corrd to target corrd.

    :param pose_src: pose matrix of soucre, [R|T], 3x4
    :param pose_tgt: pose matrix of target, [R|T], 3x4
    :param rot_coord: model/camera
    :param rot_type: quat/euler/matrix
    :return: Rm_delta
    :return: T_delta
    """
    if rot_coord.lower() == "naive":
        se3_src2tgt = se3_mul(pose_tgt, se3_inverse(pose_src))
        Rm_delta = se3_src2tgt[:, :3]
        T_delta = se3_src2tgt[:, 3].reshape(3)
    else:
        Rm_delta = R_inv_transform(pose_src[:3, :3], pose_tgt[:3, :3], rot_coord)
        T_delta = T_inv_transform(
            pose_src[:, 3], pose_tgt[:, 3], T_means, T_stds, rot_coord
        )

    if rot_type.lower() == "quat":
        r = mat2quat(Rm_delta)
    elif rot_type.lower() == "euler":
        r = mat2euler(Rm_delta)
    elif rot_type.lower() == "matrix":
        r = Rm_delta
    else:
        raise Exception("Unknown rot_type: {}".format(rot_type))
    t = np.squeeze(T_delta)

    return r, t


def R_transform(R_src, R_delta, rot_coord="MODEL"):
    """transform R_src use R_delta.

    :param R_src: matrix
    :param R_delta:
    :param rot_coord:
    :return:
    """
    if rot_coord.lower() == "model":
        R_output = np.dot(R_src, R_delta)
    elif (
        rot_coord.lower() == "camera"
        or rot_coord.lower() == "naive"
        or rot_coord.lower() == "camera_new"
    ):
        R_output = np.dot(R_delta, R_src)
    else:
        raise Exception("Unknown rot_coord in R_transform: {}".format(rot_coord))
    return R_output


def R_inv_transform(R_src, R_tgt, rot_coord):
    if rot_coord.lower() == "model":
        # dR_m2m = R_src_c2m x R_tgt_m2c
        R_delta = np.dot(R_src.transpose(), R_tgt)
    elif rot_coord.lower() == "camera" or rot_coord.lower() == "camera_new":
        # dR_c2c = R_tgt_m2c x R_src_c2m
        R_delta = np.dot(R_tgt, R_src.transpose())
    else:
        raise Exception("Unknown rot_coord in R_inv_transform: {}".format(rot_coord))
    return R_delta


def T_transform(T_src, T_delta, T_means, T_stds, rot_coord):
    """
    :param T_src: (x1, y1, z1)
    :param T_delta: (dx, dy, dz), normed
    :return: T_tgt: (x2, y2, z2)
    """
    # print("T_src: {}".format(T_src))
    assert T_src[2] != 0, "T_src: {}".format(T_src)
    T_delta_1 = T_delta * T_stds + T_means
    T_tgt = np.zeros((3,))
    z2 = T_src[2] / np.exp(T_delta_1[2])
    T_tgt[2] = z2
    if rot_coord.lower() == "camera" or rot_coord.lower() == "model":
        # use this
        T_tgt[0] = z2 * (T_delta_1[0] + T_src[0] / T_src[2])
        T_tgt[1] = z2 * (T_delta_1[1] + T_src[1] / T_src[2])
    elif rot_coord.lower() == "camera_new":
        T_tgt[0] = T_src[2] * T_delta_1[0] + T_src[0]
        T_tgt[1] = T_src[2] * T_delta_1[1] + T_src[1]
    else:
        raise Exception("Unknown: {}".format(rot_coord))

    return T_tgt


def T_transform_naive(R_delta, T_src, T_delta):
    T_src = T_src.reshape((3, 1))
    T_delta = T_delta.reshape((3, 1))
    T_new = np.dot(R_delta, T_src) + T_delta
    return T_new.reshape((3,))


def T_inv_transform(T_src, T_tgt, T_means, T_stds, rot_coord):
    """
    :param T_src:
    :param T_tgt:
    :param T_means:
    :param T_stds:
    :return: T_delta: delta in pixel
    """
    T_delta = np.zeros((3,))
    if rot_coord.lower() == "camera_new":
        T_delta[0] = (T_tgt[0] - T_src[0]) / T_src[2]
        T_delta[1] = (T_tgt[1] - T_src[1]) / T_src[2]
    elif rot_coord.lower() == "camera" or rot_coord.lower() == "model":
        T_delta[0] = T_tgt[0] / T_tgt[2] - T_src[0] / T_src[2]
        T_delta[1] = T_tgt[1] / T_tgt[2] - T_src[1] / T_src[2]
    else:
        raise Exception("Unknown: {}".format(rot_coord))
    T_delta[2] = np.log(T_src[2] / T_tgt[2])
    T_delta_normed = (T_delta - T_means) / T_stds
    return T_delta_normed


def RT_transform(pose_src, r, t, T_means, T_stds, rot_coord="MODEL"):
    # r: 4(quat) or 3(euler) number
    # t: 3 number, (delta_x, delta_y, scale)
    r = np.squeeze(r)
    if r.shape[0] == 3:
        Rm_delta = euler2mat(r[0], r[1], r[2])
    elif r.shape[0] == 4:
        # QUAT
        quat = r / LA.norm(r)
        Rm_delta = quat2mat(quat)
    else:
        raise Exception("Unknown r shape: {}".format(r.shape))
    t_delta = np.squeeze(t)

    if rot_coord.lower() == "naive":
        se3_mx = np.zeros((3, 4))
        se3_mx[:, :3] = Rm_delta
        se3_mx[:, 3] = t
        pose_est = se3_mul(se3_mx, pose_src)
    else:
        pose_est = np.zeros((3, 4))
        pose_est[:3, :3] = R_transform(pose_src[:3, :3], Rm_delta, rot_coord)
        pose_est[:3, 3] = T_transform(
            pose_src[:, 3], t_delta, T_means, T_stds, rot_coord
        )

    return pose_est


def calc_se3(pose_src, pose_tgt):
    """
    :param pose_src: pose matrix of soucre, [R|T], 3x4
    :param pose_tgt: pose matrix of target, [R|T], 3x4
    """
    se3_src2tgt = se3_mul(pose_tgt, se3_inverse(pose_src))
    rotm = se3_src2tgt[:, :3]
    t = se3_src2tgt[:, 3].reshape(3)

    return rotm, t


def se3_q2m(se3_q):
    assert se3_q.size == 7
    se3_mx = np.zeros((3, 4))
    quat = se3_q[0:4] / LA.norm(se3_q[0:4])
    R = quat2mat(quat)
    se3_mx[:, :3] = R
    se3_mx[:, 3] = se3_q[4:]
    return se3_mx


def quat_trans_to_pose_m(quat, trans):
    se3_mx = np.zeros((3, 4))
    # quat = quat / LA.norm(quat)
    R = quat2mat(quat)  # normalize internally
    se3_mx[:, :3] = R
    se3_mx[:, 3] = trans
    return se3_mx


def allocentric_to_egocentric(allo_pose, src_type="mat", dst_type="mat"):
    """Given an allocentric (object-centric) pose, compute new camera-centric
    pose Since we do detection on the image plane and our kernels are
    2D-translationally invariant, we need to ensure that rendered objects
    always look identical, independent of where we render them.

    Since objects further away from the optical center undergo skewing,
    we try to visually correct by rotating back the amount between
    optical center ray and object centroid ray. Another way to solve
    that might be translational variance
    (https://arxiv.org/abs/1807.03247)
    """
    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = np.asarray([0, 0, 1.0])
    if src_type == "mat":
        trans = allo_pose[:3, 3]
    elif src_type == "quat":
        trans = allo_pose[4:7]
    else:
        raise ValueError("src_type should be mat or quat, got: {}".format(src_type))
    obj_ray = trans.copy() / np.linalg.norm(trans)
    angle = math.acos(cam_ray.dot(obj_ray))

    # Rotate back by that amount
    if angle > 0:
        if dst_type == "mat":
            ego_pose = np.zeros((3, 4), dtype=allo_pose.dtype)
            ego_pose[:3, 3] = trans
            rot_mat = axangle2mat(axis=np.cross(cam_ray, obj_ray), angle=angle)
            if src_type == "mat":
                ego_pose[:3, :3] = np.dot(rot_mat, allo_pose[:3, :3])
            elif src_type == "quat":
                ego_pose[:3, :3] = np.dot(rot_mat, quat2mat(allo_pose[:4]))
        elif dst_type == "quat":
            ego_pose = np.zeros((7,), dtype=allo_pose.dtype)
            ego_pose[4:7] = trans
            rot_q = axangle2quat(np.cross(cam_ray, obj_ray), angle)
            if src_type == "quat":
                ego_pose[:4] = qmult(rot_q, allo_pose[:4])
            elif src_type == "mat":
                ego_pose[:4] = qmult(rot_q, mat2quat(allo_pose[:3, :3]))
        else:
            raise ValueError("dst_type should be mat or quat, got: {}".format(dst_type))
    else:  # allo to ego
        if src_type == "mat" and dst_type == "quat":
            ego_pose = np.zeros((7,), dtype=allo_pose.dtype)
            ego_pose[:4] = mat2quat(allo_pose[:3, :3])
            ego_pose[4:7] = allo_pose[:3, 3]
        elif src_type == "quat" and dst_type == "mat":
            ego_pose = np.zeros((3, 4), dtype=allo_pose.dtype)
            ego_pose[:3, :3] = quat2mat(allo_pose[:4])
            ego_pose[:3, 3] = allo_pose[4:7]
        else:
            ego_pose = allo_pose.copy()[:3, :]
    return ego_pose


def allocentric_to_egocentric_batch_torch(
    allo_pose, src_type="mat", dst_type="mat", cam_ray=(0, 0, 1.0)
):
    # allo_pose: (B, 4, 4)
    bs = allo_pose.shape[0]
    eps = 1e-6
    cam_ray = torch.tensor(cam_ray, dtype=allo_pose.dtype, device=allo_pose.device)[
        None
    ].repeat(bs, 1)
    if src_type == "mat":
        trans = allo_pose[:, :3, 3]
    else:
        raise ValueError("src_type should be mat or quat, got: {}".format(src_type))
    obj_ray = trans / (torch.linalg.norm(trans, dim=1, keepdim=True) + eps)  # (B, 3)
    angle = torch.acos(torch.einsum("b i, b i->b", cam_ray, obj_ray))  # (B,)
    # Rotate back by that amount
    if dst_type == "mat":
        ego_t = trans
        rot_aa = torch.cross(cam_ray, obj_ray, dim=1)
        rot_aa = rot_aa / (torch.linalg.norm(rot_aa, dim=1, keepdim=True) + eps)
        rot_mat = pytorch3d.transforms.axis_angle_to_matrix(
            rot_aa * angle.unsqueeze(-1)
        )
        ego_rot = rot_mat @ allo_pose[:, :3, :3]
        ego_pose = torch.cat([ego_rot, ego_t.unsqueeze(-1)], dim=-1)  # (B, 3, 4)
        last_row = (
            torch.tensor([0, 0, 0, 1], device=allo_pose.device)
            .reshape(1, 1, 4)
            .repeat(bs, 1, 1)
        )
        ego_pose = torch.cat(
            [ego_pose, last_row],
            dim=1,
        )
    else:
        raise ValueError("dst_type should be mat or quat, got: {}".format(dst_type))
    return ego_pose


def egocentric_to_allocentric(
    ego_pose, src_type="mat", dst_type="mat", cam_ray=(0, 0, 1.0)
):
    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = np.asarray(cam_ray)
    if src_type == "mat":
        trans = ego_pose[:3, 3]
    elif src_type == "quat":
        trans = ego_pose[4:7]
    else:
        raise ValueError("src_type should be mat or quat, got: {}".format(src_type))
    obj_ray = trans.copy() / np.linalg.norm(trans)
    angle = math.acos(cam_ray.dot(obj_ray))
    # Rotate back by that amount
    if angle > 0:
        if dst_type == "mat":
            allo_pose = np.zeros((3, 4), dtype=ego_pose.dtype)
            allo_pose[:3, 3] = trans
            rot_mat = axangle2mat(axis=np.cross(cam_ray, obj_ray), angle=-angle)
            if src_type == "mat":
                allo_pose[:3, :3] = np.dot(rot_mat, ego_pose[:3, :3])
            elif src_type == "quat":
                allo_pose[:3, :3] = np.dot(rot_mat, quat2mat(ego_pose[:4]))
        elif dst_type == "quat":
            allo_pose = np.zeros((7,), dtype=ego_pose.dtype)
            allo_pose[4:7] = trans
            rot_q = axangle2quat(np.cross(cam_ray, obj_ray), -angle)
            if src_type == "quat":
                allo_pose[:4] = qmult(rot_q, ego_pose[:4])
            elif src_type == "mat":
                allo_pose[:4] = qmult(rot_q, mat2quat(ego_pose[:3, :3]))
        else:
            raise ValueError("dst_type should be mat or quat, got: {}".format(dst_type))
    else:
        if src_type == "mat" and dst_type == "quat":
            allo_pose = np.zeros((7,), dtype=ego_pose.dtype)
            allo_pose[:4] = mat2quat(ego_pose[:3, :3])
            allo_pose[4:7] = ego_pose[:3, 3]
        elif src_type == "quat" and dst_type == "mat":
            allo_pose = np.zeros((3, 4), dtype=ego_pose.dtype)
            allo_pose[:3, :3] = quat2mat(ego_pose[:4])
            allo_pose[:3, 3] = ego_pose[4:7]
        else:
            allo_pose = ego_pose.copy()
    return allo_pose


def egocentric_to_allocentric_torch(
    ego_pose, src_type="mat", dst_type="mat", cam_ray=(0, 0, 1.0)
):
    # ego_pose: (B, 4, 4)
    bs = ego_pose.shape[0]
    cam_ray = torch.tensor(cam_ray, dtype=ego_pose.dtype, device=ego_pose.device)[
        None
    ].repeat(bs, 1)
    if src_type == "mat":
        trans = ego_pose[:, :3, 3]
    else:
        raise ValueError("src_type should be mat or quat, got: {}".format(src_type))
    obj_ray = trans / torch.linalg.norm(trans, dim=1, keepdim=True)  # (B, 3)
    angle = torch.acos(torch.einsum("b i, b i->b", cam_ray, obj_ray))  # (B,)
    # Rotate back by that amount
    if dst_type == "mat":
        allo_t = trans
        rot_aa = torch.cross(cam_ray, obj_ray, dim=1)
        rot_aa = rot_aa / torch.linalg.norm(rot_aa, dim=1, keepdim=True)
        rot_mat = pytorch3d.transforms.axis_angle_to_matrix(
            rot_aa * (-angle.unsqueeze(-1))
        )
        allo_rot = rot_mat @ ego_pose[:, :3, :3]
        allo_pose = torch.cat([allo_rot, allo_t.unsqueeze(-1)], dim=-1)  # (B, 3, 4)
        last_row = (
            torch.tensor([0, 0, 0, 1], device=allo_pose.device)
            .reshape(1, 1, 4)
            .repeat(bs, 1, 1)
        )
        allo_pose = torch.cat(
            [allo_pose, last_row],
            dim=1,
        )
    else:
        raise ValueError("dst_type should be mat or quat, got: {}".format(dst_type))
    return allo_pose


def quatmul_torch(q1, q2):
    """Computes the multiplication of two quaternions.

    Note, output dims: NxMx4 with N being the batchsize and N the number
    of quaternions or 3D points to be transformed.
    """
    # RoI dimension. Unsqueeze if not fitting.
    a = q1.unsqueeze(0) if q1.dim() == 1 else q1
    b = q2.unsqueeze(0) if q2.dim() == 1 else q2

    # Corner dimension. Unsequeeze if not fitting.
    a = a.unsqueeze(1) if a.dim() == 2 else a
    b = b.unsqueeze(1) if b.dim() == 2 else b

    # Quaternion product
    x = (
        a[:, :, 1] * b[:, :, 0]
        + a[:, :, 2] * b[:, :, 3]
        - a[:, :, 3] * b[:, :, 2]
        + a[:, :, 0] * b[:, :, 1]
    )
    y = (
        -a[:, :, 1] * b[:, :, 3]
        + a[:, :, 2] * b[:, :, 0]
        + a[:, :, 3] * b[:, :, 1]
        + a[:, :, 0] * b[:, :, 2]
    )
    z = (
        a[:, :, 1] * b[:, :, 2]
        - a[:, :, 2] * b[:, :, 1]
        + a[:, :, 3] * b[:, :, 0]
        + a[:, :, 0] * b[:, :, 3]
    )
    w = (
        -a[:, :, 1] * b[:, :, 1]
        - a[:, :, 2] * b[:, :, 2]
        - a[:, :, 3] * b[:, :, 3]
        + a[:, :, 0] * b[:, :, 0]
    )

    return torch.stack((w, x, y, z), dim=2)


def allocentric_to_egocentric_torch(translation, q_allo, eps=1e-4):
    """Given an allocentric (object-centric) pose, compute new camera-centric
    pose Since we do detection on the image plane and our kernels are
    2D-translationally invariant, we need to ensure that rendered objects
    always look identical, independent of where we render them.

    Since objects further away from the optical center undergo skewing, we try to visually correct by
    rotating back the amount between optical center ray and object centroid ray.
    Another way to solve that might be translational variance (https://arxiv.org/abs/1807.03247)
    Args:
        translation: Nx3
        q_allo: Nx4
    """

    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = torch.tensor(
        [0, 0, 1.0], dtype=translation.dtype, device=translation.device
    )  # (3,)
    obj_ray = translation / (torch.norm(translation, dim=1, keepdim=True) + eps)

    # cam_ray.dot(obj_ray), assume cam_ray: (0, 0, 1)
    angle = obj_ray[:, 2:3].acos()

    # Compute rotation between ray to object centroid and optical center ray
    axis = torch.cross(cam_ray.expand_as(obj_ray), obj_ray)
    axis = axis / (torch.norm(axis, dim=1, keepdim=True) + eps)

    # Build quaternion representing the rotation around the computed axis
    # angle-axis => quat
    q_allo_to_ego = cat(
        [
            torch.cos(angle / 2.0),
            axis[:, 0:1] * torch.sin(angle / 2.0),
            axis[:, 1:2] * torch.sin(angle / 2.0),
            axis[:, 2:3] * torch.sin(angle / 2.0),
        ],
        dim=1,
    )

    # Apply quaternion for transformation from allocentric to egocentric.
    q_ego = quatmul_torch(q_allo_to_ego, q_allo)[
        :, 0
    ]  # Remove added Corner dimension here.
    return q_ego


def allo_to_ego_mat_torch(translation, rot_allo, eps=1e-4):
    """
    Args:
        translation: Nx3
        rot_allo: Nx3x3
    """
    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = torch.tensor(
        [0, 0, 1.0], dtype=translation.dtype, device=translation.device
    )  # (3,)
    obj_ray = translation / (torch.norm(translation, dim=1, keepdim=True) + eps)

    # cam_ray.dot(obj_ray), assume cam_ray: (0, 0, 1)
    angle = obj_ray[:, 2:3].acos()

    # Compute rotation between ray to object centroid and optical center ray
    axis = torch.cross(cam_ray.expand_as(obj_ray), obj_ray, dim=1)
    axis = axis / (torch.norm(axis, dim=1, keepdim=True) + eps)

    # Build quaternion representing the rotation around the computed axis
    # angle-axis => quat
    q_allo_to_ego = cat(
        [
            torch.cos(angle / 2.0),
            axis[:, 0:1] * torch.sin(angle / 2.0),
            axis[:, 1:2] * torch.sin(angle / 2.0),
            axis[:, 2:3] * torch.sin(angle / 2.0),
        ],
        dim=1,
    )
    rot_allo_to_ego = quat2mat_torch(q_allo_to_ego)
    # Apply quaternion for transformation from allocentric to egocentric.
    rot_ego = torch.matmul(rot_allo_to_ego, rot_allo)
    return rot_ego


# For testing whether a number is close to zero
_EPS4 = np.finfo(float).eps * 4.0

_MAX_FLOAT = np.finfo(np.float32).dtype
_FLOAT_EPS = np.finfo(np.float32).eps


def my_mat2quat(mat, dtype=None):
    # https://github.com/adamlwgriffiths/Pyrr/blob/master/pyrr/quaternion.py#L92
    # http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
    # optimised "alternative version" does not produce correct results
    # see issue #42
    dtype = dtype or mat.dtype

    trace = mat[0][0] + mat[1][1] + mat[2][2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qx = (mat[2][1] - mat[1][2]) * s
        qy = (mat[0][2] - mat[2][0]) * s
        qz = (mat[1][0] - mat[0][1]) * s
        qw = 0.25 / s
    elif mat[0][0] > mat[1][1] and mat[0][0] > mat[2][2]:
        s = 2.0 * np.sqrt(1.0 + mat[0][0] - mat[1][1] - mat[2][2])
        qx = 0.25 * s
        qy = (mat[0][1] + mat[1][0]) / s
        qz = (mat[0][2] + mat[2][0]) / s
        qw = (mat[2][1] - mat[1][2]) / s
    elif mat[1][1] > mat[2][2]:
        s = 2.0 * np.sqrt(1.0 + mat[1][1] - mat[0][0] - mat[2][2])
        qx = (mat[0][1] + mat[1][0]) / s
        qy = 0.25 * s
        qz = (mat[1][2] + mat[2][1]) / s
        qw = (mat[0][2] - mat[2][0]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + mat[2][2] - mat[0][0] - mat[1][1])
        qx = (mat[0][2] + mat[2][0]) / s
        qy = (mat[1][2] + mat[2][1]) / s
        qz = 0.25 * s
        qw = (mat[1][0] - mat[0][1]) / s

    # quat = np.array([qx, qy, qz, qw], dtype=dtype)
    if qw >= 0:
        quat = np.array([qw, qx, qy, qz], dtype=dtype)
    else:
        quat = -(np.array([qw, qx, qy, qz], dtype=dtype))

    return quat


def quat_inverse(q):
    q = np.squeeze(q)
    w, x, y, z = q
    Nq = w * w + x * x + y * y + z * z
    return np.array([w, -x, -y, -z] / Nq)


def cayley(A):
    # A is skew-symmetric
    assert np.equal(-A, A.T).all()
    I = np.eye(3)
    C = np.dot(LA.inv(I - A), I + A)
    return C


def cayley_1(a, b, c):
    R = (
        1
        / (1 + a * a + b * b + c * c)
        * np.array(
            [
                [
                    1 + a * a - b * b - c * c,
                    2 * a * b - 2 * c,
                    2 * a * c + 2 * b,
                ],
                [
                    2 * a * b + 2 * c,
                    1 - a * a + b * b - c * c,
                    2 * b * c - 2 * a,
                ],
                [
                    2 * a * c - 2 * b,
                    2 * b * c + 2 * a,
                    1 - a * a - b * b + c * c,
                ],
            ]
        )
    )
    return R


def inv_cayley(C):
    I = np.eye(3)
    A = np.dot(C - I, LA.inv(I + C))
    return A


def inv_cayley_1(C):
    I = np.eye(3)
    A = np.dot(C - I, LA.inv(I + C))
    a_1 = A[2, 1]
    b_1 = A[0, 2]
    c_1 = A[1, 0]
    return a_1, b_1, c_1


def ego_to_allo_v2(rot, trans, rot_type="quat"):
    assert rot_type in ["quat", "mat"], rot_type
    x, y, z = trans
    dx = np.arctan2(x, -z)
    dy = np.arctan2(y, -z)
    # print(dx, dy)
    euler_order = "sxyz"

    if rot_type == "quat":
        quat = euler2quat(-dy, -dx, 0, axes=euler_order)
        quat = qmult(quat, rot)
        return quat, trans
    elif rot_type == "mat":
        mat = euler2mat(-dy, -dx, 0, axes=euler_order)
        mat = mat.dot(rot)
        return mat, trans
    else:
        raise ValueError("Unknown rot_type: {}, should be mat or quat".format(rot_type))


def ego_pose_to_allo_pose_v2(ego_pose, rot_type="mat"):
    assert rot_type in ["quat", "mat"], rot_type
    if rot_type == "mat":
        trans = ego_pose[:3, 3]
    else:
        trans = ego_pose[4:7]

    dx = np.arctan2(trans[0], trans[2])
    dy = np.arctan2(trans[1], trans[2])
    # print(dx, dy)
    euler_order = "sxyz"

    if rot_type == "quat":
        rot = ego_pose[:4]
        quat = euler2quat(-dy, -dx, 0, axes=euler_order)
        quat = qmult(quat, rot)
        return np.concatenate([quat, trans], axis=0)
    elif rot_type == "mat":
        rot = ego_pose[:3, :3]
        mat = euler2mat(-dy, -dx, 0, axes=euler_order)
        mat = mat.dot(rot)
        return np.hstack([mat, trans.reshape(3, 1)])
    else:
        raise ValueError("Unknown rot_type: {}, should be mat or quat".format(rot_type))


def test_ego_allo():
    ego_pose = np.zeros((3, 4), dtype=np.float32)
    ego_pose[:3, :3] = axangle2mat((1, 2, 3), 1)
    ego_pose[:3, 3] = np.array([0.4, 0.5, 0.6])
    ego_pose_q = np.zeros((7,), dtype=np.float32)
    ego_pose_q[:4] = mat2quat(ego_pose[:3, :3])
    ego_pose_q[4:7] = ego_pose[:3, 3]
    ego_poses = {"mat": ego_pose, "quat": ego_pose_q}
    rot_types = ["mat", "quat"]
    for src_type in rot_types:
        for dst_type in rot_types:
            allo_pose = egocentric_to_allocentric(
                ego_poses[src_type], src_type, dst_type
            )
            ego_pose_1 = allocentric_to_egocentric(allo_pose, dst_type, src_type)
            print(src_type, dst_type)
            print("ego_pose: ", ego_poses[src_type])
            print("allo_pose from ego_pose: ", allo_pose)
            print("ego_pose from allo_pose: ", ego_pose_1)
            print(np.allclose(ego_poses[src_type], ego_pose_1))
            print("************************")


def test_ego_to_allo_v2():
    ego_pose = np.zeros((3, 4), dtype=np.float32)
    ego_pose[:3, :3] = axangle2mat((1, 2, 3), 1)
    ego_pose[:3, 3] = np.array([0.4, 0.5, 0.6])
    ego_pose_q = np.zeros((7,), dtype=np.float32)
    ego_pose_q[:4] = mat2quat(ego_pose[:3, :3])
    ego_pose_q[4:7] = ego_pose[:3, 3]
    ego_poses = {"mat": ego_pose, "quat": ego_pose_q}
    rot_types = ["mat", "quat"]
    for src_type in rot_types:
        dst_type = src_type
        allo_pose = egocentric_to_allocentric(ego_poses[src_type], src_type, dst_type)
        ego_pose_1 = allocentric_to_egocentric(allo_pose, dst_type, src_type)
        if src_type == "mat":
            allo_pose_v2 = ego_to_allo_v2(
                ego_poses[src_type][:3, :3],
                ego_poses[src_type][:3, 3],
                rot_type=src_type,
            )
            ego_pose_1_v2 = allocentric_to_egocentric(
                np.concatenate(
                    [allo_pose_v2[0], allo_pose_v2[1].reshape(3, 1)], axis=1
                ),
                dst_type,
                src_type,
            )
        else:
            allo_pose_v2 = ego_to_allo_v2(
                ego_poses[src_type][:4],
                ego_poses[src_type][4:7],
                rot_type=src_type,
            )
            ego_pose_1_v2 = allocentric_to_egocentric(
                np.concatenate(allo_pose_v2, axis=0), dst_type, src_type
            )

        print(src_type, dst_type)
        print("ego_pose: ", ego_poses[src_type])
        print("allo_pose from ego_pose: ", allo_pose)
        print("ego_pose from allo_pose: ", ego_pose_1)
        print(np.allclose(ego_poses[src_type], ego_pose_1))

        print()
        print("allo_pose from ego_pose (v2): ", allo_pose_v2)
        print("ego_pose from allo_pose (v2): ", ego_pose_1_v2)
        print(np.allclose(ego_poses[src_type], ego_pose_1_v2))
        print()
        print("************************")


def test_mat2quat():
    for i in range(1000):
        q_rand = np.random.rand(4)
        mat = quat2mat(q_rand)

        q_t3d = mat2quat(mat)
        q_my = my_mat2quat(mat)
        print(q_t3d, q_my)
        assert np.allclose(q_my, q_t3d)


def test_egocentric_to_allocentric_torch():

    ego_pose = np.zeros((3, 4), dtype=np.float32)
    ego_pose[:3, :3] = axangle2mat((1, 2, 3), 1)
    ego_pose[:3, 3] = np.array([0.4, 0.5, 0.6])
    # numpy result
    allo_pose_np = egocentric_to_allocentric(ego_pose, src_type="mat", dst_type="mat")

    ego_pose = torch.from_numpy(ego_pose).unsqueeze(0)  # (1, 3, 4)
    allo_pose = egocentric_to_allocentric_torch(
        ego_pose, src_type="mat", dst_type="mat"
    )
    print(allo_pose, allo_pose_np)


def test_allocentric_to_egocentric_batch_torch():

    allo_pose = np.zeros((3, 4), dtype=np.float32)
    allo_pose[:3, :3] = axangle2mat((1, 2, 3), 1)
    allo_pose[:3, 3] = np.array([0.4, 0.5, 0.6])
    # numpy result
    ego_pose_np = allocentric_to_egocentric(allo_pose, src_type="mat", dst_type="mat")

    allo_pose = torch.from_numpy(allo_pose).unsqueeze(0)  # (1, 3, 4)
    ego_pose = allocentric_to_egocentric_batch_torch(
        allo_pose, src_type="mat", dst_type="mat"
    )
    print(ego_pose, ego_pose_np)


# python -m src.utils.pysixd.RT_transform
if __name__ == "__main__":
    # test_ego_to_allo_v2()
    # test_mat2quat()
    # test_egocentric_to_allocentric_torch()
    test_allocentric_to_egocentric_batch_torch()
