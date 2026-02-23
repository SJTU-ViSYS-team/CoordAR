import numpy as np
import torch
import torch.nn.functional as F
import transforms3d
from itertools import chain, permutations, product
from scipy.spatial.transform import Rotation as R


def transform_pts(T, pts):
    # T: (B, 4, 4)
    # pts: (B, n, 3)
    batch_size = T.size(0)
    pts_homogeneous = torch.cat(
        (pts, torch.ones(batch_size, pts.size(1), 1).to(T)), dim=2
    )
    transformed_pts = torch.matmul(pts_homogeneous, T.permute(0, 2, 1))
    transformed_pts = transformed_pts[:, :, :3] / transformed_pts[:, :, 3:4]
    return transformed_pts


def transform_pts_Rt(pts, R, t=None):
    """
    Args:
        pts: (B,P,3)
        R: (B,3,3)
        t: (B,3,1)

    Returns:

    """
    bs = R.shape[0]
    n_pts = pts.shape[1]
    assert pts.shape == (bs, n_pts, 3)
    if t is not None:
        assert t.shape[0] == bs

    pts_transformed = R.view(bs, 1, 3, 3) @ pts.view(bs, n_pts, 3, 1)
    if t is not None:
        pts_transformed += t.view(bs, 1, 3, 1)
    return pts_transformed.squeeze(-1)  # (B, P, 3)


def combine_R_and_T(R, T):
    # R: (B,3,3)
    # T: (B,3)
    bs = len(R)
    matrix4x4 = torch.cat([R, T.unsqueeze(-1)], dim=-1)
    matrix4x4 = torch.cat(
        [
            matrix4x4,
            torch.tensor([0, 0, 0, 1], device=R.device).view(1, 1, 4).repeat(bs, 1, 1),
        ],
        dim=-2,
    )
    return matrix4x4


def rot6d_to_mat_batch(d6):
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix.
    Args:
        d6: 6D rotation representation, of size (*, 6)
    Returns:
        batch of rotation matrices of size (*, 3, 3)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks. CVPR 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    # poses
    x_raw = d6[..., 0:3]  # bx3
    y_raw = d6[..., 3:6]  # bx3

    x = F.normalize(x_raw, p=2, dim=-1)  # bx3
    z = torch.cross(x, y_raw, dim=-1)  # bx3
    z = F.normalize(z, p=2, dim=-1)  # bx3
    y = torch.cross(z, x, dim=-1)  # bx3

    # (*,3)x3 --> (*,3,3)
    return torch.stack((x, y, z), dim=-1)  # (b,3,3)


def mat_to_rot6d_batch(rot):
    """Converts a batch of rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last col. Note that 6D representation is not unique.
    Args:
        rot: batch of rotation matrices of size (b, 3, 3)
    Returns:
        6D rotation representation, of size (b, 6)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    x = rot[..., :, 0]  # b x 3
    y = rot[..., :, 1]  # b x 3
    rot6d = torch.cat([x, y], dim=-1)  # b x 6
    return rot6d


def mat_to_rot6d_np(rot):
    """numpy version for only one matrix.
    Converts a single rotation matrix to 6D rotation representation by Zhou et al. [1]
    by dropping the last col. Note that 6D representation is not unique.
    Args:
        rot: rotation matrix of size (3, 3)
    Returns:
        6D rotation representation, of size (6,)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    x = rot[:3, 0]  # col x
    y = rot[:3, 1]  # col y
    rot6d = np.concatenate([x, y])  # (6,)
    return rot6d


rad2deg = lambda t: ((180 / np.pi) * t)
deg2rad = lambda t: ((np.pi / 180) * t)


def add_noise(TCO, euler_deg_std=[15, 15, 15], trans_std=[0.01, 0.01, 0.05]):
    TCO_out = TCO.clone()
    device = TCO_out.device
    bsz = TCO.shape[0]
    euler_noise_deg = np.concatenate(
        [
            torch.normal(
                mean=torch.zeros(bsz, 1), std=torch.full((bsz, 1), euler_deg_std_i)
            ).numpy()
            for euler_deg_std_i in euler_deg_std
        ],
        axis=1,
    )
    euler_noise_rad = deg2rad(euler_noise_deg)
    R_noise = (
        torch.tensor([transforms3d.euler.euler2mat(*xyz) for xyz in euler_noise_rad])
        .float()
        .to(device)
    )

    trans_noise = np.concatenate(
        [
            torch.normal(
                mean=torch.zeros(bsz, 1), std=torch.full((bsz, 1), trans_std_i)
            ).numpy()
            for trans_std_i in trans_std
        ],
        axis=1,
    )
    trans_noise = torch.tensor(trans_noise).float().to(device)
    TCO_out[:, :3, :3] = TCO_out[:, :3, :3] @ R_noise
    TCO_out[:, :3, 3] += trans_noise
    return TCO_out


def add_noise_with_diameter(
    TCO, diameter, euler_deg_std=[15, 15, 15], trans_std_ratio=[0.05, 0.05, 0.1]
):
    """
    Add noise to pose transformation with translation std scaled by object diameter.

    Args:
        TCO: (B, 4, 4) camera to object transformation
        diameter: (B,) or scalar, object diameter in meters
        euler_deg_std: [x, y, z] rotation noise standard deviation in degrees
        trans_std_ratio: [x, y, z] translation noise as ratio of diameter
                         Default: [0.05, 0.05, 0.1] means 5%, 5%, 10% of diameter

    Returns:
        TCO_out: (B, 4, 4) noisy transformation
    """
    TCO_out = TCO.clone()
    device = TCO_out.device
    bsz = TCO.shape[0]

    # Handle diameter input
    if isinstance(diameter, (int, float)):
        diameter_tensor = torch.full((bsz,), diameter, device=device)
    elif isinstance(diameter, np.ndarray):
        diameter_tensor = torch.from_numpy(diameter).float().to(device)
    else:
        diameter_tensor = diameter.float().to(device)

    if diameter_tensor.dim() == 0:
        diameter_tensor = diameter_tensor.unsqueeze(0).expand(bsz)

    # Add rotation noise
    euler_noise_deg = np.concatenate(
        [
            torch.normal(
                mean=torch.zeros(bsz, 1), std=torch.full((bsz, 1), euler_deg_std_i)
            ).numpy()
            for euler_deg_std_i in euler_deg_std
        ],
        axis=1,
    )
    euler_noise_rad = deg2rad(euler_noise_deg)
    R_noise = (
        torch.tensor([transforms3d.euler.euler2mat(*xyz) for xyz in euler_noise_rad])
        .float()
        .to(device)
    )

    # Add translation noise scaled by diameter
    # trans_std = diameter * trans_std_ratio for each axis
    trans_noise = torch.stack(
        [
            torch.normal(
                mean=torch.zeros(bsz, device=device),
                std=diameter_tensor * trans_std_ratio_i,
            )
            for trans_std_ratio_i in trans_std_ratio
        ],
        dim=1,
    )  # (B, 3)

    TCO_out[:, :3, :3] = TCO_out[:, :3, :3] @ R_noise
    TCO_out[:, :3, 3] += trans_noise
    return TCO_out


def get_perturbations(TCO: torch.Tensor, extra_views=False) -> torch.Tensor:
    params = product(permutations("XYZ", 1), [-45 / 2, 45 / 2])
    if extra_views:
        params = chain(params, product(permutations("XYZ", 1), [-45, 45]))
    perturbations = torch.eye(4, device=TCO.device).tile(1, 7 + extra_views * 6, 1, 1)
    perturbations[0, 1:, :3, :3] = torch.stack(
        [
            torch.from_numpy(R.from_euler("".join(a), s, degrees=True).as_matrix())
            for a, s in params
        ],
        dim=0,
    )
    return TCO.unsqueeze(1) @ perturbations
