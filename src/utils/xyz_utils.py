import torch
from scipy.spatial.distance import cdist

from src.utils.transform_tensor import transform_pts


def calc_xyz_bp_batch(depth, R, T, K, fmt="BHWC"):
    """
    Args:
        depth: BxHxW rendered depth
        R: Bx3x3
        T: Bx3
        K: Bx3x3
    -------
        xyz: (B,3,H,W)
    """

    assert depth.ndim == 3, depth.shape
    bs, height, width = depth.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    X = grid_x.expand(bs, height, width) - K[:, 0, 2].view(bs, 1, 1)
    Y = grid_y.expand(bs, height, width) - K[:, 1, 2].view(bs, 1, 1)

    if fmt == "BHWC":
        xyz_cam = torch.stack(
            (
                X * depth / K[:, 0, 0].view(bs, 1, 1),
                Y * depth / K[:, 1, 1].view(bs, 1, 1),
                depth,
            ),
            dim=-1,
        )
        xyz_cam = xyz_cam.view(bs, height, width, 3, 1)
        Rinv_expand = (
            R.permute(0, 2, 1).view(bs, 1, 1, 3, 3).expand(bs, height, width, 3, 3)
        )
        T_expand = T.view(bs, 1, 1, 3, 1).expand(bs, height, width, 3, 1)
        mask = (depth != 0).to(depth).view(bs, height, width, 1)
        # xyz = torch.matmul(Rinv_expand, xyz_cam - T_expand).squeeze() * mask
        xyz = torch.einsum("bhwij,bhwjk->bhwi", Rinv_expand, xyz_cam - T_expand) * mask
    else:  # BCHW
        xyz_cam = torch.stack(
            (
                X * depth / K[:, 0, 0].view(bs, 1, 1),
                Y * depth / K[:, 1, 1].view(bs, 1, 1),
                depth,
            ),
            dim=-3,
        )
        xyz_cam = xyz_cam.view(bs, 3, 1, height, width)
        Rinv_expand = (
            R.permute(0, 2, 1).view(bs, 3, 3, 1, 1).expand(bs, 3, 3, height, width)
        )
        T_expand = T.view(bs, 3, 1, 1, 1).expand(bs, 3, 1, height, width)
        mask = (depth != 0).to(depth).view(bs, 1, height, width)
        xyz = torch.einsum("bijhw,bjkhw->bihw", Rinv_expand, xyz_cam - T_expand) * mask

    return xyz


def xyz_to_region_batch(xyz, fps_points, mask=None):
    """
    Args:
        xyz: (b,h,w,3)
        fps_points: (b,f,3)
    Returns:
        (b,h,w) 1 to num_fps, 0 is bg
    """
    assert xyz.shape[-1] == 3 and xyz.ndim == 4, xyz.shape
    assert fps_points.shape[-1] == 3 and fps_points.ndim == 3, fps_points.shape
    bs, h, w = xyz.shape[:3]

    if mask is None:
        mask = torch.ones((bs, h, w), dtype=torch.float32, device=xyz.device)

    dists = torch.cdist(xyz.view(bs, -1, 3), fps_points, p=2)  # b,hw,f
    region = dists.argmin(-1).view(bs, h, w) + 1  # NOTE: 1 to num_fps
    # b,h,w
    return (region * mask).to(torch.long)


def calc_sym_targets_batch(
    symmetries, xyz_patch, model_extent, model_center, mask, fps_points, black_pixels
):
    # symmetries: (b, n_sym, 4, 4)
    # xyz_patch: (b, h, w, 3)
    # model_extent: (b, 3)
    # model_center: (b, 3)
    # mask: (b, h, w)
    # black_pixels: (b, h, w)
    # fps_points: (b, f, 3)
    bs, n_sym = symmetries.shape[:2]
    h, w = mask.shape[1:]
    symmetries = symmetries.flatten(0, 1)
    # symmetries: (b*n_sym, 4, 4)
    xyz_patch = (
        xyz_patch.unsqueeze(1)
        .expand(-1, n_sym, -1, -1, -1)
        .reshape(bs * n_sym, h * w, 3)
    )
    # xyz_patch: (b*n_sym, h*w,  3)
    # transform xyz
    symmetries_inv = torch.linalg.inv(symmetries)
    xyz_patch = transform_pts(symmetries_inv, xyz_patch)

    # normalize xyz
    model_center = (
        model_center.unsqueeze(1).repeat(1, n_sym, 1).flatten(0, 1).unsqueeze(1)
    )
    model_extent = (
        model_extent.unsqueeze(1).repeat(1, n_sym, 1).flatten(0, 1).unsqueeze(1)
    )
    # model_center: (b*n_sym, 1, 3)
    new_center = transform_pts(symmetries_inv, model_center)
    nocs_patch = ((xyz_patch - new_center) / model_extent).float() + 0.5
    # nocs_patch: (b*n_sym, h*w,  3)

    fps_points = fps_points.unsqueeze(1).expand(bs, n_sym, -1, -1).flatten(0, 1)
    # fps_points: (b*n_sym, f, 3)
    mask = mask.unsqueeze(1).expand(-1, n_sym, -1, -1).flatten(0, 1)

    xyz_patch = xyz_patch.reshape(bs * n_sym, h, w, 3)
    region_patch = xyz_to_region_batch(xyz_patch, fps_points, mask=mask)

    nocs_patch = nocs_patch.view(bs, n_sym, h, w, 3)
    region_patch = region_patch.view(bs, n_sym, h, w)
    xyz_patch = xyz_patch.view(bs, n_sym, h, w, 3).clone()
    black_pixels = black_pixels.unsqueeze(1).expand(-1, n_sym, -1, -1)
    nocs_patch[black_pixels] = 0.5
    region_patch[black_pixels] = 0

    return nocs_patch, region_patch, xyz_patch


def normalize_xyz(xyz, model_center, model_extent, mask=None):
    # xyz: (b, h, w, 3)
    # mask: (b, h, w)
    # model_center: (b, 3)
    # model_extent: (b, 3)
    xyz_norm = (xyz - model_center[:, None, None, :]) / model_extent[
        :, None, None, :
    ] + 0.5
    if mask is None:
        return xyz_norm
    bg_pixels = mask.unsqueeze(3).repeat(1, 1, 1, 3) < 0.001
    xyz_norm[bg_pixels] = 0.5
    return xyz_norm


def denormalize_xyz(xyz, model_center, model_extent, format="BHWC"):
    # xyz: (b, h, w, 3)
    # model_center: (b, 3)
    # model_extent: (b, 3)
    if format == "BHWC":
        xyz_dnorm = (xyz - 0.5) * model_extent[:, None, None, :] + model_center[
            :, None, None, :
        ]
    elif format == "BCHW":
        xyz_dnorm = (xyz - 0.5) * model_extent[:, :, None, None] + model_center[
            :, :, None, None
        ]
    else:
        raise ValueError("Invalid format")
    return xyz_dnorm
