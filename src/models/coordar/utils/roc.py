from einops import rearrange
import ipdb
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

resize_as = lambda x, y: F.interpolate(
    x.float(),
    size=y.shape[2:],
    mode="nearest",
)


def depth_to_point_map(depth, K):
    """
    Convert depth map to point map.
    Args:
        depth: (B, H, W) tensor of depth values.
        K: (B, 3, 3) tensor of camera intrinsic matrix.
    Returns:
        point_map: (B, H, W, 3) tensor of 3D points in camera coordinates.
    """
    B, H, W = depth.shape
    u = torch.arange(W, device=depth.device).view(1, 1, W).expand(B, H, W)
    v = torch.arange(H, device=depth.device).view(1, H, 1).expand(B, H, W)
    K = K[..., None, None]

    x = (u - K[:, 0, 2]) * depth / K[:, 0, 0]
    y = (v - K[:, 1, 2]) * depth / K[:, 1, 1]
    z = depth

    point_map = torch.stack([x, y, z], dim=-1)
    return point_map


def solve_scale_translation(x_w, x_r, mask, threshold=0.03, refine=True):
    # x_w: (b, n, 3), x_r: (b, n, 3)
    # mask: (b, n)
    b, n, _ = x_w.shape
    eps = 1e-3

    # Compute means
    x_w_mean = (
        (x_w * mask.unsqueeze(-1)).sum(1) / (mask.sum(1, keepdim=True) + eps)
    ).unsqueeze(
        1
    )  # (b, 1, 3)
    x_r_mean = (
        (x_r * mask.unsqueeze(-1)).sum(1) / (mask.sum(1, keepdim=True) + eps)
    ).unsqueeze(
        1
    )  # (b, 1, 3)

    # Compute A and B
    A = x_w - x_w_mean  # (b, n, 3)
    B = x_r - x_r_mean  # (b, n, 3)

    # Solve for s, s.t. A * s = B
    s = ((B / A) * mask.unsqueeze(-1)).mean(-1).sum(1) / (mask.sum(1) + eps)  # (b)

    # Solve for t
    t = x_w_mean - x_r_mean / s.reshape(b, 1, 1)
    t = t.squeeze(1)  # (b, 3)

    # error
    error = x_w - (x_r / s.view(b, 1, 1) + t.unsqueeze(1))
    error = error.norm(dim=-1)  # (b, n)
    accurate_mask = error < threshold
    mask = mask * accurate_mask
    if refine:
        s, t = solve_scale_translation(
            x_w, x_r, mask, threshold=threshold, refine=False
        )

    return s, t


def fit_pose_least_squere(roc_hat, depth, mask, K, diameter):
    # roc_hat: [B, 3, H, W]
    # depth: [B, H, W]
    # mask: [B, H, W]
    B, C, H, W = roc_hat.shape

    roc_hat = rearrange(roc_hat, "b c h w -> b (h w) c")

    point_map = depth_to_point_map(depth, K)
    point_map = rearrange(point_map, "b h w c -> b (h w) c")

    mask = rearrange(mask, "b h w -> b (h w)")
    s_hat, t_hat = solve_scale_translation(point_map, roc_hat, mask)

    point_map_hat = roc_hat / s_hat.view(B, 1, 1) + t_hat.unsqueeze(1)
    depth_hat = point_map_hat[..., 2]

    point_map_hat = rearrange(point_map_hat, "b (h w) c -> b h w c", h=H, w=W)
    depth_hat = rearrange(depth_hat, "b (h w) -> b h w", h=H, w=W)

    return t_hat, depth_hat, point_map_hat


def test_roc_fit_least_square():

    data = np.load(
        "logs/predict/runs/2025-06-25_13-00-52/vis/predict_vis/000000/data.npz"
    )
    roc_hat = torch.tensor(data["pred_roc"]).float()
    depth = torch.tensor(data["depth"]).float()
    mask = torch.tensor(data["mask"]).float()
    pred_mask = (torch.tensor(data["pred_mask"]) > 0.5).float()
    diameter = torch.tensor(data["diameter"]).float()
    K = torch.tensor(data["K"]).float()
    pose = torch.tensor(data["pose"]).float()

    pred_mask = resize_as(pred_mask.unsqueeze(1), depth.unsqueeze(1)).squeeze(1)
    roc_hat = resize_as(roc_hat, depth.unsqueeze(1))
    fit_mask = pred_mask * (depth > 0.001)

    t_hat, depth_hat, point_map_hat = fit_pose_least_squere(
        roc_hat, depth, fit_mask, K, diameter
    )

    print("t_hat:", t_hat)
    print("pose:", pose[:, :3, 3])

    # show the result
    bs = roc_hat.shape[0]
    fig, axs = plt.subplots(bs, 6, figsize=(15, 15))
    for i in range(bs):
        axs[i, 0].imshow(depth[i].cpu().numpy(), cmap="gray")
        axs[i, 0].set_title("Depth")
        axs[i, 1].imshow(mask[i].cpu().numpy(), cmap="gray")
        axs[i, 1].set_title("Mask")
        axs[i, 2].imshow(roc_hat[i].permute(1, 2, 0).cpu().numpy())
        axs[i, 2].set_title("ROC Hat")
        axs[i, 3].imshow(pred_mask[i].cpu().numpy(), cmap="gray")
        axs[i, 3].set_title("Pred Mask")
        axs[i, 4].imshow(depth_hat[i].cpu().numpy(), cmap="gray")
        axs[i, 4].set_title("Depth Hat")
        # depth error
        axs[i, 5].imshow(
            ((depth_hat[i] - depth[i]) * pred_mask[i]).abs().cpu().numpy(),
            cmap="jet",
            vmin=0,
            vmax=0.015,
        )
        axs[i, 5].set_title("Depth Error")

    for ax in axs.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# python -m src.models.coordar.utils.roc
if __name__ == "__main__":
    test_roc_fit_least_square()
