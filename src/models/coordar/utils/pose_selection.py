from einops import einsum
import ipdb
from matplotlib import pyplot as plt
import numpy as torch
import torch
from torchvision.utils import make_grid

from src.models.coordar.utils.metrics import calc_mssd_recall


def uv_grid(bs, height, width, device=None):
    if device is None:
        device = torch.device("cpu")
    u, v = torch.meshgrid(
        torch.arange(width, device=device),
        torch.arange(height, device=device),
        indexing="xy",
    )
    u = u.reshape(1, height, width).repeat(bs, 1, 1)
    v = v.reshape(1, height, width).repeat(bs, 1, 1)
    return u, v


def reproject_depth(
    query_depth, query_mask, relative_pose, K, template_K, template_depth
):
    # query_depth: (b, h, w)
    # query_mask: (b, h, w)
    # relative_pose: (b, v, 4, 4)
    # K: (b, 3, 3)
    # template_K: (b, v, 3, 3)
    bs, height, width = query_depth.shape
    n_views = relative_pose.shape[1]
    device = query_depth.device
    u, v = uv_grid(bs, height, width, device=device)
    K = K[..., None, None]
    points_cam = torch.stack(
        [
            (u - K[:, 0, 2]) * query_depth / K[:, 0, 0],
            (v - K[:, 1, 2]) * query_depth / K[:, 1, 1],
            query_depth,
            torch.ones_like(u),
        ],
        dim=-1,
    ).reshape(bs, -1, 4)

    W_Q = einsum(relative_pose, points_cam, "b v m n, b p n -> b v p m")
    points_img = einsum(template_K, W_Q[..., :3], "b v m n, b v p n -> b v p m")
    points_img[..., :2] /= points_img[..., 2:3].clamp(min=1e-6)

    UV = points_img[..., :2]  # (bs, n_views, height * width, 2)
    reprojected_depth = points_img[..., 2]  # (bs, n_views, height * width)
    depth_matched = (reprojected_depth - template_depth.flatten(2, 3)).abs() < 0.015
    reprojected_mask = torch.zeros((bs, n_views, height * width)).to(query_depth.device)
    valid = (
        (UV[..., 0] >= 0)
        & (UV[..., 0] < width)
        & (UV[..., 1] >= 0)
        & (UV[..., 1] < height)
    )
    valid = valid & (query_mask.unsqueeze(1).repeat(1, n_views, 1, 1).flatten(2, 3) > 0)
    # scatter
    u_idx = UV[..., 0].long().clamp(0, width - 1)
    v_idx = UV[..., 1].long().clamp(0, height - 1)
    reprojected_mask.scatter_(
        dim=-1, index=v_idx * width + u_idx, src=valid.float()  # 展平后的索引
    )
    reprojected_mask = reprojected_mask.reshape(bs, n_views, height, width)

    return reprojected_mask


def compute_IoU(mask_rproj, mask_ref, agg="mean"):
    # mask_rproj: (b, v, k, h, w)
    # mask_ref: (b, v, h, w)
    bs = mask_rproj.shape[0]
    n_views = mask_rproj.shape[1]
    mask_ref = mask_ref.unsqueeze(2)
    intersection = torch.logical_and(mask_rproj, mask_ref).sum([3, 4])
    union = torch.logical_or(mask_rproj, mask_ref).sum([3, 4])
    iou = intersection / union.clip(min=1e-6)
    if agg == "mean":
        miou = iou.mean(dim=1)  # average over views
    elif agg == "max":
        miou = iou.max(dim=1).values
    else:
        raise ValueError(f"Unsupported aggregation method: {agg}")
    return miou


def compute_AP(mask_rproj, mask_ref):
    # mask_rproj: (b, v, k, h, w)
    # mask_ref: (b, v, h, w)
    bs = mask_rproj.shape[0]
    n_views = mask_rproj.shape[1]
    mask_ref = mask_ref.unsqueeze(2)
    intersection = torch.logical_and(mask_rproj, mask_ref).sum([3, 4])
    total = mask_rproj.sum([3, 4])
    precision = intersection / total.clip(min=1e-6)
    AP = precision.mean(dim=1)  # average over views
    return AP


def pose_selection(batch, split_batches, out_dicts, method="IoU"):
    device = split_batches[0]["query_TCO"].device
    sample_count = len(split_batches[0]["query_TCO"])
    num_views = len(out_dicts)
    assert len(split_batches) == len(out_dicts)
    pred_tco = torch.stack([out["tco_pred"] for out in out_dicts], 1)

    if method in ["IoU", "AP"]:
        relative_pose = batch["template_tco"].unsqueeze(2) @ torch.linalg.inv(
            pred_tco
        ).unsqueeze(1)
        reproj_mask = reproject_depth(
            batch["query_depth"],
            batch["query_mask"],
            relative_pose.flatten(1, 2),
            batch["query_K_crop"],
            batch["template_K_crop"].repeat_interleave(num_views, dim=1),
            batch["template_depths"].repeat_interleave(num_views, dim=1),
        ).unflatten(1, (num_views, num_views))
        if method == "IoU":
            miou = compute_IoU(reproj_mask, batch["template_masks_visib"], agg="mean")
        elif method == "AP":
            miou = compute_AP(reproj_mask, batch["template_masks_visib"])
        else:
            raise ValueError(f"Unsupported method: {method}")
        best = torch.argmax(miou, dim=1)
    elif method == "best":
        expand = lambda x: x.repeat_interleave(num_views, dim=0)
        mssd_recall = calc_mssd_recall(
            pred_tco.flatten(0, 1),
            expand(batch["query_TCO"]),
            expand(batch["points"]),
            expand(batch["symmetries"]),
            expand(batch["diameter"]),
        ).unflatten(0, (sample_count, num_views))
        best = torch.argmax(mssd_recall, dim=1)

    best_out = []
    keys = out_dicts[0].keys()
    for sample_idx in range(sample_count):
        sample = {}
        for key in keys:
            value = out_dicts[best[sample_idx].item()][key]
            if isinstance(value, torch.Tensor):
                if value.ndim > 0 and value.shape[0] == sample_count:
                    value = value[sample_idx]
            else:
                raise ValueError(f"Unsupported type {type(value)} for key {key}")
            sample[key] = value
        best_out.append(sample)
    best_outdict = {key: torch.stack([out[key] for out in best_out]) for key in keys}

    return best_outdict
