import os
import cv2
from einops import rearrange
import imageio
import ipdb
from matplotlib import pyplot as plt
import numpy as np
import skimage
import torch
from torchvision.utils import make_grid
import torch.nn.functional as F

from src.utils.misc import prepare_dir
from src.utils.vis.pose import (
    draw_projected_box3d,
    draw_projected_box3d_xyz,
    get_bbox3d_from_center_ext,
    points_to_2D,
)
from src.utils.point_vis import show_multiple_point_clouds_plotly

resize_as = lambda x, y: F.interpolate(
    x.float(),
    size=y.shape[2:],
    mode="nearest",
)


def normalize_depth_bp_torch(depth_bp):
    max_value = torch.max(depth_bp.flatten(2, 3), dim=2)[0].view(-1, 3, 1, 1)
    min_value = torch.min(depth_bp.flatten(2, 3), dim=2)[0].view(-1, 3, 1, 1)
    depth_bp = (depth_bp - min_value) / (max_value - min_value + 1e-6)
    return depth_bp


def show_batch(batch, log_dir="logs/debug"):

    template_imgs = batch["template_imgs_raw"]
    template_rel_rocs = batch["template_rel_rocs"]
    template_rel_rocs = batch["template_rel_rocs"]
    template_rocs = batch["template_rocs"]
    template_depths = batch["template_depths"]
    template_masks = batch["template_masks_visib"]
    bs = len(template_imgs)
    template_depth_bp = batch["template_depth_bp"]
    query = batch["query_raw"]
    query_rel_roc = batch["query_rel_roc"]
    query_roc = batch["query_roc"]
    query_roc_loc = batch["query_roc_loc"].cpu().numpy()
    query_depth = batch["query_depth"]
    query_depth_bp = batch["query_depth_bp"]
    query_rel_roc = batch["query_rel_roc"]
    query_mask = batch["query_mask"]
    diameter = batch["diameter"].tolist()
    template_obj_size = batch["template_obj_size"].tolist()

    bs = len(template_imgs)

    fig, axs = plt.subplots(bs, 14, figsize=(14 * 2, bs * 2))
    if bs == 1:
        axs = [axs]
    individual_image_dir = os.path.join(log_dir, "individual_images")
    prepare_dir(individual_image_dir)
    for i in range(bs):
        scene_id = batch["scene_id"][i].item()
        im_id = batch["im_id"][i].item()
        gt_id = batch["gt_id"][i].item()
        obj_id = batch["obj_id"][i].item()
        axs[i][0].imshow(make_grid(template_imgs[i]).permute(1, 2, 0).cpu().numpy())
        axs[i][0].set_title(
            f"template_img, diameter: {template_obj_size[i][0]*1000:.1f}mm"
        )
        skimage.io.imsave(
            os.path.join(
                individual_image_dir,
                f"{i:06d}_template_img.png",
            ),
            make_grid(template_imgs[i]).permute(1, 2, 0).cpu().numpy(),
        )
        axs[i][1].imshow(template_depths[i][0].cpu().numpy(), cmap="gray")
        axs[i][1].set_title("template_depth")
        axs[i][2].imshow(query[i].permute(1, 2, 0).cpu().numpy())
        axs[i][2].set_title(f"query: {scene_id}_{im_id}_{obj_id}_{gt_id}")
        skimage.io.imsave(
            os.path.join(
                individual_image_dir,
                f"{i:06d}_query.png",
            ),
            query[i].permute(1, 2, 0).cpu().numpy(),
        )
        axs[i][3].imshow(query_depth[i].cpu().numpy(), cmap="gray")
        axs[i][3].set_title(f"query_depth, diameter:{diameter[i]*1000:.1f}")
        axs[i][4].imshow(
            make_grid(template_rel_rocs[i]).permute(1, 2, 0).clip(0, 1).cpu().numpy()
        )
        axs[i][4].set_title("template_rel_rocs")
        axs[i][5].imshow(
            make_grid(template_rocs[i]).permute(1, 2, 0).clip(0, 1).cpu().numpy()
        )
        axs[i][5].set_title("template_rocs")
        skimage.io.imsave(
            os.path.join(
                individual_image_dir,
                f"{i:06d}_template_rocs.png",
            ),
            (
                make_grid(template_rocs[i]).permute(1, 2, 0).clip(0, 1).cpu().numpy()
                * 255
            ).astype(np.uint8),
        )

        axs[i][6].imshow(
            make_grid(template_rel_rocs[i]).permute(1, 2, 0).clip(0, 1).cpu().numpy()
        )
        axs[i][6].set_title("template_rel_rocs")

        axs[i][7].imshow(query_rel_roc[i].permute(1, 2, 0).clip(0, 1).cpu().numpy())
        axs[i][7].set_title("query_rel_roc")
        axs[i][8].imshow(query_roc[i].permute(1, 2, 0).clip(0, 1).cpu().numpy())
        axs[i][8].set_title("query_roc")
        axs[i][8].scatter(
            query_roc_loc[i][:, 0],
            query_roc_loc[i][:, 1],
            s=0.01,
            c="b",
        )

        axs[i][9].imshow(query_rel_roc[i].permute(1, 2, 0).clip(0, 1).cpu().numpy())
        axs[i][9].set_title("query_rel_roc")

        axs[i][10].imshow(template_masks[i][0].cpu().numpy(), cmap="grey")
        axs[i][10].set_title("template_mask")
        axs[i][11].imshow(query_mask[i].cpu().numpy(), cmap="grey")
        axs[i][11].set_title("query_mask")

        axs[i][12].imshow(query_depth_bp[i].permute(1, 2, 0).cpu().numpy())
        axs[i][12].set_title("query_depth_bp")

        axs[i][13].imshow(
            make_grid(template_depth_bp[i]).permute(1, 2, 0).clip(0, 1).cpu().numpy()
        )
        axs[i][13].set_title("template_depth_bp")

        for j in range(14):
            axs[i][j].axis("off")
    plt.tight_layout()
    plt.savefig(f"{log_dir}/batch.png")
    plt.close()



def show_pred(batch, pred, log_dir):
    get_pred = lambda name, placeholder: (
        pred[name].float() if name in pred else torch.zeros_like(placeholder)
    )
    query = batch["query"]
    query_rel_roc = batch["query_rel_roc"]
    query_roc = batch["query_roc"]
    query_mask = batch["query_mask"]
    pred_roc = get_pred("roc", batch["query_rel_roc"])
    pred_mask = get_pred("mask", batch["query_mask"]).sigmoid()
    pred_geo = get_pred("geo", batch["query_rel_roc"])
    pred_geo_recon = get_pred("geo_recon", batch["query_rel_roc"])[:, :3]

    individual_image_dir = os.path.join(log_dir, "individual_images")
    prepare_dir(individual_image_dir)

    bs = len(query)
    fig, axs = plt.subplots(bs, 9, figsize=(20, 3 * bs))
    if bs == 1:
        axs = [axs]
    for i in range(bs):
        axs[i][0].imshow(query[i].permute(1, 2, 0).cpu().numpy())
        axs[i][0].set_title("query")
        axs[i][1].imshow(query_rel_roc[i].permute(1, 2, 0).clip(0, 1).cpu().numpy())
        axs[i][1].set_title("query_rel_roc")
        skimage.io.imsave(
            os.path.join(
                individual_image_dir,
                f"{i:06d}_query_gt_roc.png",
            ),
            (query_rel_roc[i].permute(1, 2, 0).clip(0, 1).cpu().numpy() * 255).astype(
                np.uint8
            ),
        )
        axs[i][2].imshow(query_roc[i].permute(1, 2, 0).clip(0, 1).cpu().numpy())
        axs[i][2].set_title("query_roc")
        axs[i][3].imshow(query_mask[i].cpu().numpy())
        axs[i][3].set_title("query_mask")
        axs[i][4].imshow(pred_geo[i].permute(1, 2, 0).clip(0, 1).cpu().numpy())
        axs[i][4].set_title("pred_geo")
        axs[i][5].imshow(pred_roc[i].permute(1, 2, 0).clip(0, 1).cpu().numpy())
        axs[i][5].set_title("pred_roc")
        axs[i][6].imshow(pred_mask[i].cpu().numpy())
        axs[i][6].set_title("pred_mask")
        axs[i][7].imshow(pred_geo_recon[i].permute(1, 2, 0).clip(0, 1).cpu().numpy())
        axs[i][7].set_title("geo_recon")
        axs[i][8].set_title("pose_geo")

        for j in range(5):
            axs[i][j].axis("off")
    plt.tight_layout()
    plt.savefig(f"{log_dir}/pred.png")
    plt.close()


def show_digit(batch, pred, log_dir):
    if "digit" not in pred:
        return
    get_pred = lambda name, placeholder: (
        pred[name].float() if name in pred else torch.zeros_like(placeholder)
    )
    query = batch["query"]
    pred_digit = pred["digit"]
    query_rel_roc = batch["query_rel_roc"]
    query_mask = batch["query_mask"]
    digit_vis = pred_digit.softmax(dim=1).max(dim=1)[0]
    pred_nocs = get_pred("nocs", batch["query_rel_roc"])

    order_mask = batch.get(
        "order_mask",
        digit_vis
        * resize_as(query_mask.unsqueeze(1), digit_vis.unsqueeze(1)).squeeze(1),
    )

    bs = len(query)
    fig, axs = plt.subplots(bs, 5, figsize=(20, 3 * bs))
    if bs == 1:
        axs = [axs]
    for i in range(bs):
        axs[i][0].imshow(query[i].permute(1, 2, 0).cpu().numpy())
        axs[i][0].set_title("query")
        axs[i][1].imshow(query_rel_roc[i].permute(1, 2, 0).clip(0, 1).cpu().numpy())
        axs[i][1].set_title("query_rel_roc")
        axs[i][2].imshow(pred_nocs[i].permute(1, 2, 0).clip(0, 1).cpu().numpy())
        axs[i][2].set_title("pred_nocs")
        axs[i][3].imshow(digit_vis[i].cpu().numpy())
        axs[i][3].set_title("digit confidence")
        axs[i][4].imshow(order_mask[i].cpu().numpy())
        axs[i][4].set_title("order_mask")

        for j in range(3):
            axs[i][j].axis("off")
    plt.tight_layout()
    plt.savefig(f"{log_dir}/digit.png")
    plt.close()


def show_pose(batch, pred, log_dir, mssd_recall):
    if "tco_pred" not in pred:
        return
    get_pred = lambda name, placeholder: (
        pred[name].float() if name in pred else torch.zeros_like(placeholder)
    )
    query = (
        batch["query_raw"]
        .permute(0, 2, 3, 1)
        .cpu()
        .contiguous()
        .numpy()
        .astype(np.uint8)
    )
    query_rel_roc = batch["query_rel_roc"]
    pred_geo = (
        (get_pred("geo", batch["query_rel_roc"]).clip(0, 1) * 255)
        .permute(0, 2, 3, 1)
        .contiguous()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )
    extents = batch["extents"].cpu().numpy()
    TCO = batch["query_TCO"].cpu().numpy()
    depth = batch["query_depth"].cpu().numpy()
    K_crop = batch["query_K_crop"].cpu().numpy()
    tco_pred = pred["tco_pred"].cpu().numpy()
    pred_mask = get_pred("mask", batch["query_mask"]).sigmoid()
    pred_depth = get_pred("depth", batch["query_depth"]).cpu().numpy()
    template_rel_rocs = batch["template_rel_rocs"]
    procruste_mask = get_pred("procrustes_mask", batch["query_mask"])
    obj_id = batch["obj_id"].cpu().numpy()

    individual_image_dir = os.path.join(log_dir, "individual_images")
    prepare_dir(individual_image_dir)

    bs = len(query)
    fig, axs = plt.subplots(bs, 9, figsize=(3 * 9, 3 * bs))
    if bs == 1:
        axs = [axs]
    for i in range(bs):
        bbox3d = get_bbox3d_from_center_ext(np.zeros(3), extents[i])
        corners_2d, _ = points_to_2D(bbox3d, TCO[i, :3, :3], TCO[i, :3, 3], K_crop[i])
        rgb_with_box = draw_projected_box3d_xyz(
            query[i].copy(),
            corners_2d,
            thickness=5,
        )
        axs[i][0].imshow(rgb_with_box)
        axs[i][0].set_title(f"query, obj_id: {obj_id[i]}")
        skimage.io.imsave(
            os.path.join(
                individual_image_dir,
                f"{i:06d}_rgb_with_gt_pose.png",
            ),
            rgb_with_box,
        )

        axs[i][1].imshow(depth[i], cmap="gray")
        axs[i][1].set_title("depth")

        axs[i][2].imshow(
            make_grid(template_rel_rocs[i]).permute(1, 2, 0).clip(0, 1).cpu().numpy()
        )
        axs[i][2].set_title("template_rel_rocs")

        axs[i][3].imshow(query_rel_roc[i].permute(1, 2, 0).clip(0, 1).cpu().numpy())
        axs[i][3].set_title("query_rel_roc")

        axs[i][4].imshow(pred_mask[i].cpu().numpy())
        axs[i][4].set_title("pred_mask")

        axs[i][5].imshow(procruste_mask[i].clip(0, 1).cpu().numpy())
        axs[i][5].set_title("procruste_mask")

        axs[i][6].imshow(pred_depth[i], cmap="gray")
        axs[i][6].set_title("pred_depth")

        corners_2d, _ = points_to_2D(
            bbox3d, tco_pred[i, :3, :3], tco_pred[i, :3, 3], K_crop[i]
        )
        scale = pred_geo[i].shape[0] / query.shape[1]
        roc_with_box = draw_projected_box3d_xyz(
            pred_geo[i].copy(),
            corners_2d * scale,
            thickness=5,
        )
        rgb_with_box = draw_projected_box3d_xyz(
            query[i].copy(),
            corners_2d * scale,
            thickness=5,
        )
        axs[i][7].imshow(roc_with_box)
        skimage.io.imsave(
            os.path.join(
                individual_image_dir,
                f"{i:06d}_pred_roc_with_pose.png",
            ),
            roc_with_box,
        )
        skimage.io.imsave(
            os.path.join(
                individual_image_dir,
                f"{i:06d}_pred_roc.png",
            ),
            pred_geo[i],
        )
        skimage.io.imsave(
            os.path.join(
                individual_image_dir,
                f"{i:06d}_pred_rgb_with_pose.png",
            ),
            rgb_with_box,
        )
        axs[i][8].imshow(roc_with_box)
        if mssd_recall is not None:
            axs[i][8].set_title(f"mssd_recall: {mssd_recall[i]:.2f}")

        for j in range(9):
            axs[i][j].axis("off")
    plt.tight_layout()
    plt.savefig(f"{log_dir}/pose.png")
    plt.close()
    # plt.show()


def save_numpy_arrays_as_gif(arrays, output_path, duration=0.1, loop=None):
    """
    使用 imageio 将 NumPy 数组列表保存为 GIF 文件

    参数:
        arrays (list): NumPy 数组列表 (形状为 [H,W] 或 [H,W,3] 或 [H,W,4])
        output_path (str): 输出的 GIF 文件路径
        duration (float): 每帧显示时间(秒)，默认为 0.1 秒
        loop (int): 循环次数，0 表示无限循环，默认为 0
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 确保数组是 uint8 类型
    processed_arrays = []
    for array in arrays:
        if array.dtype != np.uint8:
            array = (array.clip(0, 1) * 255).astype(np.uint8)
        processed_arrays.append(array)

    # 保存为 GIF
    imageio.mimsave(output_path, processed_arrays, duration=duration, loop=loop)


def save_ar_process(arrays, output_dir):
    prepare_dir(output_dir)
    for i, array in enumerate(arrays):
        # 确保数组是 uint8 类型
        if array.dtype != np.uint8:
            array = (array.clip(0, 1) * 255).astype(np.uint8)
        imageio.imwrite(
            os.path.join(output_dir, f"steps_{i:03d}.png"),
            array,
        )


def show_ar(batch, pred, log_dir):
    if "geo_steps" not in pred:
        return
    geo_steps = pred["geo_steps"]
    mask_steps = pred.get("mask_steps", torch.zeros_like(geo_steps[:, :, 0]))
    confidence_steps = pred["confidence_steps"]
    bs = len(geo_steps)
    geo_steps = rearrange(geo_steps, "b t c h w -> t c (b h) w")
    mask_steps = 1 - rearrange(mask_steps, "b t h w -> t (b h) w").unsqueeze(1)
    confidence_steps = (
        confidence_steps - confidence_steps.amin(dim=[2, 3], keepdim=True)
    ) / (
        confidence_steps.amax(dim=[2, 3], keepdim=True)
        - confidence_steps.amin(dim=[2, 3], keepdim=True)
        + 1e-6
    )
    confidence_steps = rearrange(confidence_steps, "b t h w -> t (b h) w").unsqueeze(
        1
    )  # [t, 1, b*H, W]
    confidence_steps = (
        resize_as(confidence_steps, geo_steps).permute(0, 2, 3, 1).repeat(1, 1, 1, 3)
    )

    vis_steps = resize_as(mask_steps, geo_steps) * geo_steps
    vis_steps = vis_steps.permute(0, 2, 3, 1).cpu().numpy()  # [t, b*H, W, C]

    # save gif
    save_numpy_arrays_as_gif(
        vis_steps,
        os.path.join(log_dir, "ar_steps.gif"),
        duration=1,
    )
    vis_steps = rearrange(
        np.stack([vis_steps, confidence_steps.cpu().numpy()], axis=2),
        "t h n w c -> t h (n w) c",
    )  # [t, b*H, W, c]
    save_ar_process(
        rearrange(vis_steps, "t (b h) w c -> b h (t w) c", b=bs),
        os.path.join(log_dir, "ar_steps"),
    )


def save_data(batch, pred, log_dir):
    # save depth, mask, pred mask, pred_roc
    depth = batch["query_depth"].cpu().numpy()
    mask = batch["query_mask"].cpu().numpy()
    pred_mask = pred["mask"].cpu().numpy()
    pred_roc = pred["roc"].cpu().numpy()
    tco_pred = pred["tco_pred"].cpu().numpy()
    template_tco = batch["template_tco"].cpu().numpy()
    pose = batch["query_TCO"].cpu().numpy()
    K = batch["query_K_crop"].cpu().numpy()
    diameter = batch["diameter"].cpu().numpy()
    template_masks = batch["template_masks_visib"].cpu().numpy()
    template_K_crop = batch["template_K_crop"].cpu().numpy()

    # save npy
    data = dict(
        depth=depth,
        mask=mask,
        pred_mask=pred_mask,
        pred_roc=pred_roc,
        pose=pose,
        tco_pred=tco_pred,
        K=K,
        diameter=diameter,
        template_masks=template_masks,
        template_tco=template_tco,
        template_K_crop=template_K_crop,
    )
    np.savez(os.path.join(log_dir, "data.npz"), **data)


def show_rocs(batch, log_dir="logs/debug"):
    #
    bs = len(batch["query"])
    query_roc = batch["query_roc"].cpu().numpy()
    template_rocs = batch["template_rocs"].cpu().numpy()[:, 0]  # only the first

    save_dir = os.path.join(log_dir, "rocs")
    prepare_dir(save_dir)

    for i in range(bs):
        query_roc_i = query_roc[i].transpose(1, 2, 0).reshape(-1, 3)
        template_rocs_i = template_rocs[i].transpose(1, 2, 0).reshape(-1, 3)
        show_multiple_point_clouds_plotly(
            [query_roc_i, template_rocs_i],
            title=f"roc_{i}",
            labels_list=["query_roc", "template_roc_#0"],
            save_path=os.path.join(save_dir, f"roc_{i}.html"),
            show=False,
        )


def show_point_pairs(batch, pred, log_dir="logs/debug"):
    if "valid_pts" not in batch:
        return
    max_to_show = 5
    valid_pts = batch["valid_pts"].cpu().numpy()
    tgt_loc = batch["tgt_loc"].cpu().numpy()
    query_roc_loc = batch["query_roc_loc"].cpu().numpy()
    query_roc = batch["query_rel_roc"].permute(0, 2, 3, 1).cpu().numpy().clip(0, 1)
    template_roc = (
        batch["template_rocs"][:, 0].permute(0, 2, 3, 1).cpu().numpy().clip(0, 1)
    )

    bs = len(batch["query"])

    fig, axs = plt.subplots(bs, 3, figsize=(10, 3 * bs))
    if bs == 1:
        axs = [axs]
    for i in range(bs):
        tgt_loc_i = tgt_loc[i][:max_to_show]
        query_roc_loc_i = query_roc_loc[i][:max_to_show]
        matches = [
            cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0)
            for i in range(len(tgt_loc_i))
        ]
        img_matches = cv2.drawMatches(
            (template_roc * 255).astype(np.uint8)[i],
            [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=1) for p in tgt_loc_i],
            (query_roc * 255).astype(np.uint8)[i],
            [
                cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=1)
                for p in query_roc_loc_i
            ],
            matches,
            None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        axs[i][0].imshow(img_matches)
        axs[i][0].set_title("point pairs")

        axs[i][1].imshow((template_roc * 255).astype(np.uint8)[i])
        axs[i][1].scatter(
            tgt_loc[i][valid_pts[i]][:, 0],
            tgt_loc[i][valid_pts[i]][:, 1],
            s=1,
            c="r",
        )
        axs[i][1].set_title("src pts on template")

        axs[i][2].imshow((query_roc * 255).astype(np.uint8)[i])
        axs[i][2].scatter(
            query_roc_loc[i][valid_pts[i]][:, 0],
            query_roc_loc[i][valid_pts[i]][:, 1],
            s=1,
            c="r",
        )
        axs[i][2].set_title("src pts on query")

    plt.savefig(f"{log_dir}/point_pairs.png")
    plt.close()
