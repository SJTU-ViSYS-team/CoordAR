import copy
import hashlib
import os
import pickle
import warnings
import cv2
from einops import rearrange
import ipdb
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import skimage
from mmcv.image.io import imread
import os.path as osp
import torch
from torch.utils.data import Dataset
from tqdm import tqdm, trange
from bop_toolkit_lib.misc import calc_pts_diameter2
from src.data.bop.data_utils import idx2coords
from src.utils.geometry import depth_to_cam_coords_points
from src.third_party.bop_toolkit.bop_toolkit_lib import inout
from src.data.bop.scene_dataset import BOPSceneDataset
from src.data.megapose.obj_ds.bop_object_dataset import BOPObjectDataset
from src.data.megapose.shapenet import (
    depth_backproject,
    depth_to_roc_map,
    depth_to_xyz,
    normalize_depth_bp_minmax,
    normalize_depth_bp_zscore,
)
from src.third_party.bop_toolkit.bop_toolkit_lib.dataset_params import (
    get_model_params,
    get_present_scene_ids,
    get_split_params,
)
from src.utils.augment import aug_bbox_DZI
from src.utils.cropping import (
    crop_resize_by_warp_affine,
    get_K_crop_resize,
    get_affine_transform,
    xywh2xyxy,
)
from src.utils.inout import convert_list_to_dataframe


from src.utils.logging import get_logger
from src.utils.mask_utils import binary_mask_to_rle, cocosegm2mask
from src.utils.misc import prepare_dir
from src.utils.pysixd.RT_transform import allocentric_to_egocentric

logger = get_logger(__name__)


def remove_depth_outliers_by_diameter(depth_map, K, mask, diameter, background_value=0):
    """
    去除深度图中超出指定直径范围的异常值。

    Args:
        depth_map (np.ndarray): 输入深度图（float32 或 uint16）。
        K (np.ndarray): 相机内参矩阵（3x3）。
        mask (np.ndarray): 二值掩码（True/1 表示有效区域）。用于计算中心
        diameter (float): 深度值的允许波动直径（总范围）。
        background_value (float): 异常值的填充值（默认 0）。

    Returns:
        np.ndarray: 处理后的深度图。
    """
    # 确保 mask 是布尔类型
    mask = mask.astype(bool)
    empty_depth = depth_map < 0.001
    valid_mask = mask & ~empty_depth  # 最终有效区域

    # 提取 mask 区域内的深度值
    masked_depth = depth_map[mask]

    if len(masked_depth) == 0:
        return depth_map, np.zeros_like(depth_map).astype(
            np.bool
        )  # 如果 mask 为空，直接返回原图

    point_map = depth_to_cam_coords_points(depth_map, K)
    center_point = np.mean(point_map[valid_mask], axis=0)
    distances = np.linalg.norm(point_map - center_point, axis=-1)
    outlier_mask = distances > (diameter / 2)

    # 创建输出深度图（避免修改原图）
    processed_depth = depth_map.copy()

    # 在 mask 区域内，将超出范围的深度值置为 background_value
    processed_depth[outlier_mask] = background_value

    return processed_depth, outlier_mask


def largest_connected_component_bbox(mask):
    """
    找出二值掩码中面积最大的连通区域并返回其包围框

    参数:
        mask (numpy.ndarray): 二值掩码图像(0和1或0和255)

    返回:
        tuple: (x, y, w, h) 包围框坐标和宽高
              如果mask全为0，则返回None
    """
    # 确保mask是二值图像(0和255)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if np.max(mask) == 1:
        mask = mask * 255

    # 查找连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    # 如果没有连通区域(只有背景)，返回None
    if num_labels < 2:
        return None

    # 找到面积最大的区域(跳过背景区域0)
    max_area = 0
    max_idx = 1  # 从1开始，0是背景

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > max_area:
            max_area = area
            max_idx = i

    # 获取最大区域的包围框 (x, y, w, h)
    x = stats[max_idx, cv2.CC_STAT_LEFT]
    y = stats[max_idx, cv2.CC_STAT_TOP]
    w = stats[max_idx, cv2.CC_STAT_WIDTH]
    h = stats[max_idx, cv2.CC_STAT_HEIGHT]

    return (x, y, w, h)


def estimate_obj_size(depth, mask, K, TCO, percentile_clip=10):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    ys, xs = np.where(mask)
    zs = depth[ys, xs]
    valid = zs > 0.01
    xs, ys, zs = xs[valid], ys[valid], zs[valid]

    if len(zs) == 0:
        return np.zeros(3, dtype=np.float32), 0

    X = (xs - cx) * zs / fx
    Y = (ys - cy) * zs / fy
    Z = zs

    # 根据mask大小粗略估计直径
    mask_bbox = largest_connected_component_bbox(mask)
    if mask_bbox is not None:
        # Get bbox coordinates
        _, _, w, h = mask_bbox
        # Calculate diagonal length in pixels
        bbox_diagonal_px = np.sqrt(w**2 + h**2)
        # Use median depth for conversion
        median_depth = np.median(zs)
        # Convert to 3D (approximate diameter)
        approx_diameter = bbox_diagonal_px * median_depth / fx
    else:
        approx_diameter = 550  # very large

    # # 去除极端值
    # def percentile_mask(arr):
    #     low, high = np.percentile(arr, percentile_clip), np.percentile(
    #         arr, 100 - percentile_clip
    #     )
    #     return (arr >= low) & (arr <= high)

    # valid = percentile_mask(X) & percentile_mask(Y) & percentile_mask(Z)

    # X = X[valid]
    # Y = Y[valid]
    # Z = Z[valid]

    # # 计算点云中心
    # points = np.stack([X, Y, Z], axis=-1)
    # center = np.median(points, axis=0)

    # # 去除中心之外超出半径1.5倍距离的点
    # distances = np.linalg.norm(points - center, axis=1)
    # valid = distances < (1.5 * approx_diameter / 2)
    # points = points[valid]

    # points = points[np.random.permutation(len(points))[:1000]]  # 随机采样1000个点
    # diameter = calc_pts_diameter2(points).astype(np.float32)

    diameter = np.float32(approx_diameter)

    extents = np.ones(3, dtype=np.float32) * diameter * np.sqrt(1 / 3)
    return extents, diameter


class BOPInstanceDataset(Dataset):

    def __init__(
        self,
        scene_ds: BOPSceneDataset,
        obj_ds: BOPObjectDataset,
        dzi_config={},
        zoom_size=(224, 224),
        load_cad=False,
        clean_bg=False,
        normalize_by="diameter",
        diameter="oracle",
        mask_type="GT",
        remove_depth_by_diameter=True,
        n_sample_observed_point=2048,
    ):
        super().__init__()

        self.scene_ds = scene_ds
        self.obj_ds = obj_ds
        self.dzi_config = dzi_config
        self.zoom_size = np.array(zoom_size)
        self.load_cad = load_cad
        self.clean_bg = clean_bg
        self.normalize_by = normalize_by
        self.diameter = diameter
        self.mask_type = mask_type
        self.remove_depth_by_diameter = remove_depth_by_diameter
        self.n_sample_observed_point = n_sample_observed_point

        self.build_index()

    def get_signature(self):
        hashed_file_name = hashlib.md5(
            (
                "{}_instances".format(
                    self.scene_ds.get_signature(),
                )
            ).encode("utf-8")
        ).hexdigest()
        return hashed_file_name

    def build_index(self):
        cache_path = (
            f".cache/bop_{self.scene_ds.dataset}_instances_{self.get_signature()}.pkl"
        )
        if osp.exists(cache_path):
            logger.info("get instances from cache file: {}".format(cache_path))
            metaDatas = pickle.load(open(cache_path, "rb"))
            logger.info("num instances: {}".format(len(metaDatas)))
            assert len(metaDatas) > 0
        else:
            metaDatas = []
            scene_meta = self.scene_ds.metaDatas
            num_scenes = len(scene_meta)

            for i in trange(num_scenes):
                scene_data = scene_meta.iloc[i]
                scene_id = scene_data["scene_id"]
                im_id = scene_data["im_id"]
                instances = scene_data["instances"]
                for j, instance in enumerate(instances):
                    label = instance["label"]
                    TCO = instance["TCO"]
                    rot = allocentric_to_egocentric(TCO)[:3, :3]
                    metaDatas.append(
                        dict(
                            scene_idx=i,
                            instance_idx=j,
                            label=label,
                            rot=rot,
                            TCO=TCO,
                            obj_id=int(label.split("_")[-1]),
                            scene_id=scene_id,
                            im_id=im_id,
                            gt_id=instance["gt_id"],
                            det_id=instance["det_id"],
                        )
                    )
            prepare_dir(osp.dirname(cache_path))
            pickle.dump(metaDatas, open(cache_path, "wb"))
        self.metaDatas = convert_list_to_dataframe(metaDatas)
        logger.info(f"num instances:{len(self.metaDatas)}")

    def look_up(self, scene_id, im_id, obj_id):
        idx = self.metaDatas.query(
            f"scene_id == {scene_id} and im_id == {im_id} and obj_id == {obj_id}"
        ).index
        if len(idx) == 0:
            return None
        idx = idx[0]
        return self.__getitem__(idx)

    def __len__(self):
        return len(self.metaDatas)

    def __getitem__(self, idx: int):

        metaData = self.metaDatas.iloc[idx]
        scene_idx = metaData["scene_idx"]
        instance_idx = metaData["instance_idx"]
        label = metaData["label"]

        scene_data = self.scene_ds[scene_idx]
        scene_meta = self.scene_ds.metaDatas.iloc[scene_idx]
        instances = scene_meta["instances"]
        instance = instances[instance_idx]

        K = scene_meta["K"]
        scene_id = scene_meta["scene_id"]
        im_id = scene_meta["im_id"]

        rgb = rearrange(scene_data["rgb"], "c h w -> h w c")
        depth = scene_data["depth"]
        TCO = instance["TCO"]
        gt_id = instance["gt_id"]
        score = instance["score"]
        det_id = instance["det_id"]

        gt_t = TCO[:3, 3]
        gt_2d = gt_t @ K.T
        gt_2d[0] /= gt_2d[2]
        gt_2d[1] /= gt_2d[2]
        center_in_img = gt_2d[:2]

        obj = self.obj_ds.get_object_by_label(label)
        model_center = obj.model_center
        points = obj.points
        extents = obj.extents
        diameter = obj.diameter_meters
        gt_diameter = diameter
        symmetries = obj.make_symmetry_poses()
        if self.load_cad:
            mesh = obj.model_p3d
        obj_id = int(label.split("_")[-1])

        im_H, im_W = rgb.shape[:2]  # h, w
        mask_path = self.scene_ds.dp_split["mask_visib_tpath"].format(
            scene_id=scene_id, im_id=im_id, gt_id=gt_id
        )
        if self.mask_type != "GT":
            mask_path = mask_path.replace("mask_visib", f"mask_{self.mask_type}")
        if gt_id >= 0:
            mask_visib = cv2.imread(mask_path)[..., 0] // 255
        else:
            mask_visib = np.zeros_like(rgb[..., 0])
        h, w = rgb.shape[:2]

        bbox = instance["bbox"]
        bbox_xyxy = xywh2xyxy(bbox)

        detection_center_2d_gt = (bbox_xyxy[:2] + bbox_xyxy[2:]) / 2
        if self.dzi_config:
            detection_center_2d, crop_size = aug_bbox_DZI(
                bbox_xyxy, h, w, **self.dzi_config, return_box=False
            )
            detection_center_2d = detection_center_2d
        else:
            detection_center_2d = detection_center_2d_gt
            crop_size = (bbox_xyxy[2:] - bbox_xyxy[:2]).max()
        crop_size = float(crop_size)
        crop_image = lambda x, interpolation: crop_resize_by_warp_affine(
            x,
            detection_center_2d,
            crop_size,
            self.zoom_size,
            interpolation,
        )
        vis_mask_patch = crop_image(mask_visib, cv2.INTER_NEAREST)
        rgb_patch = crop_image(rgb, cv2.INTER_LINEAR)
        depth_patch = crop_image(depth, cv2.INTER_NEAREST)

        # calc bbox of the affine transform
        crop_trans = get_affine_transform(
            detection_center_2d, crop_size, 0, self.zoom_size
        )
        x1 = (0 - crop_trans[0, 2]) / crop_trans[0, 0]
        y1 = (0 - crop_trans[1, 2]) / crop_trans[1, 1]
        x2 = (self.zoom_size[0].item() - crop_trans[0, 2]) / crop_trans[0, 0]
        y2 = (self.zoom_size[1].item() - crop_trans[1, 2]) / crop_trans[1, 1]
        bbox_affine = np.array([x1, y1, x2, y2]).astype(np.float32)
        bw = max(bbox_affine[2] - bbox_affine[0], 1)
        bh = max(bbox_affine[3] - bbox_affine[1], 1)
        roi_wh = torch.tensor([bw, bh], dtype=torch.float32)

        delta_c = center_in_img - detection_center_2d
        center_in_crop = np.array([delta_c[0] / bw, delta_c[1] / bh])
        resize_ratio = self.zoom_size[0] / crop_size
        scale_in_crop = 1 / (gt_t[2]) * resize_ratio

        K_crop = get_K_crop_resize(
            K,
            bbox_affine,
            orig_size=(h, w),
            crop_resize=self.zoom_size,
        )[0].numpy()

        if self.diameter == "estimated":
            try:
                extents, diameter = estimate_obj_size(
                    depth_patch, vis_mask_patch, K_crop, TCO
                )
            except Exception as e:  # noqa: E722
                logger.warning(
                    f"Failed to estimate extents for {scene_id} {im_id} {gt_id}: {e}"
                )
                raise ValueError()
        elif self.diameter == "measured":
            diameter = obj.diameter_meters * np.random.uniform(0.9, 1.1)
            extents = np.ones(3, dtype=np.float32) * diameter * np.sqrt(1 / 3)
        else:
            diameter = np.array(diameter, dtype=np.float32)

        nocs = depth_to_xyz(
            depth_patch * vis_mask_patch,
            K_crop,
            TCO[:3, :3],
            TCO[:3, 3],
            extents=extents,
            normalize_by=self.normalize_by,
        )
        roc = depth_to_roc_map(
            depth_patch * vis_mask_patch,
            K_crop,
            TCO[:3, 3],
            extents=extents,
            normalize_by=self.normalize_by,
        )

        if self.remove_depth_by_diameter:
            depth_patch, outlier_mask = remove_depth_outliers_by_diameter(
                depth_patch,
                K_crop,
                vis_mask_patch,
                diameter=np.linalg.norm(extents),
                background_value=0,
            )
            depth_bp_mask = vis_mask_patch & ~outlier_mask
        else:
            depth_bp_mask = vis_mask_patch

        points_in_cam = depth_backproject(
            depth_patch,
            K_crop,
        )
        depth_bp = (
            normalize_depth_bp_zscore(
                points_in_cam,
                depth_bp_mask,
            )
            * depth_bp_mask[..., None]
        )

        roc_loc = depth_bp_mask.astype(np.float32).flatten().nonzero()[0]
        if len(roc_loc) == 0:
            # 如果没有有效点，随机选择一些点，避免报错
            roc_loc = np.random.choice(
                np.arange(self.zoom_size[0] * self.zoom_size[1]),
                size=min(10, self.zoom_size[0] * self.zoom_size[1]),
                replace=False,
            )
            warnings.warn(
                f"No valid depth points for {scene_id} {im_id} {gt_id}, using random points."
            )
        choose_idx = np.random.choice(
            np.arange(len(roc_loc)),
            size=self.n_sample_observed_point,
            replace=len(roc_loc) <= self.n_sample_observed_point,
        )
        roc_loc = idx2coords(roc_loc[choose_idx], self.zoom_size)
        pts_in_mask = points_in_cam[
            roc_loc[:, 1].astype(np.int32), roc_loc[:, 0].astype(np.int32)
        ].astype(np.float32)

        rgb_patch_raw = rgb_patch.copy()
        if self.clean_bg:
            rgb_patch = rgb_patch * vis_mask_patch[..., None]

        data = dict(
            scene_id=scene_id,
            im_id=im_id,
            gt_id=gt_id,
            det_id=det_id,
            obj_id=obj_id,
            score=score,
            bbox=bbox.astype(np.float32),
            vis_mask_patch=vis_mask_patch,
            rgb_patch=rearrange(rgb_patch, "h w c -> c h w"),
            rgb_patch_raw=rearrange(rgb_patch_raw, "h w c -> c h w"),
            depth_patch=depth_patch,
            depth_bp=rearrange(depth_bp, "h w c -> c h w"),
            TCO=TCO.astype(np.float32),
            K=K,
            K_crop=K_crop,
            nocs=rearrange(nocs, "h w c -> c h w"),
            roc=rearrange(roc, "h w c -> c h w").clip(0, 1),
            roc_loc=roc_loc.astype(np.float32),
            extents=extents,
            diameter=diameter,
            model_center=model_center.astype(np.float32),
            center_in_crop=center_in_crop.astype(np.float32),
            scale_in_crop=scale_in_crop.astype(np.float32),
            points=points,
            symmetries=torch.from_numpy(symmetries).float(),
            resize_ratio=resize_ratio.astype(np.float32),
            gt_diameter=gt_diameter,
            obj_scale=np.float32(1.0),
            pts_in_mask=pts_in_mask.astype(np.float32),
            roi_center=detection_center_2d,
            roi_wh=roi_wh,
        )
        if self.load_cad:
            data.update(dict(mesh=mesh))
        return data


# python -m src.data.bop.instance_dataset
if __name__ == "__main__":

    scene_dataset = BOPSceneDataset(
        "data/BOP",
        "ycbv",
        "test",
        only_bop19_test=True,
        choose_obj=[18],
    )
    obj_ds = BOPObjectDataset("data/BOP/ycbv/models")
    instance_dataset = BOPInstanceDataset(scene_dataset, obj_ds, diameter="estimated")
    print(len(instance_dataset))

    for i in trange(len(instance_dataset)):
        data = instance_dataset[i]
        print(data["diameter"])
