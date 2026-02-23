import copy
import cv2
import einops
import ipdb
from matplotlib import pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
import torchvision
from einops import rearrange
import random
import json
from scipy.spatial.transform import Rotation

from tqdm import trange
from src.utils.logging import get_logger
from src.utils.lib3d.numpy import (
    get_obj_poses_from_template_level,
    inverse_transform,
    load_index_level0_in_level2,
)
from src.utils.lib3d.rotation_conversions import convert_rotation_representation
from src.utils.shapeNet_utils import (
    train_categories,
    test_categories,
    open_pose,
    open_image,
)
from src.utils.inout import convert_list_to_dataframe
from pytorch_lightning import seed_everything

logger = get_logger(__name__)


def depth_to_xyz(depth, K, R, T, extents, normalize_by, base_rot=None):
    """
    Convert depth to xyz
    Args:
        depth: (H, W)
        K: (3, 3)
        R: (3, 3)
        T: (3,)
    Returns:
        xyz: (H, W, 3)
    """
    H, W = depth.shape
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    x = (grid_x - K[0, 2]) * depth / K[0, 0]
    y = (grid_y - K[1, 2]) * depth / K[1, 1]
    z = depth

    non_zero = np.isclose(depth, 0.0, atol=1e-3)[..., None]
    mask = ~non_zero

    points_cam = np.stack((x, y, z), axis=-1)
    points_obj = (R.T @ (points_cam.reshape(-1, 3).T - T.reshape(3, 1))).T
    if normalize_by == "axis":
        xyz = points_obj / extents
    elif normalize_by == "max_axis":
        xyz = points_obj / (np.max(extents) * 1.1)
    elif normalize_by == "diameter":
        diameter = np.linalg.norm(extents)
        xyz = points_obj / (diameter + 1e-6)
    else:
        raise ValueError("normalize_by should be either 'axis' or 'obj_size'")
    if base_rot is not None:
        xyz = (base_rot @ xyz.T).T
    xyz = xyz.reshape(H, W, 3) * mask
    xyz = xyz + 0.5
    return xyz.astype(np.float32)


def normalize_depth_bp_minmax(depth_bp, mask, lower_pct=1, upper_pct=99):
    # depth_bp： (H, W, C)
    # 计算截断边界
    if mask.sum() == 0:
        return np.zeros_like(depth_bp) + 0.5
    mask = mask.astype(np.bool)
    lower = np.percentile(depth_bp[mask], lower_pct, axis=0).reshape(1, 1, 3)
    upper = np.percentile(depth_bp[mask], upper_pct, axis=0).reshape(1, 1, 3)

    # 截断 + 线性归一化
    clipped = np.clip(depth_bp, lower, upper)
    normalized = (clipped - lower) / (upper - lower + 1e-6)
    return normalized


def normalize_depth_bp_zscore(depth_bp, mask, n_std=3):
    # depth_bp： (H, W, C)
    if mask.sum() == 0:
        return np.zeros_like(depth_bp) + 0.5
    mask = mask.astype(np.bool)
    mean = np.mean(depth_bp[mask], axis=0).reshape(1, 1, 3)
    std = np.std(depth_bp[mask], axis=0).reshape(1, 1, 3)
    normalized = (depth_bp - mean) / (std * n_std + 1e-6) + 0.5
    return normalized.clip(0, 1)


def depth_backproject(depth, K):
    """
    Backproject depth to 3D points in camera coordinates.
    Args:
        depth: (H, W)
        K: (3, 3)
    Returns:
        points_cam: (H, W, 3)
    """
    H, W = depth.shape
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    x = (grid_x - K[0, 2]) * depth / K[0, 0]
    y = (grid_y - K[1, 2]) * depth / K[1, 1]
    z = depth

    points_cam = np.stack((x, y, z), axis=-1)
    return points_cam.astype(np.float32)


def depth_to_roc_map(depth, K, T, extents, normalize_by):
    """
    Convert depth to xyz
    Args:
        depth: (H, W)
        K: (3, 3)
        R: (3, 3)
        T: (3,)
    Returns:
        xyz: (H, W, 3)
    """
    H, W = depth.shape
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    x = (grid_x - K[0, 2]) * depth / K[0, 0]
    y = (grid_y - K[1, 2]) * depth / K[1, 1]
    z = depth

    non_zero = np.isclose(depth, 0.0, atol=1e-3)[..., None]
    mask = ~non_zero

    points_cam = np.stack((x, y, z), axis=-1)
    points_obj = points_cam.reshape(-1, 3) - T.reshape(1, 3)
    if normalize_by == "axis":
        xyz = points_obj / extents
    elif normalize_by == "max_axis":
        xyz = points_obj / (np.max(extents) * 1.1)
    elif normalize_by == "diameter":
        diameter = np.linalg.norm(extents)
        xyz = points_obj / (diameter + 1e-6)
    else:
        raise ValueError("normalize_by should be either 'axis' or 'obj_size'")
    xyz = xyz.reshape(H, W, 3) * mask
    xyz = xyz + 0.5
    return xyz.astype(np.float32)


def roc_to_nocs_map(roc_map, R, extents, mask, normalize_by, base_rot=None):
    H, W, _ = roc_map.shape
    roc_map = roc_map - 0.5
    points_obj = (R.T @ (obj_size * roc_map.reshape(-1, 3).T)).T
    if normalize_by == "axis":
        xyz = points_obj / extents
    elif normalize_by == "max_axis":
        xyz = points_obj / (np.max(extents) * 1.1)
    elif normalize_by == "diameter":
        xyz = points_obj / np.linalg.norm(extents)
    else:
        raise ValueError("normalize_by should be either 'axis' or 'obj_size'")

    if base_rot is not None:
        xyz = (base_rot @ xyz.T).T
    xyz = xyz.reshape(H, W, 3) * mask[..., None]
    xyz = xyz + 0.5
    return xyz


def rot_to_nocs_map_batch(roc_map, R, extents, mask, normalize_by, base_rot=None):
    B, H, W, _ = roc_map.shape
    roc_map = roc_map - 0.5
    points_obj = (
        R.transpose(-2, -1) @ (obj_size * roc_map.reshape(B, -1, 3).transpose(1, 2))
    ).transpose(1, 2)
    if normalize_by == "axis":
        xyz = points_obj / extents.unsqueeze(1)
    elif normalize_by == "obj_size":
        obj_size = torch.max(extents, dim=-1).reshape(B, 1, 1) * 1.1
        xyz = points_obj / obj_size
    elif normalize_by == "diameter":
        diameter = torch.linalg.norm(extents, dim=-1).reshape(B, 1, 1)
        xyz = points_obj / diameter
    else:
        raise ValueError("normalize_by should be either 'axis' or 'obj_size'")
    if base_rot is not None:
        xyz = (base_rot @ xyz.transpose(1, 2)).transpose(1, 2)
    xyz = xyz.reshape(B, H, W, 3) * mask[..., None]
    xyz = xyz + 0.5
    return xyz


def fit_rotation(src, tgt, mask):
    src = rearrange(src, "b h w c -> b c (h w)")
    tgt = rearrange(tgt, "b h w c -> b c (h w)")
    mask = mask.unsqueeze(1).flatten(2)

    # Apply mask
    src_masked = src * mask
    tgt_masked = tgt * mask
    mask_sum = mask.sum(dim=-1, keepdim=True).clamp(min=1e-6)  # avoid div by 0

    #  Compute centroids
    centroid1 = src_masked.sum(dim=-1, keepdim=True) / mask_sum
    centroid2 = tgt_masked.sum(dim=-1, keepdim=True) / mask_sum

    # Center the points
    src_centered = src_masked - centroid1
    tgt_centered = tgt_masked - centroid2

    # Covariance matrix
    cov = torch.einsum("b i n, b j n -> b i j", src_centered, tgt_centered)  # (B, 3, 3)

    # SVD
    U, S, Vh = torch.linalg.svd(cov, full_matrices=True)
    R = torch.matmul(Vh.transpose(-2, -1), U.transpose(-2, -1))  # R = V U^T

    # Ensure proper rotation (det(R) == 1)
    det = torch.det(R)
    Vh[:, :, -1] *= torch.sign(det).unsqueeze(-1)
    R = torch.matmul(Vh.transpose(-2, -1), U.transpose(-2, -1))

    return R  # (B, 3, 3)


class ShapeNet(Dataset):
    def __init__(
        self,
        root_dir,
        split,
        img_size=224,
        num_views_per_instance=5,
        max_ref_views_per_instance=10,
        random_base_rot=False,
        normalize_by="diameter",
        **kwargs,
    ):
        self.root_dir = Path(root_dir)
        self.split = split

        # implementation details
        self.img_size = img_size
        self.rotation_representation = "rotation6d"
        self.img_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                torchvision.transforms.Resize(self.img_size, antialias=True),
                transforms.Lambda(lambda x: rearrange(x * 2.0 - 1.0, "c h w -> h w c")),
            ]
        )

        self.size_transform = torchvision.transforms.Resize(
            self.img_size, antialias=True
        )

        self.num_views_per_instance = num_views_per_instance
        self.max_ref_views_per_instance = max_ref_views_per_instance
        self.random_base_rot = random_base_rot
        self.normalize_by = normalize_by

        self.K = np.array([[262.5, 0.0, 128], [0.0, 262.5, 128], [0.0, 0.0, 1.0]])

        self.load_metaData()

    def load_metaData(self):
        """
        There are three different splits:
        1. Training sets: ~1000 cads per category (with 13 categories in total)
        2. Unseen instances sets: 50 cads per category (with 13 categories in total)
        3. Unseen categories sets: 1000 cads per category (with 10 categories in total)
        """
        self.is_testing_split = False if self.split == "training" else True
        if self.split in ["training", "unseen_training"]:
            categories = train_categories
            num_cad_per_category = 1000 if self.split == "training" else 50
        elif self.split == "testing":
            categories = test_categories
            num_cad_per_category = 100
        else:
            categories = [self.split]
            num_cad_per_category = 100

        # keep only instances belong to correct split
        all_metaDatas = json.load(open(self.root_dir / "metaData_shapeNet.json"))

        # counter for number of instances for each category
        counters = {cat: 0 for cat in categories}

        self.metaDatas = []
        for obj_id, metaData in enumerate(all_metaDatas):
            cat = metaData["category_name"]
            if cat in categories:
                if counters[cat] >= num_cad_per_category:
                    continue
                counters[cat] += 1
                metaData["symmetry"] = 2 if cat in ["bottle"] else 0
                metaData["obj_id_orig"] = metaData["obj_id"]
                metaData["obj_id"] = obj_id
                for view_id in range(self.num_views_per_instance):
                    metaData_ = metaData.copy()
                    metaData_["view_id"] = int(view_id)
                    self.metaDatas.append(metaData_)

        self.metaDatas = convert_list_to_dataframe(self.metaDatas)
        self.metaDatas = self.metaDatas.sample(frac=1).reset_index(drop=True)
        num_cads = sum([counters[cat] for cat in categories])
        logger.info(
            f"Loaded {len(self.metaDatas)} images for {num_cads} CAD models from split {self.split}!"
        )

    def __len__(self):
        return len(self.metaDatas)

    def __getitem__(self, index):
        obj_id = self.metaDatas["obj_id"].iloc[index]
        obj_id_orig = self.metaDatas["obj_id_orig"].iloc[index]
        obj_dir = self.root_dir / "images" / f"{obj_id:06d}"
        view_id = int(self.metaDatas["view_id"].iloc[index])
        symmetry = int(self.metaDatas["symmetry"].iloc[index])
        diameter = self.metaDatas["diameter"].iloc[index]
        extents = self.metaDatas["extents"].iloc[index]
        origin_bounds = self.metaDatas["origin_bounds"].iloc[index]
        # default rotation fo  shapenet due to history
        # export tool (3ds max) and blender
        rx90 = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

        query = open_image(obj_dir / f"{view_id:06d}_query.png")
        query = self.img_transform(query)
        query_pose = open_pose(obj_dir / "poses.npz", "query", view_id)
        query_depth = (
            cv2.imread(
                obj_dir / f"{view_id:06d}_query_depth.png",
                cv2.IMREAD_UNCHANGED,
            ).astype(np.float32)
            / 1000
        )

        base_rot = np.eye(3)
        if self.random_base_rot:
            base_rot = Rotation.random().as_matrix()

        query_tco = query_pose @ rx90

        query_nocs = depth_to_xyz(
            query_depth,
            self.K,
            query_tco[:3, :3],
            query_tco[:3, 3],
            extents=extents,
            base_rot=base_rot,
            normalize_by=self.normalize_by,
        )
        query_roc = depth_to_roc_map(
            query_depth,
            self.K,
            query_tco[:3, 3],
            extents=np.array(extents),
            normalize_by=self.normalize_by,
        )

        """
            unfortunately, equivalent_tco[:3,:3] is not othogonal
        """
        # normalize_mat = np.eye(4)
        # normalize_mat[:3, :3] = np.diag(np.array(extents))
        # base_rot_t = np.eye(4)
        # base_rot_t[:3, :3] = base_rot.T
        # equivalent_tco = (
        #     query_tco @ np.linalg.inv(normalize_mat) @ base_rot_t @ normalize_mat
        # )

        """
        # test if utils for query_roc is correct
        """
        # recon_xyz = roc_to_nocs_map(
        #     query_roc,
        #     query_tco[:3, :3] @ base_rot.T,
        #     extents=np.array(extents),
        #     mask=(query_depth > 0).astype(np.float32),
        # )
        # psudo_predicted_xyz = roc_to_nocs_map(
        #     query_roc,
        #     np.eye(3),
        #     extents=np.array(extents),
        #     mask=(query_depth > 0).astype(np.float32),
        # )
        # psudo_gt = fit_rotation(
        #     (torch.from_numpy(psudo_predicted_xyz).unsqueeze(0) - 0.5) * diameter,
        #     (torch.from_numpy(query_nocs).unsqueeze(0) - 0.5) * diameter,
        #     torch.from_numpy((query_depth > 0).astype(np.float32)[None]).unsqueeze(0),
        # ).permute(0, 2, 1)
        # print(psudo_gt, query_tco[:3, :3] @ base_rot.T)
        # fig, axs = plt.subplots(2, 3, figsize=(15, 5))
        # axs[0, 0].imshow(query_nocs)
        # axs[0, 0].set_title("query_nocs")
        # axs[0, 1].imshow(query_roc)
        # axs[0, 1].set_title("query_roc")
        # axs[0, 2].imshow(recon_xyz)
        # axs[0, 2].set_title("recon_xyz")
        # axs[1, 0].imshow(psudo_predicted_xyz)
        # axs[1, 0].set_title("psudo_predicted_xyz")
        # plt.show()

        query_nocs = self.size_transform(torch.from_numpy(query_nocs).permute(2, 0, 1))
        query_roc = self.size_transform(torch.from_numpy(query_roc).permute(2, 0, 1))

        query_mask = (query_depth > 0).astype(np.float32)
        query_mask = self.size_transform(torch.from_numpy(query_mask[None]))[0]

        template_data = {
            "template_imgs": [],
            "template_depths": [],
            "template_masks_visib": [],
            "template_masks_full": [],
            "template_nocs": [],
            "template_rocs": [],
            "template_rots": [],
        }
        obj_template_dir = self.root_dir / "templates" / f"{obj_id:06d}"
        template_poses = np.load(obj_template_dir / "poses.npz")["template_poses"]
        template_indices = list(range(len(template_poses)))
        template_indices = np.random.permutation(template_indices)[
            : self.max_ref_views_per_instance
        ]
        for idx in template_indices:
            template = open_image(obj_template_dir / f"{idx:06d}.png")
            template = self.img_transform(template)
            template_depth = (
                cv2.imread(
                    obj_template_dir / f"{idx:06d}_depth.png",
                    cv2.IMREAD_UNCHANGED,
                ).astype(np.float32)
                / 1000
            )
            two = template_poses[idx]
            template_tco = two @ rx90

            # depth to xyz
            xyz = depth_to_xyz(
                template_depth,
                self.K,
                template_tco[:3, :3],
                template_tco[:3, 3],
                extents=extents,
                base_rot=base_rot,
                normalize_by=self.normalize_by,
            )
            roc = depth_to_roc_map(
                template_depth,
                self.K,
                template_tco[:3, 3],
                extents=np.array(extents),
                normalize_by=self.normalize_by,
            )

            xyz = self.size_transform(torch.from_numpy(xyz).permute(2, 0, 1))
            roc = self.size_transform(torch.from_numpy(roc).permute(2, 0, 1))

            template_masks = self.size_transform(
                torch.from_numpy((template_depth > 0).astype(np.float32)[None])
            )[0]

            template_data["template_imgs"].append(template.permute(2, 0, 1))
            template_data["template_depths"].append(torch.from_numpy(template_depth))
            template_data["template_nocs"].append(xyz.float().clip(0, 1))
            template_data["template_rocs"].append(roc.float().clip(0, 1))
            template_data["template_masks_full"].append(template_masks)
            template_data["template_masks_visib"].append(template_masks)
            template_data["template_rots"].append(
                torch.from_numpy(template_tco[:3, :3] @ base_rot.T).float()
            )
        for k in template_data.keys():
            template_data[k] = torch.stack(template_data[k]).float()

        return {
            "query": query.permute(2, 0, 1).float(),
            "query_rot": torch.from_numpy(
                query_tco[:3, :3] @ base_rot.T
            ).float(),  # for evaluation,
            "query_nocs": query_nocs.float().clip(0, 1),
            "query_mask": query_mask.long(),
            "query_roc": query_roc.float().clip(0, 1),
            "symmetry": torch.tensor(symmetry),
            "extents": torch.tensor(extents).float(),
            **template_data,
        }


# python -m src.data.megapose.shapenet

# check consistency with blender rendering
# blenderproc run src/utils/lib3d/blenderproc.py data/MegaPose-ShapeNetCore/shapenetcorev2/models_orig/03211117/1063cfe209bdaeb340ff33d80c1d7d1e/models/model_normalized.obj  data/MegaPose-ShapeNetCore/shapenetcorev2/templates/000000  7
# blender center is array([-0.00927101, -0.0383855 ,  0.215873  ])
# our center is [-0.009271000000000001, 0.215873, 0.038385499999999996]
# ShapeNet数据集中的.obj文件普遍存在一个默认的90度X轴旋转（即rotation_euler=[90°, 0, 0]）。这是因历史导出工具（如3DS Max）的坐标系约定与Blender不同导致的。

if __name__ == "__main__":
    from torchvision.utils import save_image
    from src.models.utils import unnormalize_to_zero_to_one

    seed_everything(2025)

    root_dir = "data/MegaPose-ShapeNetCore/shapenetcorev2"
    dataset = ShapeNet(
        root_dir, "training", max_ref_views_per_instance=10, random_base_rot=True
    )

    save_dir = Path("./logs/debug")
    save_dir.mkdir(exist_ok=True, parents=True)

    # 18-> obj 0
    # 127 -> obj 1
    # 1154 ->  obj 2

    for i in trange(len(dataset)):
        batch = dataset[i]
        # ipdb.set_trace()
        # save_image(batch["query"], save_dir / f"{i:06d}_query.png")
        save_image(
            batch["template_imgs"],
            save_dir / f"{i:06d}_gt_template.png",
        )
        save_image(
            batch["template_nocs"],
            save_dir / f"{i:06d}_gt_template_nocs.png",
        )
        save_image(
            batch["template_rocs"],
            save_dir / f"{i:06d}_gt_template_rocs.png",
        )
        logger.info("-----------------")
