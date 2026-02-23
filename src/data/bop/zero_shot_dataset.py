from functools import partial
import hashlib
import os
import pickle
import random
import time
from einops import einsum, rearrange
import ipdb
import numpy as np
import torch
from tqdm import tqdm, trange
from torch.utils.data import Dataset
import os.path as osp
from torchvision import transforms
from multiprocessing import Pool
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist

from src.data.bop.instance_dataset import BOPInstanceDataset
from src.data.bop.scene_dataset import BOPSceneDataset
from src.data.collate_importer import default_collate_fn
from src.data.foundation_pose.zero_shot_dataset import show_batch
from src.data.megapose.obj_ds.bop_object_dataset import BOPObjectDataset
from src.utils.inout import convert_list_to_dataframe
from src.utils.logging import get_logger
from src.utils.misc import prepare_dir
from src.utils.pysixd.RT_transform import (
    allocentric_to_egocentric,
    egocentric_to_allocentric,
)

logger = get_logger(__name__)


def farthest_point_sampling_rotations(rotations, k):
    """
    使用四元数距离的最远点采样，直接操作旋转矩阵
    输入: rotations - (n, 3, 3) 的旋转矩阵数组
    输出: (k, 3, 3) 的采样旋转矩阵
    """
    # 将旋转矩阵转换为 scipy Rotation 对象（支持四元数距离计算）
    rots = Rotation.from_matrix(rotations)  # shape: (n,)
    quats = rots.as_quat()  # shape: (n, 4)

    # 预计算所有四元数对的距离矩阵
    def quat_distance(q1, q2):
        return min(np.linalg.norm(q1 - q2), np.linalg.norm(q1 + q2))

    # 使用 scipy 的 cdist 加速距离计算（自定义度量）
    dist_matrix = cdist(quats, quats, metric=quat_distance)

    # 最远点采样
    n = len(rotations)
    selected = np.zeros(k, dtype=int)
    remaining = np.arange(n)

    # 随机选择第一个点
    selected[0] = np.random.choice(remaining)
    remaining = np.delete(remaining, np.where(remaining == selected[0]))

    # 初始化最小距离
    min_distances = dist_matrix[selected[0], remaining]

    for i in range(1, k):
        # 选择当前最远点
        farthest_idx = np.argmax(min_distances)
        selected[i] = remaining[farthest_idx]
        remaining = np.delete(remaining, farthest_idx)
        min_distances = np.delete(min_distances, farthest_idx)

        if len(remaining) == 0:
            break

        # 更新最小距离
        new_distances = dist_matrix[selected[i], remaining]
        min_distances = np.minimum(min_distances, new_distances)

    # 返回采样后的旋转矩阵（直接原数组切片）
    return rotations[selected], selected


class BOPZeroShot(Dataset):

    def __init__(
        self,
        instance_ds: BOPInstanceDataset,
        num_ref=5,
    ):
        super().__init__()

        self.instance_ds = instance_ds
        self.num_ref = num_ref
        self.normalize_by = instance_ds.normalize_by

        self.build_index()

    def get_signature(self):
        hash_code = hashlib.md5(
            f"{self.instance_ds.get_signature()}_{self.num_ref}".encode("utf-8")
        ).hexdigest()
        return hash_code

    def build_index(self):
        cache_path = f".cache/bop_fewshot_{self.get_signature()}.pkl"
        if osp.exists(cache_path):
            print("get instances from cache file: {}".format(cache_path))
            metaDatas = pickle.load(open(cache_path, "rb"))
            print("num instances: {}".format(len(metaDatas)))
            assert len(metaDatas) > 0
        else:
            print("building few shot indices ...")
            instances = self.instance_ds.metaDatas
            metaDatas = []
            labels = sorted(instances["label"].unique())
            label_to_indices = instances.groupby("label").indices
            # sample rotation by farthest rotations for every label
            for label in tqdm(labels):
                rotations = []
                instance_indices = label_to_indices[label]
                for idx in instance_indices:
                    instance = instances.iloc[idx]
                    rot = instance["rot"]
                    rotations.append(rot)
                rotations = np.stack(rotations)
                sampled_rotations, sample_indices = farthest_point_sampling_rotations(
                    rotations, self.num_ref
                )
                ref_indices = instance_indices[sample_indices].tolist()
                for idx in instance_indices:
                    metaDatas.append(dict(idx_in_instance=idx, ref_indices=ref_indices))
            assert len(metaDatas) > 0
            prepare_dir(osp.dirname(cache_path))
            pickle.dump(metaDatas, open(cache_path, "wb"))
        self.metaDatas = convert_list_to_dataframe(metaDatas)
        logger.info(f"Loaded {len(self.metaDatas)} samples!")

    def __len__(self):
        return len(self.metaDatas)

    def __getitem__(self, idx):
        idx_in_instance = self.metaDatas.iloc[idx]["idx_in_instance"]
        ref_indices = self.metaDatas.iloc[idx]["ref_indices"]

        instance = self.instance_ds[idx_in_instance]

        instance_list = []
        for i in ref_indices + [idx]:
            instance = self.instance_ds[i]
            rgb = instance["rgb_patch"]
            depth = instance["depth_patch"]
            mask = instance["vis_mask_patch"]
            roc = instance["roc"]
            TCO = instance["TCO"]
            rot = instance["TCO"][:3, :3]
            rot_allo = egocentric_to_allocentric(TCO)[:3, :3]
            extents = instance["extents"]
            model_center = instance["model_center"]
            diameter = instance["diameter"]
            if self.normalize_by == "diameter":
                obj_size = diameter
            elif self.normalize_by == "max_axis":
                obj_size = np.max(extents) * 1.1
            else:
                raise ValueError(f"Unknown normalization type: {self.normalize_by}")
            instance_list.append(
                dict(
                    rgb=rgb,
                    depth=depth,
                    mask=mask,
                    roc=roc,
                    rot=rot,
                    rot_allo=rot_allo,
                    extents=extents,
                    model_center=model_center,
                    diameter=diameter,
                    obj_size=obj_size,
                    TCO=TCO,
                    K_crop=instance["K_crop"],
                    center_in_crop=instance["center_in_crop"],
                    scale_in_crop=instance["scale_in_crop"],
                    points=instance["points"],
                    symmetries=instance["symmetries"],
                    resize_ratio=instance["resize_ratio"],
                )
            )

        # permute
        instance_list = np.random.permutation(instance_list)

        stack = lambda key: np.stack(
            [instance_list[i][key] for i in range(len(instance_list))]
        ).astype(np.float32)
        center_in_crop = stack("center_in_crop")
        rel_center_in_crop = (
            center_in_crop - center_in_crop[0:1, :]
        )  # first as reference

        scale_in_crop = stack("scale_in_crop").clip(min=1e-3)
        rel_scale_in_crop = np.log(
            scale_in_crop / scale_in_crop[0:1]
        )  # first as reference

        rot_allo = stack("rot_allo")
        rel_rot_allo = rot_allo @ rot_allo[0:1].transpose(0, 2, 1)  # first as reference

        # relative roc
        roc = stack("roc")
        TCO = stack("TCO")
        mask = stack("mask")
        trans = TCO[:, :3, 3]
        rel_pose = TCO[0:1] @ np.linalg.inv(TCO)  # first as reference
        obj_size = stack("obj_size").reshape(-1, 1, 1, 1)
        rel_roc = (
            (
                einsum(
                    rel_pose[:, :3, :3],
                    obj_size * (roc - 0.5) + trans[:, :, None, None],
                    "b i j, b j h w -> b i h w",
                )
                + rel_pose[:, :3, 3, None, None]
            )
            - trans[:1, :, None, None]
        ) * mask[:, None] / obj_size + 0.5

        data = dict(
            rgb=stack("rgb"),
            depth=stack("depth"),
            mask=stack("mask"),
            roc=roc,
            extents=stack("extents"),
            model_center=stack("model_center"),
            TCO=stack("TCO"),
            K_crop=stack("K_crop"),
            center_in_crop=center_in_crop,
            rel_center_in_crop=rel_center_in_crop,
            scale_in_crop=scale_in_crop,
            rel_scale_in_crop=rel_scale_in_crop,
            rot_allo=rot_allo,
            rel_rot_allo=rel_rot_allo,
            rel_roc=rel_roc.clip(0, 1),
            points=instance_list[0]["points"],
            diameter=instance_list[0]["diameter"],
            obj_size=instance_list[0]["obj_size"],
            symmetries=instance_list[0]["symmetries"],
            resize_ratio=stack("resize_ratio"),
        )
        return data


# python -m src.data.bop.zero_shot_dataset
if __name__ == "__main__":
    scene_dataset = BOPSceneDataset(
        "data/BOP",
        "lm",
        "test",
        split_type=None,
        only_bop19_test=True,
        choose_obj=[],
    )
    print(len(scene_dataset))
    dzi_config = dict(
        dzi_type="uniform", dzi_pad_scale=1.5, dzi_scale_ratio=0.0, dzi_shift_ratio=0.0
    )
    obj_ds = BOPObjectDataset("data/BOP/lm/models")
    instance_dataset = BOPInstanceDataset(scene_dataset, obj_ds, dzi_config=dzi_config)
    print(len(instance_dataset))

    few_shot_ds = BOPZeroShot(instance_dataset)

    for item in tqdm(few_shot_ds):
        break

    data_loader = torch.utils.data.DataLoader(
        few_shot_ds,
        num_workers=8,
        collate_fn=default_collate_fn,
        batch_size=8,
    )

    for batch in tqdm(data_loader):
        pass
        # ipdb.set_trace()
        show_batch(batch)
