from functools import partial
import hashlib
import json
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

from bop_toolkit_lib import inout
from src.data.bop.instance_dataset import BOPInstanceDataset
from src.data.bop.scene_dataset import BOPSceneDataset
from src.data.bop.ycbv_filter import YCBVKeyframeFilter
from src.data.collate_importer import default_collate_fn
from src.utils.geometry import project_points, transform_coord
from src.models.coordar.visualization import (
    show_batch,
    show_point_pairs,
    show_rocs,
)
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


class BOPFewShot(Dataset):

    def __init__(
        self,
        instance_ds: BOPInstanceDataset,
        num_ref=5,
        ref="random",
        manual_ref="",
    ):
        super().__init__()

        self.instance_ds = instance_ds
        self.num_ref = num_ref
        self.normalize_by = instance_ds.normalize_by
        self.ref = ref
        self.ref_file = manual_ref
        if ref == "manual" and manual_ref:
            self.manual_ref = json.load(open(manual_ref, "r"))

        self.build_index()

    def get_signature(self):
        ref_with_mod_time = self.ref + (
            f"_{os.path.getmtime(self.ref_file)}" if self.ref == "manual" else ""
        )
        hash_code = hashlib.md5(
            f"{self.instance_ds.get_signature()}_{self.num_ref}_{ref_with_mod_time}".encode(
                "utf-8"
            )
        ).hexdigest()
        return hash_code

    def build_index(self):
        cache_path = f".cache/bop_{self.instance_ds.scene_ds.dataset}_fewshot_{self.get_signature()}.pkl"
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
            predefined_poses = np.load(
                "src/utils/lib3d/predefined_poses/obj_poses_level1.npy"
            )
            for label in tqdm(labels):
                rotations = []
                instance_indices = label_to_indices[label]
                if self.ref == "random":
                    for idx in instance_indices:
                        instance = instances.iloc[idx]
                        rot = instance["rot"]
                        rotations.append(rot)
                    rotations = np.stack(rotations)
                    sampled_rotations, sample_indices = (
                        farthest_point_sampling_rotations(rotations, self.num_ref)
                    )
                    ref_indices = instance_indices[sample_indices].tolist()
                elif self.ref == "first_frame":
                    pass
                elif self.ref == "first":
                    pass
                elif self.ref == "manual":
                    obj_id = int(label.split("_")[-1])
                    ref = self.manual_ref[str(obj_id)]
                    ref_data = instances.query(
                        f"scene_id == {ref['scene_id']} and im_id == {ref['im_id']} and gt_id == {ref['gt_id']}"
                    )
                    assert (
                        len(ref_data) == 1
                    ), f"Expected one reference frame, but got {len(ref_data)} for object {obj_id}"
                    ref_indices = ref_data.index.tolist()
                else:
                    raise ValueError(f"Unknown reference type: {self.ref}")

                for idx in instance_indices:
                    if self.ref == "first_frame":
                        # find first frame by current instance
                        curr_instance = instances.iloc[idx]
                        tgt_instances = instances.loc[
                            (instances["label"] == curr_instance["label"])
                            & (instances["scene_id"] == curr_instance["scene_id"])
                        ]
                        ref_indices = (
                            tgt_instances.assign(
                                im_id_int=lambda x: x["im_id"].astype(int)
                            )
                            .nsmallest(1, "im_id_int")
                            .index.tolist()
                        )
                        if len(ref_indices) != 1:
                            ipdb.set_trace()
                            raise ValueError(
                                f"Expected one reference frame, but got {len(ref_indices)} for instance {idx}"
                            )
                    elif self.ref == "first":
                        # find first view by current instance
                        curr_instance = instances.iloc[idx]
                        tgt_instances = instances.loc[
                            (instances["label"] == curr_instance["label"])
                        ]
                        # order by scene_id and im_id
                        ref_indices = (
                            tgt_instances.assign(
                                scene_id_int=lambda x: x["scene_id"].astype(int),
                                im_id_int=lambda x: x["im_id"].astype(int),
                            )
                            .sort_values(by=["scene_id_int", "im_id_int"])
                            .head(1)
                            .index.tolist()
                        )
                        assert (
                            len(ref_indices) == 1
                        ), f"Expected one reference frame, but got {len(ref_indices)} for instance {idx}"
                    metaDatas.append(dict(idx_in_instance=idx, ref_indices=ref_indices))
            assert len(metaDatas) > 0
            prepare_dir(osp.dirname(cache_path))
            pickle.dump(metaDatas, open(cache_path, "wb"))
        self.metaDatas = convert_list_to_dataframe(metaDatas)
        logger.info(f"Loaded {len(self.metaDatas)} samples!")

    def export_ref(self, save_path):
        """
        Export the reference indices to a JSON file.
        """
        ref_data = {}
        for idx, row in self.metaDatas.iterrows():
            ref_indices = row["ref_indices"]
            if len(ref_indices) == 0:
                continue
            instance_meta = self.instance_ds.metaDatas.iloc[ref_indices[0]]
            ref_data[str(instance_meta["obj_id"])] = {
                "scene_id": int(instance_meta["scene_id"]),
                "im_id": int(instance_meta["im_id"]),
                "gt_id": int(instance_meta["gt_id"]),
            }
        inout.save_json(save_path, ref_data)

    def get_test_targets(self):
        # generate test_targets_bop19 for tyol and real275 using sampled image pairs
        instances = []
        logger.info("Generating test targets...")
        for i in trange(len(self)):
            query_index = self.metaDatas.iloc[i]["idx_in_instance"]
            scene_ds = self.instance_ds.scene_ds
            instance_meta = self.instance_ds.metaDatas.iloc[query_index]
            scene_idx = instance_meta["scene_idx"]
            instance_idx = instance_meta["instance_idx"]
            scene_meta = scene_ds.metaDatas.iloc[scene_idx]
            scene_instances = scene_meta["instances"]

            instance = scene_instances[instance_idx]
            instances.append(
                dict(
                    scene_id=instance["scene_id"],
                    im_id=instance["im_id"],
                    obj_id=instance["obj_id"],
                    gt_id=instance["gt_id"],
                )
            )
        instance_cnt = {}
        for instance in instances:
            scene_id = instance["scene_id"]
            im_id = instance["im_id"]
            obj_id = instance["obj_id"]
            gt_id = instance["gt_id"]

            instance_cnt.setdefault(scene_id, {}).setdefault(im_id, {}).setdefault(
                obj_id, set()
            ).add(gt_id)

        test_target = []
        for scene_id, im_dict in sorted(instance_cnt.items()):
            for im_id, obj_dict in sorted(im_dict.items()):
                for obj_id, gt_ids in sorted(obj_dict.items()):
                    test_target.append(
                        dict(
                            scene_id=scene_id,
                            im_id=im_id,
                            obj_id=obj_id,
                            inst_count=len(gt_ids),
                        )
                    )
        return test_target

    def __len__(self):
        return len(self.metaDatas)

    def __getitem__(self, idx):
        idx_in_instance = self.metaDatas.iloc[idx]["idx_in_instance"]
        ref_indices = self.metaDatas.iloc[idx]["ref_indices"]

        instance_list = []
        for i in ref_indices + [idx_in_instance]:
            instance = self.instance_ds[i]
            rgb = instance["rgb_patch"]
            depth = instance["depth_patch"]
            mask = instance["vis_mask_patch"]
            TCO = instance["TCO"]
            nocs = instance["nocs"]
            roc = instance["roc"]
            extents = instance["extents"]
            diameter = instance["diameter"]
            gt_diameter = instance["gt_diameter"]
            if self.normalize_by == "diameter":
                obj_size = diameter
            elif self.normalize_by == "max_axis":
                obj_size = np.max(extents) * 1.1
            else:
                raise ValueError(f"Unknown normalization type: {self.normalize_by}")
            model_center = instance["model_center"]
            rot = instance["TCO"][:3, :3]
            rot_allo = egocentric_to_allocentric(TCO)[:3, :3]
            instance_list.append(
                dict(
                    rgb=rgb,
                    depth=depth,
                    mask=mask,
                    rot=rot,
                    rot_allo=rot_allo,
                    nocs=nocs,
                    roc=roc,
                    extents=extents,
                    model_center=model_center,
                    diameter=diameter,
                    gt_diameter=gt_diameter,
                    obj_size=obj_size,
                    TCO=TCO,
                    K_crop=instance["K_crop"],
                    K=instance["K"],
                    center_in_crop=instance["center_in_crop"],
                    scale_in_crop=instance["scale_in_crop"],
                    points=instance["points"],
                    symmetries=instance["symmetries"],
                    scene_id=instance["scene_id"],
                    im_id=instance["im_id"],
                    gt_id=instance["gt_id"],
                    det_id=instance["det_id"],
                    obj_id=instance["obj_id"],
                    score=instance["score"],
                    bbox=instance["bbox"],
                    depth_bp=instance["depth_bp"],
                    rgb_patch_raw=instance["rgb_patch_raw"],
                    resize_ratio=instance["resize_ratio"],
                    roc_loc=instance["roc_loc"],
                    pts_in_mask=instance["pts_in_mask"],
                    roi_center=instance["roi_center"],
                    roi_wh=instance["roi_wh"],
                )
            )

        query = instance_list[-1]
        templates = instance_list[:-1]
        template_data = {}

        stack = lambda key: np.stack([templates[i][key] for i in range(len(templates))])
        template_data["template_imgs"] = stack("rgb")
        template_data["template_imgs_raw"] = stack("rgb_patch_raw")
        template_data["template_depths"] = stack("depth")
        template_data["template_masks_visib"] = stack("mask")
        template_data["template_rots"] = stack("rot")
        template_data["template_nocs"] = stack("nocs")
        template_data["template_rocs"] = stack("roc")
        template_data["template_tco"] = stack("TCO")
        template_data["template_K_crop"] = stack("K_crop")
        template_data["template_depth_bp"] = stack("depth_bp")
        template_data["template_obj_size"] = stack("obj_size").astype(np.float32)
        template_data["template_resize_ratio"] = stack("resize_ratio").astype(
            np.float32
        )
        template_data["template_roc_loc"] = stack("roc_loc").astype(np.float32)

        # relative data
        stack = lambda key: np.stack(
            [instance_list[i][key] for i in range(len(instance_list))]
        ).astype(np.float32)

        center_in_crop = stack("center_in_crop")
        rel_center_in_crop = (
            center_in_crop - center_in_crop[0:1, :]
        )  # first as reference

        scale_in_crop = stack("scale_in_crop").clip(min=1e-3)  # avoid log(0)
        rel_scale_in_crop = scale_in_crop / scale_in_crop[0:1]  # first as reference

        rot_allo = stack("rot_allo")
        rel_rot_allo = rot_allo[0:1] @ rot_allo.transpose(
            0, 2, 1
        )  # first as reference, to template 0

        roc = stack("roc")
        TCO = stack("TCO")
        mask = stack("mask")
        trans = TCO[:, :3, 3]
        rel_pose = TCO[0:1] @ np.linalg.inv(TCO)  # first as reference, to template 0
        obj_size = stack("obj_size").reshape(-1, 1, 1, 1)
        template_obj_size = template_data["template_obj_size"]

        rel_rocs = (
            (
                einsum(
                    rel_pose[:, :3, :3],
                    obj_size * (roc - 0.5) + trans[:, :, None, None],
                    "b i j, b j h w -> b i h w",
                )
                + rel_pose[:, :3, 3, None, None]
            )
            - trans[:1, :, None, None]
        ) * mask[:, None] / template_obj_size[:1] + 0.5
        template_data["template_rel_rocs"] = rel_rocs[:-1]
        template_data["template_rel_scale"] = rel_scale_in_crop[:-1]
        template_data["template_rot_allo"] = rot_allo[:-1]
        template_data["template_rel_rot_allo"] = rel_rot_allo[:-1]
        template_data["template_center"] = center_in_crop[:-1]
        template_data["template_rel_center"] = rel_center_in_crop[:-1]
        template_data["template_pts_in_mask"] = stack("pts_in_mask")[:-1]

        # consistent with template object size
        scaled_roc = (obj_size * (roc - 0.5)) * mask[:, None] / (
            template_obj_size[0] + 1e-6
        ) + 0.5
        query_roc = scaled_roc[-1]

        # target points
        tgt_pts = transform_coord(rel_pose, stack("pts_in_mask"))  # to reference frame
        K_crop = stack("K_crop")
        tgt_roc_loc = project_points(K_crop[0:1], tgt_pts)  # first as reference
        # filter occluded points
        tgt_roc_loc, tgt_z = tgt_roc_loc[:, :, :2], tgt_roc_loc[:, :, 2]
        obs_depth = stack("depth")
        h, w = obs_depth.shape[1:]
        obs_z = obs_depth.flatten()[
            tgt_roc_loc[:, :, 1].clip(0, h - 1).astype(np.int32) * w
            + tgt_roc_loc[:, :, 0].clip(0, w - 1).astype(np.int32)
        ]
        occ_pts = tgt_z - obs_z > 0.01  # 1cm threshold
        # oob points
        oob_pts = (
            (tgt_roc_loc[:, :, 0] < 0)
            | (tgt_roc_loc[:, :, 0] >= w)
            | (tgt_roc_loc[:, :, 1] < 0)
            | (tgt_roc_loc[:, :, 1] >= h)
        )
        valid_pts = ~(occ_pts | oob_pts)

        tgt_roc_loc[:, :, 0] = tgt_roc_loc[:, :, 0].clip(0, w - 1)
        tgt_roc_loc[:, :, 1] = tgt_roc_loc[:, :, 1].clip(0, h - 1)

        data = dict(
            query=query["rgb"],
            query_raw=query["rgb_patch_raw"],
            query_rot=query["rot"],
            query_depth=query["depth"],
            query_depth_bp=query["depth_bp"],
            query_mask=query["mask"],
            query_nocs=query["nocs"],
            query_roc=query_roc,
            query_roc_loc=query["roc_loc"],
            query_rel_roc=rel_rocs[-1],
            query_resize_ratio=query["resize_ratio"],
            query_center=center_in_crop[-1],
            query_rel_center=rel_center_in_crop[-1],
            query_scale=scale_in_crop[-1],
            query_rel_scale=rel_scale_in_crop[-1],
            query_rot_allo=rot_allo[-1],
            query_rel_rot_allo=rel_rot_allo[-1],
            query_pts_in_mask=query["pts_in_mask"],
            extents=query["extents"],
            model_center=query["model_center"],
            gt_diameter=query["gt_diameter"],
            diameter=query["diameter"],
            query_obj_size=query["obj_size"],
            query_TCO=query["TCO"],
            query_rel_TCO=rel_pose[-1],
            query_K_crop=query["K_crop"],
            query_K=query["K"],
            query_roi_center=query["roi_center"],
            query_roi_wh=query["roi_wh"],
            points=query["points"],
            symmetries=query["symmetries"],
            scene_id=query["scene_id"],
            im_id=query["im_id"],
            obj_id=query["obj_id"],
            score=query["score"],
            bbox=query["bbox"],
            gt_id=query["gt_id"],
            det_id=query["det_id"],
            tgt_loc=tgt_roc_loc[-1],
            valid_pts=valid_pts[-1],
        )
        data.update(dict(**template_data))
        return data


def test_ycbv():
    ycbv_filter = YCBVKeyframeFilter("./data/BOP")
    scene_dataset = BOPSceneDataset(
        "data/BOP",
        "ycbv",
        "test",
        split_type=None,
        only_bop19_test=False,
        choose_obj=[],
        filters=[ycbv_filter],
    )
    print(len(scene_dataset))
    dzi_config = dict(
        dzi_type="uniform", dzi_pad_scale=1.5, dzi_scale_ratio=0.0, dzi_shift_ratio=0.0
    )
    obj_ds = BOPObjectDataset("data/BOP/ycbv/models")
    instance_dataset = BOPInstanceDataset(scene_dataset, obj_ds, dzi_config=dzi_config)
    print(len(instance_dataset))

    few_shot_ds = BOPFewShot(instance_dataset, num_ref=1, ref="first_frame")

    test_targets = few_shot_ds.get_test_targets()

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
        show_batch(batch)
        show_rocs(batch)
        ipdb.set_trace()
        print(batch["symmetries"].shape)


def test_lm():
    scene_dataset = BOPSceneDataset(
        "data/BOP",
        "lm",
        "test",
        split_type=None,
        only_bop19_test=True,
        choose_obj=[1],
    )
    dzi_config = dict(
        dzi_type="uniform", dzi_pad_scale=1.5, dzi_scale_ratio=0.0, dzi_shift_ratio=0.0
    )
    obj_ds = BOPObjectDataset("data/BOP/lm/models")
    instance_dataset = BOPInstanceDataset(
        scene_dataset, obj_ds, dzi_config=dzi_config, diameter="estimated"
    )
    print(len(instance_dataset))

    # few_shot_ds = BOPFewShot(
    #     instance_dataset,
    #     num_ref=1,
    #     ref="manual",
    #     manual_ref="src/data/bop/manual_ref/lm_manual.json",
    # )
    few_shot_ds = BOPFewShot(
        instance_dataset,
        num_ref=1,
        ref="first_frame",
    )
    few_shot_ds[0]
    data_loader = torch.utils.data.DataLoader(
        few_shot_ds,
        num_workers=8,
        collate_fn=default_collate_fn,
        batch_size=8,
    )
    # few_shot_ds.export_ref("src/data/bop/manual_ref/lm.json")
    for batch in tqdm(data_loader):
        show_batch(batch)
        show_rocs(batch)
        show_point_pairs(batch, None)
        ipdb.set_trace()


# python -m src.data.bop.few_shot_dataset
if __name__ == "__main__":
    test_lm()
