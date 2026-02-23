import hashlib
import json
import os
import pickle
from random import sample
import cv2
from einops import einsum
import ipdb
from matplotlib import pyplot as plt
import numpy as np
import torch
import os.path as osp
from torch.utils.data import Dataset
from tqdm import tqdm

from src.data.bop.instance_dataset import BOPInstanceDataset
from src.data.bop.scene_dataset import BOPSceneDataset
from src.data.megapose.obj_ds.bop_object_dataset import BOPObjectDataset
from src.models.dust3r.cloud_opt import GlobalAlignerMode, global_aligner
from src.models.scanar.geo_solver import GeoSolver
from src.models.scanar.utils.device import collate_with_cat
from src.utils.inout import convert_list_to_dataframe


def label_with_number(img, num, color=(255, 0, 0), scale=5, thickness=10):
    """在图像中心写一个大大的数字。"""
    img_copy = img.copy()
    text = str(num)

    # 获取图像尺寸
    h, w = img.shape[:2]

    # 计算文字尺寸
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )

    # 计算居中位置（左下角坐标）
    x = (w - text_w) // 2
    y = (h + text_h) // 2

    # 在图像上写字
    cv2.putText(
        img_copy,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )

    return img_copy


class ReconDataset(Dataset):
    def __init__(self, instance_ds, num_frames, mark_frames=False):
        super().__init__()
        self.instance_ds = instance_ds
        self.num_frames = num_frames
        self.mark_frames = mark_frames
        self.build_index()

    def __len__(self):
        return len(self.metaDatas)

    def build_index(self):
        print("building few shot indices ...")
        instances = self.instance_ds.metaDatas
        metaDatas = []
        obj_ids = sorted(instances["obj_id"].unique())
        obj_id_to_indices = instances.groupby("obj_id").indices
        for obj_id in obj_ids:
            instance_indices = obj_id_to_indices[obj_id]
            assert (
                len(instance_indices) >= self.num_frames
            ), "not enough frames for object {}".format(obj_id)
            sampled_indices = np.linspace(
                0, len(instance_indices) - 1, self.num_frames, dtype=int
            )
            indices_in_instance = []
            for idx in sampled_indices:
                indices_in_instance.append(instance_indices[idx])
            metaDatas.append(dict(indices_in_instance=indices_in_instance))
            assert len(metaDatas) > 0
            self.metaDatas = convert_list_to_dataframe(metaDatas)

        print("num recon: {}".format(len(metaDatas)))
        assert len(metaDatas) > 0

    def __getitem__(self, idx):
        indices_in_instance = self.metaDatas.iloc[idx]["indices_in_instance"]
        instances = [self.instance_ds[idx] for idx in indices_in_instance]

        views = []
        for i in range(len(instances)):
            instance = instances[i]
            img = {}
            rgb = instance["rgb_patch"]
            if self.mark_frames:
                rgb = label_with_number(rgb.transpose(1, 2, 0), i)
            img["img"] = rgb
            img["depth"] = instance["depth_patch"]
            img["mask"] = instance["vis_mask_patch"]
            img["diameter"] = instance["diameter"]
            img["extents"] = instance["extents"]
            img["roc"] = instance["roc"]
            img["K_crop"] = instance["K_crop"]
            img["tco"] = instance["TCO"]
            img["true_shape"] = instance["rgb_patch"].shape[1:3]
            img["instance"] = f"{i}"
            img["obj_id"] = instance["obj_id"]
            img["idx"] = i
            views.append(img)

        pairs = []
        targets = []
        for i in range(len(views)):
            for j in range(len(views)):
                if i != j:
                    view1 = views[i]
                    view2 = views[j]
                    # compute pairwise coordinates
                    rel_pose = instances[i]["TCO"] @ np.linalg.inv(
                        instances[j]["TCO"]
                    )  # first as reference, to template 0
                    trans1 = instances[i]["TCO"][:3, 3]
                    trans2 = instances[j]["TCO"][:3, 3]
                    obj_size1 = instances[i]["diameter"].reshape(1, 1, 1)
                    obj_size2 = instances[j]["diameter"].reshape(1, 1, 1)
                    roc = instances[i]["roc"]
                    rel_roc = (
                        (
                            einsum(
                                rel_pose[:3, :3],
                                obj_size2 * (instances[j]["roc"] - 0.5)
                                + trans2[:, None, None],
                                "i j, j h w -> i h w",
                            )
                            + rel_pose[:3, 3, None, None]
                        )
                        - trans1[:, None, None]
                    ) * instances[j]["vis_mask_patch"][None] / obj_size1 + 0.5

                    point11 = (roc - 0.5) * obj_size1
                    point21 = (rel_roc - 0.5) * obj_size1
                    target11 = dict(
                        pts3d=point11.transpose(1, 2, 0)[None],
                        conf=np.exp(
                            instances[i]["vis_mask_patch"]
                            * (instances[i]["depth_patch"] > 0)
                        ).astype(np.float32)[None],
                    )
                    target21 = dict(
                        pts3d_in_other_view=point21.transpose(1, 2, 0)[None],
                        conf=np.exp(
                            instances[j]["vis_mask_patch"]
                            * (instances[j]["depth_patch"] > 0)
                        ).astype(np.float32)[None],
                    )

                    pair = (view1, view2)
                    # show
                    # fig, axs = plt.subplots(1, 2)
                    # axs[0].imshow(point11.transpose(1, 2, 0))
                    # axs[1].imshow(point21.transpose(1, 2, 0))
                    # plt.show()

                    pairs.append(pair)
                    targets.append((target11, target21))
        data = dict(
            views=views,
            pairs=pairs,
            targets=targets,
        )
        return data


def get_solver_input():
    scene_dataset = BOPSceneDataset(
        "data/BOP",
        "lm",
        "test",
        only_bop19_test=True,
        choose_obj=[9],
    )
    obj_ds = BOPObjectDataset("data/BOP/lm/models")
    instance_dataset = BOPInstanceDataset(scene_dataset, obj_ds, diameter="oracle")
    print(len(instance_dataset))

    recon_ds = ReconDataset(
        instance_dataset,
        num_frames=5,
    )

    sample = recon_ds[0]
    imgs, pairs, targets = sample["imgs"], sample["pairs"], sample["targets"]
    pairs = collate_with_cat(pairs)
    targets = collate_with_cat(targets)

    view1, view2 = pairs
    pred1, pred2 = targets
    K = [img["K_crop"] for img in imgs]
    K_indices = np.arange(0, len(K), dtype=np.long)

    tco = [img["tco"] for img in imgs]
    tco_indices = np.arange(0, len(tco), dtype=np.long)

    depths = [img["depth"] for img in imgs]
    depth_indices = np.arange(0, len(depths), dtype=np.long)

    return (
        view1,
        view2,
        pred1,
        pred2,
        tco,
        tco_indices,
        K,
        K_indices,
        depths,
        depth_indices,
    )


def test_solver():
    device = "cuda"
    schedule = "cosine"
    lr = 0
    niter = 1

    (
        view1,
        view2,
        pred1,
        pred2,
        tco,
        tco_indices,
        K,
        K_indices,
        depths,
        depth_indices,
    ) = get_solver_input()

    solver = GeoSolver(
        view1,
        view2,
        pred1,
        pred2,
        conf="log",
        verbose=True,
        rot_type="allocentric",
        base_scale=1.0,
    )
    solver.preset_pose(tco, tco_indices, no_grad=True)
    solver.preset_intrinsics(K, K_indices)
    solver.preset_depthmap(depths, depth_indices, no_grad=True)
    solver.cuda()

    solver.compute_global_alignment(init="mst", niter=niter, lr=lr)

    # visualize reconstruction
    poses = solver.get_im_poses()
    pts3d = solver.get_pts3d()
    ipdb.set_trace()
    solver.show()


def test_dust3r():
    device = "cuda"
    schedule = "cosine"
    lr = 0.1
    niter = 300

    (view1, view2, pred1, pred2, tco, tco_indices, K, K_indices, _, _) = (
        get_solver_input()
    )
    output = dict(view1=view1, view2=view2, pred1=pred1, pred2=pred2)

    scene = global_aligner(
        output,
        device=device,
        mode=GlobalAlignerMode.ModularPointCloudOptimizer,
        min_conf_thr=0.5,
    )
    scene.preset_intrinsics(K, K_indices)
    scene.preset_pose(tco, tco_indices, no_grad=False)
    loss = scene.compute_global_alignment(
        init="mst", niter=niter, schedule=schedule, lr=lr
    )
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    scene.show()


# python -m src.data.bop.recon_dataset
if __name__ == "__main__":
    # Example usage
    test_solver()
    # test_dust3r()
