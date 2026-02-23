import copy
import hashlib
import os
import pickle
import cv2
from einops import rearrange
import ipdb
import numpy as np
import pandas as pd
import skimage
import os.path as osp
import torch
import torchvision
from torch.utils.data import Dataset
from tqdm import tqdm, trange
from mmcv.image.io import imread
from bop_toolkit_lib import inout
from src.third_party.bop_toolkit.bop_toolkit_lib.dataset_params import (
    get_model_params,
    get_present_scene_ids,
    get_split_params,
)
from src.utils.inout import convert_list_to_dataframe


from src.utils.logging import get_logger
from src.utils.mask_utils import binary_mask_to_rle
from src.utils.misc import prepare_dir

logger = get_logger(__name__)


class BOPSceneDataset(Dataset):

    def __init__(
        self,
        datasets_path: str,
        dataset: str,
        split="test",
        split_type=None,
        only_bop19_test=False,
        detection="",  # detection file, in bop format
        detection_threshold=0.0,
        min_visib=0.0,
        choose_obj=[],
        top_k_per_obj=100,
        filters=[],
        transforms=None,
    ):
        super().__init__()

        self.datasets_path = datasets_path
        self.dataset = dataset
        self.split = split
        self.split_type = split_type
        self.only_bop19_test = only_bop19_test
        self.detection = detection
        self.detection_threshold = detection_threshold
        self.min_visib = min_visib
        self.choose_obj = choose_obj
        self.top_k_per_obj = top_k_per_obj
        self.filters = filters
        self.transforms = transforms

        self.dp_split = get_split_params(
            self.datasets_path, self.dataset, self.split, self.split_type
        )

        self.dp_eval_model = get_model_params(self.datasets_path, self.dataset, "eval")

        self.scene_ids = get_present_scene_ids(self.dp_split)
        self.obj_ids = self.dp_eval_model["obj_ids"]

        self.build_index()

    def get_signature(self):
        hashed_file_name = hashlib.md5(
            (
                "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_scene".format(
                    self.dataset,
                    self.split,
                    self.split_type,
                    self.only_bop19_test,
                    self.detection,
                    self.detection_threshold,
                    self.min_visib,
                    self.top_k_per_obj,
                    "".join([f.get_signature() for f in self.filters]),
                    "-".join(
                        str(self.choose_obj) if len(self.choose_obj) > 0 else ["all"]
                    ),
                )
            ).encode("utf-8")
        ).hexdigest()
        return hashed_file_name

    def load_detections_(self):
        print("loading detections...")
        targets = pd.read_json(
            f"{self.datasets_path}/{self.dataset}/test_targets_bop19.json"
        )

        def is_filtered(scene_id, im_id, obj_id):
            for f in self.filters:
                if f.is_filtered(scene_id, im_id, obj_id):
                    return True
            return False

        if self.detection:
            detection_path = self.detection
            assert os.path.isfile(detection_path), f"{detection_path} not found"
            detections = inout.load_json(detection_path, True)
            bop_detections = copy.deepcopy(detections)
            detections = {}
            obj_cnt = {}
            for i, est in tqdm(enumerate(bop_detections)):
                scene_id = est["scene_id"]
                obj_id = est["category_id"]
                im_id = est["image_id"]
                if est["score"] < self.detection_threshold:
                    continue
                if len(self.choose_obj) > 0 and obj_id not in self.choose_obj:
                    continue
                if self.only_bop19_test:
                    t = targets.loc[
                        (targets["scene_id"] == scene_id)
                        & (targets["im_id"] == im_id)
                        & (targets["obj_id"] == obj_id)
                    ]
                    if t.empty:
                        continue
                if (
                    self.top_k_per_obj > 0
                    and obj_cnt.get(obj_id, 0) >= self.top_k_per_obj
                ):
                    print("filterd by topk")
                    continue
                if is_filtered(scene_id, im_id, obj_id):
                    continue
                TCO = np.eye(4, dtype=np.float32)
                TCO[2, 3] = 1.0
                detections.setdefault(scene_id, {}).setdefault(im_id, []).append(
                    dict(
                        bbox=np.array(est["bbox"]),  # in xywh format
                        score=est["score"],
                        scene_id=int(scene_id),
                        im_id=int(im_id),
                        time=est["time"],
                        label=f"obj_{est['category_id']:06d}",
                        obj_id=est["category_id"],
                        gt_id=-1,
                        det_id=i,
                        TCO=TCO,  # placeholder
                    )
                )
            print("num detections before filtering:", len(bop_detections))
        else:
            # use gt detection
            instance_id = -1
            num_instances_without_valid_segmentation = 0
            detections = {}
            for scene_id in tqdm(self.scene_ids):
                gt_info = inout.load_scene_gt_info(
                    self.dp_split["scene_gt_info_tpath"].format(scene_id=scene_id)
                )
                scene_gt = inout.load_scene_gt(
                    self.dp_split["scene_gt_tpath"].format(scene_id=scene_id)
                )
                for im_id in sorted(scene_gt.keys()):
                    info = gt_info[im_id]
                    for gt_id, inst_gt in enumerate(scene_gt[im_id]):
                        obj_id = inst_gt["obj_id"]
                        visibility = info[gt_id]["visib_fract"]
                        instance_id += 1
                        if visibility < self.min_visib:
                            continue
                        if len(self.choose_obj) > 0 and obj_id not in self.choose_obj:
                            continue
                        if self.only_bop19_test:
                            t = targets.loc[
                                (targets["scene_id"] == scene_id)
                                & (targets["im_id"] == im_id)
                                & (targets["obj_id"] == obj_id)
                            ]
                            if t.empty:
                                continue
                        if is_filtered(scene_id, im_id, obj_id):
                            continue
                        cam_R_m2c = np.array(inst_gt["cam_R_m2c"]).reshape(3, 3)
                        cam_t_m2c = np.array(inst_gt["cam_t_m2c"]).reshape(3)
                        TCO = np.eye(4, dtype=np.float32)
                        TCO[:3, :3] = cam_R_m2c
                        TCO[:3, 3] = cam_t_m2c / 1000  # mm to m
                        bbox = gt_info[im_id][gt_id]["bbox_obj"]

                        instance = dict(
                            bbox=bbox,  # in xywh format
                            obj_id=obj_id,
                            label=f"obj_{obj_id:06d}",
                            score=1.0,
                            scene_id=int(scene_id),
                            im_id=int(im_id),
                            time=-1,
                            gt_id=gt_id,
                            det_id=-1,
                            TCO=TCO,
                            visibility=visibility,
                        )
                        # cache mask
                        mask_visib_file = self.dp_split["mask_visib_tpath"].format(
                            scene_id=scene_id, im_id=im_id, gt_id=gt_id
                        )
                        mask_file = self.dp_split["mask_tpath"].format(
                            scene_id=scene_id, im_id=im_id, gt_id=gt_id
                        )
                        assert osp.exists(mask_file), mask_file
                        assert osp.exists(mask_visib_file), mask_visib_file
                        mask_single = imread(mask_visib_file, "unchanged")
                        area = mask_single.sum()
                        if (
                            area <= 64
                        ):  # filter out too small or nearly invisible instances
                            num_instances_without_valid_segmentation += 1
                            continue
                        mask_rle = binary_mask_to_rle(mask_single, compressed=True)
                        mask_full = imread(mask_file, "unchanged")
                        mask_full = mask_full.astype("bool")
                        mask_full_rle = binary_mask_to_rle(mask_full, compressed=True)
                        instance.update(
                            dict(
                                mask_visib_rle=mask_rle,
                                mask_amodal_rle=mask_full_rle,
                            )
                        )
                        detections.setdefault(int(scene_id), {}).setdefault(
                            int(im_id), []
                        ).append(instance)
            print(
                "num_instances_without_valid_segmentation:",
                num_instances_without_valid_segmentation,
            )

        return detections

    def build_index(self):
        cache_path = f".cache/bop_{self.dataset}_scene_{self.get_signature()}.pkl"
        if osp.exists(cache_path):
            print("get scene from cache file: {}".format(cache_path))
            metaDatas = pickle.load(open(cache_path, "rb"))
            print("num scene: {}".format(len(metaDatas)))
            assert len(metaDatas) > 0
        else:
            detections = self.load_detections_()
            pbar = tqdm(self.scene_ids)
            metaDatas = []
            for scene_id in pbar:
                scene_camera = inout.load_scene_camera(
                    self.dp_split["scene_camera_tpath"].format(scene_id=scene_id)
                )
                scene_gt = inout.load_scene_gt(
                    self.dp_split["scene_gt_tpath"].format(scene_id=scene_id)
                )
                for im_id in sorted(scene_gt.keys()):
                    pbar.set_description(f"loading scene_{scene_id}_{im_id}")
                    K = scene_camera[im_id]["cam_K"]
                    depth_scale = scene_camera[im_id]["depth_scale"]
                    if scene_id in detections and im_id in detections[scene_id]:
                        metaDatas.append(
                            dict(
                                scene_id=scene_id,
                                im_id=int(im_id),
                                K=K,
                                depth_scale=depth_scale,
                                instances=detections[scene_id][im_id],
                            )
                        )
            prepare_dir(osp.dirname(cache_path))
            pickle.dump(metaDatas, open(cache_path, "wb"))
        self.metaDatas = convert_list_to_dataframe(metaDatas)
        logger.info(f"Loaded {len(self.metaDatas)} images!")

    def __len__(self):
        return len(self.metaDatas)

    def __getitem__(self, idx: int):

        meta_data = self.metaDatas.iloc[idx]

        scene_id = meta_data["scene_id"]
        im_id = meta_data["im_id"]
        depth_scale = meta_data["depth_scale"]

        if self.dataset == "itodd" and self.split_type != "pbr":
            gray = skimage.io.imread(
                self.dp_split["gray_tpath"].format(scene_id=scene_id, im_id=im_id),
            )
            rgb = np.stack([gray] * 3, axis=-1)
        else:
            rgb = skimage.io.imread(
                self.dp_split["rgb_tpath"].format(scene_id=scene_id, im_id=im_id)
            )
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)

        depth = (
            cv2.imread(
                self.dp_split["depth_tpath"].format(scene_id=scene_id, im_id=im_id),
                cv2.IMREAD_UNCHANGED,
            ).astype(np.float32)
            * depth_scale
            * 0.001
        )
        rgb = rearrange(rgb, "h w c -> c h w")
        if self.transforms:
            rgb = torch.tensor(rgb)
            rgb = self.transforms(rgb).numpy()

        return dict(
            rgb=rgb,
            depth=depth.astype(np.float32),
        )


# python -m src.data.bop.scene_dataset
if __name__ == "__main__":
    scene_dataset = BOPSceneDataset(
        "data/BOP",
        "tless",
        "test",
        split_type="primesense",
        only_bop19_test=True,
        choose_obj=[1],
        detection="data/BOP/default_detections/core19_model_based_unseen/cnos-fastsam/cnos-fastsam_tless-test_8ca61cb0-4472-4f11-bce7-1362a12d396f.json",
        transforms=torchvision.transforms.RandomResizedCrop((224, 224)),
    )
    print(len(scene_dataset))

    for i in trange(len(scene_dataset)):
        scene_dataset[i]
