import copy
import glob
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
from torch.utils.data import Dataset
from tqdm import tqdm, trange
from mmcv.image.io import imread
from bop_toolkit_lib import inout
from src.data.megapose.obj_ds.real275_dataset import Real275ObjectDataset
from src.utils.inout import convert_list_to_dataframe


from src.utils.logging import get_logger
from src.utils.mask_utils import binary_mask_to_rle
from src.utils.misc import prepare_dir

logger = get_logger(__name__)


def mask_to_bbox(mask: np.ndarray, margin_ratio: float = 0.0) -> np.ndarray:
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("输入mask必须是二维numpy数组")

    # 找到所有前景像素的坐标
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    # 如果没有前景像素，返回全0
    if not np.any(rows) or not np.any(cols):
        return np.zeros(4, dtype=np.float32)

    # 计算原始bbox [y_min, y_max, x_min, x_max]
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # 计算原始宽高
    w = x_max - x_min + 1
    h = y_max - y_min + 1

    # 计算margin
    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)

    # 应用margin并处理越界
    x_min = max(0, x_min - margin_x)
    y_min = max(0, y_min - margin_y)
    x_max = min(mask.shape[1] - 1, x_max + margin_x)
    y_max = min(mask.shape[0] - 1, y_max + margin_y)

    # 转换为xywh格式
    x = x_min
    y = y_min
    w = x_max - x_min + 1
    h = y_max - y_min + 1

    return np.asarray([x, y, w, h], dtype=np.float32)


class Real275SceneDataset(Dataset):

    def __init__(
        self,
        root_dir,
        split="test",
        split_type="real",
        detection="",  # detection file, in bop format
        detection_threshold=0.0,
        min_visib=0.0,
        choose_obj=[],
        top_k_per_obj=100,
    ):
        super().__init__()

        self.root_dir = root_dir
        self.dataset = "real275"
        self.split = split
        self.split_type = split_type
        self.detection = detection
        self.detection_threshold = detection_threshold
        self.min_visib = min_visib
        self.choose_obj = choose_obj
        self.top_k_per_obj = top_k_per_obj

        self.scene_ids = self.get_scene_ids()  # list of ints
        self.obj_ds = Real275ObjectDataset(
            root_dir=self.root_dir,
            split=self.split,
            split_type=self.split_type,
        )
        self.intrinsics = np.array(
            [[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]]
        ).astype(np.float32)

        self.gts = self.load_gts()
        self.metas = self.load_meta(self.gts)

        self.build_index()

    def get_scene_ids(self):
        sub_dirs = glob.glob(f"{self.root_dir}/{self.split_type}_{self.split}/*/")
        scene_ids = []
        for sub_dir in sub_dirs:
            scene_id = osp.basename(osp.normpath(sub_dir))
            scene_ids.append(int(scene_id.split("_")[1]))
        return sorted(scene_ids)

    def get_signature(self):
        hashed_file_name = hashlib.md5(
            (
                "{}_{}_{}_{}_{}_{}_{}_scene".format(
                    self.split,
                    self.split_type,
                    self.detection,
                    self.detection_threshold,
                    self.min_visib,
                    self.top_k_per_obj,
                    "-".join(
                        str(self.choose_obj) if len(self.choose_obj) > 0 else ["all"]
                    ),
                )
            ).encode("utf-8")
        ).hexdigest()
        return hashed_file_name

    def load_gts(self):
        gt_dir = f"{self.root_dir}/gts/{self.split_type}_{self.split}"
        gt_files = glob.glob(f"{gt_dir}/*.pkl")
        gts = {}
        for gt_file in sorted(gt_files):
            scene_id = int(osp.basename(gt_file).split(".")[0].split("_")[4])
            im_id = int(osp.basename(gt_file).split(".")[0].split("_")[5])
            gt = pickle.load(open(gt_file, "rb"))
            gts.setdefault(scene_id, {}).setdefault(im_id, gt["gt_RTs"])
        return gts

    def load_meta(self, gts):
        metas = {}
        for scene_id in tqdm(self.scene_ids):
            scene_gt = gts[scene_id]
            for im_id in sorted(scene_gt.keys()):
                meta_file = f"{self.root_dir}/{self.split_type}_{self.split}/scene_{scene_id}/{im_id:04d}_meta.txt"
                meta = np.loadtxt(
                    meta_file,
                    dtype={
                        "names": ("id", "num", "name"),
                        "formats": ("i4", "i4", "U50"),
                    },
                )
                metas.setdefault(int(scene_id), {}).setdefault(int(im_id), meta)
        return metas

    def load_detections_(self):
        print("loading detections...")
        if self.detection:
            raise NotImplementedError(
                "Loading detections from a file is not implemented yet."
            )
        else:
            # use gt detection
            instance_id = -1
            num_instances_without_valid_segmentation = 0
            detections = {}
            for scene_id in tqdm(self.scene_ids):
                scene_gt = self.gts[scene_id]
                for im_id in sorted(scene_gt.keys()):
                    gt = scene_gt[im_id]
                    mask_visib_file = f"{self.root_dir}/{self.split_type}_{self.split}/scene_{scene_id}/{im_id:04d}_mask.png"
                    mask_file = f"{self.root_dir}/{self.split_type}_{self.split}/scene_{scene_id}/{im_id:04d}_mask.png"
                    assert osp.exists(mask_file), mask_file
                    assert osp.exists(mask_visib_file), mask_visib_file
                    mask_scene = imread(mask_visib_file, "unchanged")
                    for gt_id, inst_gt in enumerate(gt):
                        label = str(self.metas[scene_id][im_id][gt_id]["name"])
                        obj_id = self.obj_ds.label2id.get(label, -1)
                        visibility = 1.0
                        instance_id += 1
                        if len(self.choose_obj) > 0 and obj_id not in self.choose_obj:
                            continue
                        TCO = inst_gt
                        mask_single = mask_scene == (gt_id + 1)
                        area = mask_single.sum()
                        bbox = mask_to_bbox(mask_single, margin_ratio=0.1)
                        if (
                            area <= 64
                        ):  # filter out too small or nearly invisible instances
                            num_instances_without_valid_segmentation += 1
                            continue

                        instance = dict(
                            bbox=bbox,  # in xywh format
                            obj_id=obj_id,
                            label=f"real275_{label}",
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
        cache_path = f".cache/{self.dataset}_scene_{self.get_signature()}.pkl"
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
                scene_gt = self.gts[scene_id]
                for im_id in sorted(scene_gt.keys()):
                    pbar.set_description(f"loading scene_{scene_id}_{im_id}")
                    depth_scale = 1.0
                    if scene_id in detections and im_id in detections[scene_id]:
                        metaDatas.append(
                            dict(
                                scene_id=scene_id,
                                im_id=int(im_id),
                                K=self.intrinsics,
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

        rgb = skimage.io.imread(
            f"{self.root_dir}/{self.split_type}_{self.split}/scene_{scene_id}/{im_id:04d}_color.png"
        )

        depth = (
            cv2.imread(
                f"{self.root_dir}/{self.split_type}_{self.split}/scene_{scene_id}/{im_id:04d}_depth.png",
                cv2.IMREAD_UNCHANGED,
            ).astype(np.float32)
            * depth_scale
            * 0.001
        )

        return dict(
            rgb=rearrange(rgb, "h w c -> c h w"),
            depth=depth.astype(np.float32),
        )


# python -m src.data.real275.scene_dataset
if __name__ == "__main__":
    scene_dataset = Real275SceneDataset(
        "data/Real275",
        choose_obj=[],
    )
    print(len(scene_dataset))

    for i in trange(len(scene_dataset)):
        scene_dataset[i]
