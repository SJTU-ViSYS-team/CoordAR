import copy
import hashlib
import os
import pickle
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
from src.data.bop.instance_dataset import remove_depth_outliers_by_diameter
from src.data.collate_importer import default_collate_fn
from src.data.foundation_pose.instance_dataset import show_batch
from src.data.megapose.obj_ds.real275_dataset import Real275ObjectDataset
from src.data.real275.scene_dataset import Real275SceneDataset
from src.third_party.bop_toolkit.bop_toolkit_lib import inout
from src.data.megapose.shapenet import depth_to_roc_map, depth_to_xyz
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
from src.utils.misc import prepare_dir, to_device
from src.utils.pysixd.RT_transform import allocentric_to_egocentric

logger = get_logger(__name__)


class Real275InstanceDataset(Dataset):

    def __init__(
        self,
        scene_ds: Real275SceneDataset,
        obj_ds: Real275ObjectDataset,
        dzi_config={},
        zoom_size=(224, 224),
        load_cad=False,
        clean_bg=False,
        normalize_by="diameter",
    ):
        super().__init__()

        self.scene_ds = scene_ds
        self.obj_ds = obj_ds
        self.dzi_config = dzi_config
        self.zoom_size = np.array(zoom_size)
        self.load_cad = load_cad
        self.clean_bg = clean_bg
        self.normalize_by = normalize_by

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
        cache_path = f".cache/real275_instances_{self.get_signature()}.pkl"
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
                instances = scene_data["instances"]
                for j, instance in enumerate(instances):
                    label = instance["label"]
                    TCO = instance["TCO"]
                    rot = allocentric_to_egocentric(TCO)[:3, :3]
                    metaDatas.append(
                        dict(scene_idx=i, instance_idx=j, label=label, rot=rot)
                    )
            prepare_dir(osp.dirname(cache_path))
            pickle.dump(metaDatas, open(cache_path, "wb"))
        self.metaDatas = convert_list_to_dataframe(metaDatas)
        logger.info(f"num instances:{len(self.metaDatas)}")

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
        visibility = instance["visibility"]
        obj_id = instance["obj_id"]

        # decomposite TCO
        R = TCO[:3, :3]
        gt_t = TCO[:3, 3]
        mesh_scale = np.sqrt((R.T @ R)[0][0])
        R = R / mesh_scale
        TCO = np.eye(4, dtype=np.float32)
        TCO[:3, :3] = R
        TCO[:3, 3] = gt_t

        gt_2d = gt_t @ K.T
        gt_2d[0] /= gt_2d[2]
        gt_2d[1] /= gt_2d[2]
        center_in_img = gt_2d[:2]

        obj = self.obj_ds.get_object_by_label(label)
        extents = obj.extents
        model_center = obj.model_center
        points = obj.points
        diameter = obj.diameter_meters
        symmetries = obj.make_symmetry_poses()
        if self.load_cad:
            mesh = obj.model_p3d

        im_H, im_W = rgb.shape[:2]  # h, w
        if gt_id >= 0:
            mask_visib = (
                cv2.imread(
                    f"{self.scene_ds.root_dir}/{self.scene_ds.split_type}_{self.scene_ds.split}/scene_{scene_id}/{im_id:04d}_mask.png"
                )[..., 0]
                // 255
            )
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
        vis_mask_patch = crop_image(mask_visib, cv2.INTER_LINEAR) > 0
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

        delta_c = center_in_img - detection_center_2d
        center_in_crop = (
            np.array([delta_c[0] / bw, delta_c[1] / bh]) + 0.5
        ) * self.zoom_size
        resize_ratio = self.zoom_size[0] / crop_size
        scale_in_crop = 1 / (gt_t[2]) * resize_ratio

        K_crop = get_K_crop_resize(
            K,
            bbox_affine,
            orig_size=(h, w),
            crop_resize=self.zoom_size,
        )[0].numpy()

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

        depth_patch = remove_depth_outliers_by_diameter(
            depth_patch,
            vis_mask_patch,
            diameter=np.linalg.norm(extents),
            background_value=0,
        )
        if self.clean_bg:
            rgb_patch = rgb_patch * vis_mask_patch[..., None]

        data = dict(
            label=label,
            scene_id=scene_id,
            im_id=im_id,
            gt_id=gt_id,
            det_id=det_id,
            obj_id=obj_id,
            score=score,
            bbox=bbox.astype(np.float32),
            vis_mask_patch=vis_mask_patch,
            rgb_patch=rearrange(rgb_patch, "h w c -> c h w"),
            depth_patch=depth_patch,
            TCO=TCO.astype(np.float32),
            K_crop=K_crop,
            nocs=rearrange(nocs, "h w c -> c h w").clip(0, 1),
            roc=rearrange(roc, "h w c -> c h w").clip(0, 1),
            extents=extents,
            diameter=diameter,
            model_center=model_center.astype(np.float32),
            center_in_crop=center_in_crop.astype(np.float32),
            scale_in_crop=scale_in_crop.astype(np.float32),
            points=points,
            symmetries=torch.from_numpy(symmetries).float(),
            resize_ratio=resize_ratio.astype(np.float32),
            occlusionRatio=1 - visibility,
        )
        if self.load_cad:
            data.update(dict(mesh=mesh))
        return data


# python -m src.data.real275.instance_dataset
if __name__ == "__main__":
    scene_dataset = Real275SceneDataset(
        "data/Real275",
        choose_obj=[],
    )
    obj_ds = Real275ObjectDataset("data/Real275")
    instance_dataset = Real275InstanceDataset(scene_dataset, obj_ds, load_cad=True)
    print(len(instance_dataset))

    for i in trange(len(instance_dataset)):
        data = instance_dataset[i]
        ipdb.set_trace()
        break

    data_loader = torch.utils.data.DataLoader(
        instance_dataset,
        num_workers=1,
        collate_fn=default_collate_fn,
        batch_size=4,
    )

    for batch in tqdm(data_loader):
        batch = to_device(batch, "cuda:0")
        s = show_batch(batch)
