"""
Tools to manipulate bop-webdataset format

bop-webdataset is composed of several shards (a .tar file), 
each containing amaximum of 1000 images. Because images and 
annotations are stored in a .tar file,they can be read sequentially 
to achieve faster reading speeds compared to the other
file formats.

├─ dataset
│  ├─ key_to_shard.json
│  ├─ shard-000000.tar
│  ├─ shard-000001.tar
│  ├─ ...

Each shard contains a chunk of the bop-imagewise format.
The images are typically stored after shuffling, to achieve random 
sampling of the dataset even if the data is read sequentially. E.g.

├─ shard-000000.tar
│  ├─ 00004_00015.rgb.jpg
│  ├─ 00004_00015.camera.json
│  ├─ 00004_00015.gt.json
│  ├─ 00004_00015.gt_info.json
│  ├─ 00004_00015.mask.json
│  ├─ 00004_00015.mask_visib.json
│  ├─ 00021_00777.rgb.jpg
│  ├─ 00021_00777.camera.json
│  ├─ 00021_00777.gt.json
│  ├─ 00021_00777.gt_info.json
│  ├─ 00021_00777.mask.json
│  ├─ 00021_00777.mask_visib.json


The file key_to_shard.json maps an image key to the index of the shard
where it is stored. This can be used to read an individual image
directly in a .tar file, but beware that this may be slow because 
random access in a .tar file
requires to seek the correpsonding file in the entire byte sequence.
"""

import json
import io
import tarfile
import ipdb
import cv2
import numpy as np

from src.third_party.bop_toolkit.bop_toolkit_lib import inout
from .bop_imagewise import io_load_gt, io_load_masks


def decode_sample(
    sample,
    decode_camera,
    decode_rgb,
    decode_gray,
    decode_depth,
    decode_gt,
    decode_gt_info,
    decode_mask,
    decode_mask_visib,
    rescale_depth=True,
    rgb_suffix=".jpg",
    instance_ids=None,
):
    image_data = {
        "__key__": sample["__key__"],
        "__url__": sample["__url__"],
        "camera": None,
        "im_rgb": None,
        "im_gray": None,
        "mask": None,
        "mask_visib": None,
        "gt": None,
        "gt_info": None,
    }

    if decode_camera:
        image_data["camera"] = json.loads(sample["camera.json"])

    if decode_rgb:
        # TODO: KeyError: 'rgb.jpg'
        image_data["im_rgb"] = inout.load_im(sample["rgb" + rgb_suffix]).astype(
            np.uint8
        )

    if decode_gray:
        image_data["im_gray"] = inout.load_im(sample["gray.tiff"]).astype(np.uint8)

    if decode_depth:
        file_bytes = np.asarray(bytearray(sample["depth.png"]), dtype=np.uint8)
        im_depth = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED).astype(np.float32)
        if rescale_depth:
            im_depth *= image_data["camera"]["depth_scale"]
        image_data["im_depth"] = im_depth

    if decode_gt:
        image_data["gt"] = io_load_gt(
            io.BytesIO(sample["gt.json"]), instance_ids=instance_ids
        )

    if decode_gt_info:
        image_data["gt_info"] = io_load_gt(
            io.BytesIO(sample["gt_info.json"]), instance_ids=instance_ids
        )

    if decode_mask_visib:
        image_data["mask_visib"] = io_load_masks(
            io.BytesIO(sample["mask_visib.json"]), instance_ids=instance_ids
        )

    if decode_mask:
        image_data["mask"] = io_load_masks(
            io.BytesIO(sample["mask.json"]), instance_ids=instance_ids
        )

    return image_data


def load_image_data(
    shard_path,
    image_key,
    load_rgb=True,
    load_gray=False,
    load_depth=True,
    load_mask_visib=True,
    load_mask=False,
    load_gt=False,
    load_gt_info=False,
    rescale_depth=True,
    instance_ids=None,
    rgb_suffix=".jpg",
):
    tar = tarfile.open(shard_path, mode="r")

    def _load(ext, read=True):
        buffered_reader = tar.extractfile(f"{image_key}.{ext}")
        if read:
            return buffered_reader.read()
        else:
            return buffered_reader

    image_data = dict(
        camera=None,
        im_rgb=None,
        im_gray=None,
        mask=None,
        mask_visib=None,
        gt=None,
        gt_info=None,
    )
    camera = json.load(_load("camera.json", read=False))
    image_data["camera"] = camera

    if load_rgb:
        image_data["im_rgb"] = inout.load_im(_load("rgb" + rgb_suffix)).astype(np.uint8)

    if load_gray:
        image_data["im_gray"] = inout.load_im(_load("gray.tiff")).astype(np.uint8)

    if load_depth:
        im_depth = inout.load_im(_load("depth.png")).astype(np.float32)
        if rescale_depth:
            im_depth *= camera["depth_scale"]
        image_data["im_depth"] = im_depth

    if load_gt:
        image_data["gt"] = io_load_gt(
            _load("gt.json", read=False), instance_ids=instance_ids
        )

    if load_gt_info:
        image_data["gt_info"] = io_load_gt(
            _load("gt_info.json", read=False), instance_ids=instance_ids
        )

    if load_mask_visib:
        image_data["mask_visib"] = io_load_masks(
            _load("mask_visib.json", read=False), instance_ids=instance_ids
        )

    if load_mask:
        image_data["mask"] = io_load_masks(
            _load("mask.json", read=False), instance_ids=instance_ids
        )

    tar.close()
    return image_data
