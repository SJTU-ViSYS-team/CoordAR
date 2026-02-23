import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.data.datasets.megapose.obj_ds.bop_object_dataset import BOPObjectDataset
from src.data.datasets.megapose.obj_ds.gso_dataset import GoogleScannedObjectDataset
from src.data.datasets.megapose.obj_ds.object_dataset import RigidObjectDataset
from src.data.datasets.megapose.obj_ds.shapenet_object_dataset import (
    ShapeNetObjectDataset,
)

BOP_DS_DIR = "./data/BOP"
GSO_DIR = "./data/google_scanned_objects"
SHAPENET_DIR = "./data/MegaPose-ShapeNet"
SHAPENET_MODELNET_CATEGORIES = set(
    [
        "guitar",
        "bathtub,bathing tub,bath,tub",
        "bookshelf",
        "sofa,couch,lounge",
    ]
)


def make_object_dataset(ds_name: str) -> RigidObjectDataset:
    # BOP original models
    if ds_name == "tless.cad":
        ds: RigidObjectDataset = BOPObjectDataset(
            f"{BOP_DS_DIR}/tless/models_cad", label_format="tless-{label}"
        )
    elif ds_name == "tless.eval":
        ds = BOPObjectDataset(
            f"{BOP_DS_DIR}/tless/models_eval", label_format="tless-{label}"
        )
    elif ds_name == "tless.reconst":
        ds = BOPObjectDataset(
            f"{BOP_DS_DIR}/tless/models_reconst", label_format="tless-{label}"
        )
    elif ds_name == "ycbv":
        ds = BOPObjectDataset(f"{BOP_DS_DIR}/ycbv/models", label_format="ycbv-{label}")
    elif ds_name == "hb":
        ds = BOPObjectDataset(f"{BOP_DS_DIR}/hb/models", label_format="hb-{label}")
    elif ds_name == "icbin":
        ds = BOPObjectDataset(
            f"{BOP_DS_DIR}/icbin/models", label_format="icbin-{label}"
        )
    elif ds_name == "itodd":
        ds = BOPObjectDataset(
            f"{BOP_DS_DIR}/itodd/models", label_format="itodd-{label}"
        )
    elif ds_name in {"lm", "lmo"}:
        ds = BOPObjectDataset(f"{BOP_DS_DIR}/lm/models", label_format="lm-{label}")
    elif ds_name == "tudl":
        ds = BOPObjectDataset(f"{BOP_DS_DIR}/tudl/models", label_format="tudl-{label}")
    elif ds_name == "tyol":
        ds = BOPObjectDataset(f"{BOP_DS_DIR}/tyol/models", label_format="tyol-{label}")
    elif ds_name == "ruapc":
        ds = BOPObjectDataset(
            f"{BOP_DS_DIR}/ruapc/models", label_format="ruapc-{label}"
        )
    elif ds_name == "hope":
        ds = BOPObjectDataset(f"{BOP_DS_DIR}/hope/models", label_format="hope-{label}")

    # GSO
    elif ds_name == "gso.orig":
        ds = GoogleScannedObjectDataset(GSO_DIR, split="orig")
    elif ds_name == "gso.normalized":
        ds = GoogleScannedObjectDataset(GSO_DIR, split="normalized")
    elif ds_name == "gso.panda3d":
        ds = GoogleScannedObjectDataset(GSO_DIR, split="panda3d")
    # ShapeNet
    # shapenet.{filters=20mb_50k,remove_modelnet,...}.split
    elif ds_name.startswith("shapenet."):
        ds_name = ds_name[len("shapenet.") :]

        filters_list: List[str] = []
        if ds_name.startswith("filters="):
            filter_str = ds_name.split(".")[0]
            filters_list = filter_str.split("filters=")[1].split(",")
            ds_name = ds_name[len(filter_str) + 1 :]

        model_split = ds_name
        ds = ShapeNetObjectDataset(SHAPENET_DIR, split=model_split)

        for filter_str in filters_list:
            if filter_str == "remove_modelnet":
                keep_labels = set(
                    [
                        obj.label
                        for obj in ds.objects
                        if obj.category not in SHAPENET_MODELNET_CATEGORIES
                    ]
                )
            else:
                keep_labels = set(
                    json.load(
                        open(f"{SHAPENET_DIR}/stats/shapenet_{filter_str}.json", "r")
                    )
                )
            ds = ds.filter_objects(keep_labels)
    # GSO
    # gso.{nobjects=500,...}.split
    elif ds_name.startswith("gso."):
        ds_name = ds_name[len("gso.") :]

        n_objects_: Optional[int] = None
        if ds_name.startswith("nobjects="):
            nobjects_str = ds_name.split(".")[0]
            n_objects_ = int(nobjects_str.split("=")[1])
            ds_name = ds_name[len(nobjects_str) + 1 :]

        model_split = ds_name
        ds = GoogleScannedObjectDataset(GSO_DIR, split=model_split)
        if n_objects_ is not None:
            np_random = np.random.RandomState(0)
            keep_labels = set(
                np_random.choice(
                    [obj.label for obj in ds.objects], n_objects_, replace=False
                ).tolist()
            )
            ds = ds.filter_objects(keep_labels)

    else:
        raise ValueError(ds_name)
    return ds
