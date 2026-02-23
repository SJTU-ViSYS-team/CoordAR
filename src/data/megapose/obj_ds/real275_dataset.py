# Standard Library
import glob
import json
import os
from pathlib import Path
from typing import Dict, List
import os.path as osp
import ipdb
import numpy as np

from bop_toolkit_lib.misc import calc_pts_diameter2

# Local Folder
from .object_dataset import MEMORY, RigidObject, RigidObjectDataset


def extents_from_points(points: np.ndarray) -> np.ndarray:
    """Calculate the extents of a point cloud."""
    min_point = np.min(points, axis=0)
    max_point = np.max(points, axis=0)
    return max_point - min_point


def get_obj_ids(root_dir: str, split_type: str, split: str):
    obj_files = glob.glob(f"{root_dir}/obj_models/{split_type}_{split}/*.obj")
    obj_ids = []
    id2label = {}
    label2id = {}
    for obj_id, obj_file in enumerate(sorted(obj_files), start=1):
        obj_label = osp.basename(obj_file).split(".")[0]
        obj_ids.append(obj_id)
        id2label[obj_id] = obj_label
        label2id[obj_label] = obj_id
    return obj_ids, id2label, label2id


class Real275ObjectDataset(RigidObjectDataset):
    def __init__(self, root_dir: str, split="test", split_type="real"):
        root_dir = os.path.abspath(root_dir)
        self.root_dir = Path(root_dir)
        self.split = split
        self.split_type = split_type

        scaling_factor = 1.0

        self.obj_ids, self.id2label, self.label2id = get_obj_ids(
            root_dir, split_type, split
        )
        objects = []
        for object_id in self.obj_ids:
            label = self.id2label[object_id]
            model_path = (
                self.root_dir
                / "obj_models"
                / f"{self.split_type}_{self.split}"
                / f"{label}.obj"
            )
            vertices = np.loadtxt(
                str(model_path).replace(".obj", "_vertices.txt"), dtype=np.float32
            )
            extents = extents_from_points(vertices)
            vertices_sample = vertices[
                np.random.choice(
                    vertices.shape[0], size=min(1000, vertices.shape[0]), replace=False
                )
            ]
            diameter = calc_pts_diameter2(vertices_sample).item()

            # raise NotImplementedError(
            #     "Real275ObjectDataset does not support symmetries yet."
            # )

            obj = RigidObject(
                label=f"real275_{label}",
                mesh_path=model_path,
                point_path=model_path,
                mesh_diameter=diameter,
                scaling_factor=scaling_factor,
                extents=extents,
            )
            objects.append(obj)
        super().__init__(objects)


# python -m src.data.megapose.obj_ds.real275_dataset
if __name__ == "__main__":
    real275_dataset = Real275ObjectDataset("data/Real275")
    obj = real275_dataset.get_object_by_label("real275_bottle_red_stanford_norm")
    ipdb.set_trace()
