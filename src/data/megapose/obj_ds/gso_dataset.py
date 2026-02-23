"""
Copyright (c) 2022 Inria & NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Standard Library
import json
import os
from pathlib import Path
from typing import Dict, List

import ipdb
import numpy as np

# Local Folder
from .object_dataset import MEMORY, RigidObject, RigidObjectDataset


@MEMORY.cache
def make_gso_infos(gso_dir: Path, model_name: str = "model.obj") -> List[str]:
    gso_dir = Path(gso_dir)
    models_dir = gso_dir.iterdir()
    invalid_ids = set(json.loads((gso_dir.parent / "invalid_meshes.json").read_text()))
    object_ids = []
    for model_dir in models_dir:
        if (model_dir / "meshes" / model_name).exists():
            object_id = model_dir.name
            if object_id not in invalid_ids:
                object_ids.append(object_id)
    object_ids.sort()
    return object_ids


@MEMORY.cache
def load_gso_id2name(megapose_root: str):
    obj_id_infos = json.load((Path(megapose_root) / "gso_models.json").open())
    id_mapping = {}
    for info in obj_id_infos:
        id_mapping[info["obj_id"]] = f"gso_{info['gso_id']}"
    return id_mapping


def load_gso_id2meta(gso_root: str) -> Dict:
    meta_path = Path(gso_root) / "metaData_GSO.json"
    if not meta_path.exists():
        return {}
    obj_id_infos = json.load(meta_path.open())
    id_mapping = {}
    for info in obj_id_infos:
        id_mapping[info["label"]] = np.array(info["extents"], dtype=np.float32)
    return id_mapping


class GoogleScannedObjectDataset(RigidObjectDataset):
    def __init__(self, gso_root: str, split: str = "orig", number_of_points=2048):
        gso_root = os.path.abspath(gso_root)
        self.gso_root = Path(gso_root)
        self.gso_dir = self.gso_root / f"models_{split}"

        if split == "orig":
            scaling_factor = 1.0
        elif split in {"normalized", "pointcloud"}:
            scaling_factor = 0.1

        object_ids = make_gso_infos(self.gso_dir)
        self.meta_data = load_gso_id2meta(gso_root)
        objects = []
        for object_id in object_ids:
            model_path = self.gso_dir / object_id / "meshes" / "model.obj"
            label = f"gso_{object_id}"
            # object_id is the name of the object, not the number
            point_path = (
                Path(str(self.gso_dir).replace(f"models_{split}", "models_pointcloud"))
                / object_id
                / "meshes"
                / "model.obj"
            )
            # point_path = model_path
            obj = RigidObject(
                label=label,
                mesh_path=model_path,
                point_path=point_path,
                normalize_verts=True,
                scaling_factor=scaling_factor,
                extents=self.meta_data.get(label, None),
                number_of_points=number_of_points,
                point_scale=1.0,
                point_normalize=True,
            )
            objects.append(obj)
        super().__init__(objects)


# python -m src.data.megapose.obj_ds.gso_dataset
if __name__ == "__main__":
    # Threshold_Porcelain_Teapot_White
    gso_dataset = GoogleScannedObjectDataset("data/google_scanned_objects")
    obj = gso_dataset.get_object_by_label("gso_Threshold_Porcelain_Teapot_White")
    ipdb.set_trace()
