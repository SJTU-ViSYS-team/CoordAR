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

import ipdb
from matplotlib import pyplot as plt
import numpy as np
import torch
from src.data.megapose.config import MEMORY
from src.utils.pytorch3d.diff_render import render_posed_object

# Local Folder
from .object_dataset import RigidObject, RigidObjectDataset


class ShapeNetSynset:
    def __init__(self, synset_id, name):
        self.synset_id = synset_id
        self.name = name
        self.parents = []
        self.children = []
        self.models = []
        self.models_descendants = []


class ShapeNetModel:
    def __init__(self, synset_id, source_id):
        self.synset_id = synset_id
        self.source_id = source_id


@MEMORY.cache
def make_shapenet_infos(shapenet_dir, model_name):
    # TODO: This probably has issues / is poorly implemented and very slow
    shapenet_dir = Path(shapenet_dir)
    taxonomy_path = shapenet_dir / "taxonomy.json"
    taxonomy = json.loads(taxonomy_path.read_text())

    synset_id_to_synset = dict()

    def get_synset(synset_id):
        if synset_id not in synset_id_to_synset:
            synset = ShapeNetSynset(synset_id, synset_dict["name"])
            synset_id_to_synset[synset_id] = synset
        else:
            synset = synset_id_to_synset[synset_id]
        return synset

    for synset_dict in taxonomy:
        synset_id = synset_dict["synsetId"]
        synset = get_synset(synset_id)
        for child_synset_id in synset_dict["children"]:
            child_synset = get_synset(child_synset_id)
            child_synset.parents.append(synset)

    def model_exists(model_dir):
        model_dir_ = model_dir / "models"
        return (model_dir_ / model_name).exists()

    for synset in synset_id_to_synset.values():
        synset_dir = shapenet_dir / synset.synset_id
        if synset_dir.exists():
            model_dirs = list(synset_dir.iterdir())
        else:
            model_dirs = []
        model_names = [
            model_dir.name for model_dir in model_dirs if model_exists(model_dir)
        ]
        synset.models = model_names

    def get_descendants(synset):
        if len(synset.children) == 0:
            return synset.models
        else:
            return sum([get_descendants(child) for child in children])

    for synset in synset_id_to_synset.values():
        synset.models_descendants = get_descendants(synset)
    return list(synset_id_to_synset.values())


@MEMORY.cache
def load_shapenet_id2name(megapose_dir):
    obj_id_infos = json.load((Path(megapose_dir) / "shapenet_models.json").open())
    id_mapping = {}
    for info in obj_id_infos:
        id_mapping[info["obj_id"]] = (
            f"shapenet_{info['shapenet_synset_id']}_{info['shapenet_source_id']}"
        )
    return id_mapping


class ShapeNetObjectDataset(RigidObjectDataset):
    def __init__(
        self,
        shapenet_root: str,
        split: str = "orig",
    ):
        self.shapenet_dir = Path(os.path.abspath(shapenet_root))

        if split == "orig":
            model_name = "model_normalized.obj"
            ypr_offset_deg = (0.0, 0.0, 0.0)
        elif split == "panda3d_bam":
            model_name = "model_normalized_binormals.bam"
            ypr_offset_deg = (0.0, -90.0, 0.0)
        elif split == "pointcloud":
            model_name = "model_normalized_pointcloud.obj"
            ypr_offset_deg = (0.0, 0.0, 0.0)
        else:
            raise ValueError("split")

        synsets = make_shapenet_infos(self.shapenet_dir, model_name)
        main_synsets = [
            synset
            for synset in synsets
            if len(synset.parents) == 0 and len(synset.models_descendants) > 0
        ]
        objects = []

        for synset in main_synsets:
            for source_id in synset.models_descendants:
                model_path = (
                    self.shapenet_dir
                    / synset.synset_id
                    / source_id
                    / "models"
                    / model_name
                )
                label = f"shapenet_{synset.synset_id}_{source_id}"
                category = synset.name
                obj = RigidObject(
                    label=label,
                    category=category,
                    mesh_path=model_path,
                    scaling_factor=0.1,
                    ypr_offset_deg=ypr_offset_deg,
                )
                objects.append(obj)
        super().__init__(objects)


# python -m src.data.megapose.obj_ds.shapenet_object_dataset
if __name__ == "__main__":
    from pytorch3d.structures import join_meshes_as_scene, join_meshes_as_batch

    dataset = ShapeNetObjectDataset(
        "data/MegaPose-ShapeNetCore/shapenetcorev2/models_orig"
    )
    intrinsic = torch.tensor([[525, 0.0, 256], [0.0, 525, 256], [0.0, 0.0, 1.0]])
    img_size = [512, 512]

    obj_idx = 10001

    pose = torch.eye(4)
    pose[2, 3] = 0.1

    print("number of objects:", len(dataset))
    print(dataset[obj_idx].model_p3d)
    print(dataset[obj_idx].points.diameter)
    print(dataset[obj_idx].mesh_path)

    # the result is strange
    rgb, depth = render_posed_object(
        join_meshes_as_batch([dataset[obj_idx].model_p3d]),
        pose.unsqueeze(0),
        intrinsic.unsqueeze(0),
        img_size,
        re_render_iters=10,
        light_loc=(0.1, -0.1, -0.1),
        znear=0.001,
        zfar=10,
    )
    depth = depth.squeeze(1)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(rgb[0].permute(1, 2, 0).cpu().numpy())
    ax[1].imshow(depth[0].cpu().numpy())
    plt.show()
