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

# Third Party
import ipdb
import numpy as np

from src.data.megapose.lib3d.symmetries import ContinuousSymmetry, DiscreteSymmetry

# Local Folder
from .object_dataset import RigidObject, RigidObjectDataset


def get_model_info(bop_info):
    size_x = bop_info["size_x"]
    size_y = bop_info["size_y"]
    size_z = bop_info["size_z"]
    extents = np.array([size_x, size_y, size_z]).astype(np.float32)
    center_x = bop_info["min_x"] + size_x / 2.0
    center_y = bop_info["min_y"] + size_y / 2.0
    center_z = bop_info["min_z"] + size_z / 2.0
    center = np.array([center_x, center_y, center_z]).astype(np.float32)

    return center, extents


class BOPObjectDataset(RigidObjectDataset):
    def __init__(
        self, ds_dir_str: str, label_format: str = "{label}", number_of_points=2048
    ):
        ds_dir = Path(os.path.abspath(ds_dir_str))
        infos_file = ds_dir / "models_info.json"
        infos = json.loads(infos_file.read_text())
        objects = []
        for obj_id, bop_info in infos.items():
            obj_id = int(obj_id)
            obj_label = f"obj_{obj_id:06d}"
            mesh_path = (ds_dir / obj_label).with_suffix(".ply").as_posix()
            symmetries_discrete = [
                DiscreteSymmetry(pose=np.array(x).reshape((4, 4)))
                for x in bop_info.get("symmetries_discrete", [])
            ]
            symmetries_continuous = [
                ContinuousSymmetry(offset=d["offset"], axis=d["axis"])
                for d in bop_info.get("symmetries_continuous", [])
            ]
            center, extents = get_model_info(bop_info)
            obj = RigidObject(
                label=label_format.format(label=obj_label),
                mesh_path=Path(mesh_path),
                point_path=Path(mesh_path),
                mesh_units="mm",
                symmetries_discrete=symmetries_discrete,
                symmetries_continuous=symmetries_continuous,
                mesh_diameter=bop_info["diameter"],
                extents=extents,
                center=center,
                number_of_points=number_of_points,
                point_normalize=False,
                point_scale=0.001,
            )
            objects.append(obj)

        self.ds_dir = ds_dir
        super().__init__(objects)


# python -m src.data.megapose.obj_ds.bop_object_dataset
if __name__ == "__main__":
    gso_dataset = BOPObjectDataset("data/Real275/BOP/real275/models")
    obj = gso_dataset.get_object_by_label("obj_000001")

    sm = obj.make_symmetry_poses()
    pts = obj.points
    # show 3d points using matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # transform pts by symmetry poses
    ipdb.set_trace()
    pts_sym = (sm[32][:3, :3] @ pts.T + sm[32][:3, 3:4]).T

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="r", marker="o", s=1)
    ax.scatter(pts_sym[:, 0], pts_sym[:, 1], pts_sym[:, 2], c="b", marker="o", s=1)
    # show axis
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()
