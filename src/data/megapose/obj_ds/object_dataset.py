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
from dataclasses import dataclass
from io import BytesIO
import itertools
from pathlib import Path
from typing import List, Optional, Set, Tuple

# Third Party
import ipdb
from matplotlib import pyplot as plt
import numpy as np
import open3d as o3d
import pytorch3d.io as pio
import torch
from pytorch3d.structures import Meshes
from joblib import Memory
from scipy.spatial import distance

from src.third_party.bop_toolkit.bop_toolkit_lib import inout
from src.utils.pytorch3d.diff_render import normalize_verts, setup_meshes

# MegaPose
from src.data.megapose.lib3d.symmetries import (
    ContinuousSymmetry,
    DiscreteSymmetry,
    make_symmetries_poses,
)
from src.utils.pytorch3d.diff_render import render_posed_object
from src.utils.pytorch3d.mesh_utils import repeat_meshes
from src.utils.tensor_collection import TensorCollection
from src.utils.xyz_utils import calc_xyz_bp_batch, normalize_xyz


CACHE_DIR = "./.cache/obj_ds"
MEMORY = Memory(CACHE_DIR, verbose=0)


def load_mesh_p3d(mesh_path, normalize, scale):
    # load center and extent
    mesh = setup_meshes(mesh_path, scale, disable_textures=False, normalize=normalize)
    return mesh


def load_points(mesh_path, number_of_points=1000, scale=1.0, normalize=False):

    if ".obj" == mesh_path[-4:]:
        verts, faces, aux = pio.load_obj(mesh_path, load_textures=False)
        verts = verts.numpy().astype(np.float32)
    elif ".ply" == mesh_path[-4:]:
        model = inout.load_ply(mesh_path)
        verts = np.array(model["pts"], dtype=np.float32)
    else:
        raise NotImplementedError(f"Unsupported mesh format: {mesh_path}")

    if len(verts) >= number_of_points:
        # If we have enough vertices, sample without replacement
        indices = np.random.choice(len(verts), size=number_of_points, replace=False)
    else:
        # If not enough vertices, sample with replacement
        indices = np.random.choice(len(verts), size=number_of_points, replace=True)
    points = verts[indices]
    if normalize:
        points = normalize_verts(points)
    points = points * scale
    return points.astype(np.float32)


@dataclass
class PointCloud:
    points: Optional[np.ndarray] = None
    center: Optional[np.ndarray] = None
    extent: Optional[np.ndarray] = None
    diameter: float = -1.0


class RigidObject:
    def __init__(
        self,
        label: str,
        mesh_path: Path,
        point_path: Path,
        category: Optional[str] = None,
        center: Optional[np.ndarray] = None,
        extents: Optional[np.ndarray] = None,
        mesh_diameter: float = -1,
        mesh_units: str = "m",
        symmetries_discrete: List[DiscreteSymmetry] = [],
        symmetries_continuous: List[ContinuousSymmetry] = [],
        ypr_offset_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        normalize_verts=False,
        scaling_factor: float = 1.0,
        scaling_factor_mesh_units_to_meters: Optional[float] = None,
        number_of_points=1000,
        render_fov=50,
        render_dist=1.0,
        point_normalize=False,
        point_scale=1.0,
    ):
        """
        Args:
            label (str): A unique label to identify an object.
            mesh_path (Path): Path to a mesh. Multiple object types are supported.
                Please refer to downstream usage of this class for the supported formats.
                For example, when a `RigidObjectDataset`is passed to a `Panda3dSceneRenderer`,
                the user must ensure that the mesh can be loaded correctly.
            category (Optional[str], optional): Can be used to identify the object
                as one of a known category,  e.g. mug or shoes.  In the general case, an
                object does not need to belong to a category. The notion of category can also
                ambiguous. In this codebase, this is only used to parse the categories of the
                ShapeNet dataset in order to remove the instances that overlap with the test
                categories of the ModelNet dataset.
            mesh_diameter (Optional[float], optional): Diameter of the object, expressed
                the in unit of the meshes.
                This is useful for computing error some metrics like ADD<0.1d or ADD-S<0.1d.
            mesh_units (str, optional): Units in which the vertex positions are expressed.
                Can be `m`or `mm`, defaults to `m`. In the operations of this codebase,
                all mesh coordinates and poses must be expressed in meters.
                When an object is loaded, a scaling will be applied to the mesh
                to ensure its coordinates are in meters when in memory.
            symmetries_discrete (List[ContinuousSymmetry], optional):
                See https://github.com/thodan/bop_toolkit/blob/master/bop_toolkit_lib/misc.py
            symmetries_continuous (List[DiscreteSymmetry], optional):
                See https://github.com/thodan/bop_toolkit/blob/master/bop_toolkit_lib/misc.py
            ypr_offset_deg (np.ndarray, optional): A rotation offset applied to the mesh
                **only when loaded in Panda3D**. This can be useful to correct
                some mesh conventions where axes are flipped.
                Defaults to np.zeros(3, dtype=float).
            scaling_factor (float, optional): An extra scaling factor that can
                be applied to the mesh to rescale it. Defaults to 1.0. Please note
                that this is applied on top applying the scale factor to resize the
                mesh to meters.
                For example, if you have a mesh with coordinates expressed in `mm`
                which you want to resize to 10% of its size,
                you should pass `mesh_units=mm`and `scaling_factor=0.1`.
                Note that `mesh_units=m` and `scaling_factor=100` would be strictly equivalent.
            scaling_factor_mesh_units_to_meters (float, optional): Can be used
                instead of the mesh_units argument. This is the scale that converts
                mesh units to meters.
        """

        self.label = label
        self.category = category
        self.mesh_path = mesh_path
        self.point_path = point_path
        self.mesh_units = mesh_units
        self.number_of_points = number_of_points
        self.normalize_verts = normalize_verts
        self.render_fov = render_fov
        self.render_dist = render_dist

        self.point_normalize = point_normalize
        self.point_scale = point_scale

        if scaling_factor_mesh_units_to_meters is not None:
            self.scaling_factor_mesh_units_to_meters = (
                scaling_factor_mesh_units_to_meters
            )
        else:
            self.scaling_factor_mesh_units_to_meters = {
                "m": 1.0,
                "mm": 0.001,
            }[self.mesh_units]
        self.scaling_factor = scaling_factor

        self.diameter_meters = None
        if mesh_diameter is not None:
            self.mesh_diameter = mesh_diameter
            self.diameter_meters = (
                mesh_diameter * self.scaling_factor_mesh_units_to_meters
            )
        self.extents = None
        if extents is not None:
            self.extents = extents * self.scaling_factor_mesh_units_to_meters

        self.center = None
        if center is not None:
            self.center = center * self.scaling_factor_mesh_units_to_meters

        self.symmetries_discrete = symmetries_discrete
        self.symmetries_continuous = symmetries_continuous
        self.ypr_offset_deg = ypr_offset_deg

    @property
    def is_symmetric(self) -> bool:
        return len(self.symmetries_discrete) > 0 or len(self.symmetries_continuous) > 0

    @property
    def scale(self) -> float:
        """Returns the scale factor that converts the mesh to desired units."""
        return self.scaling_factor_mesh_units_to_meters * self.scaling_factor

    def make_symmetry_poses(self, n_symmetries_continuous: int = 64) -> np.ndarray:
        """Generates the set of object symmetries.

        Returns:
            (num_symmetries, 4, 4) array
        """
        return make_symmetries_poses(
            self.symmetries_discrete,
            self.symmetries_continuous,
            n_symmetries_continuous=n_symmetries_continuous,
            scale=self.scale,
        )

    @property
    def model_p3d(self):
        return load_mesh_p3d(
            str(self.mesh_path), self.normalize_verts, scale=self.scale
        )

    @property
    def model_center(self):
        if self.normalize_verts:
            return np.zeros(3, dtype=np.float32)
        elif self.center is not None:
            return self.center
        return np.mean(self.points, 0)

    @property
    def points(self):
        return load_points(
            str(self.point_path),
            number_of_points=self.number_of_points,
            scale=self.point_scale,
            normalize=self.point_normalize,
        )


class RigidObjectDataset:
    def __init__(
        self,
        objects: List[RigidObject],
    ):
        self.list_objects = objects
        self.label_to_objects = {obj.label: obj for obj in objects}

        self.label2index = {obj.label: idx for idx, obj in enumerate(objects)}

        if len(self.list_objects) != len(self.label_to_objects):
            raise RuntimeError("There are objects with duplicate labels")

    def __getitem__(self, idx: int) -> RigidObject:
        return self.list_objects[idx]

    def get_object_by_label(self, label: str) -> RigidObject:
        return self.label_to_objects[label]

    def get_id_by_label(self, label: str) -> int:
        """Returns the index of the object with the given label."""
        if label not in self.label2index:
            raise KeyError(f"Label {label} not found in the dataset.")
        return self.label2index[label]

    def __len__(self) -> int:
        return len(self.list_objects)

    def contains(self, label: str) -> bool:
        return label in self.label_to_objects

    @property
    def objects(self) -> List[RigidObject]:
        """Returns a list of objects in this dataset."""
        return self.list_objects

    def filter_objects(self, keep_labels: Set[str]) -> "RigidObjectDataset":
        list_objects = [obj for obj in self.list_objects if obj.label in keep_labels]
        return RigidObjectDataset(list_objects)


def append_dataset_name_to_object_labels(
    ds_name: str, object_dataset: RigidObjectDataset
) -> RigidObjectDataset:
    for obj in object_dataset.list_objects:
        obj.label = f"ds_name={ds_name}_{obj.label}"
    return object_dataset


def concat_object_datasets(datasets: List[RigidObjectDataset]) -> RigidObjectDataset:
    objects = list(itertools.chain.from_iterable([ds.list_objects for ds in datasets]))
    return RigidObjectDataset(objects)
