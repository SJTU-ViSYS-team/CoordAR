# Standard Library
import json
import os
from pathlib import Path
from typing import Dict, List

import ipdb
import numpy as np

from src.data.megapose.obj_ds.bop_object_dataset import get_model_info

# Local Folder
from .object_dataset import MEMORY, RigidObject, RigidObjectDataset


class OV9DObjectDataset(RigidObjectDataset):
    def __init__(self, ov9d_root: str, number_of_points=2048):
        ov9d_root = os.path.abspath(ov9d_root)
        self.ov9d_root = Path(ov9d_root)

        scaling_factor = 1.0

        self.models_info = json.load((Path(ov9d_root) / "models_info.json").open())
        objects = []
        for object_id in self.models_info.keys():
            object_id = int(object_id)
            info = self.models_info[str(object_id)]
            model_path = self.ov9d_root / "models" / f"obj_{object_id:06d}.ply"
            label = f"ov9d_{object_id:06d}"
            center, extents = get_model_info(info)
            obj = RigidObject(
                label=label,
                mesh_units="mm",
                mesh_path=model_path,
                point_path=model_path,
                mesh_diameter=info["diameter"],
                scaling_factor=scaling_factor,
                extents=extents,
                center=center,
                number_of_points=number_of_points,
            )
            objects.append(obj)
        super().__init__(objects)

    def get_obj_ids(self) -> List[int]:
        return sorted([int(object_id) for object_id in self.models_info.keys()])


# python -m src.data.megapose.obj_ds.ov9d_dataset
if __name__ == "__main__":
    # Threshold_Porcelain_Teapot_White
    ov9d_dataset = OV9DObjectDataset("data/OV9D")
    obj = ov9d_dataset.get_object_by_label("ov9d_000001")
    ipdb.set_trace()
