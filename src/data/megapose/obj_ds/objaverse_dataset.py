import glob
import json
import os
from pathlib import Path
from typing import Dict

import ipdb
import numpy as np
import objaverse


from .object_dataset import MEMORY, RigidObject, RigidObjectDataset


def load_objaverse_id2meta(gso_root: str) -> Dict:
    meta_path = Path(gso_root) / "metaData_Objaverse.json"
    if not meta_path.exists():
        return {}
    obj_id_infos = json.load(meta_path.open())
    id_mapping = {}
    for info in obj_id_infos:
        id_mapping[info["label"]] = np.array(info["extents"], dtype=np.float32)
    return id_mapping


class ObjaverseDataset(RigidObjectDataset):
    def __init__(self, objaverse_root: str, split: str = "orig", number_of_points=2048):
        objaverse_root = os.path.abspath(objaverse_root)
        self.objaverse_root = Path(objaverse_root)
        objaverse.BASE_PATH = objaverse_root
        objaverse._VERSIONED_PATH = os.path.join(objaverse.BASE_PATH, "hf-objaverse-v1")

        print("Loading Objaverse dataset from:", objaverse_root)
        lvis_annotations = objaverse.load_lvis_annotations()
        lvis_uids = []
        for category, uids in lvis_annotations.items():
            lvis_uids.extend(uids)

        print("Total number of LVIS uids:", len(lvis_uids))
        # list all glb path
        glb_paths = glob.glob(
            os.path.join(objaverse._VERSIONED_PATH, "glbs", "**", "*.glb"),
            recursive=True,
        )
        # filter lvis_uids by glb_uids
        uid_to_path = {Path(path).stem: path for path in glb_paths}
        self.meta_data = load_objaverse_id2meta(objaverse_root)

        objects = []
        for object_id in lvis_uids:
            model_path = uid_to_path.get(object_id, None)
            if model_path is None:
                print(f"Object {object_id} not found in glb paths.")
                continue
            label = f"objaverse_{object_id}"
            # object_id is the name of the object, not the number
            point_path = Path(f"{objaverse_root}/points/objaverse_{object_id}.ply")
            obj = RigidObject(
                label=label,
                mesh_path=Path(model_path),
                point_path=point_path,
                normalize_verts=True,
                scaling_factor=0.5,
                extents=self.meta_data.get(label, None),
                number_of_points=number_of_points,
                point_scale=1.0,
                point_normalize=False,
            )
            objects.append(obj)
        super().__init__(objects)


# python -m src.data.megapose.obj_ds.objaverse_dataset
if __name__ == "__main__":
    # 00a1a602456f4eb188b522d7ef19e81b
    # 0f81453b4d6e49fea0cf56264698016c
    # 6abb1ba39e414450a20d7e8ac096d016 # has inherit rotation
    # dbfb19290759455db5fc75c23a878249 # still error
    # 059ddf8d773748a0aa32c778897e711e # still error
    # d177e1fbcce940cba32e434cc5a62f1a # still error
    # ebf5bea9caca46ffaf01714a78893f55 # still error
    # 8907f8c7c21842b7bf059b077ff99a35 # still error
    # 56df1826f18f4669a99fbbfb1204e706 # large depth error
    objaverse_dataset = ObjaverseDataset("./data/Objaverse_cache")
    obj = objaverse_dataset.get_object_by_label(
        "objaverse_0a0c7e40a66d4fd090f549599f2f2c9d"
    )
    obj.model_p3d
    ipdb.set_trace()
