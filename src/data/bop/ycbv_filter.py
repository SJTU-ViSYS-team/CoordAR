from glob import glob

import ipdb
import numpy as np


class YCBVKeyframeFilter:
    """
    Filter for YCBV keyframes.
    """

    def __init__(self, bop_root):
        self.bop_root = bop_root
        self.keyframes = self._load_keyframes()

    def _load_keyframes(self):
        """
        Load the list of keyframes from the BOP dataset.
        """
        key_frame_path = f"{self.bop_root}/ycbv/keyframe.txt"
        # readlines
        keyframes = {}
        with open(key_frame_path, "r") as f:
            for line in f.readlines():
                scene_id, frame_id = map(int, line.strip().split("/"))
                keyframes.setdefault(scene_id, {}).setdefault(
                    frame_id, True
                )  # keyframe starts from 0, bop starts from 1
        return keyframes

    def get_signature(self):
        return "ycbv_keyframe_filter"

    def is_filtered(self, scene_id, im_id, obj_id):
        return scene_id not in self.keyframes or im_id not in self.keyframes[scene_id]


# python -m src.data.bop.ycbv_filter
if __name__ == "__main__":
    filter = YCBVKeyframeFilter("./data/BOP")
    print(filter.keyframes)
    print(filter.is_filtered(1, 1, 1))
    print(filter.is_filtered(48, 1, 1))
