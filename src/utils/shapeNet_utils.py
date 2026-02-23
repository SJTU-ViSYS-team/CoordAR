import os
import json
import numpy as np
from PIL import Image
from src.utils.lib3d.numpy import get_root_project

intrinsic = np.array([[262.5, 0.0, 128], [0.0, 262.5, 128], [0.0, 0.0, 1.0]])
black_rgb = Image.new(
    "RGB",
    (256, 256),
    (0, 0, 0),
)
train_categories = [
    "airplane",
    "bench",
    "cabinet",
    "car",
    "chair",
    "display",
    "lamp",
    "loudspeaker",
    "rifle",
    "sofa",
    "table",
    "telephone",
    "watercraft",  # "vessel" in the paper
]

test_categories = [
    "bottle",
    "bus",
    "clock",
    "dishwasher",
    "guitar",
    "mug",
    "pistol",
    "skateboard",
    "train",
    "washer",
]


def get_shapeNet_mapping():
    root_repo = get_root_project()
    path_shapenet_id2cat = os.path.join(root_repo, "src/utils/shapeNet_id2cat_v2.json")
    with open(path_shapenet_id2cat) as json_file:
        id2cat_mapping = json.load(json_file)
    # create inverse mapping
    cat2id_mapping = {}
    for key, value in id2cat_mapping.items():
        cat2id_mapping[value] = key
    return id2cat_mapping, cat2id_mapping


def open_image(path):
    img = Image.open(path)
    rgb = black_rgb.copy()
    rgb.paste(img, mask=img.getchannel("A"))
    return np.array(rgb)


def open_pose(path, img_name, view_id):
    poses = np.load(path)
    return poses[img_name][view_id]
