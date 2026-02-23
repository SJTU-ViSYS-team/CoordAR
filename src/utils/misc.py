import os
import random
import shutil
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import importlib
from torch.autograd import Variable
import functools


def LD2DL(LD: List[Dict]) -> Dict[str, List[Any]]:
    # 数据量大时慎用，容易耗尽内存
    all_keys = set([])
    for d in LD:
        for k in d.keys():
            if k not in all_keys:
                all_keys.add(k)
    DL = {}
    for k in all_keys:
        DL[k] = []
        for dic in LD:
            if k in dic:
                DL[k].append(dic[k])
    return DL


def DL2LD(DL: Dict[str, List]) -> List[Dict]:
    all_keys = list(DL.keys())
    n = len(DL[all_keys[0]])
    LD = [{} for _ in range(n)]
    for k in all_keys:
        for i in range(n):
            LD[i][k] = DL[k][i]

    return LD


def prepare_dir(dir_name: str, exist_ok=True, clean=False):
    if clean and os.path.isdir(dir_name):
        shutil.rmtree(dir_name)
    if os.path.splitext(dir_name)[1] != "":  # file like
        dir_name = os.path.split(dir_name)[0]
    os.makedirs(dir_name, exist_ok=exist_ok)
    return dir_name


def visualize_internal_images(output, images, fontScale=1, org=(50, 50)):
    img_list = []
    plt.figure()
    for i in range(len(images)):
        name = images[i][0]
        if isinstance(images[i][1], np.ndarray):
            image = images[i][1]
        else:
            image = images[i][1].numpy().astype(np.uint8)
        image = np.ascontiguousarray(image)
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # org

        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 2
        image = cv2.putText(
            image, f"{name}", org, font, fontScale, color, thickness, cv2.LINE_AA
        )
        img_list.append(image)
    cv2.imwrite(output, np.concatenate(img_list, axis=1))


class nameddict(dict):
    __getattr__ = dict.__getitem__

    def __init__(self, *args, **kwargs):
        return super(nameddict, self).__init__(*args, **kwargs)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def to_device(src_data, device):
    if isinstance(src_data, tuple) or isinstance(src_data, list):
        dst_data = []
        for item in src_data:
            dst_data.append(to_device(item, device))
        if isinstance(src_data, tuple):
            dst_data = tuple(dst_data)
    elif isinstance(src_data, dict):
        dst_data = {}
        for key, value in src_data.items():
            dst_data[key] = to_device(value, device)
    elif isinstance(src_data, torch.Tensor):
        dst_data = src_data.detach().to(device)
    else:
        try:
            dst_data = src_data.to(device)
        except:
            dst_data = src_data
    # elif isinstance(src_data, str):
    #     return src_data
    # else:
    # raise TypeError(f"unexpected type:{type(src_data)},{src_data}")
    return dst_data


def to_half(src_data):
    if isinstance(src_data, tuple) or isinstance(src_data, list):
        dst_data = []
        for item in src_data:
            dst_data.append(to_half(item))
        if isinstance(src_data, tuple):
            dst_data = tuple(dst_data)
    elif isinstance(src_data, dict):
        dst_data = {}
        for key, value in src_data.items():
            dst_data[key] = to_half(value)
    elif isinstance(src_data, torch.Tensor) and (
        src_data.dtype == torch.float32 or src_data.dtype == torch.float64
    ):
        dst_data = src_data.half()
    else:
        dst_data = src_data
    return dst_data


def read_file_by_lines(file_path):
    lines = []
    try:
        with open(file_path, "r") as file:
            for line in file:
                lines.append(
                    line.strip()
                )  # Add each line to the list after removing leading/trailing whitespaces and newlines
    except FileNotFoundError:
        print("File not found. Please provide a valid file path.")
    except Exception as e:
        print("Error occurred while reading the file:", str(e))

    return lines


def init_instance(args):
    if isinstance(args, list):
        for i in range(len(args)):
            args[i] = init_instance(args[i])
        return args
    elif isinstance(args, dict):
        if "class_path" in args and "init_args" in args:
            class_path = args["class_path"]
            cls = getattr(
                importlib.import_module(".".join(class_path.split(".")[:-1])),
                class_path.split(".")[-1],
            )
            init_args = init_instance(args["init_args"])
            return cls(**init_args)
        instance = {}
        for k, v in args.items():
            instance[k] = init_instance(v)
        return instance
    else:
        return args


def collate_dcnet(data):
    tensors = LD2DL(data)
    for k, v in tensors.items():
        if isinstance(v[0], np.ndarray):
            v = [torch.from_numpy(item) for item in v]
            tensors[k] = torch.stack(v, 0)
        elif isinstance(v[0], torch.Tensor):
            tensors[k] = torch.stack(v, 0)
        elif isinstance(v[0], str):
            tensors[k] = v
        elif isinstance(v[0], int):
            tensors[k] = torch.tensor(v)
    return tensors


def lazy_property(function):
    # https://danijar.com/structuring-your-tensorflow-models/
    attribute = "_cache_" + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


def print_model_param_size(model):
    # 定义总参数量、可训练参数量及非可训练参数量变量
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    # 遍历model.parameters()返回的全局参数列表
    for param in model.parameters():
        mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
        Total_params += mulValue  # 总参数量
        if param.requires_grad:
            Trainable_params += mulValue  # 可训练参数量
        else:
            NonTrainable_params += mulValue  # 非可训练参数量

    print(f"Total params: {Total_params/1024/1024:.2f}M")
    print(f"Trainable params: {Trainable_params/1024/1024:.2f}M")
    print(f"Non-trainable params: {NonTrainable_params/1024/1024:.2f}M")
