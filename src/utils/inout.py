import os
import errno
import shutil
import numpy as np
import json
import pandas as pd
import ruamel.yaml as yaml


def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass


def del_folder(path):
    try:
        shutil.rmtree(path)
    except OSError:
        pass


def write_txt(path, list_files):
    with open(path, "w") as f:
        for idx in list_files:
            f.write(idx + "\n")
        f.close()


def open_txt(path):
    with open(path, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines


def save_json(path, info):
    # save to json without sorting keys or changing format
    with open(path, "w") as f:
        json.dump(info, f, indent=4)


def get_root_project():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def casting_format_to_save_json(data):
    # casting for every keys in dict to list so that it can be saved as json
    for key in data.keys():
        if (
            isinstance(data[key][0], np.ndarray)
            or isinstance(data[key][0], np.float32)
            or isinstance(data[key][0], np.float64)
            or isinstance(data[key][0], np.int32)
            or isinstance(data[key][0], np.int64)
        ):
            data[key] = np.array(data[key]).tolist()
    return data


def convert_dict_to_dataframe(data_dict, column_names, convert_to_list=True):
    if convert_to_list:
        data_list = list(data_dict.items())
    else:
        data_list = data_dict
    df = pd.DataFrame(data_list, columns=column_names)
    return df


def convert_list_to_dataframe(data_list):
    column_names = [k for k in data_list[0].keys()]
    data = [[] for _ in range(len(data_list))]
    for idx, item in enumerate(data_list):
        for key in item.keys():
            data[idx].append(item[key])
    df = pd.DataFrame(data, columns=column_names)
    return df


if __name__ == "__main__":
    root_dir = get_root_project()
    print(root_dir)
