import os
import shutil


def prepare_dir(dir_name: str, exist_ok=True, clean=False):
    if clean and os.path.isdir(dir_name):
        shutil.rmtree(dir_name)
    if os.path.splitext(dir_name)[1] != "":  # file like
        dir_name = os.path.split(dir_name)[0]
    os.makedirs(dir_name, exist_ok=exist_ok)
    return dir_name
