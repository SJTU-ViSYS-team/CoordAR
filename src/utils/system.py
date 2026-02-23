import functools
import gc
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path
import os
import fcntl
import time

import torch


def show_memory():
    print("*" * 60)
    objects_list = []
    for obj in gc.get_objects():
        size = sys.getsizeof(obj)
        objects_list.append((obj, size))
    for obj, size in sorted(objects_list, key=lambda x: x[1], reverse=True)[:10]:
        print(
            f"OBJ: {id(obj)}, TYPE: {type(obj)} SIZE: {size/1024/1024:.2f}MB {str(obj)[:100]}"
        )


def get_time_str(fmt="%Y%m%d_%H%M%S", hours_offset=8):
    # get UTC+8 time by default
    # set hours_offset to 0 to get UTC time
    # use utc time to avoid the problem of mis-configured timezone on some machines
    return (datetime.utcnow() + timedelta(hours=hours_offset)).strftime(fmt)


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    """
    Increments a file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.

    If the path exists and exist_ok is not set to True, the path will be incremented by appending a number and sep to
    the end of the path. If the path is a file, the file extension will be preserved. If the path is a directory, the
    number will be appended directly to the end of the path. If mkdir is set to True, the path will be created as a
    directory if it does not already exist.

    Args:
        path (str, pathlib.Path): Path to increment.
        exist_ok (bool, optional): If True, the path will not be incremented and returned as-is. Defaults to False.
        sep (str, optional): Separator to use between the path and the incrementation number. Defaults to ''.
        mkdir (bool, optional): Create a directory if it does not exist. Defaults to False.

    Returns:
        (pathlib.Path): Incremented path.
    """
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (
            (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        )

        # Method 1
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


class Timer:
    def __init__(self):
        self.start_times = {}
        self.end_times = {}

    def start(self, event: str):
        self.start_times[event] = datetime.now()

    def end(self, event: str):
        if event in self.start_times:
            self.end_times[event] = datetime.now()
            return self.end_times[event] - self.start_times[event]
        else:
            return None

    def elapsed(self, event: str):
        if event in self.start_times:
            if event in self.end_times:
                return self.end_times[event] - self.start_times[event]
            else:
                return datetime.now() - self.start_times[event]
        else:
            return None

    def elapsed_seconds(self, event: str):
        elapsed_time = self.elapsed(event)
        if elapsed_time is not None:
            return elapsed_time.total_seconds()
        else:
            return -1

    def summary(self):
        for event in self.start_times:
            if event in self.end_times:
                print(f"{event}: {self.end_times[event] - self.start_times[event]}s")
            else:
                print(f"{event}: {datetime.now() - self.start_times[event]}s")


def dump_with_lock(file_path, content):
    try:
        # 打开文件并获取锁
        with open(file_path, "wb") as file:
            fcntl.flock(file, fcntl.LOCK_EX)  # 获取独占锁

            pickle.dump(content, file)

            # 释放锁
            fcntl.flock(file, fcntl.LOCK_UN)
    except Exception as e:
        print(f"Error writing to file: {e}")


def print_time(func):
    def wrapper(*args, **kwargs):
        import time

        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} takes {time.time() - start:.4f} seconds")
        return result

    return wrapper


def show_gpu_memory(device="cuda:0"):
    import torch

    # 打印当前分配的显存
    current_memory = torch.cuda.memory_allocated(device) / (1024**3)  # 转换为GB
    print(f"当前显存占用: {current_memory:.2f} GB")

    # 打印最大显存使用量
    max_memory = torch.cuda.max_memory_allocated(device) / (1024**3)  # 转换为GB
    print(f"最大显存占用: {max_memory:.2f} GB")


def show_param_size(model):
    # 打印模型参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params/1024**2:.2f}M")


def profile_time_memory(func):
    """
    装饰器：打印函数执行时间和 GPU 显存消耗
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            print("⚠️ CUDA 不可用，只统计耗时。")
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"[{func.__name__}] 耗时: {end - start:.4f} 秒")
            return result

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()
        start_time = time.time()

        result = func(*args, **kwargs)

        torch.cuda.synchronize()
        end_time = time.time()
        end_mem = torch.cuda.memory_allocated()
        peak_mem = torch.cuda.max_memory_allocated()

        print(f"[{func.__name__}] 耗时: {end_time - start_time:.4f} 秒")
        print(
            f"[{func.__name__}] 显存: "
            f"起始 {start_mem/1024**2:.2f} MB, "
            f"结束 {end_mem/1024**2:.2f} MB, "
            f"峰值 {peak_mem/1024**2:.2f} MB"
        )

        return result

    return wrapper


if __name__ == "__main__":
    show_memory()
