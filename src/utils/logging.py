import logging
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import wandb
import torch
from PIL import Image
import numpy as np


def get_logger(name: str, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 检查是否已存在处理器（避免重复添加）
    if not logger.handlers:
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # 设置日志格式
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)

        # 添加处理器到logger
        logger.addHandler(console_handler)

    return logger


class LevelsFilter(logging.Filter):
    def __init__(self, levels):
        self.levels = [getattr(logging, level) for level in levels]

    def filter(self, record):
        return record.levelno in self.levels


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        import tqdm

        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def log_image(logger, name, path=None):
    if isinstance(logger, WandbLogger):
        assert isinstance(path, str), "image must be a path to an image"
        logger.experiment.log(
            {f"{name}": wandb.Image(path)},
        )
    elif isinstance(logger, TensorBoardLogger):
        image = Image.open(path)
        image = torch.tensor(np.array(image) / 255.0)
        image = image.permute(2, 0, 1).float()
        assert isinstance(image, torch.Tensor), "image must be a tensor"
        logger.experiment.add_image(
            f"{name}",
            image,
        )


def log_video(logger, name, path=None):
    if isinstance(logger, WandbLogger):
        assert isinstance(path, str), "image must be a path to an image"
        logger.experiment.log(
            {f"{name}": wandb.Video(path)},
        )
    elif isinstance(logger, TensorBoardLogger):
        raise NotImplementedError("Tensorboard does not support video logging")
