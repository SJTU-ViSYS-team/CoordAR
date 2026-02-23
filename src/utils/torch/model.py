import torch.nn as nn

from src.utils.logging import get_logger

logger = get_logger(__name__)


def log_hierarchical_summary(model, max_depth=2, indent=0, current_depth=0):
    """
    按层级打印模型结构，可控制最大深度
    Args:
        max_depth (int): 最大递归深度（0表示仅顶层）
        indent (int): 缩进空格数（内部用，无需手动设置）
        current_depth (int): 当前深度（内部用，无需手动设置）
    """
    if current_depth > max_depth:
        return  # 终止递归

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "    " * indent
        + f"Model: {type(model).__name__}, Total Params: {total_params:,}"
    )
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        # 打印当前模块信息（带缩进）
        logger.info(
            "    " * (indent + 1)
            + f"└─ {name} [{type(module).__name__}], Params: {num_params:,}"
        )
        # 递归打印子模块（深度+1）
        if isinstance(module, nn.Module) and list(module.named_children()):
            log_hierarchical_summary(
                module,
                max_depth=max_depth,
                indent=indent + 1,
                current_depth=current_depth + 1,
            )

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):    # taken from timm
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):  # taken from timm
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
    def extra_repr(self):
        return f'(drop_prob=...)'


# python -m src.utils.torch.model
if __name__ == "__main__":
    # 示例用法
    class ExampleModel(nn.Module):
        def __init__(self):
            super(ExampleModel, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
            self.fc = nn.Linear(32 * 6 * 6, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    model = ExampleModel()
    log_hierarchical_summary(model, max_depth=2)
