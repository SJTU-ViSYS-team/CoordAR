import lightning as pl
import torch
import copy


class EMACallback(pl.Callback):
    def __init__(self, decay=0.9999):
        self.decay = decay
        self.ema_state_dict = None

    def on_train_start(self, trainer, pl_module):
        # 初始化 EMA 参数
        self.ema_state_dict = copy.deepcopy(pl_module.state_dict())
        print("ema weights initialized")

    def on_after_backward(self, trainer, pl_module):
        # EMA 更新
        with torch.no_grad():
            for name, param in pl_module.named_parameters():
                if param.requires_grad:
                    ema_param = self.ema_state_dict[name]
                    ema_param.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def on_validation_start(self, trainer, pl_module):
        # 保存当前模型参数
        self.saved_state_dict = copy.deepcopy(pl_module.state_dict())
        # 使用 EMA 参数
        if self.ema_state_dict is not None:
            pl_module.load_state_dict(self.ema_state_dict)

    def on_validation_end(self, trainer, pl_module):
        # 恢复原始模型参数
        pl_module.load_state_dict(self.saved_state_dict)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["ema_state_dict"] = self.ema_state_dict

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        self.ema_state_dict = checkpoint["ema_state_dict"]
