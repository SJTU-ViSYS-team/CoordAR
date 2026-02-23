from collections import defaultdict
import os
import re
from typing import Any, Dict, List, Tuple

import ipdb
from matplotlib import pyplot as plt
import numpy as np
from pytest import param
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric
from torchmetrics.classification.accuracy import Accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from ranger import Ranger

from src.data.collate_importer import FunctionImporter
from src.utils.distributed import collect_results
from src.utils.folder import prepare_dir
from src.utils.misc import LD2DL
from src.utils.tensor_collection import TensorCollection
from src.utils.torch.lr_scheduler import (
    flat_and_anneal_lr_scheduler_epoch,
    step_scheduler_with_warmup,
)


class TokenizerModule(LightningModule):

    def __init__(
        self,
        tokenizer: torch.nn.Module,
        optimizer_init: Dict[str, Any] = {},
        lr_monitor="val_loss",
        scheduler="cosine",
        lr_schedule_cfg: Dict[str, Any] = {},
        train_vis_step=1000,
        val_vis_step=1000,
        predict_vis_step=1000,
        test_vis_step=1000,
        optimizer="adam",
        preprocess_batch: FunctionImporter = None,
        accumulate_grad_batches=1,
        debug=False,
        vis_dir: str = "logs/vis",
        tags=["cir"],
        compile: bool = False,
        gradient_clip_val=10,
    ):
        super(TokenizerModule, self).__init__()

        self.save_hyperparameters(
            ignore=[
                "tokenizer",
                "lr_schedule_cfg",
                "optimizer",
                "scheduler",
                "preprocess_batch",
                "debug",
            ],
            logger=False,
        )

        self.tokenizer = tokenizer
        self.lr_monitor = lr_monitor
        self.lr_schedule_cfg = lr_schedule_cfg
        self.optimizer_init = optimizer_init
        self.train_vis_step = train_vis_step
        self.val_vis_step = val_vis_step
        self.test_vis_step = test_vis_step
        self.predict_vis_step = predict_vis_step
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.preprocess_batch = preprocess_batch
        self.accumulate_grad_batches = accumulate_grad_batches
        self.debug = debug
        self.vis_dir = vis_dir
        self.tags = tags
        self.gradient_clip_val = gradient_clip_val

        self.no_decay = ["bias", "LayerNorm.weight"]
        self.tokenizer.maybe_freeze_components()

        self.automatic_optimization = False
        self.val_loss = MeanMetric()
        self.pred_outs = []
        # self.strict_loading = False

    def setup(self, stage):
        if self.hparams.compile and stage == "fit":
            self.tokenizer = torch.compile(self.tokenizer)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

    def forward(self, batch):
        return self.tokenizer(batch)

    def log_metrics(self, metric_dict, stage, on_step=True, on_epoch=False):
        for k, v in metric_dict.items():
            self.log(
                f"{stage}/{k}",
                v.mean(),
                on_step=on_step,
                on_epoch=on_epoch,
                sync_dist=True,
                prog_bar=True,
            )

    # @print_time
    def training_step(self, batch, batch_idx):
        vis_dir = f"{self.vis_dir}/train_vis"
        prepare_dir(vis_dir)
        if self.preprocess_batch is not None:
            batch = self.preprocess_batch(batch, stage="train")

        opts = self.optimizers()
        schedulers = self.lr_schedulers()

        if not isinstance(opts, list):
            opts = [opts]
            schedulers = [schedulers]

        for optimizer_idx, opt in enumerate(opts):
            if (
                hasattr(self.trainer, "limit_train_batches")
                and self.trainer.limit_train_batches > 1.0
            ):
                num_iters = self.trainer.limit_train_batches
            else:
                num_iters = len(self.trainer.train_dataloader)
            global_step = self.current_epoch * num_iters + batch_idx

            loss_dict, log = self.tokenizer.calc_losses(
                batch,
                batch_idx,
                log_dir=vis_dir,
                optimizer_idx=optimizer_idx,
                global_step=global_step,
            )
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            self.manual_backward(loss)

            if (
                batch_idx % self.accumulate_grad_batches == 0
                or batch_idx == num_iters - 1
            ):
                self.clip_gradients(
                    opt,
                    gradient_clip_val=self.gradient_clip_val,
                    gradient_clip_algorithm="norm",
                )
                opt.step()
                train_propotion = (
                    self.trainer.limit_train_batches
                    if (self.trainer.limit_train_batches is not None)
                    and (self.trainer.limit_train_batches < 1)
                    else 1.0
                )
                num_batches = int(num_iters * train_propotion)
                schedulers[optimizer_idx].step()
            self.log_metrics(loss_dict, "train")
            self.log_metrics(log, "train")

        if self.train_vis_step > 0 and batch_idx % self.train_vis_step == 0:
            self.tokenizer.calc_metrics(batch, batch_idx, verbose=True, log_dir=vis_dir)

    def validation_step(self, batch, batch_idx):
        if self.preprocess_batch is not None:
            batch = self.preprocess_batch(batch, stage="val")

        vis_dir = f"{self.vis_dir}/val_vis"
        prepare_dir(vis_dir)
        verbose = False
        if self.val_vis_step > 0 and batch_idx % self.val_vis_step == 0:
            verbose = True
        _, metrics = self.tokenizer.calc_metrics(
            batch,
            batch_idx,
            verbose=verbose,
            log_dir=vis_dir,
        )
        self.val_loss(metrics["loss"])
        self.log_metrics(metrics, "val", on_epoch=True, on_step=False)
        return metrics

    def test_step(self, batch, batch_idx):
        debug = self.debug
        if self.preprocess_batch is not None:
            batch = self.preprocess_batch(batch, stage="test")
        vis_dir = f"{self.vis_dir}/test_vis"
        verbose = False
        if self.test_vis_step > 0 and batch_idx % self.test_vis_step == 0:
            verbose = True
        if debug:
            vis_dir = f"{vis_dir}/{self.tokenizer.get_tracing_dir(batch)}"
            verbose = True
        prepare_dir(vis_dir)
        predictions, metrics = self.tokenizer.calc_metrics(
            batch,
            batch_idx,
            verbose=verbose,
            log_dir=vis_dir,
        )
        self.log_metrics(metrics, "test", on_epoch=True, on_step=False)
        return predictions, metrics

    def on_test_batch_end(self, outputs, batch, batch_idx):
        self.pred_outs.append(outputs[0])

    def on_validation_epoch_end(self):
        torch.cuda.empty_cache()

    def on_test_epoch_end(self):
        # If compute only gets called on process 0 then it will wait indefinitely on the other processes trying to reach the barrier (which they never will).
        predictions = self.pred_outs
        # break batches
        flat_predictions = []
        for i in range(len(predictions)):
            batch = predictions[i]
            for j in range(len(batch)):
                flat_predictions.append(batch[j])
        flat_predictions = collect_results(flat_predictions)
        # ddp workers will return empty list
        if len(flat_predictions) == 0:
            return

        pred_dir = os.path.join(f"{self.vis_dir}/test", "predictions")
        prepare_dir(pred_dir)
        torch.cuda.empty_cache()
        self.pred_outs = []

    def collect_metrics(self, step_outputs):
        metrics = LD2DL(step_outputs)
        metric_sum = defaultdict()
        for k in metrics.keys():
            vals = torch.Tensor(metrics[k])
            mean_value = torch.mean(vals, 0).item()
            metric_sum[k] = mean_value
        return metric_sum

    def configure_optimizers(self):
        # params = self.tokenizer.parameters()
        opt_params = []
        tokenizer_params = self.group_params(self.tokenizer, self.optimizer_init)
        opt_params.append(tokenizer_params)
        if (
            hasattr(self.tokenizer, "vqgan_loss")
            and self.tokenizer.vqgan_loss is not None
        ):
            vqgan_params = self.group_params(
                self.tokenizer.vqgan_loss, self.optimizer_init
            )
            opt_params.append(vqgan_params)
        if hasattr(self.tokenizer, "gan_loss") and self.tokenizer.gan_loss is not None:
            gan_params = self.group_params(self.tokenizer.gan_loss, self.optimizer_init)
            opt_params.append(gan_params)
        optimizers = []
        scheduler_configs = []

        for params in opt_params:
            if self.optimizer == "adam":
                optimizer = torch.optim.Adam(params)
            elif self.optimizer == "adamw":
                optimizer = torch.optim.AdamW(params)
            elif self.optimizer == "sgd":
                optimizer = torch.optim.SGD(params)
            elif self.optimizer == "ranger":
                optimizer = Ranger(params)
            else:
                raise NotImplementedError(f"Not implemented {self.optimizer}")
            # optimizer = Ranger(params, **self.optimizer_init)
            if self.scheduler == "step_with_warmup":
                lr_scheduler = step_scheduler_with_warmup(
                    optimizer, **self.lr_schedule_cfg
                )
            elif self.scheduler == "step":
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, **self.lr_schedule_cfg
                )
            elif self.scheduler == "one_cycle":
                lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    **self.lr_schedule_cfg,
                )
            else:
                lr_scheduler = ReduceLROnPlateau(optimizer, **self.lr_schedule_cfg)
            lr_scheduler_config = {
                "scheduler": lr_scheduler,
                "interval": "step",  # or 'epoch'
                "frequency": 1,
                "monitor": self.lr_monitor,
            }
            optimizers.append(optimizer)
            scheduler_configs.append(lr_scheduler_config)
        return optimizers, scheduler_configs

    def group_params(self, module, optimizer_init):
        """
        lr_dict: 定义不同层的学习率规则，例如：
            {"encoder.*": 1e-5, "decoder.*": 1e-4, "classifier.*": 1e-3}
        """
        lr_dict = optimizer_init["lr_dict"]
        groups = {}
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            lr = next(
                (v for k, v in lr_dict.items() if re.match(k, name)), 1e-4
            )  # 默认值
            weight_decay = optimizer_init["weight_decay"]
            if any(nd in name for nd in self.no_decay):
                weight_decay = 0.0

            groups.setdefault((lr, weight_decay), []).append((name, param))

        # 转换为参数组
        param_groups = []
        for key, param_list in groups.items():
            lr, weight_decay = key
            names = [n for n, _ in param_list]
            params = [p for _, p in param_list]
            print(f"Group params: {names} with lr={lr}, weight_decay={weight_decay}")
            param_groups.append(
                {
                    "params": params,
                    "lr": lr,
                    "weight_decay": weight_decay,
                }
            )
        return param_groups

    def on_train_epoch_end(self):
        # training_step_outputs is on gpu
        torch.cuda.empty_cache()

    def predict_step(self, batch, batch_idx):
        debug = self.debug
        if self.preprocess_batch is not None:
            batch = self.preprocess_batch(batch, batch_idx, stage="predict")

        vis_dir = f"{self.vis_dir}/predict_vis"
        verbose = False
        if debug:
            vis_dir = f"{vis_dir}/{self.tokenizer.get_tracing_dir(batch)}"
            verbose = True
        prepare_dir(vis_dir)
        if self.predict_vis_step > 0 and batch_idx % self.predict_vis_step == 0:
            verbose = True
        predictions = self.tokenizer.predict(batch, batch_idx, verbose, vis_dir)
        return predictions

    def on_predict_batch_end(self, outputs, batch, batch_idx):
        self.pred_outs.append(outputs)

    def on_predict_epoch_end(self):
        pass


if __name__ == "__main__":
    pass
