from collections import defaultdict
import copy
import os
import re
from typing import Any, Dict, List
import torch
import torch.nn.functional as F
from lightning import LightningModule
import numpy as np
from torchmetrics import CatMetric, MeanMetric
import torchvision
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from ranger import Ranger
import matplotlib.pyplot as plt
import ipdb


from bop_toolkit_lib import inout
from src.utils.distributed import collect_results
from src.utils.logging import get_logger
from src.utils.misc import LD2DL
from src.utils.folder import prepare_dir
from src.data.collate_importer import FunctionImporter
from src.utils.bop_eval import save_and_eval_results, save_test_predictions
from src.utils.system import increment_path
from src.utils.torch.lr_scheduler import (
    flat_and_anneal_lr_scheduler_epoch,
    step_scheduler_with_warmup,
)
from src.third_party.bop_toolkit.bop_toolkit_lib import dataset_params
from src.utils.torch.model import log_hierarchical_summary

logger = get_logger(__name__)


class CoordsMapModule(LightningModule):

    def __init__(
        self,
        estimator: torch.nn.Module,
        optimizer_init: Dict[str, Any] = {},
        lr_monitor="val_loss",
        scheduler="cosine",
        lr_schedule_cfg: Dict[str, Any] = {},
        train_vis_step=1000,
        val_vis_step=1000,
        predict_vis_step=1000,
        test_vis_step=1000,
        vis_all_samples=False,
        optimizer="adam",
        preprocess_batch: FunctionImporter = None,
        accumulate_grad_batches=1,
        debug=False,
        vis_dir: str = "logs/vis",
        tags=["cir"],
        compile: bool = False,
        gradient_clip_val=10,
    ):
        super(CoordsMapModule, self).__init__()

        self.save_hyperparameters(
            ignore=[
                "estimator",
                "lr_schedule_cfg",
                "optimizer",
                "scheduler",
                "preprocess_batch",
                "vis_all_samples",
                "debug",
            ],
            logger=False,
        )

        self.estimator = estimator
        self.lr_monitor = lr_monitor
        self.lr_schedule_cfg = lr_schedule_cfg
        self.optimizer_init = optimizer_init
        self.train_vis_step = train_vis_step
        self.val_vis_step = val_vis_step
        self.test_vis_step = test_vis_step
        self.predict_vis_step = predict_vis_step
        self.vis_all_samples = vis_all_samples
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.preprocess_batch = preprocess_batch
        self.accumulate_grad_batches = accumulate_grad_batches
        self.debug = debug
        self.vis_dir = vis_dir
        self.tags = tags
        self.gradient_clip_val = gradient_clip_val

        self.no_decay = ["bias", "LayerNorm.weight"]
        self.estimator.maybe_freeze_components()
        if optimizer == "ranger":
            self.automatic_optimization = False
        else:
            assert (
                accumulate_grad_batches == 1
            ), "Only support accumulate_grad_batches>1 for ranger"

        self.val_loss = MeanMetric()
        self.pred_outs = []

    def setup(self, stage):
        if self.hparams.compile and stage == "fit":
            self.estimator = torch.compile(self.estimator)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

    def on_test_start(self):
        log_hierarchical_summary(self.estimator, max_depth=2)

    def forward(self, batch):
        return self.estimator(batch)

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

        if self.preprocess_batch is not None:
            batch = self.preprocess_batch(batch, stage="train")

        if self.optimizer == "ranger":
            opt = self.optimizers()
            scheduler = self.lr_schedulers()
            opt.zero_grad()

        if (
            hasattr(self.trainer, "limit_train_batches")
            and self.trainer.limit_train_batches > 1.0
        ):
            num_iters = self.trainer.limit_train_batches
        else:
            num_iters = len(self.trainer.train_dataloader)

        training_process = (
            self.current_epoch + batch_idx / num_iters
        ) / self.trainer.max_epochs

        loss_dict = self.estimator.calc_losses(
            batch,
            batch_idx,
            log_dir=f"{self.vis_dir}/train_samples",
            training_process=training_process,
        )
        vis_dir = f"{self.vis_dir}/train_vis"
        prepare_dir(vis_dir)
        loss = loss_dict["loss"].mean()
        self.log_metrics(loss_dict, "train")
        if self.train_vis_step > 0 and batch_idx % self.train_vis_step == 0:
            self.estimator.calc_metrics(
                batch, batch_idx, verbose=True, log_dir=vis_dir, vis_all_samples=False
            )

        if self.optimizer == "ranger":
            self.manual_backward(loss)

            if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                self.clip_gradients(
                    opt,
                    gradient_clip_val=self.gradient_clip_val,
                    gradient_clip_algorithm="norm",
                )
                opt.step()
                opt.zero_grad()
                train_propotion = (
                    self.trainer.limit_train_batches
                    if (self.trainer.limit_train_batches is not None)
                    and (self.trainer.limit_train_batches < 1)
                    else 1.0
                )
                num_batches = int(num_iters * train_propotion)
                scheduler.step()
        else:
            return loss

    def validation_step(self, batch, batch_idx):
        if self.preprocess_batch is not None:
            batch = self.preprocess_batch(batch, stage="val")

        vis_dir = f"{self.vis_dir}/val_vis"
        prepare_dir(vis_dir)
        verbose = False
        if self.val_vis_step > 0 and batch_idx % self.val_vis_step == 0:
            verbose = True
        _, metrics = self.estimator.calc_metrics(
            batch,
            batch_idx,
            verbose=verbose,
            log_dir=vis_dir,
            vis_all_samples=self.vis_all_samples,
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
            vis_dir = f"{vis_dir}/{self.estimator.get_tracing_dir(batch)}"
            verbose = True
        prepare_dir(vis_dir)
        predictions, metrics = self.estimator.calc_metrics(
            batch,
            batch_idx,
            verbose=verbose,
            log_dir=vis_dir,
            vis_all_samples=self.vis_all_samples,
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
        save_test_predictions(flat_predictions, pred_dir)
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
        # params = self.estimator.parameters()
        params = self._init_param_groups()
        # params = list(filter(lambda x: x.requires_grad, self.estimator.parameters()))
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(params, **self.optimizer_init)
        elif self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(params, **self.optimizer_init)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(params, **self.optimizer_init)
        elif self.optimizer == "ranger":
            optimizer = Ranger(params, **self.optimizer_init)
        else:
            raise NotImplementedError(f"Not implemented {self.optimizer}")
        # optimizer = Ranger(params, **self.optimizer_init)
        scheduler_config = copy.deepcopy(self.lr_schedule_cfg)
        interval = scheduler_config.pop("interval", "step")
        if "steps_per_epoch" in scheduler_config:
            scheduler_config["steps_per_epoch"] //= self.accumulate_grad_batches
        if self.optimizer == "ranger":
            if self.scheduler == "flat_and_anneal":
                lr_scheduler = flat_and_anneal_lr_scheduler_epoch(
                    optimizer, self.trainer.max_epochs, **scheduler_config
                )
            elif self.scheduler == "cosine":
                lr_scheduler = CosineAnnealingWarmRestarts(
                    optimizer, **scheduler_config
                )
            elif self.scheduler == "step":
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, **scheduler_config
                )
            elif self.scheduler == "one_cycle":
                lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    **scheduler_config,
                )
            elif self.scheduler == "reduce_on_plateau":
                lr_scheduler = ReduceLROnPlateau(optimizer, **scheduler_config)
            else:
                raise NotImplementedError(
                    f"Not implemented {self.scheduler} for ranger"
                )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": interval,
                    "frequency": 1,
                    "monitor": self.lr_monitor,
                },
            }
        elif self.scheduler == "step_with_warmup":
            lr_scheduler = step_scheduler_with_warmup(optimizer, **scheduler_config)
        elif self.scheduler == "step":
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, **scheduler_config
            )
            lr_scheduler_config = {
                "scheduler": lr_scheduler,
                "interval": interval,
                "frequency": 1,
                "monitor": self.lr_monitor,
            }
        elif self.scheduler == "one_cycle":
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                **scheduler_config,
            )
        elif self.scheduler == "reduce_on_plateau":
            lr_scheduler = ReduceLROnPlateau(optimizer, **scheduler_config)
        else:
            raise NotImplementedError(f"Not implemented {self.scheduler}")
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": interval,
            "frequency": 1,
            "monitor": self.lr_monitor,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def _init_param_groups(self) -> List[Dict]:
        """Initialize the parameter groups. Used to ensure weight_decay is not applied to our specified bias
        parameters when we initialize the optimizer.

        Returns:
            List[Dict]: A list of parameter group dictionaries.
        """
        return [
            {
                "params": [
                    p
                    for n, p in self.estimator.named_parameters()
                    if not any(nd in n for nd in self.no_decay) and p.requires_grad
                ],
                "weight_decay": self.hparams.optimizer_init["weight_decay"],
            },
            {
                "params": [
                    p
                    for n, p in self.estimator.named_parameters()
                    if any(nd in n for nd in self.no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

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
            vis_dir = f"{vis_dir}/{self.estimator.get_tracing_dir(batch)}"
            verbose = True
        prepare_dir(vis_dir)
        if self.predict_vis_step > 0 and batch_idx % self.predict_vis_step == 0:
            verbose = True
        predictions = self.estimator.predict(
            batch, batch_idx, verbose, vis_dir, vis_all_samples=self.vis_all_samples
        )
        return predictions

    def on_predict_start(self):
        log_hierarchical_summary(self.estimator, max_depth=2)

    def on_predict_batch_end(self, outputs, batch, batch_idx):
        self.pred_outs.append(outputs)

    def on_predict_epoch_end(self):
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

        if not hasattr(
            self.trainer.predict_dataloaders.dataset.instance_ds, "scene_ds"
        ):
            return

        dataset = self.trainer.predict_dataloaders.dataset.instance_ds.scene_ds
        dp_split = dataset_params.get_split_params(
            dataset.datasets_path,
            dataset.dataset,
            dataset.split,
            dataset.split_type,
        )
        # pred_path = dp_split["pred_tpath"].format(tag=f"{self.logger.name}_{self.logger.version}")
        split_type_str = "" if dataset.split_type is None else f"-{dataset.split_type}"
        eval_dir = f"{self.vis_dir}/../eval"
        method = "_".join(self.tags)
        pred_name = f"{method}-{dp_split['name']}-{dataset.split}{split_type_str}.csv"

        if dataset.dataset in ["tyol", "real275"]:
            targets_filename = "test_targets_rand2000.json"
            test_targets = self.trainer.predict_dataloaders.dataset.get_test_targets()
            inout.save_json(f"{dp_split['base_path']}/{targets_filename}", test_targets)
            # save copy to logs
            prepare_dir(eval_dir)
            inout.save_json(f"{eval_dir}/{targets_filename}", test_targets)
        elif dataset.only_bop19_test == False:
            targets_filename = "test_targets_keyframes.json"
            test_targets = self.trainer.predict_dataloaders.dataset.get_test_targets()
            inout.save_json(f"{dp_split['base_path']}/{targets_filename}", test_targets)
        else:
            targets_filename = "test_targets_bop19.json"

        ADD_eval_cfg = {
            "targets_filename": targets_filename,
            "error_types": ["ad"],
        }
        save_and_eval_results(
            dp_split,
            f"{eval_dir}/ADD(-S)",
            pred_name,
            flat_predictions,
            **ADD_eval_cfg,
        )

        AR_eval_cfg = {
            "targets_filename": targets_filename,
            "error_types": ["mssd", "mspd", "vsd"],
        }
        save_and_eval_results(
            dp_split,
            f"{eval_dir}/AR",
            pred_name,
            flat_predictions,
            **AR_eval_cfg,
        )

        ADD_eval_cfg = {
            "targets_filename": targets_filename,
            "error_types": ["adoryon"],
        }
        save_and_eval_results(
            dp_split,
            f"{eval_dir}/ADD_ORYON",
            pred_name,
            flat_predictions,
            **ADD_eval_cfg,
        )

        AUC_ADD_eval_cfg = {
            "targets_filename": targets_filename,
            "error_types": ["AUCadd"],
        }
        save_and_eval_results(
            dp_split,
            f"{eval_dir}/AUC_ADD",
            pred_name,
            flat_predictions,
            **AUC_ADD_eval_cfg,
        )

        AUC_ADI_eval_cfg = {
            "targets_filename": targets_filename,
            "error_types": ["AUCadi"],
        }
        save_and_eval_results(
            dp_split,
            f"{eval_dir}/AUC_ADI",
            pred_name,
            flat_predictions,
            **AUC_ADI_eval_cfg,
        )
        self.pred_outs = []
