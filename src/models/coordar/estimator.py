import copy
import logging
import os
import re
from time import sleep
import time
from einops import rearrange
import open3d
from tqdm import tqdm
import torch.utils.data as data
import torch
import torch.nn as nn
import random
import cv2
import numpy as np
import skimage
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import ipdb
from torchvision import transforms

from src.data.collate_importer import preprocess_batch
from src.data.megapose.shapenet import rot_to_nocs_map_batch
from src.models.coordar.utils.pose_selection import pose_selection, uv_grid
from src.models.coordar.utils.metrics import calc_add_recall, calc_mssd_recall
from src.models.coordar.utils.procrustes import WeightedProcrustes
from src.models.coordar.utils.roc import depth_to_point_map, fit_pose_least_squere
from src.models.coordar.visualization import (
    save_data,
    show_ar,
    show_batch,
    show_digit,
    show_point_pairs,
    show_pose,
    show_pred,
)
from src.utils.ckpt_loader import CkptLoader
from src.utils.logging import get_logger
from src.utils.misc import prepare_dir
from src.utils.pysixd.RT_transform import (
    allocentric_to_egocentric,
    allocentric_to_egocentric_batch_torch,
    egocentric_to_allocentric,
)
from src.utils.ransac_utils import tensor2pcd
from src.utils.status import Status
from src.utils.system import Timer
from src.utils.torch.model import log_hierarchical_summary
from src.utils.transform_tensor import mat_to_rot6d_batch, rot6d_to_mat_batch


def dice_coefficient(y_true, y_pred):
    intersection = torch.sum(y_true * y_pred, dim=[1, 2])
    union = torch.sum(y_true, dim=[1, 2]) + torch.sum(y_pred, dim=[1, 2])
    return (2.0 * intersection) / (union + 1e-7)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


def resize_as(x, y):
    return F.interpolate(
        x.float(),
        size=y.shape[2:],
        mode="nearest",
    )


MASK_LOSS_FUNCS = {
    "l1": nn.L1Loss(reduction="mean"),
    "l2": nn.MSELoss(reduction="mean"),
    "dice": dice_loss,
}

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


def cam_to_obj(pts, tco):
    # pts: (b, n, 3)
    # tco: (b, 4, 4)
    pts = pts - tco[:, None, :3, 3]
    pts = torch.einsum("bij,bnj->bni", tco[:, :3, :3].transpose(1, 2), pts)
    return pts


def obj_to_cam(pts, tco):
    # pts: (b, n, 3)
    # tco: (b, 4, 4)
    pts = torch.einsum("bij,bnj->bni", tco[:, :3, :3], pts)
    pts = pts + tco[:, None, :3, 3]
    return pts


class NovelEstimator(torch.nn.Module):

    def __init__(
        self,
        memo: torch.nn.Module,
        ckpt_loader: CkptLoader,
        loss_weights={},
        pose_method="procrustes",
        use_estimated_depth=False,
        use_gt_mask=True,
        pose_selecton="IoU",
        multiview="one2any",
    ) -> None:
        super(NovelEstimator, self).__init__()

        self.memo = memo
        self.loss_weights = loss_weights
        self.pose_method = pose_method
        self.use_estimated_depth = use_estimated_depth
        self.use_gt_mask = use_gt_mask
        self.pose_selecton = pose_selecton
        self.multiview = multiview

        self.procruste = WeightedProcrustes(return_transform=True, weight_thresh=0.5)

        ckpt_loader.load_from_ckpt(self)

        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name, torch.FloatTensor(value).view(1, 3, 1, 1), persistent=False
            )

    def maybe_freeze_components(self):
        pass

    def split_batch(self, batch):
        """
        Split the batch into multiple batches if there are multiple templates.
        """
        template_imgs = batch["template_imgs"]
        N = template_imgs.shape[1]
        split_batches = []
        for i in range(N):
            new_batch = copy.deepcopy(batch)
            new_batch["template_imgs"] = template_imgs[:, i : i + 1]
            new_batch["template_tco"] = batch["template_tco"][:, i : i + 1]
            new_batch["template_K_crop"] = batch["template_K_crop"][:, i : i + 1]
            new_batch["template_rel_rocs"] = batch["template_rel_rocs"][:, i : i + 1]
            new_batch["template_rocs"] = batch["template_rocs"][:, i : i + 1]
            new_batch["template_masks_visib"] = batch["template_masks_visib"][
                :, i : i + 1
            ]
            new_batch["template_depth_bp"] = batch["template_depth_bp"][:, i : i + 1]
            new_batch["template_obj_size"] = batch["template_obj_size"][:, i : i + 1]
            split_batches.append(new_batch)
        return split_batches

    def forward(self, batch):
        # if multiple templates, use the best one
        if self.multiview == "one2any" and batch["template_imgs"].shape[1] > 1:
            # split into multiple batches
            split_batches = self.split_batch(batch)
            out_dicts = []
            for split_batch in split_batches:
                out_dict = self.forward(split_batch)
                out_dicts.append(out_dict)
            out_dict = pose_selection(
                batch, split_batches, out_dicts, self.pose_selecton
            )
            return out_dict
        timer = Timer()
        timer.start("forward")
        template_imgs = (
            batch["template_imgs"] / 255.0 - self._resnet_mean.unsqueeze(1)
        ) / self._resnet_std.unsqueeze(1)
        template_rocs = batch["template_rocs"]
        template_masks = batch["template_masks_visib"].unsqueeze(2)  # (B, N, 1, H, W)
        query = (batch["query"] / 255.0 - self._resnet_mean) / self._resnet_std
        query_roc_mask = torch.cat(
            [batch["query_roc"], batch["query_mask"].unsqueeze(1).float()], dim=1
        )  # (B, 4, H, W)

        N = template_imgs.shape[1]
        # learning_step = np.random.randint(0, N) if self.training else None
        learning_step = None
        ref_values = torch.cat(
            [template_rocs, template_masks],
            dim=2,
        )
        query_target = None
        if "query_rel_roc" in batch:
            query_target = torch.cat(
                [
                    batch["query_rel_roc"],
                    batch["query_mask"].unsqueeze(1).float(),
                ],
                dim=1,
            )
        contrastive_data = {}
        if "valid_pts" in batch:
            contrastive_data = {
                "src_pts": batch["query_roc_loc"],
                "tgt_pts": batch["tgt_loc"],
                "mask": batch["valid_pts"],
            }
        point_data = {
            "query_pts": (
                batch["query_pts_in_mask"] - batch["query_TCO"][:, None, :3, 3]
            )
            / batch["diameter"][:, None, None]
            * 2,
            "ref_pts": (
                batch["template_pts_in_mask"][:, 0]
                - batch["template_tco"][:, 0, None, :3, 3]
            )
            / batch["template_obj_size"][:, 0, None, None]
            * 2,
        }
        timer.start("model")
        out_dict = self.memo(
            rearrange(template_imgs, "b n c h w -> n b c h w"),
            rearrange(ref_values, "b n c h w -> n b c h w"),
            query,
            query_roc_mask,
            learning_step,
            query_target,
            contrastive_data=contrastive_data,
            point_data=point_data,
        )
        timer.end("model")
        resize = lambda x: F.interpolate(
            x,
            size=(28, 28),
            mode="bilinear",
            align_corners=False,
        )
        if self.use_gt_mask:
            pred_mask_bin = batch["query_mask"].float()
        else:
            pred_mask_bin = (out_dict["mask"].squeeze(1) > 0.5).float()

        def to_real_points(point_map, trans, obj_size):
            points = rearrange(
                point_map - 0.5, "b c h w->b (h w) c"
            ) * obj_size.reshape(-1, 1, 1) + trans.unsqueeze(1)
            return points

        timer.start("procruste")
        if not self.training:
            procrustes_mask = resize_as(
                (
                    resize_as(
                        pred_mask_bin.unsqueeze(1), batch["query_depth"].unsqueeze(1)
                    ).squeeze(1)
                    * (
                        (out_dict["depth"] > 1e-3)
                        if "depth" in out_dict
                        else (batch["query_depth"] > 1e-3)
                    )
                )
                .unsqueeze(1)
                .float(),
                out_dict["geo"],
            ).squeeze(1)
            point_from_depth = self.get_point_from_depth(batch, out_dict)
            out_dict["procrustes_mask"] = procrustes_mask
        if not self.training and self.pose_method == "procrustes":
            rel_pose = self.procruste(
                to_real_points(
                    out_dict["geo"],
                    batch["template_tco"][:, 0, :3, 3],
                    batch["template_obj_size"][:, 0],
                ),
                rearrange(
                    point_from_depth,
                    "b c h w->b (h w) c",
                ),
                rearrange(
                    procrustes_mask,
                    "b h w->b (h w)",
                ),
            )
            # rel_pose = self.procruste(
            #     to_real_points(
            #         batch["query_rel_roc"], batch["template_tco"][:, 0, :3, 3]
            #     ),
            #     to_real_points(batch["query_roc"], batch["query_TCO"][:, :3, 3]),
            #     rearrange(batch["query_mask"], "b h w->b (h w)"),
            # )  # debug with gt
            out_dict["tco_pred"] = rel_pose @ batch["template_tco"][:, 0]
        else:
            out_dict["tco_pred"] = batch["template_tco"][
                :, 0
            ]  # placeholder when training

        timer.end("procruste")
        timer.end("forward")
        out_dict["time_forward"] = torch.tensor(timer.elapsed_seconds("forward")).to(
            batch["query"].device
        )
        out_dict["time_model"] = torch.tensor(timer.elapsed_seconds("model")).to(
            batch["query"].device
        )
        out_dict["time_procruste"] = torch.tensor(
            timer.elapsed_seconds("procruste")
        ).to(batch["query"].device)

        return out_dict

    def get_point_from_depth(self, batch, pred):
        pointmap_obs = resize_as(
            depth_to_point_map(
                batch["query_depth"],
                batch["query_K_crop"],
            ).permute(0, 3, 1, 2),
            pred["geo"],
        )
        if not self.use_estimated_depth:
            return pointmap_obs

        """
        use estimated point map
        """

        pred_mask = resize_as(
            pred["mask"].unsqueeze(1).sigmoid(), batch["query_depth"].unsqueeze(1)
        ).squeeze(1)
        roc_hat = resize_as(pred["roc"], batch["query_depth"].unsqueeze(1))
        fit_mask = pred_mask * (batch["query_depth"] > 0.001)
        t_hat, depth_hat, pointmap = fit_pose_least_squere(
            roc_hat,
            batch["query_depth"],
            fit_mask,
            batch["query_K_crop"],
            batch["diameter"],
        )
        pointmap = rearrange(pointmap, "b h w c -> b c h w")
        pointmap = resize_as(pointmap, pred["geo"])
        # replace nan with obs
        pointmap[torch.isnan(pointmap)] = pointmap_obs[torch.isnan(pointmap)]

        pred["depth"] = depth_hat

        return pointmap

    def calc_losses(
        self,
        batch,
        batch_idx,
        log_dir,
        training_process=0,
        pred=None,
    ):
        if pred is None:
            pred = self.forward(batch)
        loss_dict = {}
        loss_dict.update(self.get_mask_loss(batch, pred))
        loss_dict.update(self.get_roc_loss(batch, pred))
        if "geo_loss" in pred:
            loss_dict.update(dict(geo_loss=pred["geo_loss"]))
        else:
            loss_dict.update(self.get_geo_loss(batch, pred))
        loss_dict.update(self.get_recon_loss(batch, pred))
        loss = 0
        for name in loss_dict.keys():
            loss += loss_dict[name] * self.loss_weights.get(name, 1.0)

        loss_dict["loss"] = loss
        return loss_dict

    def get_mask_loss(self, batch, pred):
        if "mask" not in pred:
            return dict()
        visib_mask_pred = pred["mask"]
        gt_visib_mask = F.interpolate(
            batch["query_mask"].float().unsqueeze(1),
            size=visib_mask_pred.shape[1:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(
            1
        )  # resize to the same size as visib_mask_pred
        mask_loss = F.binary_cross_entropy_with_logits(
            visib_mask_pred, gt_visib_mask.float()
        )

        return dict(
            mask_loss=mask_loss,
        )

    def get_geo_loss(self, batch, pred):
        if "geo" not in pred:
            return dict()
        pred_geo = pred["geo"]
        gt_geo = F.interpolate(
            batch["query_rel_roc"],
            size=pred_geo.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        gt_visib_mask = F.interpolate(
            batch["query_mask"].float().unsqueeze(1),
            size=pred_geo.shape[2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        geo_loss = ((pred_geo - gt_geo) * gt_visib_mask.unsqueeze(1)).abs().sum() / (
            gt_visib_mask.sum() + 1
        )
        return dict(
            geo_loss=geo_loss,
        )

    def get_roc_loss(self, batch, pred):
        if "roc" not in pred:
            return dict()
        pred_roc = pred["roc"]
        gt_roc = F.interpolate(
            batch["query_roc"],
            size=pred_roc.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        gt_visib_mask = F.interpolate(
            batch["query_mask"].float().unsqueeze(1),
            size=pred_roc.shape[2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        roc_loss = ((pred_roc - gt_roc) * gt_visib_mask.unsqueeze(1)).abs().sum() / (
            gt_visib_mask.sum() + 1
        )
        return dict(
            roc_loss=roc_loss,
        )


    def get_recon_loss(self, batch, pred):
        if "recon_loss" not in pred:
            return dict()
        return dict(
            recon_loss=pred["recon_loss"],
        )

    @torch.no_grad()
    def calc_metrics(
        self, batch, batch_idx, verbose=False, log_dir="./debug", vis_all_samples=False
    ):
        pred = self.forward(batch)

        symmetries = batch["symmetries"]
        gt_diameter = batch["gt_diameter"]
        model_pts = batch["points"]
        TCO_gt = batch["query_TCO"]
        TCO_pred = pred["tco_pred"]
        mssd_recall = calc_mssd_recall(
            TCO_pred, TCO_gt, model_pts, symmetries, gt_diameter
        )
        add_recall = calc_add_recall(
            TCO_pred, TCO_gt, model_pts, symmetries, gt_diameter
        )

        losses = self.calc_losses(
            batch,
            batch_idx,
            log_dir,
            pred=pred,
        )
        metrics = dict(
            loss=losses["loss"].mean(),
            mssd_recall=mssd_recall.mean(),
            add_recall=add_recall.mean(),
            time_forward=pred["time_forward"],
            time_model=pred["time_model"],
            time_procruste=pred["time_procruste"],
        )
        for k, v in losses.items():
            metrics.update({k: losses[k]})

        predictions = []
        pred_pose = pred["tco_pred"]
        bs = batch["query"].shape[0]
        for i in range(bs):
            p = dict()
            TCO = pred_pose[i].cpu().numpy()
            TCO[:3, 3] *= 1000
            p.update(
                dict(
                    status=Status.SUCCESS,
                    scene_id=batch["scene_id"][i].item(),
                    im_id=batch["im_id"][i].item(),
                    obj_id=batch["obj_id"][i].item(),
                    gt_id=batch["gt_id"][i].item(),
                    det_id=batch["det_id"][i].item(),
                    score=batch["score"][i].item(),
                    TCO_ref=batch["template_tco"][i].cpu().numpy(),
                    TCO=TCO,
                    TCO_gt=batch["query_TCO"][i].cpu().numpy(),
                    bbox=batch["bbox"][i].cpu().numpy(),
                    add_recall=add_recall[i].item(),
                    mssd_recall=mssd_recall[i].item(),
                )
            )
            predictions.append(p)

        if verbose:
            if vis_all_samples:
                log_dir = f"{log_dir}/{batch_idx:06d}"
                prepare_dir(log_dir)
            self.visualize(batch, pred, log_dir, vis_gt=False, mssd_recall=mssd_recall)

        return predictions, metrics

    @torch.no_grad()
    def predict(
        self,
        batch,
        batch_idx,
        verbose=False,
        log_dir="./debug",
        vis_all_samples=False,
    ):
        pred = self.forward(batch)
        if verbose:
            if vis_all_samples:
                log_dir = f"{log_dir}/{batch_idx:06d}"
                prepare_dir(log_dir)
            self.visualize(batch, pred, log_dir, vis_gt=True)

        bs = batch["query"].shape[0]
        pred_pose = pred["tco_pred"]

        predictions = []
        for i in range(bs):
            p = dict()
            TCO = pred_pose[i].cpu().numpy()
            TCO[:3, 3] *= 1000
            p.update(
                dict(
                    status=Status.SUCCESS,
                    scene_id=batch["scene_id"][i].item(),
                    im_id=batch["im_id"][i].item(),
                    obj_id=batch["obj_id"][i].item(),
                    gt_id=batch["gt_id"][i].item(),
                    det_id=batch["det_id"][i].item(),
                    score=batch["score"][i].item(),
                    TCO=TCO,
                    bbox=batch["bbox"][i].cpu().numpy(),
                )
            )
            predictions.append(p)
        return predictions

    def visualize(self, batch, pred, log_dir, vis_gt=True, mssd_recall=None):
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
        show_batch(batch, log_dir)
        show_pred(batch, pred, log_dir)
        show_digit(batch, pred, log_dir)
        show_pose(batch, pred, log_dir, mssd_recall)
        show_ar(batch, pred, log_dir)
        show_point_pairs(batch, pred, log_dir)
        # save_data(batch, pred, log_dir)


# python -m src.models.coordar.estimator
if __name__ == "__main__":
    from tqdm import tqdm
    from hydra.experimental import compose, initialize
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    import rootutils
    from torchvision.utils import save_image
    import matplotlib

    # matplotlib.use("Agg")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = get_logger(__name__, logging.DEBUG)

    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    with initialize(config_path="../../../configs/"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["experiment=coordar/ar_paper_dinovae"],
        )
    OmegaConf.set_struct(cfg, False)
    estimator = instantiate(cfg.model.estimator).cuda()

    log_hierarchical_summary(estimator, max_depth=2)

    datamodule = instantiate(cfg.data)
    dataloader = datamodule.train_dataloader()
    print(len(dataloader))
    estimator.eval()

    for i, batch in tqdm(enumerate(dataloader)):
        batch = preprocess_batch(batch)
        # losses = estimator.calc_losses(batch, i, "logs/debug")
        # print(losses["loss"])

        # ipdb.set_trace()

        _, metrics = estimator.calc_metrics(
            batch, i, verbose=True, log_dir="logs/debug"
        )
        print(metrics)

        # estimator.predict(batch, i, verbose=True, log_dir="logs/debug")
