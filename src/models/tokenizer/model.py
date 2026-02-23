import logging
import os
import time
from typing import Any, Dict
from diffusers.models.autoencoders.vq_model import VQModel
from einops import rearrange
import ipdb
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import lpips

from src.data.collate_importer import preprocess_batch
from src.models.tokenizer.visualization import show_pred
from src.utils.ckpt_loader import CkptLoader
from src.utils.logging import get_logger


class FocalL1Loss(nn.Module):
    def __init__(self, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        weight = (1 - torch.exp(-diff)) ** self.gamma
        loss = weight * diff

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class Tokenizer(nn.Module):

    def __init__(
        self,
        vq_args: Dict[str, Any],
        ckpt_loader: CkptLoader,
        loss_weights={},
        key="nocs",
        vqgan_loss=None,
        quantization="vqvae",
        lfq_args: Dict[str, Any] = None,
        focal_loss_gamma: float = 0.0,
        use_lpips_loss=False,
    ):
        super().__init__()
        self.vq_vae = VQModel(
            **vq_args,
        )
        self.vq_vae.quantize.sane_index_shape = True
        self.num_vq_embeddings = vq_args["num_vq_embeddings"]
        self.vq_embed_dim = vq_args["latent_channels"]
        self.key = key
        self.vqgan_loss = vqgan_loss
        self.quantization = quantization

        self.loss_weights = loss_weights
        if focal_loss_gamma > 0.0:
            self.l1_loss_fn = FocalL1Loss(gamma=focal_loss_gamma, reduction="mean")
        else:
            self.l1_loss_fn = nn.L1Loss(reduction="mean")
        ckpt_loader.load_from_ckpt(self)

    def maybe_freeze_components(self):
        pass

    def get_vocab_size(self):
        return self.num_vq_embeddings

    def get_emb_dim(self):
        return self.vq_embed_dim

    def get_step_size(self):
        raise NotImplementedError("Step size varies based on input image size.")

    def tokenize(self, data):
        h, z_q, min_encoding_indices, _ = self.encode(data - 0.5)
        tokens = rearrange(min_encoding_indices, "b h w -> b (h w)")
        return tokens

    def detokenize(self, tokens):
        b = tokens.shape[0]
        l = tokens.shape[1]
        h, w = int(l**0.5), int(l**0.5)
        assert h * w == l, f"Token length {l} is not a perfect square."
        indices = rearrange(tokens, "b (h w) -> b h w", h=h, w=w)
        recon = self.decode_indices(indices)
        return recon

    def encode(self, xyz):
        h = self.vq_vae.encode(xyz).latents
        z_q, loss, (_, _, min_encoding_indices) = self.vq_vae.quantize(h)
        return h, z_q, min_encoding_indices, loss

    def decode_indices(self, indices):
        shape = (*indices.shape, self.vq_embed_dim)
        quant = self.vq_vae.quantize.get_codebook_entry(indices, shape)

        quant2 = self.vq_vae.post_quant_conv(quant)
        dec = self.vq_vae.decoder(quant2, None) + 0.5
        return dec

    def discretize(self, nocs):
        h, z_q, min_encoding_indices, _ = self.encode(nocs - 0.5)
        return min_encoding_indices

    def emb(self, indices):
        z = self.vq_vae.quantize.get_codebook_entry(indices, shape=None)
        return z

    def decode(self, z):
        quant2 = self.vq_vae.post_quant_conv(z)
        dec = self.vq_vae.decoder(quant2, None)
        return dec

    def forward(self, batch):
        img = batch[self.key]
        h, z_q, min_encoding_indices, vq_loss = self.encode(img - 0.5)

        dec = self.vq_vae.decode(h)
        recon = dec.sample + 0.5

        usage_counts = torch.zeros(self.num_vq_embeddings).to(img)
        indices = torch.cat([indices.view(-1) for indices in min_encoding_indices])
        unique_indices, counts = torch.unique(indices, return_counts=True)
        usage_counts[unique_indices] += counts
        usage_rate = (usage_counts > 0).float().mean()

        return dict(
            min_encoding_indices=min_encoding_indices,
            recon=recon,
            vq_loss=vq_loss,
            usage_rate=usage_rate,
        )

    def get_last_layer(self):
        return self.vq_vae.decoder.conv_out.weight

    def calc_losses(
        self,
        batch,
        batch_idx,
        log_dir,
        optimizer_idx=0,
        global_step=0,
        pred=None,
    ):
        if pred is None:
            pred = self.forward(batch)

        loss_dict = {}
        log = {}
        if self.vqgan_loss is not None:
            loss, log = self.vqgan_loss(
                pred["vq_loss"],
                batch[self.key],
                pred["recon"],
                optimizer_idx,
                global_step,
                last_layer=self.get_last_layer(),
            )
            loss_dict.update({"vqgan_loss": loss})

        if self.vqgan_loss is None:
            recon = pred["recon"]
            target = batch[self.key]
            loss_dict["recon_loss"] = self.l1_loss_fn(target, recon)
            loss_dict["vq_loss"] = pred["vq_loss"]

        if self.use_lpips_loss:
            lpips_loss_fn = self.get_lpips_fn()
            normalize = lambda x: (x - 0.5) * 2.0
            loss_dict["lpips_loss"] = lpips_loss_fn(
                normalize(pred["recon"]), normalize(batch[self.key])
            ).mean()

        loss = 0
        for name in loss_dict.keys():
            loss += loss_dict[name] * self.loss_weights.get(name, 1.0)

        loss_dict["loss"] = loss
        return loss_dict, log

    def get_lpips_fn(self):
        if hasattr(self, "_lpips_loss_fn"):
            return self._lpips_loss_fn
        lpips_loss_fn = lpips.LPIPS(net="alex").to(next(self.parameters()).device)
        self._lpips_loss_fn = lpips_loss_fn
        return lpips_loss_fn

    @torch.no_grad()
    def calc_metrics(
        self,
        batch,
        batch_idx,
        verbose=False,
        log_dir="./debug",
    ):
        pred = self.forward(batch)

        predictions = {}
        losses, log = self.calc_losses(
            batch,
            batch_idx,
            log_dir,
            pred=pred,
        )
        metrics = dict(
            loss=losses["loss"].cpu().numpy().tolist(),
            usage_rate=pred["usage_rate"].cpu().numpy(),
        )
        for k, v in losses.items():
            metrics.update({k: losses[k]})
        for k, v in log.items():
            metrics.update({k: v.cpu().numpy()})

        if verbose:
            self.visualize(
                batch,
                pred,
                log_dir,
                vis_gt=True,
            )

        return predictions, metrics

    def predict(
        self,
        batch,
        verbose=False,
        log_dir="./debug",
    ):
        pred = self.forward(batch)
        return pred

    def visualize(self, batch, pred, log_dir, vis_gt=True):
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
        show_pred(batch, pred, log_dir, key=self.key)

    def get_teacher_forcing(self, tokens):
        tokens = torch.cat(tokens, dim=-1)[:, :-1]
        return self.emb(tokens)

    def get_next_autoregressive_input(self, prev_tokens: torch.Tensor) -> torch.Tensor:
        """
        获取下一个自回归输入
        Args:
            prev_tokens: (B, t) long tensor, 前t个token
        Returns:
            input: (B, vocab_size) one-hot编码
        """
        assert (
            prev_tokens.max().item() < self.num_vq_embeddings
        ), f"Token id {prev_tokens.max().item()} exceeds vocab size {self.num_vq_embeddings}."
        input = self.emb(prev_tokens[:, -1])
        return input


# python -m src.models.tokenizer.model
if __name__ == "__main__":
    from tqdm import tqdm
    from hydra.experimental import compose, initialize
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    import rootutils
    from torchvision.utils import save_image
    import matplotlib

    matplotlib.use("Agg")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = get_logger(__name__, logging.DEBUG)

    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    with initialize(config_path="../../../configs/"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["experiment=tokenizer/roc_l8"],
        )
    OmegaConf.set_struct(cfg, False)

    tokenizer = instantiate(cfg.model.tokenizer).cuda()
    datamodule = instantiate(cfg.data)
    dataloader = datamodule.train_dataloader()
    tokenizer.eval()

    for i, batch in tqdm(enumerate(dataloader)):
        batch = preprocess_batch(batch)
        losses, _ = tokenizer.calc_losses(batch, i, "logs/debug")
        print(losses["loss"])

        # ipdb.set_trace()

        _, metrics = tokenizer.calc_metrics(
            batch, i, verbose=True, log_dir="logs/debug"
        )
        print(metrics)

        # tokenizer.predict(batch, verbose=True, log_dir="logs/debug")
