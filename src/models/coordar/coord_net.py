from einops import rearrange
import ipdb
from sympy import rf
import torch
import torch.nn as nn
from src.models.layers.transformer import TransformerDecoderLayer
import timm
import torch.nn.functional as F
import diffusers
from torchvision.utils import save_image

from src.models.coordar.backbones.vae import VAE
from src.models.coordar.heads.geo_ar_reg import GeoRegHead
from src.models.coordar.memory.pixel_aligned_mem import SlotMem
from src.models.layers.adaln import AdaLN_EW
from src.utils.system import show_gpu_memory


class MemoCoordMap(nn.Module):

    def __init__(
        self,
        key_encoder,
        value_encoder,
        reg_head,
        geo_head=None,
        image_size=(224, 224),
        patch_size=16,
        decode_steps=4,
        decode_heads=4,
        embed_dim=768,
        freeze_components=[],
        fused_attn=False,
    ):
        super().__init__()

        self.feature_size = (
            image_size[0] // patch_size,  # width
            image_size[1] // patch_size,  # height
        )

        self.key_encoder = key_encoder
        self.value_encoder = value_encoder
        self.reg_head = reg_head
        self.geo_head = geo_head
        self.memo = SlotMem(
            embed_dim=embed_dim,
            decode_steps=decode_steps,
            decode_heads=decode_heads,
            fused_attn=fused_attn,
        )


        for name in freeze_components:
            if hasattr(self, name):
                module = getattr(self, name)
                if isinstance(module, nn.Module):
                    for param in module.parameters():
                        param.requires_grad = False
                else:
                    raise ValueError(f"{name} is not a nn.Module")
                print(f"Freezing {name} parameters")

    def forward(
        self,
        refs,
        ref_roc_masks,
        query,
        query_roc_masks,
        learning_step=None,
        query_target=None,
        contrastive_data={},
        point_data={},
    ):
        B = query.shape[0]
        H, W = query.shape[2], query.shape[3]
        N = refs.shape[0]
        # Extract features from the backbone
        query_feature = self.key_encoder(query)
        key_features = []
        value_features = []
        assert N == 1
        for i in range(N):
            if learning_step is not None and i != learning_step:
                with torch.inference_mode():
                    key_features.append(self.key_encoder(refs[i]))
                    value_features.append(self.value_encoder(ref_roc_masks[i]))
            else:
                key_features.append(self.key_encoder(refs[i]))
                value_features.append(self.value_encoder(ref_roc_masks[i]))


        query_feature = self.memo(key_features, value_features, query_feature)

        def vit2conv(x):
            return rearrange(
                x[:, x.shape[1] - self.feature_size[0] * self.feature_size[1] :],
                "b (h w) d -> b d h w",
                h=self.feature_size[1],
                w=self.feature_size[0],
            )

        resize_as = lambda x, y: F.interpolate(
            x.float(),
            size=y.shape[2:],
            mode="nearest",
        )

        query_feature = vit2conv(query_feature)
        out_data = {}

        # for regression models
        if self.reg_head == "vae.decoder":
            # VAE decoder
            out_data["reg_out"] = self.value_encoder.backbone_impl.vae_impl.decode(
                query_feature
            ).sample
        else:
            out_data["reg_out"] = self.reg_head(query_feature)

        if "reg_out" in out_data:
            roc, mask = out_data["reg_out"].split([3, 1], 1)
            out_data["roc"] = roc
            out_data["mask"] = mask.squeeze(1)

        if self.geo_head is not None:
            if not (
                isinstance(self.geo_head, diffusers.models.autoencoders.vae.Decoder)
                or isinstance(self.geo_head, GeoRegHead)
            ):
                kw_args = {}
                # if isinstance(self.geo_head, NOCSARHead):
                #     kw_args["visib_mask"] = resize_as(
                #         out_data["mask"].unsqueeze(1), query_feature
                #     ).squeeze(1)
                kw_args["show_steps"] = False if self.training else True
                geo_out = self.geo_head(
                    query_feature,
                    # query_target[:, :3], # for debug
                    query_target[:, :3] if self.training else None,
                    **kw_args,
                )
                out_data.update(geo_out)
            else:
                out_data["geo"] = self.geo_head(query_feature)

        """
            reconstruct input
        """
        if query_target is not None:
            if isinstance(self.value_encoder.backbone_impl, VAE):
                posterior = self.value_encoder.backbone_impl.encode(query_target)
                z = posterior.sample()
                out_data["geo_recon"] = self.value_encoder.backbone_impl.decode(z)
                out_data["kl_loss"] = posterior.kl().mean()
            else:
                if isinstance(
                    self.geo_head, diffusers.models.autoencoders.vae.Decoder
                ) or isinstance(self.geo_head, GeoRegHead):
                    recon_feature = self.value_encoder(query_target)
                    recon_feature = vit2conv(recon_feature)
                    out_data["geo_recon"] = self.geo_head(recon_feature)

            if "geo_recon" in out_data:
                gt = resize_as(query_target[:, :3, ...], out_data["geo_recon"])
                out_data["recon_loss"] = (out_data["geo_recon"] - gt).abs().mean()

        return out_data


# python -m src.models.coordar.memo
if __name__ == "__main__":
    # Example usage
    batch_size = 8
    key_in_channels = 3
    value_in_channels = 3
    out_channels = 3
    mem_size = 20
    embed_dim = 768

    device = "cuda:7"

    memo = MemoCoordMap(
        key_in_channels,
        value_in_channels,
        out_channels,
        mem_size=mem_size,
    ).to(device)

    num_examples = 100

    refs = torch.stack(
        [
            torch.randn(batch_size, key_in_channels, 224, 224).to(device)
            for _ in range(num_examples)
        ],
        0,
    )
    values = torch.stack(
        [
            torch.randn(batch_size, value_in_channels, 224, 224).to(device)
            for _ in range(num_examples)
        ],
        0,
    )
    queries = torch.randn(batch_size, key_in_channels, 224, 224).to(device)
    output = memo(refs, values, queries)

    print(
        "Output shape:", output.shape
    )  # Should be [batch_size, out_channels, 224, 224]

    show_gpu_memory()
