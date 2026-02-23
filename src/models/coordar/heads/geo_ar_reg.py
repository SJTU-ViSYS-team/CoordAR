import math
from einops import rearrange
import ipdb
import numpy as np
import torch
import torch.nn as nn
import scipy.stats as stats
from timm.models.vision_transformer import Block
from tqdm import tqdm


from src.models.layers.transformer import TransformerDecoderLayer
from src.models.tokenizer.model import Tokenizer
from src.utils.ckpt_loader import CkptLoader


def mask_by_order(mask_len, order, bsz, seq_len, device):
    masking = torch.zeros(bsz, seq_len).to(device)
    masking = torch.scatter(
        masking,
        dim=-1,
        index=order[:, : mask_len.long()],
        src=torch.ones(bsz, seq_len).to(device),
    ).bool()
    return masking


# a head similar to AR but based on regression
class GeoRegHead(nn.Module):

    def __init__(
        self,
        in_dim,
        decoder_depth=16,
        decoder_embed_dim=768,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        proj_dropout=0.1,
        attn_dropout=0.1,
        buffer_size=64,
        seq_len=14 * 14,
        mask_ratio_min=0.7,
        decode_from_memory=False,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.num_iter = 1
        self.decoder_embed_dim = decoder_embed_dim
        self.condition_proj = nn.Linear(in_dim, decoder_embed_dim, bias=True)
        self.seq_len = seq_len
        self.decode_from_memory = decode_from_memory

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed_learned = nn.Parameter(
            torch.zeros(1, self.seq_len + self.buffer_size, decoder_embed_dim)
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.out = nn.Linear(decoder_embed_dim, 3 * 16 * 16)

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    proj_drop=proj_dropout,
                    attn_drop=attn_dropout,
                )
                for _ in range(decoder_depth)
            ]
        )
        if self.decode_from_memory:
            self.memory_reading = nn.ModuleList(
                [
                    TransformerDecoderLayer(decoder_embed_dim, decoder_num_heads)
                    for _ in range(decoder_depth)
                ]
            )

        self.loss = nn.L1Loss(reduction="none")

        # --------------------------------------------------------------------------
        # MAR variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        self.mask_ratio_generator = stats.truncnorm(
            (mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25
        )

    def one_step_masking(self, x, orders):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        num_masked_tokens = orders.shape[1]
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(
            mask,
            dim=-1,
            index=orders[:, :num_masked_tokens],
            src=torch.ones(bsz, seq_len, device=x.device),
        )
        return mask

    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def forward_decoder(self, mask, pixel_condition, memory=None):
        bsz = mask.shape[0]
        embed_dim = self.decoder_embed_dim
        device = mask.device

        x = torch.zeros(bsz, self.buffer_size, embed_dim, device=device)
        mask_with_buffer = torch.cat(
            [torch.zeros(x.size(0), self.buffer_size, device=device), mask], dim=1
        )

        # pad mask tokens
        mask_tokens = self.mask_token.repeat(
            mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1
        ).to(x.dtype)
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(
            x.shape[0] * x.shape[1], x.shape[2]
        )
        pixel_condition_pad = torch.cat(
            [
                torch.zeros(bsz, self.buffer_size, embed_dim, device=device),
                pixel_condition,
            ],
            dim=1,
        )
        # decoder position embedding
        x = x_after_pad + self.decoder_pos_embed_learned

        for i, block in enumerate(self.decoder_blocks):
            x = block(x)
            x = x + pixel_condition_pad
            # cross attention with memory
            if self.decode_from_memory and memory is not None:
                x = self.memory_reading[i](x, memory["keys"], memory["values"])
        x = self.decoder_norm(x)

        x = x[:, self.buffer_size :]
        return x

    def forward_loss(self, xyz, target):
        bsz, seq_len, _, _ = target.shape
        xyz = rearrange(
            xyz, "b (h w) (c fh fw) -> b c (h fh) (w  fw)", c=3, fh=16, h=14
        )
        loss = self.loss(xyz, target)
        return loss.mean()

    def forward(self, latents, gt_geo=None, memory=None, show_steps=False):
        out_dict = {}
        h, w = latents.shape[2], latents.shape[3]
        latents = rearrange(latents, "b c h w -> b (h w) c")
        pixel_condition = self.condition_proj(latents)
        if gt_geo is not None:
            orders = self.sample_orders(bsz=latents.size(0))
            mask = self.one_step_masking(latents, orders)  # mask all
            # mae decoder
            z = self.forward_decoder(mask, pixel_condition, memory)
            xyz = self.out(z)
            out_dict["geo_loss"] = self.forward_loss(xyz=xyz, target=gt_geo)
        else:
            out_dict.update(
                self.forward_ar_(
                    pixel_condition,
                    h,
                    w,
                    memory=memory,
                    show_steps=show_steps,
                )
            )
        return out_dict

    def forward_ar_(
        self,
        pixel_condition,
        h,
        w,
        progress=False,
        orders=None,
        memory=None,
        show_steps=False,
    ):
        bsz = pixel_condition.size(0)
        out_dict = {}
        device = pixel_condition.device
        # init and sample generation orders
        mask = torch.ones(bsz, self.seq_len).to(device)
        if orders is None:
            orders = self.sample_orders(bsz)
        orders = self.sample_orders(bsz)
        indices = list(range(self.num_iter))
        if progress:
            indices = tqdm(indices)

        xyz = torch.zeros(bsz, self.seq_len, 3 * 16 * 16).to(device)
        # generate latents
        for step in indices:

            current_xyz = xyz.clone()

            # mae decoder
            z = self.forward_decoder(mask, pixel_condition, memory)

            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2.0 * (step + 1) / self.num_iter)
            mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).to(device)

            # masks out at least one for the next iteration
            mask_len = torch.maximum(
                torch.Tensor([1]).to(device),
                torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len),
            )

            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len, device)
            if step >= self.num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
            mask_curr = mask
            mask = mask_next
            # sample token latents for this step
            z = z[mask_to_pred.nonzero(as_tuple=True)]

            xyz_pred = self.out(z)
            current_xyz[mask_to_pred.nonzero(as_tuple=True)] = xyz_pred
            xyz = rearrange(
                current_xyz, "b (h w) (c fh fw) -> b c (h fh) (w  fw)", c=3, fh=16, h=14
            )
            geo = xyz
            if show_steps:
                out_dict.setdefault("geo_steps", []).append(geo)
                out_dict.setdefault("mask_steps", []).append(
                    rearrange(mask_curr.float(), "b (h w) -> b h w", h=h, w=w)
                )
        if show_steps:
            out_dict.setdefault("geo_steps", []).append(geo)
            out_dict.setdefault("mask_steps", []).append(
                rearrange(torch.zeros_like(mask).float(), "b (h w) -> b h w", h=h, w=w)
            )

            out_dict["geo_steps"] = torch.stack(out_dict["geo_steps"], dim=1)
            out_dict["mask_steps"] = torch.stack(out_dict["mask_steps"], dim=1)

        out_dict["geo"] = geo

        return out_dict


# python -m src.models.coordar.heads.geo_ar
if __name__ == "__main__":
    from src.models.tokenizer.model import Tokenizer

    tokenizer = Tokenizer(
        vq_args=dict(
            num_vq_embeddings=8192,
            latent_channels=768,
            layers_per_block=1,
            down_block_types=[
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ],
            up_block_types=[
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
            ],
            block_out_channels=[32, 64, 128, 256, 512],
            norm_num_groups=8,
        ),
        ckpt_loader=CkptLoader(
            base_ckpt="logs/train/runs/2025-05-30_21-37-21/checkpoints/last.ckpt",
            base_prefix="tokenizer.",
            target_prefix="",
        ),
    ).cuda()
    model = GeoRegHead(
        in_dim=768,
        tokenizer=tokenizer,
        decoder_depth=16,
        decoder_embed_dim=768,
        decoder_num_heads=16,
    ).cuda()
    x = torch.randn(2, 768, 14, 14).float().cuda()

    # loss
    gt_geo = torch.randn(2, 3, 224, 224).float().cuda()
    out = model.forward(x, gt_geo=gt_geo)
    print(out["geo_loss"])

    # inference
    out = model.forward(x)
    print(out["geo"].shape)
