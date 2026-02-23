import math
from einops import rearrange
import ipdb
import numpy as np
import torch
import torch.nn as nn
import scipy.stats as stats
from timm.models.vision_transformer import Block
from tqdm import tqdm
import torch.nn.functional as F

from src.models.layers.adaln import AdaLN_EW
from src.models.layers.transformer import TransformerDecoderLayer
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


class GeoARHead(nn.Module):

    def __init__(
        self,
        in_dim,
        tokenizer: torch.nn.Module,
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
        mask_scale=0.25,
        num_iter=64,
        decode_from_memory=False,
        scheduler="cosine",
        infer_order="random",
        do_sample=False,
        temperature=1.0,
        condition_type="v1",
        cfg=1.0,
        cfg_schedule="constant",
        label_drop_prob=0.1,
        reorder=False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.buffer_size = buffer_size
        self.num_iter = num_iter
        self.seq_len = seq_len
        self.decode_from_memory = decode_from_memory
        self.scheduler = scheduler
        self.infer_order = infer_order
        self.do_sample = do_sample
        self.temperature = temperature
        self.condition_type = condition_type
        self.cfg = cfg
        self.cfg_schedule = cfg_schedule
        self.label_drop_prob = label_drop_prob
        self.decoder_embed_dim = decoder_embed_dim

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed_learned = nn.Parameter(
            torch.zeros(1, self.seq_len + self.buffer_size, decoder_embed_dim)
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.out = nn.Linear(decoder_embed_dim, tokenizer.num_vq_embeddings)
        self.reorder = reorder

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
        if self.condition_type == "v1":
            self.condition_proj = nn.Linear(in_dim, decoder_embed_dim, bias=True)
        elif self.condition_type == "v2":
            self.condition_adalns = nn.ModuleList(
                [AdaLN_EW(decoder_embed_dim, in_dim) for _ in range(decoder_depth - 1)]
            )

        if self.decode_from_memory:
            self.memory_reading = nn.ModuleList(
                [
                    TransformerDecoderLayer(decoder_embed_dim, decoder_num_heads)
                    for _ in range(decoder_depth)
                ]
            )

        self.loss = nn.CrossEntropyLoss(reduction="none")

        # --------------------------------------------------------------------------
        # MAR variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std mask_scale
        self.mask_ratio_generator = stats.truncnorm(
            (mask_ratio_min - 1.0) / mask_scale, 0, loc=1.0, scale=mask_scale
        )

        # freeze tokenizer components
        for param in self.tokenizer.parameters():
            param.requires_grad = False

        self.initialize_weights()

    def initialize_weights(self):
        pass

    def random_masking(self, x, orders):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
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

    def raster_orders(self, bsz):
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))[::-1]
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def forward_decoder(self, x, mask, cond, memory=None):
        bsz, seq_len, embed_dim = x.shape

        # random drop condition embedding during training
        x = torch.cat(
            [torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1
        )

        mask_with_buffer = torch.cat(
            [torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1
        )
        # dropping
        x = x[(1 - mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)

        # pad mask tokens
        mask_tokens = self.mask_token.repeat(
            mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1
        ).to(x.dtype)
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(
            x.shape[0] * x.shape[1], x.shape[2]
        )
        if self.condition_type == "v1":
            cond = self.condition_proj(cond)
            cond_pad = torch.cat(
                [
                    torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device),
                    cond,
                ],
                dim=1,
            )
        elif self.condition_type == "v2":
            cond_pad = torch.cat(
                [
                    torch.zeros(bsz, self.buffer_size, cond.size(2)).to(x),
                    cond,
                ],
                dim=1,
            )
        else:
            raise NotImplementedError(f"invalid condition type {self.condition_type}")
        # decoder position embedding
        x = x_after_pad + self.decoder_pos_embed_learned

        for i, block in enumerate(self.decoder_blocks):
            if self.condition_type == "v1":
                x = block(x)
                x = x + cond_pad
            elif self.condition_type == "v2":
                x = block(x)
                if i != len(self.decoder_blocks) - 1:
                    x = self.condition_adalns[i](x, cond_pad)
            else:
                raise NotImplementedError(
                    f"invalid condition type {self.condition_type}"
                )
            # cross attention with memory
            if self.decode_from_memory and memory is not None:
                x = self.memory_reading[i](x, memory["keys"], memory["values"])
        x = self.decoder_norm(x)

        x = x[:, self.buffer_size :]
        return x

    def forward_loss(self, z, target):

        digit = self.out(z)
        digit = rearrange(digit, "b l c -> b c l")
        loss = self.loss(digit, target)
        return loss.mean()

    def forward(self, cond, gt_geo=None, memory=None, show_steps=False):
        out_dict = {}
        bsz, h, w = cond.shape[0], cond.shape[2], cond.shape[3]
        cond = rearrange(cond, "b c h w -> b (h w) c")
        if gt_geo is not None:
            orders = self.sample_orders(bsz=bsz)
            mask = self.random_masking(cond, orders)

            gt_indices = self.tokenizer.encode(gt_geo - 0.5)[2]
            gt_indices = rearrange(gt_indices, "b h w -> b (h w)")
            gt_tokens = self.tokenizer.emb(gt_indices)
            z = self.forward_decoder(gt_tokens, mask, cond, memory)
            out_dict["geo_loss"] = self.forward_loss(z, target=gt_indices)
        else:
            orders = None
            if self.reorder:
                ourder_out = self.forward_ar_(
                    cond,
                    h,
                    w,
                    memory=memory,
                    show_steps=show_steps,
                    num_iter=1,
                )
                confidence = ourder_out["confidence_steps"][:, -1, :, :]
                orders = torch.argsort(
                    rearrange(confidence, "b h w -> b (h w)"), dim=-1
                )
            out_dict.update(
                self.forward_ar_(
                    cond,
                    h,
                    w,
                    memory=memory,
                    show_steps=show_steps,
                    num_iter=self.num_iter,
                    orders=orders,
                )
            )
        return out_dict

    def forward_ar_(
        self,
        conditions,
        h,
        w,
        progress=False,
        orders=None,
        memory=None,
        show_steps=False,
        num_iter=1,
    ):
        bsz = conditions.size(0)
        out_dict = {}
        device = conditions.device
        # init and sample generation orders
        mask = torch.ones(bsz, self.seq_len).to(device)
        tokens = torch.zeros(bsz, self.seq_len, self.tokenizer.vq_embed_dim).to(device)
        confidence = torch.zeros(bsz, self.seq_len).to(tokens)
        if orders is None:
            if self.infer_order == "random":
                orders = self.sample_orders(bsz)
            elif self.infer_order == "raster":
                orders = self.raster_orders(bsz)
            else:
                raise NotImplementedError(f"invalid order {self.infer_order}")
        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)

        if not self.cfg == 1.0:
            conditions = torch.cat(
                [conditions, self.fake_latent.repeat(bsz, 1, 1)], dim=0
            )

        # generate latents
        for step in indices:
            cur_tokens = tokens.clone()
            # condition embedding and CFG
            if not self.cfg == 1.0:
                tokens = torch.cat([tokens, tokens], dim=0)
                mask = torch.cat([mask, mask], dim=0)

            # mae decoder
            z = self.forward_decoder(tokens, mask, conditions, memory)

            # mask ratio for the next round, following MaskGIT and MAGE.
            if self.scheduler == "cosine":
                mask_ratio = np.cos(math.pi / 2.0 * (step + 1) / num_iter)
            elif self.scheduler == "linear":
                mask_ratio = 1 - (step + 1) / num_iter
            else:
                raise NotImplementedError(f"invalid scheduler {self.scheduler}")
            mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).to(device)

            # masks out at least one for the next iteration
            mask_len = torch.maximum(
                torch.Tensor([1]).to(device),
                torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len),
            )

            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len, device)
            if step >= num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
            mask_curr = mask
            mask = mask_next
            if not self.cfg == 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)
            # sample token latents for this step
            z = z[mask_to_pred.nonzero(as_tuple=True)]

            sampled_token_latent, confidence_ = self.sample_discrete(z)
            confidence[mask_to_pred.nonzero(as_tuple=True)] = confidence_

            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()
            if show_steps:
                tokens_ = rearrange(tokens, "b (h w) c -> b c h w", h=h, w=w)
                geo = self.tokenizer.decode(tokens_) + 0.5
                out_dict.setdefault("geo_steps", []).append(geo)
                out_dict.setdefault("confidence_steps", []).append(
                    rearrange(confidence.clone(), "b (h w) -> b h w", h=h, w=w)
                )
                out_dict.setdefault("mask_steps", []).append(
                    rearrange(mask_curr.float(), "b (h w) -> b h w", h=h, w=w)
                )
        if show_steps:
            tokens_ = rearrange(tokens, "b (h w) c -> b c h w", h=h, w=w)
            geo = self.tokenizer.decode(tokens_) + 0.5
            out_dict.setdefault("geo_steps", []).append(geo)
            out_dict.setdefault("confidence_steps", []).append(
                rearrange(confidence.clone(), "b (h w) -> b h w", h=h, w=w)
            )
            out_dict.setdefault("mask_steps", []).append(
                rearrange(torch.zeros_like(mask).float(), "b (h w) -> b h w", h=h, w=w)
            )

            out_dict["geo_steps"] = torch.stack(out_dict["geo_steps"], dim=1)
            out_dict["mask_steps"] = torch.stack(out_dict["mask_steps"], dim=1)
            out_dict["confidence_steps"] = torch.stack(
                out_dict["confidence_steps"], dim=1
            )

        tokens = rearrange(tokens, "b (h w) c -> b c h w", h=h, w=w)
        # decode tokens
        geo = self.tokenizer.decode(tokens) + 0.5
        out_dict["geo"] = geo

        return out_dict

    def sample_discrete(self, z):
        sampled_token_digit = self.out(z)
        if self.do_sample:
            probs = F.softmax(sampled_token_digit / self.temperature, dim=-1)
            sampled_token_indices = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            sampled_token_indices = torch.argmax(sampled_token_digit, dim=1)
        confidence = torch.max(torch.softmax(sampled_token_digit, dim=1), dim=1).values

        sampled_token_latent = self.tokenizer.emb(sampled_token_indices)
        return sampled_token_latent, confidence


# python -m src.models.coordar.heads.geo_ar
if __name__ == "__main__":
    from src.models.tokenizer.vae import Tokenizer

    tokenizer = Tokenizer(
        vae_args=dict(embed_dim=16, ch_mult=[1, 1, 2, 2, 4]),
        ckpt_loader=CkptLoader(
            base_ckpt="logs/roc_tokenization/archive/roc_vae/2025-10-30_20-32-07/checkpoints/last.ckpt",
            base_prefix="tokenizer.",
            target_prefix="",
            base_strict=False,
        ),
    ).cuda()
    model = GeoARHead(
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
