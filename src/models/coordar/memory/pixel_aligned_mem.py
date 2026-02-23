import ipdb
import torch
import torch.nn as nn
from src.models.layers.transformer import TransformerDecoderLayer


class SlotMem(nn.Module):

    def __init__(
        self,
        embed_dim=768,
        decode_steps=4,
        decode_heads=4,
        buffer_size=64,
        fused_attn=False,
    ):
        super(SlotMem, self).__init__()

        self.embed_dim = embed_dim
        self.buffer_size = buffer_size
        self.decode_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(embed_dim, decode_heads, fused_attn=fused_attn)
                for _ in range(decode_steps)
            ]
        )
        self.pos_embed = nn.Parameter(torch.randn(1, 1000, embed_dim))

    def forward(self, ks, vs, q):
        B = q.shape[0]
        N = len(ks)
        mem_k = []
        mem_v = []
        for i in range(N):
            assert ks[i].ndim == 3
            mem_k.append(ks[i])
            mem_v.append(vs[i])
        mem_k = torch.cat(mem_k, dim=1)
        mem_v = torch.cat(mem_v, dim=1)

        out = self.decode(q, mem_k, mem_v)
        return out

    def decode(self, q, mem_k, mem_v):
        B, L = q.shape[:2]
        assert (
            L <= self.pos_embed.shape[1]
        ), f"Input length {L} exceeds max length {self.pos_embed.shape[1]}"

        # concat buffer
        q = torch.cat(
            [torch.zeros(B, self.buffer_size, self.embed_dim, device=q.device), q],
            dim=1,
        )

        for layer in self.decode_layers:
            q = q + self.pos_embed[:, : L + self.buffer_size]
            q = layer(q, mem_k, mem_v)

        q = q[:, self.buffer_size :]
        return q
