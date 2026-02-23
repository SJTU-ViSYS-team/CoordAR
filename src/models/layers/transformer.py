import time
import torch
import torch.nn as nn
from torch.nn import functional as F

from src.models.layers.attention import Attention

class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        ffn_expand=4,
        dropout=0.0,
        enable_self_attn=True,
        fused_attn=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.ffn_expand = ffn_expand
        self.enable_self_attn = enable_self_attn
        self.fused_attn = fused_attn
        attn_cls = nn.MultiheadAttention
        if self.fused_attn:
            attn_cls = Attention

        # Self-Attention (带掩码)
        if enable_self_attn:
            self.self_attn = attn_cls(d_model, nhead, dropout=dropout, batch_first=True)
        # Cross-Attention (编码器-解码器注意力)
        self.cross_attn = attn_cls(d_model, nhead, dropout=dropout, batch_first=True)

        # Feed-Forward Network (按扩展倍数计算维度)
        self.linear1 = nn.Linear(d_model, d_model * ffn_expand)
        self.linear2 = nn.Linear(d_model * ffn_expand, d_model)

        # Normalization layers
        if enable_self_attn:
            self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # 或 ReLU

    def forward(
        self,
        tgt,  # 解码器输入 (batch, seq_len, d_model)
        memory_keys,  # 编码器输出 (batch, seq_len, d_model)
        memory_values,  # 编码器输出 (batch, seq_len, d_model)
        tgt_mask=None,  # 解码器自注意力掩码（防止未来信息泄漏）
        memory_mask=None,  # 编码器-解码器注意力掩码（通常为None）
        tgt_key_padding_mask=None,  # 解码器输入的padding掩码
        memory_key_padding_mask=None,  # 编码器输出的padding掩码
    ):
        # Pre-LayerNorm 结构（更稳定）

        # 1. Masked Self-Attention + 残差
        if self.enable_self_attn:
            tgt2 = self.norm1(tgt)
            tgt2 = self.self_attn(
                tgt2,
                tgt2,
                tgt2,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
            )[0]
            tgt = tgt + self.dropout(tgt2)

        # 2. Cross-Attention + 残差
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn(
            tgt2,
            memory_keys,
            memory_values,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        # 3. Feed-Forward Network + 残差
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)

        return tgt  # 输出: (batch, seq_len, d_model)

# python -m src.models.layers.transformer
if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size = 8
    seq_len_q = 100
    seq_len_kv = 10000
    embed_dim = 768
    num_heads = 8
    fused_attn = True
    # 初始化模型
    block = TransformerDecoderLayer(
        embed_dim, num_heads, dropout=0.1, fused_attn=fused_attn
    ).cuda()

    # 随机输入
    q = torch.randn(batch_size, seq_len_q, embed_dim, requires_grad=True, device="cuda")
    k = torch.randn(
        batch_size, seq_len_kv, embed_dim, requires_grad=True, device="cuda"
    )
    v = torch.randn(
        batch_size, seq_len_kv, embed_dim, requires_grad=True, device="cuda"
    )
    attn_mask = torch.randint(
        0, 2, (seq_len_q, seq_len_kv), dtype=torch.bool, device="cuda"
    )
    start_mem = torch.cuda.memory_allocated()
    start_time = time.time()

    x = q
    for i in range(10):
        x = block(x, k, v)
    loss = x.sum()
    loss.backward()
    print("Backward success, grad shape:", q.grad.shape)

    end_time = time.time()
    end_mem = torch.cuda.memory_allocated()
    peak_mem = torch.cuda.max_memory_allocated()
    print(f"[block] 耗时: {end_time - start_time:.4f} 秒")
    print(
        f"[block] 显存: "
        f"起始 {start_mem/1024**2:.2f} MB, "
        f"结束 {end_mem/1024**2:.2f} MB, "
        f"峰值 {peak_mem/1024**2:.2f} MB"
    )
