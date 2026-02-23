import time
from typing import Optional
from einops import rearrange
import ipdb
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.amp import autocast
from torch.nn import functional as F
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_


class Attention(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.bias = bias
        self.batch_first = batch_first

        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.wq = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.wk = nn.Linear(self.kdim, self.embed_dim, bias=bias)
        self.wv = nn.Linear(self.vdim, self.embed_dim, bias=bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.wq.weight)
        xavier_uniform_(self.wk.weight)
        xavier_uniform_(self.wv.weight)
        xavier_uniform_(self.out_proj.weight)

        if self.bias:
            constant_(self.wq.bias, 0.0)
            constant_(self.wk.bias, 0.0)
            constant_(self.wv.bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        q,
        k,
        v,
        attn_mask=None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        # attn_mask: (B, L, S)
        # key_padding_mask: (B, S)
        # 如果输入是 (B, L, E) 且 batch_first=True，保持不变
        # 如果是 (L, B, E)，就换成 (B, L, E)
        if not self.batch_first:
            # (L, B, E) -> (B, L, E)
            q, k, v = [x.transpose(0, 1) for x in (q, k, v)]
        B, L, _ = q.shape
        B, S, _ = k.shape
        q, k, v = self.wq(q), self.wk(k), self.wv(v)

        # 拆成多头: (B, L, num_heads, head_dim) -> (B, num_heads, L, head_dim)
        def shape(x):
            return x.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = map(shape, (q, k, v))
        if attn_mask is None:
            attn_mask = torch.ones(
                B, self.num_heads, L, S, device=q.device, dtype=torch.bool
            )
        if key_padding_mask is not None:
            key_mask = key_padding_mask[:, None, :].to(torch.bool)  # (B, 1, S)
            if attn_mask is None:
                attn_mask = key_mask
            else:
                attn_mask = attn_mask | key_mask  # 屏蔽的地方取并集

        # with sdpa_kernel(SDPBackend.FLASH_ATTENTION): # 快，但不支持复杂的attn_mask，仅限half，bf16
        with sdpa_kernel(
            SDPBackend.EFFICIENT_ATTENTION
        ):  # 较快，支持复杂的mask，支持float
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,  # fail if attn_mask is given
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )  # (B, num_heads, L, head_dim)
        # 还原维度: (B, L, E)
        attn_output = rearrange(attn_output, "b h l d -> b l (h d)")

        # 最后的输出 projection
        out = self.out_proj(attn_output)

        # (B, L, E) -> (L, B, E) 如果 batch_first=False
        if not self.batch_first:
            out = out.transpose(0, 1)

        attn_output_weights = None  # 和原生API保持一致

        return out, attn_output_weights


def test_attn():
    x = torch.randn(2, 8, 1000, 64, device="cuda", dtype=torch.float16)
    out = F.scaled_dot_product_attention(x, x, x)  # warm up

    start_mem = torch.cuda.memory_allocated()
    start_time = time.time()

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for i in range(100):
            F.scaled_dot_product_attention(x, x, x)
    end_time = time.time()
    end_mem = torch.cuda.memory_allocated()
    peak_mem = torch.cuda.max_memory_allocated()
    print(f"[flash] 耗时: {end_time - start_time:.4f} 秒")
    print(
        f"[flash] 显存: "
        f"起始 {start_mem/1024**2:.2f} MB, "
        f"结束 {end_mem/1024**2:.2f} MB, "
        f"峰值 {peak_mem/1024**2:.2f} MB"
    )

    start_mem = torch.cuda.memory_allocated()
    start_time = time.time()
    with sdpa_kernel([SDPBackend.MATH]):
        for i in range(100):
            F.scaled_dot_product_attention(x, x, x)
    end_time = time.time()
    end_mem = torch.cuda.memory_allocated()
    peak_mem = torch.cuda.max_memory_allocated()
    print(f"[pytorch] 耗时: {end_time - start_time:.4f} 秒")
    print(
        f"[pytorch] 显存: "
        f"起始 {start_mem/1024**2:.2f} MB, "
        f"结束 {end_mem/1024**2:.2f} MB, "
        f"峰值 {peak_mem/1024**2:.2f} MB"
    )

    start_mem = torch.cuda.memory_allocated()
    start_time = time.time()
    for i in range(100):
        F.scaled_dot_product_attention(x, x, x)
    end_time = time.time()
    end_mem = torch.cuda.memory_allocated()
    peak_mem = torch.cuda.max_memory_allocated()
    print(f"[auto] 耗时: {end_time - start_time:.4f} 秒")
    print(
        f"[auto] 显存: "
        f"起始 {start_mem/1024**2:.2f} MB, "
        f"结束 {end_mem/1024**2:.2f} MB, "
        f"峰值 {peak_mem/1024**2:.2f} MB"
    )


def test_attention():
    torch.manual_seed(42)
    batch_size = 8
    seq_len_q = 100
    seq_len_kv = 10000
    embed_dim = 768
    num_heads = 8
    # 初始化模型
    attn = Attention(embed_dim, num_heads, dropout=0.1, batch_first=True).cuda()

    # 随机输入
    q = torch.randn(batch_size, seq_len_q, embed_dim, requires_grad=True, device="cuda")
    k = torch.randn(
        batch_size, seq_len_kv, embed_dim, requires_grad=True, device="cuda"
    )
    v = torch.randn(
        batch_size, seq_len_kv, embed_dim, requires_grad=True, device="cuda"
    )

    # ====== 随机生成 attn_mask ======
    # mask 的形状: (seq_len_q, seq_len_kv)，其中 True 表示 -inf
    attn_mask = torch.randint(
        0, 2, (seq_len_q, seq_len_kv), dtype=torch.bool, device="cuda"
    )
    # print("attn_mask:\n", attn_mask)

    # 前向
    start_mem = torch.cuda.memory_allocated()
    start_time = time.time()

    out = attn(q, k, v, attn_mask)
    end_time = time.time()
    print("output shape:", out.shape)  # [batch, seq_len_q, embed_dim]
    # print("output (first batch):\n", out[0])
    print(f"[forward] 耗时: {end_time - start_time:.4f} 秒")
    # 反向传播测试
    loss = out.sum()
    loss.backward()
    print("Backward success, grad shape:", q.grad.shape)

    end_mem = torch.cuda.memory_allocated()
    peak_mem = torch.cuda.max_memory_allocated()
    print(
        f"[forwad] 显存: "
        f"起始 {start_mem/1024**2:.2f} MB, "
        f"结束 {end_mem/1024**2:.2f} MB, "
        f"峰值 {peak_mem/1024**2:.2f} MB"
    )


def compare_attetion():
    torch.manual_seed(42)
    batch_size = 8
    seq_len_q = 100
    seq_len_kv = 10000
    embed_dim = 768
    num_heads = 8
    # 初始化模型
    attn = Attention(embed_dim, num_heads, dropout=0.1, batch_first=True).cuda()

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

    for i in range(10):
        out = attn(q, k, v, attn_mask)
    end_time = time.time()
    end_mem = torch.cuda.memory_allocated()
    peak_mem = torch.cuda.max_memory_allocated()
    # print("output (first batch):\n", out[0])
    print(f"[custom] 耗时: {end_time - start_time:.4f} 秒")
    print(
        f"[custom] 显存: "
        f"起始 {start_mem/1024**2:.2f} MB, "
        f"结束 {end_mem/1024**2:.2f} MB, "
        f"峰值 {peak_mem/1024**2:.2f} MB"
    )

    pt_attn = nn.MultiheadAttention(
        embed_dim, num_heads, dropout=0.1, batch_first=True
    ).cuda()
    start_mem = torch.cuda.memory_allocated()
    start_time = time.time()
    for i in range(10):
        out = pt_attn(q, k, v, attn_mask=attn_mask)[0]
    end_time = time.time()
    end_mem = torch.cuda.memory_allocated()
    peak_mem = torch.cuda.max_memory_allocated()
    # print("output (first batch):\n", out[0])
    print(f"[pytorch] 耗时: {end_time - start_time:.4f} 秒")
    print(
        f"[pytorch] 显存: "
        f"起始 {start_mem/1024**2:.2f} MB, "
        f"结束 {end_mem/1024**2:.2f} MB, "
        f"峰值 {peak_mem/1024**2:.2f} MB"
    )


# python -m src.models.layers.attention
if __name__ == "__main__":
    # test_attn()
    # test_attention()
    compare_attetion()
