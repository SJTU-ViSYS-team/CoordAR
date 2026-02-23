import torch
import torch.nn as nn


class AdaLN(nn.Module):
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        # 把条件向量映射成 gamma 和 beta
        self.fc = nn.Linear(cond_dim, hidden_dim * 2)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        # elementwise_affine=False 表示去掉固定 gamma, beta

    def forward(self, x, cond):
        """
        x: [B, N, hidden_dim]  输入特征
        cond: [B, cond_dim]    条件向量 (如 time embedding)
        """
        # 生成 gamma 和 beta
        gamma, beta = self.fc(cond).chunk(2, dim=-1)  # 各 [B, hidden_dim]

        # LN
        x_norm = self.norm(x)

        # 加条件调制
        out = x_norm * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        return out


class AdaLN_EW(nn.Module):
    """
    Element-wise Adaptive Layer Normalization
    Applies conditioning at each token position independently
    """

    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        # 把条件向量映射成 gamma 和 beta
        self.fc = nn.Linear(cond_dim, hidden_dim * 2)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        # elementwise_affine=False 表示去掉固定 gamma, beta

        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, cond):
        """
        x: [B, N, hidden_dim]  输入特征
        cond: [B, N, cond_dim] 条件向量 (每个token位置都有对应的条件)
        """
        B, N, hidden_dim = x.shape

        # 对每个位置的条件向量生成 gamma 和 beta
        cond_reshaped = cond.reshape(-1, cond.size(-1))  # [B*N, cond_dim]
        gamma_beta = self.fc(cond_reshaped)  # [B*N, hidden_dim*2]
        gamma_beta = gamma_beta.reshape(B, N, -1)  # [B, N, hidden_dim*2]

        gamma, beta = gamma_beta.chunk(2, dim=-1)  # 各 [B, N, hidden_dim]

        # LN
        x_norm = self.norm(x)

        # 加条件调制 (element-wise)
        out = x_norm * (1 + gamma) + beta
        return out


class AdaLN_Spatial(nn.Module):
    """
    Spatial Element-wise Adaptive Layer Normalization for 2D feature maps
    Applies conditioning at each spatial position independently
    """

    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        # 把条件向量映射成 gamma 和 beta (使用1x1卷积)
        self.conv = nn.Conv2d(cond_dim, hidden_dim * 2, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=hidden_dim, affine=False)
        # affine=False 表示去掉固定 gamma, beta

        nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x, cond):
        """
        x: [B, hidden_dim, H, W]  输入特征
        cond: [B, cond_dim, H, W] 条件特征 (每个空间位置都有对应的条件)
        """
        # 对每个位置的条件特征生成 gamma 和 beta
        gamma_beta = self.conv(cond)  # [B, hidden_dim*2, H, W]
        gamma, beta = gamma_beta.chunk(2, dim=1)  # 各 [B, hidden_dim, H, W]

        # GroupNorm
        x_norm = self.norm(x)

        # 加条件调制 (element-wise)
        out = x_norm * (1 + gamma) + beta
        return out


# python -m src.models.layers.adaln
if __name__ == "__main__":
    # Test AdaLN_EW
    B, N, hidden_dim, cond_dim = 2, 8, 512, 128

    # Create test data
    x = torch.randn(B, N, hidden_dim)
    cond = torch.randn(B, N, cond_dim)

    # Initialize AdaLN_EW
    adaln_ew = AdaLN_EW(hidden_dim, cond_dim)

    # Forward pass
    out = adaln_ew(x, cond)

    print(f"Input shape: {x.shape}")
    print(f"Condition shape: {cond.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output mean: {out.mean():.4f}, std: {out.std():.4f}")

    # Test that each position gets different conditioning
    out1 = adaln_ew(x, cond)
    cond_modified = cond.clone()
    cond_modified[:, 0] += 1.0  # Modify first position conditioning
    out2 = adaln_ew(x, cond_modified)

    # Check that only the first position output changed significantly
    diff = (out2 - out1).abs()
    print(f"Max difference at pos 0: {diff[:, 0].max():.4f}")
    print(f"Max difference at pos 1: {diff[:, 1].max():.4f}")
    print("Test passed!" if diff[:, 0].max() > diff[:, 1].max() else "Test failed!")
