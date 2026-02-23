import torch
import torch.nn as nn
from diffusers import AutoencoderKL


class FiLM(nn.Module):
    def __init__(self, cond_dim, feat_dim):
        super().__init__()
        self.gamma_fc = nn.Linear(cond_dim, feat_dim)
        self.beta_fc = nn.Linear(cond_dim, feat_dim)

    def forward(self, x, cond):
        gamma = self.gamma_fc(cond).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = self.beta_fc(cond).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return gamma * x + beta


class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim=4, cond_dim=128, **init_args):
        super().__init__()
        self.vae_impl = AutoencoderKL(**init_args)
        self.cond_proj = nn.Linear(cond_dim, latent_dim)  # 把条件投影到 latent 维度

    def forward(self, x, conditioning=None):
        posterior = self.vae_impl.encode(x).latent_dist
        z = posterior.sample()

        if conditioning is not None:
            cond_feat = self.cond_proj(conditioning)  # (B, latent_dim)
            cond_feat = cond_feat.unsqueeze(-1).unsqueeze(-1)  # (B, latent_dim, 1, 1)
            cond_feat = cond_feat.expand(-1, -1, z.shape[-2], z.shape[-1])  # broadcast
            z = z + cond_feat  # 融合条件（也可拼接）

        return z


# python -m src.models.coordar.backbones.conditional_vae
if __name__ == "__main__":
    vae = ConditionalVAE(
        latent_dim=4,
        cond_dim=6,
        in_channels=3,
        out_channels=3,
        block_out_channels=(128, 256, 512, 512),
        latent_channels=4,
        down_block_types=[
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
        ],
    )

    x = torch.randn(4, 3, 256, 256)
    cond = torch.randn(4, 6)
    z = vae(x, conditioning=cond)
