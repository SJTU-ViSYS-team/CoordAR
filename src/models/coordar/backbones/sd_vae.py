from diffusers import AutoencoderKL
import ipdb
import torch
import torch.nn as nn


class VAETokenizer(nn.Module):
    def __init__(self, freeze=True, in_dim=3, out_dim=4):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="vae"
        )
        if in_dim != 3:
            # modify the input channels of VAE encoder
            old_layer = self.vae.encoder.conv_in
            self.vae.encoder.conv_in = torch.nn.Conv2d(
                in_dim,
                old_layer.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            # init weights from old conv
            if in_dim > 3:
                self.vae.encoder.conv_in.weight.data[:, :3, :, :] = (
                    old_layer.weight.data
                )
                for i in range(3, in_dim):
                    self.vae.encoder.conv_in.weight.data[:, i : i + 1, :, :] = 0
            else:
                self.vae.encoder.conv_in.weight.data = old_layer.weight.data[
                    :, :in_dim, :, :
                ]
        self.out_dim = out_dim
        if out_dim != 4:
            # modify the output channels of VAE decoder
            self.conv_out = torch.nn.Conv2d(
                4,
                out_dim,
                kernel_size=1,
            )
        # freeze VAE
        if freeze:
            for param in self.vae.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def forward(self, images):
        latents = self.vae.encode(images).latent_dist.mean  # [B, C, H, W]
        if self.out_dim != 4:
            latents = self.conv_out(latents)
        latents = latents.flatten(2).transpose(1, 2)  # [B, H*W, C]（类似序列）
        return latents


# python -m src.models.coordar.backbones.sd_vae
if __name__ == "__main__":
    model = VAETokenizer(freeze=False, in_dim=4, out_dim=768).cuda()
    x = torch.randn(2, 4, 224, 224).cuda()
    with torch.no_grad():
        latents = model(x)
    print(latents.shape)  # should be [2, H*W, 768]
