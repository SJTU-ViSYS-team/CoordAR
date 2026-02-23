from einops import rearrange
import ipdb
import torch
from diffusers import AutoencoderKL
import torch.nn as nn

from src.models.layers.adaln import AdaLN_Spatial


class DINO_SDVAE(nn.Module):
    def __init__(
        self,
        dino="dinov2_vitl14",
        patch_size=14,
        target_size=(16, 16),  # w, h
    ):
        super().__init__()
        self.dinov2 = torch.hub.load("facebookresearch/dinov2", dino)
        self.patch_size = patch_size
        self.target_size = (int(target_size[0]), int(target_size[1]))

        self.vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="vae"
        )

        self.fuse = AdaLN_Spatial(self.dinov2.embed_dim, 4)

        # freeze VAE
        for param in self.vae.parameters():
            param.requires_grad = False

    def forward(self, x):
        h, w = x.shape[-2:]
        dino_features_dict = self.dinov2.forward_features(x)
        dino_features = dino_features_dict["x_norm_patchtokens"]

        resize = lambda t, size: torch.nn.functional.interpolate(
            t,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        dino_features = rearrange(
            dino_features,
            "b (h w) c -> b c h w",
            h=h // self.patch_size,
            w=w // self.patch_size,
        )
        # reshape to 28*28
        dino_features = resize(dino_features, size=(28, 28))

        # SD VAE encoding
        vae_latents = self.vae.encode(x).latent_dist.mean

        # fuse DINO and VAE features
        features = self.fuse(dino_features, vae_latents)
        if self.target_size != (28, 28):
            features = torch.nn.functional.interpolate(
                features,
                size=self.target_size,
                mode="bilinear",
                align_corners=False,
            )
        return features


# python -m src.models.coordar.backbones.dino_sdvae
if __name__ == "__main__":

    tokenizer = DINO_SDVAE().to("cuda")
    images = torch.randn(2, 3, 224, 224).to("cuda")
    latents = tokenizer(images)
    print(latents.shape)  # [2, 784, 4]
