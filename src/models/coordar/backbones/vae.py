from diffusers import AutoencoderKL
import ipdb
import torch
import torch.nn as nn


class VAE(nn.Module):

    def __init__(self, **init_args):
        super().__init__()
        self.vae_impl = AutoencoderKL(**init_args)

    def forward(self, x):
        posterior = self.vae_impl.encode(x).latent_dist
        x = posterior.sample()
        return x

    def encode(self, x):
        posterior = self.vae_impl.encode(x).latent_dist
        return posterior

    def decode(self, z):
        return self.vae_impl.decode(z).sample
