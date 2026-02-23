from loguru import logger
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class PosEncoding(nn.Module):
    def __init__(
        self, d_model=32, scale=1000, min_freq=1e-3, std=1, center=0, normalize=True
    ):
        super(PosEncoding, self).__init__()
        self.d_model = d_model
        self.scale = scale
        self.min_freq = min_freq
        self.std = std
        self.center = center
        self.normalize = normalize

    def forward(self, x):
        # x: (b) in pixel coordinate or 3D space, expected to be [0,1] when scale=1000, and min_freq=1e-3
        # output: (b, d_model)

        # normalize to [0,1]
        if self.normalize:
            x = (x - self.center) / (6 * self.std) + 0.5
            if torch.logical_and(x >= 0, x <= 1).sum() / torch.ones_like(x).sum() < 0.9:
                print("warning!, expect x in [0,1] for positional encoding.")
            x = x.clip(0, 1)
        x = x * self.scale
        # freqs \in [min_freq,1]
        freqs = self.min_freq ** (
            2 * (torch.arange(self.d_model).to(x) // 2) / self.d_model
        )
        # [min_freq, 1]

        pos_enc = x.unsqueeze(1) * freqs.view(1, self.d_model)
        # phase: (b, d_model)
        pos_enc[:, ::2] = pos_enc[:, ::2].cos()
        pos_enc[:, 1::2] = pos_enc[:, 1::2].sin()
        return pos_enc

    def reweight(self, loss):
        # weights for loss
        # loss (b, d_model)
        freqs = self.min_freq ** (
            2 * (torch.arange(self.d_model).to(loss) // 2) / self.d_model
        )
        w = 1 / freqs * self.min_freq
        return w.unsqueeze(0) * loss

    def reverse(self, v, weighted=False, verbose=False):
        # v: (b, d_model)
        l = torch.linspace(0, 1, self.scale + 1).to(v) * self.scale
        freqs = self.min_freq ** (
            2 * (torch.arange(self.d_model).to(v) // 2) / self.d_model
        )
        lookup = l.unsqueeze(1) * freqs.unsqueeze(0)

        lookup[:, ::2] = lookup[:, ::2].cos()
        lookup[:, 1::2] = lookup[:, 1::2].sin()

        # weighted by frequency
        if weighted:
            w = 1 / freqs
        else:
            w = torch.ones_like(freqs)
        v_w = w * v
        lookup_w = w * lookup
        dist = torch.cdist(v_w, lookup_w)
        match_indices = dist.argmin(dim=1)

        # (scale, d_model)
        if verbose:
            fig = plt.figure()
            ax = fig.add_subplot(221)
            ax.plot(freqs)
            ax.set_title("frequency")
            ax = fig.add_subplot(222)
            ax.plot(dist[0])
            ax.set_title("distance distribution")
            ax = fig.add_subplot(223)
            ax.plot(v[0], label="encoded")
            ax.plot(lookup[match_indices[0]], label="best match")
            ax.plot(lookup[8000], label="should match")
            plt.legend()
            ax.set_title("encoding")
            plt.savefig("debug/dist.png")
            plt.close()
        best_match = l[match_indices] / self.scale
        if self.normalize:
            best_match = (best_match - 0.5) * (6 * self.std) + self.center
        return best_match


@logger.catch
def run_reverse_pos():
    x = torch.rand(8)
    # x = torch.tensor([0.8000, 0.80005])
    pos = PosEncoding(scale=10000, min_freq=1e-4, normalize=False)
    v = pos(x)

    noise_acc = []
    for noise_level in torch.logspace(start=-5, end=0, steps=51):
        x_h = pos.reverse(v + torch.randn_like(v) * noise_level, weighted=False)
        noise_acc.append([noise_level, (x - x_h).norm()])

    noise_acc = torch.tensor(noise_acc)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(noise_acc[:, 0], noise_acc[:, 1])
    ax.set_xlabel("noise_level")
    ax.set_xscale("log")
    ax.set_ylabel("reconstruction error")
    plt.savefig("debug/noise_acc.png")
    plt.close()


if __name__ == "__main__":
    run_reverse_pos()
