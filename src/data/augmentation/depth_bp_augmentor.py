from matplotlib import pyplot as plt
import numpy as np
from perlin_numpy import (
    generate_fractal_noise_2d,
    generate_fractal_noise_3d,
    generate_perlin_noise_2d,
    generate_perlin_noise_3d,
)
import matplotlib.animation as animation


class DepthBPAugmentor:

    def __init__(self, noise_size=(256, 256), noise_std=0.2):
        self.res_list = [
            (16, 16),
            (8, 8),
            (4, 4),
            (2, 2),
            (1, 1),
        ]
        self.noise_size = noise_size
        self.noise_std = noise_std

    def __call__(self, depth_bp):
        h, w = depth_bp.shape[:2]
        res = self.res_list[np.random.randint(len(self.res_list))]
        noise = generate_fractal_noise_2d(self.noise_size, res, 5)
        mask = noise < 0.5
        mask_depth_bp = mask[:h, :w]
        depth_bp_add_noise = (
            depth_bp + self.noise_std * np.random.randn(*depth_bp.shape)
        ).clip(0, 1)
        depth_bp_aug = depth_bp_add_noise * mask_depth_bp[..., None]
        return depth_bp_aug.astype(depth_bp.dtype)


# python -m src.data.augmentation.depth_augmentor
if __name__ == "__main__":
    np.random.seed(0)
    res = (16, 16)
    # res = (8, 8)
    # res = (4, 4)
    # res = (2, 2)
    # res = (1, 1)
    noise = generate_perlin_noise_2d((256, 256), res)
    plt.imshow(noise, cmap="gray", interpolation="lanczos")
    plt.colorbar()
    np.random.seed(0)
    noise = generate_fractal_noise_2d((256, 256), res, 5)
    plt.figure()
    plt.imshow(noise < 0.5, cmap="gray", interpolation="lanczos")
    plt.colorbar()
    plt.show()
