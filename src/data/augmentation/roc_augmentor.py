from matplotlib import pyplot as plt
import numpy as np
from perlin_numpy import (
    generate_fractal_noise_2d,
    generate_fractal_noise_3d,
    generate_perlin_noise_2d,
    generate_perlin_noise_3d,
)
import matplotlib.animation as animation


class ROCAugmentor:

    def __init__(self, noise_size=(256, 256)):
        self.res_list = [
            (16, 16),
            (8, 8),
            (4, 4),
            (2, 2),
            (1, 1),
        ]
        self.noise_size = noise_size

    def __call__(self, roc):
        h, w = roc.shape[:2]
        res = self.res_list[np.random.randint(len(self.res_list))]
        noise = generate_fractal_noise_2d(self.noise_size, res, 5)
        mask = noise < 0.5
        mask_roc = mask[:h, :w]
        roc_aug = roc * mask_roc[..., None] + 0.5 * (1 - mask_roc[..., None])
        return roc_aug.astype(roc.dtype)


# python -m src.data.augmentation.roc_augmentor
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
