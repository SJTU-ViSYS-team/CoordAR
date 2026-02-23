from matplotlib import pyplot as plt
import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmenters import (
    Sequential,
    SomeOf,
    OneOf,
    Sometimes,
    WithColorspace,
    WithChannels,
    Noop,
    Lambda,
    AssertLambda,
    AssertShape,
    Scale,
    CropAndPad,
    Pad,
    Crop,
    Fliplr,
    Flipud,
    Superpixels,
    ChangeColorspace,
    PerspectiveTransform,
    Grayscale,
    GaussianBlur,
    AverageBlur,
    MedianBlur,
    Convolve,
    Sharpen,
    Emboss,
    EdgeDetect,
    DirectedEdgeDetect,
    Add,
    AddElementwise,
    AdditiveGaussianNoise,
    Multiply,
    MultiplyElementwise,
    Dropout,
    CoarseDropout,
    Invert,
    ContrastNormalization,
    Affine,
    PiecewiseAffine,
    ElasticTransformation,
    pillike,
    LinearContrast,
)
import torchvision.transforms as transforms

AUG_CONFIG = {
    "base": transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    "gdr": Sequential(
        [
            Sometimes(0.5, CoarseDropout(p=0.2, size_percent=0.05)),
            Sometimes(0.4, GaussianBlur((0.0, 3.0))),
            Sometimes(0.3, pillike.EnhanceSharpness(factor=(0.0, 50.0))),
            Sometimes(0.3, pillike.EnhanceContrast(factor=(0.2, 50.0))),
            Sometimes(0.5, pillike.EnhanceBrightness(factor=(0.1, 6.0))),
            Sometimes(0.3, pillike.EnhanceColor(factor=(0.0, 20.0))),
            Sometimes(0.5, Add((-25, 25), per_channel=0.3)),
            Sometimes(0.3, Invert(0.2, per_channel=True)),
            Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),
            Sometimes(0.5, Multiply((0.6, 1.4))),
            Sometimes(0.1, AdditiveGaussianNoise(scale=10, per_channel=True)),
            Sometimes(0.5, iaa.contrast.LinearContrast((0.5, 2.2), per_channel=0.3)),
            Sometimes(0.5, Grayscale(alpha=(0.0, 1.0))),  # maybe remove for det
        ],
        random_order=True,
    ),
    "refine": Sequential(
        [
            Sometimes(0.4, GaussianBlur((0.0, 3.0))),
            Sometimes(0.3, pillike.EnhanceSharpness(factor=(0.0, 50.0))),
            Sometimes(0.3, pillike.EnhanceContrast(factor=(0.2, 50.0))),
            Sometimes(0.5, pillike.EnhanceBrightness(factor=(0.1, 6.0))),
            Sometimes(0.3, pillike.EnhanceColor(factor=(0.0, 20.0))),
        ]
    ),
}


class ColorAugmentor:

    def __init__(self, color_aug_prob=0.8, method="gdr"):
        self.color_aug_prob = color_aug_prob
        self.method = method

    def __call__(self, rgb_patch):
        if np.random.rand() < self.color_aug_prob:
            aug_color = AUG_CONFIG[self.method]._to_deterministic()
            rgb_patch = aug_color.augment_image(rgb_patch)
        return rgb_patch


# python -m src.data.components.color_augmentor
if __name__ == "__main__":
    from tqdm import tqdm
    from hydra.experimental import compose, initialize
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    import rootutils
    from torchvision.utils import save_image

    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    with initialize(config_path="../../../configs/"):
        cfg = compose(config_name="train.yaml")
    OmegaConf.set_struct(cfg, False)

    datamodule = instantiate(cfg.data)
    dataloader = datamodule.train_dataloader()

    augmentor = ColorAugmentor(1.0)
    for i, batch in tqdm(enumerate(dataloader)):
        rgb = batch["rgb_patch"].permute(0, 2, 3, 1).cpu().numpy()
        mask = batch["vis_mask_patch"].cpu().numpy()

        bs = len(rgb)
        fig, axs = plt.subplots(bs, 3)
        for i in range(bs):
            axs[i, 0].imshow((rgb[i]))
            axs[i, 1].imshow(mask[i])
            rgb_aug = augmentor(rgb[i])
            axs[i, 2].imshow(rgb_aug)

        plt.show()
