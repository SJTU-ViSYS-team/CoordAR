import hashlib
import os
import pickle
import random
import os.path as osp
from matplotlib import pyplot as plt
import numpy as np
import skimage

from src.utils.folder import prepare_dir
from src.utils.img_utils import resize_short_edge
from src.utils.python_ext import lazy_property
from src.utils.system import dump_with_lock


class BGAugmentor:
    def __init__(self, bg_type, bg_imgs_root, num_bg_imgs=10000):
        self.bg_type = bg_type
        self.bg_imgs_root = bg_imgs_root
        self.num_bg_imgs = num_bg_imgs

    def __call__(self, rgb, mask):
        bg_types = ["original", "clean", "random"]
        bg_type = random.choice(bg_types)
        if bg_type == "original":
            return rgb
        elif bg_type == "clean":
            rgb_aug = rgb * mask[..., None]
        elif bg_type == "random":
            rgb_aug = self.replace_bg(rgb, mask)
        return rgb_aug

    def replace_bg(self, im, im_mask, return_mask=False, truncate_fg=False):
        # add background to the image
        H, W = im.shape[:2]
        ind = random.randint(0, len(self._bg_img_paths) - 1)
        filename = self._bg_img_paths[ind]

        bg_img = self.get_bg_image(filename, H, W)

        if len(bg_img.shape) != 3:
            bg_img = np.zeros((H, W, 3), dtype=np.uint8)
            print("bad background image: {}".format(filename))

        mask = im_mask.copy().astype(np.bool_)

        mask_bg = ~mask
        im[mask_bg] = bg_img[mask_bg]
        im = im.astype(np.uint8)

        rets = [im]
        if return_mask:  # bool fg mask
            rets.append(mask)
        return tuple(rets) if len(rets) > 1 else rets[0]

    def get_bg_image(self, filename, imH, imW, channel=3):
        """keep aspect ratio of bg during resize target image size:

        imHximWxchannel.
        """
        target_size = min(imH, imW)
        max_size = max(imH, imW)
        real_hw_ratio = float(imH) / float(imW)
        bg_image = skimage.io.imread(filename)
        if len(bg_image.shape) == 2:
            bg_image = np.expand_dims(bg_image, 2).repeat(3, axis=2)
        bg_h, bg_w = bg_image.shape[:2]
        bg_image_resize = np.zeros((imH, imW, channel), dtype="uint8")

        if (float(imH) / float(imW) < 1 and float(bg_h) / float(bg_w) < 1) or (
            float(imH) / float(imW) >= 1 and float(bg_h) / float(bg_w) >= 1
        ):
            if bg_h >= bg_w:
                bg_h_new = int(np.ceil(bg_w * real_hw_ratio))
                if bg_h_new < bg_h:
                    bg_image_crop = bg_image[0:bg_h_new, 0:bg_w, :]
                else:
                    bg_image_crop = bg_image
            else:
                bg_w_new = int(np.ceil(bg_h / real_hw_ratio))
                if bg_w_new < bg_w:
                    bg_image_crop = bg_image[0:bg_h, 0:bg_w_new, :]
                else:
                    bg_image_crop = bg_image
        else:
            if bg_h >= bg_w:
                bg_h_new = int(np.ceil(bg_w * real_hw_ratio))
                bg_image_crop = bg_image[0:bg_h_new, 0:bg_w, :]
            else:  # bg_h < bg_w
                bg_w_new = int(np.ceil(bg_h / real_hw_ratio))
                # logger.info(bg_w_new)
                bg_image_crop = bg_image[0:bg_h, 0:bg_w_new, :]

        bg_image_resize_0 = resize_short_edge(bg_image_crop, target_size, max_size)
        h, w, c = bg_image_resize_0.shape
        bg_image_resize[0:h, 0:w, :] = bg_image_resize_0

        return bg_image_resize.squeeze()

    @lazy_property
    def _bg_img_paths(self):
        # random.choice(bg_img_paths)
        bg_type = self.bg_type
        bg_root = self.bg_imgs_root
        num_bg_imgs = self.num_bg_imgs
        hashed_file_name = hashlib.md5(
            ("{}_{}_{}_get_bg_imgs".format(bg_root, num_bg_imgs, bg_type)).encode(
                "utf-8"
            )
        ).hexdigest()
        cache_path = osp.join(
            ".cache/bg_paths_{}_{}.pkl".format(bg_type, hashed_file_name)
        )
        prepare_dir(osp.dirname(cache_path))
        if osp.exists(cache_path):
            # print("get bg_paths from cache file: {}".format(cache_path))
            with open(cache_path, "rb") as cache_path:
                bg_img_paths = pickle.load(cache_path)
            # print("num bg imgs: {}".format(len(bg_img_paths)))
            assert len(bg_img_paths) > 0
            return bg_img_paths

        print("building bg imgs cache {}...".format(bg_type))
        assert osp.exists(bg_root), f"BG ROOT: {bg_root} does not exist"
        if bg_type == "coco":
            img_paths = [
                osp.join(bg_root, fn.name)
                for fn in os.scandir(bg_root)
                if ".png" in fn.name or "jpg" in fn.name
            ]
        elif bg_type == "VOC_table":  # used in original deepim
            VOC_root = bg_root  # path to "VOCdevkit/VOC2012"
            VOC_image_set_dir = osp.join(VOC_root, "ImageSets/Main")
            VOC_bg_list_path = osp.join(VOC_image_set_dir, "diningtable_trainval.txt")
            with open(VOC_bg_list_path, "r") as f:
                VOC_bg_list = [
                    line.strip("\r\n").split()[0]
                    for line in f.readlines()
                    if line.strip("\r\n").split()[1] == "1"
                ]
            img_paths = [
                osp.join(VOC_root, "JPEGImages/{}.jpg".format(bg_idx))
                for bg_idx in VOC_bg_list
            ]
        elif bg_type == "VOC":
            VOC_root = bg_root  # path to "VOCdevkit/VOC2012"
            img_paths = [
                osp.join(VOC_root, "JPEGImages", fn.name)
                for fn in os.scandir(osp.join(bg_root, "JPEGImages"))
                if ".jpg" in fn.name
            ]
        elif bg_type == "SUN2012":
            img_paths = [
                osp.join(bg_root, "JPEGImages", fn.name)
                for fn in os.scandir(osp.join(bg_root, "JPEGImages"))
                if ".jpg" in fn.name
            ]
        else:
            raise ValueError(f"BG_TYPE: {bg_type} is not supported")
        assert len(img_paths) > 0, len(img_paths)

        num_bg_imgs = min(len(img_paths), num_bg_imgs)
        indices = [i for i in range(len(img_paths))]
        sel_indices = np.random.choice(indices, num_bg_imgs)
        bg_img_paths = [img_paths[idx] for idx in sel_indices]

        dump_with_lock(cache_path, bg_img_paths)
        print("num bg imgs: {}".format(len(bg_img_paths)))
        assert len(bg_img_paths) > 0
        return bg_img_paths


# python -m src.data.components.bg_augmentor
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

    augmentor = BGAugmentor("VOC", "./data/VOC2012/VOCdevkit/VOC2012", 10000)
    for i, batch in tqdm(enumerate(dataloader)):
        rgb = batch["rgb_patch"].permute(0, 2, 3, 1).cpu().numpy()
        mask = batch["vis_mask_patch"].cpu().numpy()

        bs = len(rgb)
        fig, axs = plt.subplots(bs, 3)
        for i in range(bs):
            axs[i, 0].imshow((rgb[i]))
            axs[i, 1].imshow(mask[i])
            rgb_aug = augmentor(rgb[i], mask[i])
            axs[i, 2].imshow(rgb_aug)

        plt.show()
