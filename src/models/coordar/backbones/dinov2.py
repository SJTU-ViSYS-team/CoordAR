import ipdb
import torch


class DINOv2(torch.nn.Module):
    def __init__(
        self,
        name="dinov2_vitl14",
        patch_size=14,
        target_size=(16, 16),  # w, h
    ):
        super(DINOv2, self).__init__()
        self.dinov2 = torch.hub.load("facebookresearch/dinov2", name)
        self.patch_size = patch_size
        self.target_size = (int(target_size[0]), int(target_size[1]))

    def forward(self, x):
        h, w = x.shape[-2:]
        features_dict = self.dinov2.forward_features(x)
        features = features_dict["x_norm_patchtokens"]
        features = features.permute(0, 2, 1).reshape(
            features.shape[0], -1, h // self.patch_size, w // self.patch_size
        )
        if self.target_size != (16, 16):
            features = torch.nn.functional.interpolate(
                features,
                size=self.target_size,
                mode="bilinear",
                align_corners=False,
            )
        return features
