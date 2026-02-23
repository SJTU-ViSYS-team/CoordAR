from einops import rearrange
import ipdb
import torch
import timm

from src.models.coordar.backbones.conditional_vae import ConditionalVAE
from src.models.coordar.backbones.dino_sdvae import DINO_SDVAE
from src.models.coordar.backbones.dinov2 import DINOv2
from src.models.coordar.backbones.sd_vae import VAETokenizer
from src.models.coordar.backbones.vae import VAE
from src.models.coordar.backbones.timm_utils import my_create_timm_model

BACKBONES = {
    "vae_tokenizer": VAETokenizer,
    "vae": VAE,
    "conditional_vae": ConditionalVAE,
    "dinov2": DINOv2,
    "dinovae": DINO_SDVAE,
}


class Backbone(torch.nn.Module):
    def __init__(self, init_args, bchw2blc=False, remove_cls=False):
        super(Backbone, self).__init__()
        self.init_args = init_args
        self.bchw2blc = bchw2blc

        backbone_type = init_args.pop("model_name")
        if "timm/" in backbone_type or "tv/" in backbone_type:
            init_args["model_name"] = backbone_type.split("/")[-1]
            self.backbone_impl = my_create_timm_model(**init_args)
        else:
            self.backbone_impl = BACKBONES[backbone_type](**init_args)

        self.remove_cls = remove_cls

    def forward(self, x, **kwargs):
        features = self.backbone_impl(x, **kwargs)
        if "return_nodes" in self.init_args:
            assert (
                len(self.init_args["return_nodes"]) == 1
            ), "Only one return node is supported"
            features = features[list(self.init_args["return_nodes"].values())[0]]
        elif "features_only" in self.init_args:
            assert len(features) == 1, "Only one feature is supported"
            features = features[0]
        if self.bchw2blc:
            # bchw -> blc
            features = rearrange(features, "b c h w -> b (h w) c")
        if self.remove_cls:
            # remove cls token
            features = features[:, 1:, :]
        return features


# python -m src.models.coordar.backbones.backbone
if __name__ == "__main__":
    # init_args = {
    #     "model_name": "timm/convnext_base",
    #     "pretrained": True,
    #     "in_chans": 6,  # will copy rgb weight to depth weights according to https://github.com/huggingface/pytorch-image-models/blob/5dce71010174ad6599653da4e8ba37fd5f9fa572/timm/models/_manipulate.py#L275
    #     "features_only": True,
    #     "out_indices": [3],
    # }

    # init_args = {
    #     "model_name": "timm/vit_base_patch16_224",
    #     "pretrained": True,
    #     "in_chans": 6,
    #     "return_nodes": {
    #         "norm": "pre_logits",
    #     },
    # }

    init_args = {
        "model_name": "dinov2",
        "name": "dinov2_vitb14",
        "target_size": (14, 14),
    }

    # test
    model = Backbone(init_args, bchw2blc=False)

    x = torch.randn(1, 3, 224, 224)
    features = model(x)  # 返回的是 List[Tensor]

    print("features:", features.shape)
