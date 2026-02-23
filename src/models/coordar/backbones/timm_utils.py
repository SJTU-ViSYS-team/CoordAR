import copy
import logging
import ipdb
import timm
import pathlib
from torchvision.models.feature_extraction import create_feature_extractor
import torch

_logger = logging.getLogger(__name__)


def my_create_timm_model(**init_args):
    if "return_nodes" in init_args:
        return_nodes = init_args.pop("return_nodes")
        model = create_feature_extractor(
            timm.create_model(**init_args),
            return_nodes=return_nodes,
        )
    else:
        model = timm.create_model(**init_args)
    return model


# python -m src.models.coordar.backbones.timm_utils
if __name__ == "__main__":
    # init_args = {
    #     "model_name": "timm/convnext_base",
    #     "pretrained": True,
    #     "in_chans": 6,  # will copy rgb weight to depth weights according to https://github.com/huggingface/pytorch-image-models/blob/5dce71010174ad6599653da4e8ba37fd5f9fa572/timm/models/_manipulate.py#L275
    #     "features_only": True,
    #     "out_indices": [3],
    # }

    init_args = {
        "model_name": "timm/vit_base_patch16_224",
        "pretrained": True,
        "in_chans": 6,
        "return_nodes": {
            "norm": "pre_logits",
        },
    }

    # test
    model = my_create_timm_model(**init_args)

    x = torch.randn(1, 6, 224, 224)
    features = model(x)  # 返回的是 List[Tensor]

    print("features:", features)
