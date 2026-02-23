import re
import ipdb
import torch


class CkptLoader:
    def __init__(
        self,
        base_ckpt: str = "",
        base_prefix: str = "",
        target_prefix: str = "",
        exclude_layers=[],
        base_strict=True,
    ):
        super().__init__()

        self.base_ckpt = base_ckpt
        self.base_prefix = base_prefix
        self.target_prefix = target_prefix
        self.exclude_layers = exclude_layers
        self.base_strict = base_strict

    def load_from_ckpt(self, model):
        if self.base_ckpt:
            device = "cpu"
            # if torch.distributed.is_initialized():
            #     device = "cuda:{}".format(torch.distributed.get_rank())
            ckpt_weights = torch.load(
                self.base_ckpt, map_location=device, weights_only=False
            )
            # ema_state_dict
            if "ema_state_dict" in ckpt_weights:
                ckpt_weights = ckpt_weights["ema_state_dict"]
            elif "state_dict" in ckpt_weights:
                ckpt_weights = ckpt_weights["state_dict"]
            elif "model" in ckpt_weights:
                ckpt_weights = ckpt_weights["model"]

            tgt_ckpt_weights = {}
            # remove prefix
            for layername in ckpt_weights.keys():
                if self.base_prefix == "":
                    inner_name = self.target_prefix + layername
                else:
                    inner_name = layername.replace(self.base_prefix, self.target_prefix)
                # skip loading layer in ckpt
                exclude = any(
                    re.match(pattern, inner_name) for pattern in self.exclude_layers
                )
                if not exclude:
                    tgt_ckpt_weights[inner_name] = ckpt_weights[layername]
                else:
                    print(f"resuming has excluded {layername}")
            model.load_state_dict(tgt_ckpt_weights, strict=self.base_strict)
            print(f"weights loaded from {self.base_ckpt}")
