from einops import rearrange
import ipdb
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm


try:
    from mmcv.cnn import normal_init, constant_init
except ImportError:
    from mmengine.model import normal_init, constant_init

from src.models.layers.transformer import TransformerDecoderLayer
from src.utils.torch_utils.layers.conv_module import ConvModule, _get_deconv_pad_outpad
from src.utils.torch_utils.layers.layer_utils import get_nn_act_func, get_norm
from src.utils.torch_utils.layers.std_conv_transpose import StdConvTranspose2d


class MemReading(nn.Module):

    def __init__(self, in_dim, feat_dim, num_heads=8):
        super().__init__()
        self.in_dim = in_dim
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.cross_attn = TransformerDecoderLayer(feat_dim, self.num_heads)
        if in_dim != feat_dim:
            self.in_proj = nn.Linear(in_dim, feat_dim)
            self.out_proj = nn.Linear(feat_dim, in_dim)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, query, keys, values):
        h, w = query.shape[2], query.shape[3]
        query = rearrange(query, "b c h w -> b (h w) c")
        if self.in_dim != self.feat_dim:
            readout = self.cross_attn(self.in_proj(query), keys, values)
            readout = self.out_proj(readout)
        else:
            readout = self.cross_attn(query, keys, values)
        query = self.norm(query + readout)
        query = rearrange(query, "b (h w) c -> b c h w", h=h)
        return query


class XYZMaskReg(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        up_types=("deconv", "bilinear", "bilinear"),
        deconv_kernel_size=3,
        num_conv_per_block=2,
        feat_dim=256,
        feat_kernel_size=3,
        use_ws=False,
        use_ws_deconv=False,
        norm="GN",
        num_gn_groups=32,
        act="GELU",
        out_kernel_size=1,
        mask_out_dim=2,
        xyz_out_dim=3,
        enable_cross_attn=False,
        num_heads=8,
        memory_dim=768,
    ):
        """
        Args:
            up_types: use up-conv or deconv for each up-sampling layer
                ("bilinear", "bilinear", "bilinear")
                ("deconv", "bilinear", "bilinear")  # CDPNv2 rot head
                ("deconv", "deconv", "deconv")  # CDPNv1 rot head
                ("nearest", "nearest", "nearest")  # implement here but maybe won't use
        NOTE: default from stride 32 to stride 4 (3 ups)
        """
        super().__init__()
        assert out_kernel_size in [
            1,
            3,
        ], "Only support output kernel size: 1 and 3"
        assert deconv_kernel_size in [
            1,
            3,
            4,
        ], "Only support deconv kernel size: 1, 3, and 4"
        assert len(up_types) > 0, up_types
        self.enable_cross_attn = enable_cross_attn
        self.num_heads = num_heads

        self.features = nn.ModuleList()
        for i, up_type in enumerate(up_types):
            _in_dim = in_dim if i == 0 else feat_dim
            if self.enable_cross_attn:
                self.features.append(
                    MemReading(_in_dim, memory_dim, num_heads=num_heads)
                )
            if up_type == "deconv":
                (
                    deconv_kernel,
                    deconv_pad,
                    deconv_out_pad,
                ) = _get_deconv_pad_outpad(deconv_kernel_size)
                deconv_layer = (
                    StdConvTranspose2d if use_ws_deconv else nn.ConvTranspose2d
                )
                self.features.append(
                    deconv_layer(
                        _in_dim,
                        feat_dim,
                        kernel_size=deconv_kernel,
                        stride=2,
                        padding=deconv_pad,
                        output_padding=deconv_out_pad,
                        bias=False,
                    )
                )
                self.features.append(
                    get_norm(norm, feat_dim, num_gn_groups=num_gn_groups)
                )
                self.features.append(get_nn_act_func(act))
            elif up_type == "bilinear":
                self.features.append(nn.UpsamplingBilinear2d(scale_factor=2))
            elif up_type == "nearest":
                self.features.append(nn.UpsamplingNearest2d(scale_factor=2))
            else:
                raise ValueError(f"Unknown up_type: {up_type}")

            if up_type in ["bilinear", "nearest"]:
                assert num_conv_per_block >= 1, num_conv_per_block
            for i_conv in range(num_conv_per_block):
                if i == 0 and i_conv == 0 and up_type in ["bilinear", "nearest"]:
                    conv_in_dim = in_dim
                else:
                    conv_in_dim = feat_dim

                if use_ws:
                    conv_cfg = dict(type="StdConv2d")
                else:
                    conv_cfg = None

                self.features.append(
                    ConvModule(
                        conv_in_dim,
                        feat_dim,
                        kernel_size=feat_kernel_size,
                        padding=(feat_kernel_size - 1) // 2,
                        conv_cfg=conv_cfg,
                        norm=norm,
                        num_gn_groups=num_gn_groups,
                        act=act,
                    )
                )

        self.mask_out_dim = mask_out_dim
        self.xyz_out_dim = xyz_out_dim
        out_dim = self.mask_out_dim + self.xyz_out_dim
        self.out_layer = nn.Conv2d(
            feat_dim,
            out_dim,
            kernel_size=out_kernel_size,
            padding=(out_kernel_size - 1) // 2,
            bias=True,
        )

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
        # init output layers
        normal_init(self.out_layer, std=0.01)

    def forward(self, x, memory=None):
        for i, l in enumerate(self.features):
            if isinstance(l, MemReading):
                if memory is None:
                    continue
                x = l(x, memory["keys"], memory["values"])
            else:
                x = l(x)
        x = self.out_layer(x)
        return x


# python -m src.models.coordar.heads.xyz_reg
if __name__ == "__main__":
    x = torch.randn(1, 128, 7, 7)
    memory = {
        "keys": torch.randn(1, 100, 768),
        "values": torch.randn(1, 100, 768),
    }
    model = XYZMaskReg(
        in_dim=128,
        up_types=("deconv", "bilinear", "bilinear"),
        enable_cross_attn=True,
    )
    out = model(x, memory)
    print(out.shape)
