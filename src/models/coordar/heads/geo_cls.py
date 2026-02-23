import ipdb
import torch
import torch.nn as nn

from src.models.tokenizer.model import Tokenizer


class GeoClsHead(nn.Module):
    def __init__(self, in_dim, tokenizer: Tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

        self.in_dim = in_dim

        self.out = nn.Conv2d(self.in_dim, tokenizer.num_vq_embeddings, 1)
        self.loss = nn.CrossEntropyLoss()

        # freeze tokenizer components
        for param in self.tokenizer.parameters():
            param.requires_grad = False

    def forward(self, x, gt_geo=None):
        digit = self.out(x)
        indices = torch.argmax(digit, dim=1)
        geo = self.tokenizer.decode_indices(indices)
        geo = geo + 0.5
        out_dict = {}
        if gt_geo is not None:
            gt_indices = self.tokenizer.encode(gt_geo - 0.5)[2]
            geo_loss = self.loss(digit, gt_indices)
            out_dict["geo_loss"] = geo_loss
        out_dict["digit"] = digit
        out_dict["geo"] = geo
        return out_dict
