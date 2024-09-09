from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn


class OffEmbeddingBag(nn.Module):
    def __init__(
        self,
        num_hot,
        num_cold,
        embedding_dim,
        hot_dict,
        device,
    ):
        super(OffEmbeddingBag, self).__init__()
        self.num_hot = num_hot
        self.num_cold = num_cold
        if num_hot > 0:
            self.weight_hot = nn.EmbeddingBag(
                num_hot, embedding_dim, mode="sum", sparse=True
            )
        else:
            self.weight_hot = None
        if num_cold > 0:
            self.weight_cold = nn.EmbeddingBag(
                num_cold, embedding_dim, mode="sum", sparse=True
            )
        else:
            self.weight_cold = None
        self.hot_dict = torch.from_numpy(hot_dict).to(device)

    def forward(self, input, offsets=None):
        new_indices = self.hot_dict[input]
        if self.weight_hot is not None:
            embed_hot = self.weight_hot(new_indices % self.num_hot, offsets)
        else:
            embed_hot = None
        if self.weight_cold is not None:
            embed_cold = self.weight_cold(input % self.num_cold, offsets)
        else:
            embed_cold = None
        if embed_hot is None:
            embed = embed_cold
        elif embed_cold is None:
            embed = embed_hot
        else:
            hot_entries = (new_indices >= 0).unsqueeze(-1)
            embed = torch.where(hot_entries, embed_hot, embed_cold)
        return embed
