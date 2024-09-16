# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Quotient-Remainder Trick
#
# Description: Applies quotient remainder-trick to embeddings to reduce
# embedding sizes.
#
# References:
# [1] Hao-Jun Michael Shi, Dheevatsa Mudigere, Maxim Naumov, Jiyan Yang,
# "Compositional Embeddings Using Complementary Partitions for Memory-Efficient
# Recommendation Systems", CoRR, arXiv:1909.02107, 2019


from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class QREmbeddingBag(nn.Module):
    def __init__(
        self,
        num_categories,
        embedding_dim,
        num_collisions,
    ):
        super(QREmbeddingBag, self).__init__()

        self.num_categories = num_categories
        if isinstance(embedding_dim, int) or len(embedding_dim) == 1:
            self.embedding_dim = [embedding_dim, embedding_dim]
        else:
            self.embedding_dim = embedding_dim
        self.num_collisions = num_collisions

        self.num_embeddings = [
            int(np.ceil(num_categories / num_collisions)) + 1,
            num_collisions,
        ]

        self.weight_q = Parameter(
            torch.Tensor(self.num_embeddings[0], self.embedding_dim[0])
        )
        self.weight_r = Parameter(
            torch.Tensor(self.num_embeddings[1], self.embedding_dim[1])
        )
        self.reset_parameters()

    def reset_parameters(self):
        scale = np.sqrt(1 / self.num_categories)
        nn.init.uniform_(self.weight_q, -scale, scale)
        nn.init.uniform_(self.weight_r, -scale, scale)

    def forward(self, input, offsets=None):
        input_q = (input / self.num_collisions).long()
        input_r = torch.remainder(input, self.num_collisions).long()

        embed_q = F.embedding_bag(
            input_q,
            self.weight_q,
            offsets,
            sparse=True,
        )
        embed_r = F.embedding_bag(
            input_r,
            self.weight_r,
            offsets,
            sparse=True,
        )

        return embed_q + embed_r
