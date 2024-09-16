# References
# AdaEmbed: Adaptive Embedding for Large-Scale Recommendation Models
# Fan Lai, et al. OSDI, 2023.


from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaEmbeddingBag(nn.Module):
    def __init__(
        self,
        offset,
        weight,
        dic,
        device,
    ):
        super(AdaEmbeddingBag, self).__init__()
        self.offset = offset
        self.device = device
        self.weight = weight
        self.dic = dic

    def forward(self, input, offsets=None):
        with torch.no_grad():
            self.weight[0] = 0
        input = input + self.offset
        dic = self.dic[input]
        dic = torch.tensor(dic).to(self.device)
        embed = F.embedding_bag(
            dic,
            self.weight,
            offsets.to(self.device),
            sparse=True,
        )

        return embed
