import torch.nn as nn
import math


class HashEmbeddingBag(nn.EmbeddingBag):
    def __init__(
        self,
        embedding_num,
        embedding_dim,
        compress_rate,
        mode="mean",
        sparse=False,
    ):
        self.real_n = int(math.ceil(embedding_num * compress_rate))
        super(HashEmbeddingBag, self).__init__(self.real_n, embedding_dim, mode=mode, sparse=sparse)

    def forward(self, input, offsets=None):
        input = input % self.real_n
        return super().forward(input, offsets)
