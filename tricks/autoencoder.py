from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, base_dim):
        super(AutoEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.base_dim = base_dim
        print(f"embedding_dim: {embedding_dim}, base_dim: {base_dim}")
        self.embs = nn.EmbeddingBag(
            num_embeddings, embedding_dim, mode="sum", sparse=True
        )
        torch.nn.init.xavier_uniform_(self.embs.weight)
        if embedding_dim < base_dim:
            self.proj = nn.Linear(embedding_dim, base_dim)
            torch.nn.init.xavier_uniform_(self.proj.weight)
        elif embedding_dim == base_dim:
            self.proj = nn.Identity()
        else:
            raise ValueError(
                "Embedding dim " + str(embedding_dim) + " > base dim " + str(base_dim)
            )
        self.fc1 = nn.Linear(base_dim, embedding_dim)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(embedding_dim, num_embeddings)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def get_emb(self, input, offsets=None, per_sample_weights=None):
        with torch.no_grad():
            emb = self.proj(
                self.embs(input, offsets=offsets, per_sample_weights=per_sample_weights)
            )
        return emb
    def forward(self, input, offsets=None, per_sample_weights=None):
        return self.fc2(self.fc1(self.proj(
            self.embs(input, offsets=offsets, per_sample_weights=per_sample_weights)
        )))
