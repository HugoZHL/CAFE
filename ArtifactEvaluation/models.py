import numpy as np
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(
        self,
        embedding_layer,
        device,
        embedding_dim,
        num_sparse,
        num_dense,
        ln_bot=None,
        ln_top=None,
    ):
        super(BaseModel, self).__init__()

        self.embedding_layer = embedding_layer
        self.device = device

        self.create_model(embedding_dim, num_sparse, num_dense, ln_bot, ln_top)
        self.loss_fn = torch.nn.BCELoss(reduction="mean")

    def create_model(self, embedding_dim, num_sparse, num_dense, ln_bot, ln_top):
        raise NotImplementedError

    def create_mlp(self, ln, sigmoid_layer=-1):
        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]
            LL = nn.Linear(int(n), int(m), bias=True)
            nn.init.normal_(LL.weight, 0.0, np.sqrt(2 / (m + n)))
            nn.init.normal_(LL.bias, 0.0, np.sqrt(1 / m))
            layers.append(LL)
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())

        return torch.nn.Sequential(*layers)

    def apply_mlp(self, x, layers):
        if x == None:
            return None
        return layers(x)


class DLRM_Net(BaseModel):
    def create_model(self, embedding_dim, num_sparse, num_dense, ln_bot, ln_top):
        self.bot_mlp = self.create_mlp(ln_bot)
        self.top_mlp = self.create_mlp(ln_top, ln_top.size-2)

    def interact_features(self, x, feats):
        batch_size, dim = feats[0].shape
        if x is not None:
            feats = [x] + feats
        T = torch.cat(feats, dim=1).view((batch_size, -1, dim))
        Z = torch.bmm(T, torch.transpose(T, 1, 2))
        _, ni, nj = Z.shape
        triu_indices = torch.triu_indices(ni, nj, 1)
        R = Z[:, triu_indices[0], triu_indices[1]]
        if x is not None:
            R = torch.cat([x, R], dim=1)
        return R

    def forward(self, dense, offsets, indices):
        x = self.apply_mlp(dense, self.bot_mlp)
        feats = self.embedding_layer(offsets, indices)
        x = self.interact_features(x, feats)
        x = self.apply_mlp(x, self.top_mlp)
        return x


class WDL_Net(BaseModel):
    def create_model(self, embedding_dim, num_sparse, num_dense, ln_bot, ln_top):
        ln_top = np.array([embedding_dim * num_sparse + num_dense, 256, 256, 1])
        self.wide = nn.Linear(ln_top[0], 1)
        self.top_mlp = self.create_mlp(ln_top, ln_top.size-2)
        nn.init.normal_(self.wide.weight, mean=0, std=0.0001)

    def forward(self, dense, offsets, indices):
        feats = self.embedding_layer(offsets, indices)
        batch_size = indices.shape[0]
        feats = torch.cat(feats, dim = 1).view(batch_size, -1)
        if dense is not None:
            feats = torch.cat([dense, feats], dim = 1)
        deep_p = self.apply_mlp(feats, self.top_mlp)
        wide_p = self.wide(feats)
        return torch.sigmoid(deep_p + wide_p)


class DCN_Net(BaseModel):
    def create_model(self, embedding_dim, num_sparse, num_dense, ln_bot, ln_top):
        self.cross_layer_n = 3
        input_dim = embedding_dim * num_sparse + num_dense
        ln_top = np.array([input_dim, 256, 256, 256])
        self.cross_weight = [
            nn.Parameter(torch.normal(mean=0.0, std=0.0001,
            size=(input_dim, 1), device=self.device), requires_grad=True)
            for i in range(self.cross_layer_n)
        ]

        self.cross_biases = [
            nn.Parameter(torch.zeros(input_dim, device=self.device), requires_grad=True)
            for i in range(self.cross_layer_n)
        ]
        self.last_layer = nn.Linear(input_dim + 256, 1)
        self.top_mlp = self.create_mlp(ln_top, ln_top.size-2)

    def cross_layer(self, x0, x1, i):
        x1w = torch.mm(x1, self.cross_weight[i])
        y = x0 * x1w + self.cross_biases[i]
        return y

    def apply_cross(self, x0):
        x1 = x0.clone()
        for i in range(self.cross_layer_n):
            x1 = self.cross_layer(x0, x1, i)
        return x1

    def forward(self, dense, offsets, indices):
        feats = self.embedding_layer(offsets, indices)
        batch_size = indices.shape[0]
        feats = torch.cat(feats, dim = 1).view(batch_size, -1)
        if dense is not None:
            feats = torch.cat([dense, feats], dim = 1)
        deep_p = self.apply_mlp(feats, self.top_mlp)
        cross_p = self.apply_cross(feats)
        last_input = torch.cat([deep_p, cross_p], dim=1)
        last_output = self.last_layer(last_input)
        return torch.sigmoid(last_output)
