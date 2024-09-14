import numpy as np
import random
import math
import ctypes
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from tricks.qr_embedding_bag import QREmbeddingBag
from tricks.sk_embedding_bag import SKEmbeddingBag
from tricks.md_embedding_bag import PrEmbeddingBag
from tricks.adaembed import adaEmbeddingBag

class BaseModel(nn.Module):
    def __init__(
        self,
        compress_rate,
        ada_flag,
        lib,
        hotn,
        device,
        sketch_flag,
        hash_rate,
        sketch_threshold,
        embedding_dim,
        embedding_nums,
        ln_bot=None,
        ln_top=None,
        qr_flag=False,
        qr_operation="mult",
        qr_collisions=0,
        qr_threshold=200,
        md_flag=False,
        md_threshold=200,
    ):
        super(BaseModel, self).__init__()

        # save arguments
        self.embedding_nums = embedding_nums
        self.compress_rate = compress_rate
        self.ada_flag = ada_flag
        if self.ada_flag:
            self.dic = []
            self.weight = None
            self.hot_rate = 0
            self.hotn = 0
            self.d_time = 0
        self.ada_emb = np.zeros(len(embedding_nums))
        self.lib = lib
        self.output_d = 0
        self.parallel_model_batch_size = -1
        self.parallel_model_is_not_prepared = True
        self.device = device
        # create variables for QR embedding if applicable
        self.qr_flag = qr_flag
        if self.qr_flag:
            self.qr_collisions = qr_collisions
            self.qr_operation = qr_operation
            self.qr_threshold = qr_threshold
        # create variables for MD embedding if applicable
        self.md_flag = md_flag
        if self.md_flag:
            self.md_threshold = md_threshold
        self.sketch_flag = sketch_flag
        self.hash_rate = hash_rate
        self.hotn = hotn
        self.grad_norm = []
        self.f_offset = np.zeros(embedding_nums.size, dtype=np.int32)
        if self.sketch_flag:
            self.weight_high = Parameter(
                torch.Tensor(hotn, embedding_dim),
                requires_grad=True,
            )
            self.register_buffer('sketch_buffer', torch.zeros(hotn * 12, dtype = torch.int32, device = 'cpu'))
            init = lib.init
            init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
            init.restype = None
            numpy_array = self.sketch_buffer.numpy()
            data_ptr = numpy_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            init(hotn, sketch_threshold, data_ptr)

        self.sketch_emb = np.zeros(self.embedding_nums.size)
        # nn.init.uniform_(self.weight_high, np.sqrt(1 / 10000000))
        # If running distributed, get local slice of embedding tables

        # create operators
        self.emb_l = self.create_emb(embedding_dim, embedding_nums)

        # create model
        self.create_model(embedding_dim, embedding_nums, ln_bot, ln_top)

        # specify the loss function
        self.loss_fn = torch.nn.BCELoss(reduction="mean")

    def create_model(self, embedding_dim, embedding_nums, ln_bot, ln_top):
        raise NotImplementedError

    def create_mlp(self, ln, sigmoid_layer=-1):
        # build MLP layer by layer
        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]

            # construct fully connected operator
            LL = nn.Linear(int(n), int(m), bias=True)

            # initialize the weights
            # with torch.no_grad():
            # custom Xavier input, output or two-sided fill
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            # approach 1
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            # approach 2
            # LL.weight.data.copy_(torch.tensor(W))
            # LL.bias.data.copy_(torch.tensor(bt))
            # approach 3
            # LL.weight = Parameter(torch.tensor(W),requires_grad=True)
            # LL.bias = Parameter(torch.tensor(bt),requires_grad=True)
            layers.append(LL)

            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())

        # approach 1: use ModuleList
        # return layers
        # approach 2: use Sequential container to wrap all layers
        return torch.nn.Sequential(*layers)

    def create_emb(self, m, ln):
        emb_l = nn.ModuleList()
        print(ln)
        tot_nums = 0
        N = 0
        if self.ada_flag:
            for i in range(0, ln.size):
                if ln[i] > 2000 * self.compress_rate:
                    N += ln[i]
            self.dic = np.zeros(N, dtype=np.int32)
            self.hotn = int((N * m * self.compress_rate - N * 2) / m)
            self.weight = Parameter(
                torch.Tensor(self.hotn + 1, m),
                requires_grad=True,
            )
            self.hot_rate = self.hotn / N
            self.grad_norm = np.zeros(N, dtype=np.float64)
            self.tmp_grad_norm = np.zeros(N, dtype=np.float64)
            print(f"hotn: {self.hotn}, hot_rate: {self.hot_rate}")
        N = 0
        for i in range(0, ln.size):
            n = ln[i]

            # construct embedding operator
            if self.qr_flag and n > self.qr_threshold:
                EE = QREmbeddingBag(
                    n,
                    m,
                    self.qr_collisions,
                    operation=self.qr_operation,
                    mode="sum",
                    sparse=True,
                )
            elif self.md_flag and n > self.md_threshold:
                base = max(m)
                _m = m[i] if n > self.md_threshold else base
                EE = PrEmbeddingBag(n, _m, base)
                # use np initialization as below for consistency...
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, _m)
                ).astype(np.float32)
                EE.embs.weight.data = torch.tensor(W, requires_grad=True)
            elif self.ada_flag and n > 2000 * self.compress_rate:
                EE = adaEmbeddingBag(
                    N,
                    self.weight,
                    self.dic,
                    self.compress_rate,
                    self.device,
                    n,
                    m,
                )
                self.ada_emb[i] = True
                self.f_offset[i] = N
                N += n
            elif self.sketch_flag and n > 2000 * self.compress_rate:
                EE = SKEmbeddingBag(
                    N,
                    self.hotn,
                    self.lib,
                    self.weight_high,
                    self.device,
                    n,
                    m,
                    tot_nums,
                    round(self.hash_rate * n + 0.51),
                )
                self.sketch_emb[i] = True
                N += 1
            else:
                if self.md_flag:
                    EE = nn.EmbeddingBag(n, max(m), mode="sum", sparse=True)
                else:
                    EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
                # initialize embeddings
                # nn.init.uniform_(EE.weight, a=-np.sqrt(1 / n), b=np.sqrt(1 / n))
                tmp_n = max(n, 5)

                if self.md_flag:
                    W = np.random.uniform(
                        low=-np.sqrt(1 / tmp_n), high=np.sqrt(1 / tmp_n), size=(n, max(m))
                    ).astype(np.float32)
                else:
                    W = np.random.uniform(
                        low=-np.sqrt(1 / tmp_n), high=np.sqrt(1 / tmp_n), size=(n, m)
                    ).astype(np.float32)
                # approach 1
                EE.weight.data = torch.tensor(W, requires_grad=True)
                # approach 2
                # EE.weight.data.copy_(torch.tensor(W))
                # approach 3
                # EE.weight = Parameter(torch.tensor(W),requires_grad=True)
            emb_l.append(EE)
            if self.ada_flag == False:
                self.f_offset[i] = tot_nums
            tot_nums += n
        return emb_l

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        def compute_should_use_set_data(tensor, tensor_applied):
            if torch._has_compatible_shallow_copy_type(tensor, tensor_applied):
                # If the new tensor has compatible tensor type as the existing tensor,
                # the current behavior is to change the tensor in-place using `.data =`,
                # and the future behavior is to overwrite the existing tensor. However,
                # changing the current behavior is a BC-breaking change, and we want it
                # to happen in future releases. So for now we introduce the
                # `torch.__future__.get_overwrite_module_params_on_conversion()`
                # global flag to let the user control whether they want the future
                # behavior of overwriting the existing tensor or not.
                return not torch.__future__.get_overwrite_module_params_on_conversion()
            else:
                return False

        for key, param in self._parameters.items():
            if param is None:
                continue
            # Tensors stored in modules are graph leaves, and we don't want to
            # track autograd history of `param_applied`, so we have to use
            # `with torch.no_grad():`
            with torch.no_grad():
                param_applied = fn(param)
            should_use_set_data = compute_should_use_set_data(param, param_applied)
            if should_use_set_data:
                param.data = param_applied
                out_param = param
            else:
                assert isinstance(param, Parameter)
                assert param.is_leaf
                out_param = Parameter(param_applied, param.requires_grad)
                self._parameters[key] = out_param

            if param.grad is not None:
                with torch.no_grad():
                    grad_applied = fn(param.grad)
                should_use_set_data = compute_should_use_set_data(param.grad, grad_applied)
                if should_use_set_data:
                    assert out_param.grad is not None
                    out_param.grad.data = grad_applied
                else:
                    assert param.grad.is_leaf
                    out_param.grad = grad_applied.requires_grad_(param.grad.requires_grad)

        for key, buf in self._buffers.items():
            if buf is not None and key != "sketch_buffer":
                self._buffers[key] = fn(buf)

        return self

    def apply_mlp(self, x, layers):
        # approach 1: use ModuleList
        # for layer in layers:
        #     x = layer(x)
        # return x
        # approach 2: use Sequential container to wrap all layers
        if x == None:
            return None
        return layers(x)

    def apply_emb(self, lS_o, lS_i, emb_l, test):
        # WARNING: notice that we are processing the batch at once. We implicitly
        # assume that the data is laid out such that:
        # 1. each embedding is indexed with a group of sparse indices,
        #   corresponding to a single lookup
        # 2. for each embedding the lookups are further organized into a batch
        # 3. for a list of embedding tables there is a list of batched lookups

        ly = []
        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = lS_o[k]

            # embedding lookup
            # We are using EmbeddingBag, which implicitly uses sum operator.
            # The embeddings are represented as tall matrices, with sum
            # happening vertically across 0 axis, resulting in a row vector
            # E = emb_l[k]

            E = emb_l[k]
            if (self.sketch_emb[k] == True):
                V = E(
                    sparse_index_group_batch,
                    sparse_offset_group_batch,
                    test=test,
                )
            else:

                V = E(
                    sparse_index_group_batch.to(self.device),
                    sparse_offset_group_batch.to(self.device),
                )

            ly.append(V)

        # print(ly)
        return ly

    def ada_decay(self):
        self.grad_norm *= 0.8

    def ada_rebuild(self):
        new_cnt = self.grad_norm[:]
        for i in range(self.embedding_nums.size):
            if self.ada_emb[i]:
                l = self.f_offset[i]
                r = self.f_offset[i] + self.embedding_nums[i]
                p = np.percentile(self.grad_norm[l: r], 95)
                if (p != 0):
                    new_cnt[l: r] /= p
        ind_1 = set(np.argsort(-new_cnt)[:self.hotn])  # all hot features
        ind_2 = set(np.where(self.dic != 0)[0])  # old hot features
        admit_ = np.array(list(ind_1 - ind_2))
        evict_ = np.array(list(ind_2 - ind_1))
        if len(admit_) != len(evict_):
            if (len(evict_) != 0):
                print("error")
                exit(0)
            self.dic[admit_] = np.arange(1, self.hotn + 1)
        else:
            self.dic[admit_] = self.dic[evict_]
            with torch.no_grad():
                self.weight[self.dic[admit_]] = 0
            self.dic[evict_] = 0

    def ada_check(self):
        N = 1000000
        sample = random.sample(range(0, len(self.grad_norm)), N)
        dic = self.dic[sample]
        cnt = self.grad_norm[sample]
        m = math.ceil(N * self.hot_rate)

        ind_1 = set(np.argsort(-cnt)[:m])  # all hot features
        ind_2 = set(np.where(dic != 0)[0])  # old hot features
        admit_ = np.array(list(ind_1 - ind_2))
        evict_ = np.array(list(ind_2 - ind_1))
        lim = cnt[np.argsort(-cnt)[m]]  # all hot features
        if (len(admit_) > m * 0.05):
            self.ada_rebuild()

    def insert_adagrad(self, lS_o):
        tmp = 0
        N = len(lS_o[0])
        for k, input in enumerate(lS_o):
            if self.ada_emb[k] == True:
                grad_norm = np.array(torch.norm(self.weight.grad._values()[
                                     tmp: tmp+N], dim=1, p=2).cpu())
                np.add.at(self.grad_norm, input +
                          self.f_offset[k], grad_norm / np.sum(grad_norm) * N)
                tmp += N
        self.d_time += 1
        if (self.d_time == 1 or self.d_time % 4096 == 0):
            self.ada_check()
        if (self.d_time % 16384 == 0):
            self.ada_decay()

    def insert_grad(self, lS_o):
        for k, input in enumerate(lS_o):
            if self.sketch_emb[k] == True:
                self.emb_l[k].insert_grad(input)


class DLRM_Net(BaseModel):
    def create_model(self, embedding_dim, embedding_nums, ln_bot, ln_top):
        self.bot_l = self.create_mlp(ln_bot)
        self.top_l = self.create_mlp(ln_top, ln_top.size-2)

    def interact_features(self, x, ly):
        # concatenate dense and sparse features
        (batch_size, d) = ly[0].shape
        if x == None:
            T = torch.cat(ly, dim=1).view((batch_size, -1, d))
        else:
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
        # perform a dot product
        Z = torch.bmm(T, torch.transpose(T, 1, 2))
        # append dense feature with the interactions (into a row vector)
        # approach 1: all
        # Zflat = Z.view((batch_size, -1))
        # approach 2: unique
        _, ni, nj = Z.shape
        # approach 1: tril_indices
        # li, lj = torch.tril_indices(ni, nj, offset=offset)
        # approach 2: custom
        offset = 0
        li = torch.tensor([i for i in range(ni)
                            for j in range(i + offset)])
        lj = torch.tensor([j for i in range(nj)
                            for j in range(i + offset)])
        # print(f"ni, nj: {ni, nj}, li, lj: {li, lj}")
        Zflat = Z[:, li, lj]
        # concatenate dense features and interactions
        if x != None:
            R = torch.cat([x] + [Zflat], dim=1)
        else:
            R = torch.cat([Zflat], dim=1)

        return R

    def forward(self, dense_x, lS_o, lS_i, test):
        # process dense features (using bottom mlp), resulting in a row vector
        x = self.apply_mlp(dense_x, self.bot_l)
        # debug prints
        # print("intermediate")
        # print(x.detach().cpu().numpy())

        # process sparse features(using embeddings), resulting in a list of row vectors
        ly = self.apply_emb(lS_o, lS_i, self.emb_l, test)
        # for y in ly:
        #     print(y.detach().cpu().numpy())

        # interact features (dense and sparse)
        z = self.interact_features(x, ly)
        # print(z.detach().cpu().numpy())

        # obtain probability of a click (using top mlp)
        p = self.apply_mlp(z, self.top_l)

        z = p

        return z


class WDL_Net(BaseModel):
    def create_model(self, embedding_dim, embedding_nums, ln_bot, ln_top):
        m_den = 13
        ln_top = np.array([embedding_dim * len(embedding_nums) + m_den,256,256,1])
        self.wide = nn.Linear(ln_top[0], 1)
        self.top_l = self.create_mlp(ln_top, ln_top.size-2)
        nn.init.normal_(self.wide.weight, mean=0, std=0.0001)

    def forward(self, dense_x, lS_o, lS_i, test):
        # process dense features (using bottom mlp), resulting in a row vector

        # process sparse features(using embeddings), resulting in a list of row vectors
        ly = self.apply_emb(lS_o, lS_i, self.emb_l, test)

        # interact features (dense and sparse)
        batch_size, d = dense_x.shape
        ly = torch.cat(ly, dim = 1).view(batch_size, -1)

        z = torch.cat([dense_x, ly], dim = 1)
        # print(z.shape)
        # print(z.detach().cpu().numpy())

        # obtain probability of a click (using top mlp)
        deep_p = self.apply_mlp(z, self.top_l)
        wide_p = self.wide(z)
        return torch.sigmoid(deep_p + wide_p)


class DCN_Net(BaseModel):
    def create_model(self, embedding_dim, embedding_nums, ln_bot, ln_top):
        self.cross_layer_n = 3
        m_den = 13
        input_dim = embedding_dim * len(embedding_nums) + m_den
        ln_top = np.array([input_dim,256,256,256])
        self.cross_weight = [nn.Parameter(torch.normal(mean=0.0, std=0.0001,
                                requires_grad=True, size=(input_dim, 1), device=self.device))
                                for i in range(self.cross_layer_n)]

        self.cross_biases = [nn.Parameter(torch.zeros(input_dim, device=self.device))
                                for i in range(self.cross_layer_n)]
        self.last_layer = nn.Linear(input_dim+256, 1)
        self.top_l = self.create_mlp(ln_top, ln_top.size-2)

    def cross_layer(self, x0, x1, i):
        x1w = torch.mm(x1, self.cross_weight[i])
        y = x0 * x1w + self.cross_biases[i]
        return y

    def apply_cross(self, x0):
        x1 = x0.clone().to(x0.device)
        for i in range(self.cross_layer_n):
            x1 = self.cross_layer(x0, x1, i)
        return x1

    def forward(self, dense_x, lS_o, lS_i, test):
        # process dense features (using bottom mlp), resulting in a row vector

        # process sparse features(using embeddings), resulting in a list of row vectors
        ly = self.apply_emb(lS_o, lS_i, self.emb_l, test)

        # interact features (dense and sparse)
        batch_size, d = dense_x.shape
        ly = torch.cat(ly, dim = 1).view(batch_size, -1)

        z = torch.cat([dense_x, ly], dim = 1)
        # print(z.shape)
        # print(z.detach().cpu().numpy())

        # obtain probability of a click (using top mlp)
        deep_p = self.apply_mlp(z, self.top_l)
        cross_p = self.apply_cross(z)
        last_input = torch.cat([deep_p, cross_p], dim=1)
        last_output = self.last_layer(last_input)
        return torch.sigmoid(last_output)

