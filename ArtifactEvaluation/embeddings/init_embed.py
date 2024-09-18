import os
import os.path as osp
import ctypes
import numpy as np
import math
import random
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from .hash_embedding_bag import HashEmbeddingBag
from .qr_embedding_bag import QREmbeddingBag
from .sk_embedding_bag import SKEmbeddingBag
from .md_embedding_bag import PrEmbeddingBag, md_solver
from .ada_embedding_bag import AdaEmbeddingBag


class EmbeddingLayer(nn.Module):
    def __init__(
        self,
        args,
        embedding_dim,
        embedding_nums,
        device,
    ):
        super(EmbeddingLayer, self).__init__()
        self.device = device
        self.compress_method = args.compress_method
        compress_rate = args.compress_rate
        # here we use adaptive threshold w.r.t. compress rate
        compress_threshold = 400 / (1 - np.log2(compress_rate))
        self.embedding_nums = embedding_nums

        embs = nn.ModuleList()
        tot_nums = 0
        N = 0
        self.sketch_emb = np.zeros(self.embedding_nums.size)
        self.ada_emb = np.zeros(self.embedding_nums.size)
        if self.compress_method == 'mde':
            md_round_dims = args.md_round_dims

            l = 0.0001
            r = 0.5
            while r - l > 0.0001:
                mid = (l + r) / 2
                embedding_dim_ = md_solver(
                    torch.tensor(embedding_nums),
                    mid,  # alpha
                    d0=embedding_dim,
                    round_dim=md_round_dims,
                ).tolist()
                cr = sum(embedding_dim_ * embedding_nums) / (np.sum(embedding_nums) * embedding_dim)
                if cr > compress_rate:
                    l = mid
                else:
                    r = mid
            new_embedding_dim = md_solver(
                torch.tensor(embedding_nums),
                r,  # alpha
                d0=embedding_dim,
                round_dim=md_round_dims,
            ).tolist()
        elif self.compress_method == 'qr':
            ntotal = sum(embedding_nums)
            tobe_sqrt = (ntotal * compress_rate) ** 2 - 4 * ntotal
            if tobe_sqrt < 0:
                raise AssertionError(f'Q-R trick cannot support compress rate: {compress_rate}')
            qr_collisions = int(math.ceil((ntotal * compress_rate - math.sqrt(tobe_sqrt)) / 2))
        elif self.compress_method == 'ada':
            self.dic = []
            self.weight = None
            self.hot_rate = 0
            self.hotn = 0
            self.d_time = 0
            self.grad_norm = []
            self.f_offset = np.zeros(embedding_nums.size, dtype=np.int32)

            for i in range(0, embedding_nums.size):
                if embedding_nums[i] > compress_threshold:
                    N += embedding_nums[i]
            self.dic = np.zeros(N, dtype=np.int32)
            self.hotn = int((N * embedding_dim * compress_rate - N * 2) / embedding_dim)
            self.weight = Parameter(
                torch.Tensor(self.hotn + 1, embedding_dim),
                requires_grad=True,
            )
            self.hot_rate = self.hotn / N
            self.grad_norm = np.zeros(N, dtype=np.float64)
            self.tmp_grad_norm = np.zeros(N, dtype=np.float64)
            print(f"hotn: {self.hotn}, hot_rate: {self.hot_rate}")
        elif self.compress_method == 'cafe':
            cafe_hash_rate = args.cafe_hash_rate
            cafe_sketch_threshold = args.cafe_sketch_threshold
            cafe_decay = args.cafe_decay
            cur_dir = osp.join(osp.split(osp.abspath(__file__))[0])
            if not osp.exists(f'{cur_dir}/sklib.so'):
                os.system(f"g++ -fPIC -shared -o {cur_dir}/sklib.so -g -rdynamic -mavx2 -mbmi -mavx512bw -mavx512dq --std=c++17 -O3 -fopenmp {cur_dir}/sketch.cpp")
            lib = ctypes.CDLL(f'{cur_dir}/sklib.so')
            self.lib = lib

            totn = sum(embedding_nums)
            hotn = int(totn * compress_rate * (1 - cafe_hash_rate)
                    * (embedding_dim * 4 / (embedding_dim * 4 + 48)))
            cafe_hash_rate = compress_rate * cafe_hash_rate
            self.hotn = hotn
            print(f"cafe_hash_rate: {cafe_hash_rate}, hotn: {hotn}")

            self.weight_high = Parameter(
                torch.Tensor(hotn, embedding_dim),
                requires_grad=True,
            )
            scale = np.sqrt(1 / max(embedding_nums))
            nn.init.uniform_(self.weight_high, -scale, scale)
            self.register_buffer('sketch_buffer', torch.zeros(hotn * 12, dtype = torch.int32, device = 'cpu'))
            init = lib.init
            init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
            init.restype = None
            numpy_array = self.sketch_buffer.numpy()
            data_ptr = numpy_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            init(hotn, cafe_sketch_threshold, data_ptr, ctypes.c_double(cafe_decay))
        N = 0
        for i in range(embedding_nums.size):
            n = embedding_nums[i]

            tmp_n = max(n, 5)
            scale = np.sqrt(1 / tmp_n)
            if self.compress_method == 'mde' and n > compress_threshold:
                base = embedding_dim
                _m = new_embedding_dim[i]
                EE = PrEmbeddingBag(n, _m, base)
            elif self.compress_method == 'qr' and n > compress_threshold:
                EE = QREmbeddingBag(
                    n,
                    embedding_dim,
                    qr_collisions,
                )
            elif self.compress_method == 'ada' and n > compress_threshold:
                EE = AdaEmbeddingBag(
                    N,
                    self.weight,
                    self.dic,
                    self.device,
                )
                self.ada_emb[i] = True
                self.f_offset[i] = N
                N += n
            elif self.compress_method == 'cafe' and n > compress_threshold:
                EE = SKEmbeddingBag(
                    N,
                    self.hotn,
                    self.lib,
                    self.weight_high,
                    self.device,
                    n,
                    embedding_dim,
                    tot_nums,
                    int(math.ceil(cafe_hash_rate * n)),
                )
                self.sketch_emb[i] = True
                N += 1
            elif self.compress_method == 'hash' and n > compress_threshold:
                EE = HashEmbeddingBag(n, embedding_dim, compress_rate, mode="sum", sparse=True)
                nn.init.uniform_(EE.weight, -scale, scale)
            else:
                EE = nn.EmbeddingBag(n, embedding_dim, mode="sum", sparse=True)
                nn.init.uniform_(EE.weight, -scale, scale)

            embs.append(EE)
            tot_nums += n
        self.embeddings = embs

    def forward(self, lS_o, lS_i):
        ly = []
        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = lS_o[k]
            E = self.embeddings[k]
            if self.sketch_emb[k] or self.ada_emb[k]:
                V = E(
                    sparse_index_group_batch,
                    sparse_offset_group_batch,
                )
            else:
                V = E(
                    sparse_index_group_batch.to(self.device),
                    sparse_offset_group_batch.to(self.device),
                )
            ly.append(V)

        return ly

    def on_load(self):
        if self.compress_method == 'cafe':
            self.lib.update()

    def insert_grad(self, lS_i):
        if self.compress_method == 'cafe':
            for k, input in enumerate(lS_i):
                if self.sketch_emb[k] == True:
                    self.embeddings[k].insert_grad(input)
        elif self.compress_method == 'ada':
            tmp = 0
            N = len(lS_i[0])
            for k, input in enumerate(lS_i):
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
                raise AssertionError('Number of to be evicted features should be 0.')
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
        if (len(admit_) > m * 0.05):
            self.ada_rebuild()


