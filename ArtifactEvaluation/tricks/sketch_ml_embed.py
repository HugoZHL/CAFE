from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from numpy import random as ra
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import ctypes
import time
import math


sketch_time = 0


def get_sketch_time():
    global sketch_time
    return sketch_time


def reset_sketch_time():
    global sketch_time
    sketch_time = 0


class MLEmbeddingBag(nn.Module):
    r"""Computes sums or means over two 'bags' of embeddings, one using the quotient
    of the indices and the other using the remainder of the indices, without
    instantiating the intermediate embeddings, then performs an operation to combine these.

    For bags of constant length and no :attr:`per_sample_weights`, this class

        * with ``mode="sum"`` is equivalent to :class:`~torch.nn.Embedding` followed by ``torch.sum(dim=0)``,
        * with ``mode="mean"`` is equivalent to :class:`~torch.nn.Embedding` followed by ``torch.mean(dim=0)``,
        * with ``mode="max"`` is equivalent to :class:`~torch.nn.Embedding` followed by ``torch.max(dim=0)``.

    However, :class:`~torch.nn.EmbeddingBag` is much more time and memory efficient than using a chain of these
    operations.

    QREmbeddingBag also supports per-sample weights as an argument to the forward
    pass. This scales the output of the Embedding before performing a weighted
    reduction as specified by ``mode``. If :attr:`per_sample_weights`` is passed, the
    only supported ``mode`` is ``"sum"``, which computes a weighted sum according to
    :attr:`per_sample_weights`.

    Known Issues:
    Autograd breaks with multiple GPUs. It breaks only with multiple embeddings.

    Args:
        num_categories (int): total number of unique categories. The input indices must be in
                              0, 1, ..., num_categories - 1.
        embedding_dim (list): list of sizes for each embedding vector in each table. If ``"add"``
                              or ``"mult"`` operation are used, these embedding dimensions must be
                              the same. If a single embedding_dim is used, then it will use this
                              embedding_dim for both embedding tables.
        num_collisions (int): number of collisions to enforce.
        operation (string, optional): ``"concat"``, ``"add"``, or ``"mult". Specifies the operation
                                      to compose embeddings. ``"concat"`` concatenates the embeddings,
                                      ``"add"`` sums the embeddings, and ``"mult"`` multiplies
                                      (component-wise) the embeddings.
                                      Default: ``"mult"``
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                    is renormalized to have norm :attr:`max_norm`.
        norm_type (float, optional): The p of the p-norm to compute for the :attr:`max_norm` option. Default ``2``.
        scale_grad_by_freq (boolean, optional): if given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
                                                Note: this option is not supported when ``mode="max"``.
        mode (string, optional): ``"sum"``, ``"mean"`` or ``"max"``. Specifies the way to reduce the bag.
                                 ``"sum"`` computes the weighted sum, taking :attr:`per_sample_weights`
                                 into consideration. ``"mean"`` computes the average of the values
                                 in the bag, ``"max"`` computes the max value over each bag.
                                 Default: ``"mean"``
        sparse (bool, optional): if ``True``, gradient w.r.t. :attr:`weight` matrix will be a sparse tensor. See
                                 Notes for more details regarding sparse gradients. Note: this option is not
                                 supported when ``mode="max"``.

    Attributes:
        weight (Tensor): the learnable weights of each embedding table is the module of shape
                         `(num_embeddings, embedding_dim)` initialized using a uniform distribution
                         with sqrt(1 / num_categories).

    Inputs: :attr:`input` (LongTensor), :attr:`offsets` (LongTensor, optional), and
        :attr:`per_index_weights` (Tensor, optional)

        - If :attr:`input` is 2D of shape `(B, N)`,

          it will be treated as ``B`` bags (sequences) each of fixed length ``N``, and
          this will return ``B`` values aggregated in a way depending on the :attr:`mode`.
          :attr:`offsets` is ignored and required to be ``None`` in this case.

        - If :attr:`input` is 1D of shape `(N)`,

          it will be treated as a concatenation of multiple bags (sequences).
          :attr:`offsets` is required to be a 1D tensor containing the
          starting index positions of each bag in :attr:`input`. Therefore,
          for :attr:`offsets` of shape `(B)`, :attr:`input` will be viewed as
          having ``B`` bags. Empty bags (i.e., having 0-length) will have
          returned vectors filled by zeros.

        per_sample_weights (Tensor, optional): a tensor of float / double weights, or None
            to indicate all weights should be taken to be ``1``. If specified, :attr:`per_sample_weights`
            must have exactly the same shape as input and is treated as having the same
            :attr:`offsets`, if those are not ``None``. Only supported for ``mode='sum'``.


    Output shape: `(B, embedding_dim)`

    """
    """
    __constants__ = [
        "num_categories",
        "embedding_dim",
        "num_collisions",
        "operation",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "mode",
        "sparse",
    ]
    """

    def __init__(
        self,
        field_num,
        hotn,
        lib,
        weight_high,
        device,
        num_categories,
        embedding_dim,
        f_offset,
        hash_size,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
    ):
        super(MLEmbeddingBag, self).__init__()
        # self.weight_high = weight_high
        self.field_num = field_num
        self.lib = lib
        self.offset = f_offset
        self.hash_size = hash_size
        self.hot_nums = hotn
        self.p = 7

        self.num_categories = num_categories
        self.weight_h = weight_high
        self.embedding_dim = embedding_dim
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.device = device
        self.grad_norm = 0
        self.threshold1 = 500
        self.threshold2 = self.threshold1 * 0.2

        self.ins = self.lib.batch_insert
        self.ins.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.ins.restype = ctypes.POINTER(ctypes.c_int)

        self.que = self.lib.batch_query
        self.que.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.que.restype = ctypes.POINTER(ctypes.c_int)

        self.inv = self.lib.batch_insert_val
        self.inv.argtypes = [ctypes.POINTER(
            ctypes.c_int), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        self.inv.restype = ctypes.POINTER(ctypes.c_int)

        self.que_cnt = self.lib.batch_query_cnt
        self.que_cnt.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.que_cnt.restype = ctypes.POINTER(ctypes.c_float)

        self.level_rate = 0.7
        self.hash_size1 = int(math.ceil(self.hash_size * self.level_rate))
        self.hash_size2 = int(
            math.ceil(self.hash_size * (1 - self.level_rate)))
        self.weight_hash = Parameter(
            torch.Tensor(self.hash_size1, self.embedding_dim)
        )
        self.weight_hash2 = Parameter(
            torch.Tensor(self.hash_size2, self.embedding_dim)
        )

        print(f"hash_size: {self.hash_size}")
        self.reset_parameters()
        self.sparse = sparse

    def insert(self, input):
        N = len(input)
        input_l = np.array((input + self.offset), dtype=np.int32)
        addr = input_l.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        input_c = ctypes.cast(addr, ctypes.POINTER(ctypes.c_int))

        mask_ptr = self.ins(input_c, N)
        dic_ptr = self.que(input_c, N)

        # mask = torch.tensor(ctypes.cast(mask_ptr, ctypes.POINTER(ctypes.c_int * len(input))).contents)
        # dic = torch.tensor(ctypes.cast(dic_ptr, ctypes.POINTER(ctypes.c_int * len(input))).contents)
        mask = torch.frombuffer(ctypes.cast(mask_ptr, ctypes.POINTER(
            ctypes.c_int * N)).contents, dtype=torch.int32, count=N)
        dic = torch.frombuffer(ctypes.cast(dic_ptr, ctypes.POINTER(
            ctypes.c_int * N)).contents, dtype=torch.int32, count=N)

        dic_mask = (dic < 0)
        dic = torch.abs(dic)
        return mask, dic_mask, dic

    def query_cnt(self, input):
        N = len(input)
        input_l = np.array((input + self.offset), dtype=np.int32)
        addr = input_l.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        input_c = ctypes.cast(addr, ctypes.POINTER(ctypes.c_int))

        dic_ptr = self.que_cnt(input_c, N)
        cnt = torch.frombuffer(ctypes.cast(dic_ptr, ctypes.POINTER(
            ctypes.c_float * N)).contents, dtype=torch.float32, count=N)
        return cnt

    def query(self, input):
        N = len(input)
        input_l = np.array((input + self.offset), dtype=np.int32)
        addr = input_l.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        input_c = ctypes.cast(addr, ctypes.POINTER(ctypes.c_int))

        dic_ptr = self.que(input_c, N)
        dic = torch.frombuffer(ctypes.cast(dic_ptr, ctypes.POINTER(
            ctypes.c_int * N)).contents, dtype=torch.int32, count=N)
        dic_mask = (dic < 0)
        dic = torch.abs(dic)
        return dic_mask, dic

    def reset_parameters(self):
        nn.init.uniform_(self.weight_hash, -np.sqrt(1 /
                         self.num_categories), np.sqrt(1 / self.num_categories))
        nn.init.uniform_(self.weight_hash2, -np.sqrt(1 /
                         self.num_categories), np.sqrt(1 / self.num_categories))

    def forward(self, input, offsets=None, per_sample_weights=None, test=False):
        start_time = time.time()
        dic_mask, dic = self.query(input)
        # if test:
        #     dic_mask, dic = self.query(input)
        # else:
        #     mask, dic_mask, dic = self.insert(input) #cold to hot features, cold or hot features, features
        #     #idx = torch.nonzero(mask)
        #     # with torch.no_grad():
        #     #     for x in idx[:, 0]:
        #     #         self.weight_h[dic[x]] = self.weight_hash[input[x] % self.hash_size]
        cnt = self.query_cnt(input)
        cnt = torch.clip(cnt, self.threshold2, self.threshold1)
        rate = np.array(((cnt - self.threshold2) > 0), dtype=np.int32)
        rate = torch.Tensor(rate).to(self.device)
        rate1 = (1 - rate).unsqueeze(1)
        rate2 = rate.unsqueeze(1)

        start_time = time.time()
        dic = dic.to(self.device)
        # print(f"dic: {dic} {dic_mask}")
        offsets = offsets.to(self.device)
        dic_mask = dic_mask.to(self.device).unsqueeze(1)
        # embed = self.weight_hash[input % self.hash_size]
        embed_high = F.embedding_bag(
            dic % self.hot_nums,
            self.weight_h,
            offsets,
            sparse=True,
        )
        embed_hash1 = F.embedding_bag(
            dic % self.hash_size1,
            self.weight_hash,
            offsets,
            sparse=True,
        )
        embed_hash2 = F.embedding_bag(
            dic * self.p % self.hash_size2,
            self.weight_hash2,
            offsets,
            sparse=True,
        )
        embed = torch.where(
            dic_mask,
            embed_high,
            embed_hash1 + embed_hash2,
        )

        # embed[dic_mask == False] = self.weight_hash[dic[dic_mask == False]%self.hash_size]
        # embed[dic_mask == True] = self.weight_h[dic[dic_mask == True]]
        global sketch_time
        sketch_time += time.time() - start_time
        return embed

    def insert_grad(self, input):
        # print(f"input: {input}")

        start_time = time.time()
        N = len(input)
        dic_mask, dic = self.query(input)

        cnt = self.query_cnt(input)
        cnt = torch.clip(cnt, self.threshold2, self.threshold1)
        rate = np.array(((cnt - self.threshold2) > 0), dtype=np.int32)
        rate = torch.Tensor(rate).to(self.device)
        rate1 = (1 - rate).unsqueeze(1)
        rate2 = rate.unsqueeze(1)
        g1 = self.weight_hash.grad._values()
        # g1 = g1 * rate1
        g2 = self.weight_hash2.grad._values()
        # g2 = g2 * rate2

        l = self.field_num * N
        r = l + N
        grad_norm = torch.where(
            dic_mask.to(self.device),
            torch.norm(self.weight_h.grad._values()[l: r], dim=1, p=2),
            torch.norm(g1 + g2, dim=1, p=2),
        )
        # print(f"grad norm: {grad_norm}")

        if self.grad_norm == 0:
            self.grad_norm = torch.sum(grad_norm)
        else:
            self.grad_norm = self.grad_norm * 0.8 + torch.sum(grad_norm) * 0.2

        grad_norm = grad_norm * N / self.grad_norm
        # print(f"grad_norm: {grad_norm} {torch.max(grad_norm)}")

        input_l = np.array((input + self.offset), dtype=np.int32)
        addr = input_l.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        input_c = ctypes.cast(addr, ctypes.POINTER(ctypes.c_int))

        grad_norm_np = np.array(grad_norm.cpu(), dtype=np.float32)
        grad_norm_addr = grad_norm_np.ctypes.data_as(
            ctypes.POINTER(ctypes.c_float))
        grad_norm_c = ctypes.cast(
            grad_norm_addr, ctypes.POINTER(ctypes.c_float))

        mask_ptr = self.inv(input_c, grad_norm_c, N)
        # mask = torch.frombuffer(ctypes.cast(mask_ptr, ctypes.POINTER(ctypes.c_int * N)).contents, dtype=torch.int32, count = N)
        # idx = torch.nonzero(mask)

        # dic_ptr = self.que(input_c, N)
        # dic = torch.frombuffer(ctypes.cast(dic_ptr, ctypes.POINTER(ctypes.c_int * N)).contents, dtype=torch.int32, count = N)
        # dic = torch.abs(dic)
        # #print(f"idx: {idx}, dic: {dic}, input: {input_l}")
        # with torch.no_grad():
        #     for x in idx[:, 0]:
        #         self.weight_h[dic[x]] = self.weight_hash[input[x] % self.hash_size2]

    def extra_repr(self):
        s = "{num_embeddings}, {embedding_dim}"
        if self.max_norm is not None:
            s += ", max_norm={max_norm}"
        if self.norm_type != 2:
            s += ", norm_type={norm_type}"
        if self.scale_grad_by_freq is not False:
            s += ", scale_grad_by_freq={scale_grad_by_freq}"
        s += ", mode={mode}"
        return s.format(**self.__dict__)
