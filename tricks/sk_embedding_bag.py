from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from numpy import random as ra
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import ctypes
import time


sketch_time = 0


def get_sketch_time():
    global sketch_time
    return sketch_time


def reset_sketch_time():
    global sketch_time
    sketch_time = 0


class SKEmbeddingBag(nn.Module):

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
        super(SKEmbeddingBag, self).__init__()
        # self.weight_high = weight_high
        self.field_num = field_num
        self.lib = lib
        self.offset = f_offset
        self.hash_size = hash_size
        self.hot_nums = hotn
        self.input_c = None

        self.num_categories = num_categories
        self.weight_h = weight_high
        self.embedding_dim = embedding_dim
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.device = device
        self.grad_norm = 0
        self.ins = self.lib.batch_insert
        self.ins.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.ins.restype = ctypes.POINTER(ctypes.c_int)
        self.query_dic = None

        self.que = self.lib.batch_query
        self.que.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.que.restype = ctypes.POINTER(ctypes.c_int)

        self.inv = self.lib.batch_insert_val
        self.inv.argtypes = [ctypes.POINTER(
            ctypes.c_int), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        self.inv.restype = ctypes.POINTER(ctypes.c_int)

        self.weight_hash = Parameter(
            torch.Tensor(self.hash_size, self.embedding_dim)
        )

        # print(f"hash_size: {self.hash_size}")
        self.reset_parameters()
        self.sparse = sparse

    def insert(self, input):
        N = len(input)
        input_l = (input + self.offset).numpy().astype(np.int32)
        addr = input_l.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        input_c = ctypes.cast(addr, ctypes.POINTER(ctypes.c_int))

        mask_ptr = self.ins(input_c, N)
        dic_ptr = self.que(input_c, N)

        mask = torch.frombuffer(ctypes.cast(mask_ptr, ctypes.POINTER(
            ctypes.c_int * N)).contents, dtype=torch.int32, count=N)
        dic = torch.frombuffer(ctypes.cast(dic_ptr, ctypes.POINTER(
            ctypes.c_int * N)).contents, dtype=torch.int32, count=N)

        dic_mask = (dic < 0)
        dic = torch.abs(dic)
        return mask, dic_mask, dic

    def query(self, input):
        N = len(input)
        input_l = (input + self.offset).numpy().astype(np.int32)
        addr = input_l.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        input_c = ctypes.cast(addr, ctypes.POINTER(ctypes.c_int))
        self.input_c = input_c

        dic_ptr = self.que(input_c, N)
        dic = torch.frombuffer(ctypes.cast(dic_ptr, ctypes.POINTER(
            ctypes.c_int * N)).contents, dtype=torch.int32, count=N)
        dic_mask = (dic < 0)
        dic = torch.abs(dic)
        return dic_mask, dic

    def reset_parameters(self):
        nn.init.uniform_(self.weight_hash, -np.sqrt(1 /
                         self.num_categories), np.sqrt(1 / self.num_categories))

    def forward(self, input, offsets=None, per_sample_weights=None, test=False):

        dic_mask, dic = self.query(input)
        # if test:
        #     dic_mask, dic = self.query(input)
        # else:
        #     mask, dic_mask, dic = self.insert(input) #cold to hot features, cold or hot features, features
            #idx = torch.nonzero(mask)
            # with torch.no_grad():
            #     for x in idx[:, 0]:
            #         self.weight_h[dic[x]] = self.weight_hash[input[x] % self.hash_size]
        self.query_dic = dic_mask.numpy()
        dic = dic.to(self.device)
        offsets = offsets.to(self.device)
        dic_mask = dic_mask.to(self.device).unsqueeze(1)
        embed_high = F.embedding_bag(
            dic % self.hot_nums,
            self.weight_h,
            offsets,
            sparse=True,
        )
        embed_hash = F.embedding_bag(
            dic % self.hash_size,
            self.weight_hash,
            offsets,
            sparse=True,
        )
        embed = torch.where(
            dic_mask,
            embed_high,
            embed_hash,
        )

        return embed

    def query_norm(self, input):
        N = len(input)
        l = self.field_num * N
        r = l + N
        dic_mask = self.query_dic
        grad_norm = torch.where(
            dic_mask,
            torch.norm(self.weight_h.grad._values()[l: r], dim=1, p=2),
            torch.norm(self.weight_hash.grad._values(), dim=1, p=2),
        )
        lst = self.grad_norm
        if self.grad_norm == 0:
            self.grad_norm = torch.sum(grad_norm)
        else:
            self.grad_norm = self.grad_norm * 0.8 + torch.sum(grad_norm) * 0.2
        grad_norm = grad_norm * N / self.grad_norm
        self.grad_norm = lst
        return grad_norm.cpu()

    def insert_grad(self, input):

        N = len(input)
        l = self.field_num * N
        r = l + N
        grad_norm = torch.where(
            torch.from_numpy(self.query_dic).to(self.device),
            torch.norm(self.weight_h.grad._values()[l: r], dim=1, p=2),
            torch.norm(self.weight_hash.grad._values(), dim=1, p=2),
        )

        # if self.grad_norm == 0:
        #     self.grad_norm = torch.sum(grad_norm)
        # else:
        #     self.grad_norm = self.grad_norm * 0.8 + torch.sum(grad_norm) * 0.2

        grad_norm = grad_norm * N / torch.sum(grad_norm)
        grad_norm_np = grad_norm.cpu().numpy()
        grad_norm_addr = grad_norm_np.ctypes.data_as(
            ctypes.POINTER(ctypes.c_float))
        grad_norm_c = ctypes.cast(
            grad_norm_addr, ctypes.POINTER(ctypes.c_float))

        mask_ptr = self.inv(self.input_c, grad_norm_c, N)
        mask = torch.frombuffer(ctypes.cast(mask_ptr, ctypes.POINTER(
            ctypes.c_int * N)).contents, dtype=torch.int32, count=N)
        idx = torch.nonzero(mask)

        dic_ptr = self.que(self.input_c, N)
        dic = torch.frombuffer(ctypes.cast(dic_ptr, ctypes.POINTER(
            ctypes.c_int * N)).contents, dtype=torch.int32, count=N)
        dic = torch.abs(dic)
        with torch.no_grad():
            for x in idx[:, 0]:
                self.weight_h[dic[x]] = self.weight_hash[input[x] %
                                                         self.hash_size]

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