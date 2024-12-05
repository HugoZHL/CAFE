import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import ctypes
import numpy as np

class SKEmbedding(nn.Module):
    def __init__(
        self,
        lib,
        hotn,
        hash_size,
        embedding_dim,
        padding_idx,
        device,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
    ):
        super(SKEmbedding, self).__init__()
        self.lib = lib
        self.hash_size = hash_size
        self.hot_nums = hotn
        self.padding_idx = padding_idx

        self.embedding_dim = embedding_dim
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.device = device

        self.que = self.lib.batch_query
        self.que.argtypes = [ctypes.POINTER(ctypes.c_uint), ctypes.c_int]
        self.que.restype = ctypes.POINTER(ctypes.c_int)

        self.inv = self.lib.batch_insert_val
        self.inv.argtypes = [ctypes.POINTER(
            ctypes.c_uint), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        self.inv.restype = ctypes.POINTER(ctypes.c_int)
        
        self.in_start = self.lib.batch_insert_start
        self.in_start.argtypes = None
        self.in_start.restype = None

        self.weight_h = Parameter(
            torch.zeros(self.hot_nums, self.embedding_dim).to(self.device)
        )
        self.weight_hash = Parameter(
            torch.zeros(self.hash_size, self.embedding_dim).to(self.device)
        )
        nn.init.xavier_uniform_(self.weight_h)
        nn.init.xavier_uniform_(self.weight_hash)
        
        self.input_queue = []
        self.query_dic_queue = []
        self.query_dic_mask_queue = []
        
        self.save_ss = self.lib.save_state
        self.save_ss.argtypes = [ctypes.c_char_p]
        self.save_ss.restype = None
        
        self.load_ss = self.lib.load_state
        self.load_ss.argtypes = [ctypes.c_char_p]
        self.load_ss.restype = None

    def query(self, input):
        N = len(input)
        input_l = input.cpu().numpy().astype(np.uint32)
        addr = input_l.ctypes.data_as(ctypes.POINTER(ctypes.c_uint))
        input_c = ctypes.cast(addr, ctypes.POINTER(ctypes.c_uint))

        dic_ptr = self.que(input_c, N)
        dic = torch.frombuffer(ctypes.cast(dic_ptr, ctypes.POINTER(
            ctypes.c_int * N)).contents, dtype=torch.int32, count=N)
        dic_mask = (dic < 0)
        dic = torch.abs(dic)
        return dic_mask, dic

    def forward(self, input):
        shape = input.shape
        input = input.flatten()

        dic_mask, dic = self.query(input)
        
        dic = dic.to(self.device)
        dic_mask = dic_mask.to(self.device).unsqueeze(1)
        
        self.input_queue.append(input.clone())
        self.query_dic_queue.append(dic.clone())
        self.query_dic_mask_queue.append(dic_mask.clone())

        embed_high = F.embedding(
            dic % self.hot_nums,
            self.weight_h,
        )
        
        non_padding_mask = (input != self.padding_idx).unsqueeze(1)
        embed_hash = F.embedding(
            dic % self.hash_size,
            self.weight_hash,
        ) * non_padding_mask

        embed = torch.where(
            dic_mask,
            embed_high,
            embed_hash,
        )

        embed = embed.view(*shape, self.embedding_dim)

        return embed

    def insert_grad(self, input, dic, dic_mask):
        N = len(input)
        
        input_l = input.cpu().numpy().astype(np.int32)
        addr = input_l.ctypes.data_as(ctypes.POINTER(ctypes.c_uint))
        input_c = ctypes.cast(addr, ctypes.POINTER(ctypes.c_uint))
        
        grad_high = F.embedding(
            dic % self.hot_nums,
            self.weight_h.grad,
        )
        grad_hash = F.embedding(
            dic % self.hash_size,
            self.weight_hash.grad,
        )
        
        grad = torch.where(
            dic_mask,
            grad_high,
            grad_hash,
        )
        
        grad_norm = torch.norm(grad, dim=-1, p=2)
        grad_norm = grad_norm * N / torch.sum(grad_norm)
        grad_norm_np = grad_norm.cpu().numpy()
        grad_norm_addr = grad_norm_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        grad_norm_c = ctypes.cast(grad_norm_addr, ctypes.POINTER(ctypes.c_float))
        
        mask_ptr = self.inv(input_c, grad_norm_c, N)
        mask = torch.frombuffer(ctypes.cast(mask_ptr, ctypes.POINTER(
            ctypes.c_int * N)).contents, dtype=torch.int32, count=N)
        idx = torch.nonzero(mask)

        dic_ptr = self.que(input_c, N)
        dic = torch.frombuffer(ctypes.cast(dic_ptr, ctypes.POINTER(
            ctypes.c_int * N)).contents, dtype=torch.int32, count=N)
        dic = torch.abs(dic)
        with torch.no_grad():
            try:
                for x in idx[:, 0]:
                    self.weight_h[dic[x]] = self.weight_hash[input[x] % self.hash_size]
            except:
                for x in idx[:, 0]:
                    print(dic[x], input[x])
                print(dic)
                exit(0)

    def insert_all_grad(self):
        if self.hot_nums > 1:
            self.in_start()
            for i in range(len(self.input_queue)):
                self.insert_grad(self.input_queue[i], self.query_dic_queue[i], self.query_dic_mask_queue[i])
        self.input_queue = []
        self.query_dic_queue = []
        self.query_dic_mask_queue = []

    def save_sketch(self, path):
        self.save_ss(ctypes.c_char_p(path.encode('utf-8')))

    def load_sketch(self, path):
        self.load_ss(ctypes.c_char_p(path.encode('utf-8')))
