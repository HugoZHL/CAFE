import ctypes
import numpy as np
import time
import math
import os
import os.path as osp
import pandas as pd

cur_dir = osp.split(osp.abspath(__file__))[0]

os.system(f"g++ -fPIC -shared -o {cur_dir}/sklibtest.so --std=c++17 -O3 -fopenmp {cur_dir}/sketchtest.cpp")
lib = ctypes.CDLL(f'{cur_dir}/sklibtest.so')

init = lib.init
init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
init.restype = None

ins = lib.batch_insert
ins.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
ins.restype = ctypes.POINTER(ctypes.c_int)

que = lib.batch_query
que.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
que.restype = ctypes.POINTER(ctypes.c_int)

inv = lib.batch_insert_val
inv.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(
    ctypes.c_float), ctypes.c_int]
inv.restype = ctypes.POINTER(ctypes.c_int)

analyse = lib.analyse
analyse.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
analyse.restype = ctypes.c_float

criteo_dir = osp.join(cur_dir, '../datasets/criteo')

data = np.memmap(osp.join(criteo_dir, "processed_sparse_sep.bin"),
                 dtype=np.int32, mode='r', shape=(45840617, 26))
print(data)

count = np.fromfile(osp.join(criteo_dir, 'processed_count.bin'), dtype=np.int32)
print(count)

grad_norm_npy = np.zeros(np.sum(count), dtype=np.float32)

hotn = int(np.sum(count) * 0.7 * 0.001 * (16 / 28))
print(hotn)
len = 45840617 * 6 // 7
batch_size = 128
total_time = 0
step = math.floor(len / batch_size)
q_time = 0

rate = [1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4]
mem = [633, 633 * 1.2, 633 * 1.4, 633 * 1.6, 633 * 1.8, 633 * 2, 633 * 2.2, 633 * 2.4]
mem_recall = pd.DataFrame(index = mem, columns=[4, 8, 16, 32])

for c in [4, 8, 16, 32]:
    for k in range(8):
        init(int(hotn * rate[k]), 300, hotn, c)
        grad_norm_npy = np.zeros(np.sum(count), dtype=np.float32)
        for i in range(math.floor(len / batch_size)):
            l = i * batch_size
            r = l + batch_size
            for j in range(26):
                input = np.array(data[l: r, j], dtype=np.int32)
                addr = input.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
                input_c = ctypes.cast(addr, ctypes.POINTER(ctypes.c_int))
                np.add.at(grad_norm_npy, input, 1)
                # print(f"grad_norm: {grad_norm}")

                mask_ptr = ins(input_c, batch_size)

            if (i % 2048 == 0):
                print(f"Finished insert {i}/{math.floor(len / batch_size)}")

        hotn = int(np.sum(count) * 0.7 * 0.001 * (16 / 28))
        ind = np.argsort(-grad_norm_npy)[:hotn]
        ind = np.array(np.sort(ind), dtype=np.int32)
        print(ind)
        ind_addr = ind.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        ind_c = ctypes.cast(ind_addr, ctypes.POINTER(ctypes.c_int))
        Recall = analyse(ind_c, hotn)
        mem_recall.loc[mem[k], c] = Recall
        print(mem_recall)

mem_recall.to_csv(osp.join(cur_dir, "sketch_mem_recall.csv"))


throughput_df = pd.DataFrame(index=[4,8,16,32], columns=["Insert", "Query"])
step = math.floor(len / batch_size / 7)
for c in [4, 8, 16, 32]:
    init(int(hotn), 300, hotn, c)
    total_time = 0
    q_time = 0
    for i in range(step):
        l = i * batch_size
        r = l + batch_size
        for j in range(26):
            input = np.array(data[l: r, j], dtype=np.int32)
            addr = input.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            input_c = ctypes.cast(addr, ctypes.POINTER(ctypes.c_int))
            np.add.at(grad_norm_npy, input, 1)
            # print(f"grad_norm: {grad_norm}")
            start_time = time.time()

            mask_ptr = ins(input_c, batch_size)

            end_time = time.time()
            total_time += end_time - start_time
            start_time = time.time()
            dic = que(input_c, batch_size)
            end_time = time.time()
            q_time += end_time - start_time
        if (i % 2048 == 0):
            print(f"Test throughput: Finished insert {i}/{math.floor(len / batch_size)}")
    
    tot_num = step * batch_size * 26
    per_entry_time = total_time / tot_num
    throughput = 1.0 / per_entry_time
    per_entry_time2 = q_time / tot_num
    throughput2 = 1.0 / per_entry_time2
    throughput_df.loc[c, "Query"] = throughput2
    throughput_df.loc[c, "Insert"] = throughput

throughput_df.to_csv(osp.join(cur_dir, "throughput.csv"))


day = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
time_recall_1000 = pd.DataFrame(index=day, columns=["Sliding-window Topk", "Up-to-date Topk"])
step = math.floor(len / batch_size)
hotn = int(np.sum(count) * 0.7 * 0.001 * (16 / 28))
init(int(hotn), 300, hotn, 4)
for i in range(step):
    l = i * batch_size
    r = l + batch_size
    grad_norm_npy = np.zeros(np.sum(count), dtype=np.float32)
    grad_norm_npy_window = np.zeros(np.sum(count), dtype=np.float32)
    for j in range(26):
        input = np.array(data[l: r, j], dtype=np.int32)
        addr = input.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        input_c = ctypes.cast(addr, ctypes.POINTER(ctypes.c_int))
        np.add.at(grad_norm_npy, input, 1)
        np.add.at(grad_norm_npy_window, input, 1)
        mask_ptr = ins(input_c, batch_size)
    if (i % 25000 == 0 and i != 0):        
        hotn = int(np.sum(count) * 0.7 * 0.001 * (16 / 28))
        ind = np.argsort(-grad_norm_npy)[:hotn]
        ind = np.array(np.sort(ind), dtype=np.int32)
        print(ind)
        ind_addr = ind.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        ind_c = ctypes.cast(ind_addr, ctypes.POINTER(ctypes.c_int))
        Recall = analyse(ind_c, hotn)
        time_recall_1000.loc[i / 50000, "Up-to-date Topk"] = Recall

        ind = np.argsort(-grad_norm_npy_window)[:hotn]
        ind = np.array(np.sort(ind), dtype=np.int32)
        print(ind)
        ind_addr = ind.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        ind_c = ctypes.cast(ind_addr, ctypes.POINTER(ctypes.c_int))
        Recall = analyse(ind_c, hotn)
        time_recall_1000.loc[i / 50000, "Sliding-window Topk"] = Recall
        grad_norm_npy_window = np.zeros(np.sum(count), dtype=np.float32)

time_recall_1000.to_csv(osp.join(cur_dir, "time_recall_1000.csv"))


hotn = int(np.sum(count) * 0.3 * 0.01 * (16 / 28))
day = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
time_recall_100 = pd.DataFrame(index=day, columns=["Sliding-window Topk", "Up-to-date Topk"])
step = math.floor(len / batch_size)
init(int(hotn), 50, hotn, 4)
for i in range(step):
    l = i * batch_size
    r = l + batch_size
    grad_norm_npy = np.zeros(np.sum(count), dtype=np.float32)
    grad_norm_npy_window = np.zeros(np.sum(count), dtype=np.float32)
    for j in range(26):
        input = np.array(data[l: r, j], dtype=np.int32)
        addr = input.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        input_c = ctypes.cast(addr, ctypes.POINTER(ctypes.c_int))
        np.add.at(grad_norm_npy, input, 1)
        np.add.at(grad_norm_npy_window, input, 1)
        mask_ptr = ins(input_c, batch_size)
    if (i % 25000 == 0 and i != 0):        
        hotn = int(np.sum(count) * 0.3 * 0.01 * (16 / 28))
        ind = np.argsort(-grad_norm_npy)[:hotn]
        ind = np.array(np.sort(ind), dtype=np.int32)
        print(ind)
        ind_addr = ind.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        ind_c = ctypes.cast(ind_addr, ctypes.POINTER(ctypes.c_int))
        Recall = analyse(ind_c, hotn)
        time_recall_100.loc[i / 50000, "Up-to-date Topk"] = Recall

        ind = np.argsort(-grad_norm_npy_window)[:hotn]
        ind = np.array(np.sort(ind), dtype=np.int32)
        print(ind)
        ind_addr = ind.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        ind_c = ctypes.cast(ind_addr, ctypes.POINTER(ctypes.c_int))
        Recall = analyse(ind_c, hotn)
        time_recall_100.loc[i / 50000, "Sliding-window Topk"] = Recall
        grad_norm_npy_window = np.zeros(np.sum(count), dtype=np.float32)

time_recall_100.to_csv(osp.join(cur_dir, "time_recall_100.csv"))
