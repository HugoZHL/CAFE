import ctypes
import numpy as np
import time
import math


lib = ctypes.CDLL('./sklibtest.so')


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
inv.restype = None

get_sketch_ans = lib.ans
get_sketch_ans.argtypes = None
get_sketch_ans.restype = ctypes.POINTER(ctypes.c_int)

grad = np.load("../importance.npy")
print(grad)
print(grad.shape)
print(np.sum(grad))


data = np.memmap("../../criteo/origin/kaggle_processed_sparse.bin",
                 dtype=np.int32, mode='r', shape=(45840617, 26))
print(data)

count = [1460, 583, 10131227, 2202608, 305,
         24, 12517, 633, 3, 93145, 5683,
         8351593, 3194, 27, 14992, 5461306,
         10, 5652, 2173, 4, 7046547, 18,
         15, 286181, 105, 142572]

grad_norm_npy = np.zeros(np.sum(count), dtype=np.float32)

hotn = int(np.sum(count) * 0.7 * 0.001 * (16 / 28))
print(hotn)
len = 45840617 * 6 // 40
batch_size = 128
total_time = 0
step = math.floor(len / batch_size)
q_time = 0
for i in range(math.floor(len / batch_size)):
    l = i * batch_size
    r = l + batch_size
    for j in range(26):
        input = np.array(data[l: r, j], dtype=np.int32)
        addr = input.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        input_c = ctypes.cast(addr, ctypes.POINTER(ctypes.c_int))
        grad_norm = np.array(grad[j, l: r], dtype=np.float32)
        grad_norm = grad_norm / np.sum(grad_norm) * batch_size
        np.add.at(grad_norm_npy, input, grad_norm)
        grad_norm_addr = grad_norm.ctypes.data_as(
            ctypes.POINTER(ctypes.c_float))
        # print(f"grad_norm: {grad_norm}")
        grad_norm_c = ctypes.cast(
            grad_norm_addr, ctypes.POINTER(ctypes.c_float))
        start_time = time.time()

        mask_ptr = inv(input_c, grad_norm_c, batch_size)

        end_time = time.time()
        total_time += end_time - start_time
        start_time = time.time()
        dic = que(input_c, batch_size)
        end_time = time.time()
        q_time += end_time - start_time
    if (i % 2048 == 0):
        print(f"Finished insert {i}/{math.floor(len / batch_size)}")


# np.save("grad_norm_kaggle.npy", grad_norm_npy)
hotn = int(np.sum(count) * 0.7 * 0.001 * (16 / 28))
ind = np.argsort(-grad_norm_npy)[:hotn]
ind = np.array(np.sort(ind), dtype=np.int32)
print(ind)


ind_addr = ind.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
ind_c = ctypes.cast(ind_addr, ctypes.POINTER(ctypes.c_int))
analyse(ind_c, hotn)
tot_num = step * batch_size * 26
per_entry_time = total_time / tot_num
throughput = 1.0 / per_entry_time
per_entry_time2 = q_time / tot_num
throughput2 = 1.0 / per_entry_time2
print(f"total_time: {total_time}, entrys: {tot_num}")
print(f"per entry time: {per_entry_time}, throughput: {throughput}")
print(
    f"query_time: {q_time}, per query time: {per_entry_time2}, throughput: {throughput2}")
sketch_topk = get_sketch_ans()
sketch_topk = np.frombuffer(ctypes.cast(sketch_topk, ctypes.POINTER(
    ctypes.c_int * (hotn - 1))).contents, dtype=np.int32, count=hotn-1)
print(sketch_topk)
print(ind)
print(grad_norm_npy[sketch_topk])
print(grad_norm_npy[ind])
print(grad_norm_npy[sketch_topk], grad_norm_npy[ind])
print(np.sum(grad_norm_npy[sketch_topk]), np.sum(grad_norm_npy[ind]))
