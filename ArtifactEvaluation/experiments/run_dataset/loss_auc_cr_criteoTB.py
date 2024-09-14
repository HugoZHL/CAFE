import numpy as np
import pandas as pd
import os

method = ["ada", "sketch", "qr", "md", "hash"]

compress_rate = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
sketch_rate = [0.7, 0.5, 0.5, 0.5, 0.3, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1]
sketch_threshold = [10, 10, 20, 30, 50, 100, 200, 500, 500, 500, 500, 500]

method_cr = [3, 10, 7, 4, 10]

for i in range(5):
    for c in range(method_cr[i]):
        md = method[i]
        cr = compress_rate[c]

        ops = ("../../bench/criteo_terabyte.sh" + 
                " \"--compress_rate=" + str(1.0 / cr) + 
                " --" + md + "-flag")
        if i == 1:
            ops += (" --sketch_threshold=" + str(sketch_threshold[c]) + 
                   " --hash_rate=" + str(sketch_rate[c]) + 
                   " --notinsert_test")

        ops += " --tensor_board_filename=\"../board/criteotb/" + str(md) + str(cr) + "\"\""
        os.system(ops)

