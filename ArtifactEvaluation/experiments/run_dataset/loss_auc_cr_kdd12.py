import numpy as np
import pandas as pd
import os

method = ["ada", "sketch", "qr", "md", "hash"]

compress_rate = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
sketch_rate = [0.7, 0.5, 0.5, 0.5, 0.3, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1]
sketch_threshold = [10, 10, 20, 30, 50, 100, 200, 500, 500, 500, 500, 500]

method_cr = [4, 12, 7, 5, 12]

for i in range(5):
    for c in range(method_cr[i]):
        md = method[i]
        cr = compress_rate[c]

        ops = ("../../bench/kdd12.sh" + 
                " \"--compress-rate=" + str(1.0 / cr) + 
                " --" + md + "-flag")
        if i == 1:
            ops += (" --sketch-threshold=" + str(sketch_threshold[c]) + 
                   " --hash-rate=" + str(sketch_rate[c]) + 
                   " --notinsert-test")

        ops += " --tensor-board-filename=\"../board/kdd12/" + str(md) + str(cr) + "\"\""
        print(f"command: {ops}")
        os.system(ops)

