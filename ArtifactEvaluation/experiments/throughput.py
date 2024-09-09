import numpy as np
import pandas as pd
import os

method = ["ada", "sketch", "qr", "md", "hash"]

compress_rate = [10]
sketch_rate = [0.5]
sketch_threshold = [10]

method_cr = [1,1,1,1,1]

df = pd.DataFrame(index = ["Train", "Test"], columns= ["Hash", "Q-R Trick", "MDE", "AdaEmbed", "CAFE(ours)"])
df.to_csv("./excels/throughput.csv")
for i in range(5):
    for c in range(method_cr[i]):
        md = method[i]
        cr = compress_rate[c]

        ops = ("./bench/criteo_kaggle.sh" + 
                " \"--compress-rate=" + str(1.0 / cr) + 
                " --" + md + "-flag" + 
                " --test-throughput")
        if i == 1:
            ops += (" --sketch-threshold=" + str(sketch_threshold[c]) + 
                   " --hash-rate=" + str(sketch_rate) + 
                   " --notinsert-test")

        ops += " --tensor-board-filename=\"board/criteo/" + str(method) + str(cr) + "\"\""
        os.system(ops)

