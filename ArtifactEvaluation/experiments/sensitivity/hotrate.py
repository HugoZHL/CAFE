import tensorboard
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import os.path as osp

import os


method = ["sketch"]
compress_rate = [1000]
sketch_rate = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.00001]


for rate in sketch_rate:
    md = method[0]
    cr = compress_rate[0]

    ops = ("../../bench/criteo_kaggle.sh" + 
            " \"--compress_rate=" + str(1.0 / cr) + 
            " --" + md + "-flag")
    ops += (" --cafe_sketch_threshold=500" + 
            " --cafe_hash_rate=" + str(rate) + 
            " --notinsert_test")

    ops += " --tensor_board_filename=\"../board/sensitivity/hotrate" + str(rate) + "\"\""
    os.system(ops)