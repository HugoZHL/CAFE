import tensorboard
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import os.path as osp

import os


method = ["sketch"]
compress_rate = [1000]
sketch_rate = [0.7, 0.5, 0.5, 0.5, 0.3, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1]
sketch_decay = [0.9, 0.95, 0.98, 0.99, 1]

method_cr = [2, 12, 7, 3, 12]

for decay in sketch_decay:
    md = method[0]
    cr = compress_rate[0]

    ops = ("../../bench/criteo_kaggle.sh" + 
            " \"--compress-rate=" + str(1.0 / cr) + 
            " --" + md + "-flag")
    ops += (" --sketch-threshold=500" + 
            " --hash-rate=0.2" + 
            " --notinsert-test" + 
            " --sketch-decay=" + str(decay))

    ops += " --tensor-board-filename=\"../board/sensitivity/decay" + str(decay) + "\"\""
    os.system(ops)