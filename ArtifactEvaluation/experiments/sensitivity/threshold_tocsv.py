import tensorboard
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import os.path as osp

method = ["sketch"]
compress_rate = [1000]
sketch_rate = [0.7, 0.5, 0.5, 0.5, 0.3, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1]
sketch_threshold = [100, 300, 500, 700, 900]

auc_csv = pd.DataFrame(index = sketch_threshold, columns=['auc'])

for i in range(5):
    md = method[0]
    cr = compress_rate[0]
    log_file = osp.join("./board", "sensitivity",  "threshold" + str(sketch_threshold[i]))

    ea = event_accumulator.EventAccumulator(log_file)
    ea.Reload()
    scalar_data = ea.scalars
    print(scalar_data.Keys())
    loss = ea.scalars.Items('Train/Loss')
    auc = ea.scalars.Items('roc_auc')
    print(len(loss))
    print(loss)
    # pd.DataFrame(ea.Scalars('Train/Loss')).to_csv('./excels/' + str(md) + str(cr) + '_loss.csv')
    lst = 0
    avr_loss = 0
    avr_auc = auc[-2].value
    for x in loss:
        avr_loss += (x.step - lst) * x.value
        lst = x.step
    avr_loss /= lst
    print(f"loss: {avr_loss}")
    # pd.DataFrame(ea.Scalars('roc_auc')[:-1]).to_csv('./excels/' + str(md) + str(cr) + '_auc.csv')
    auc_csv.loc[cr, md] = ea.Scalars('roc_auc')[-2].value

auc_csv.to_csv("./excels/decay.csv")