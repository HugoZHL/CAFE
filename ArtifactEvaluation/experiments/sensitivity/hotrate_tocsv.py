import tensorboard
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import os.path as osp


method = ["sketch"]
compress_rate = [1000]
sketch_rate = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.00001]
idx = ["0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "loo"]

auc_csv = pd.DataFrame(index = idx, columns=['auc'])

for i in range(7):
    md = method[0]
    cr = compress_rate[0]
    log_file = osp.join("./board", "sensitivity",  "hotrate" + str(sketch_rate[i]))

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

auc_csv.to_csv("./excels/hotrate.csv")