import tensorboard
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import os.path as osp


dataset = "criteo"
method = ["ada", "sketch", "qr", "md", "hash"]

compress_rate = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
sketch_rate = [0.7, 0.5, 0.5, 0.5, 0.3, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1]
sketch_threshold = [10, 10, 20, 30, 50, 100, 200, 500, 500, 500, 500, 500]

method_cr = [2, 12, 7, 3, 12]

auc_csv = pd.DataFrame(index = compress_rate, columns=method)
loss_csv = pd.DataFrame(index = compress_rate, columns=method)

for i in range(5):
    for c in range(method_cr[i]):
        md = method[i]
        cr = compress_rate[c]
        log_file = osp.join("./board", dataset, md + str(cr))

        ea = event_accumulator.EventAccumulator(log_file)
        ea.Reload()
        scalar_data = ea.scalars
        print(scalar_data.Keys())
        loss = ea.scalars.Items('Train/Loss')
        auc = ea.scalars.Items('roc_auc')
        print(len(loss))
        print(loss)
        pd.DataFrame(ea.Scalars('Train/Loss')).to_csv('./excels/criteo/' + str(md) + str(cr) + '_loss.csv')
        lst = 0
        avr_loss = 0
        avr_auc = auc[-2].value
        for x in loss:
            avr_loss += (x.step - lst) * x.value
            lst = x.step
        avr_loss /= lst
        print(f"loss: {avr_loss}")
        pd.DataFrame(ea.Scalars('roc_auc')[:-1]).to_csv('./excels/criteo/' + str(md) + str(cr) + '_auc.csv')
        auc_csv.loc[cr, md] = ea.Scalars('roc_auc')[-2].value
        loss_csv.loc(cr, md) = avr_loss

auc_csv.to_csv("./excels/criteo/auc.csv")
loss_csv.to_csv("./excels/criteo/loss.csv")