import pandas as pd
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import os.path as osp

board_dir = osp.join(osp.split(osp.split(osp.abspath(__file__))[0])[0], 'board')


def load_from_board(log_dir):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    # scalar_data = ea.scalars
    # print(scalar_data.Keys())
    # output: ['Train/Loss', 'recall', 'precision', 'f1', 'ap', 'roc_auc', 'Test/Acc']
    loss = ea.scalars.Items('Train/Loss')
    auc = ea.scalars.Items('roc_auc')
    pd.DataFrame(loss).to_csv(osp.join(log_dir, 'loss_iter.csv'))
    pd.DataFrame(auc).to_csv(osp.join(log_dir, 'auc_iter.csv'))
    lst = 0
    avr_loss = 0
    for x in loss:
        avr_loss += (x.step - lst) * x.value
        lst = x.step
    avr_loss /= lst
    # auc_cr = max([a.value for a in auc])
    ## align the auc point
    auc_cr = auc[-2].value
    loss_cr = avr_loss
    metrics = {'loss': loss_cr, 'auc': auc_cr}
    pd.DataFrame([metrics]).to_csv(osp.join(log_dir, 'metrics.csv'), index=False)


def get_log_dir(dataset, task_name, non_exist_ok=False):
    log_dir = osp.join(board_dir, dataset, task_name)
    if not osp.exists(log_dir):
        if non_exist_ok:
            return None
        raise AssertionError(f'Log dir {log_dir} not exists!')
    return log_dir


def get_metric_cr(dataset, task_name, metric):
    log_dir = get_log_dir(dataset, task_name, non_exist_ok=True)
    if log_dir is None:
        return np.nan
    if metric not in ['loss', 'auc']:
        raise AssertionError(f'Not support metric: {metric} now.')
    csv_file = osp.join(log_dir, 'metrics.csv')
    if not osp.exists(csv_file):
        load_from_board(log_dir)
    metrics = pd.read_csv(csv_file)
    return metrics[metric][0]


def get_loss_cr(dataset, task_name):
    return get_metric_cr(dataset, task_name, 'loss')


def get_auc_cr(dataset, task_name):
    return get_metric_cr(dataset, task_name, 'auc')


def get_loss_iter(dataset, task_name):
    log_dir = get_log_dir(dataset, task_name)
    csv_file = osp.join(log_dir, 'loss_iter.csv')
    if not osp.exists(csv_file):
        load_from_board(log_dir)
    losses = pd.read_csv(csv_file)
    return losses


def get_auc_iter(dataset, task_name):
    log_dir = get_log_dir(dataset, task_name)
    csv_file = osp.join(log_dir, 'auc_iter.csv')
    if not osp.exists(csv_file):
        load_from_board(log_dir)
    aucs = pd.read_csv(csv_file)
    return aucs
