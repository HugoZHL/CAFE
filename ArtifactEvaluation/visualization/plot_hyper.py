#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
from board_reader import get_auc_cr

work_dir = osp.split(osp.split(osp.abspath(__file__))[0])[0]
png_dir = osp.join(work_dir, 'pngs')
os.makedirs(png_dir, exist_ok=True)


def bar(yvalues, results, ylabel, xlim, fname, color, height, xticks=None):
    yvalues = [str(xl) for xl in yvalues]
    plt.rc('font', family='Arial')
    plt.figure(figsize=(6, 4.5))
    ax = plt.gca()
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.5)
    plt.xscale('linear')
    plt.tick_params(labelsize=19)
    plt.grid(axis='x', linestyle='--', zorder=0)
    plt.barh(yvalues, results, color=color, edgecolor='black',
             height=height, hatch='x', zorder=2, linewidth=2.5)
    plt.yticks(yvalues)
    plt.xticks(xticks)
    plt.xlim(xlim)
    interval = 0.12 * (xlim[1] - xlim[0])
    for a, b in zip(yvalues, results):
        plt.text(b + interval, a, ('%.5f' %
                 b)[1:], ha='center', va='center', fontsize=19)
    if ylabel is not None:
        plt.ylabel(ylabel, fontweight='bold', fontsize=24)
    plt.xlabel('Test AUC', fontweight='bold', fontsize=24)
    plt.tight_layout()
    plt.savefig(osp.join(png_dir, fname))
    plt.close()


def div100(x):
    return [xx / 100 for xx in x]


if __name__ == '__main__':
    baseline = get_auc_cr('criteo', 'cafe0.001')

    # plot top percent in 1000 cr
    yvalues = ['loo', 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    results = np.array([get_auc_cr('sensitivity', f'hot_percentage{hp}') for hp in [0.00001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    if np.isnan(results).sum() != 1:
        raise AssertionError("Should have a nan to be replaced by baseline.")
    results[np.isnan(results)] = baseline
    bar(yvalues, results, 'Hot Percentage', (0.78, 0.80),
        'toppercent.png', 'white', 0.5, [0.78, 0.785, 0.79, 0.795, 0.8])

    # plot threshold in 1000 cr
    yvalues = [100, 300, 500, 700, 900][::-1]
    results = np.array([get_auc_cr('sensitivity', f'threshold{thres}') for thres in yvalues])
    if np.isnan(results).sum() != 1:
        raise AssertionError("Should have a nan to be replaced by baseline.")
    results[np.isnan(results)] = baseline
    bar(yvalues, results, 'Threshold', (0.794, 0.796), 'threshold.png',
        'white', 0.5, [0.794, 0.7945, 0.795, 0.7955, 0.796])

    # plot decay in 1000 cr
    yvalues = ['0.90', '0.95', '0.98', '0.99', 'w/o'][::-1]
    results = np.array([get_auc_cr('sensitivity', f'decay{dec}') for dec in [1, 0.99, 0.98, 0.95, 0.9]])
    if np.isnan(results).sum() != 1:
        raise AssertionError("Should have a nan to be replaced by baseline.")
    results[np.isnan(results)] = baseline
    bar(yvalues, results, 'Decay', (0.793, 0.796),
        'decay.png', 'white', 0.5, [0.793, 0.794, 0.795, 0.796])

    # # # others in 1000 cr
    # # # yvalues = ['CAFE', 'Field', 'w/o \nNorm', 'Freq'][::-1]
    # # # results = [79.531, 76.547, 79.526, 79.504][::-1]
    # # yvalues = ['CAFE', 'Field', 'Freq'][::-1]
    # # results = div100([79.531, 76.547, 79.504][::-1])
    # # bar(yvalues, results, 'Design', (0.76, 0.81), 'others.png',
    # #     'white', 0.5, div100([76, 77, 78, 79, 80, 81]))
