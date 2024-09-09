#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp

cur_dir = osp.split(osp.split(osp.abspath(__file__))[0])[0]
png_dir = osp.join(cur_dir, 'pngs')
excel_dir = osp.join(cur_dir, 'excels')
os.makedirs(png_dir, exist_ok=True)


def bar_latency():
    datafile = osp.join(excel_dir, f'throughput.csv')
    data = pd.read_csv(datafile)
    ngroup = len(data.index)
    ntype = len(data.columns)
    columns = data.columns
    data.loc['Train', columns] = 1e3 * 2048 / data.loc['Train', columns]
    data.loc['Test', columns] = 1e3 * 16384 / data.loc['Test', columns]

    plt.rc('font', family='Arial')

    plt.figure(figsize=(12, 4.5))
    plt.yscale('linear')
    plt.tick_params(labelsize=19)

    plt.xlabel(u'', fontweight='bold', fontsize=24)  # 设置x轴，并设定字号大小
    plt.ylabel(u'Latency (ms)', fontweight='bold',
               fontsize=24)  # 设置y轴，并设定字号大小

    x = np.arange(ngroup)  # 总共有几组，就设置成几，我们这里有三组，所以设置为3
    total_width, n = 0.8, ntype    # 有多少个类型，只需更改n即可，比如这里我们对比了四个，那么就把n设成4
    width = total_width / n
    x = x - (total_width - width) / ngroup

    name_list = data.index

    plt.bar(x, data['Hash'], width=width, color='C0',
            label='Hash', edgecolor='black')
    plt.bar(x + width, data['Q-R Trick'], width=width, color='C1',
            label='Q-R Trick', edgecolor='black')
    plt.bar(x + 2*width, data['MDE'], width=width, color='C5',
            label='MDE', edgecolor='black')
    plt.bar(x + 3*width, data['AdaEmbed'], width=width, color='C2',
            label='AdaEmbed', edgecolor='black')  # ada freq
    plt.bar(x + 4*width, data['CAFE(ours)'], width=width, color='C3',
            label='CAFE (ours)', edgecolor='black')  # sketch freq
    plt.bar(x + width*2, [0] * ngroup, tick_label=name_list)

    plt.legend(ncol=2, loc='best', prop={'size': 16})
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontweight='bold', fontsize=20)
    plt.xticks(size=24, weight="bold")

    plt.grid(True, linestyle='--', axis='y')
    plt.grid(True, linestyle='--', axis='x')
    plt.tight_layout()

    plt.savefig(osp.join(png_dir, 'latency.png'))


if __name__ == '__main__':
    bar_latency()
