#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp

work_dir = osp.split(osp.split(osp.abspath(__file__))[0])[0]
png_dir = osp.join(work_dir, 'pngs')
os.makedirs(png_dir, exist_ok=True)


def read_latency():
    methods = ['hash', 'qr', 'mde', 'ada', 'cafe']
    latencies = pd.DataFrame(columns=['train', 'test'], index=methods)
    lat_dir = osp.join(work_dir, 'board/latency')
    for met in methods:
        cur_lat = pd.read_csv(osp.join(lat_dir, met, 'latency.csv'))
        latencies.loc[met] = cur_lat.iloc[0]
    return latencies


def bar_latency(data):
    ngroup = len(data.columns)
    ntype = len(data.index)

    plt.rc('font', family='Arial')

    plt.figure(figsize=(12, 4.5))
    plt.yscale('linear')
    plt.tick_params(labelsize=19)

    plt.xlabel(u'', fontweight='bold', fontsize=24)
    plt.ylabel(u'Latency (ms)', fontweight='bold',
               fontsize=24)

    x = np.arange(ngroup)
    total_width, n = 0.8, ntype
    width = total_width / n
    x = x - (total_width - width) / ngroup

    name_list = data.columns

    plt.bar(x, data.loc['hash'], width=width, color='C0',
            label='Hash', edgecolor='black')
    plt.bar(x + width, data.loc['qr'], width=width, color='C1',
            label='Q-R Trick', edgecolor='black')
    plt.bar(x + 2*width, data.loc['mde'], width=width, color='C5',
            label='MDE', edgecolor='black')
    plt.bar(x + 3*width, data.loc['ada'], width=width, color='C2',
            label='AdaEmbed', edgecolor='black')
    plt.bar(x + 4*width, data.loc['cafe'], width=width, color='C3',
            label='CAFE (ours)', edgecolor='black')
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


def bar_throughput(data):
    ngroup = len(data.columns)
    ntype = len(data.index)
    data['train'] = 2048 / data['train']
    data['test'] = 16384 / data['test']

    plt.rc('font', family='Arial')

    plt.figure(figsize=(12, 4.5))
    plt.yscale('linear')
    plt.tick_params(labelsize=19)

    plt.xlabel(u'', fontweight='bold', fontsize=24)
    plt.ylabel(u'Throughput (K/s)', fontweight='bold',
               fontsize=24)

    x = np.arange(ngroup)
    total_width, n = 0.8, ntype
    width = total_width / n
    x = x - (total_width - width) / ngroup

    name_list = data.columns

    plt.bar(x, data.loc['hash'] / 1e3, width=width, color='C0',
            label='Hash', edgecolor='black')
    plt.bar(x + width, data.loc['qr'] / 1e3, width=width, color='C1',
            label='Q-R Trick', edgecolor='black')
    plt.bar(x + 2*width, data.loc['mde'] / 1e3, width=width, color='C5',
            label='MDE', edgecolor='black')
    plt.bar(x + 3*width, data.loc['ada'] / 1e3, width=width, color='C2',
            label='AdaEmbed', edgecolor='black')  # ada freq
    plt.bar(x + 4*width, data.loc['cafe'] / 1e3, width=width, color='C3',
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

    plt.savefig(osp.join(png_dir, 'throughput.png'))


if __name__ == '__main__':
    latencies = read_latency()
    bar_latency(latencies)
    bar_throughput(latencies)
