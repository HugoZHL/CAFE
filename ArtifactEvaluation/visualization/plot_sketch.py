#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp

work_dir = osp.split(osp.split(osp.abspath(__file__))[0])[0]
excel_dir = osp.join(work_dir, 'sketch_expr')
png_dir = osp.join(work_dir, 'pngs')
os.makedirs(png_dir, exist_ok=True)


def plot_recall():
    datafile = osp.join(excel_dir, 'sketch_mem_recall.csv')
    data = pd.read_csv(datafile, index_col=0)
    plt.rc('font', family='Arial')
    plt.figure(figsize=(6, 4.5))
    plt.yscale('linear')
    plt.xscale('linear')
    plt.tick_params(labelsize=19)

    plt.xlabel('Memory (KB)', fontweight='bold', fontsize=24)
    plt.ylabel('Recall', fontweight='bold', fontsize=24)

    plt.plot(data.index, data['4'], label='c=4', linestyle='-', marker='h',
             markersize=7, alpha=1, linewidth=2, markerfacecolor='none', markeredgewidth=2)
    plt.plot(data.index, data['8'], label='c=8', linestyle='-', marker='h',
             markersize=7, alpha=1, linewidth=2, markerfacecolor='none', markeredgewidth=2)
    plt.plot(data.index, data['16'], label='c=16', linestyle='-', marker='h',
             markersize=7, alpha=1, linewidth=2, markerfacecolor='none', markeredgewidth=2)
    plt.plot(data.index, data['32'], label='c=32', linestyle='-', marker='h',
             markersize=7, alpha=1, linewidth=2, markerfacecolor='none', markeredgewidth=2)

    plt.legend(loc='best', ncol=1, handlelength=3.3, prop={'size': 10})
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontweight='bold', fontsize=20)

    plt.grid(True, linestyle='--', axis='y', zorder=0)
    plt.grid(True, linestyle='--', axis='x', zorder=0)
    plt.tight_layout()

    plt.savefig(osp.join(png_dir, f'sketch_mem_recall.png'))


def bar_throughput():
    datafile = osp.join(excel_dir, f'throughput.csv')
    data = pd.read_csv(datafile, index_col=0)
    data = data.transpose()
    ngroup = len(data.index)
    ntype = len(data.columns)

    plt.rc('font', family='Arial')

    plt.figure(figsize=(6, 4.5))
    plt.yscale('linear')
    plt.tick_params(labelsize=19)

    plt.xlabel(u'', fontweight='bold', fontsize=24)
    plt.ylabel(u'Throughput (M/s)', fontweight='bold',
               fontsize=24)

    x = np.arange(ngroup)
    total_width, n = 0.8, ntype
    width = total_width / n
    x = x - (total_width - width) / ngroup

    name_list = data.index

    plt.bar(x, data[4] / 1e6, width=width, label='c=4', edgecolor='black')
    plt.bar(x + width, data[8] / 1e6, width=width,
            label='c=8', edgecolor='black')
    plt.bar(x + 2*width, data[16] / 1e6, width=width,
            label='c=16', edgecolor='black')
    plt.bar(x + 3*width, data[32] / 1e6, width=width,
            label='c=32', edgecolor='black')
    plt.bar(x + width*1.5, [0] * ngroup, tick_label=name_list)

    plt.legend(loc='best', prop={'size': 16})
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontweight='bold', fontsize=19)
    plt.xticks(size=24, weight="bold")

    plt.grid(True, linestyle='--', axis='y')
    plt.grid(True, linestyle='--', axis='x')
    plt.tight_layout()

    plt.savefig(osp.join(png_dir, 'sketch_throughput.png'))


def plot_dynamic(cr):
    datafile = osp.join(excel_dir, f'time_recall_{cr}.csv')
    data = pd.read_csv(datafile, index_col=0)
    plt.rc('font', family='Arial')
    plt.figure(figsize=(6, 4.5))
    plt.yscale('linear')
    plt.xscale('linear')
    plt.tick_params(labelsize=19)

    plt.xlabel('Days', fontweight='bold', fontsize=24)
    plt.ylabel('Recall', fontweight='bold', fontsize=24)
    plt.ylim(bottom=0.87, top=0.96)
    xdata = np.arange(1, 13) / 2

    plt.plot(xdata, data["Sliding-window Topk"], label='Sliding-window Top-k', linestyle='-', marker='h',
             markersize=7, alpha=1, linewidth=2, markerfacecolor='none', markeredgewidth=2)
    plt.plot(xdata, data["Up-to-date Topk"], label='Up-to-date Top-k', linestyle='-', marker='h',
             markersize=7, alpha=1, linewidth=2, markerfacecolor='none', markeredgewidth=2)

    plt.legend(loc='lower center', ncol=1, handlelength=3.3, prop={'size': 10})
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontweight='bold', fontsize=20)

    plt.grid(True, linestyle='--', axis='y', zorder=0)
    plt.grid(True, linestyle='--', axis='x', zorder=0)
    plt.tight_layout()

    plt.savefig(osp.join(png_dir, f'time_recall_{cr}.png'))


if __name__ == '__main__':
    plot_recall()
    bar_throughput()
    plot_dynamic(100)
    plot_dynamic(1000)
