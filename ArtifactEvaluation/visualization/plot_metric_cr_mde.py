#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path as osp

cur_dir = osp.split(osp.split(osp.abspath(__file__))[0])[0]
png_dir = osp.join(cur_dir, 'pngs')
excel_dir = osp.join(cur_dir, 'excels')
os.makedirs(png_dir, exist_ok=True)


# currently hard coded results of non-compress models
full_loss = {
    'avazu': 0.3851184168384841,
    'criteo': 0.452088838,
}
full_auc = {
    'avazu': 0.74704,
    'criteotb': 0.8025,
    'criteo': 0.80212,
    'kdd12': 0.81725,
}


def plot_cr(dataset, isauc):
    if isauc:
        suffix = 'auc'
        full_result = full_auc.get(dataset, None)
    else:
        suffix = 'loss'
        full_result = full_loss.get(dataset, None)
    datafile = osp.join(excel_dir, f'{dataset}/{suffix}.csv')
    data = pd.read_csv(datafile, nrows=3)
    xlabels = data.index
    plt.rc('font', family='Arial')
    plt.figure(figsize=(6, 4.5))
    plt.yscale('linear')
    plt.xscale('log')
    plt.tick_params(labelsize=19)

    #plt.title('Examples of line chart',fontsize=20)
    plt.xlabel('Compression Ratio', fontweight='bold', fontsize=24)
    if isauc:
        ylabel = 'Test AUC'
    else:
        ylabel = 'Train Loss'
    plt.ylabel(ylabel, fontweight='bold', fontsize=24)
    if dataset == 'criteo':
        plt.xlim(left=1.6667, right=12)
        plt.xticks([2, 5, 10], ['2', '5', '10'])
    else:
        plt.xlim(left=8.3333, right=60)
        if suffix == 'auc':
            plt.ylim(bottom=0.7935, top=0.803)
        else:
            plt.ylim(bottom=0.12275, top=0.12475)
        plt.xticks([10, 20, 50], ['10', '20', '50'])
    plt.minorticks_off()

    if full_result is not None:
        plt.axhline(y=full_result, color='black', linestyle='dashed')
        # if isauc:
        #     if dataset == 'criteo':
        #         yoffset = full_result - 0.001
        #     elif dataset == 'criteotb':
        #         yoffset = full_result - 0.001
        # else:
        #     yoffset = full_result + 0.0001
        # pos = 35 if dataset == 'criteotb' else 8
        # plt.text(pos, yoffset, 'Ideal', fontsize=20, fontweight='bold')

    plt.plot(xlabels, data['hash'], label='Hash', linestyle='-', marker='D', color='C0',
             markersize=10, alpha=1, linewidth=2, markerfacecolor='none', markeredgewidth=2)
    # plt.plot(xlabels, data['qr'], label='Q-R Trick', linestyle='-', marker='s', color='C1',
    #          markersize=10, alpha=1, linewidth=2, markerfacecolor='none', markeredgewidth=2)
    # plt.plot(xlabels, data['ada'], label='AdaEmbed', linestyle='-', marker='v', color='C2',
    #          markersize=11.5, alpha=1, linewidth=2, markerfacecolor='none', markeredgewidth=2)
    plt.plot(xlabels, data['mde'], label='MDE', linestyle='-', marker='h', color='C5',
             markersize=12, alpha=1, linewidth=2, markerfacecolor='none', markeredgewidth=2)
    plt.plot(xlabels, data['sketch'], label='CAFE (ours)', linestyle='-', marker='o', color='C3',
             markersize=11.3, alpha=1, linewidth=2, markerfacecolor='none', markeredgewidth=2)

    loc = 'best'
    if dataset == 'criteotb':
        if suffix == 'auc':
            loc = 'lower left'
        else:
            loc = 'upper left'
    plt.legend(loc=loc, ncol=1, handlelength=3.3, prop={'size': 10})
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontweight='bold', fontsize=20)

    plt.grid(True, linestyle='--', axis='y', zorder=0)
    plt.grid(True, linestyle='--', axis='x', zorder=0)
    plt.tight_layout()

    plt.savefig(osp.join(png_dir, f'{dataset}_mde_{suffix}_cr.png'))


if __name__ == '__main__':
    plot_cr('criteo', True)
    plot_cr('criteo', False)
    plot_cr('criteotb', True)
    plot_cr('criteotb', False)
