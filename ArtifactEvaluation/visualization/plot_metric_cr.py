#!/usr/bin/env python
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
from board_reader import get_auc_cr, get_loss_cr

work_dir = osp.split(osp.split(osp.abspath(__file__))[0])[0]
png_dir = osp.join(work_dir, 'pngs')
os.makedirs(png_dir, exist_ok=True)


def plot_figure(dataset, metric, methods, crs):
    if metric not in ['loss', 'auc']:
        raise AssertionError(f'Metric {metric} not supported.')
    isauc = metric == 'auc'
    if isauc:
        func = get_auc_cr
    else:
        func = get_loss_cr
    xlabels = np.array([int(1 / cr) for cr in crs], dtype=int)
    plt.rc('font', family='Arial')
    plt.figure(figsize=(6, 4.5))
    plt.yscale('linear')
    plt.xscale('log')
    plt.tick_params(labelsize=19)

    plt.xlabel('Compression Ratio', fontweight='bold', fontsize=24)
    if isauc:
        ylabel = 'Test AUC'
    else:
        ylabel = 'Train Loss'
    plt.ylabel(ylabel, fontweight='bold', fontsize=24)
    loc = 'best'
    plotting_mde = 'mde' in methods and len(crs) == 3
    if plotting_mde:
        # tune mde figure layout
        if dataset == 'criteo':
            plt.xlim(left=1.6667, right=12)
            plt.xticks([2, 5, 10], ['2', '5', '10'])
        else:
            plt.xlim(left=8.3333, right=60)
            if metric == 'auc':
                loc = 'lower left'
                plt.ylim(bottom=0.7935, top=0.803)
            else:
                loc = 'upper left'
                plt.ylim(bottom=0.12275, top=0.12475)
            plt.xticks([10, 20, 50], ['10', '20', '50'])
        plt.minorticks_off()

    if 'full' in methods:
        if isauc and dataset.endswith('criteotb'):
            # hard-coded from dlrm repo and paper: https://github.com/facebookresearch/dlrm
            # we don't run full criteotb since it's too large
            full_result = 0.8025
        else:
            full_result = func(dataset, 'full')
        if not np.isnan(full_result):
            plt.axhline(y=full_result, color='black', linestyle='dashed')
            if not plotting_mde:
                if isauc:
                    if dataset == 'criteo':
                        yoffset = full_result - 0.005
                    elif dataset.endswith('criteotb'):
                        yoffset = full_result - 0.004
                    else:
                        yoffset = full_result - 0.013
                else:
                    yoffset = full_result + 0.001
                plt.text(3000, yoffset, 'Ideal', fontsize=20, fontweight='bold')

    if 'hash' in methods:
        data = np.array([func(dataset, f'hash{cr}') for cr in crs])
        mask = ~np.isnan(data)
        plt.plot(xlabels[mask], data[mask], label='Hash', linestyle='-', marker='D', color='C0',
                markersize=10, alpha=1, linewidth=2, markerfacecolor='none', markeredgewidth=2)
    if 'qr' in methods:
        data = np.array([func(dataset, f'qr{cr}') for cr in crs])
        mask = ~np.isnan(data)
        plt.plot(xlabels[mask], data[mask], label='Q-R Trick', linestyle='-', marker='s', color='C1',
                markersize=10, alpha=1, linewidth=2, markerfacecolor='none', markeredgewidth=2)
    if 'mde' in methods:
        data = np.array([func(dataset, f'mde{cr}') for cr in crs])
        mask = ~np.isnan(data)
        plt.plot(xlabels[mask], data[mask], label='MDE', linestyle='-', marker='h', color='C5',
                markersize=12, alpha=1, linewidth=2, markerfacecolor='none', markeredgewidth=2)
    if 'ada' in methods:
        data = np.array([func(dataset, f'ada{cr}') for cr in crs])
        mask = ~np.isnan(data)
        plt.plot(xlabels[mask], data[mask], label='AdaEmbed', linestyle='-', marker='v', color='C2',
                markersize=11.5, alpha=1, linewidth=2, markerfacecolor='none', markeredgewidth=2)
    if 'off' in methods:
        data = np.array([func(dataset, f'off{cr}') for cr in crs])
        mask = ~np.isnan(data)
        plt.plot(xlabels[mask], data[mask], label='Offline', linestyle='-', marker='^', color='C4',
             markersize=11.5, alpha=1, linewidth=2, markerfacecolor='none', markeredgewidth=2)
    if 'cafe' in methods:
        data = np.array([func(dataset, f'cafe{cr}') for cr in crs])
        mask = ~np.isnan(data)
        plt.plot(xlabels[mask], data[mask], label='CAFE (ours)', linestyle='-', marker='o', color='C3',
                markersize=11.3, alpha=1, linewidth=2, markerfacecolor='none', markeredgewidth=2)

    plt.legend(loc=loc, ncol=1, handlelength=3.3, prop={'size': 10})
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontweight='bold', fontsize=20)

    plt.grid(True, linestyle='--', axis='y', zorder=0)
    plt.grid(True, linestyle='--', axis='x', zorder=0)
    plt.tight_layout()

    suffix = ''.join([m[0] for m in methods])
    plt.savefig(osp.join(png_dir, f'{dataset}_{metric}_cr_{suffix}.png'))


if __name__ == '__main__':
    # it's ok to directly pass all crs, since those non-exist will not be plotted
    all_crs = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    methods = ['full', 'hash', 'qr', 'ada', 'cafe']
    no_full_methods = ['hash', 'qr', 'ada', 'cafe']
    plot_figure('criteo', 'auc', methods, all_crs)
    plot_figure('criteo', 'loss', methods, all_crs)
    plot_figure('kdd12', 'auc', methods, all_crs)
    plot_figure('avazu', 'loss', methods, all_crs)
    plot_figure('criteotb', 'auc', methods, all_crs)
    plot_figure('criteotb', 'loss', methods, all_crs)

    # plot wdl and dcn
    crs = [0.05, 0.01, 0.002, 0.0005, 0.0001]
    plot_figure('wdl_criteotb', 'auc', no_full_methods, crs)
    plot_figure('wdl_criteotb', 'loss', no_full_methods, crs)
    plot_figure('wdl_criteotb', 'auc', no_full_methods, crs)
    plot_figure('wdl_criteotb', 'loss', no_full_methods, crs)

    # plot mde
    plot_figure('criteo', 'auc', ['full', 'hash', 'mde', 'cafe'], [0.5, 0.2, 0.1])
    plot_figure('criteo', 'loss', ['full', 'hash', 'mde', 'cafe'], [0.5, 0.2, 0.1])
    plot_figure('criteotb', 'auc', ['full', 'hash', 'mde', 'cafe'], [0.1, 0.05, 0.02])
    plot_figure('criteotb', 'loss', ['full', 'hash', 'mde', 'cafe'], [0.1, 0.05, 0.02])

    # plot offline
    plot_figure('criteo', 'auc', ['full', 'off', 'cafe'], [0.1, 0.01, 0.001, 0.0001])

    # plot criteotb-1/3
    plot_figure('criteotb13', 'auc', no_full_methods, [0.1, 0.05, 0.02])
    plot_figure('criteotb13', 'loss', no_full_methods, [0.1, 0.05, 0.02])
