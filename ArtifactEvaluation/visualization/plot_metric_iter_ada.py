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


def get_data(fname):
    data = pd.read_csv(fname)
    return data['step'], data['value']


def plot_auc(dataset, cr):
    data = {}
    data_dir = osp.join(excel_dir, dataset)
    # full
    fullfile = osp.join(data_dir, f'full_auc.csv')
    hasfull = osp.exists(fullfile)
    if hasfull:
        stepdata, fulldata = get_data(fullfile)
        data['step'] = stepdata
        data['full'] = fulldata
        nstep = len(stepdata)
    # ada
    stepdata, adadata = get_data(osp.join(data_dir, f'ada{cr}_auc.csv'))
    if 'step' not in data:
        data['step'] = stepdata
        nstep = len(stepdata)
    data['ada'] = adadata[:nstep]
    # sketch
    _, cafedata = get_data(osp.join(data_dir, f'sketch{cr}_auc.csv'))
    data['sketch'] = cafedata[:nstep]

    plt.rc('font', family='Arial')
    plt.figure(figsize=(6, 4.5))
    plt.yscale('linear')
    plt.xscale('linear')
    plt.tick_params(labelsize=19)

    #plt.title('Examples of line chart',fontsize=20)
    plt.xlabel('Iterations', fontweight='bold', fontsize=24)
    plt.ylabel('Test AUC', fontweight='bold', fontsize=24)

    if hasfull:
        plt.plot(data['step'], data['full'], label='Ideal', linestyle='-', marker='*', color='black',
                 markersize=10, alpha=1, linewidth=2, markerfacecolor='none', markeredgewidth=2)
    plt.plot(data['step'], data['ada'], label='AdaEmbed', linestyle='-', marker='v', color='C2',
             markersize=11.5, alpha=1, linewidth=2, markerfacecolor='none', markeredgewidth=2)
    plt.plot(data['step'], data['sketch'], label='CAFE (ours)', linestyle='-', marker='o', color='C3',
             markersize=11.3, alpha=1, linewidth=2, markerfacecolor='none', markeredgewidth=2)

    ax = plt.gca()
    ax.xaxis.get_offset_text().set(size=19)

    plt.legend(loc='lower right', ncol=1, handlelength=3.3, prop={'size': 10})
    leg = ax.get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontweight='bold', fontsize=20)

    plt.grid(True, linestyle='--', axis='y', zorder=0)
    plt.grid(True, linestyle='--', axis='x', zorder=0)
    plt.tight_layout()

    plt.savefig(osp.join(png_dir, f'{dataset}{cr}_ada_auc_iter.png'))


def plot_loss(dataset, cr):
    data = {}
    data_dir = osp.join(excel_dir, dataset)
    # full
    fullfile = osp.join(data_dir, f'full_loss.csv')
    hasfull = osp.exists(fullfile)
    if hasfull:
        stepdata, fulldata = get_data(fullfile)
        data['step'] = stepdata
        data['full'] = fulldata
        nstep = len(stepdata)
    # ada
    stepdata, adadata = get_data(osp.join(data_dir, f'ada{cr}_loss.csv'))
    if 'step' not in data:
        data['step'] = stepdata
        nstep = len(stepdata)
    data['ada'] = adadata[:nstep]
    # sketch
    _, cafedata = get_data(osp.join(data_dir, f'sketch{cr}_loss.csv'))
    data['sketch'] = cafedata[:nstep]
    for ind, s in enumerate(data['step']):
        if s >= 1024:
            break
    for k, v in data.items():
        if k == 'step':
            data[k] = [ind // 2] + v[ind:].tolist()
        else:
            data[k] = [sum(v[:ind]) / ind] + v[ind:].tolist()

    plt.rc('font', family='Arial')
    plt.figure(figsize=(12, 4.5))
    plt.yscale('linear')
    plt.xscale('linear')
    plt.tick_params(labelsize=19)

    if dataset == 'criteotb':
        plt.ylim(0.10, 0.16)

    #plt.title('Examples of line chart',fontsize=20)
    plt.xlabel('Iterations', fontweight='bold', fontsize=24)
    plt.ylabel('Train Loss', fontweight='bold', fontsize=24)

    if hasfull:
        plt.plot(data['step'], data['full'], label='Ideal', linestyle='-', marker=None, color='black',
                 alpha=1, linewidth=1)
    plt.plot(data['step'], data['ada'], label='AdaEmbed', linestyle='-', marker=None, color='C2',
             alpha=1, linewidth=1)
    plt.plot(data['step'], data['sketch'], label='CAFE (ours)', linestyle='-', marker=None, color='C3',
             alpha=1, linewidth=1)

    ax = plt.gca()
    ax.xaxis.get_offset_text().set(size=19)

    plt.legend(loc='upper right', ncol=1, handlelength=3.3, prop={'size': 10})
    leg = ax.get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontweight='bold', fontsize=20)

    plt.grid(True, linestyle='--', axis='y', zorder=0)
    plt.grid(True, linestyle='--', axis='x', zorder=0)
    plt.tight_layout()

    plt.savefig(osp.join(png_dir, f'{dataset}{cr}_ada_loss_iter.png'))


if __name__ == '__main__':
    # plot_auc('criteo', 2)
    plot_auc('criteo', 5)
    # plot_auc('criteotb', 20)
    plot_auc('criteotb', 50)
    # plot_loss('criteo', 2)
    plot_loss('criteo', 5)
    # plot_loss('criteotb', 20)
    plot_loss('criteotb', 50)
