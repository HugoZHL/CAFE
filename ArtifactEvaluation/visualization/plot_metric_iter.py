#!/usr/bin/env python
# coding=utf-8

import matplotlib.pyplot as plt
import os
import os.path as osp
from board_reader import get_auc_iter, get_loss_iter

work_dir = osp.split(osp.split(osp.abspath(__file__))[0])[0]
png_dir = osp.join(work_dir, 'pngs')
os.makedirs(png_dir, exist_ok=True)


def get_loss_data(dataset, task_name):
    iters = get_loss_iter(dataset, task_name)
    return iters['step'], iters['value']


def get_auc_data(dataset, task_name):
    iters = get_auc_iter(dataset, task_name)
    return iters['step'], iters['value']


def plot_auc(dataset, cr, methods):
    data = {}
    # full; we don't run full criteotb, since it's too large
    if 'full' in methods:
        stepdata, fulldata = get_auc_data(dataset, 'full')
        data['step'] = stepdata[:-1]
        data['full'] = fulldata[:-1]
        nstep = len(data['step'])
    # hash
    if 'hash' in methods:
        stepdata, hashdata = get_auc_data(dataset, f'hash{cr}')
        if 'step' not in data:
            data['step'] = stepdata[:-1]
            nstep = len(data['step'])
        data['hash'] = hashdata[:nstep]
    # qr
    if 'qr' in methods:
        stepdata, qrdata = get_auc_data(dataset, f'qr{cr}')
        if 'step' not in data:
            data['step'] = stepdata[:-1]
            nstep = len(data['step'])
        data['qr'] = qrdata[:nstep]
    # ada
    if 'ada' in methods:
        stepdata, cafedata = get_auc_data(dataset, f'ada{cr}')
        if 'step' not in data:
            data['step'] = stepdata[:-1]
            nstep = len(data['step'])
        data['ada'] = cafedata[:nstep]
    # cafe
    if 'cafe' in methods:
        stepdata, cafedata = get_auc_data(dataset, f'cafe{cr}')
        if 'step' not in data:
            data['step'] = stepdata[:-1]
            nstep = len(data['step'])
        data['cafe'] = cafedata[:nstep]

    plt.rc('font', family='Arial')
    plt.figure(figsize=(6, 4.5))
    plt.yscale('linear')
    plt.xscale('linear')
    plt.tick_params(labelsize=19)

    plt.xlabel('Iterations', fontweight='bold', fontsize=24)
    plt.ylabel('Test AUC', fontweight='bold', fontsize=24)

    if 'full' in methods:
        plt.plot(data['step'], data['full'], label='Ideal', linestyle='-', marker='*', color='black',
                 markersize=10, alpha=1, linewidth=2, markerfacecolor='none', markeredgewidth=2)
    if 'hash' in methods:
        plt.plot(data['step'], data['hash'], label='Hash', linestyle='-', marker='D', color='C0',
                markersize=10, alpha=1, linewidth=2, markerfacecolor='none', markeredgewidth=2)
    if 'qr' in methods:
        plt.plot(data['step'], data['qr'], label='Q-R Trick', linestyle='-', marker='s', color='C1',
                markersize=10, alpha=1, linewidth=2, markerfacecolor='none', markeredgewidth=2)
    if 'ada' in methods:
        plt.plot(data['step'], data['ada'], label='AdaEmbed', linestyle='-', marker='v', color='C2',
                markersize=11.5, alpha=1, linewidth=2, markerfacecolor='none', markeredgewidth=2)
    if 'cafe' in methods:
        plt.plot(data['step'], data['cafe'], label='CAFE (ours)', linestyle='-', marker='o', color='C3',
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

    suffix = ''.join([m[0] for m in methods])
    plt.savefig(osp.join(png_dir, f'{dataset}{cr}_auc_iter_{suffix}.png'))


def plot_loss(dataset, cr, methods):
    data = {}
    # full; we don't run full criteotb, since it's too large
    if 'full' in methods:
        stepdata, fulldata = get_loss_data(dataset, 'full')
        data['step'] = stepdata
        data['full'] = fulldata
        nstep = len(stepdata)
    # hash
    if 'hash' in methods:
        stepdata, hashdata = get_loss_data(dataset, f'hash{cr}')
        if 'step' not in data:
            data['step'] = stepdata
            nstep = len(stepdata)
        data['hash'] = hashdata[:nstep]
    # qr
    if 'qr' in methods:
        stepdata, qrdata = get_loss_data(dataset, f'qr{cr}')
        if 'step' not in data:
            data['step'] = stepdata
            nstep = len(stepdata)
        data['qr'] = qrdata[:nstep]
    # ada
    if 'ada' in methods:
        stepdata, cafedata = get_loss_data(dataset, f'ada{cr}')
        if 'step' not in data:
            data['step'] = stepdata
            nstep = len(stepdata)
        data['ada'] = cafedata[:nstep]
    # cafe
    if 'cafe' in methods:
        stepdata, cafedata = get_loss_data(dataset, f'cafe{cr}')
        if 'step' not in data:
            data['step'] = stepdata
            nstep = len(stepdata)
        data['cafe'] = cafedata[:nstep]
    for ind, s in enumerate(data['step']):
        if s >= 1024:
            # remove early too high losses to improve visualization
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

    plt.xlabel('Iterations', fontweight='bold', fontsize=24)
    plt.ylabel('Train Loss', fontweight='bold', fontsize=24)

    if 'full' in methods:
        plt.plot(data['step'], data['full'], label='Ideal', linestyle='-', marker=None, color='black',
                alpha=1, linewidth=1)
    if 'hash' in methods:
        plt.plot(data['step'], data['hash'], label='Hash', linestyle='-', marker=None, color='C0',
                alpha=1, linewidth=1)
    if 'qr' in methods:
        plt.plot(data['step'], data['qr'], label='Q-R Trick', linestyle='-', marker=None, color='C1',
                alpha=1, linewidth=1)
    if 'ada' in methods:
        plt.plot(data['step'], data['ada'], label='AdaEmbed', linestyle='-', marker=None, color='C2',
                alpha=1, linewidth=1)
    if 'cafe' in methods:
        plt.plot(data['step'], data['cafe'], label='CAFE (ours)', linestyle='-', marker=None, color='C3',
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

    suffix = ''.join([m[0] for m in methods])
    plt.savefig(osp.join(png_dir, f'{dataset}{cr}_loss_iter_{suffix}.png'))


if __name__ == '__main__':
    plot_auc('criteo', 0.01, methods=['full', 'hash', 'qr', 'cafe'])
    plot_loss('criteo', 0.01, methods=['full', 'hash', 'qr', 'cafe'])
    plot_auc('criteo', 0.2, methods=['full', 'ada', 'cafe'])
    plot_loss('criteo', 0.2, methods=['full', 'ada', 'cafe'])

    plot_loss('avazu', 0.2, methods=['full', 'hash', 'qr', 'ada', 'cafe'])

    # we don't have full for criteotb, since the embedding table is too large
    plot_auc('criteotb', 0.01, methods=['hash', 'qr', 'cafe'])
    plot_loss('criteotb', 0.01, methods=['hash', 'qr', 'cafe'])
    plot_auc('criteotb', 0.02, methods=['ada', 'cafe'])
    plot_loss('criteotb', 0.02, methods=['ada', 'cafe'])
