import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path as osp
cur_dir = osp.split(osp.split(osp.abspath(__file__))[0])[0]
png_dir = osp.join(cur_dir, 'pngs')
excel_dir = osp.join(cur_dir, 'excels')
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
    # plot top percent in 1000 cr
    # yvalues = ['loo', 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    # results = div100([78.545, 79.508, 79.517, 79.531, 79.487, 79.387, 79.321])
    data = pd.read_csv(osp.join(excel_dir, 'hotrate.csv'))
    yvalues = data.index
    results = div100(data['auc'])
    bar(yvalues, results, 'Hot Percentage', (0.78, 0.80),
        'toppercent.png', 'white', 0.5, div100([78.0, 78.5, 79.0, 79.5, 80.0]))

    # plot threshold in 1000 cr
    # yvalues = [100, 300, 500, 700, 900][::-1]
    # results = div100([79.461, 79.488, 79.531, 79.504, 79.499][::-1])
    data = pd.read_csv(osp.join(excel_dir, 'threshold.csv'))
    yvalues = data.index
    results = div100(data['auc'])
    bar(yvalues, results, 'Threshold', (0.794, 0.796), 'threshold.png',
        'white', 0.5, div100([79.4, 79.45, 79.5, 79.55, 79.6]))

    # plot decay in 1000 cr
    # yvalues = ['0.90', '0.95', '0.98', '0.99', 'w/o'][::-1]
    # results = div100([79.392, 79.515, 79.531, 79.517, 79.518][::-1])
    data = pd.read_csv(osp.join(excel_dir, 'decay.csv'))
    yvalues = data.index
    results = div100(data['auc'])
    bar(yvalues, results, 'Decay', (0.793, 0.796),
        'decay.png', 'white', 0.5, div100([79.3, 79.4, 79.5, 79.6]))

    # # others in 1000 cr
    # # yvalues = ['CAFE', 'Field', 'w/o \nNorm', 'Freq'][::-1]
    # # results = [79.531, 76.547, 79.526, 79.504][::-1]
    # yvalues = ['CAFE', 'Field', 'Freq'][::-1]
    # results = div100([79.531, 76.547, 79.504][::-1])
    # bar(yvalues, results, 'Design', (0.76, 0.81), 'others.png',
    #     'white', 0.5, div100([76, 77, 78, 79, 80, 81]))
