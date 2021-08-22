import json
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import math
import json
from scipy.stats import ttest_ind
# from ppo.a2c_ppo_acktr.utils import CompressPDF

font = {'size': 20}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 3
matplotlib.rcParams['ps.fonttype'] = 3
plt.rcParams['figure.constrained_layout.use'] = True

colors = [[0, 0.447, 0.7410], [0.85, 0.325, 0.098],  [0.466, 0.674, 0.188], [0.929, 0.694, 0.125],
          [0.494, 0.1844, 0.556],[0.3010, 0.745, 0.933], [137/255,145/255,145/255],
          [0.466, 0.674, 0.8], [0.929, 0.04, 0.125],
          [0.3010, 0.245, 0.33], [0.635, 0.078, 0.184], [0.35, 0.78, 0.504]]

RESULTS_PATH = '/home/ahmed/Documents/eta-optimality/eta_otpimality/results'
SAVE_PATH = '/home/ahmed/Documents/eta-optimality/eta_otpimality/results/plots/'

LINE = 'mean'
ERR = 'std'
DPI = 30
N_SEEDS = None
N_EPOCHS = None
LINEWIDTH = 3
MARKERSIZE = 0.3
ALPHA = 0.3
ALPHA_TEST = 0.05
MARKERS = ['o', 'v', 's', 'P', 'D', 'X', "*", 'v', 's', 'p', 'P', '1']
FREQ = 1
LAST_EP = 300
# line, err_min, err_plus = get_stat_func(line=LINE, err=ERR)
# COMPRESSOR = CompressPDF(4)


def setup_figure(xlabel=None, ylabel=None, xlim=None, ylim=None):
    fig = plt.figure(figsize=(22, 15), frameon=False)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_linewidth(6)
    ax.spines['right'].set_linewidth(6)
    ax.spines['bottom'].set_linewidth(6)
    ax.spines['left'].set_linewidth(6)
    ax.tick_params(width=10, direction='in', length=20, labelsize='55')
    artists = ()
    if xlabel:
        xlab = plt.xlabel(xlabel)
        artists += (xlab,)
    if ylabel:
        ylab = plt.ylabel(ylabel)
        artists += (ylab,)
    if ylim:
        plt.ylim(ylim)
    if xlim:
        plt.xlim(xlim)
    plt.grid()
    return artists, ax

def setup_n_figs(n, xlabels=None, ylabels=None, xlims=None, ylims=None):
    fig, axs = plt.subplots(n, 1, figsize=(22, 15), frameon=False)
    axs = axs.ravel()
    artists = ()
    for i_ax, ax in enumerate(axs):
        ax.spines['top'].set_linewidth(3)
        ax.spines['right'].set_linewidth(3)
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['left'].set_linewidth(3)
        ax.tick_params(width=7, direction='in', length=15, labelsize='55', zorder=10)
        if xlabels[i_ax]:
            xlab = ax.set_xlabel(xlabels[i_ax])
            artists += (xlab,)
        if ylabels[i_ax]:
            ylab = ax.set_ylabel(ylabels[i_ax])
            artists += (ylab,)
        if ylims[i_ax]:
            ax.set_ylim(ylims[i_ax])
        if xlims[i_ax]:
            ax.set_xlim(xlims[i_ax])
    return artists, axs

def save_fig(path, artists):
    plt.savefig(os.path.join(path), bbox_extra_artists=artists, bbox_inches='tight', dpi=DPI)
    plt.close('all')


def check_length_and_seeds(experiment_path):
    conditions = os.listdir(experiment_path)
    # check max_length and nb seeds
    max_len = 0
    max_seeds = 0
    min_len = 1e6
    min_seeds = 1e6

    for cond in conditions:
        cond_path = experiment_path + cond + '/'
        list_runs = sorted(os.listdir(cond_path))
        if len(list_runs) > max_seeds:
            max_seeds = len(list_runs)
        if len(list_runs) < min_seeds:
            min_seeds = len(list_runs)
        for run in list_runs:
            try:
                run_path = cond_path + run + '/'
                data_run = pd.read_csv(run_path + 'progress.csv')
                nb_epochs = len(data_run)
                if nb_epochs > max_len:
                    max_len = nb_epochs
                if nb_epochs < min_len:
                    min_len = nb_epochs
            except:
                pass
    return max_len, max_seeds, min_len, min_seeds


def plot_sr_av(max_len, experiment_path, folders):

    artists, ax = setup_figure(
        xlabel='million steps',
        ylabel='average reward',
        xlim=[-0.1, 1],
        ylim=[-900, 15900])

    for i, folder in enumerate(folders):
        condition_path = experiment_path + folder + '/'
        list_runs = sorted(os.listdir(condition_path))
        x_eps = np.arange(0, max_len, FREQ) * 2048 / 1000000
        T = max_len
        REW = np.zeros([len(list_runs), T])
        for i_run, run in enumerate(list_runs):
            run_path = condition_path + run + '/'
            data_run = pd.read_csv(run_path + 'progress.csv')
            REW[i_run, :] = data_run['mean_reward'][:T]

        mean_rew = np.mean(REW, axis=0)
        std_rew = np.std(REW, axis=0)
        plt.plot(x_eps, mean_rew, color=colors[i], marker=MARKERS[i], markersize=MARKERSIZE, linewidth=LINEWIDTH)
        plt.fill_between(x_eps, mean_rew - std_rew, mean_rew + std_rew, color=colors[i], alpha=ALPHA)
    leg = plt.legend(['{}'.format(folder) for folder in folders],
                     loc='upper center',
                     bbox_to_anchor=(0.5, 1.45),
                     ncol=3,
                     fancybox=True,
                     shadow=True,
                     prop={'size': 20, 'weight': 'bold'},
                     markerscale=1)
    artists += (leg,)
    save_fig(path=SAVE_PATH + '_sr.pdf', artists=artists)


if __name__ == '__main__':
    print('\n\tPlotting')
    # if PLOT == 'init_study':
    experiment_path = RESULTS_PATH + '/'

    plot_sr_av(480, experiment_path, ['ppo', 'uniform_uniform', 'uniform_poisson_10', 'uniform_poisson_20',
                                      'uniform_poisson_30', 'uniform_geometric_05', 'uniform_geometric_07'])



