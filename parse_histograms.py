"""
Authors:
    Kee-Myoung Nam

Last updated:
    1/20/2023
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3_circles
import seaborn as sns

###########################################################################
def get_histogram_per_mismatch_2d(xvals, yvals, nbins, indices=list(range(20)),
                                  xmin=None, xmax=None, ymin=None, ymax=None):
    # First identify the min/max x- and y-values over all mismatch positions
    if xmin is None:
        xmin = round(np.min(xvals))
    if xmax is None:
        xmax = round(np.max(xvals))
    if ymin is None:
        ymin = round(np.min(yvals))
    if ymax is None:
        ymax = round(np.max(yvals))

    # Maintain all 2-D histograms and their bin edges in 3-D arrays
    x_bin_edges = np.linspace(xmin, xmax, nbins + 1)
    y_bin_edges = np.linspace(ymin, ymax, nbins + 1)
    histograms = np.zeros((len(indices), nbins, nbins), dtype=np.float64) 

    # Compute the 2-D histograms
    for k, i in enumerate(indices):
        hist, _, _ = np.histogram2d(
            xvals[:, i], yvals[:, i], bins=nbins,
            range=[[xmin, xmax], [ymin, ymax]]
        )
        histograms[k, :, :] = hist
    
    return histograms, x_bin_edges, y_bin_edges

###########################################################################
def main():
    # ---------------------------------------------------------------- # 
    # Parse the output metrics for single-mismatch substrates  
    # ---------------------------------------------------------------- #
    logrates = np.loadtxt('data/line-3-w6-v2-minusbind-single-logrates-subset.tsv')
    probs = np.loadtxt('data/line-3-w6-v2-minusbind-single-probs-subset.tsv')
    specs = np.loadtxt('data/line-3-w6-v2-minusbind-single-specs-subset.tsv')
    cleave = np.loadtxt('data/line-3-w6-v2-minusbind-single-cleave-subset.tsv')
    unbind = np.loadtxt('data/line-3-w6-v2-minusbind-single-unbind-subset.tsv')
    rapid = np.loadtxt('data/line-3-w6-v2-minusbind-single-rapid-subset.tsv')
    dead_dissoc = np.loadtxt('data/line-3-w6-v2-minusbind-single-deaddissoc-subset.tsv')
    ratio_min = -6
    ratio_max = 6
    
    # Cleavage probabilities on perfect-match substrates
    activities = np.tile(probs[:, 0].reshape((probs.shape[0]), 1), 20)

    # Cleavage rates on perfect-match substrates 
    speeds = np.tile(cleave[:, 0].reshape((cleave.shape[0]), 1), 20)

    # Histogram of c = b/d over all sampled parameter vectors
    hist_c_total, hist_c_bin_edges = np.histogram(
        logrates[:, 0] - logrates[:, 1],
        bins=20,
        range=[ratio_min, ratio_max]
    )

    # ---------------------------------------------------------------- #
    # Plot how specific rapidity changes with mismatch position
    # ---------------------------------------------------------------- #
    histograms, spec_x_bin_edges, rapid_y_bin_edges = get_histogram_per_mismatch_2d(
        specs, rapid, 20, indices=list(range(20)), xmin=0
    )
    low_spec_threshold = spec_x_bin_edges[1] 
    print(
        spec_x_bin_edges[0], spec_x_bin_edges[-1],
        rapid_y_bin_edges[0], rapid_y_bin_edges[-1]
    )

    # ---------------------------------------------------------------- #
    # Get distributions of parameter ratios for parameter vectors that yield
    # high specific rapidity for each bin along the specificity axis
    # ---------------------------------------------------------------- #
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(11, 11))
    c_total = logrates[:, 0] - logrates[:, 1]        # c = b / d
    cp_total = logrates[:, 2] - logrates[:, 3]       # cp = b' / d'
    p_total = logrates[:, 2] - logrates[:, 1]        # p = b' / d
    q_total = logrates[:, 0] - logrates[:, 3]        # q = b / d'
    c_valid = [c_total]
    cp_valid = [cp_total]
    p_valid = [p_total]
    q_valid = [q_total]
    start_idx = 2
    for i in range(start_idx, 20):       # For each mismatch position ...
        c_valid_i = []
        cp_valid_i = []
        p_valid_i = []
        q_valid_i = []
        for j in range(20):   # ... and each bin along the x-axis ... 
            # Get indices of parameter vectors for top 1000 rapidity values 
            # in the j-th column of the i-th histogram
            in_column = (
                (specs[:, i] >= spec_x_bin_edges[j]) &
                (specs[:, i] < spec_x_bin_edges[j+1])
            )
            rapid_top1000_idx = np.argsort(rapid[in_column, i])[-1000:]
            logrates_valid = logrates[in_column, :][rapid_top1000_idx]
            for k in range(logrates_valid.shape[0]):
                c_valid_i.append(logrates_valid[k, 0] - logrates_valid[k, 1])
                cp_valid_i.append(logrates_valid[k, 2] - logrates_valid[k, 3])
                p_valid_i.append(logrates_valid[k, 2] - logrates_valid[k, 1])
                q_valid_i.append(logrates_valid[k, 0] - logrates_valid[k, 3])
        c_valid.append(c_valid_i)
        cp_valid.append(cp_valid_i)
        p_valid.append(p_valid_i)
        q_valid.append(q_valid_i)
    sns.boxplot(
        data=c_valid, orient='v', width=0.7, showfliers=False, linewidth=2,
        ax=axes[0], whis=(5, 95)
    )
    sns.boxplot(
        data=p_valid, orient='v', width=0.7, showfliers=False, linewidth=2,
        ax=axes[1], whis=(5, 95)
    )
    sns.boxplot(
        data=q_valid, orient='v', width=0.7, showfliers=False, linewidth=2,
        ax=axes[2], whis=(5, 95)
    )
    sns.boxplot(
        data=cp_valid, orient='v', width=0.7, showfliers=False, linewidth=2,
        ax=axes[3], whis=(5, 95)
    )
    for i in range(4):
        axes[i].patches[0].set_facecolor(sns.color_palette('deep')[1])
        for j in range(1, 21 - start_idx):
            axes[i].patches[j].set_facecolor(sns.color_palette('deep')[0])
        axes[i].set_xticklabels(['All'] + [str(k) for k in range(start_idx, 20)])
        axes[i].set_ylim([ratio_min - 0.2, ratio_max + 0.2])
    axes[0].set_ylabel(r"$\log_{10}(b/d)$")
    axes[1].set_ylabel(r"$\log_{10}(b'/d)$")
    axes[2].set_ylabel(r"$\log_{10}(b/d')$")
    axes[3].set_ylabel(r"$\log_{10}(b'/d')$")
    plt.tight_layout()
    plt.savefig('plots/line-3-w6-v2-minusbind-single-highrapid-boxplot.pdf')
    plt.close()

    # ---------------------------------------------------------------- #
    # Get distributions of parameter ratios for parameter vectors that yield
    # negative specific rapidity 
    # ---------------------------------------------------------------- #
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(11, 11))
    c_valid = [c_total]
    cp_valid = [cp_total]
    p_valid = [p_total]
    q_valid = [q_total]
    for i in range(20):
        neg_rapid = (rapid[:, i] < 0)
        logrates_valid = logrates[neg_rapid, :]
        c_valid.append(logrates_valid[:, 0] - logrates_valid[:, 1])
        cp_valid.append(logrates_valid[:, 2] - logrates_valid[:, 3])
        p_valid.append(logrates_valid[:, 2] - logrates_valid[:, 1])
        q_valid.append(logrates_valid[:, 0] - logrates_valid[:, 3])
    sns.boxplot(
        data=c_valid, orient='v', width=0.7, showfliers=False, linewidth=2,
        ax=axes[0], whis=(5, 95)
    )
    sns.boxplot(
        data=p_valid, orient='v', width=0.7, showfliers=False, linewidth=2,
        ax=axes[1], whis=(5, 95)
    )
    sns.boxplot(
        data=q_valid, orient='v', width=0.7, showfliers=False, linewidth=2,
        ax=axes[2], whis=(5, 95)
    )
    sns.boxplot(
        data=cp_valid, orient='v', width=0.7, showfliers=False, linewidth=2,
        ax=axes[3], whis=(5, 95)
    )
    for i in range(4):
        axes[i].patches[0].set_facecolor(sns.color_palette('deep')[1])
        for j in range(1, 21):
            axes[i].patches[j].set_facecolor(sns.color_palette('deep')[0])
        axes[i].set_xticklabels(['All'] + [str(k) for k in range(20)])
        axes[i].set_ylim([ratio_min - 0.2, ratio_max + 0.2])
    axes[0].set_ylabel(r"$\log_{10}(b/d)$")
    axes[1].set_ylabel(r"$\log_{10}(b'/d)$")
    axes[2].set_ylabel(r"$\log_{10}(b/d')$")
    axes[3].set_ylabel(r"$\log_{10}(b'/d')$")
    plt.tight_layout()
    plt.savefig('plots/line-3-w6-v2-minusbind-single-negrapid-boxplot.pdf')
    plt.close()

    # ---------------------------------------------------------------- #
    # Plot how dead specific dissociativity changes with mismatch position
    # ---------------------------------------------------------------- #
    histograms, spec_x_bin_edges, dissoc_y_bin_edges = get_histogram_per_mismatch_2d(
        specs, dead_dissoc, 20, indices=list(range(20)), xmin=0, ymin=0
    )
    print(
        spec_x_bin_edges[0], spec_x_bin_edges[-1],
        dissoc_y_bin_edges[0], dissoc_y_bin_edges[-1]
    )

    # For each histogram, identify the upper limit of the uppermost bin in the
    # second column 
    dissoc_threshold_indices = {}
    for i in range(2, 20):
        spec_within_range = (specs[:, i] >= spec_x_bin_edges[1])
        dissoc_threshold_indices[i] = np.nonzero(
            dissoc_y_bin_edges > np.max(dead_dissoc[spec_within_range, i])
        )[0][0]
    dissoc_thresholds = {i: dissoc_y_bin_edges[dissoc_threshold_indices[i]] for i in range(2, 20)}
    print(dissoc_threshold_indices)
    print(dissoc_thresholds)

    # ---------------------------------------------------------------- #
    # Get distributions of parameter ratios for parameter vectors that yield
    # high dissociativity and low specificity
    # ---------------------------------------------------------------- #
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(11, 11))
    c_valid = [c_total]
    cp_valid = [cp_total]
    p_valid = [p_total]
    q_valid = [q_total]
    start_idx = 2
    for i in range(start_idx, 20):       # For each mismatch position ...
        c_valid_i = []
        cp_valid_i = []
        p_valid_i = []
        q_valid_i = []
        low_spec = (specs[:, i] < low_spec_threshold)
        high_dissoc = (dead_dissoc[:, i] > dissoc_thresholds[i])
        logrates_valid = logrates[low_spec & high_dissoc, :]
        c_valid.append(logrates_valid[:, 0] - logrates_valid[:, 1])
        cp_valid.append(logrates_valid[:, 2] - logrates_valid[:, 3])
        p_valid.append(logrates_valid[:, 2] - logrates_valid[:, 1])
        q_valid.append(logrates_valid[:, 0] - logrates_valid[:, 3])
    sns.boxplot(
        data=c_valid, orient='v', width=0.7, showfliers=False, linewidth=2,
        ax=axes[0], whis=(5, 95)
    )
    sns.boxplot(
        data=p_valid, orient='v', width=0.7, showfliers=False, linewidth=2,
        ax=axes[1], whis=(5, 95)
    )
    sns.boxplot(
        data=q_valid, orient='v', width=0.7, showfliers=False, linewidth=2,
        ax=axes[2], whis=(5, 95)
    )
    sns.boxplot(
        data=cp_valid, orient='v', width=0.7, showfliers=False, linewidth=2,
        ax=axes[3], whis=(5, 95)
    )
    for i in range(4):
        axes[i].patches[0].set_facecolor(sns.color_palette('deep')[1])
        for j in range(1, 21 - start_idx):
            axes[i].patches[j].set_facecolor(sns.color_palette('deep')[0])
        axes[i].set_xticklabels(['All'] + [str(k) for k in range(start_idx, 20)])
        axes[i].set_ylim([ratio_min - 0.2, ratio_max + 0.2])
    axes[0].set_ylabel(r"$\log_{10}(b/d)$")
    axes[1].set_ylabel(r"$\log_{10}(b'/d)$")
    axes[2].set_ylabel(r"$\log_{10}(b/d')$")
    axes[3].set_ylabel(r"$\log_{10}(b'/d')$")
    plt.tight_layout()
    plt.savefig('plots/line-3-w6-v2-minusbind-single-highdissoc-boxplot.pdf')
    plt.close()

###########################################################################
if __name__ == '__main__':
    main()
