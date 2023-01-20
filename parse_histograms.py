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

def main():
    # ---------------------------------------------------------------- # 
    # Parse the output metrics for single-mismatch substrates  
    # ---------------------------------------------------------------- #
    logrates = np.loadtxt('data/line-3-w6-v2-minusbind-single-logrates.tsv')
    probs = np.loadtxt('data/line-3-w6-v2-minusbind-single-probs.tsv')
    specs = np.loadtxt('data/line-3-w6-v2-minusbind-single-specs.tsv')
    cleave = np.loadtxt('data/line-3-w6-v2-minusbind-single-cleave.tsv')
    unbind = np.loadtxt('data/line-3-w6-v2-minusbind-single-unbind.tsv')
    rapid = np.loadtxt('data/line-3-w6-v2-minusbind-single-rapid.tsv')
    dead_dissoc = np.loadtxt('data/line-3-w6-v2-minusbind-single-deaddissoc.tsv')
    c_min, c_max = -6, 6
    cp_min, cp_max = -6, 6
    
    # Cleavage probabilities on perfect-match substrates
    activities = np.tile(probs[:, 0].reshape((probs.shape[0]), 1), 20)

    # Cleavage rates on perfect-match substrates 
    speeds = np.tile(cleave[:, 0].reshape((cleave.shape[0]), 1), 20)

    # Histogram of c = b/d over all sampled parameter vectors
    hist_c_total, hist_c_bin_edges = np.histogram(
        logrates[:, 0] - logrates[:, 1],
        bins=20,
        range=[c_min, c_max]
    )

    ######################################################################
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

    ######################################################################
    # ---------------------------------------------------------------- #
    # Count parameter vectors that yield high activity and low specificity
    # ---------------------------------------------------------------- #
    histograms, activity_x_bin_edges, spec_y_bin_edges = get_histogram_per_mismatch_2d(
        activities, specs, 20, indices=list(range(20)), xmin=0, ymin=0
    )
    high_activity_threshold = activity_x_bin_edges[-2]
    low_spec_threshold = spec_y_bin_edges[1]
    high_activity = (probs[:, 0] > high_activity_threshold)
    # Print the total number of parameter vectors, the number of parameter 
    # vectors with c > 1, and the number of parameter vectors in each bin 
    # of the histogram for c from 1 to 2
    outstr = 'All & [{},{}] & '.format(hist_c_bin_edges[10], hist_c_bin_edges[-1])
    outstr += ' & '.join('[{},{}]'.format(hist_c_bin_edges[i], hist_c_bin_edges[i+1]) for i in range(20))
    outstr += ' \\\\ \n'
    outstr += '{:d} & {:d} & '.format(hist_c_total.sum(), hist_c_total[10:].sum())
    outstr += ' & '.join(['{:d}'.format(int(x)) for x in hist_c_total])
    outstr += ' \\\\ \n'
    print(outstr)
    # For each mismatch position i, get the histogram of c over all parameter 
    # vectors that exhibit high activity and low specificity
    hist_c_HALS = np.zeros((20, 20), dtype=np.float64)    # Each row = mismatch position
    for i in range(20):
        low_spec = (specs[:, i] < low_spec_threshold)
        logrates_HALS = logrates[(high_activity & low_spec), :]
        c_HALS = logrates_HALS[:, 0] - logrates_HALS[:, 1]
        hist_c_HALS[i, :] = np.histogram(
            c_HALS,
            bins=hist_c_bin_edges
        )[0]
    outstr = ''
    for i in range(20):
        # Print the total number of HA-LS parameter vectors (for mismatch 
        # position i), the total number of HA-LS parameter vectors with 
        # c > 1, and the number of HA-LS parameter vectors in each bin of
        # the histogram for c from 0 to 2
        outstr += '{:d} & {:d} & '.format(
            int(hist_c_HALS[i, :].sum()), int(hist_c_HALS[i, 10:].sum())
        )
        outstr += ' & '.join([
            '{:d}'.format(int(x)) for x in hist_c_HALS[i, :]
        ])
        outstr += ' \\\\ \n'
    print(outstr)

    # ---------------------------------------------------------------- #
    # Plot how specific rapidity changes with mismatch position
    # ---------------------------------------------------------------- #
    histograms, spec_x_bin_edges, rapid_y_bin_edges = get_histogram_per_mismatch_2d(
        specs, rapid, 20, indices=list(range(20)), xmin=0
    )
    print(
        spec_x_bin_edges[0], spec_x_bin_edges[-1],
        rapid_y_bin_edges[0], rapid_y_bin_edges[-1]
    )

    # ---------------------------------------------------------------- #
    # Get distributions of b / d and b' / d' values for parameter vectors
    # that yield high specific rapidity for each bin along the specificity
    # axis
    # ---------------------------------------------------------------- #
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(11, 8))
    c_total = logrates[:, 0] - logrates[:, 1]        # c = b / d
    cp_total = logrates[:, 3] - logrates[:, 2]       # cp = b' / d'
    c_valid = [c_total]
    c_valid_max = [c_total.max()]
    c_valid_perc99 = [np.percentile(c_total, 99)]
    c_valid_perc95 = [np.percentile(c_total, 95)]
    cp_valid = [cp_total]
    cp_valid_min = [cp_total.min()]
    cp_valid_perc1 = [np.percentile(cp_total, 1)]
    cp_valid_perc5 = [np.percentile(cp_total, 5)]
    start_idx = 2
    for i in range(start_idx, 20):       # For each mismatch position ...
        c_valid_i = []
        cp_valid_i = []
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
                cp_valid_i.append(logrates_valid[k, 3] - logrates_valid[k, 2])
        c_valid.append(c_valid_i)
        c_valid_max.append(np.max(c_valid_i))
        c_valid_perc99.append(np.percentile(c_valid_i, 99))
        c_valid_perc95.append(np.percentile(c_valid_i, 95))
        cp_valid.append(cp_valid_i)
        cp_valid_min.append(np.min(cp_valid_i))
        cp_valid_perc1.append(np.percentile(cp_valid_i, 1))
        cp_valid_perc5.append(np.percentile(cp_valid_i, 5))
    sns.boxplot(
        data=c_valid, orient='v', width=0.7, showfliers=False, linewidth=2,
        ax=axes[0], whis=(5, 95)
    )
    sns.boxplot(
        data=cp_valid, orient='v', width=0.7, showfliers=False, linewidth=2,
        ax=axes[1], whis=(5, 95)
    )
    #axes[0].scatter(
    #    list(range(1, 21 - start_idx)), c_valid_max[1:], marker='X',
    #    color=sns.color_palette()[4], s=50, zorder=2
    #)
    #axes[0].scatter(
    #    list(range(1, 21 - start_idx)), c_valid_perc99[1:], marker='X',
    #    color=sns.color_palette()[3], s=50, zorder=2
    #)
    #axes[0].scatter(
    #    list(range(1, 21 - start_idx)), c_valid_perc95[1:], marker='X',
    #    color=sns.color_palette()[6], s=50, zorder=2
    #)
    #axes[1].scatter(
    #    list(range(1, 21 - start_idx)), cp_valid_min[1:], marker='X',
    #    color=sns.color_palette()[4], s=50, zorder=2 
    #)
    #axes[1].scatter(
    #    list(range(1, 21 - start_idx)), cp_valid_perc1[1:], marker='X',
    #    color=sns.color_palette()[3], s=50, zorder=2
    #)
    #axes[1].scatter(
    #    list(range(1, 21 - start_idx)), cp_valid_perc5[1:], marker='X',
    #    color=sns.color_palette()[6], s=50, zorder=2
    #)
    axes[0].patches[0].set_facecolor(sns.color_palette('deep')[1])
    axes[1].patches[0].set_facecolor(sns.color_palette('deep')[1])
    for j in range(1, 21 - start_idx):
        axes[0].patches[j].set_facecolor(sns.color_palette('deep')[0])
        axes[1].patches[j].set_facecolor(sns.color_palette('deep')[0])
    axes[0].set_xticklabels(['All\nvalid\nparameter\nvectors'] + [str(k) for k in range(start_idx, 20)])
    axes[1].set_xticklabels(['All\nvalid\nparameter\nvectors'] + [str(k) for k in range(start_idx, 20)])
    axes[0].set_ylim([c_min - 0.2, c_max + 0.2])
    axes[1].set_ylim([cp_min - 0.2, cp_max + 0.2])
    axes[0].set_ylabel(r"$\log_{10}(b/d)$")
    axes[1].set_ylabel(r"$\log_{10}(d'/b')$")
    plt.tight_layout()
    plt.savefig('plots/line-3-w6-v2-minusbind-single-highrapid-dpbp-boxplot.pdf')
    plt.close()

    # ---------------------------------------------------------------- #
    # Count parameter vectors that yield negative specific rapidity and 
    # have a small b / d ratio
    # ---------------------------------------------------------------- #
    high_activity = (probs[:, 0] > high_activity_threshold)
    outstr = 'All & [{},{}] & '.format(hist_c_bin_edges[10], hist_c_bin_edges[-1])
    outstr += ' & '.join('[{},{}]'.format(hist_c_bin_edges[i], hist_c_bin_edges[i+1]) for i in range(20))
    outstr += ' \\\\ \n'
    outstr = '{:d} & {:d} & '.format(hist_c_total.sum(), hist_c_total[:10].sum())
    outstr += ' & '.join(['{:d}'.format(int(x)) for x in hist_c_total])
    outstr += ' \\\\ \n'
    print(outstr)
    hist_c_valid = np.zeros((20, 20), dtype=np.float64)    # Each row = mismatch position
    for i in range(20):
        negative_rapid = (rapid[:, i] < 0)
        logrates_valid = logrates[negative_rapid, :]
        c_valid = logrates_valid[:, 0] - logrates_valid[:, 1]
        hist_c_valid[i, :] = np.histogram(
            c_valid,
            bins=hist_c_bin_edges
        )[0]
    outstr = ''
    for i in range(20):
        outstr += '{:d} & {:d} & '.format(
            int(hist_c_valid[i, :].sum()), int(hist_c_valid[i, :].sum())
        )
        outstr += ' & '.join([
            '{:d}'.format(int(x)) for x in hist_c_valid[i, :]
        ])
        outstr += ' \\\\ \n'
    print(outstr)

    # ---------------------------------------------------------------- #
    # Get distribution of b / d values for parameter vectors that yield 
    # negative specific rapidity 
    # ---------------------------------------------------------------- #
    fig = plt.figure(figsize=(11, 4))
    ax = plt.gca()
    c_total = logrates[:, 0] - logrates[:, 1]
    c_valid = [c_total]
    c_valid_max = [c_total.max()]
    for i in range(5):
        for j in range(4):
            k = 4 * i + j
            neg_rapid = (rapid[:, k] < 0)
            logrates_valid = logrates[neg_rapid, :]
            c_valid.append(logrates_valid[:, 0] - logrates_valid[:, 1])
            c_valid_max.append((logrates_valid[:, 0] - logrates_valid[:, 1]).max())
    sns.boxplot(
        data=c_valid, orient='v', width=0.7, showfliers=False, linewidth=2, ax=ax,
        whis=(5, 95)
    )
    #ax.scatter(
    #    list(range(1, 21)), c_valid_max[1:], marker='X', color=sns.color_palette()[4],
    #    s=50, zorder=2 
    #)
    ax.patches[0].set_facecolor(sns.color_palette('deep')[1])
    for i in range(1, 21):
        ax.patches[i].set_facecolor(sns.color_palette('deep')[0])
    ax.set_xticklabels(['All\nvalid\nparameter\nvectors'] + [str(k) for k in range(20)])
    ax.set_ylim([c_min - 0.2, c_max + 0.2])
    ax.set_ylabel(r'$\log_{10}(b/d)$')
    plt.tight_layout()
    plt.savefig('plots/line-3-w6-v2-minusbind-single-negrapid-bd-boxplot.pdf')
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

    # Determine the distribution of specificity values among low-specificity and 
    # high-dissociativity parameter vectors
    for i in range(2, 20):
        low_spec = (specs[:, i] < low_spec_threshold)
        high_dissoc = (dead_dissoc[:, i] > dissoc_thresholds[i])
        specs_valid = specs[low_spec & high_dissoc, i]
        specs_bins = [0, 1e-20, 1e-10, 1e-5, 1e-2, spec_x_bin_edges[1]]
        specs_valid_hist, _ = np.histogram(specs_valid, bins=specs_bins)
        print(specs_valid_hist)

    # Determine the intersection of high activity, low specificity, and 
    # high dissociativity parameter vectors for each mismatch position
    fig, axes = plt.subplot_mosaic(
        [['{}{}'.format(i, j) for j in range(4)] for i in range(5)],
        figsize=(12, 15)
    )
    axes['00'].set_axis_off()
    axes['01'].set_axis_off()
    high_activity = (probs[:, 0] > high_activity_threshold)
    for i in range(2, 20):
        j, k = i // 4, i % 4
        low_spec = (specs[:, i] < low_spec_threshold)
        high_dissoc = (dead_dissoc[:, i] > dissoc_thresholds[i])
        Abc = (high_activity & ~low_spec & ~high_dissoc).sum()
        aBc = (~high_activity & low_spec & ~high_dissoc).sum()
        ABc = (high_activity & low_spec & ~high_dissoc).sum()
        abC = (~high_activity & ~low_spec & high_dissoc).sum()
        AbC = (high_activity & ~low_spec & high_dissoc).sum()
        aBC = (~high_activity & low_spec & high_dissoc).sum()
        ABC = (high_activity & low_spec & high_dissoc).sum()
        print(
            i, high_activity.sum(), low_spec.sum(), high_dissoc.sum(),
            Abc, aBc, ABc, abC, AbC, aBC, ABC
        )
        v = venn3_circles(
            [Abc, aBc, ABc, abC, AbC, aBC, ABC], alpha=0.3, linewidth=0,
            ax=axes['{}{}'.format(j, k)]
        )
        v[0].set_color('red')
        v[0].set_edgecolor('red')
        v[1].set_color('blue')
        v[1].set_edgecolor('blue')
        v[2].set_color('green')
        v[2].set_edgecolor('green')
    plt.savefig('plots/line-3-w6-v2-minusbind-single-activity-spec-deaddissoc-venn.pdf')
    plt.close()

    # ----------------------------------------------------------------------- #
    # Plot how speed (on-target cleavage rate) changes with mismatch position
    # ----------------------------------------------------------------------- #
    histograms, speed_x_bin_edges, spec_y_bin_edges = get_histogram_per_mismatch_2d(
        speeds, specs, 20, indices=list(range(20)), xmin=0, ymin=0
    )
    print(
        speed_x_bin_edges[0], speed_x_bin_edges[-1],
        spec_y_bin_edges[0], spec_y_bin_edges[-1]
    )

    # For each histogram, identify the upper limit of the rightmost bin in the
    # second-to-bottommost column 
    speed_threshold_indices = {}
    for i in range(4, 20):
        spec_within_range = (specs[:, i] >= spec_y_bin_edges[1])
        speed_threshold_indices[i] = np.nonzero(
            speed_x_bin_edges > np.max(speeds[spec_within_range, i])
        )[0][0]
    speed_thresholds = {i: speed_x_bin_edges[speed_threshold_indices[i]] for i in range(4, 20)}
    print(speed_threshold_indices)
    print(speed_thresholds)

    # Determine the distribution of specificity values among low-specificity and 
    # high-speed parameter vectors
    for i in range(4, 20):
        low_spec = (specs[:, i] < low_spec_threshold)
        high_speed = (speeds[:, i] > speed_thresholds[i])
        specs_valid = specs[low_spec & high_speed, i]
        specs_bins = [0, 1e-20, 1e-10, 1e-5, 1e-2, spec_y_bin_edges[1]]
        specs_valid_hist, _ = np.histogram(specs_valid, bins=specs_bins)
        print(specs_valid_hist)

    """
    counts = np.zeros((18, 6), dtype=int)
    for i in range(2, 20):
        high_dissoc = (dead_dissoc[:, i] > dissoc_thresholds[i])
        low_specs = specs[high_dissoc, i]
        count = (low_specs >= 0.1).sum()
        counts[i - 2, 0] = count
        exponents = [-1, -5, -10, -20]
        for j in range(len(exponents)-1):
            count = ((low_specs < 10 ** exponents[j]) & (low_specs >= 10 ** exponents[j+1])).sum()
            counts[i - 2, j + 1] = count
        count = (low_specs < 1e-20).sum()
        counts[i - 2, 5] = count
    outstr = ''
    for i in range(2, 20):
        high_dissoc = (dead_dissoc[:, i] > dissoc_thresholds[i])
        low_specs = specs[high_dissoc, i]
        outstr += ' & '.join(['{:d}'.format(x) for x in counts[i - 2, :]])
        outstr += ' \\\\ \n'
        #outstr += ' & '.join([
        #    '({:#.3g}\\%)'.format(100 * x / low_specs.size) for x in counts[i - 2, :]
        #])
        #outstr += ' \\\\ \n'
    print(outstr)
    """
   
    """
    # ---------------------------------------------------------------- #
    # Count parameter vectors that yield low specificity and high dissociativity
    # and have a large b / d ratio 
    # ---------------------------------------------------------------- #
    hist_c_valid1 = np.zeros((20, 20), dtype=np.float64)    # (i, j) = mismatch at i-th position, j-th bin
    hist_c_valid2 = np.zeros((20, 20), dtype=np.float64)    # (i, j) = mismatch at i-th position, j-th bin
    hist_c_valid3 = np.zeros((20, 20), dtype=np.float64)    # (i, j) = mismatch at i-th position, j-th bin
    high_activity = (probs[:, 0] > high_activity_threshold)
    print(high_activity.sum())
    for i in range(2, 20):
        low_spec = (specs[:, i] < low_spec_threshold)
        high_dissoc = (dead_dissoc[:, i] >= y_bin_edges[dissoc_threshold_indices[i]])
        print(low_spec.sum(), high_dissoc.sum())
        logrates_valid1 = logrates[(high_dissoc & low_spec), :]
        logrates_valid2 = logrates[(high_dissoc & high_activity), :]
        logrates_valid3 = logrates[(high_dissoc & high_activity & low_spec), :]
        c_valid1 = logrates_valid1[:, 0] - logrates_valid1[:, 1]
        c_valid2 = logrates_valid2[:, 0] - logrates_valid2[:, 1]
        c_valid3 = logrates_valid3[:, 0] - logrates_valid3[:, 1]
        hist_c_valid1[i, :] = np.histogram(c_valid1, bins=20, range=[0, 12])[0]
        hist_c_valid2[i, :] = np.histogram(c_valid2, bins=20, range=[0, 12])[0]
        hist_c_valid3[i, :] = np.histogram(c_valid3, bins=20, range=[0, 12])[0]
    outstr = ''
    for i in range(20):
        outstr += '{:d} & {:d} & '.format(
            int(hist_c_valid1[i, :].sum()), int(hist_c_valid1[i, 10:].sum())
        )
        outstr += ' & '.join([
            '{:d}'.format(int(x)) for x in hist_c_valid1[i, 10:]
        ])
        outstr += ' \\\\ \n'
        outstr += '({:#.3g}\\%) & ({:#.3g}\\%) & '.format(
            100 * hist_c_valid1[i, :].sum() / hist_c_total.sum(),
            100 * hist_c_valid1[i, 10:].sum() / hist_c_total[10:].sum()
        )
        outstr += ' & '.join([
            '({:#.3g}\\%)'.format(100 * x / y)
            for x, y in zip(hist_c_valid1[i, 10:], hist_c_total[10:])
        ])
        outstr += ' \\\\ \n'
    print(outstr)
    outstr = ''
    for i in range(20):
        outstr += '{:d} & {:d} & '.format(
            int(hist_c_valid2[i, :].sum()), int(hist_c_valid2[i, 10:].sum())
        )
        outstr += ' & '.join([
            '{:d}'.format(int(x)) for x in hist_c_valid2[i, 10:]
        ])
        outstr += ' \\\\ \n'
        outstr += '({:#.3g}\\%) & ({:#.3g}\\%) & '.format(
            100 * hist_c_valid2[i, :].sum() / hist_c_total.sum(),
            100 * hist_c_valid2[i, 10:].sum() / hist_c_total[10:].sum()
        )
        outstr += ' & '.join([
            '({:#.3g}\\%)'.format(100 * x / y)
            for x, y in zip(hist_c_valid2[i, 10:], hist_c_total[10:])
        ])
        outstr += ' \\\\ \n'
    print(outstr)
    outstr = ''
    for i in range(20):
        outstr += '{:d} & {:d} & '.format(
            int(hist_c_valid3[i, :].sum()), int(hist_c_valid3[i, 10:].sum())
        )
        outstr += ' & '.join([
            '{:d}'.format(int(x)) for x in hist_c_valid3[i, 10:]
        ])
        outstr += ' \\\\ \n'
        outstr += '({:#.3g}\\%) & ({:#.3g}\\%) & '.format(
            100 * hist_c_valid3[i, :].sum() / hist_c_total.sum(),
            100 * hist_c_valid3[i, 10:].sum() / hist_c_total[10:].sum()
        )
        outstr += ' & '.join([
            '({:#.3g}\\%)'.format(100 * x / y)
            for x, y in zip(hist_c_valid3[i, 10:], hist_c_total[10:])
        ])
        outstr += ' \\\\ \n'
    print(outstr)
    """

    """
    # ---------------------------------------------------------------- # 
    # Parse the output metrics for distal-mismatch substrates  
    # ---------------------------------------------------------------- #
    probs = np.loadtxt('data/line-4-w8-minusbind-distal-probs.tsv')
    probs[:, 1:] = probs[:, 1:][:, ::-1]
    specs = np.loadtxt('data/line-4-w8-minusbind-distal-specs.tsv')[:, ::-1]
    rapid = np.loadtxt('data/line-4-w8-minusbind-distal-rapid.tsv')[:, ::-1]
    dead_dissoc = np.loadtxt('data/line-4-w8-minusbind-distal-dead-dissoc.tsv')[:, ::-1]
    live_dissoc = np.loadtxt('data/line-4-w8-minusbind-distal-live-dissoc.tsv')[:, ::-1]

    # Plot activity vs. specificity for all 20 mismatch positions 
    fig, axes = plt.subplot_mosaic(
        [['{}{}'.format(i, j) for j in range(4)] for i in range(5)],
        figsize=(12, 15)
    )
    _, x_bin_edges, y_bin_edges = plot_metrics_by_mismatch(
        activities, specs, 20,
        r'$\phi(\mathbf{u}^{\mathrm{P}})$',
        r'$\log_{10}(\phi(\mathbf{u}^{\mathrm{P}}) / \phi(\mathbf{u}^M))$',
        axes, list(range(20)), xmin=0, ymin=0,
        annotate_fmt=r'$M = [ {} ]$'
    )
    print(x_bin_edges[0], x_bin_edges[-1], y_bin_edges[0], y_bin_edges[-1])
    plt.tight_layout()
    plt.savefig('plots/line-4-w8-minusbind-distal-prob-by-mismatch-all.pdf')
    plt.close()

    # ---------------------------------------------------------------- #
    # Count parameter vectors that yield high activity and low specificity
    # and have a large b / d ratio 
    # ---------------------------------------------------------------- #
    high_activity = (probs[:, 0] > high_activity_threshold)
    hist_c_total = np.histogram(logrates[:, 0] - logrates[:, 1], bins=20, range=[0, 12])[0]
    outstr = '{:d} & '.format(hist_c_total[10:].sum())
    outstr += ' & '.join(['{:d}'.format(int(x)) for x in hist_c_total[10:]])
    outstr += ' \\\\ \n'
    print(outstr)
    hist_c_valid = np.zeros((20, 20), dtype=np.float64)    # Each row = mismatch position
    for i in range(20):
        low_spec = (specs[:, i] < low_spec_threshold)
        logrates_valid = logrates[(high_activity & low_spec), :]
        c_valid = logrates_valid[:, 0] - logrates_valid[:, 1]
        hist_c_valid[i, :] = np.histogram(c_valid, bins=20, range=[0, 12])[0]
    outstr = ''
    for i in range(20):
        outstr += '{:d} & {:d} & '.format(
            int(hist_c_valid[i, :].sum()), int(hist_c_valid[i, 10:].sum())
        )
        outstr += ' & '.join([
            '{:d}'.format(int(x)) for x in hist_c_valid[i, 10:]
        ])
        outstr += ' \\\\ \n'
        outstr += '({:#.3g}\\%) & ({:#.3g}\\%) & '.format(
            100 * hist_c_valid[i, :].sum() / hist_c_total.sum(),
            100 * hist_c_valid[i, 10:].sum() / hist_c_total[10:].sum()
        )
        outstr += ' & '.join([
            '({:#.3g}\\%)'.format(100 * x / y)
            for x, y in zip(hist_c_valid[i, 10:], hist_c_total[10:])
        ])
        outstr += ' \\\\ \n'
    print(outstr)

    # ---------------------------------------------------------------- #
    # Plot how specific rapidity changes with mismatch position
    # ---------------------------------------------------------------- #
    fig, axes = plt.subplot_mosaic(    # Plot all 20 subplots 
        [['{}{}'.format(i, j) for j in range(4)] for i in range(5)],
        figsize=(12, 15)
    )
    _, x_bin_edges, y_bin_edges = plot_metrics_by_mismatch(
        specs, rapid, 20,
        r'$\log_{10}(\phi(\mathbf{u}^{\mathrm{P}}) / \phi(\mathbf{u}^M))$',
        r'$\log_{10}(\sigma_{*}(\mathbf{u}^{\mathrm{P}}) / \sigma_{*}(\mathbf{u}^M))$',
        axes, list(range(20)), xmin=0,
        annotate_fmt=r'$M = [ {} ]$'
    )
    print(x_bin_edges[0], x_bin_edges[-1], y_bin_edges[0], y_bin_edges[-1])
    plt.tight_layout()
    plt.savefig('plots/line-4-w8-minusbind-distal-spec-vs-rapid-by-mismatch-all.pdf')
    plt.close()

    # ---------------------------------------------------------------- #
    # Plot how dead specific dissociativity changes with mismatch position
    # ---------------------------------------------------------------- #
    fig, axes = plt.subplot_mosaic(    # Plot all 20 subplots 
        [['{}{}'.format(i, j) for j in range(4)] for i in range(5)],
        figsize=(12, 15)
    )
    _, x_bin_edges, y_bin_edges = plot_metrics_by_mismatch(
        specs, dead_dissoc, 20,
        r'$\log_{10}(\phi(\mathbf{u}^{\mathrm{P}}) / \phi(\mathbf{u}^M))$',
        r'$\log_{10}(\sigma(\mathbf{u}^M) / \sigma(\mathbf{u}^{\mathrm{P}}))$',
        axes, list(range(20)), xmin=0, ymin=0,
        annotate_fmt=r'$M = [ {} ]$'
    )
    print(x_bin_edges[0], x_bin_edges[-1], y_bin_edges[0], y_bin_edges[-1])
    plt.tight_layout()
    plt.savefig('plots/line-4-w8-minusbind-distal-spec-vs-deaddissoc-by-mismatch-all.pdf')
    plt.close()
    """

#######################################################################
if __name__ == '__main__':
    main()
