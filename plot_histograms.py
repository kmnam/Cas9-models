"""
Authors:
    Kee-Myoung Nam

Last updated:
    1/12/2023
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpmath import mp
mp.dps = 100

def logsumexp(logx):
    """
    Log-sum-exp function for list of mpmath.mpf objects. 
    """
    logmax = max(logx)
    return logmax + mp.log(mp.fsum([mp.exp(a - logmax) for a in logx]))

def main():
    # ---------------------------------------------------------------- # 
    # Parse the output metrics for single-mismatch substrates  
    # ---------------------------------------------------------------- #
    logrates = np.loadtxt('data/line-2-w2-minusbind-single-logrates.tsv')
    probs = np.loadtxt('data/line-2-w2-minusbind-single-probs.tsv')
    specs = np.loadtxt('data/line-2-w2-minusbind-single-specs.tsv')
    cleave = np.loadtxt('data/line-2-w2-minusbind-single-cleave.tsv')
    unbind = np.loadtxt('data/line-2-w2-minusbind-single-unbind.tsv')
    rapid = np.loadtxt('data/line-2-w2-minusbind-single-rapid.tsv')
    dead_dissoc = np.loadtxt('data/line-2-w2-minusbind-single-deaddissoc.tsv')

    # Cleavage probabilities on perfect-match substrates
    activities = np.tile(probs[:, 0].reshape((probs.shape[0]), 1), 20)

    # Cleavage rates on perfect-match substrates 
    speeds = np.log10(np.tile(cleave[:, 0].reshape((cleave.shape[0]), 1), 20))

    ######################################################################
    def plot_metrics_by_mismatch_2d(xvals, yvals, nbins, xlabel, ylabel, axes,
                                    indices=list(range(20)), xmin=None, xmax=None,
                                    ymin=None, ymax=None, labelsize=12,
                                    ticklabelsize=8, cbar_ticklabelsize=8,
                                    annotate_fmt=r'$M = \{{ {} \}}$'):
        # First identify the min/max x- and y-values over all mismatch positions
        if xmin is None:
            xmin = round(np.min(xvals))
        if xmax is None:
            xmax = round(np.max(xvals))
        if ymin is None:
            ymin = round(np.min(yvals))
        if ymax is None:
            ymax = round(np.max(yvals))

        # Maintain all plotted 2-D histograms and their bin edges
        x_bin_edges = np.linspace(xmin, xmax, nbins + 1)
        y_bin_edges = np.linspace(ymin, ymax, nbins + 1)
        histograms = np.zeros((len(indices), nbins, nbins), dtype=np.float64) 

        # Re-compute and plot the 2-D histograms with the maximum frequency  
        ax_indices = ['{}{}'.format(i, j) for i in range(5) for j in range(4)][:len(indices)]
        max_row = str((len(indices) - 1) // 4)
        for k, (i, j) in enumerate(zip(indices, ax_indices)):
            sns.histplot(
                x=xvals[:, i], y=yvals[:, i], ax=axes[j], bins=nbins,
                binrange=[[xmin, xmax], [ymin, ymax]], stat='probability',
                cbar=True
            )
            hist, _, _ = np.histogram2d(
                xvals[:, i], yvals[:, i], bins=nbins,
                range=[[xmin, xmax], [ymin, ymax]]
            )
            histograms[k, :, :] = hist
            if j.startswith(max_row):
                axes[j].set_xlabel(xlabel, size=labelsize)
            if j.endswith('0'):
                axes[j].set_ylabel(ylabel, size=labelsize)
            axes[j].tick_params(axis='both', labelsize=ticklabelsize)

            # Fix colorbar tick labels so that they increase in increments
            # of 0.1, 0.05, or 0.01 (depending on the maximum value)
            _, cbar_max = plt.gcf().axes[-1].get_ylim()
            if cbar_max > 0.2:
                new_cbar_ticks = [0]
                while new_cbar_ticks[-1] < cbar_max:
                    new_cbar_ticks.append(new_cbar_ticks[-1] + 0.1)
            elif cbar_max > 0.1:
                new_cbar_ticks = [0]
                while new_cbar_ticks[-1] < cbar_max:
                    new_cbar_ticks.append(new_cbar_ticks[-1] + 0.05)
            elif cbar_max > 0.02:
                new_cbar_ticks = [0]
                while new_cbar_ticks[-1] < cbar_max:
                    new_cbar_ticks.append(new_cbar_ticks[-1] + 0.01)
            else:
                new_cbar_ticks = plt.gcf().axes[-1].get_yticks()
            if new_cbar_ticks[-1] > cbar_max:
                new_cbar_ticks = new_cbar_ticks[:-1]
            plt.gcf().axes[-1].set_yticks(new_cbar_ticks)
            plt.gcf().axes[-1].tick_params(labelsize=cbar_ticklabelsize)
            
            # Add padding to the upper side of the y-axis and add an 
            # annotation on the top-right
            ax_ymin, ax_ymax = axes[j].get_ylim()
            axes[j].set_ylim(
                bottom=ax_ymin, top=(ax_ymin + 1.1 * (ax_ymax - ax_ymin))
            )
            axes[j].annotate(
                annotate_fmt.format(str(i)),
                xy=(0.97, 0.97),
                xycoords='axes fraction',
                horizontalalignment='right',
                verticalalignment='top',
                size=9
            )
        
        return histograms, x_bin_edges, y_bin_edges

    ######################################################################
    # ---------------------------------------------------------------- #
    # Plot how cleavage probability depends on mismatch position ...
    # ---------------------------------------------------------------- #
    # First plot cleavage probability and specificity for 500 handpicked
    # parameter combinations ... 
    n = 500
    length = 20
    fig, axes = plt.subplot_mosaic(
        [['A0', 'A0', 'A1', 'A1']] + [
            ['{}{}'.format(i, j) for j in range(4)] for i in range(2)
        ],
        figsize=(12, 8)
    )
    for i in range(n):     # Plot cleavage probabilities as a function of mismatch position 
        axes['A0'].plot(
            list(range(20)), probs[i, 1:], c=(0.9, 0.9, 0.9), marker=None,
            zorder=0
        )
    for i in range(n):     # Plot cleavage specificities as a function of mismatch position 
        axes['A1'].plot(
            list(range(20)), np.power(10.0, -specs[i, :]), c=(0.9, 0.9, 0.9),
            marker=None, zorder=0
        )
    indices = [            # Handpicked plots to be highlighted in different colors 
        134,  # blue
        400,  # orange 
        768,  # green 
        284,  # red
        534,  # purple
        230,  # brown (update to yellow)
        307   # pink
    ]
    palette = sns.color_palette('colorblind', len(indices))
    palette[-2] = sns.color_palette('colorblind', 9)[8]    # Replace brown with yellow
    for i, j in enumerate(indices):
        axes['A0'].plot(
            list(range(20)), probs[j, 1:], c=palette[i], marker=None, zorder=1,
            linewidth=3
        )
        axes['A1'].plot(
            list(range(20)), np.power(10.0, -specs[j, :]), c=palette[i],
            marker=None, zorder=1, linewidth=3
        )
    axes['A0'].set_xlim([-0.5, 19.5])
    axes['A0'].set_xlabel(r'$m$', size=10)
    axes['A0'].set_xticks(list(range(20)))
    axes['A0'].set_xticklabels([str(i) for i in range(20)], size=8)
    axes['A0'].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes['A0'].set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
    axes['A0'].set_ylabel(r'$\phi(\mathbf{u}^{\{m\}})$', size=10)
    axes['A1'].set_xlim([-0.5, 19.5])
    axes['A1'].set_xlabel(r'$m$', size=10)
    axes['A1'].set_xticks(list(range(20)))
    axes['A1'].set_xticklabels([str(i) for i in range(20)], size=8)
    axes['A1'].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes['A1'].set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
    axes['A1'].set_ylabel(r'$\phi(\mathbf{u}^{\{m\}}) \,/\, \phi(\mathbf{u}^{\mathrm{P}})$', size=10)

    # Then plot the histograms for mismatch positions 0, 2, 4, 6, 13, 15, 17, 19 ... 
    _, x_bin_edges, y_bin_edges = plot_metrics_by_mismatch_2d(
        activities, specs, 20,
        r'$\phi(\mathbf{u}^{\mathrm{P}})$',
        r'$\log_{10}(\phi(\mathbf{u}^{\mathrm{P}}) / \phi(\mathbf{u}^M))$',
        axes, indices=[0, 2, 4, 6, 13, 15, 17, 19], xmin=0, xmax=1, ymin=0,
        labelsize=10
    )
    #print(x_bin_edges[0], x_bin_edges[-1], y_bin_edges[0], y_bin_edges[-1])
    plt.subplots_adjust(
        left=0.1,
        right=0.9,
        bottom=0.18,
        top=0.9,
        hspace=0.32,
        wspace=0.3
    )
    plt.savefig('plots/line-2-w2-minusbind-single-prob-by-mismatch.pdf')
    plt.close()

    # Then plot the histograms for all 20 mismatch positions ... 
    fig, axes = plt.subplot_mosaic(
        [['{}{}'.format(i, j) for j in range(4)] for i in range(5)],
        figsize=(12, 15)
    )
    _, x_bin_edges, y_bin_edges = plot_metrics_by_mismatch_2d(
        activities, specs, 20,
        r'$\phi(\mathbf{u}^{\mathrm{P}})$',
        r'$\log_{10}(\phi(\mathbf{u}^{\mathrm{P}}) / \phi(\mathbf{u}^M))$',
        axes, indices=list(range(20)), xmin=0, ymin=0
    )
    #print(x_bin_edges[0], x_bin_edges[-1], y_bin_edges[0], y_bin_edges[-1])
    plt.tight_layout()
    plt.savefig('plots/line-2-w2-minusbind-single-prob-by-mismatch-all.pdf')
    plt.close()

    # ------------------------------------------------------------- #
    # Plot how speed and specificity changes with mismatch position
    # ------------------------------------------------------------- #
    # First plot the histograms for mismatch positions 0, 6, 12, 19 ... 
    fig, axes = plt.subplot_mosaic(
        [['{}{}'.format(i, j) for j in range(4)] for i in range(1)],
        figsize=(13, 3)
    )
    _, x_bin_edges, y_bin_edges = plot_metrics_by_mismatch_2d(
        speeds, specs, 20,
        r'$\log_{10}(\sigma_{*}(\mathbf{u}^{\mathrm{P}}))$',
        r'$\log_{10}(\phi(\mathbf{u}^{\mathrm{P}}) / \phi(\mathbf{u}^M))$',
        axes, indices=[0, 6, 12, 19], ymin=0
    )
    #print(x_bin_edges[0], x_bin_edges[-1], y_bin_edges[0], y_bin_edges[-1])
    plt.subplots_adjust(
        left=0.1,
        right=0.9,
        bottom=0.18,
        top=0.9,
        hspace=0.3,
        wspace=0.3
    )
    plt.savefig('plots/line-2-w2-minusbind-single-speed-vs-spec-by-mismatch.pdf')
    plt.close()

    # Then plot the histograms for all 20 mismatch positions ... 
    fig, axes = plt.subplot_mosaic(
        [['{}{}'.format(i, j) for j in range(4)] for i in range(5)],
        figsize=(12, 15)
    )
    histograms, x_bin_edges, y_bin_edges = plot_metrics_by_mismatch_2d(
        speeds, specs, 20,
        r'$\log_{10}(\sigma_{*}(\mathbf{u}^{\mathrm{P}}))$',
        r'$\log_{10}(\phi(\mathbf{u}^{\mathrm{P}}) / \phi(\mathbf{u}^M))$',
        axes, indices=list(range(20)), ymin=0
    )
    #print(x_bin_edges[0], x_bin_edges[-1], y_bin_edges[0], y_bin_edges[-1])
    plt.tight_layout()
    plt.savefig('plots/line-2-w2-minusbind-single-speed-vs-spec-by-mismatch-all.pdf')
    plt.close()

    # ------------------------------------------------------------------- #
    # Plot how speed and specific rapidity changes with mismatch position
    # ------------------------------------------------------------------- #
    # First plot the histograms for mismatch positions 0, 6, 12, 19 ... 
    fig, axes = plt.subplot_mosaic(
        [['{}{}'.format(i, j) for j in range(4)] for i in range(1)],
        figsize=(13, 3)
    )
    _, x_bin_edges, y_bin_edges = plot_metrics_by_mismatch_2d(
        speeds, rapid, 20,
        r'$\log_{10}(\sigma_{*}(\mathbf{u}^{\mathrm{P}}))$',
        r'$\log_{10}(\sigma_{*}(\mathbf{u}^{\mathrm{P}}) / \sigma_{*}(\mathbf{u}^M))$',
        axes, indices=[0, 6, 12, 19], ymin=0
    )
    #print(x_bin_edges[0], x_bin_edges[-1], y_bin_edges[0], y_bin_edges[-1])
    plt.subplots_adjust(
        left=0.1,
        right=0.9,
        bottom=0.18,
        top=0.9,
        hspace=0.3,
        wspace=0.3
    )
    plt.savefig('plots/line-2-w2-minusbind-single-speed-vs-rapid-by-mismatch.pdf')
    plt.close()

    # Then plot the histograms for all 20 mismatch positions ... 
    fig, axes = plt.subplot_mosaic(
        [['{}{}'.format(i, j) for j in range(4)] for i in range(5)],
        figsize=(12, 15)
    )
    histograms, x_bin_edges, y_bin_edges = plot_metrics_by_mismatch_2d(
        speeds, rapid, 20,
        r'$\log_{10}(\sigma_{*}(\mathbf{u}^{\mathrm{P}}))$',
        r'$\log_{10}(\sigma_{*}(\mathbf{u}^{\mathrm{P}}) / \sigma_{*}(\mathbf{u}^M))$',
        axes, indices=list(range(20)), ymin=0
    )
    #print(x_bin_edges[0], x_bin_edges[-1], y_bin_edges[0], y_bin_edges[-1])
    plt.tight_layout()
    plt.savefig('plots/line-2-w2-minusbind-single-speed-vs-rapid-by-mismatch-all.pdf')
    plt.close()

    # ------------------------------------------------------------------------- #
    # Plot how specificity and specific rapidity changes with mismatch position
    # ------------------------------------------------------------------------- #
    # First plot the histograms for mismatch positions 0, 6, 12, 19 ... 
    fig, axes = plt.subplot_mosaic(
        [['{}{}'.format(i, j) for j in range(4)] for i in range(1)],
        figsize=(13, 3)
    )
    _, x_bin_edges, y_bin_edges = plot_metrics_by_mismatch_2d(
        specs, rapid, 20,
        r'$\log_{10}(\phi(\mathbf{u}^{\mathrm{P}}) / \phi(\mathbf{u}^M))$',
        r'$\log_{10}(\sigma_{*}(\mathbf{u}^{\mathrm{P}}) / \sigma_{*}(\mathbf{u}^M))$',
        axes, indices=[0, 6, 12, 19], xmin=0
    )
    #print(x_bin_edges[0], x_bin_edges[-1], y_bin_edges[0], y_bin_edges[-1])
    plt.subplots_adjust(
        left=0.1,
        right=0.9,
        bottom=0.18,
        top=0.9,
        hspace=0.3,
        wspace=0.3
    )
    plt.savefig('plots/line-2-w2-minusbind-single-spec-vs-rapid-by-mismatch.pdf')
    plt.close()

    # Then plot the histograms for all 20 mismatch positions ... 
    fig, axes = plt.subplot_mosaic(
        [['{}{}'.format(i, j) for j in range(4)] for i in range(5)],
        figsize=(12, 15)
    )
    histograms, x_bin_edges, y_bin_edges = plot_metrics_by_mismatch_2d(
        specs, rapid, 20,
        r'$\log_{10}(\phi(\mathbf{u}^{\mathrm{P}}) / \phi(\mathbf{u}^M))$',
        r'$\log_{10}(\sigma_{*}(\mathbf{u}^{\mathrm{P}}) / \sigma_{*}(\mathbf{u}^M))$',
        axes, indices=list(range(20)), xmin=0
    )
    #print(x_bin_edges[0], x_bin_edges[-1], y_bin_edges[0], y_bin_edges[-1])
    plt.tight_layout()
    plt.savefig('plots/line-2-w2-minusbind-single-spec-vs-rapid-by-mismatch-all.pdf')
    plt.close()

    # ------------------------------------------------------------------------------- #
    # Plot how specificity and specific dissociativity changes with mismatch position
    # ------------------------------------------------------------------------------- #
    fig, axes = plt.subplot_mosaic(    # Then plot all 20 subplots 
        [['{}{}'.format(i, j) for j in range(4)] for i in range(5)],
        figsize=(12, 15)
    )
    _, x_bin_edges, y_bin_edges = plot_metrics_by_mismatch_2d(
        specs, dead_dissoc, 20,
        r'$\log_{10}(\phi(\mathbf{u}^{\mathrm{P}}) / \phi(\mathbf{u}^M))$',
        r'$\log_{10}(\sigma(\mathbf{u}^M) / \sigma(\mathbf{u}^{\mathrm{P}}))$',
        axes, indices=list(range(20)), xmin=0, ymin=0
    )
    #print(x_bin_edges[0], x_bin_edges[-1], y_bin_edges[0], y_bin_edges[-1])

    # For each plot, identify the upper limit of the uppermost bin in the
    # second column 
    dissoc_threshold_indices = {}
    for i in range(2, 20):
        spec_within_range = (specs[:, i] >= x_bin_edges[1]) & (specs[:, i] < x_bin_edges[2]) 
        dissoc_threshold_indices[i] = np.nonzero(
            y_bin_edges > np.max(dead_dissoc[spec_within_range, i])
        )[0][0]
    dissoc_thresholds = {i: y_bin_edges[dissoc_threshold_indices[i]] for i in range(2, 20)}
    #print(dissoc_threshold_indices)
    #print(dissoc_thresholds)
    for i in range(5):
        for j in range(4):
            k = 4 * i + j
            if k >= 2:
                key = '{}{}'.format(i, j)
                xlim = axes[key].get_xlim()
                axes[key].plot(
                    [x_bin_edges[0], x_bin_edges[-1]],
                    [dissoc_thresholds[k], dissoc_thresholds[k]],
                    c='red', linewidth=2, linestyle='--'
                )
                ylim = axes[key].get_ylim()
                yaxis_fraction = (dissoc_thresholds[k] - ylim[0]) / (ylim[1] - ylim[0])
                axes[key].annotate(
                    '{:.1f}'.format(dissoc_thresholds[k]),
                    xy=(0.97, yaxis_fraction + 0.01),
                    xycoords='axes fraction',
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    size=9,
                    color='red'
                )
    plt.tight_layout()
    plt.savefig('plots/line-2-w2-minusbind-single-spec-vs-deaddissoc-by-mismatch-all.pdf')
    plt.close()

    fig, axes = plt.subplot_mosaic(    # Separately plot only four of the subplots
        [['{}{}'.format(i, j) for j in range(4)] for i in range(1)],
        figsize=(13, 3)
    )
    _, x_bin_edges, y_bin_edges = plot_metrics_by_mismatch_2d(
        specs, dead_dissoc, 20,
        r'$\log_{10}(\phi(\mathbf{u}^{\mathrm{P}}) / \phi(\mathbf{u}^M))$',
        r'$\log_{10}(\sigma(\mathbf{u}^M) / \sigma(\mathbf{u}^{\mathrm{P}}))$',
        axes, indices=[0, 6, 12, 19], xmin=0, ymin=0
    )
    print(x_bin_edges[0], x_bin_edges[-1], y_bin_edges[0], y_bin_edges[-1])
    for i, k in enumerate([0, 6, 12, 19]):
        if k >= 2:
            key = '0{}'.format(i)
            xlim = axes[key].get_xlim()
            axes[key].plot(
                [x_bin_edges[0], x_bin_edges[-1]],
                [dissoc_thresholds[k], dissoc_thresholds[k]],
                c='red', linewidth=2, linestyle='--'
            )
            ylim = axes[key].get_ylim()
            yaxis_fraction = (dissoc_thresholds[k] - ylim[0]) / (ylim[1] - ylim[0])
            axes[key].annotate(
                '{:.1f}'.format(dissoc_thresholds[k]),
                xy=(0.97, yaxis_fraction + 0.01),
                xycoords='axes fraction',
                horizontalalignment='right',
                verticalalignment='bottom',
                size=9,
                color='red'
            )
    plt.subplots_adjust(
        left=0.1,
        right=0.9,
        bottom=0.18,
        top=0.9,
        hspace=0.3,
        wspace=0.3
    )
    plt.savefig('plots/line-2-w2-minusbind-single-spec-vs-deaddissoc-by-mismatch.pdf')
    plt.close()
    
#######################################################################
if __name__ == '__main__':
    main()
