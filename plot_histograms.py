"""
Authors:
    Kee-Myoung Nam

Last updated:
    1/15/2023
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

##########################################################################
def plot_histograms(filenames, output_prefix):
    # ---------------------------------------------------------------- # 
    # Parse the output metrics for single-mismatch substrates  
    # ---------------------------------------------------------------- #
    logrates = np.loadtxt(filenames['logrates'])
    probs = np.loadtxt(filenames['probs'])
    specs = np.loadtxt(filenames['specs'])
    cleave = np.loadtxt(filenames['cleave'])
    unbind = np.loadtxt(filenames['unbind'])
    rapid = np.loadtxt(filenames['rapid'])
    dead_dissoc = np.loadtxt(filenames['deaddissoc'])

    # Cleavage probabilities on perfect-match substrates
    activities = np.tile(probs[:, 0].reshape((probs.shape[0]), 1), 20)

    # Cleavage rates on perfect-match substrates 
    speeds = np.tile(cleave[:, 0].reshape((cleave.shape[0]), 1), 20)

    ######################################################################
    def plot_metrics_by_mismatch_2d(xvals, yvals, nbins, xlabel, ylabel, axes,
                                    indices=list(range(20)),
                                    ax_indices=[(i, j) for i in range(5) for j in range(4)],
                                    xmin=None, xmax=None, ymin=None, ymax=None,
                                    labelsize=10, ticklabelsize=8, cbar_ticklabelsize=8,
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
        max_row = max(int(c[0]) for c in ax_indices)
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
            if type(j) == str:
                if j.startswith(str(max_row)):
                    axes[j].set_xlabel(xlabel, size=labelsize)
                if j.endswith('0'):
                    axes[j].set_ylabel(ylabel, size=labelsize)
            elif type(j) == tuple:
                if j[0] == max_row:
                    axes[j].set_xlabel(xlabel, size=labelsize)
                if j[1] == 0:
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
    fig_main, axes_main = plt.subplot_mosaic(
        [['A0', 'A0', 'A1', 'A1']] +\
        [['{}{}'.format(j, k) for k in range(4)] for j in range(4)],
        figsize=(12, 14)
    )
    for i in range(n):     # Plot cleavage probabilities as a function of mismatch position 
        axes_main['A0'].plot(
            list(range(20)), probs[i, 1:], c=(0.9, 0.9, 0.9), marker=None,
            zorder=0
        )
    for i in range(n):     # Plot cleavage specificities as a function of mismatch position 
        axes_main['A1'].plot(
            list(range(20)), np.power(10.0, -specs[i, :]), c=(0.9, 0.9, 0.9),
            marker=None, zorder=0
        )
    indices = [            # Handpicked plots to be highlighted in different colors 
        704,  # blue
        146,  # orange 
        830,  # green 
        697,  # red
        440,  # purple
        105,  # brown (update to yellow)
        139   # pink
    ]
    palette = sns.color_palette('colorblind', len(indices))
    palette[-2] = sns.color_palette('colorblind', 9)[8]    # Replace brown with yellow
    for i, j in enumerate(indices):
        axes_main['A0'].plot(
            list(range(20)), probs[j, 1:], c=palette[i], marker=None, zorder=1,
            linewidth=2
        )
        axes_main['A1'].plot(
            list(range(20)), np.power(10.0, -specs[j, :]), c=palette[i],
            marker=None, zorder=1, linewidth=2
        )
    axes_main['A0'].set_xlim([-0.5, 19.5])
    axes_main['A0'].set_xlabel(r'$m$', size=10)
    axes_main['A0'].set_xticks(list(range(20)))
    axes_main['A0'].set_xticklabels([str(i) for i in range(20)], size=8)
    axes_main['A0'].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes_main['A0'].set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
    axes_main['A0'].set_ylabel(r'$\phi(\mathbf{u}^{\{m\}})$', size=10)
    axes_main['A1'].set_xlim([-0.5, 19.5])
    axes_main['A1'].set_xlabel(r'$m$', size=10)
    axes_main['A1'].set_xticks(list(range(20)))
    axes_main['A1'].set_xticklabels([str(i) for i in range(20)], size=8)
    axes_main['A1'].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes_main['A1'].set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
    axes_main['A1'].set_ylabel(r'$\phi(\mathbf{u}^{\{m\}}) \,/\, \phi(\mathbf{u}^{\mathrm{P}})$', size=10)

    # Then plot the histograms for mismatch positions 0, 6, 12, 19 ...
    _, x_bin_edges, y_bin_edges = plot_metrics_by_mismatch_2d(
        activities, specs, 20,
        r'$\phi(\mathbf{u}^{\mathrm{P}})$',
        r'$\log_{10}(\phi(\mathbf{u}^{\mathrm{P}}) / \phi(\mathbf{u}^M))$',
        axes_main, indices=[0, 6, 12, 19], ax_indices=['00', '01', '02', '03'],
        xmin=0, xmax=1, ymin=0
    )

    # Then plot the histograms for all 20 mismatch positions ... 
    fig, axes = plt.subplot_mosaic(
        [['{}{}'.format(i, j) for j in range(4)] for i in range(5)],
        figsize=(12, 14)
    )
    _, x_bin_edges, y_bin_edges = plot_metrics_by_mismatch_2d(
        activities, specs, 20,
        r'$\phi(\mathbf{u}^{\mathrm{P}})$',
        r'$\log_{10}(\phi(\mathbf{u}^{\mathrm{P}}) / \phi(\mathbf{u}^M))$',
        axes, indices=list(range(20)), xmin=0, ymin=0,
        ax_indices=['{}{}'.format(i, j) for i in range(5) for j in range(4)],
    )
    print(
        'bin edges for activity vs specificity:',
        x_bin_edges[0], x_bin_edges[-1], y_bin_edges[0], y_bin_edges[-1]
    )
    plt.tight_layout()
    plt.savefig('plots/{}-prob-by-mismatch-all.pdf'.format(output_prefix))
    plt.close()

    # ------------------------------------------------------------- #
    # Plot how speed and specificity change with mismatch position
    # ------------------------------------------------------------- #
    # First plot the histograms for all 20 mismatch positions ...
    fig, axes = plt.subplot_mosaic(
        [['{}{}'.format(i, j) for j in range(4)] for i in range(5)],
        figsize=(12, 14)
    )
    histograms, x_bin_edges, y_bin_edges = plot_metrics_by_mismatch_2d(
        speeds, specs, 20,
        r'$\sigma_{*}(\mathbf{u}^{\mathrm{P}})$',
        r'$\log_{10}(\phi(\mathbf{u}^{\mathrm{P}}) / \phi(\mathbf{u}^M))$',
        axes, indices=list(range(20)), ymin=0,
        ax_indices=['{}{}'.format(i, j) for i in range(5) for j in range(4)],
    )
    print(
        'bin edges for speed vs specificity:',
        x_bin_edges[0], x_bin_edges[-1], y_bin_edges[0], y_bin_edges[-1]
    )

    # For each histogram, identify the upper limit of the rightmost bin in the
    # second-to-bottommost column 
    speed_threshold_indices = {}
    for i in range(4, 20):
        spec_within_range = (
            (specs[:, i] >= y_bin_edges[1]) &
            (specs[:, i] < y_bin_edges[2])
        )
        speed_threshold_indices[i] = np.nonzero(
            x_bin_edges > np.max(speeds[spec_within_range, i])
        )[0][0]
    speed_thresholds = {i: x_bin_edges[speed_threshold_indices[i]] for i in range(4, 20)}
    print(speed_threshold_indices)
    print(speed_thresholds)
    for i in range(5):
        for j in range(4):
            k = 4 * i + j
            if k >= 4:
                key = '{}{}'.format(i, j)
                xlim = axes[key].get_xlim()
                axes[key].plot(
                    [speed_thresholds[k], speed_thresholds[k]],
                    [y_bin_edges[0], y_bin_edges[-1]],
                    c='red', linewidth=2, linestyle='--'
                )
                xlim = axes[key].get_xlim()
                xaxis_fraction = (speed_thresholds[k] - xlim[0]) / (xlim[1] - xlim[0])
                axes[key].annotate(
                    '{:.1f}'.format(speed_thresholds[k]),
                    xy=(xaxis_fraction + 0.02, 0.08),
                    xycoords='axes fraction',
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    size=9,
                    color='red'
                )
    plt.tight_layout()
    plt.savefig('plots/{}-speed-vs-spec-by-mismatch-all.pdf'.format(output_prefix))
    plt.close()

    # Then plot the histograms for mismatch positions 4, 9, 14, 19 ...
    indices = [4, 9, 14, 19]
    _, x_bin_edges, y_bin_edges = plot_metrics_by_mismatch_2d(
        speeds, specs, 20,
        r'$\sigma_{*}(\mathbf{u}^{\mathrm{P}})$',
        r'$\log_{10}(\phi(\mathbf{u}^{\mathrm{P}}) / \phi(\mathbf{u}^M))$',
        axes_main, indices=indices, ax_indices=['10', '11', '12', '13'], ymin=0
    )
    for i, k in enumerate(indices):
        key = '1{}'.format(i)
        xlim = axes_main[key].get_xlim()
        axes_main[key].plot(
            [speed_thresholds[k], speed_thresholds[k]],
            [y_bin_edges[0], y_bin_edges[-1]],
            c='red', linewidth=2, linestyle='--'
        )
        xlim = axes_main[key].get_xlim()
        xaxis_fraction = (speed_thresholds[k] - xlim[0]) / (xlim[1] - xlim[0])
        axes_main[key].annotate(
            '{:.1f}'.format(speed_thresholds[k]),
            xy=(xaxis_fraction + 0.02, 0.08),
            xycoords='axes fraction',
            horizontalalignment='left',
            verticalalignment='bottom',
            size=9,
            color='red'
        )

    # ------------------------------------------------------------------- #
    # Plot how speed and specific rapidity change with mismatch position
    # ------------------------------------------------------------------- #
    # Then plot the histograms for all 20 mismatch positions ... 
    fig, axes = plt.subplot_mosaic(
        [['{}{}'.format(i, j) for j in range(4)] for i in range(5)],
        figsize=(12, 14)
    )
    histograms, x_bin_edges, y_bin_edges = plot_metrics_by_mismatch_2d(
        speeds, rapid, 20,
        r'$\sigma_{*}(\mathbf{u}^{\mathrm{P}})$',
        r'$\log_{10}(\sigma_{*}(\mathbf{u}^{\mathrm{P}}) / \sigma_{*}(\mathbf{u}^M))$',
        axes, indices=list(range(20)), ymin=0,
        ax_indices=['{}{}'.format(i, j) for i in range(5) for j in range(4)]
    )
    #print(x_bin_edges[0], x_bin_edges[-1], y_bin_edges[0], y_bin_edges[-1])
    plt.tight_layout()
    plt.savefig('plots/{}-speed-vs-rapid-by-mismatch-all.pdf'.format(output_prefix))
    plt.close()

    # ------------------------------------------------------------------------- #
    # Plot how specificity and specific rapidity change with mismatch position
    # ------------------------------------------------------------------------- #
    # First plot the histograms for mismatch positions 0, 6, 12, 19 ... 
    _, x_bin_edges, y_bin_edges = plot_metrics_by_mismatch_2d(
        specs, rapid, 20,
        r'$\log_{10}(\phi(\mathbf{u}^{\mathrm{P}}) / \phi(\mathbf{u}^M))$',
        r'$\log_{10}(\sigma_{*}(\mathbf{u}^{\mathrm{P}}) / \sigma_{*}(\mathbf{u}^M))$',
        axes_main, indices=[0, 6, 12, 19], ax_indices=['20', '21', '22', '23'], xmin=0
    )

    # Then plot the histograms for all 20 mismatch positions ... 
    fig, axes = plt.subplot_mosaic(
        [['{}{}'.format(i, j) for j in range(4)] for i in range(5)],
        figsize=(12, 14)
    )
    histograms, x_bin_edges, y_bin_edges = plot_metrics_by_mismatch_2d(
        specs, rapid, 20,
        r'$\log_{10}(\phi(\mathbf{u}^{\mathrm{P}}) / \phi(\mathbf{u}^M))$',
        r'$\log_{10}(\sigma_{*}(\mathbf{u}^{\mathrm{P}}) / \sigma_{*}(\mathbf{u}^M))$',
        axes, indices=list(range(20)), xmin=0,
        ax_indices=['{}{}'.format(i, j) for i in range(5) for j in range(4)]
    )
    #print(x_bin_edges[0], x_bin_edges[-1], y_bin_edges[0], y_bin_edges[-1])
    plt.tight_layout()
    plt.savefig('plots/{}-spec-vs-rapid-by-mismatch-all.pdf'.format(output_prefix))
    plt.close()

    # ------------------------------------------------------------------------------- #
    # Plot how specificity and specific dissociativity change with mismatch position
    # ------------------------------------------------------------------------------- #
    fig, axes = plt.subplot_mosaic(    # Then plot all 20 subplots 
        [['{}{}'.format(i, j) for j in range(4)] for i in range(5)],
        figsize=(12, 14)
    )
    _, x_bin_edges, y_bin_edges = plot_metrics_by_mismatch_2d(
        specs, dead_dissoc, 20,
        r'$\log_{10}(\phi(\mathbf{u}^{\mathrm{P}}) / \phi(\mathbf{u}^M))$',
        r'$\log_{10}(\sigma(\mathbf{u}^M) / \sigma(\mathbf{u}^{\mathrm{P}}))$',
        axes, indices=list(range(20)), xmin=0, ymin=0,
        ax_indices=['{}{}'.format(i, j) for i in range(5) for j in range(4)]
    )
    print(x_bin_edges[0], x_bin_edges[-1], y_bin_edges[0], y_bin_edges[-1])

    # For each plot, identify the upper limit of the uppermost bin in the
    # second column 
    dissoc_threshold_indices = {}
    for i in range(2, 20):
        spec_within_range = (specs[:, i] >= x_bin_edges[1]) & (specs[:, i] < x_bin_edges[2]) 
        dissoc_threshold_indices[i] = np.nonzero(
            y_bin_edges > np.max(dead_dissoc[spec_within_range, i])
        )[0][0]
    dissoc_thresholds = {i: y_bin_edges[dissoc_threshold_indices[i]] for i in range(2, 20)}
    print(dissoc_threshold_indices)
    print(dissoc_thresholds)
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
    plt.savefig('plots/{}-spec-vs-deaddissoc-by-mismatch-all.pdf'.format(output_prefix))
    plt.close()

    indices = [2, 10, 16, 19]
    _, x_bin_edges, y_bin_edges = plot_metrics_by_mismatch_2d(
        specs, dead_dissoc, 20,
        r'$\log_{10}(\phi(\mathbf{u}^{\mathrm{P}}) / \phi(\mathbf{u}^M))$',
        r'$\log_{10}(\sigma(\mathbf{u}^M) / \sigma(\mathbf{u}^{\mathrm{P}}))$',
        axes_main, indices=indices, ax_indices=['30', '31', '32', '33'], xmin=0, ymin=0
    )
    print(x_bin_edges[0], x_bin_edges[-1], y_bin_edges[0], y_bin_edges[-1])
    for i, k in enumerate(indices):
        key = '3{}'.format(i)
        xlim = axes_main[key].get_xlim()
        axes_main[key].plot(
            [x_bin_edges[0], x_bin_edges[-1]],
            [dissoc_thresholds[k], dissoc_thresholds[k]],
            c='red', linewidth=2, linestyle='--'
        )
        ylim = axes_main[key].get_ylim()
        yaxis_fraction = (dissoc_thresholds[k] - ylim[0]) / (ylim[1] - ylim[0])
        axes_main[key].annotate(
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
        hspace=0.35,
        wspace=0.3
    )
    plt.savefig('plots/{}-main.pdf'.format(output_prefix))
    plt.close()
    
##########################################################################
def main():
    filenames = {
        'logrates': 'data/line-2-w2-minusbind-single-logrates.tsv',
        'probs': 'data/line-2-w2-minusbind-single-probs.tsv',
        'specs': 'data/line-2-w2-minusbind-single-specs.tsv',
        'cleave': 'data/line-2-w2-minusbind-single-cleave.tsv',
        'unbind': 'data/line-2-w2-minusbind-single-unbind.tsv',
        'rapid': 'data/line-2-w2-minusbind-single-rapid.tsv',
        'deaddissoc': 'data/line-2-w2-minusbind-single-deaddissoc.tsv'
    }
    plot_histograms(filenames, 'line-2-w2-minusbind-single')
    filenames = {
        'logrates': 'data/line-2-w2-minusbind-distal-logrates.tsv',
        'probs': 'data/line-2-w2-minusbind-distal-probs.tsv',
        'specs': 'data/line-2-w2-minusbind-distal-specs.tsv',
        'cleave': 'data/line-2-w2-minusbind-distal-cleave.tsv',
        'unbind': 'data/line-2-w2-minusbind-distal-unbind.tsv',
        'rapid': 'data/line-2-w2-minusbind-distal-rapid.tsv',
        'deaddissoc': 'data/line-2-w2-minusbind-distal-deaddissoc.tsv'
    }
    plot_histograms(filenames, 'line-2-w2-minusbind-distal')

##########################################################################
if __name__ == '__main__':
    main()
