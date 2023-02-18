"""
Authors:
    Kee-Myoung Nam

Last updated:
    1/20/2023
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

##########################################################################
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

##########################################################################
def plot_histograms(filenames, output_prefix, plot_main=False,
                    highlight_plot_indices=None, label_speed_thresholds=False,
                    label_dissoc_thresholds=True, label_speed_thresholds_from=3,
                    label_dissoc_thresholds_from=2):
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

    # Pick plots to highlight if not already specified 
    if plot_main and highlight_plot_indices is None:
        highlight_plot_indices = [
            probs[:500, 1].argmax(),
            np.power(10, -specs[:500, 19]).argmin(),
            np.abs(probs[:500, 6] - probs[:500, 5]).argmax(),
            np.abs(probs[:500, 13] - probs[:500, 12]).argmax(),
            np.abs(probs[:500, 17] - probs[:500, 16]).argmax()
            #np.abs(np.power(10, -specs[:500, 19]) - 0.6).argmin(),
            #np.abs(np.power(10, -specs[:500, 12]) - np.power(10, -specs[:500, 11])).argmax(),
        ]

    ######################################################################
    # ---------------------------------------------------------------- #
    # Plot how cleavage probability depends on mismatch position ...
    # ---------------------------------------------------------------- #
    if plot_main:
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
        palette = sns.color_palette('colorblind', len(highlight_plot_indices))
        for i, j in enumerate(highlight_plot_indices):
            axes_main['A0'].plot(
                list(range(20)), probs[j, 1:], c=palette[i], marker=None, zorder=1,
                linewidth=3
            )
            axes_main['A1'].plot(
                list(range(20)), np.power(10.0, -specs[j, :]), c=palette[i],
                marker=None, zorder=1, linewidth=3
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

    # For each histogram, identify the upper limit within all bins but the 
    # bottommost
    if label_speed_thresholds:
        speed_threshold_indices = {}
        for i in range(label_speed_thresholds_from, 20):
            spec_within_range = (specs[:, i] >= y_bin_edges[1])
            try:
                speed_threshold_indices[i] = np.nonzero(
                    x_bin_edges > np.max(speeds[spec_within_range, i])
                )[0][0]
            except IndexError:
                speed_threshold_indices[i] = None
        speed_thresholds = {
            i: None if speed_threshold_indices[i] is None 
            else x_bin_edges[speed_threshold_indices[i]]
            for i in range(label_speed_thresholds_from, 20)
        }
        for i in range(5):
            for j in range(4):
                k = 4 * i + j
                if k >= label_speed_thresholds_from and speed_thresholds[k] is not None:
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

    # Then plot the histograms for mismatch positions 0, 6, 12, 19 ...
    if plot_main:
        indices = [0, 6, 12, 19]
        _, x_bin_edges, y_bin_edges = plot_metrics_by_mismatch_2d(
            speeds, specs, 20,
            r'$\sigma_{*}(\mathbf{u}^{\mathrm{P}})$',
            r'$\log_{10}(\phi(\mathbf{u}^{\mathrm{P}}) / \phi(\mathbf{u}^M))$',
            axes_main, indices=indices, ax_indices=['10', '11', '12', '13'], ymin=0
        )
        if label_speed_thresholds:
            for i, k in enumerate(indices):
                key = '1{}'.format(i)
                if k >= label_speed_thresholds_from and speed_thresholds[k] is not None:
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

    # ------------------------------------------------------------------------- #
    # Plot how specificity and specific rapidity change with mismatch position
    # ------------------------------------------------------------------------- #
    # First plot the histograms for mismatch positions 0, 6, 12, 19 ... 
    if plot_main:
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
    plt.tight_layout()
    plt.savefig('plots/{}-spec-vs-rapid-by-mismatch-all.pdf'.format(output_prefix))
    plt.close()

    # ------------------------------------------------------------------------------- #
    # Plot how specificity and specific dissociativity change with mismatch position
    # ------------------------------------------------------------------------------- #
    # First plot the histograms for all 20 mismatch positions ... 
    fig, axes = plt.subplot_mosaic(
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

    # For each histogram, identify the upper limit within all bins but the 
    # rightmost 
    if label_dissoc_thresholds:
        dissoc_threshold_indices = {}
        for i in range(label_dissoc_thresholds_from, 20):
            spec_within_range = (specs[:, i] >= x_bin_edges[1])
            dissoc_threshold_indices[i] = np.nonzero(
                y_bin_edges > np.max(dead_dissoc[spec_within_range, i])
            )[0][0]
        dissoc_thresholds = {
            i: y_bin_edges[dissoc_threshold_indices[i]]
            for i in range(label_dissoc_thresholds_from, 20)
        }
        for i in range(5):
            for j in range(4):
                k = 4 * i + j
                if k >= label_dissoc_thresholds_from:
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

    # Then plot the histograms for mismatch positions 0, 6, 12, 19 ...
    if plot_main:
        indices = [0, 6, 12, 19]
        _, x_bin_edges, y_bin_edges = plot_metrics_by_mismatch_2d(
            specs, dead_dissoc, 20,
            r'$\log_{10}(\phi(\mathbf{u}^{\mathrm{P}}) / \phi(\mathbf{u}^M))$',
            r'$\log_{10}(\sigma(\mathbf{u}^M) / \sigma(\mathbf{u}^{\mathrm{P}}))$',
            axes_main, indices=indices, ax_indices=['30', '31', '32', '33'], xmin=0, ymin=0
        )
        if label_dissoc_thresholds:
            for i, k in enumerate(indices):
                key = '3{}'.format(i)
                if k >= label_dissoc_thresholds_from:
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
        'logrates': 'data/line_3_diff1_combined_single-logrates.tsv',
        'probs': 'data/line_3_diff1_combined_single-probs.tsv',
        'specs': 'data/line_3_diff1_combined_single-specs.tsv',
        'cleave': 'data/line_3_diff1_combined_single-cleave.tsv',
        'unbind': 'data/line_3_diff1_combined_single-unbind.tsv',
        'rapid': 'data/line_3_diff1_combined_single-rapid.tsv',
        'deaddissoc': 'data/line_3_diff1_combined_single-deaddissoc.tsv'
    }
    plot_histograms(
        filenames, 'line_3_diff1_combined_single', plot_main=True, 
        highlight_plot_indices=None, label_speed_thresholds=True,
        label_dissoc_thresholds=True, label_speed_thresholds_from=1,
        label_dissoc_thresholds_from=3
    )

##########################################################################
if __name__ == '__main__':
    main()
