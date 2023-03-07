"""
Authors:
    Kee-Myoung Nam

Last updated:
    3/7/2023
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
                                labelsize=12, ticklabelsize=8, cbar_ticklabelsize=8,
                                annotate_fmt=r'$M = \{{ {} \}}$'):
    # First identify the min/max x- and y-values over all mismatch positions
    if xmin is None:
        xmin = round(np.min(xvals[:, indices]))
    if xmax is None:
        xmax = round(np.max(xvals[:, indices]))
    if ymin is None:
        ymin = round(np.min(yvals[:, indices]))
    if ymax is None:
        ymax = round(np.max(yvals[:, indices]))

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
            xy=(0.98, 0.96),
            xycoords='axes fraction',
            horizontalalignment='right',
            verticalalignment='top',
            size=10
        )
    
    return histograms, x_bin_edges, y_bin_edges

##########################################################################
def plot_histograms(filenames, output_prefix, label_speed_thresholds=False,
                    label_speed_thresholds_from=3):
    # ---------------------------------------------------------------- # 
    # Parse the output metrics for single-mismatch substrates  
    # ---------------------------------------------------------------- #
    logrates = np.loadtxt(filenames['logrates'])
    probs = np.loadtxt(filenames['probs'])
    specs = np.loadtxt(filenames['specs'])
    cleave = np.loadtxt(filenames['cleave'])
    rapid = np.loadtxt(filenames['rapid'])

    # Cleavage probabilities on perfect-match substrates
    activities = np.tile(probs[:, 0].reshape((probs.shape[0]), 1), 20)

    # Cleavage rates on perfect-match substrates 
    speeds = np.tile(cleave[:, 0].reshape((cleave.shape[0]), 1), 20)

    ######################################################################
    # ---------------------------------------------------------------- #
    # Plot how cleavage probability depends on mismatch position ...
    # ---------------------------------------------------------------- #
    fig, axes = plt.subplot_mosaic(
        [['{}{}'.format(i, j) for j in range(4)] for i in range(5)],
        figsize=(12, 14)
    )
    _, x_bin_edges, y_bin_edges = plot_metrics_by_mismatch_2d(
        activities, specs, 20,
        #r'$\phi(\mathbf{u}^{\mathrm{P}})$',
        r'$\mathrm{Prob}(\mathbf{u}^{\mathrm{P}})$',
        #r'$\log_{10}(\phi(\mathbf{u}^{\mathrm{P}}) / \phi(\mathbf{u}^M))$',
        r'$\log_{10}{(\mathrm{Spec}(\mathbf{u}^M))}$',
        axes, indices=list(range(20)), xmin=0, ymin=0,
        ax_indices=['{}{}'.format(i, j) for i in range(5) for j in range(4)],
    )
    plt.tight_layout()
    plt.savefig('plots/{}-prob-by-mismatch-all.pdf'.format(output_prefix))
    plt.close()

    # ------------------------------------------------------------- #
    # Plot how speed and specificity change with mismatch position
    # ------------------------------------------------------------- #
    fig, axes = plt.subplot_mosaic(
        [['{}{}'.format(i, j) for j in range(4)] for i in range(5)],
        figsize=(12, 14)
    )
    histograms, x_bin_edges, y_bin_edges = plot_metrics_by_mismatch_2d(
        speeds, specs, 20,
        #r'$\sigma_{*}(\mathbf{u}^{\mathrm{P}})$',
        r'$\mathrm{Rate}(\mathbf{u}^{\mathrm{P}})$',
        #r'$\log_{10}(\phi(\mathbf{u}^{\mathrm{P}}) / \phi(\mathbf{u}^M))$',
        r'$\log_{10}{(\mathrm{Spec}(\mathbf{u}^M))}$',
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
                        size=10,
                        color='red'
                    )
    plt.tight_layout()
    plt.savefig('plots/{}-speed-vs-spec-by-mismatch-all.pdf'.format(output_prefix))
    plt.close()

    # ------------------------------------------------------------------------- #
    # Plot how specificity and specific rapidity change with mismatch position
    # ------------------------------------------------------------------------- #
    fig, axes = plt.subplot_mosaic(
        [['{}{}'.format(i, j) for j in range(4)] for i in range(5)],
        figsize=(12, 14)
    )
    histograms, x_bin_edges, y_bin_edges = plot_metrics_by_mismatch_2d(
        specs, rapid, 20,
        #r'$\log_{10}(\phi(\mathbf{u}^{\mathrm{P}}) / \phi(\mathbf{u}^M))$',
        r'$\log_{10}{(\mathrm{Spec}(\mathbf{u}^M))}$',
        #r'$\log_{10}(\sigma_{*}(\mathbf{u}^{\mathrm{P}}) / \sigma_{*}(\mathbf{u}^M))$',
        r'$\log_{10}{(\mathrm{Rapid}(\mathbf{u}^M))}$',
        axes, indices=list(range(20)), xmin=0,
        ax_indices=['{}{}'.format(i, j) for i in range(5) for j in range(4)]
    )
    plt.tight_layout()
    plt.savefig('plots/{}-spec-vs-rapid-by-mismatch-all.pdf'.format(output_prefix))
    plt.close()

    # Plot a subset of histograms for mismatch positions 5, 7, 9, 11, 13, 15, 17, 19
    fig, axes = plt.subplot_mosaic(
        [['{}{}'.format(i, j) for j in range(4)] for i in range(2)],
        figsize=(12, 5)
    )
    histograms, x_bin_edges, y_bin_edges = plot_metrics_by_mismatch_2d(
        specs, rapid, 20,
        #r'$\log_{10}(\phi(\mathbf{u}^{\mathrm{P}}) / \phi(\mathbf{u}^M))$',
        r'$\log_{10}{(\mathrm{Spec}(\mathbf{u}^M))}$',
        #r'$\log_{10}(\sigma_{*}(\mathbf{u}^{\mathrm{P}}) / \sigma_{*}(\mathbf{u}^M))$',
        r'$\log_{10}{(\mathrm{Rapid}(\mathbf{u}^M))}$',
        axes, indices=[5, 7, 9, 11, 13, 15, 17, 19], xmin=0,
        ax_indices=['{}{}'.format(i, j) for i in range(2) for j in range(4)]
    )
    plt.tight_layout()
    plt.savefig('plots/{}-spec-vs-rapid-by-mismatch-main.pdf'.format(output_prefix))
    plt.close()

##########################################################################
def main():
    filenames = {
        'logrates': 'data/line_3_diff1_combined_single-logrates-subset.tsv',
        'probs': 'data/line_3_diff1_combined_single-probs-subset.tsv',
        'specs': 'data/line_3_diff1_combined_single-specs-subset.tsv',
        'cleave': 'data/line_3_diff1_combined_single-cleave-subset.tsv',
        'rapid': 'data/line_3_diff1_combined_single-rapid-subset.tsv',
    }
    plot_histograms(
        filenames, 'line_3_diff1_combined_single', label_speed_thresholds=False,
        label_speed_thresholds_from=1
    )

##########################################################################
if __name__ == '__main__':
    main()
