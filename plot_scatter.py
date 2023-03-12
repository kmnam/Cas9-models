"""
Authors:
    Kee-Myoung Nam

Last updated:
    3/12/2023
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

##########################################################################
def plot_metrics_by_mismatch_scatter(xvals, yvals, nbins, xlabel, ylabel, axes,
                                     npoints_per_bin=1, indices=list(range(20)),
                                     ax_indices=[(i, j) for i in range(5) for j in range(4)],
                                     alpha=1.0, color=sns.color_palette()[0],
                                     xmin=None, xmax=None, ymin=None, ymax=None,
                                     labelsize=12, ticklabelsize=8, cbar_ticklabelsize=8,
                                     annotate_fmt=r'$M = \{{ {} \}}$'):
    # First identify the min/max x- and y-values over all mismatch positions
    if xmin is None:
        xmin = np.min(xvals[:, indices])
    if xmax is None:
        xmax = np.max(xvals[:, indices])
    if ymin is None:
        ymin = np.min(yvals[:, indices])
    if ymax is None:
        ymax = np.max(yvals[:, indices])

    # Divide the x-axis into bins 
    x_bin_edges = np.linspace(xmin, xmax, nbins + 1)

    # Produce scatterplots for each mismatch position 
    max_row = max(int(c[0]) for c in ax_indices)
    for k, (i, j) in enumerate(zip(indices, ax_indices)):
        # For each bin ...
        for m in range(nbins):
            # ... extract the points that fall within the bin with the 
            # greatest y-values
            subset_in_bin = (xvals[:, i] >= x_bin_edges[m]) & (xvals[:, i] < x_bin_edges[m+1])
            xvals_subset = xvals[subset_in_bin, i]
            yvals_subset = yvals[subset_in_bin, i]
            subset_to_plot = np.argsort(yvals_subset)[::-1][:npoints_per_bin] 
            axes[j].scatter(
                xvals_subset[subset_to_plot], yvals_subset[subset_to_plot]
            )
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

##########################################################################
def plot_scatter(filenames, output_prefix):
    # Parse the output metrics for single-mismatch substrates  
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
    # Plot how cleavage probability depends on mismatch position ...
    fig, axes = plt.subplot_mosaic(
        [['{}{}'.format(i, j) for j in range(4)] for i in range(5)],
        figsize=(12, 14)
    )
    plot_metrics_by_mismatch_scatter(
        activities, specs, 100,
        r'$\mathrm{Prob}(\mathbf{u}^{\mathrm{P}})$',
        r'$\log_{10}{(\mathrm{Spec}(\mathbf{u}^M))}$',
        axes, npoints_per_bin=1, indices=list(range(20)), xmin=0, ymin=0,
        ax_indices=['{}{}'.format(i, j) for i in range(5) for j in range(4)],
    )
    plt.tight_layout()
    plt.savefig('plots/{}-prob-by-mismatch-all.pdf'.format(output_prefix))
    plt.close()

    # Plot how speed and specificity change with mismatch position
    fig, axes = plt.subplot_mosaic(
        [['{}{}'.format(i, j) for j in range(4)] for i in range(5)],
        figsize=(12, 14)
    )
    plot_metrics_by_mismatch_scatter(
        speeds, specs, 100,
        r'$\mathrm{Rate}(\mathbf{u}^{\mathrm{P}})$',
        r'$\log_{10}{(\mathrm{Spec}(\mathbf{u}^M))}$',
        axes, npoints_per_bin=1, indices=list(range(20)), ymin=0,
        ax_indices=['{}{}'.format(i, j) for i in range(5) for j in range(4)],
    )
    plt.tight_layout()
    plt.savefig('plots/{}-speed-vs-spec-by-mismatch-all.pdf'.format(output_prefix))
    plt.close()

    # Plot how specificity and specific rapidity change with mismatch position
    fig, axes = plt.subplot_mosaic(
        [['{}{}'.format(i, j) for j in range(4)] for i in range(5)],
        figsize=(12, 14)
    )
    plot_metrics_by_mismatch_scatter(
        specs, rapid, 100,
        r'$\log_{10}{(\mathrm{Spec}(\mathbf{u}^M))}$',
        r'$\log_{10}{(\mathrm{Rapid}(\mathbf{u}^M))}$',
        axes, npoints_per_bin=1, indices=list(range(20)), xmin=0,
        ax_indices=['{}{}'.format(i, j) for i in range(5) for j in range(4)]
    )
    plt.tight_layout()
    plt.savefig('plots/{}-spec-vs-rapid-by-mismatch-all.pdf'.format(output_prefix))
    plt.close()

    # Plot a subset of plots for mismatch positions 5, 7, 9, 11, 13, 15, 17, 19
    fig, axes = plt.subplot_mosaic(
        [['{}{}'.format(i, j) for j in range(4)] for i in range(2)],
        figsize=(12, 5)
    )
    plot_metrics_by_mismatch_scatter(
        specs, rapid, 100,
        r'$\log_{10}{(\mathrm{Spec}(\mathbf{u}^M))}$',
        r'$\log_{10}{(\mathrm{Rapid}(\mathbf{u}^M))}$',
        axes, npoints_per_bin=1, indices=[5, 7, 9, 11, 13, 15, 17, 19], xmin=0,
        ax_indices=['{}{}'.format(i, j) for i in range(2) for j in range(4)]
    )
    plt.tight_layout()
    plt.savefig('plots/{}-spec-vs-rapid-by-mismatch-main.pdf'.format(output_prefix))
    plt.close()

##########################################################################
def main():
    filenames = {
        'logrates': 'data/line-3-combined-single-logrates-subset.tsv',
        'probs': 'data/line-3-combined-single-probs-subset.tsv',
        'specs': 'data/line-3-combined-single-specs-subset.tsv',
        'cleave': 'data/line-3-combined-single-cleave-subset.tsv',
        'rapid': 'data/line-3-combined-single-rapid-subset.tsv',
    }
    plot_scatter(filenames, 'line-3-combined-single')

##########################################################################
if __name__ == '__main__':
    main()