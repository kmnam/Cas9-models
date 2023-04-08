"""
Authors:
    Kee-Myoung Nam

Last updated:
    4/8/2023
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

##########################################################################
def plot_metrics_by_mismatch_scatter(xvals, yvals, nbins, xlabel, ylabel, rng,
                                     axes, npoints_per_bin=1, plot_subsample=True,
                                     color=sns.color_palette()[0],
                                     color_subsample=sns.color_palette('pastel')[0],
                                     indices=list(range(20)),
                                     ax_indices=[(i, j) for i in range(5) for j in range(4)],
                                     xmin=None, xmax=None, ymin=None, ymax=None,
                                     labelsize=12, ticklabelsize=8, cbar_ticklabelsize=8,
                                     adjust_axes=True, annotate_fmt=None):
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
                xvals_subset[subset_to_plot], yvals_subset[subset_to_plot],
                color=color, alpha=1.0, zorder=1
            )
        # Then plot a random subsample of 10000 points
        if plot_subsample:
            idx = rng.choice(xvals.shape[0], 10000) 
            axes[j].scatter(
                xvals[idx, i], yvals[idx, i], color=color_subsample,
                alpha=1.0, zorder=0, rasterized=True
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

    # Equalize all axes limits if desired 
    if adjust_axes:
        axes_xmin = min(axes[j].get_xlim()[0] for j in ax_indices)
        axes_xmax = max(axes[j].get_xlim()[1] for j in ax_indices)
        axes_ymin = min(axes[j].get_ylim()[0] for j in ax_indices)
        axes_ymax = max(axes[j].get_ylim()[1] for j in ax_indices)
        for i, j in zip(indices, ax_indices):
            # While equalizing all axes limits, also add padding to the upper
            # side of each y-axis and add annotation on the top-right
            axes[j].set_xlim(left=axes_xmin, right=axes_xmax)
            if annotate_fmt is None:
                axes[j].set_ylim(bottom=axes_ymin, top=axes_ymax)
            else:
                axes[j].set_ylim(
                    bottom=axes_ymin, top=(axes_ymin + 1.1 * (axes_ymax - axes_ymin))
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
def main():
    rng = np.random.default_rng(1234567890)
    
    # Parse the output metrics for single-mismatch substrates
    filename_fmts = {
        'logrates': 'data/line-{}-combined-single-logrates.tsv',
        'probs': 'data/line-{}-combined-single-probs.tsv',
        'specs': 'data/line-{}-combined-single-specs.tsv',
        'cleave': 'data/line-{}-combined-single-cleave.tsv',
        'rapid': 'data/line-{}-combined-single-rapid.tsv'
    }
    output_prefix_fmt = 'plots/line-{}-combined-single'
    logrates = np.loadtxt(filename_fmts['logrates'].format(3))
    probs = np.loadtxt(filename_fmts['probs'].format(3))
    specs = np.loadtxt(filename_fmts['specs'].format(3))
    cleave = np.loadtxt(filename_fmts['cleave'].format(3))
    rapid = np.loadtxt(filename_fmts['rapid'].format(3))

    # Cleavage probabilities on perfect-match substrates
    activities = np.tile(probs[:, 0].reshape((probs.shape[0]), 1), 20)

    # Cleavage rates on perfect-match substrates 
    speeds = np.tile(cleave[:, 0].reshape((cleave.shape[0]), 1), 20)

    # Ratios of model parameters
    c_total = logrates[:, 0] - logrates[:, 1]        # c = b / d
    cp_total = logrates[:, 2] - logrates[:, 3]       # cp = b' / d'
    p_total = logrates[:, 2] - logrates[:, 1]        # p = b' / d
    q_total = logrates[:, 0] - logrates[:, 3]        # q = b / d'

    ######################################################################
    # Plot how cleavage probability depends on mismatch position ...
    fig, axes = plt.subplot_mosaic(
        [['{}{}'.format(i, j) for j in range(4)] for i in range(5)],
        figsize=(12, 14)
    )
    plot_metrics_by_mismatch_scatter(
        activities, specs, 50,
        r'$\mathrm{Prob}(\mathbf{u}^{\mathrm{P}})$',
        r'$\log_{10}{(\mathrm{Spec}(\mathbf{u}^{\{m\}}))}$',
        rng, axes, npoints_per_bin=1, indices=list(range(20)), xmin=0, ymin=0,
        ax_indices=['{}{}'.format(i, j) for i in range(5) for j in range(4)],
        annotate_fmt=r'$m = {}$'
    )
    plt.tight_layout()
    plt.savefig(
        '{}-prob-by-mismatch-all.pdf'.format(output_prefix_fmt.format(3)),
        dpi=600, transparent=True
    )
    plt.close()

    # Plot how speed and specificity change with mismatch position
    fig, axes = plt.subplot_mosaic(
        [['{}{}'.format(i, j) for j in range(4)] for i in range(5)],
        figsize=(12, 14)
    )
    plot_metrics_by_mismatch_scatter(
        speeds, specs, 50,
        r'$\mathrm{Rate}(\mathbf{u}^{\mathrm{P}})$',
        r'$\log_{10}{(\mathrm{Spec}(\mathbf{u}^{\{m\}}))}$',
        rng, axes, npoints_per_bin=1, indices=list(range(20)), ymin=0,
        ax_indices=['{}{}'.format(i, j) for i in range(5) for j in range(4)],
        annotate_fmt=r'$m = {}$'
    )
    plt.tight_layout()
    plt.savefig(
        '{}-speed-vs-spec-by-mismatch-all.pdf'.format(output_prefix_fmt.format(3)),
        dpi=600, transparent=True
    )
    plt.close()

    # Plot how specificity and specific rapidity change with mismatch position
    fig, axes = plt.subplot_mosaic(
        [['{}{}'.format(i, j) for j in range(4)] for i in range(5)],
        figsize=(12, 14)
    )
    plot_metrics_by_mismatch_scatter(
        specs, rapid, 50,
        r'$\log_{10}{(\mathrm{Spec}(\mathbf{u}^{\{m\}}))}$',
        r'$\log_{10}{(\mathrm{Rapid}(\mathbf{u}^{\{m\}}))}$',
        rng, axes, npoints_per_bin=1, indices=list(range(20)), xmin=0,
        ax_indices=['{}{}'.format(i, j) for i in range(5) for j in range(4)],
        annotate_fmt=r'$m = {}$'
    )
    plt.tight_layout()
    plt.savefig(
        '{}-spec-vs-rapid-by-mismatch-all.pdf'.format(output_prefix_fmt.format(3)),
        dpi=600, transparent=True
    )
    plt.close()

    # Plot a subset of plots for mismatch positions 5, 7, 9, 11, 13, 15, 17, 19
    fig, axes = plt.subplot_mosaic(
        [['{}{}'.format(i, j) for j in range(4)] for i in range(2)],
        figsize=(12, 6)
    )
    plot_metrics_by_mismatch_scatter(
        specs, rapid, 50,
        r'$\log_{10}{(\mathrm{Spec}(\mathbf{u}^{\{m\}}))}$',
        r'$\log_{10}{(\mathrm{Rapid}(\mathbf{u}^{\{m\}}))}$',
        rng, axes, npoints_per_bin=1, indices=[5, 7, 9, 11, 13, 15, 17, 19], xmin=0,
        ax_indices=['{}{}'.format(i, j) for i in range(2) for j in range(4)],
        annotate_fmt=r'$m = {}$'
    )
    plt.tight_layout()
    plt.savefig(
        '{}-spec-vs-rapid-by-mismatch-main.pdf'.format(output_prefix_fmt.format(3)),
        dpi=600, transparent=True
    )
    plt.close()

    ######################################################################
    # Plot output metrics from *all* parametric polytopes 
    fig1, axes1 = plt.subplot_mosaic(
        [['{}{}'.format(i, j) for j in range(4)] for i in range(5)],
        figsize=(12, 14)
    )
    fig2, axes2 = plt.subplot_mosaic(
        [['{}{}'.format(i, j) for j in range(4)] for i in range(5)],
        figsize=(12, 14)
    )
    fig3, axes3 = plt.subplot_mosaic(
        [['{}{}'.format(i, j) for j in range(4)] for i in range(5)],
        figsize=(12, 14)
    )
    colors = [
        sns.color_palette('deep')[3],
        sns.color_palette('deep')[0],
        sns.color_palette('deep')[1],
        sns.color_palette('deep')[2]
    ]
    colors_subsample = [
        sns.color_palette('pastel')[3],
        sns.color_palette('pastel')[0],
        sns.color_palette('pastel')[1],
        sns.color_palette('pastel')[2]
    ]
    for i, j in enumerate(range(2, 6)):
        # Parse the output metrics for single-mismatch substrates
        logrates = np.loadtxt(filename_fmts['logrates'].format(j))
        probs = np.loadtxt(filename_fmts['probs'].format(j))
        specs = np.loadtxt(filename_fmts['specs'].format(j))
        cleave = np.loadtxt(filename_fmts['cleave'].format(j))
        rapid = np.loadtxt(filename_fmts['rapid'].format(j))

        # Cleavage probabilities on perfect-match substrates
        activities = np.tile(probs[:, 0].reshape((probs.shape[0]), 1), 20)

        # Cleavage rates on perfect-match substrates 
        speeds = np.tile(cleave[:, 0].reshape((cleave.shape[0]), 1), 20)

        plot_metrics_by_mismatch_scatter(
            activities, specs, 50,
            r'$\mathrm{Prob}(\mathbf{u}^{\mathrm{P}})$',
            r'$\log_{10}{(\mathrm{Spec}(\mathbf{u}^{\{m\}}))}$',
            rng, axes1, npoints_per_bin=1, plot_subsample=False,
            color=colors[i], color_subsample=colors_subsample[i],
            indices=list(range(20)), xmin=0, ymin=0,
            ax_indices=['{}{}'.format(p, q) for p in range(5) for q in range(4)],
            adjust_axes=(j == 5), annotate_fmt=(None if j < 5 else r'$m = {}$') 
        )
        plot_metrics_by_mismatch_scatter(
            speeds, specs, 50,
            r'$\mathrm{Rate}(\mathbf{u}^{\mathrm{P}})$',
            r'$\log_{10}{(\mathrm{Spec}(\mathbf{u}^{\{m\}}))}$',
            rng, axes2, npoints_per_bin=1, plot_subsample=False,
            color=colors[i], color_subsample=colors_subsample[i], 
            indices=list(range(20)), ymin=0,
            ax_indices=['{}{}'.format(p, q) for p in range(5) for q in range(4)],
            adjust_axes=(j == 5), annotate_fmt=(None if j < 5 else r'$m = {}$')
        )
        plot_metrics_by_mismatch_scatter(
            specs, rapid, 50,
            r'$\log_{10}{(\mathrm{Spec}(\mathbf{u}^{\{m\}}))}$',
            r'$\log_{10}{(\mathrm{Rapid}(\mathbf{u}^{\{m\}}))}$',
            rng, axes3, npoints_per_bin=1, plot_subsample=False,
            color=colors[i], color_subsample=colors_subsample[i],
            indices=list(range(20)), xmin=0,
            ax_indices=['{}{}'.format(p, q) for p in range(5) for q in range(4)],
            adjust_axes=(j == 5), annotate_fmt=(None if j < 5 else r'$m = {}$')
        )

    fig1.tight_layout()
    fig1.savefig(
        '{}-prob-by-mismatch-all.pdf'.format(output_prefix_fmt.format('all')),
        dpi=600, transparent=True
    )
    plt.close(fig1)
    fig2.tight_layout()
    fig2.savefig(
        '{}-speed-vs-spec-by-mismatch-all.pdf'.format(output_prefix_fmt.format('all')),
        dpi=600, transparent=True
    )
    plt.close(fig2)
    fig3.tight_layout()
    fig3.savefig(
        '{}-spec-vs-rapid-by-mismatch-all.pdf'.format(output_prefix_fmt.format('all')),
        dpi=600, transparent=True
    )
    plt.close(fig3)

    ######################################################################
    # Plot boxplots of parametric ratios for high-rapidity parameter vectors
    # from the main parametric polytope 
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 10))
    logrates = np.loadtxt(filename_fmts['logrates'].format(3))
    specs = np.loadtxt(filename_fmts['specs'].format(3))
    rapid = np.loadtxt(filename_fmts['rapid'].format(3))

    # Ratios of model parameters
    c_total = logrates[:, 0] - logrates[:, 1]        # c = b / d
    cp_total = logrates[:, 2] - logrates[:, 3]       # cp = b' / d'
    p_total = logrates[:, 2] - logrates[:, 1]        # p = b' / d
    q_total = logrates[:, 0] - logrates[:, 3]        # q = b / d'

    c_valid = [c_total]
    cp_valid = [cp_total]
    p_valid = [p_total]
    q_valid = [q_total]
    nbins = 50
    npoints_per_bin = 1
    start_idx = 2
    xmin = 0
    xmax = specs.max()
    ratio_min = -6
    ratio_max = 6
    x_bin_edges = np.linspace(xmin, xmax, nbins + 1)
    for i in range(start_idx, 20):       # For each mismatch position ...
        c_valid_i = []
        cp_valid_i = []
        p_valid_i = []
        q_valid_i = []
        for j in range(nbins):   # ... and each bin along the specificity axis ... 
            # Get indices of parameter vectors for top rapidity values in the 
            # j-th bin
            in_column = (
                (specs[:, i] >= x_bin_edges[j]) & (specs[:, i] < x_bin_edges[j+1])
            )
            rapid_top_idx = np.argsort(rapid[in_column, i])[-npoints_per_bin:]
            logrates_valid = logrates[in_column, :][rapid_top_idx]
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
        axes[i].patches[0].set_facecolor(sns.color_palette('deep')[4])
        for j in range(1, 21 - start_idx):
            axes[i].patches[j].set_facecolor(sns.color_palette('deep')[0])
        axes[i].set_xticklabels(['All'] + [str(k) for k in range(start_idx, 20)])
        axes[i].set_ylim([ratio_min - 0.2, ratio_max + 0.2])
    axes[0].set_ylabel(r"$\log_{10}(b/d)$", size=12)
    axes[1].set_ylabel(r"$\log_{10}(b'/d)$", size=12)
    axes[2].set_ylabel(r"$\log_{10}(b/d')$", size=12)
    axes[3].set_ylabel(r"$\log_{10}(b'/d')$", size=12)
    plt.tight_layout()
    plt.savefig(
        '{}-highrapid-boxplot.pdf'.format(output_prefix_fmt.format(3)),
        transparent=True
    )
    plt.close()

    ######################################################################
    # Plot boxplots of parametric ratios for high-rapidity parameter vectors
    # from *all* parametric polytopes
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 10))
    for m, exp in enumerate(range(2, 6)):
        # Parse output metrics
        logrates = np.loadtxt(filename_fmts['logrates'].format(exp))
        specs = np.loadtxt(filename_fmts['specs'].format(exp))
        rapid = np.loadtxt(filename_fmts['rapid'].format(exp))

        # Ratios of model parameters
        cp_total = logrates[:, 2] - logrates[:, 3]       # cp = b' / d'

        cp_valid = [cp_total]
        nbins = 50
        npoints_per_bin = 1
        start_idx = 2
        xmin = 0
        xmax = specs.max()
        ratio_min = -2 * exp
        ratio_max = 2 * exp
        x_bin_edges = np.linspace(xmin, xmax, nbins + 1)
        for i in range(start_idx, 20):       # For each mismatch position ...
            cp_valid_i = []
            for j in range(nbins):   # ... and each bin along the specificity axis ... 
                # Get indices of parameter vectors for top rapidity values in the 
                # j-th bin
                in_column = (
                    (specs[:, i] >= x_bin_edges[j]) & (specs[:, i] < x_bin_edges[j+1])
                )
                rapid_top_idx = np.argsort(rapid[in_column, i])[-npoints_per_bin:]
                logrates_valid = logrates[in_column, :][rapid_top_idx]
                for k in range(logrates_valid.shape[0]):
                    cp_valid_i.append(logrates_valid[k, 2] - logrates_valid[k, 3])
            cp_valid.append(cp_valid_i)
        sns.boxplot(
            data=cp_valid, orient='v', width=0.7, showfliers=False, linewidth=2,
            ax=axes[m], whis=(5, 95)
        )
        for i in range(4):
            axes[i].patches[0].set_facecolor(sns.color_palette('deep')[4])
            for j in range(1, 21 - start_idx):
                axes[i].patches[j].set_facecolor(colors[m])
            axes[i].set_xticklabels(['All'] + [str(k) for k in range(start_idx, 20)])
            axes[i].set_ylim([ratio_min - 0.2, ratio_max + 0.2])
        axes[m].set_ylabel(r"$\log_{10}(b'/d')$", size=12)
    
    plt.tight_layout()
    plt.savefig(
        '{}-highrapid-boxplot.pdf'.format(output_prefix_fmt.format('all')),
        transparent=True
    )
    plt.close()

##########################################################################
if __name__ == '__main__':
    main()

