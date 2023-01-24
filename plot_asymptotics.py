"""
Authors:
    Kee-Myoung Nam

Last updated:
    1/25/2023
"""
import sys
import os
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

###############################################################
# Define a 5x4 grid of plots with an additional horizontal colorbar at the bottom
filename_asymp_fmt = 'data/line-3-w6-v2-minusbind-single-asymptotic-{}-asymp-{}.tsv'
filename_exact_fmt = 'data/line-3-w6-v2-minusbind-single-asymptotic-{}-exact-{}.tsv'
length = 20

xlabels = {
    'activity': r'$\log_{10}{(\phi(\mathbf{u}^{\mathrm{P}}))}$',
    'spec_0': r'$\log_{10}{(\phi(\mathbf{u}^{\mathrm{P}}) / \phi(\mathbf{u}^{\{0\}}))}$',
    'rapid': (
        r'$\log_{10}{(\phi(\mathbf{u}^{\mathrm{P}}) \, \sigma_{*}(\mathbf{u}^{\mathrm{P}}) '
        r'/ (\phi(\mathbf{u}^M) \, \sigma_{*}(\mathbf{u}^M)))}$'
    )
}
ylabels = {
    'activity': r"$\log_{10}(1 + k_\varnothing / b)$",
    'spec_0':   r"$\log_{10}(1 + k_\varnothing / b')$",
    'rapid':    r"$\log_{10}((c / c') (\alpha_m / \alpha))$"
}

nmax = 1000
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
exact_probs = np.loadtxt(filename_exact_fmt.format('largematch', 'probs'))
exact_specs = np.loadtxt(filename_exact_fmt.format('largematch', 'specs'))
asymp_activities_specs = np.loadtxt(filename_asymp_fmt.format('largematch', 'activities-specs'))

# Plot the set of exact and asymptotic activities
axes[0].scatter(
    x=exact_probs[:nmax, 0], y=np.power(10.0, asymp_activities_specs[:nmax, 0]),
    color=sns.color_palette()[0], alpha=0.05
)
axes[0].set_xlabel(xlabels['activity'])
axes[0].set_ylabel(ylabels['activity'])
xmin, xmax = axes[0].get_xlim()    # Plot a straight line for y = x
axes[0].plot([xmin, xmax], [xmin, xmax], c='red', linestyle='--')
axes[0].tick_params(axis='both', labelsize=10)

# Plot the set of exact and asymptotic specificities for mismatch position 0
axes[1].scatter(
    x=exact_specs[:nmax, 0], y=asymp_activities_specs[:nmax, 1],
    color=sns.color_palette()[0], alpha=0.05
)
axes[1].set_xlabel(xlabels['spec_0'])
axes[1].set_ylabel(ylabels['spec_0'])
xmin, xmax = axes[1].get_xlim()    # Plot a straight line for y = x
axes[1].plot([xmin, xmax], [xmin, xmax], c='red', linestyle='--')
axes[1].tick_params(axis='both', labelsize=10)

# Print the quantiles of the distribution of exact specificities for all
# other mismatch positions, which should ideally all be close to 1 (or 0
# in log-scale)
for k in range(1, 20):
    deciles = np.quantile(exact_specs[:, k], np.linspace(0.1, 1.0, 10))
    print(deciles)

# Generate the PDF file
plt.tight_layout()
plt.savefig(
    'plots/line-3-w6-v2-minusbind-single-asymptotic-largematch-activities-specs.pdf'
)
plt.close(fig)

# ------------------------------------------------------------------- #
fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(12, 15))
exact_specs = np.loadtxt(filename_exact_fmt.format('smallmismatch', 'specs'))
exact_rapid = np.loadtxt(filename_exact_fmt.format('smallmismatch', 'rapid'))
asymp_tradeoff = np.loadtxt(filename_asymp_fmt.format('smallmismatch', 'rapid'))

# For each mismatch position ...
for i in range(5):
    for j in range(4):
        # ... plot the corresponding set of exact and asymptotic specificity-
        # rapidity tradeoff constants
        k = 4 * i + j
        axes[i, j].scatter(
            x=exact_specs[:nmax, k] + exact_rapid[:nmax, k], y=asymp_tradeoff[:nmax, k],
            color=sns.color_palette()[0], alpha=0.05
        )
        xmin, xmax = axes[i, j].get_xlim()    # Plot a straight line for y = x
        axes[i, j].plot([xmin, xmax], [xmin, xmax], c='red', linestyle='--')
       
        # Configure axes limits
        ymin, ymax = axes[i, j].get_ylim()
        axes[i, j].set_ylim([ymin, ymin + 1.15 * (ymax - ymin)])

        # Introduce plot titles
        text_title_kwargs = {
            'size': 10,
            'xy': (0.97, 0.95),
            'xycoords': 'axes fraction',
            'horizontalalignment': 'right',
            'verticalalignment': 'top'
        }
        axes[i, j].annotate(r'$M = \{{ {} \}}$'.format(k), **text_title_kwargs)
        axes[i, j].tick_params(axis='both', labelsize=8)

# Introduce axes labels
for i in range(5):
    axes[i, 0].set_ylabel(ylabels['rapid'], size=10)
for j in range(4):
    axes[4, j].set_xlabel(xlabels['rapid'], size=10)

# Generate the PDF file
plt.tight_layout()
plt.savefig(
    'plots/line-3-w6-v2-minusbind-single-asymptotic-smallmismatch-rapid.pdf'
)
plt.close(fig)

# ------------------------------------------------------------------- #
fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(12, 15))
optimize_logrates = np.loadtxt(
    'data/line-3-w6-v2-minusbind-single-rapid-optimize-smallmismatch-logrates.tsv'
)
optimize_data = np.loadtxt(
    'data/line-3-w6-v2-minusbind-single-rapid-optimize-smallmismatch-asymp-rapid.tsv'
)

n = 49 * 100
step = 49
xrange = np.linspace(-1, 1, 49)

# For each mismatch position ... 
for i in range(5):
    for j in range(4):
        k = 4 * i + j
        # ... and each fixed set of parameter values (other than d) ... 
        for m in range(0, n, step):
            data_m = optimize_data[m:m+step, k]
            axes[i, j].plot(xrange, data_m[::-1])    # Keep colors varied

        # Configure axes limits
        ymin, ymax = axes[i, j].get_ylim()
        axes[i, j].set_ylim([ymin, ymin + 1.15 * (ymax - ymin)])
        ymin, ymax = axes[i, j].get_ylim()    # Plot a straight line for y = 0
        axes[i, j].plot([0, 0], [ymin, ymax], c='red', linestyle='--')

        # Introduce plot titles and label all axes
        text_title_kwargs = {
            'size': 10,
            'xy': (0.97, 0.95),
            'xycoords': 'axes fraction',
            'horizontalalignment': 'right',
            'verticalalignment': 'top'
        }
        axes[i, j].annotate(r'$M = \{{ {} \}}$'.format(k), **text_title_kwargs)
        axes[i, j].tick_params(axis='both', labelsize=8)

# Introduce axes labels
for i in range(5):
    axes[i, 0].set_ylabel(ylabels['rapid'], size=10)
for j in range(4):
    axes[4, j].set_xlabel(r'$\log{(b/d)}$', size=10)

# Generate the PDF file
plt.tight_layout()
plt.savefig(
    'plots/line-3-w6-v2-minusbind-single-rapid-optimize-smallmismatch-asymp-rapid.pdf'
)
plt.close(fig)

"""
for family in ['single', 'distal']:
for metric in ['rapid', 'deaddissoc']:
for exp in [2, 4, 6]:
    for lim in ['largematch', 'smallmismatch']:
        # Parse corresponding cleavage statistics and asymptotics
        if exp == 6:
            specs = np.loadtxt(filename_exact_fmt.format(exp, family, lim, 'spec'))
            stats = np.loadtxt(filename_exact_fmt.format(exp, family, lim, metric))
            asymp = np.loadtxt(filename_asymp_fmt.format(exp, family, lim, metric))
        else:
            specs = np.loadtxt(filename_exact_fmt.format(exp, family, 'exp6-' + lim, 'spec'))
            stats = np.loadtxt(filename_exact_fmt.format(exp, family, 'exp6-' + lim, metric))
            asymp = np.loadtxt(filename_asymp_fmt.format(exp, family, 'exp6-' + lim, metric))
        if specs.shape[0] > nmax:
            specs = specs[:nmax, :]
            stats = stats[:nmax, :]
                asymp = asymp[:nmax, :]
            fig, axd = plt.subplot_mosaic(
                mosaic, gridspec_kw=gridspec_kw, figsize=(10, 12),
                constrained_layout=True
            )
            # For each mismatch position ...
            for i in range(5):
                for j in range(4): 
                    # ... plot the corresponding set of exact and asymptotic tradeoff
                    # constants 
                    k = 4 * i + j
                    key = '{}{}'.format(i, j)
                    axd[key].scatter(
                        x=(specs[:, k] + stats[:, k]), y=asymp[:, k],
                        color=sns.color_palette()[0],
                        alpha=0.05
                    )
                    
                    # Plot a straight line for y = x
                    xmin, xmax = axd[key].get_xlim()
                    axd[key].plot([xmin, xmax], [xmin, xmax], c='red', linestyle='--')

                    # Configure axes limits
                    ymin, ymax = axd[key].get_ylim()
                    axd[key].set_ylim([ymin, ymin + 1.15 * (ymax - ymin)])

                        # Introduce plot titles and label all axes
                        text_title_kwargs = {
                            'size': 10,
                            'xy': (0.97, 0.95),
                            'xycoords': 'axes fraction',
                            'horizontalalignment': 'right',
                            'verticalalignment': 'top'
                        }
                        if family == 'single':
                            axd[key].annotate(r'$M = \{{ {} \}}$'.format(k), **text_title_kwargs)
                        elif family == 'distal':
                            axd[key].annotate(r'$M = [ {} ]$'.format(k), **text_title_kwargs)
                        axd[key].set_xlabel(xlabels[metric], size=8)
                        axd[key].set_ylabel(ylabels[(family, metric, lim, k)], size=10)
                        axd[key].tick_params(axis='both', labelsize=8)

                # Generate the PDF file
                plt.savefig(
                    'plots/asymptotics/line-{}-unbindingunity-{}-asymptotics-{}-{}.pdf'.format(
                        exp, family, lim if exp == 6 else 'exp6-' + lim, metric
                    )
                )
                plt.close(fig)
"""
