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
        axes[i, j].tick_params(axis='both', labelsize=10)

# Introduce axes labels
for i in range(5):
    axes[i, 0].set_ylabel(ylabels['rapid'], size=10)
for j in range(4):
    axes[4, j].set_xlabel(xlabels['rapid'], size=12)

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
        axes[i, j].tick_params(axis='both', labelsize=10)

# Introduce axes labels
for i in range(5):
    axes[i, 0].set_ylabel(ylabels['rapid'], size=10)
for j in range(4):
    axes[4, j].set_xlabel(r'$\log{(b/d)}$', size=12)

# Generate the PDF file
plt.tight_layout()
plt.savefig(
    'plots/line-3-w6-v2-minusbind-single-rapid-optimize-smallmismatch-asymp-rapid.pdf'
)
plt.close(fig)

