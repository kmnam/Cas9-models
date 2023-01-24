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
sys.path.append(os.path.abspath('../boundaries/python'))
from boundaries import Boundary2D

###############################################################
filename_fmt = 'data/boundaries/line-{}-single-{}-mm{}-boundary-final-simplified.txt'

# Define x-axis and y-axis labels for each class of boundaries 
xlabels = {
    'activity':   r'$\phi(\mathbf{u}^{\mathrm{P}})$',
    'speed':      r'$\sigma_*(\mathbf{u}^{\mathrm{P}})$',
    'rapid':      r'$\log_{10}{(\phi(\mathbf{u}^{\mathrm{P}}) / \phi(\mathbf{u}^{M}))}$', 
    'livedissoc': r'$\log_{10}{(\phi(\mathbf{u}^{\mathrm{P}}) / \phi(\mathbf{u}^{M}))}$',
    'deaddissoc': r'$\log_{10}{(\phi(\mathbf{u}^{\mathrm{P}}) / \phi(\mathbf{u}^{M}))}$'
}
ylabels = {
    'activity':   xlabels['rapid'],
    'speed':      xlabels['rapid'],
    'rapid':      r'$\log_{10}{(\sigma_*(\mathbf{u}^{\mathrm{P}}) / \sigma_*(\mathbf{u}^{M}))}$', 
    'livedissoc': r'$\log_{10}{(\sigma_\varnothing(\mathbf{u}^{M}) / \sigma_\varnothing(\mathbf{u}^{\mathrm{P}}))}$',
    'deaddissoc': r'$\log_{10}{(\sigma(\mathbf{u}^{M}) / \sigma(\mathbf{u}^{\mathrm{P}}))}$'
}

text_kwargs = {
    'size': 10,
    'xy': (0.97, 0.95),
    'xycoords': 'axes fraction',
    'horizontalalignment': 'right',
    'verticalalignment': 'top'
}
suffixes = [
    '3-w6-v2-minusbind', '3-w6-minusbind',
    '4-w8-v2-minusbind', '4-w8-minusbind',
    '5-w10-v2-minusbind', '5-w10-minusbind'
]
linestyles = ['-', '--', '-', '--', '-', '--']
linealphas = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
colors = [
    sns.color_palette('colorblind')[0],
    sns.color_palette('colorblind')[0],
    sns.color_palette('colorblind')[1],
    sns.color_palette('colorblind')[1],
    sns.color_palette('colorblind')[2],
    sns.color_palette('colorblind')[2]
]
# For each class of boundaries ...
for name in ['activity', 'speed', 'rapid', 'deaddissoc']:
    # Set up a new 5x4 collection of axes for each metric 
    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(11, 13))
    for m, run_type in enumerate(suffixes):
        # Skip over speed boundaries for larger polytopes 
        if name == 'speed' and not run_type.startswith('3'):
            continue
        # For each mismatch position ...
        for i in range(5):
            for j in range(4):
                # ... parse the corresponding boundary and plot it 
                k = 4 * i + j
                filename = filename_fmt.format(run_type, name, k)
                try:
                    b = Boundary2D.from_file(filename)
                except (FileNotFoundError, RuntimeError):
                    continue
                b.plot(
                    axes[i, j], color=colors[m], interior_color=None, linewidth=2,
                    linestyle=linestyles[m], linealpha=linealphas[m]
                )
                # Get the maximum value of the sum of the two metrics
                maxval = b.points[b.vertices, :].sum(axis=1).max()
                maxidx = b.points[b.vertices, :].sum(axis=1).argmax()

    # Equalize axes limits
    xmin = min(axes[i, j].get_xlim()[0] for i in range(5) for j in range(4))
    xmax = max(axes[i, j].get_xlim()[1] for i in range(5) for j in range(4))
    ymin = min(axes[i, j].get_ylim()[0] for i in range(5) for j in range(4))
    ymax = max(axes[i, j].get_ylim()[1] for i in range(5) for j in range(4))

    # Introduce plot titles and label all axes
    for i in range(5):
        for j in range(4):
            k = 4 * i + j
            axes[i, j].set_xlim([xmin, xmax])
            axes[i, j].set_ylim([ymin, ymin + (ymax - ymin) * 1.1])
            axes[i, j].annotate(r'$M = \{{ {} \}}$'.format(k), **text_kwargs)
            axes[i, j].tick_params(axis='both', labelsize=9)
    for i in range(5):
        axes[i, 0].set_ylabel(ylabels[name], size=12)
    for j in range(4):
        axes[4, j].set_xlabel(xlabels[name], size=12)

    # Generate the PDF file
    plt.tight_layout()
    plt.savefig(
        'plots/line-single-{}-boundaries.pdf'.format(name),
        transparent=True
    )
    plt.close()

# Define a 4x4 grid of plots for inclusion in the main text
suffixes = [
    '3-w6-v2-minusbind', '3-w6-minusbind',
    '4-w8-v2-minusbind', '4-w8-minusbind',
    '5-w10-v2-minusbind', '5-w10-minusbind'
]
linestyles = ['-', '--', '-', '--', '-', '--']
linealphas = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
colors = [
    sns.color_palette('colorblind')[0],
    sns.color_palette('colorblind')[0],
    sns.color_palette('colorblind')[1],
    sns.color_palette('colorblind')[1],
    sns.color_palette('colorblind')[2],
    sns.color_palette('colorblind')[2]
]
idx_main = np.array([
    [0, 6, 12, 19],
    [4, 9, 14, 19],
    [0, 6, 12, 19],
    [2, 10, 16, 19]
])
fig_main, axes_main = plt.subplots(nrows=4, ncols=4, figsize=(11, 10))
for i, run_type in enumerate(suffixes):
    # For each class of boundaries ... 
    for j, name in enumerate(['activity', 'speed', 'rapid', 'deaddissoc']):
        # (Skip over speed boundaries for larger polytopes)
        if name == 'speed' and not run_type.startswith('3'):
            continue
        # ... and for each selected mismatch position ...
        for k, idx in enumerate(idx_main[j, :]):
            # ... parse the corresponding boundary and plot it 
            filename = filename_fmt.format(run_type, name, idx)
            try:
                b = Boundary2D.from_file(filename)
            except (FileNotFoundError, RuntimeError):
                continue
            b.plot(
                axes_main[j, k], color=colors[i], interior_color=None,
                linewidth=2, linestyle=linestyles[i], linealpha=linealphas[i]
            )
            
for j, name in enumerate(['activity', 'speed', 'rapid', 'deaddissoc']):
    for k, idx in enumerate(idx_main[j, :]):
        # Configure axes limits
        ymin, ymax = axes_main[j, k].get_ylim()
        axes_main[j, k].set_ylim([ymin, ymin + 1.1 * (ymax - ymin)])
        
        # Introduce plot titles and label all axes
        text_kwargs = {
            'size': 10,
            'xy': (0.97, 0.95),
            'xycoords': 'axes fraction',
            'horizontalalignment': 'right',
            'verticalalignment': 'top'
        }
        axes_main[j, k].annotate(r'$M = \{{ {} \}}$'.format(idx), **text_kwargs)
        axes_main[j, k].set_xlabel(xlabels[name], size=12)
        axes_main[j, k].tick_params(axis='both', labelsize=9)
    axes_main[j, 0].set_ylabel(ylabels[name], size=12)

# Equalize axes limits along each row
xmin = [min(axes_main[j, k].get_xlim()[0] for k in range(4)) for j in range(4)]
xmax = [max(axes_main[j, k].get_xlim()[1] for k in range(4)) for j in range(4)]
ymin = [min(axes_main[j, k].get_ylim()[0] for k in range(4)) for j in range(4)]
ymax = [max(axes_main[j, k].get_ylim()[1] for k in range(4)) for j in range(4)]
for j in range(4):
    for k in range(4):
        axes_main[j, k].set_xlim([
            xmin[j] - 0.05 * abs(xmax[j] - xmin[j]), xmax[j]
        ])
        axes_main[j, k].set_ylim([
            ymin[j] - 0.05 * abs(ymax[j] - ymin[j]), ymax[j]
        ])

# Fix tick labels for rapidity plots 
#for k in range(4):
#    axes_main[2, k].set_yticks([-4, -2, 0, 2, 4, 6])

# Generate the PDF file
plt.tight_layout(pad=4, h_pad=1.04, w_pad=1.04)
plt.savefig(
    'plots/line-single-boundaries-main.pdf',
    transparent=True
)
plt.close()
