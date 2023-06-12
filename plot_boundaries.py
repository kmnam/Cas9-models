"""
Plot all numerically obtained specificity-rapidity boundaries. 

Authors:
    Kee-Myoung Nam

Last updated:
    6/12/2023
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
sys.path.append(os.path.abspath('../boundaries/python'))
sys.path.append(os.path.abspath('../markov-digraphs'))
from boundaries import Boundary2D
from pygraph import PreciseLineGraph

length = 20

#######################################################################
def main():
    # Define x-axis and y-axis labels for each class of boundaries 
    xlabel = r'$\log_{10}{(\mathrm{Spec}(\mathbf{u}^{\{m\}}))}$'
    ylabel = r'$\log_{10}{(\mathrm{Rapid}(\mathbf{u}^{\{m\}}))}$'
    filename_fmt = 'data/boundaries/{}-rapid-mm{}-boundary-final.txt'
    text_kwargs = {
        'size': 10,
        'xy': (0.98, 0.96),
        'xycoords': 'axes fraction',
        'horizontalalignment': 'right',
        'verticalalignment': 'top'
    }
    prefixes = [
        'line-5-combined-single',
        'line-4-combined-single',
        'line-3-combined-single',
        'line-2-combined-single'
    ]
    linestyles = ['-', '-', '-', '-']
    linealphas = [1.0, 1.0, 1.0, 1.0]
    linewidths = [2.0, 2.0, 2.0, 2.0] 
    colors = [
        sns.color_palette('colorblind')[2],
        sns.color_palette('colorblind')[1],
        sns.color_palette('colorblind')[0],
        sns.color_palette('colorblind')[3]
    ]
    fillcolors = [
        tuple(x + (1 - x) * 0.6 for x in sns.color_palette('pastel')[2]),
        tuple(x + (1 - x) * 0.6 for x in sns.color_palette('pastel')[1]),
        tuple(x + (1 - x) * 0.6 for x in sns.color_palette('pastel')[0]),
        tuple(x + (1 - x) * 0.6 for x in sns.color_palette('pastel')[3])
    ]
    
    # Set up a 5x4 collection of axes
    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(12, 14))
    for m, prefix in enumerate(prefixes):
        # For each mismatch position ...
        for i in range(5):
            for j in range(4):
                # ... parse the corresponding boundary and plot it 
                k = 4 * i + j
                filename = filename_fmt.format(prefix, k)
                try:
                    b = Boundary2D.from_file(filename)
                except (FileNotFoundError, RuntimeError):
                    continue
                b.plot(
                    axes[i, j], color=colors[m], interior_color=fillcolors[m],
                    linewidth=linewidths[m], linestyle=linestyles[m],
                    linealpha=linealphas[m]
                )
    
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
            axes[i, j].annotate(r'$m = {}$'.format(k), **text_kwargs)
            axes[i, j].tick_params(axis='both', labelsize=9)
    for i in range(5):
        axes[i, 0].set_ylabel(ylabel, size=12)
    for j in range(4):
        axes[4, j].set_xlabel(xlabel, size=12)

    # Generate the PDF file
    plt.tight_layout()
    plt.savefig(
        'plots/line-single-rapid-boundaries.pdf',
        transparent=True
    )
    plt.close()

    # Define a 2x2 grid of plots for inclusion in the main text
    idx_main = np.array([
        [4, 9],
        [14, 19]
    ])
    fig_main, axes_main = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
    for i, prefix in enumerate(prefixes):
        # For each class of boundaries ... 
        for j in range(2):
            for k in range(2):
                # ... and for each selected mismatch position ...
                idx = idx_main[j, k]
                # ... parse the corresponding boundary and plot it 
                filename = filename_fmt.format(prefix, idx)
                try:
                    b = Boundary2D.from_file(filename)
                except (FileNotFoundError, RuntimeError):
                    continue
                b.plot(
                    axes_main[j, k], color=colors[i], interior_color=fillcolors[i],
                    linewidth=linewidths[i], linestyle=linestyles[i],
                    linealpha=linealphas[i]
                )
                
    for j in range(2):
        for k in range(2):
            idx = idx_main[j, k]
            ymin, ymax = axes_main[j, k].get_ylim()
            axes_main[j, k].set_ylim([ymin, ymin + 1.1 * (ymax - ymin)])
            
            # Introduce plot titles and label all axes
            text_kwargs = {
                'size': 10,
                'xy': (0.98, 0.96),
                'xycoords': 'axes fraction',
                'horizontalalignment': 'right',
                'verticalalignment': 'top'
            }
            axes_main[j, k].annotate(r'$m = {}$'.format(idx), **text_kwargs)
            axes_main[j, k].tick_params(axis='both', labelsize=9)
        axes_main[j, 0].set_ylabel(ylabel, size=12)
    for k in range(2):
        axes_main[-1, k].set_xlabel(xlabel, size=12)

    # Equalize axes limits along each row
    xmin = [min(axes_main[j, k].get_xlim()[0] for k in range(2)) for j in range(2)]
    xmax = [max(axes_main[j, k].get_xlim()[1] for k in range(2)) for j in range(2)]
    ymin = [min(axes_main[j, k].get_ylim()[0] for k in range(2)) for j in range(2)]
    ymax = [max(axes_main[j, k].get_ylim()[1] for k in range(2)) for j in range(2)]
    for j in range(2):
        for k in range(2):
            axes_main[j, k].set_xlim([xmin[j], xmax[j]])
            axes_main[j, k].set_ylim([
                ymin[j] - 0.05 * abs(ymax[j] - ymin[j]), ymax[j]
            ])

    # Generate the PDF file
    plt.tight_layout()
    plt.savefig(
        'plots/line-single-boundaries-main.pdf',
        transparent=True
    )
    plt.close()

#######################################################################
if __name__ == '__main__':
    main()
