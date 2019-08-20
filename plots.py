"""
Various functions for plotting.

Author:
    Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
Last updated:
    8/19/2019
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_histograms(A, outpdf, nrows=None, ncols=None, figsize=(18, 18),
                    func=None, xlog=True, ylog=True, xlabel=None, ylabel=None,
                    **kwargs):
    """
    Given an M x N array `A`, generate an array of N - 1 histograms, 
    showing the (bivariate) distribution of values in the first column
    vs. values in each subsequent column.
    """
    # Initialize figure and array of subplots
    if nrows is None or ncols is None:
        nrows = 1
        ncols = A.shape[1] - 1
    else:
        if nrows * ncols != A.shape[1] - 1:
            raise ValueError("Invalid numbers of rows/columns")
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # Set `func` to identity if not given
    if func is None:
        func = lambda x, y: y

    # Plot each histogram with accompanying colorbar
    for i in range(nrows):
        for j in range(ncols):
            x = np.log10(A[:,0]) if xlog else A[:,0]
            y = func(x, np.log10(A[:,i*ncols+j+1])) if ylog else func(A[:,0], A[:,i*ncols+j+1])
            c = axes[i,j].hexbin(x, y, bins='log', **kwargs)
            plt.colorbar(c, ax=axes[i,j])

    # Label y-axes of leftmost plots
    for i in range(nrows):
        axes[i,0].set_ylabel(ylabel)

    # Label x-axes of bottommost plots
    for j in range(ncols):
        axes[-1,j].set_xlabel(xlabel)

    # Save figure to file
    plt.tight_layout()
    plt.savefig(outpdf)

###########################################################
def plot_scatter(A, outpdf, figsize=(8, 6), log=True, xlabel=None,
                 ylabel=None, **kwargs):
    """
    Given an M x N array `A`, generate a scatter-plot of points `(j, A[i,j])`
    for each row `i`.
    """
    # Initialize figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each row of A as a single curve
    ax.scatter(
        np.tile(np.arange(A.shape[1]), A.shape[0]),
        np.log10(A.reshape((1, -1))) if log else A.reshape((1, -1)),
        rasterized=True
    )

    # Label axes
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Save figure to file
    plt.savefig(outpdf)

###########################################################
if __name__ == '__main__':
    data = np.loadtxt('conf2-3-cleavage.tsv', comments=None, delimiter='\t')

    # Plot efficiency-specificity histograms
    plot_histograms(
        data, 'test-hist.pdf', nrows=5, ncols=4, figsize=(18, 18),
        func=lambda x, y: x - y, xlog=True, ylog=True,
        xlabel='Log efficiency', ylabel='Log specificity',
        cmap=plt.get_cmap('Blues'), gridsize=30
    )

    # Plot cleavage probability as a function of number of mismatches
    plot_scatter(
        data, 'test-curves.pdf', figsize=(8, 6), log=True,
        xlabel='Number of PAM-distal mismatches', ylabel='Cleavage probability'
    )

