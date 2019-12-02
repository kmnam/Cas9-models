"""
Plots specificity vs. speed ratio.  

Author:
    Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
Last updated:
    12/2/2019
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

###########################################################
if __name__ == '__main__':
    # Load in specificity and speed ratio matrices
    specs = np.loadtxt(sys.argv[1], comments=None, delimiter='\t')
    ratios = np.loadtxt(sys.argv[2], comments=None, delimiter='\t')
    
    # Plot specificity against speed ratio
    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(14, 17))
    for i in range(5):
        for j in range(4):
            # Get max(specificity + speed ratio)
            m = np.max(specs[:,i*4+j] + ratios[:,i*4+j])
            axes[i,j].scatter(
                specs[:,i*4+j], ratios[:,i*4+j], alpha=0.1, rasterized=True
            )
            tradeoff = lambda x: m - x
            axes[i,j].plot(
                [np.min(specs[:,i*4+j]), np.max(specs[:,i*4+j])],
                [tradeoff(np.min(specs[:,i*4+j])), tradeoff(np.max(specs[:,i*4+j]))],
                color='red', linestyle='--'
            )
            axes[i,j].annotate(
                '{:.2f}'.format(m),
                (0.95, 0.9),
                xytext=None,
                xycoords='axes fraction',
                size=14,
                horizontalalignment='right'
            )
            axes[i,j].set_title('{} mismatches'.format(i*4 + j + 1), size=14)
    for i in range(5):
        axes[i,0].set_ylabel(r'$\log{\,\omega(\sigma)}$', size=14)
    for j in range(4):
        axes[-1,j].set_xlabel(r'$\log{\,\psi(\sigma)}$', size=14)
    plt.tight_layout()
    plt.savefig(sys.argv[3])

