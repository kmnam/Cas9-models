"""
Author:
    Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
Last updated:
    2/25/2021
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

###########################################################
if __name__ == '__main__':
    # Load in cleavage specificity and unbinding specificity matrices
    probs = np.loadtxt(sys.argv[1], comments=None, delimiter='\t')
    rates = np.loadtxt(sys.argv[2], comments=None, delimiter='\t')
    cleavage_activity = probs[:,0]
    cleavage_specificity = probs[:,0].reshape((-1, 1)) / probs[:,1:]
    unbinding_specificity = rates[:,1:] / rates[:,0].reshape((-1, 1))
    
    # Plot activity against specificity
    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(14, 17))
    for i in range(5):
        for j in range(4):
            axes[i,j].scatter(
                cleavage_activity, np.log10(cleavage_specificity[:,i*4+j]),
                alpha=0.05, rasterized=True
            )
            axes[i,j].set_title('{} distal mms'.format(i*4 + j + 1), size=14)
    for i in range(5):
        axes[i,0].set_ylabel('Cleavage specificity', size=14)
    for j in range(4):
        axes[-1,j].set_xlabel('On-target activity', size=14)
    plt.tight_layout()
    plt.savefig(sys.argv[3] + '-activity-vs-specificity.pdf')

    # Plot cleavage specificity against unbinding specificity 
    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(5, 17))
    for i in range(5):
        for j in range(4):
            axes[i,j].scatter(
                np.log10(cleavage_specificity[:,i*4+j]),
                np.log10(unbinding_specificity[:,i*4+j]),
                alpha=0.05, rasterized=True
            )
            axes[i,j].set_title('{} distal mms'.format(i*4 + j + 1), size=14)
    for i in range(5):
        axes[i,0].set_ylabel('Unbinding specificity', size=14)
    for j in range(4):
        axes[-1,j].set_xlabel('Cleavage specificity', size=14)
    plt.tight_layout()
    plt.savefig(sys.argv[3] + '-cleavage-vs-unbinding.pdf')

