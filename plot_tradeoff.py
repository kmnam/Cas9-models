"""
Plots specificity vs. speed ratio.  

Author:
    Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
Last updated:
    11/26/2019
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
    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(18, 18))
    for i in range(5):
        for j in range(4):
            c = axes[i,j].hexbin(
                specs[:,i*4+j], ratios[:,i*4+j], bins='log', gridsize=30,
                cmap=plt.get_cmap('Blues')
            )
            plt.colorbar(c, ax=axes[i,j])
    for i in range(5):
        axes[i,0].set_ylabel('Log speed ratio')
    for j in range(4):
        axes[-1,j].set_xlabel('Log specificity')
    plt.tight_layout()
    plt.savefig(sys.argv[3])

