"""
Plots specificity vs. speed ratio boundaries.

Authors:
    Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
Last updated:
    12/2/2019
"""

import sys
sys.path.append('/Users/kmnam/Dropbox/gene-regulation/projects/boundaries/python')
import matplotlib.pyplot as plt
from boundaries import Boundary2D

if __name__ == '__main__':
    files = [
        'conf2/boundaries/conf2-mm1-boundary-pass274.txt',
        'conf2/boundaries/conf2-mm3-boundary-pass334.txt',
        'conf2/boundaries/conf2-mm5-boundary-pass343.txt',
        'conf2/boundaries/conf2-mm10-boundary-pass255.txt',
        'conf2/boundaries/conf2-mm15-boundary-pass310.txt',
        'conf2/boundaries/conf2-mm20-boundary-pass318.txt'
    ]
    mismatches = [1, 3, 5, 10, 15, 20]

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 8))
    for i, filename in enumerate(files):
        # Plot each boundary
        b = Boundary2D.from_file(filename)
        b.plot(axes[i//3, i%3], interior_color='white', boundary_color='green')

        # Plot the tradeoff inferred from each boundary
        tradeoff_constant = b.points.sum(axis=1).max()
        func = lambda x : tradeoff_constant - x
        axes[i//3, i%3].plot(
            [0, b.points[:,0].max()],
            [func(0), func(b.points[:,0].max())],
            color='red', linestyle='--'
        )
        axes[i//3, i%3].annotate(
            '{:.7f}'.format(tradeoff_constant),
            (0.95, 0.9),
            xytext=None,
            xycoords='axes fraction',
            size=14,
            horizontalalignment='right'
        )
        axes[i//3, i%3].set_title('{} mismatches'.format(mismatches[i]), size=14)

    # Label each set of axes
    for i in range(3):
        axes[-1, i].set_xlabel(r'$\log{\,\psi(\sigma)}$', size=14)
    for i in range(2): 
        axes[i, 0].set_ylabel(r'$\log{\,\omega(\sigma)}$', size=14)

    plt.savefig('conf2/boundaries/conf2-boundaries.pdf')
