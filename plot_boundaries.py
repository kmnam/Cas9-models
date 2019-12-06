"""
Plots specificity vs. speed ratio boundaries.

Authors:
    Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
Last updated:
    12/5/2019
"""

import sys
sys.path.append('/Users/kmnam/Dropbox/gene-regulation/projects/boundaries/python')
import numpy as np
import matplotlib.pyplot as plt
from boundaries import Boundary2D

def get_tradeoffs(boundary):
    """
    Given a Boundary2D object, learn the linear tradeoff(s) above (and below) 
    the boundary points.

    This function should only be applied to boundaries that clearly exhibit
    linear tradeoffs above (and below) the boundary points.
    """
    # Identify the "upper right" point (maximum x-coordinate, breaking 
    # ties by whichever point has the greatest y-coordinate)
    vertices = boundary.points[boundary.vertices, :]
    max_x = vertices[:,0].max()
    right = vertices[np.nonzero(vertices[:,0] + 1e-5 > max_x), :].reshape(-1, 2)
    upper_right = right[right[:,1].argmax(), :].ravel()

    # Run through the boundary points and get the most negative slope
    upper_slope = np.inf
    for i in range(vertices.shape[0]):
        if vertices[i,0] - upper_right[0] != 0:
            m = (vertices[i,1] - upper_right[1]) / (vertices[i,0] - upper_right[0])
            if upper_slope > m:
                upper_slope = m

    # Identify the "lower left" point (minimum x-coordinate, breaking
    # ties by whichever point has the least y-coordinate)
    vertices = boundary.points[boundary.vertices, :]
    min_x = vertices[:,0].min()
    left = vertices[np.nonzero(vertices[:,0] - 1e-5 < min_x), :].reshape(-1, 2)
    lower_left = left[left[:,1].argmin(), :].ravel()

    # Run through the boundary points and get the most negative slope
    lower_slope = np.inf
    for i in range(vertices.shape[0]):
        if vertices[i,0] - lower_left[0] != 0:
            m = (vertices[i,1] - lower_left[1]) / (vertices[i,0] - lower_left[0])
            if lower_slope > m:
                lower_slope = m

    # Return the tradeoff functions
    upper_tradeoff = lambda a: upper_right[1] + upper_slope * (a - upper_right[0])
    lower_tradeoff = lambda a: lower_left[1] + lower_slope * (a - lower_left[0])
    return upper_tradeoff, lower_tradeoff

##############################################################
def plot_boundaries(files, mismatches, outpdf, plot_lower_tradeoff=True,
                    plot_upper_tradeoff=True):
    """
    Plot a collection of six boundaries of specificity vs. speed ratio 
    regions for the one- and two-conformation models. 
    """
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 8))
    for i, filename in enumerate(files):
        # Plot each boundary
        b = Boundary2D.from_file(filename)
        b.plot(
            axes[i//3, i%3], boundary_color='green', plot_interior=False,
            shade_interior=True
        )
        axes[i//3, i%3].set_title('{} mismatches'.format(mismatches[i]), size=14)

        # Plot the tradeoff inferred from each boundary
        upper_tradeoff, lower_tradeoff = get_tradeoffs(b)
        if plot_upper_tradeoff:
            axes[i//3, i%3].plot(
                [0, b.points[:,0].max()],
                [upper_tradeoff(0), upper_tradeoff(b.points[:,0].max())],
                color='red', linestyle='--'
            )
            axes[i//3, i%3].annotate(
                '{:.5f}, {:.5f}'.format(
                    upper_tradeoff(0),
                    upper_tradeoff(1) - upper_tradeoff(0)
                ),
                (0.95, 0.9),
                xytext=None,
                xycoords='axes fraction',
                size=14,
                horizontalalignment='right'
            )
        if plot_lower_tradeoff:
            axes[i//3, i%3].plot(
                [0, b.points[:,0].max()],
                [lower_tradeoff(0), lower_tradeoff(b.points[:,0].max())],
                color='red', linestyle='--'
            )
            axes[i//3, i%3].annotate(
                '{:.5f}, {:.5f}'.format(
                    lower_tradeoff(0),
                    lower_tradeoff(1) - lower_tradeoff(0)
                ),
                (0.95, 0.8) if plot_upper_tradeoff else (0.95, 0.9),
                xytext=None,
                xycoords='axes fraction',
                size=14,
                horizontalalignment='right'
            )

    # Label each set of axes
    for i in range(3):
        axes[-1, i].set_xlabel(r'$\log{\,\psi(\sigma)}$', size=14)
    for i in range(2): 
        axes[i, 0].set_ylabel(r'$\log{\,\omega(\sigma)}$', size=14)

    plt.savefig(outpdf)

##############################################################
plot_boundaries(
    [
        'conf1/boundaries/conf1-mm1-boundary-pass226.txt',
        'conf1/boundaries/conf1-mm3-boundary-pass226.txt',
        'conf1/boundaries/conf1-mm5-boundary-pass225.txt',
        'conf1/boundaries/conf1-mm10-boundary-pass221.txt',
        'conf1/boundaries/conf1-mm15-boundary-pass238.txt',
        'conf1/boundaries/conf1-mm20-boundary-pass235.txt'
    ],
    [1, 3, 5, 10, 15, 20],
    'conf1/boundaries/conf1-boundaries.pdf',
    plot_lower_tradeoff=False, plot_upper_tradeoff=True
)
plot_boundaries(
    [
        'conf2/boundaries/conf2-mm1-boundary-pass274.txt',
        'conf2/boundaries/conf2-mm3-boundary-pass334.txt',
        'conf2/boundaries/conf2-mm5-boundary-pass343.txt',
        'conf2/boundaries/conf2-mm10-boundary-pass255.txt',
        'conf2/boundaries/conf2-mm15-boundary-pass310.txt',
        'conf2/boundaries/conf2-mm20-boundary-pass318.txt'
    ],
    [1, 3, 5, 10, 15, 20],
    'conf2/boundaries/conf2-boundaries.pdf',
    plot_lower_tradeoff=False, plot_upper_tradeoff=True
)
