"""
Authors:
    Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
Last updated:
    12/2/2019
"""
import numpy as np
import matplotlib.pyplot as plt
from simulations import GridGraph

np.random.seed(1234567890)

##############################################################
if __name__ == '__main__':
    graph = GridGraph(5)    # Define a grid graph of length 5
    pattern = '11110'
    nsim = 100000
   
    # Randomly sample 40 parameter combinations 
    params = 10.0 ** (-1 + 2 * np.random.random((20, 6)))
    stats = np.zeros((20, 2))

    # Set up a figure with 10 subplots
    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(16, 13))

    # Run 10 simulations ...
    for i in range(params.shape[0]):
        rates = {
            'b': params[i,0], 'd': params[i,1], 'bp': params[i,2],
            'dp': params[i,3], 'k': params[i,4], 'l': params[i,5]
        }
        data = graph.simulate(nsim, pattern, init=('A',0), **rates, kcat=1.0, kdis=1.0)

        # Compute cleavage probability
        prob = data[:,0].sum() / nsim

        # If cleavage was never observed, first passage time is infinite
        if prob == 0:
            time = np.inf
        else:    # Otherwise, plot distribution of first passage times
            time = data[np.nonzero(data[:,0]), 1].mean()
            axes[i//5, i%5].hist(data[np.nonzero(data[:,0]), 1].ravel())
            axes[i//5, i%5].annotate(
                '{:.2f}'.format(time),
                (0.95, 0.9),
                xytext=None,
                xycoords='axes fraction',
                size=14,
                horizontalalignment='right'
            )
        stats[i,0] = prob
        stats[i,1] = time

    plt.savefig('grid-graph-simulations.pdf')
    np.savetxt('grid-graph-simulations.tsv', np.hstack((params, stats)), delimiter='\t')
