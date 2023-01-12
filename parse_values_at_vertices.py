"""
Author:
    Kee-Myoung Nam

Last updated:
    1/12/2023
"""
import sys
import pandas as pd 
import numpy as np

# Parse the Jones et al. 2021 dataset and find min/max bounds for the composite
# cleavage rates 
Jones_filenames = [
    'data/Jones-2021-NatBiotechnol-SI2-Cas9-sgRNA1-{}-{}-seqs.txt'.format(variant, metric)
    for metric in ['cleave_rates', 'ndABAs_exp']
    for variant in ['wtCas9', 'eCas9', 'HypaCas9', 'Cas9HF1']
]
compcleave_values = {}
deaddissoc_values = {}
pseudocount = 1e-8      # Add pseudocount of 1e-8 to each value 
for filename in Jones_filenames:
    if 'cleave_rates' in filename:
        with open(filename) as f:
            f.readline()    # Skip the first line, which contains the on-target sequence
            for line in f:
                _, pattern, rate = line.split()
                if pattern not in compcleave_values:
                    compcleave_values[pattern] = []
                compcleave_values[pattern].append(float(rate) + pseudocount)
    elif 'ndABAs_exp' in filename:
        with open(filename) as f:
            f.readline()    # Skip the first line, which contains the on-target sequence
            for line in f:
                _, pattern, rate = line.split()
                if pattern not in deaddissoc_values:
                    deaddissoc_values[pattern] = []
                # Note that all ndABAs are given in linear scale and needs to 
                # be converted to log-scale (base 10) for comparisons
                deaddissoc_values[pattern].append(np.log10(float(rate)) + pseudocount)

# Add min/max measured values for composite cleavage rates at 100 nM from
# other sources (Chen et al. 2017, Singh et al. 2016, Singh et al. 2018)
compcleave_values['1' * 20] += [0.01609 + pseudocount, 0.6052 + pseudocount]
compcleave_values['1' * 19 + '0'] += [0.004983 + pseudocount, 0.7081 + pseudocount]
compcleave_values['1' * 18 + '00'] += [5.190e-4 + pseudocount, 0.4516 + pseudocount]
compcleave_values['1' * 17 + '000'] += [4.188e-5 + pseudocount, 0.1671 + pseudocount]
compcleave_values['1' * 16 + '0000'] += [1.208e-6 + pseudocount, 8.105e-4 + pseudocount]
compcleave_values['1' * 15 + '00111'] += [1e-7 + pseudocount, 1.662e-5 + pseudocount]
compcleave_values['1' * 13 + '0011111'] += [1e-7 + pseudocount, 0.001270 + pseudocount]
compcleave_values['1' * 11 + '001111111'] += [1e-7 + pseudocount, 9.262e-4 + pseudocount]
compcleave_values['1' * 9 + '00111111111'] += [1.430e-5 + pseudocount, 4.142e-4 + pseudocount]

# Enumerate min/max measured values for every other metric 
Rcompletion_bounds = {    # Liu et al. 2020, Singh et al. 2016, Singh et al. 2018
    '1' * 20:          [0.7 + pseudocount, 14.29 + pseudocount],
    '1' * 19 + '0':    [1.042 + pseudocount, 2.695 + pseudocount],
    '1' * 18 + '00':   [0.2762 + pseudocount, 2.041 + pseudocount],
    '1' * 17 + '000':  [0.01960 + pseudocount, 2.5 + pseudocount],
    '1' * 16 + '0000': [0.02289 + pseudocount, 0.5807 + pseudocount]
}
Rdissolution_bounds = {   # Liu et al. 2020, Singh et al. 2016, Singh et al. 2018 
    '1' * 20:          [0.02222 + pseudocount, 1.2 + pseudocount],
    '1' * 19 + '0':    [0.03782 + pseudocount, 3.442 + pseudocount],
    '1' * 18 + '00':   [0.3704 + pseudocount, 10.58 + pseudocount],
    '1' * 17 + '000':  [0.5 + pseudocount, 10 + pseudocount],
    '1' * 16 + '0000': [9.091 + pseudocount, 19.05 + pseudocount]
}
unbind_bounds = {         # Singh et al. 2016, Singh et al. 2018
    '1' * 8 + '0' * 12: [1.0/18.7 + pseudocount, 1.0/1.72 + pseudocount],
    '1' * 7 + '0' * 13: [1.0/5.42 + pseudocount, 1.0/1.38 + pseudocount],
    '1' * 6 + '0' * 14: [1.0/0.74 + pseudocount, 1.0/0.39 + pseudocount],
    '1' * 5 + '0' * 15: [1.0/0.84 + pseudocount, 1.0/0.26 + pseudocount],
    #'1' * 4 + '0' * 16: [1.0/0.4 + pseudocount, 1.0/0.24 + pseudocount],
    #'0' * 20:           [1.0/1.19 + pseudocount, 1.0/0.2 + pseudocount],
    '0' * 4 + '1' * 16: [1.0/0.54 + pseudocount, 1.0/0.06 + pseudocount],
    '0' * 2 + '1' * 18: [1.0/6.08 + pseudocount, 1.0/0.35 + pseudocount]
}

# Parse the computed values of each metric at the polytope vertices
values_prefix = sys.argv[1]
values_filenames = [
    '{}-{}.tsv'.format(values_prefix, suffix) for suffix in 
    ['compcleave', 'Rcompletion', 'Rdissolution', 'deaddissoc', 'unbind']
]
for filename in values_filenames:
    data = pd.read_csv(filename, sep='\t', index_col=None, header=0)
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    if filename.endswith('compcleave.tsv'):
        for pattern in compcleave_values:
            contained = all(
                j > data_min[pattern] and j < data_max[pattern] for j in compcleave_values[pattern]
            )
            print(
                pattern, contained, 
                min(compcleave_values[pattern]), max(compcleave_values[pattern]),
                data_min[pattern], data_max[pattern]
            )
    elif filename.endswith('Rcompletion.tsv'):
        for pattern in Rcompletion_bounds:
            contained = all(
                j > data_min[pattern] and j < data_max[pattern] for j in Rcompletion_bounds[pattern]
            )
            print(
                pattern, contained, 
                min(Rcompletion_bounds[pattern]), max(Rcompletion_bounds[pattern]),
                data_min[pattern], data_max[pattern]
            )
    elif filename.endswith('Rdissolution.tsv'):
        for pattern in Rdissolution_bounds:
            contained = all(
                j > data_min[pattern] and j < data_max[pattern] for j in Rdissolution_bounds[pattern]
            )
            print(
                pattern, contained, 
                min(Rdissolution_bounds[pattern]), max(Rdissolution_bounds[pattern]),
                data_min[pattern], data_max[pattern]
            )
    elif filename.endswith('deaddissoc.tsv'):
        for pattern in deaddissoc_values:
            contained = all(
                j > data_min[pattern] and j < data_max[pattern] for j in deaddissoc_values[pattern]
            )
            print(
                pattern, contained, 
                min(deaddissoc_values[pattern]), max(deaddissoc_values[pattern]),
                data_min[pattern], data_max[pattern]
            )
    elif filename.endswith('unbind.tsv'):
        for pattern in unbind_bounds:
            contained = all(
                j > data_min[pattern] and j < data_max[pattern] for j in unbind_bounds[pattern]
            )
            print(
                pattern, contained, 
                min(unbind_bounds[pattern]), max(unbind_bounds[pattern]),
                data_min[pattern], data_max[pattern]
            )

