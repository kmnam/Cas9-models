"""
Author:
    Kee-Myoung Nam

Last updated:
    3/3/2023
"""
import sys
import pandas as pd 
import numpy as np

#############################################################################
# Parse the Jones et al. 2021 dataset of overall cleavage rates and apparent 
# binding affinities
Jones_filenames = [
    'data/Jones-2021-NatBiotechnol-SI2-Cas9-sgRNA1-{}-{}-seqs.txt'.format(variant, metric)
    for metric in ['cleave_rates', 'ndABAs_exp']
    for variant in ['wtCas9', 'eCas9', 'HypaCas9', 'Cas9HF1']
]
ocr_values = {}
aba_values = {}
pseudocount = 1e-8      # Add pseudocount of 1e-8 to each value 
for filename in Jones_filenames:
    if 'cleave_rates' in filename:
        with open(filename) as f:
            f.readline()    # Skip the first line, which contains the on-target sequence
            for line in f:
                _, pattern, rate = line.split()
                if pattern not in ocr_values:
                    ocr_values[pattern] = []
                ocr_values[pattern].append(float(rate) + pseudocount)
    elif 'ndABAs_exp' in filename:
        with open(filename) as f:
            f.readline()    # Skip the first line, which contains the on-target sequence
            for line in f:
                _, pattern, rate = line.split()
                if pattern not in aba_values:
                    aba_values[pattern] = []
                # Note that all ndABAs are given in linear scale and needs to 
                # be converted to log-scale (base 10) for comparisons
                aba_values[pattern].append(np.log10(float(rate)) + pseudocount)

# Parse the Boyle et al. 2017 dataset of dead unbinding rates 
Boyle_filename = 'data/Boyle-2017-PNAS-SD3-dissoc-10nM-seqs.txt'
Boyle_unbind_values = {}
with open(Boyle_filename) as f:
    f.readline()    # Skip the first line, which contains the on-target sequence
    for line in f:
        _, pattern, rate = line.split()
        # Only parse the patterns with zero, one, or two mismatches
        if sum(int(c) for c in pattern) >= 18:
            if pattern not in Boyle_unbind_values:
                Boyle_unbind_values[pattern] = []
            Boyle_unbind_values[pattern].append(float(rate) + pseudocount)

# Add overall cleavage rates at 100 nM from other sources (Chen et al. 2017,
# Singh et al. 2016, Singh et al. 2018)
perfect = '1' * 20
mm1 = {}
mm2 = {}
mm_distal = {}
for i in range(0, 20):
    mm1[i] = '1' * i + '0' + '1' * (19 - i)
for i in range(0, 19):
    mm2[i] = '1' * i + '00' + '1' * (18 - i)
for i in range(1, 20):
    mm_distal[i] = '1' * (20 - i) + '0' * i
Chen_cleave_values = {
    perfect:      [0.1667, 0.1595, 0.1421, 0.08431],
    mm_distal[1]: [0.1667, 0.01961, 0.1459, 0.06964],
    mm_distal[2]: [0.1667, 9.522e-4, 5.888e-4, 0.05391],
    mm_distal[3]: [0.08644, 7.278e-5, 4.188e-5, 3.381e-4],
    mm_distal[4]: [6.491e-5, 1.208e-6, 1.289e-6, 7.811e-6],
    mm2[15]:      [1.662e-5, 1e-6, 1.291e-6, 1e-6],
    mm2[13]:      [0.001270, 7.302e-6, 1e-6, 1e-6],
    mm2[11]:      [9.262e-4, 2.008e-5, 1.549e-5, 1e-6],
    mm2[9]:       [5.258e-5, 4.142e-4, 2.478e-4, 1.430e-5],
}
Singh_cleave_values = {
    perfect:      [0.6052, 0.01609, 0.02850],
    mm_distal[1]: [0.7081, 0.01947, 0.004983],
    mm_distal[2]: [0.4516, 0.004079, 5.190e-4],
    mm_distal[3]: [0.1671, 0.001212, 3.346e-4],
    mm_distal[4]: [8.105e-4]
}
for pattern in Chen_cleave_values:
    if pattern not in ocr_values:
        ocr_values[pattern] = []
    ocr_values[pattern] += [x + pseudocount for x in Chen_cleave_values[pattern]]
for pattern in Singh_cleave_values:
    if pattern not in ocr_values:
        ocr_values[pattern] = []
    ocr_values[pattern] += [x + pseudocount for x in Singh_cleave_values[pattern]]

# Add R-loop completion rates from Liu et al. 2020, Singh et al. 2016, Singh et al. 2018
rcomp_values = {
    perfect:      [
        0.9709 + pseudocount, 14.29 + pseudocount, 2.5 + pseudocount,
        0.7 + pseudocount, 2.2 + pseudocount
    ],
    mm_distal[1]: [1.205 + pseudocount, 1.042 + pseudocount, 2.695 + pseudocount],
    mm_distal[2]: [1.754 + pseudocount, 0.2762 + pseudocount, 2.041 + pseudocount],
    mm_distal[3]: [
        1.299 + pseudocount, 0.01960 + pseudocount, 1.832 + pseudocount,
        1.1 + pseudocount, 0.8 + pseudocount, 2.5 + pseudocount
    ],
    mm_distal[4]: [0.02464 + pseudocount, 0.02289 + pseudocount, 0.5807 + pseudocount],
}

# Add R-loop dissolution rates from Liu et al. 2020, Singh et al. 2016, Singh et al. 2018
rdiss_values = {
    perfect:      [
        0.02222 + pseudocount, 0.02869 + pseudocount, 0.6852 + pseudocount,
        1.2 + pseudocount, 0.4 + pseudocount, 0.2 + pseudocount
    ],
    mm_distal[1]: [0.03782 + pseudocount, 1.0 + pseudocount, 3.442 + pseudocount],
    mm_distal[2]: [0.3704 + pseudocount, 10 + pseudocount, 10.58 + pseudocount],
    mm_distal[3]: [
        2.941 + pseudocount, 10 + pseudocount, 9.852 + pseudocount,
        0.5 + pseudocount, 0.4 + pseudocount, 0.7 + pseudocount
    ],
    mm_distal[4]: [9.091 + pseudocount, 10 + pseudocount, 19.05 + pseudocount]
}

# Add dead unbinding rates from Singh et al. 2016, Singh et al. 2018
unbind_values = {
    perfect:       [0.005 + pseudocount, 0.005 + pseudocount, 0.005 + pseudocount],
    mm_distal[4]:  [0.005 + pseudocount, 0.005 + pseudocount, 0.005 + pseudocount],
    mm_distal[11]: [0.005 + pseudocount, 0.005 + pseudocount],
    mm_distal[12]: [0.05348 + pseudocount, 0.2315 + pseudocount, 0.5814 + pseudocount],
    mm_distal[13]: [0.1845 + pseudocount, 0.3704 + pseudocount, 0.7246 + pseudocount],
    mm_distal[14]: [2.439 + pseudocount, 1.351 + pseudocount, 2.564 + pseudocount],
    mm_distal[15]: [1.190 + pseudocount, 2.273 + pseudocount, 3.846 + pseudocount],
    mm_distal[16]: [4.167 + pseudocount, 2.857 + pseudocount, 2.500 + pseudocount],
    mm_distal[20]: [0.8403 + pseudocount, 5.000 + pseudocount, 3.448 + pseudocount],
    '0' * 4 + '1' * 16: [1.852 + pseudocount, 3.226 + pseudocount, 16.67 + pseudocount],
    '0' * 2 + '1' * 18: [0.1645 + pseudocount, 1.538 + pseudocount, 2.857 + pseudocount]
}

# Parse the computed values of each metric at the polytope vertices
values_prefix = sys.argv[1]
values_filenames = [
    '{}-{}.tsv'.format(values_prefix, suffix) for suffix in 
    ['ocr', 'rcomp', 'rdiss', 'aba', 'unbind']
]
for filename in values_filenames:
    data = pd.read_csv(filename, sep='\t', index_col=None, header=0)
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    if filename.endswith('ocr.tsv'):
        for pattern in ocr_values:
            contained = all(
                j > data_min[pattern] and j < data_max[pattern] for j in ocr_values[pattern]
            )
            print(
                'ocr:', pattern, contained, 
                min(ocr_values[pattern]), max(ocr_values[pattern]),
                data_min[pattern], data_max[pattern]
            )
    elif filename.endswith('rcomp.tsv'):
        for pattern in rcomp_values:
            contained = all(
                j > data_min[pattern] and j < data_max[pattern] for j in rcomp_values[pattern]
            )
            print(
                'rcomp:', pattern, contained, 
                min(rcomp_values[pattern]), max(rcomp_values[pattern]),
                data_min[pattern], data_max[pattern]
            )
    elif filename.endswith('rdiss.tsv'):
        for pattern in rdiss_values:
            contained = all(
                j > data_min[pattern] and j < data_max[pattern] for j in rdiss_values[pattern]
            )
            print(
                'rdiss:', pattern, contained, 
                min(rdiss_values[pattern]), max(rdiss_values[pattern]),
                data_min[pattern], data_max[pattern]
            )
    elif filename.endswith('aba.tsv'):
        for pattern in aba_values:
            contained = all(
                j > data_min[pattern] and j < data_max[pattern] for j in aba_values[pattern]
            )
            print(
                'aba:', pattern, contained, 
                min(aba_values[pattern]), max(aba_values[pattern]),
                data_min[pattern], data_max[pattern]
            )
    elif filename.endswith('unbind.tsv'):
        for pattern in unbind_values:
            contained = all(
                j > data_min[pattern] and j < data_max[pattern] for j in unbind_values[pattern]
            )
            print(
                'unbind:', pattern, contained, 
                min(unbind_values[pattern]), max(unbind_values[pattern]),
                data_min[pattern], data_max[pattern]
            )
        for pattern in Boyle_unbind_values:
            contained = all(
                j > data_min[pattern] and j < data_max[pattern] for j in Boyle_unbind_values[pattern]
            )
            print(
                'unbind (Boyle et al. 2017):', pattern, contained,
                min(Boyle_unbind_values[pattern]), max(Boyle_unbind_values[pattern]),
                data_min[pattern], data_max[pattern]
            )

