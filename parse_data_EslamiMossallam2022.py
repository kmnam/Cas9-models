"""
Authors:
    Kee-Myoung Nam

Last updated:
    2/1/2023
"""
import numpy as np
import pandas as pd

#########################################################################
# All forward rates are set to one value 
forward = 641.289

# Reverse rates are position-specific and depend on whether a match or 
# mismatch is present 
reverse = np.array(
    [
        [361204., 103031000],
        [2510.87, 153092.],
        [7.52936, 4920.77],
        [2982.63, 3195580],
        [273.362, 143958.],
        [678.211, 1096930],
        [7707.97, 7644520],
        [153.089, 76995.3],
        [14.9964, 120814.],
        [201.732, 285090.],
        [230.198, 377944.],
        [345.102, 387712.],
        [14729.1, 33839500],
        [124.006, 329312.],
        [218.163, 455910.],
        [5526.94, 3181170],
        [226.096, 38348.9],
        [31.2425, 2185.78],
        [955.820, 325465.],
        [0.721473, 8.10370]
    ],
    dtype=np.float64
)
length = 20
fmt = '{:.6e}'

# Prepare output file with a header line 
outfilename = 'data/EslamiMossallam2022-params.tsv'
with open(outfilename, 'w') as f:
    f.write('seqid\t')
    f.write('\t'.join(['forward_{}'.format(i) for i in range(length)]) + '\t')
    f.write('\t'.join(['reverse_{}'.format(i+1) for i in range(length)]))
    f.write('\n')

    # Write an output string for the perfect match sequence
    seq = np.ones((length,), dtype=int)
    forward_arr = forward * np.ones((length,), dtype=np.float64)
    outstr = ''.join([str(x) for x in seq]) + '\t'
    outstr += '\t'.join([fmt.format(x) for x in forward_arr]) + '\t'
    outstr += '\t'.join([fmt.format(x) for x in reverse[:, 0]]) + '\n'
    f.write(outstr)

    # For each possible single-mismatch sequence ... 
    for j in range(length):
        # Write an output string for the mismatched sequence
        seq = np.ones((length,), dtype=int)
        seq[j] = 0
        outstr = ''.join([str(x) for x in seq]) + '\t'
        outstr += '\t'.join([fmt.format(x) for x in forward_arr]) + '\t'
        if j == 0:
            outstr += fmt.format(reverse[j, 1]) + '\t'
            outstr += '\t'.join([fmt.format(reverse[k, 0]) for k in range(j+1, length)]) + '\n'
        elif j > 0 and j < length - 1:
            outstr += '\t'.join([fmt.format(reverse[k, 0]) for k in range(0, j)]) + '\t'
            outstr += fmt.format(reverse[j, 1]) + '\t'
            outstr += '\t'.join([fmt.format(reverse[k, 0]) for k in range(j+1, length)]) + '\n'
        if j == length - 1:
            outstr += '\t'.join([fmt.format(reverse[k, 0]) for k in range(0, j)]) + '\t'
            outstr += fmt.format(reverse[j, 1]) + '\n'
        f.write(outstr)

