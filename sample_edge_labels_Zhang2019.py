"""
Generate edge labels for the line graph that are consistent with thermodynamic
parameter values determined by Zhang et al. (2019).

Authors:
    Kee-Myoung Nam

Last updated:
    9/14/2022
"""
import sys
import numpy as np
import pandas as pd

#########################################################################
def get_complement(rna_char):
    """
    Return the DNA nucleobase that is complementary to the given RNA nucleobase.
    """
    if rna_char == 'A':
        return 'T'
    elif rna_char == 'C':
        return 'G'
    elif rna_char == 'G':
        return 'C'
    elif rna_char == 'U':
        return 'A'
    else:
        raise ValueError('Invalid RNA nucleobase specified')

#########################################################################
def get_noncomplementary_triple(rna_char):
    """
    Return the three DNA nucleobases that are non-complementary to the given
    RNA nucleobase. 
    """
    if rna_char == 'A':
        return ['A', 'C', 'G']
    elif rna_char == 'C':
        return ['A', 'C', 'T']
    elif rna_char == 'G':
        return ['A', 'G', 'T']
    elif rna_char == 'U':
        return ['C', 'G', 'T']
    else:
        raise ValueError('Invalid RNA nucleobase specified')

#########################################################################
def get_label_pairs_for_rna_seq(rseq, m, stack_data, mismatch_data, 
                                start_stack_data, omega_dna, omega_rna,
                                psi_dna, psi_rna, rng, forward_logmin, 
                                forward_logmax, fmt):
    """
    Generate an output string containing `m` randomly sampled forward/reverse
    label pairs for the perfect-match substrate and all single-mismatch
    substrates against a given RNA sequence. 
    """
    length = len(rseq)

    # Define perfect-match DNA sequence 
    dseq = [get_complement(c) for c in rseq]

    # ------------------------------------------------------------------ # 
    # Get label ratios corresponding to the perfect-match DNA substrate 
    # ------------------------------------------------------------------ # 
    # Begin with the starting base-pair stacking interaction
    match_label_ratios = np.zeros((length,), dtype=np.float64)
    match_label_ratios[0] = np.exp(-start_stack_data[dseq[0]])

    # Then, for each subsequent dinucleotide pair, account for the corresponding
    # base-pair stacking interaction 
    for j in range(length - 1):
        total = stack_data.iloc[j][dseq[j] + dseq[j+1]]
        match_label_ratios[j+1] = np.exp(-total)

    # Generate m pairs of match/mismatch forward rates
    forward_rates = np.zeros((m, 2), dtype=np.float64)
    for j in range(m):
        a = 10 ** (forward_logmin + (forward_logmax - forward_logmin) * rng.random())
        b = 10 ** (forward_logmin + (forward_logmax - forward_logmin) * rng.random())
        if a < b:
            forward_rates[j, 0] = b
            forward_rates[j, 1] = a
        else:
            forward_rates[j, 0] = a
            forward_rates[j, 1] = b

    # Generate m pairs of cleavage/unbinding rates from the same range 
    terminal_rates = np.zeros((m, 2), dtype=np.float64)
    for j in range(m):
        terminal_rates[j, 0] = 1.0    # Fix unbinding rate to unity
        terminal_rates[j, 1] = 10 ** (forward_logmin + (forward_logmax - forward_logmin) * rng.random())

    # Generate m sets of forward/reverse label pairs that give rise to the
    # label ratios
    match_label_pairs = np.zeros((m, length, 2), dtype=np.float64)
    for j in range(m):
        match_label_pairs[j, :, 0] = forward_rates[j, 0] * np.ones(length)
        match_label_pairs[j, :, 1] = match_label_pairs[j, :, 0] / match_label_ratios

    # Write an output string for these forward/reverse label pairs 
    outstr = ''
    for j in range(m):
        outstr += '{}:{}:{}:match\t'.format(''.join(rseq), ''.join(dseq), j)
        outstr += '\t'.join([fmt.format(x) for x in match_label_pairs[j, :, 0]]) + '\t'
        outstr += '\t'.join([fmt.format(x) for x in match_label_pairs[j, :, 1]]) + '\t'
        outstr += fmt.format(terminal_rates[j, 0]) + '\t'
        outstr += fmt.format(terminal_rates[j, 1]) + '\n'

    # ------------------------------------------------------------------ # 
    # Get label ratios corresponding to each single-mismatch DNA substrate 
    # ------------------------------------------------------------------ # 
    # For each possible single-mismatch position ...
    for j in range(length):
        # ... and for each possible mismatched DNA sequence ...
        mismatched_dna_chars = get_noncomplementary_triple(rseq[j])
        for dna_char in mismatched_dna_chars:
            # Get the corresponding mismatched DNA sequence 
            dseq_mismatched = [c for c in dseq]
            dseq_mismatched[j] = dna_char

            # If j is the mismatch position, then the label ratios for 
            # j <-> j+1 and j+1 <-> j+2 are affected
            # 
            # (Note that "j" is the last state without j in the R-loop; 
            # "j+1" is the first state with j in the R-loop; and 
            # "j+2" is the second state with j in the R-loop)
            label_ratios = match_label_ratios.copy()
            if j == 0:
                label_ratios[0] = np.exp(
                    -start_stack_data[dseq_mismatched[0]]
                    - mismatch_data.iloc[0][dseq_mismatched[0] + rseq[0]]
                )
                dseq_j, dseq_next = dseq_mismatched[0], dseq_mismatched[1]
                rseq_j, rseq_next = rseq[0], rseq[1]
                total = (
                    omega_dna * stack_data.iloc[0][dseq_j + dseq_next]
                    + omega_rna * stack_data.iloc[0][
                        ('T' if rseq_j == 'U' else rseq_j)
                        + ('T' if rseq_next == 'U' else rseq_next)
                    ]
                    + mismatch_data.iloc[0][dseq_j + rseq_j]
                )
                label_ratios[1] = np.exp(-total)
            elif j == length - 1:
                dseq_prev, dseq_j = dseq_mismatched[length-2], dseq_mismatched[length-1]
                rseq_prev, rseq_j = rseq[length-2], rseq[length-1]
                total = (
                    omega_dna * stack_data.iloc[length-2][dseq_prev + dseq_j]
                    + omega_rna * stack_data.iloc[length-2][
                        ('T' if rseq_prev == 'U' else rseq_prev)
                        + ('T' if rseq_j == 'U' else rseq_j)
                    ]
                    + mismatch_data.iloc[length-1][dseq_j + rseq_j]
                )
                label_ratios[length-1] = np.exp(-total)
            else:
                dseq_prev, dseq_j, dseq_next =\
                    dseq_mismatched[j-1], dseq_mismatched[j], dseq_mismatched[j+1]
                rseq_prev, rseq_j, rseq_next = rseq[j-1], rseq[j], rseq[j+1]
                total = (
                    omega_dna * stack_data.iloc[j-1][dseq_prev + dseq_j]
                    + omega_rna * stack_data.iloc[j-1][
                        ('T' if rseq_prev == 'U' else rseq_prev)
                        + ('T' if rseq_j == 'U' else rseq_j)
                    ]
                    + mismatch_data.iloc[j][dseq_j + rseq_j]
                )
                label_ratios[j] = np.exp(-total)
                total = (
                    omega_dna * stack_data.iloc[j][dseq_j + dseq_next]
                    + omega_rna * stack_data.iloc[j][
                        ('T' if rseq_j == 'U' else rseq_j)
                        + ('T' if rseq_next == 'U' else rseq_next)
                    ]
                    + mismatch_data.iloc[j][dseq_j + rseq_j]
                )
                label_ratios[j+1] = np.exp(-total)

            # Now generate p new label pairs, one for each of the m sets of
            # label pairs for the matched DNA sequence, keeping all label pairs
            # in each set the same except for the affected label ratios
            label_pairs = match_label_pairs.copy()
            label_pairs[:, j, 0] = forward_rates[:, 1]
            label_pairs[:, j, 1] = label_pairs[:, j, 0] / label_ratios[j]
            if j < length - 1:
                label_pairs[:, j+1, 1] = label_pairs[:, j+1, 0] / label_ratios[j+1]

            # Write the mismatched label pairs to the output string
            for l in range(m):
                outstr += '{}:{}:{}:mm{}\t'.format(''.join(rseq), ''.join(dseq_mismatched), l, j)
                outstr += '\t'.join([fmt.format(x) for x in label_pairs[l, :, 0]]) + '\t'
                outstr += '\t'.join([fmt.format(x) for x in label_pairs[l, :, 1]]) + '\t'
                outstr += fmt.format(terminal_rates[l, 0]) + '\t'
                outstr += fmt.format(terminal_rates[l, 1]) + '\n'

    return outstr

#########################################################################
def main():
    stack_data = pd.read_csv(
        'data/Zhang2019-stack-free.tsv', sep='\t', index_col=None
    )
    mismatch_data = pd.read_csv(
        'data/Zhang2019-mismatch-free.tsv', sep='\t', index_col=None
    )
    start_stack_data = {
        'A': -0.09539, 'C': -0.08904, 'G': -0.09457, 'T': -0.09288
    }
    omega_dna = 2.69110
    omega_rna = 6.12679
    psi_dna = 10.4619
    psi_rna = 7.23209
    rng = np.random.default_rng(1234567890)
    rna_chars = ['A', 'C', 'G', 'U']
    dna_chars = ['T', 'G', 'C', 'A']
    length = 20
    forward_logmin = -5    # Log (base 10) of minimum value of each forward label
    forward_logmax = 5     # Log (base 10) of maximum value of each forward label
    fmt = '{:.10e}'

    # Prepare output file with a header line 
    outfilename = 'data/Zhang2019-sampled-seqs-edge-labels.tsv'
    with open(outfilename, 'w') as f:
        f.write('seqid\t')
        f.write('\t'.join(['forward_{}'.format(i) for i in range(length)]) + '\t')
        f.write('\t'.join(['reverse_{}'.format(i+1) for i in range(length)]))
        f.write('\tterminal_cleave_rate\tterminal_unbind_rate\n')

    m = int(sys.argv[2])     # Number of forward/reverse label pairs to sample for each sequence

    if sys.argv[1].isdigit():    # If a number has been specified ... 
        n = int(sys.argv[1])
        for i in range(n):
            # Sample an RNA sequence of the given length 
            rints = rng.integers(low=0, high=4, size=length)
            rseq = [rna_chars[j] for j in rints]
            # Get the output string containing m randomly sampled forward/reverse 
            # label pairs for each RNA sequence
            outstr = get_label_pairs_for_rna_seq(
                rseq, m, stack_data, mismatch_data, start_stack_data, omega_dna,
                omega_rna, psi_dna, psi_rna, rng, forward_logmin, forward_logmax,
                fmt
            ) 
            # Write the output string to file
            with open(outfilename, 'a') as f:
                f.write(outstr)
    else:                        # Otherwise ... 
        n = 1
        rseq = [c for c in sys.argv[1]]
        if len(rseq) != length:
            raise ValueError('Specified sequence has incorrect length')
        elif any(c not in rna_chars for c in rseq):
            raise ValueError('Specified sequence is not valid RNA sequence')
        # Get the output string containing m randomly sampled forward/reverse 
        # label pairs for each RNA sequence
        outstr = get_label_pairs_for_rna_seq(
            rseq, m, stack_data, mismatch_data, start_stack_data, omega_dna,
            omega_rna, psi_dna, psi_rna, rng, forward_logmin, forward_logmax,
            fmt
        ) 
        # Write the output string to file
        with open(outfilename, 'a') as f:
            f.write(outstr)

#########################################################################
if __name__ == '__main__':
    main()
