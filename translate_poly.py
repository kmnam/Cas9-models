"""
Translate a convex polytope specified in terms of its H-representation (in
a .poly file) by a given vector.

Authors:
    Kee-Myoung Nam

Last updated:
    1/15/2023
"""
import sys
from fractions import Fraction
import numpy as np 

infilename = sys.argv[1]
outfilename = sys.argv[2]
ineq_mode = None
system = []
with open(infilename) as f:
    ineq_mode = f.readline().strip()
    for line in f:
        system.append([float(x) for x in line.split()])
system = np.array(system)

v = np.array(sys.argv[3:], dtype=int)
if v.size != system.shape[1] - 1:
    raise ValueError('Invalid translation vector (incorrect dimensionality)')

# We want to replace each variable x with x' = x + a for some a, meaning 
# that x = x' - a
#
# This means that, for each polynomial constraint b + c1*x1 + ... + cN*xN >= 0,
# we can rewrite this constraint as
#
# b + c1*(x1' - a1) + ... + cN*(xN' - a1) >= 0
#
# or 
#
# b - c1*a1 - ... - cN*aN + c1*x1' + ... + cN*xN' >= 0
for i in range(system.shape[0]):
    system[i, 0] -= np.dot(system[i, 1:], v)

with open(outfilename, 'w') as f:
    f.write(ineq_mode + '\n')
    for i in range(system.shape[0]):
        f.write(' '.join(['{}'.format(Fraction(x)) for x in system[i, :]]) + '\n')

