"""
Translate a convex polytope specified in terms of its V-representation (in
a .vert file) by a given vector.

Authors:
    Kee-Myoung Nam

Last updated:
    1/15/2023
"""
import sys
import numpy as np 

infilename = sys.argv[1]
outfilename = sys.argv[2]
vertices = np.loadtxt(infilename, delimiter=' ', dtype=int)
v = np.array(sys.argv[3:], dtype=int)
if v.size != vertices.shape[1]:
    raise ValueError('Invalid translation vector (incorrect dimensionality)')
vertices += v
np.savetxt(outfilename, vertices, fmt='%d', delimiter=' ')

