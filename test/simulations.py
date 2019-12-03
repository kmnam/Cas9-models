"""
Implementations of Gillespie simulations on the line and grid graphs.

Authors:
    Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
Last updated:
    12/2/2019
"""
import numpy as np

##############################################################
class LineGraph(object):
    """
    A bare-bones implementation of the line graph.
    """
    def __init__(self, length):
        """
        Trivial constructor. 

        Parameters
        ----------
        length : int
            Length of the line graph.
        """
        self.length = length
        self.vertices = list(range(0, self.length + 1))

    ##########################################################
    def simulate(self, nsim, pattern, init=0, b=1.0, bp=1.0, d=1.0, dp=1.0,
                 kdis=1.0, kcat=1.0):
        """
        Run the given number of simulations, and return an array indicating
        the absorbing vertex and time to absorption for each trajectory.

        Parameters
        ----------
        nsim : int
            Number of simulations.
        pattern : str
            String of zeros and ones indicating the pattern of matches (1)
            and mismatches (0).
        init : int
            Starting vertex. 
        b : float
            Forward rate at a matching position.
        bp : float
            Forward rate at a mismatching position.
        d : float
            Backward rate at a matching position.
        dp : float
            Backward rate at a mismatching position.
        kdis : float
            Rate of absorption into -1.
        kcat : float
            Rate of absorption into self.length + 1.
        """
        stats = np.zeros((nsim, 2))

        for i in range(nsim):
            curr = init    # Start from the given vertex
            time = 0.0
            prob = None
            rates = None

            # While absorption has not yet occurred ...
            while -1 < curr < self.length + 1:
                # If absorption is possible at vertex 0 ...
                if curr == 0:
                    prob = kdis / (kdis + b) if pattern.startswith('1') else kdis / (kdis + bp)
                    rates = [kdis, b] if pattern.startswith('1') else [kdis, bp]
                    toss = np.random.binomial(1, prob)
                    curr = -1 if toss == 1 else 1

                # If absorption is possible at vertex self.length ...
                elif curr == self.length:
                    prob = kcat / (kcat + d) if pattern.endswith('1') else kcat / (kcat + dp)
                    rates = [kcat, d] if pattern.endswith('1') else [kcat, dp]
                    toss = np.random.binomial(1, prob)
                    curr = self.length + 1 if toss == 1 else self.length - 1

                # If absorption is not yet possible ...
                else:
                    match_next = pattern[curr]
                    match_prev = pattern[curr-1]
                    if match_next and match_prev:
                        prob = b / (b + d)
                        rates = [b, d]
                    elif match_next:
                        prob = b / (b + dp)
                        rates = [b, dp]
                    elif match_prev:
                        prob = bp / (bp + d)
                        rates = [bp, d]
                    else:
                        prob = bp / (bp + dp)
                        rates = [bp, dp]
                    toss = np.random.binomial(1, prob)
                    curr = curr + 1 if toss == 1 else curr - 1

                # Sample an exponentially distributed waiting time
                time += np.random.exponential(1.0 / sum(rates))
            
            stats[i,0] = curr
            stats[i,1] = time

        return stats

##############################################################
class GridGraph(object):
    """
    A bare-bones implementation of the grid graph.
    """
    def __init__(self, length):
        """
        Trivial constructor.

        Parameters
        ----------
        length : int
            Length of the grid graph.
        """
        self.length = length
        self.vertices = [('A', i) for i in range(0, self.length + 1)]\
            + [('B', i) for i in range(0, self.length + 1)]

    ##########################################################
    def laplacian(self, pattern, b=1.0, bp=1.0, d=1.0, dp=1.0, k=1.0, l=1.0):
        """
        Return the row Laplacian matrix of the grid graph. 

        Parameters
        ----------
        pattern : str
            String of zeros and ones indicating the pattern of matches (1)
            and mismatches (0).
        b : float
            Forward rate at a matching position.
        bp : float
            Forward rate at a mismatching position.
        d : float
            Backward rate at a matching position.
        dp : float
            Backward rate at a mismatching position.
        k : float
            Rate of conversion from (A,i) to (B,i).
        l : float
            Rate of conversion from (B,i) to (A,i).
        """
        f = lambda i: return b if pattern[i] else bp
        g = lambda i: return d if pattern[i] else dp

        laplacian = np.zeros((2 * self.length + 2, 2 * self.length + 2))
        for i in range(self.length):
            laplacian[i, i+1] = -g(i)
            laplacian[i+1, i] = -f(i)
            laplacian[self.length+1+i, self.length+1+i+1] = -g(i)
            laplacian[self.length+1+i+1, self.length+1+i] = -f(i)
        for i in range(self.length + 1):
            laplacian[i, self.length+1+i] = -k
            laplacian[self.length+1+i, i] = -l

        return laplacian

    ##########################################################
    def simulate(self, nsim, pattern, init=('A',0), b=1.0, bp=1.0, d=1.0, dp=1.0,
                 k=1.0, l=1.0, kdis=1.0, kcat=1.0):
        """
        Run the given number of simulations, and return an array indicating
        the absorbing vertex and time to absorption for each trajectory.

        Parameters
        ----------
        nsim : int
            Number of simulations.
        pattern : str
            String of zeros and ones indicating the pattern of matches (1)
            and mismatches (0).
        init : int
            Starting vertex. 
        b : float
            Forward rate at a matching position.
        bp : float
            Forward rate at a mismatching position.
        d : float
            Backward rate at a matching position.
        dp : float
            Backward rate at a mismatching position.
        k : float
            Rate of conversion from (A,i) to (B,i).
        l : float
            Rate of conversion from (B,i) to (A,i).
        kdis : float
            Rate of absorption into -1.
        kcat : float
            Rate of absorption into self.length + 1.
        """
        stats = np.zeros((nsim, 2))

        for i in range(nsim):
            curr = init    # Start from the given vertex
            time = 0.0
            prob = None
            rates = None

            # While absorption has not yet occurred ...
            while curr != ('A',-1) and curr != ('B',self.length+1):
                # If absorption is possible at vertex (A,0) ...
                if curr == ('A',0):
                    rates = [kdis, b, k] if pattern.startswith('1') else [kdis, bp, k]
                    dests = [('A',-1), ('A',1), ('B',0)]

                # If absorption is possible at vertex (B,self.length) ...
                elif curr == ('B',self.length):
                    rates = [kcat, d, l] if pattern.endswith('1') else [kcat, dp, l]
                    dests = [('B',self.length+1), ('B',self.length-1), ('A',self.length)]

                # If we are at vertex (B,0) ...
                elif curr == ('B',0):
                    rates = [b, l] if pattern.startswith('1') else [bp, l]
                    dests = [('B',1), ('A',0)]

                # If we are at vertex (A,self.length) ...
                elif curr == ('A',self.length):
                    rates = [d, k] if pattern.endswith('1') else [dp, k]
                    dests = [('A',self.length-1), ('B',self.length)]

                # If we are anywhere else ...
                else:
                    match_next = pattern[curr[1]]
                    match_prev = pattern[curr[1]-1]
                    if match_next and match_prev:
                        rates = [b, d, k] if curr[0] == 'A' else [b, d, l]
                    elif match_next:
                        rates = [b, dp, k] if curr[0] == 'A' else [b, dp, l]
                    elif match_prev:
                        rates = [bp, d, k] if curr[0] == 'A' else [bp, d, l]
                    else:
                        rates = [bp, dp, k] if curr[0] == 'A' else [bp, dp, l]
                    if curr[0] == 'A':
                        dests = [('A',curr[1]+1), ('A',curr[1]-1), ('B',curr[1])]
                    else:
                        dests = [('B',curr[1]+1), ('B',curr[1]-1), ('A',curr[1])]

                # Sample an exponentially distributed waiting time
                norm = sum(rates)
                time += np.random.exponential(1.0 / norm)

                # Sample a new vertex
                probs = [x / norm for x in rates]
                curr = dests[np.random.choice(range(len(dests)), p=probs)]
            
            stats[i,0] = 0 if curr == ('A',-1) else 1
            stats[i,1] = time

        return stats

