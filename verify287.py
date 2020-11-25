__doc__ = """
Verify eq. (287) of the original notes. 
"""


import numpy as np 
from hybrid_mc import Solution, Trajectory  

class Verify287(Solution):
    ''' Simulation of the l.h.s. of eq. (287) '''
    pass 


def SigmaSS(Nt,N, hat_u):
    ''' \Sigma^{SS} defined in eq. (281) '''
    pass 

def SigmaPP(Nt,N, hat_u):
    ''' \Sigma^{PP} defined in eq. (281) '''
    pass 

def SigmaSP(Nt,N, hat_u):
    ''' \Sigma^{SP} defined in eq. (282) '''
    pass 

def SigmaPS(Nt,N, hat_u):
    ''' \Sigma^{PP} defined in eq. (282) '''
    pass 

def rhs287(Nt, N, hat_u, q):
    ''' Calculation of the r.h.s. of eq. (287) '''
    pass 

if __name__ == '__main__':
    pass
