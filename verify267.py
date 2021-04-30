__doc__ = """
Verify eq. (267) of the original notes. 

To-do:
1. Reallly should implement a decorator for statistics function. 
"""

import numpy as np 
from hybrid_mc import Solution, Trajectory  # Trajectory needs import for Pickle

class Verify267(Solution):
    r"""
    According to notes, S_x is defined as follows:
        S_x = \sum_x S_x = \sum_x (\xi^L_x+\xi^R_x)/2 = 1/2 np.sum(xi); 
    First, auto-correlation time is calculated with sx_act; 
    Second, the ensemble average is returned in stat. 
    """

    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)  # actually does the dirty job 
        self.mat_size = self.N*self.N*self.Nt
    

    def sx_act(self):
        """
        Calculate auto-correlation time with coarsening
        """
        return self.calc_auto_correlation_with_coarsen(lambda _: np.sum(_)/2./ self.mat_size) 


    def stat(self, burnin=None):
        """
        Calculate the ensemble average of S_x
        """
        if not burnin: 
            burnin = len(self.traj.xis) // 2 

        print('Doing statistics...')
        
        sample = self.traj.xis[burnin:] 
        from math import fsum 
        return fsum(np.sum(xi) / 2. / self.mat_size for xi in sample) / len(sample)  # better use a more stable summing algorithm

    def stat_anti(self, burnin=None):
        """
        Calculate the ensemble average of S_x
        """
        if not burnin: 
            burnin = len(self.traj.xis) // 2 

        print('Doing statistics...')
        
        sample = self.traj.xis[burnin:] 
        from math import fsum 
        return fsum((np.sum(xi[:self.mat_size])-np.sum(xi[self.mat_size:])) / 2. / self.mat_size for xi in sample) / len(sample)  # better use a more stable summing algorithm




def f265(Nt, N, hat_t, hat_U):
    '''
    Function F(U) defined in eq. (265);

    Numerically, I would say the following code is completely wrong when Nt is large;
    so there's a mma counterpart of it.
    ''' 
    from energy_band import energy 

    ret = 0.
    for k1 in range(N):  # sum over k space 
        for k2 in range(N):
            ret += (1 - hat_U +hat_t*energy(k1/N, k2/N))**(Nt-1) / (1+(1 - hat_U +hat_t*energy(k1/N, k2/N))**Nt)  # one chiral index 
            ret += (1 - hat_U -hat_t*energy(k1/N, k2/N))**(Nt-1) / (1+(1 - hat_U -hat_t*energy(k1/N, k2/N))**Nt)  # the other index, with opposite energy 
    ret /= N * N 
    
    return ret 


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    hatu=.9
    hatt=.2
    ts = .3

    sol = Verify267(10,4,hatt,hatu, time_step=ts, max_epochs=2000, 
        from_file=False, 
        # filename='N5Nt50hatt2.00e-03hatU1.00e-07ts2.20e-01act40ep50000.pickle'
    )

    print("Simulation yields: ", sol.stat())
    print("Theory predicts:", -f265(10,3,hatt,hatu, )*np.sqrt(hatu))

    plt.plot(sol.sx_act())
    plt.savefig('1.png')


    sol = Verify267(10,4,hatt,hatu, time_step=ts, max_epochs=50000, 
        from_file=False, 
        # filename='N5Nt50hatt2.00e-03hatU1.00e-07ts2.20e-01act40ep50000.pickle'
    )

    print("Simulation yields: ", sol.stat())
    print("Theory predicts:", -f265(10,3,hatt,hatu, )*np.sqrt(hatu))

    plt.plot(sol.sx_act())
    plt.savefig('2.png')
    
