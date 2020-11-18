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
    

    def sx_act(self):
        """
        Calculate auto-correlation time with coarsening
        """
        return self.calc_auto_correlation(lambda _: np.sum(_)/2.) 


    def stat(self, burnin=None):
        """
        Calculate the ensemble average of S_x
        """
        if not burnin: 
            burnin = len(self.traj.xis) // 2 

        print('Doing statistics...')
        
        sample = self.traj.xis[burnin::self.act] 

        ret = 0. 
        for xi in sample:
            ret += np.sum(xi) / 2. 
        return ret 

import matplotlib.pyplot as plt
sol = Verify267(50,5,2e-3,1e-7, time_step=0.22, max_epochs=50000, act=40, from_file=True)

sol.calc_auto_correlation_with_coarsen()
for act in range(10, 200, 10):
  print(act, sol.stat(act=act))
# plt.plot(sol.sx_act())
# plt.savefig('1.png')

"""
sol = Verify267(50,5,2e-3,5e-7, time_step=0.25, max_epochs=50000, act=40, from_file=False)
print(sol.stat())
plt.plot(sol.sx_act())
plt.savefig('2.png')

sol = Verify267(50,5,2e-3,1e-6, time_step=0.28, max_epochs=50000, act=40, from_file=False)
print(sol.stat())
plt.plot(sol.sx_act())
plt.savefig('3.png')

sol = Verify267(50,5,2e-3,5e-6, time_step=0.29, max_epochs=50000, act=40, from_file=False)
print(sol.stat())
plt.plot(sol.sx_act())
plt.savefig('4.png')
"""

