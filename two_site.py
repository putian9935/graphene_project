__doc__ = """
Calculate two-site (staggered) magnetic moment, 
"""

import numpy as np 
from hybrid_mc import Solution, Trajectory  # Trajectory needs import for Pickle
from scipy.sparse.linalg import inv
from m_matrix import m_matrix_xi
from tqdm import tqdm
import matplotlib.pyplot as plt

class TwoSite(Solution):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs) 

    def charge(self, m_inv):
        return 1.-np.mean(m_inv.toarray().diagonal())

    def magnetic_moment(self, m_inv):
        buf = m_inv.toarray().diagonal()
        return -np.sum(buf[::4]+buf[2::4]-buf[1::4]-buf[3::4])/self.Nt

    def staggered_magnetic_moment(self, m_inv):
        buf = m_inv.toarray().diagonal()
        return -np.sum(buf[::4]-buf[2::4]-buf[1::4]+buf[3::4])/self.Nt

    def calc_auto_correlation_with_coarsen(self, x):
        ''' Coarsening to determine ac time.  ''' 
        from block import linear_blocking
        
        return linear_blocking(x)  

    def stat(self, stat_funcs, burnin=None):
        if not burnin: 
            burnin = len(self.traj.xis) // 2 

        res = []
        print('Doing statistics...')
        for xi in tqdm(self.traj.xis[burnin:]):
            m_inv = inv(self.traj.m_mat_indep + m_matrix_xi(self.Nt, self.N, self.hat_U, xi))

            new_entry = []
            for f in stat_funcs: 
                new_entry.append(f(m_inv))
            res.append(np.array(new_entry))
        
        res = np.array(res)

        for i in range(len(stat_funcs)):
            plt.plot(self.calc_auto_correlation_with_coarsen(res[:,i]) )
            plt.show()
        return np.mean(res, axis=0), np.std(res, axis=0) 


Nt = 20

hat_u, hat_t = .6,.1

ts=.15
sol = TwoSite(Nt,1,hat_t, hat_u, time_step=ts, max_epochs=6000, 
        from_file=False,) 

print(*sol.stat(stat_funcs=[sol.charge, sol.magnetic_moment, sol.staggered_magnetic_moment]))
