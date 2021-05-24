__doc__ = """
Calculate 4x4x16 staggered magnetic moment, double occupancy, etc...
"""

import numpy as np 
from hybrid_mc import Solution
from scipy.sparse.linalg import inv, bicgstab

from m_matrix import m_matrix_xi
from tqdm import tqdm
import matplotlib.pyplot as plt
from numpy.linalg import inv as minv

class MySol(Solution):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
    def staggered_magnetic_moment(self, m_inv):
        offset_diagonal = np.diagonal(m_inv, offset=-2*self.N*self.N)
        return -(np.mean(offset_diagonal[::2])- np.mean(offset_diagonal[::1]))*2
    
    def magnetic_moment2(self, m_inv):
        return 1*(2-np.mean(np.diagonal(m_inv, offset=-2*self.N*self.N))*2)-1


    def calc_auto_correlation_with_coarsen(self, x):
        ''' Coarsening to determine ac time.  ''' 
        from block import linear_blocking
        
        return linear_blocking(x)  

    def stat(self, stat_funcs, burnin=None):
        if not burnin: 
            burnin = min(len(self.traj.xis) // 2, 1000)

        res = []
        print('Doing statistics...')
        
        # m_inv = inv(self.traj.m_mat_indep)
        # return self.magnetic_moment(m_inv), self.magnetic_moment2(m_inv)

        for xi in tqdm(self.traj.xis[burnin:]):
            d_mat = m_matrix_xi(self.Nt, self.N, self.hat_U, xi)
            # m_inv = inv(self.traj.m_mat_indep + d_mat - .5*(self.traj.t_mat_indep@d_mat+d_mat@self.traj.t_mat_indep)-.5*d_mat@d_mat).toarray()

            m_inv = minv((self.traj.m_mat_indep + d_mat - .5*(self.traj.t_mat_indep@d_mat+d_mat@self.traj.t_mat_indep)-.5*d_mat@d_mat).toarray())
            # plt.matshow(m_inv) 
            # plt.show()
            new_entry = []
            for f in stat_funcs: 
                new_entry.append(f(m_inv))
            res.append(np.array(new_entry))
        
        res = np.array(res)

        for label, i in zip(['f1, f2'], range(len(stat_funcs))):
            plt.plot(self.calc_auto_correlation_with_coarsen(res[:,i]) , label=label)
        return np.mean(res, axis=0), np.std(res, axis=0) 


N, Nt = 4, 16 

t = 2.8 
U = 4 

res = []
for beta in [.002]:
    sol = MySol(Nt, N, beta*t/Nt, beta*U/Nt,  e=beta*.01/Nt, time_step=.14, max_epochs=20,  from_file=False,staggered=False)

    res.append(np.array(sol.stat(stat_funcs=[sol.magnetic_moment2, sol.staggered_magnetic_moment]))) 
    print(res[-1])
    exit()
    plt.xlabel('$N_B$')
    plt.ylabel('$\sigma^2$')
    plt.xlim(left=0)
    plt.grid()
    
    # plt.savefig('beta%.3e.pdf'%beta)

res = np.array(res)
print(res)

