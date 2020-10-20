__doc__  = """
Add new feature that records how hamiltonian scales with step
"""

from hybrid_mc import Trajectory 
import numpy as np 
from m_matrix import tau_n2ind, m_matrix_same4all, m_matrix_xi
from scipy.sparse.linalg import spsolve, inv, splu
from scipy import sparse
from tqdm import tqdm
from deprecated import m_matrix_xi_old 

m_matrix_xi = m_matrix_xi_old

class HamiltonianScale(Trajectory):
    rej_updates = 0
    tot_updates = 0
    def __init__(self,Nt, N, hat_t, hat_U, time_step=5e-4, max_steps=5, max_epochs=2000):
        super().__init__(Nt, N, hat_t, hat_U)
        
        def force(): 
            ''' calc the force according to eq. (221), with two chiral indices combined '''
            self.m_mat = self.m_mat_indep + m_matrix_xi(self.Nt, self.N, self.hat_U, self.xi)
            self.f_mat = self.m_mat @ self.m_mat.T
            self.xs = []

            ret = -self.xi
            solver = splu(self.f_mat)
            for phi in self.phi:
                self.xs.append(solver.solve(phi))
                y = self.m_mat.T @ self.xs[-1]  # can't figure out why need a transpose here, but it worked! 
                ret += 2 * self.hat_U **.5 * (self.xs[-1] * y) # trace is too fancy, simply an element-wise product 

            return ret


        def hamiltonian():
            ''' calc the hamiltonian according to eq. (219), without the sum of alpha '''
            nonlocal pi
            # f_mat is always updated before calcing hamiltonian
            return  .5*(pi@pi+self.xi@self.xi)+sum(phi@x for x, phi in zip(self.xs, self.phi))



        def leapfrog():
            ''' the leapfrog algorithm ''' 
            nonlocal pi
            pi += force() * (time_step / 2)
            for _ in range(max_steps - 1):
                self.xi += pi * time_step 
                pi += force() * time_step
            self.xi += pi * time_step 
            pi += force() * (time_step / 2)

        def one_step_leapfrog():
            ''' the leapfrog algorithm for only one step''' 
            nonlocal pi
            pi += force() * (time_step / 2)
            self.xi += pi * time_step 
            pi += force() * (time_step / 2)


        self._generate_phi()
        tmp_ham = []
        self.tot_delta_h = []
        for epoch in tqdm(range(max_epochs)):
            prev_xi = self.xi.copy()  # make a copy of previous state in case hamiltonian gets bad 

            pi = np.random.randn(self.half_size * 2)  # random momenta
            
            h_start = hamiltonian()  # record the hamiltonian at the beginning

            # launch leapfrog algorithm
            one_step_leapfrog()

            h_end = hamiltonian()  # record the hamiltonian at the end
            
            # two components of xi, phi are independent, 
            # so it's cool to accept one update while rejecting another


            self.tot_delta_h.append(h_end-h_start)
            if h_end < h_start: # exp might overflow
                continue
            if np.random.random() > np.exp(-h_end + h_start):
                self.xi = prev_xi  
                  


if __name__ == '__main__':
    from histogram import show_histogram
    import matplotlib.pyplot as plt 

    time_steps = [5e-2, 5e-3, 5e-4]
    for ts in time_steps:
        np.random.seed(42)
        hs = HamiltonianScale(8,2,2,1,time_step=ts)
        show_histogram(hs.tot_delta_h[hs.max_epochs//2:], '%.1e'%ts, np.std(hs.tot_delta_h[hs.max_epochs//2:]))
        print(np.std(hs.tot_delta_h[hs.max_epochs//2:]))
