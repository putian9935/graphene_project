__doc__  = """
Add new feature that records how hamiltonian scales with step
"""

from hybrid_mc import Trajectory 
import numpy as np 
from m_matrix import tau_n2ind, m_matrix_same4all, m_matrix_xi
from scipy.sparse.linalg import spsolve, inv
from scipy import sparse
from tqdm import tqdm

class HamiltonianScale(Trajectory):
    rej_updates = 0
    tot_updates = 0
    def __init__(self,Nt, N, hat_t, hat_U, time_step=0.005, max_steps=10, max_epochs=200):
        super().__init__(Nt, N, hat_t, hat_U)
        
        def force(): 
            ''' calc the force according to eq. (221), with two chiral indices combined '''
            self.m_mat = self.m_mat_indep + m_matrix_xi(self.Nt, self.N, self.hat_U, self.xi)
            self.f_mat = self.m_mat @ self.m_mat.T

            ret = np.zeros_like(self.xi) 
            for phi in self.phi:
                x = spsolve(self.f_mat, phi) 
                y = self.m_mat.T @ x  # can't figure out why need a transpose here, but it worked! 
                ret += 2 * self.hat_U **.5 * (x * y) # trace is too fancy, simply an element-wise product 

            return ret


        def hamiltonian():
            ''' calc the hamiltonian according to eq. (219), without the sum of alpha '''
            nonlocal pi
            # f_mat is always updated before calcing hamiltonian
            return  sum(phi@spsolve(self.f_mat, phi) for phi in self.phi)


        def leapfrog():
            ''' the leapfrog algorithm ''' 
            nonlocal pi
            pi -= force() * (time_step / 2)
            for _ in range(max_steps - 1):
                self.xi += pi * time_step 
                pi -= force() * time_step
            self.xi += pi * time_step 
            pi -= force() * (time_step / 2)

        self._generate_phi()
        tmp_ham = []
        self.tot_delta_h = []
        for epoch in tqdm(range(max_epochs)):
            prev_xi = self.xi.copy()  # make a copy of previous state in case hamiltonian gets bad 

            pi = np.random.randn(self.half_size * 2)  # random momenta
            
            h_start = hamiltonian()  # record the hamiltonian at the beginning

            # launch leapfrog algorithm
            leapfrog()

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
    show_histogram(HamiltonianScale(8,4,2,1,).tot_delta_h)
