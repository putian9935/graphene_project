__doc__ = """
A class used to test if derivative is calculated correctly. 
The method is to compare the derivative calculated analytically with the one calculated with finite difference. 
"""

from hybrid_mc import Trajectory
from m_matrix import m_matrix_xi
from scipy.sparse.linalg import spsolve
import numpy as np 

class TestCorrectness(Trajectory):
    def __init__(self,Nt, N, hat_t, hat_U):
        super().__init__(Nt, N, hat_t, hat_U)
        
        def force(): 
            ''' calc the force according to eq. (221), with two chiral indices combined '''
            self.m_mat = self.m_mat_indep + m_matrix_xi(self.Nt, self.N, self.hat_U, self.xi)
            self.f_mat = self.m_mat @ self.m_mat.T

            ret = -self.xi
            for phi in self.phi:
                x = spsolve(self.f_mat, phi) 
                y = self.m_mat.T @ x
                ret += 2 * self.hat_U **.5 * (x * y) # trace is too fancy, simply an element-wise product 


            return ret


        def hamiltonian():
            ''' calc the hamiltonian according to eq. (219), without the sum of alpha '''
            nonlocal pi
            # f_mat is always updated before calcing hamiltonian
            
            self.m_mat = self.m_mat_indep + m_matrix_xi(self.Nt, self.N, self.hat_U, self.xi)
            self.f_mat = self.m_mat @ self.m_mat.T
            return  .5*(pi@pi+self.xi@self.xi)+sum(phi@spsolve(self.f_mat, phi) for phi in self.phi)


        self._generate_phi()
        
        pi = np.random.randn(self.half_size * 2)  # random momenta
        
        
        h = hamiltonian()  # record the hamiltonian at the beginning 
        f = force()

        # perturbation = np.random.randn(self.half_size * 2) / 1e5  
        perturbation = np.array([1e-3] + [0] *(self.half_size*2-1))
        self.xi += perturbation 

        h_perturbed = hamiltonian() 
        
        print('Delta H computed using definition: ', h_perturbed - h) 
        
        print('Delta H computed using derivative: ', -perturbation @ f)


if __name__ == '__main__':
    # Sample usage 
    TestCorrectness(5,3,1e-3,1e-5)      
