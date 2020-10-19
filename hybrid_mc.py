__doc__ = """
HMC described in (217). 

==================
Jobs done:
1. Sanity check on hamiltonian derivative;

2. Save fixed part M matrix for acceleration;

"""

import numpy as np 
from m_matrix import tau_n2ind, m_matrix_same4all, m_matrix_xi
from scipy.sparse.linalg import spsolve, inv
from scipy import sparse
from tqdm import tqdm

np.random.seed(42)  # fix seed for reproductibility 

class Trajectory():
    """
    Save a single trajectory of Markov chain.
    """
    tot_updates = 0 # statistics on rej rate
    rej_updates = 0
    delta_ham = []
    def __init__(self, Nt, N, hat_t, hat_U):
        self.N, self.Nt, self.half_size = N, Nt, N*N*Nt
        self.hat_t, self.hat_U = hat_t, hat_U
        self.xi = np.random.randn(N*N*Nt*2)
        self.xis = []

    def _generate_phi(self):
        ''' Generate phi vector according to eq. (163) '''
        self.phi = []  # make a list for good iteration
        self.m_mat_indep = m_matrix_same4all(self.Nt, self.N, self.hat_t, self.hat_U,)
        self.m_mat = self.m_mat_indep + m_matrix_xi(self.Nt, self.N, self.hat_U, self.xi)
        self.f_mat = self.m_mat @ self.m_mat.T

        # divide by sqrt(2) since eta does not follow standard gaussian 
        self.phi.append(self.m_mat @ np.random.randn(self.half_size * 2) / 2 ** .5)
       
    
    
    def evolve(self, time_step=0.005, max_steps=10, max_epochs=400):
        """ 
        Evolve using leapfrog algorithm introduced on page 28;
        After this function call, self will be equipped with M matrix.

        Might replace leapfrog method with other higher-order symplectic integrator if necessary.
        """
        
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
        for epoch in tqdm(range(max_epochs)):
            prev_xi = self.xi.copy()  # make a copy of previous state in case hamiltonian gets bad 

            pi = np.random.randn(self.half_size * 2)  # random momenta
            
            h_start = hamiltonian()  # record the hamiltonian at the beginning

            # launch leapfrog algorithm
            leapfrog()

            h_end = hamiltonian()  # record the hamiltonian at the end
            
            # two components of xi, phi are independent, 
            # so it's cool to accept one update while rejecting another

            Trajectory.tot_updates += 1
            tmp_ham.append(['acc', h_end-h_start])
            if h_end < h_start: # exp might overflow
                self.xis.append(self.xi)
                continue
            if np.random.random() > np.exp(-h_end + h_start):
                self.xi = prev_xi  
                Trajectory.rej_updates += 1
                tmp_ham[-1][0]='rej'
            self.xis.append(self.xi)
                
        Trajectory.delta_ham.append(tmp_ham)

                
class TestCorrectness(Trajectory):
    def __init__(self,Nt, N, hat_t, hat_U):
        super().__init__(Nt, N, hat_t, hat_U)
        
        def force(): 
            ''' calc the force according to eq. (221), with two chiral indices combined '''
            self.m_mat = self.m_mat_indep + m_matrix_xi(self.Nt, self.N, self.hat_U, self.xi)
            self.f_mat = self.m_mat @ self.m_mat.T

            ret = np.zeros_like(self.xi) 
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
            return  sum(phi@spsolve(self.f_mat, phi) for phi in self.phi)


        self._generate_phi()
        tmp_ham = []
        
        prev_xi = self.xi.copy()  # make a copy of previous state in case hamiltonian gets bad 

        pi = np.random.randn(self.half_size * 2)  # random momenta
        
        
        h = hamiltonian()  # record the hamiltonian at the beginning 
        f = force()

        # perturbation = np.random.randn(self.half_size * 2) / 1e5  
        perturbation = np.array([1e-7] + [0] *(self.half_size*2-1))
        self.xi += perturbation 

        h_perturbed = hamiltonian() 
        
        print('Delta H computed using definition: ', h_perturbed - h) 
        
        print('Delta H computed using derivative: ', -perturbation @ f)
        

class Solution():  # cannot bear passing same arguments, use class instead
    def __init__(self, Nt, N, hat_t, hat_U):
        self.N, self.Nt = N, Nt
        self.hat_t, self.hat_U = hat_t, hat_U 

        self._generate_trajectories()

    def _generate_trajectories(self):
        ''' Generate trajectories using HMC ''' 
        self.traj = Trajectory(self.Nt, self.N, self.hat_t, self.hat_U)
        self.traj.evolve()
    

    def two_point_aa(self, burnin=200):
        ''' Calc observables via eq. (246) ''' 
        
        print('Doing statistics...')
        ret = np.zeros((self.Nt - 1,self.N, self.N))  # a function of R and tau

        for xi in tqdm(self.traj.xis[burnin:]):
            inv_mat = inv(self.traj.m_mat_indep + m_matrix_xi(self.Nt, self.N, self.hat_U, xi))
            
            for tau in range(self.Nt-1):
                for n1 in range(self.N):
                    for n2 in range(self.N):
                        indr = tau_n2ind(tau+1, n1, n2, self.N) 
                        ind0 = tau_n2ind(tau, 0, 0, self.N) 
                        ret[tau, n1, n2,] += inv_mat[indr, ind0] ** 2 
            ret /= len(self.traj.xis) 
        return ret


def show_plot(results):
    import matplotlib.pyplot as plt
    for i, res in enumerate(results):
        plt.imshow(res) 
        plt.savefig('%d.png' % i, )



if __name__ == '__main__':
    # TestCorrectness(2,2,1,4)
    sol = Solution(8,4,2,1)
    print('Acceptance Rate:%.2f%%,\nAcc/Tot:  %d/%d' % (100*(Trajectory.tot_updates-Trajectory.rej_updates)/Trajectory.tot_updates,Trajectory.tot_updates-Trajectory.rej_updates,Trajectory.tot_updates))
    print(np.array(Trajectory.delta_ham)[...,1].squeeze().astype('float64').mean())

    show_plot(sol.two_point_aa())
