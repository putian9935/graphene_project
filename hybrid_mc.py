import numpy as np 
from m_matrix import m_matrix, tau_n2ind  
from scipy.sparse.linalg import spsolve, inv
from tqdm import tqdm

np.random.seed(42)  # fix seed for reproductibility 

class Trajectory():
    """
    Save a single trajectory of Markov chain.
    """
    def __init__(self, Nt, N, hat_t, hat_U):
        self.N, self.Nt, self.half_size = N, Nt, N*N*Nt
        self.hat_t, self.hat_U = hat_t, hat_U
        self.xi = np.random.randn(N*N*Nt*2)


    def _generate_phi(self):
        ''' Generate phi vector according to eq. (163) '''
        self.phi = []  # make a list for good iteration
        self.m_mat = m_matrix(self.Nt, self.N, self.hat_t, self.hat_U, 
                    self.xi[:self.half_size],  # first half corresponds to chiral component L
                    self.xi[self.half_size:] # second half corresponds to chiral component R
                )
        self.f_mat = self.m_mat @ self.m_mat.T
        # divide by sqrt(2) since eta does not follow standard gaussian 
        self.phi.append(self.m_mat @ np.random.randn(self.half_size * 2) / 2 ** .5)
    
    def evolve(self, time_step=0.1, max_steps=10, max_epochs=20):
        """ 
        Evolve using leapfrog algorithm introduced on page 28;
        After this function call, self will be equipped with M matrix.

        Might replace leapfrog method with other higher-order symplectic integrator if necessary.
        """
        
        def force(): 
            ''' calc the force according to eq. (221), with two chiral indices combined '''
            self.m_mat = m_matrix(self.Nt, self.N, self.hat_t, self.hat_U, 
                        self.xi[:self.half_size],  # first half corresponds to chiral component L
                        self.xi[self.half_size:] # second half corresponds to chiral component R
                    )
            self.f_mat = self.m_mat @ self.m_mat.T

            ret = -self.xi 
            for phi in self.phi:
                x = spsolve(self.f_mat, phi) 
                y = spsolve(self.m_mat, x)
                ret += 2 * self.hat_U **.5 * (x * y) # trace is too fancy, simply an element-wise product 

            return ret


        def hamiltonian():
            ''' calc the hamiltonian according to eq. (219), without the sum of alpha '''
            nonlocal pi
            # f_mat is always updated before calcing hamiltonian
            return  .5*(pi@pi + self.xi@self.xi) + sum(phi@spsolve(self.f_mat, phi) for phi in self.phi)


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

        for epoch in range(max_epochs):
            prev_xi = self.xi.copy()  # make a copy of previous state in case hamiltonian gets bad 

            pi = np.random.randn(self.half_size * 2)  # random momenta
            
            h_start = hamiltonian()  # record the hamiltonian at the beginning

            # launch leapfrog algorithm
            leapfrog()

            h_end = hamiltonian()  # record the hamiltonian at the end
            
            # two components of xi, phi are independent, 
            # so it's cool to accept one update while rejecting another
            if h_end < h_start: continue
            if np.random.random() > np.exp(-h_end + h_start):
                self.xi = prev_xi  
                    


class Solution():  # cannot bear passing same arguments, use class instead
    def __init__(self, Nt, N, hat_t, hat_U, max_trajs=20):
        self.N, self.Nt = N, Nt
        self.hat_t, self.hat_U = hat_t, hat_U 
        self.max_trajs = max_trajs

        self._generate_trajectories()

    def _generate_trajectories(self):
        ''' Generate trajectories using HMC ''' 
        self.trajs = []

        for _ in tqdm(range(self.max_trajs)): 
            self.trajs.append(Trajectory(self.Nt, self.N, self.hat_t, self.hat_U))
            self.trajs[-1].evolve()
    

    def two_point_aa(self):
        ''' Calc observables via eq. (246) ''' 
        print(self.trajs)
        ret = np.zeros((self.Nt - 1,self.N, self.N))  # a function of R and tau
        for traj in self.trajs:
            inv_mat = inv(traj.m_mat)
            
            for tau in range(self.Nt-1):
                for n1 in range(self.N):
                    for n2 in range(self.N):
                        indr = tau_n2ind(tau+1, n1, n2, self.N) 
                        ind0 = tau_n2ind(tau, 0, 0, self.N) 
                        ret[tau, n1, n2,] += inv_mat[indr, ind0] ** 2 
        ret /= len(self.trajs) 
        return ret


def show_plot(results):
    import matplotlib.pyplot as plt
    for i, res in enumerate(results):
        plt.imshow(res) 
        plt.savefig('%d.png' % i, )



if __name__ == '__main__':
    sol = Solution(20,20,1,1.5)
    show_plot(sol.two_point_aa())

