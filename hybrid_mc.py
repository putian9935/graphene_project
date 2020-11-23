__doc__ = """
HMC described in (217). 

==================
Jobs done:
1. Sanity check on hamiltonian derivative;

2. Save fixed part M matrix for acceleration;

3. Fix wrong leapfrog (sign before force term); 

==================
To do:
1. Merge different statistics into one function 
"""

import numpy as np 
from m_matrix import tau_n2ind, m_matrix_same4all, m_matrix_xi, m_matrix_tau_free, ft2d_speedup, m_matrix_tau_shift
from scipy.sparse.linalg import spsolve, inv, cg, cgs, gcrotmk, splu, bicg, bicgstab, gmres, lgmres, minres, qmr
from scipy import sparse
from tqdm import tqdm

import pickle 

np.random.seed(42)  # fix seed for reproductibility 


class Trajectory():
    """
    Save a single trajectory of Markov chain.
    """
    tot_updates = 0 # statistics on rej rate
    rej_updates = 0
    delta_ham = []
    def __init__(self, Nt, N, hat_t, hat_U, max_epochs=400):
        self.N, self.Nt, self.half_size = N, Nt, N*N*Nt
        self.hat_t, self.hat_U = hat_t, hat_U
        self.xi = np.random.randn(N*N*Nt*2)  # the whole purpose of rescaling is to make xi follow gaussian
        self.xis = []
        self.max_epochs = max_epochs
        
        self.m_mat_indep = m_matrix_same4all(self.Nt, self.N, self.hat_t, self.hat_U,)

    def _generate_phi(self):
        ''' Generate phi vector according to eq. (163) '''
        self.phi = []  # make a list for good iteration
        self.m_mat_indep = m_matrix_same4all(self.Nt, self.N, self.hat_t, self.hat_U,)

        # Few necessary initialization
        self.m_mat = self.m_mat_indep + m_matrix_xi(self.Nt, self.N, self.hat_U, self.xi)
        self.f_mat = self.m_mat @ self.m_mat.T

        # divide by sqrt(2) since eta does not follow standard gaussian 
        self.phi.append(self.m_mat @ np.random.randn(self.half_size * 2) / 2 ** .5)
        self.phi.append(self.m_mat @ np.random.randn(self.half_size * 2) / 2 ** .5)

        
        self.xs = []
        for phi in self.phi:
            self.xs.append(bicgstab(self.f_mat,phi)[0])


    
    def evolve(self, time_step, max_steps=3):
        """ 
        Evolve using leapfrog algorithm introduced on page 28;
        After this function call, self will be equipped with M matrix.

        Might replace leapfrog method with other higher-order symplectic integrator if necessary.
        """
        
        def force(): 
            ''' calc the force according to eq. (221), with two chiral indices combined '''
            self.m_mat = self.m_mat_indep + m_matrix_xi(self.Nt, self.N, self.hat_U, self.xi)
            self.f_mat = self.m_mat @ self.m_mat.T
            self.xs = []

            ret = -self.xi

            """
            # Benchmarking different method
            # seems different method has different precision, but no matter?
            
            import time
            print('start_solving')
            for func in [cg, cgs, gcrotmk, bicg, bicgstab, gmres, lgmres, minres, qmr]:
                tt = time.clock()
                x= func(self.f_mat, self.phi[0])[0]
                print('%s: %.3f'%(func.__name__, time.clock()-tt))
                print(np.linalg.norm(self.phi[0] - self.f_mat@x))
            input('Finished')
            exit()
            """

            for phi in self.phi:
                self.xs.append(bicgstab(self.f_mat,phi)[0])
                y = self.m_mat.T @ self.xs[-1]  # can't figure out why need a transpose here, but it worked! 
                ret += 2 * self.hat_U **.5 * (self.xs[-1] * y) # trace is too fancy, simply an element-wise product 

            return ret


        def hamiltonian():
            ''' calc the hamiltonian according to eq. (219), without the sum of alpha '''
            nonlocal pi
            # f_mat, xs is always updated before calcing hamiltonian
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

        
        def two_step_leapfrog():
            ''' the leapfrog algorithm ''' 
            nonlocal pi
            pi += force() * (time_step / 2)
            self.xi += pi * time_step 
            pi += force() * time_step
            self.xi += pi * time_step 
            pi += force() * (time_step / 2)


        def one_step_leapfrog():
            ''' the leapfrog algorithm ''' 
            nonlocal pi
            pi += force() * (time_step / 2)
            self.xi += pi * time_step 
            pi += force() * (time_step / 2)


        tmp_ham = []
        for epoch in tqdm(range(self.max_epochs)):
            prev_xi = self.xi.copy()  # make a copy of previous state in case hamiltonian gets bad 

            self._generate_phi()

            pi = np.random.randn(self.half_size * 2)  # random momenta
            # print(pi[:10], self.phi[0][:10],self.phi[1][:10])
            # print(self.xi[:10])
            h_start = hamiltonian()  # record the hamiltonian at the beginning

            # launch leapfrog algorithm
            # one_step_leapfrog()
            # two_step_leapfrog()
            leapfrog()

            h_end = hamiltonian()  # record the hamiltonian at the end
            # print(pi[:10], self.phi[0][:10],self.phi[1][:10])
            # print(self.xi[:10])
            # two components of xi, phi are independent, 
            # so it's cool to accept one update while rejecting another
            # print(h_start, h_end) 
            # input()
            Trajectory.tot_updates += 1
            # for visualization 
            # tmp_ham.append(['acc', h_end-h_start, prev_xi.copy(), self.xi.copy()])

            tmp_ham.append(['acc', h_end-h_start])
            
            if h_end < h_start: # exp might overflow
                self.xis.append(self.xi.copy())  # a copy is necessary
                continue
            if np.random.random() > np.exp(-h_end + h_start):
                self.xi = prev_xi  
                Trajectory.rej_updates += 1
                tmp_ham[-1][0]='rej'

                # no continue here, save all points as sample
                # continue
            self.xis.append(self.xi.copy())
            
                
        Trajectory.delta_ham.append(tmp_ham)


class Solution():  # cannot bear passing same arguments, use class instead
    def __init__(self, Nt, N, hat_t, hat_U, max_epochs=400, time_step=0.4, from_file=False, filename=None):
        ''' Initialise as is. Results are always pickled for later ease. ''' 
        self.N, self.Nt = N, Nt
        self.hat_t, self.hat_U = hat_t, hat_U 
                
        print('''Parameters are: N=%d, Nt=%d,  
                hat t=%.3e, hat U=%.3e, 
                t=%.3e, U=%.3e,  
                Epochs=%d, delta t_0=%.3e'''
                %(N, Nt, hat_t, hat_U, Nt*hat_t, Nt*hat_U, max_epochs,time_step))

        if not filename:
            filename = 'N%dNt%dhatt%.2ehatU%.2ets%.2eep%d.pickle'%(N,Nt,hat_t,hat_U,time_step,max_epochs)
        if not from_file:
            self._generate_trajectories(max_epochs, time_step)
            with open(filename, 'wb') as f:
                pickle.dump(self.traj, f)
                print('hello')
        else: 
            with open(filename, 'rb') as f:
                self.traj = pickle.load(f)
                
                

    def _generate_trajectories(self, max_epochs, time_step):
        ''' Generate trajectories using HMC ''' 
        self.traj = Trajectory(self.Nt, self.N, self.hat_t, self.hat_U, max_epochs)
        self.traj.evolve(time_step=time_step)

        print('Acceptance Rate:%.2f%%,\nAcc/Tot:  %d/%d' % (100*(Trajectory.tot_updates-Trajectory.rej_updates)/Trajectory.tot_updates,Trajectory.tot_updates-Trajectory.rej_updates,Trajectory.tot_updates))
        print('Mean change in Hamiltonian is', np.array(Trajectory.delta_ham)[...,1].squeeze().astype('float64').mean())
    

    def spin_correl_aa(self, burnin=None):
        """ 
        Calc observables via eq. (246)

        As Feng mentioned, NEVER use matrix inverse unless you have a good reason. Following code uses cached solver to speedup solution. 

        Imaginary time average is performed from tau = 1 to Nt-1. 
        """
        
        if not burnin: 
            burnin = len(self.traj.xis) // 2 

        print('Doing statistics...')
        
        ret = np.zeros((self.N, self.N))  # a function of R and tau

        buf = np.zeros(self.N*self.N*self.Nt*2)
        
        sample = self.traj.xis[burnin:]
        for xi in tqdm(sample):
            inv_solver = splu(self.traj.m_mat_indep + m_matrix_xi(self.Nt, self.N, self.hat_U, xi))
            for tau in range(1, self.Nt-1):      
                ind0 = tau_n2ind(tau, 0, 0, self.N)  # ind0 is irrelevant to n1, n2, thus to indr as well
                buf[ind0] = 1
                sol_buf = inv_solver.solve(buf) # thus we can solve for buf once for a fixed tau
                for n1 in range(self.N):
                    for n2 in range(self.N):
                        # this line is effectively an inner product 
                        ret[n1, n2,] += sol_buf[tau_n2ind(tau+1, n1, n2, self.N)] ** 2
                buf[ind0] = 0

        ret /= len(sample) * (self.Nt-2)
        print(ret[0, 0])
        return ret


    def number_correl_aa(self, burnin=None):
        """ 
        Calc observables via eq. (204), (205)
        
        Imaginary time average is performed from tau = 1 to Nt-1. 
        """
        
        if not burnin: 
            burnin = len(self.traj.xis) // 2 

        print('Doing statistics...')
        
        ret = np.zeros((self.N, self.N))  # a function of R and tau

        buf = np.zeros(self.N*self.N*self.Nt*2)
        
        sample = self.traj.xis[burnin:]
        for xi in tqdm(sample):
            inv_solver = splu(self.traj.m_mat_indep + m_matrix_xi(self.Nt, self.N, self.hat_U, xi))
            for tau in range(1, self.Nt-1):      
                ind0 = tau_n2ind(tau, 0, 0, self.N)  # ind0 is irrelevant to n1, n2, thus to indr as well
                buf[ind0] = 1.
                sol_buf = inv_solver.solve(buf) # thus we can solve for buf once for a fixed tau
                for n1 in range(self.N):
                    for n2 in range(self.N):
                        # this line is effectively an inner product 
                        ret[n1, n2,] += sol_buf[tau_n2ind(tau+1, n1, n2, self.N)] ** 2
                buf[ind0] = 0

        ret /= len(sample) * (self.Nt-2)
        print(ret[0, 0])
        return ret


    def calc_spectra(self, burnin=None):
        r'''
        Use two point function defined in (194) to extract energy spectra. See notes for further info. 

        No need to Fourier transform the whole matrix. Only upper right and lower left is needed, and can be preprocessed.    
        '''

        def no_exception_wrapper(func):
            def ret_func(*args):
                try:
                    return func(*args)
                except RuntimeError:  # fit failed
                    return ((np.nan,),)  # simulate [0][0] style
            return ret_func

        if not burnin: 
            burnin = len(self.traj.xis) // 2 

        sample = self.traj.xis[burnin:]

        s1, s2 = m_matrix_tau_free(self.Nt,self.N,self.hat_t)
        e_rl, e_lr = ft2d_speedup(s1, self.Nt, self.N), ft2d_speedup(s2, self.Nt, self.N)  # ft2d_speedup is not very right
        tilde_m_mat_indep = m_matrix_tau_shift(self.Nt,self.N,0) \
                            +sparse.kron(np.array([[0,0],[1,0]]), e_rl) \
                            +sparse.kron(np.array([[0,1],[0,0]]), e_lr)

        ret = np.zeros((self.N, self.N, self.Nt,), dtype='complex128')
        for xi in tqdm(sample):
            tilde_m = tilde_m_mat_indep + m_matrix_xi(self.Nt, self.N, self.hat_U, xi)
            for k1 in range(self.N): 
                for k2 in range(self.N):
                    ind = k1 * self.N + k2
                    ret[k1,k2] += \
                        spsolve(
                            tilde_m[ind::self.N*self.N,ind::self.N*self.N].T,  # solve in one sub-space of k 
                            np.array([1]+[0]*(2*self.Nt-1))
                        )[:self.Nt]  # take only first half 
        ret /= len(sample)  # do average
                    
        from scipy.optimize import curve_fit  
        """
        import matplotlib.pyplot as plt               
        for k1 in range(self.N):
            for k2 in range(self.N):
                plt.plot(np.log(np.abs(ret[k1, k2])))
                plt.title('k1=%d, k2=%d'%(k1, k2))
                plt.show()
        """
        np.savetxt('buf_trivial.csv', ret[0, 0],delimiter=',')
        print(ret[0,0])
        input()
        return np.array(
            [[(no_exception_wrapper(curve_fit)(
                lambda _, a, b, c:a*(_-b)**2+c, np.linspace(0,1,self.Nt), 
                np.log(
                    np.abs(
                        ret[k1, k2]
                    )
                )
            )[0][0]*2)**.5 for k1 in range(self.N)] 
            for k2 in range(self.N)]
            ) / (self.Nt * self.hat_t)  # divide by t=N_t\times\hat t, to make it invariant 
      
                     
    def calc_auto_correlation(self, mapping_func=None, burnin=None):
        ''' Calc auto-correlation of function of xi's ''' 
        if not burnin: 
            burnin = len(self.traj.xis) // 2 
        if not mapping_func:
          mapping_func = lambda _: _  

        print('Calculating auto-correlation...')
        
        import tidynamics  # a very good library that calc acf 
        x = np.array([mapping_func(xi) for xi in self.traj.xis[burnin:]])  # apply the mapping function
        
        self.acf = tidynamics.acf(x-x.mean(0))  # subtract mean from sample
        self.acf /= self.acf[0]  # normalize the result         
        
        print('Done! ')

        return self.acf
    

    def calc_auto_correlation_with_coarsen(self, mapping_func=None, burnin=None):
        ''' Coarsening to determine ac time.  ''' 
        if not burnin: 
            burnin = len(self.traj.xis) // 2 
            # burnin=5000
        if not mapping_func:
          mapping_func = lambda _: _  

        print('Calculating auto-correlation with linear coarsening...')
        
        from block import linear_blocking
        
        x = linear_blocking(np.array([mapping_func(xi) for xi in self.traj.xis[burnin:]]))  # apply the mapping function
        
        print('Done! ')
        
        return x 

    
    
       
       
def show_single_plot(res):
    plt.figure(figsize=(6,6))
    plt.colorbar(plt.matshow(np.log(res))) 
    plt.savefig('result_sum.png')



if __name__ == '__main__':
    import matplotlib.pyplot as plt 
    np.set_printoptions(precision=4,linewidth=120)  # otherwise you will have a bad time debugging code
    # sol = Solution(50,10,1e-1,1e-3, time_step=0.35, max_epochs=1000)
    # sol = Solution(50, 5, 2e-3, 1e-7, time_step=0.3, max_epochs=20000) 
    
    sol = Solution(50,5,2e-3,1e-7, time_step=0.22, max_epochs=50000, from_file=True)
    # sol = Solution(50,5,2e-3, 0., time_step=0.22, max_epochs=100, act=40, from_file=False)  # turn off interaction

    plt.matshow(np.log(sol.number_correl_aa()))
    plt.show()
    input()
    print(sol.calc_spectra())
    # act has higher priority 
    plt.plot(sol.calc_auto_correlation_with_coarsen(lambda _: _@_))
    plt.ylabel(r'Blocked sample mean $\sigma^2_{O_B}/N_B$')
    plt.xlabel('Block size $N_B$')
    # plt.show()
    plt.savefig('norm_bin.png')

    plt.plot(sol.calc_auto_correlation_with_coarsen(lambda _: _[0]))
    plt.ylabel(r'Blocked sample mean $\sigma^2_{O_B}/N_B$')
    plt.xlabel('Block size $N_B$')
    # plt.show()
    plt.savefig('norm_first.png')


    """    
    sol.calc_auto_correlation(mapping_func=lambda _: _)
    plt.figure(figsize=(8,6))
    plt.plot(sol.acf[:200]) 
    plt.title(r'Mapping function: lambda _: _')
    plt.show()
    # plt.savefig('itself.png')
    
    sol.calc_auto_correlation(mapping_func=lambda _: _@_)
    plt.figure(figsize=(8,6))
    plt.plot(sol.acf[:200]) 
    plt.title(r'Mapping function: lambda _: _@_')
    plt.show()
    # plt.savefig('norm.png')
    
    sol.calc_auto_correlation(mapping_func=lambda _: _[0])
    plt.figure(figsize=(8,6))
    plt.plot(sol.acf[:200]) 
    plt.title(r'Mapping function: lambda _: _[0]')
    plt.show()
    # plt.savefig('first_component.png')
    """
    
