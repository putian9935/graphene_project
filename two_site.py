__doc__ = """
Calculate two-site (staggered) magnetic moment, 
"""

import numpy as np 
from hybrid_mc import Solution, Trajectory  # Trajectory needs import for Pickle
from scipy.sparse.linalg import inv

from m_matrix import m_matrix_xi
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.linalg import inv as minv

def my_expm2(m):
    return np.eye(2) + m + .5 * m @ m 

def my_expm1(m):
    return np.eye(2) + m # + .5 * m @ m 

class TwoSiteZero(Solution):
    def __init__(self, myexpm, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.m0 = -myexpm(-np.array([[-kwargs['e'],-args[2]],[-args[2],-kwargs['e']]]))
        self.e = kwargs['e']


    def charge(self, m_inv):
        return 1.-np.mean(m_inv.toarray().diagonal())

    def magnetic_moment(self, m_inv):
        buf = m_inv# .toarray()
        return -1*np.sum(buf[0,2]*2)-1
    
    def magnetic_moment2(self, m_inv):
        return 1*(2-np.mean(np.diagonal(m_inv, offset=-2))*2)-1

    def staggered_magnetic_moment2(self, m_inv):
        offset_diagonal = np.diagonal(m_inv, offset=-2)
        return (np.mean(offset_diagonal[::2])- np.mean(offset_diagonal[::1]))*2
        

    def calc_auto_correlation_with_coarsen(self, x):
        ''' Coarsening to determine ac time.  ''' 
        from block import linear_blocking
        
        return linear_blocking(x)  

    def stat(self, stat_funcs, burnin=None):
        if not burnin: 
            burnin = len(self.traj.xis) // 2 

        res = []
        print('Doing statistics...')
        # from scipy.linalg import inv as dinv
        m = self.traj.m_mat_indep.toarray()
        for i in range(self.Nt):
            m[2*i:2*(i+1),2*i:2*(i+1)] = self.m0 
            
        m_inv = inv(self.traj.m_mat_indep).toarray()
        # print(m_inv[2,0])
        # m_inv = minv(m)
        
        # return self.magnetic_moment2(m_inv)
    
        return self.staggered_magnetic_moment2(m_inv)
        

class TwoSite(Solution):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        # plt.matshow(self.traj.m_mat_indep.toarray())
        # plt.show()

    def charge(self, m_inv):
        return 1.-np.mean(m_inv.toarray().diagonal())

    def magnetic_moment2(self, m_inv):
        buf = m_inv# .toarray()
        
        # print(1*np.sum(2-(14*buf[2,0]+12*buf[4,0])/26.*2)-1)
        return 1*(2-np.mean(np.diagonal(m_inv, offset=-2))*2)-1

    def staggered_magnetic_moment(self, m_inv):
        offset_diagonal = np.diagonal(m_inv, offset=-2)
        return (np.mean(offset_diagonal[::2])- np.mean(offset_diagonal[::1]))*2
    
    def calc_auto_correlation_with_coarsen(self, x):
        ''' Coarsening to determine ac time.  ''' 
        from block import linear_blocking
        
        return linear_blocking(x)  

    def stat(self, stat_funcs, burnin=None):
        if not burnin: 
            burnin = len(self.traj.xis) // 2 

        res = []
        print('Doing statistics...')
        
        # m_inv = inv(self.traj.m_mat_indep)
        # return self.magnetic_moment(m_inv), self.magnetic_moment2(m_inv)

        for xi in tqdm(self.traj.xis[burnin:]):
            d_mat = m_matrix_xi(self.Nt, self.N, self.hat_U, xi)
            m_inv = inv(self.traj.m_mat_indep + d_mat - .5*(self.traj.t_mat_indep@d_mat+d_mat@self.traj.t_mat_indep)-.5*d_mat@d_mat).toarray()

            # plt.matshow(m_inv) 
            # plt.show()
            new_entry = []
            for f in stat_funcs: 
                new_entry.append(f(m_inv))
            res.append(np.array(new_entry))
        
        res = np.array(res)

        for i in range(len(stat_funcs)):
            plt.plot(self.calc_auto_correlation_with_coarsen(res[:,i]) )
            plt.xlabel('$N_B$')
            plt.ylabel('$\sigma^2$')
            plt.xlim(left=0)
            plt.grid()
            # plt.show()
            plt.savefig('h%.3e.pdf'%self.e)
        return np.mean(res, axis=0), np.std(res, axis=0) 


def plot_zero_tU():
    '''
    Be sure to comment out extra print lines from hybrid_mc.py 

    distracting and slows down the program
    '''
    hat_u, hat_t = .00, 0. # 1./Nt/3

    ts=.5

    emax = 6.4
    for Nt in [8, 16, 32, 64, 128]:
        em = emax / Nt
        
        res = []
        for e in np.linspace(-em, em, endpoint=True):
            sol = TwoSiteZero(Nt,1,hat_t, hat_u, e=e, time_step=ts, max_epochs=1, 
                from_file=False,) 

            res.append(np.array(sol.stat(stat_funcs=[sol.magnetic_moment]))) 
        es = np.linspace(-em, em, endpoint=True) * Nt 

        res = np.array(res)
        # plt.plot(es, res[:,0], label=r'$N_t=%d$'%Nt)
        plt.plot(es, res[:,1], label=r'$N_t=%d$'%Nt)

    es = np.linspace(-6.4, 6.4, endpoint=True)
    plt.plot(es,np.tanh(es/2.), label='ED')
    plt.xlabel(r'$\beta \mu_B B$')
    plt.ylabel(r'$\langle m \rangle$')
    plt.grid()
    plt.ylim(-1.2,1.2)
    plt.legend()
    plt.show()


def plot_staggered_zero_parameter():
    '''
    Be sure to comment out extra print lines from hybrid_mc.py 

    distracting and slows down the program
    '''
    u, t = 0, 10

    ts=.5

    emax = 3
    for Nt in [16,]:
        hat_u = u / Nt 
        hat_t = t / Nt
        em = emax / Nt
        
        res = []

        for e in np.linspace(0, em, endpoint=True):
            sol = TwoSiteZero(my_expm1,Nt,1,hat_t, hat_u, e=e, time_step=ts, max_epochs=1, 
                from_file=False,staggered=True) 

            res.append(np.array(sol.stat(stat_funcs=[sol.magnetic_moment]))) 
            
            print(hat_t*Nt, e*Nt, res[-1])
            input()
        es = np.linspace(-em, em, endpoint=True) * Nt 

        res = np.array(res)
        # plt.plot(es, res[:,0], label=r'$N_t=%d$'%Nt)
        plt.plot(es, res[:], label=r'$N_t=%d$'%Nt)

    es = np.linspace(-6.4, 6.4, endpoint=True)
    # plt.plot(es,np.tanh(es/2.), label='ED')
    plt.xlabel(r'$\beta \mu_B B$')
    plt.ylabel(r'$\langle m \rangle$')
    plt.grid()
    plt.ylim(-1.2,1.2)
    plt.legend()
    plt.show()

def plot_zero_U():
    Nt = 16
    data_ed = np.genfromtxt('zeroU.csv', delimiter=',')

    # plt.plot(data_ed[:,0]*4, data_ed[:,3], '--', c='C0', label=r'$\beta t=0.2$ (ED)')
    # plt.plot(data_ed[:,0]*4, data_ed[:,2], '--', c='C1', label=r'$\beta t=0.5$ (ED)')
    # plt.plot(data_ed[:,0]*4, data_ed[:,1], '--', c='C2', label=r'$\beta t=1.0$ (ED)')
    for hat_t in [1]:
        em = 4 / Nt
            
        res = []
        
        sol = TwoSiteZero(my_expm1,Nt,1,10/Nt, 0, e=0, time_step=.3, max_epochs=1, 
                from_file=False,) 
        print(np.array(sol.stat(stat_funcs=[sol.magnetic_moment])))
        input()

        es = np.linspace(-em, em, 50, endpoint=True) * Nt 
        for e in es:
            sol = TwoSiteZero(my_expm2, Nt, 1, 4*hat_t/Nt, 0., e=4*e/Nt, time_step=.5, max_epochs=1, 
                from_file=False,) 

            res.append(np.array(sol.stat(stat_funcs=[sol.magnetic_moment]))) 
        

        res = np.array(res)
        # plt.plot(es, res[:,0], label=r'$N_t=%d$'%Nt)
        plt.plot(es, -res[:], label=r'$\beta t=%0.1f$ (HMC)'%hat_t)
    plt.xlabel(r'$\beta\mu_B B$')
    plt.ylabel(r'$\langle m\rangle$')
    plt.legend()
    plt.grid()
    plt.show()


def plot_zero_U_susceptibility():
    Nt = 16
    data_ed = np.genfromtxt('zeroU_sus.csv', delimiter=',')
    plt.plot(data_ed[:,0],-data_ed[:,1],'--',label='ED')
    beta = 4
    hm=0.0005

    res = []
    # expm = my_expm1
    ts = np.linspace(.1/4, 4/4)
    for t in ts:
        em = beta*hm / Nt
        sol = TwoSiteZero(my_expm1, Nt, 1, beta*t/Nt/3, 0., e=0, time_step=.5, max_epochs=1, 
            from_file=False,) 
        m1 = sol.stat(stat_funcs=[sol.magnetic_moment])
        sol = TwoSiteZero(my_expm1, Nt, 1, beta*t/Nt, 0., e=em, time_step=.5, max_epochs=1, 
            from_file=False,) 
        m2 = sol.stat(stat_funcs=[sol.magnetic_moment])
        res.append((m2-m1)/hm) 
    res = np.array(res)
    plt.plot(ts, -res, label='1st ord. approx.')
    
    res = []
    # expm = my_expm2
    for t in ts:
        em = beta*hm / Nt
        sol = TwoSiteZero(my_expm2, Nt, 1, beta*t/Nt, 0., e=0, time_step=.5, max_epochs=1, 
            from_file=False,) 
        m1 = sol.stat(stat_funcs=[sol.magnetic_moment])
        sol = TwoSiteZero(my_expm2, Nt, 1, beta*t/Nt, 0., e=em, time_step=.5, max_epochs=1, 
            from_file=False,) 
        m2 = sol.stat(stat_funcs=[sol.magnetic_moment])
        res.append((m2-m1)/hm) 
    res = np.array(res)
    plt.plot(ts, -res, label='2nd ord. approx.')
    
    plt.xlabel(r'$\beta t$')
    plt.ylabel(r'$\chi$')
    plt.legend()
    plt.grid()
    plt.show()

def plot_zero_t_susceptibility():
    Nt = 16
    # data_ed = np.genfromtxt('zeroU_sus.csv', delimiter=',')
    # plt.plot(data_ed[:,0],-data_ed[:,1],'--',label='ED')
    
    beta = 4
    hm = 0.0005
    t = 0
    res = []
    # expm = my_expm1
    ts = np.linspace(.1/4, 4/4)
    for t in ts:
        em = beta*hm / Nt
        sol = TwoSite(Nt, 1, beta*t/Nt/3, beta*u/Nt, e=0, time_step=.3, max_epochs=5000, 
            from_file=False,) 
        m1 = sol.stat(stat_funcs=[sol.magnetic_moment2])
        sol = TwoSite(Nt, 1, beta*t/Nt, beta*u/Nt, e=em, time_step=.3, max_epochs=5000, 
            from_file=False,) 
        m2 = sol.stat(stat_funcs=[sol.magnetic_moment2])
        res.append((m2[0]-m1[0])/hm) 
    res = np.array(res)
    print(res)
    plt.plot(ts, -res, label='HMC')
    
    
    plt.xlabel(r'$\beta t$')
    plt.ylabel(r'$\chi$')
    plt.legend()
    plt.grid()
    plt.show()



# plot_staggered_zero_parameter()

# plot_zero_tU()
plot_zero_U()
exit()
# plot_zero_U_susceptibility()

Nt = 32
beta = 4 
t = 0. 
u = .005
h = -2e-2

# plot_zero_t_susceptibility()


sol = TwoSite(Nt,1, beta*t/Nt,beta*u/Nt,e=beta*h/Nt, time_step=0.35, max_epochs=400, 
                from_file=False,staggered=False)


print(sol.stat(stat_funcs=[sol.magnetic_moment2]))

