__doc__ = """
Use preconditioning technique to speed up CG algorithm. 
"""


from m_matrix import m_matrix_same4all, m_matrix_xi, m_matrix_shifted_same4all
import numpy as np
from scipy.sparse.linalg import cg, inv, lsmr, cgs
from time import perf_counter

def show_mat(sparse_m):
    import matplotlib.pyplot as plt 
    plt.matshow(sparse_m.toarray())
    plt.show()

class PreconditionTestF():
    def __init__(self, Nt, N, hat_t, hat_u):
        self.Nt = Nt 
        self.N = N 
        self.hat_t = hat_t 
        self.hat_u = hat_u
        # self.m_mat_indep = m_matrix_same4all(Nt, N, hat_t, hat_u)
        self.m_mat_indep = m_matrix_shifted_same4all(Nt, N, hat_t, hat_u)
        self.mat_size = Nt * N * N 
        self.phi = self.m_mat_indep @ np.random.random(self.mat_size * 2)
        self.m_mat = self.m_mat_indep + m_matrix_xi(self.Nt, self.N, self.hat_u, np.random.random(self.mat_size * 2)) 
        # show_mat(self.m_mat_indep)
        # self.f_mat_indep = self.m_mat_indep @ self.m_mat_indep.T
        
        
        self.m_mat_zero_t = m_matrix_same4all(Nt, N, 0, hat_u).tocsc()

        self.f_mat_indep = self.m_mat_indep @ self.m_mat_indep.T

        import scipy.sparse.linalg as spla
        import scipy.sparse as sparse
        M_x = lambda x: spla.spsolve(self.f_mat_indep, x)
        self.f_mat_inv_op = spla.LinearOperator((2*N*N*Nt, 2*N*N*Nt), M_x)
        self.f_mat_inv = inv(self.m_mat_zero_t @ self.m_mat_zero_t.T + sparse.eye(2*N*N*Nt)*.1)
        
        print(np.linalg.cond(self.m_mat.toarray()))
        print(np.linalg.cond((self.m_mat @ self.m_mat.T).toarray()))

        self.iter = 0 
    

    def call_back(self, *args):
        self.iter+=1 

    def _solve_without_pc(self):
        f_mat = self.m_mat @ self.m_mat.T  # this is always done, actaully only once 
        self.iter = 0
        return cg(f_mat, self.phi, callback = self.call_back)
    
    def _solve_with_pc(self):
        f_mat = self.m_mat @ self.m_mat.T  # this is always done, actaully only once
        self.iter = 0
        return cg(f_mat, self.phi, M=self.f_mat_inv, callback=self.call_back)
    

    def benchmark(self):
        tt = perf_counter()
        for _ in range(20):
            self._solve_without_pc()
        print('Without preconditioning: ', perf_counter() - tt, '\n# of iterations: ', self.iter)


        tt = perf_counter()
        for _ in range(20):
            self._solve_with_pc()
        print('With preconditioning(F): ', perf_counter() - tt, '\n# of iterations: ', self.iter)

        tt = perf_counter()
        f_mat = (self.m_mat @ self.m_mat.T).toarray()
        self.iter = 0
        for _ in range(20):
            cg(f_mat, self.phi,callback = self.call_back)
        print('Without preconditioning (dense): ', perf_counter() - tt, '\n# of iterations: ', self.iter)

        
        tt = perf_counter()
        f_mat = (self.m_mat @ self.m_mat.T).toarray()
        self.iter = 0
        for _ in range(20):
            cg(f_mat, self.phi, M=self.f_mat_inv.toarray(), callback = self.call_back)
        print('With preconditioning (dense): ', perf_counter() - tt, '\n# of iterations: ', self.iter)



np.random.seed(42)

sol_F = PreconditionTestF(10, 4, 2, 9) 
sol_F.benchmark()


# sol_M = PreconditionTestM(10, 3, 1e-2, 1e-5)
# sol_M.benchmark()

