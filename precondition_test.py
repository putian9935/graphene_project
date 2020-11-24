__doc__ = """
Use preconditioning technique to speed up CG algorithm. 
"""


from m_matrix import m_matrix_same4all, m_matrix_xi
import numpy as np
from scipy.sparse.linalg import cg, inv
from time import perf_counter

class PreconditionTestM():
    def __init__(self, Nt, N, hat_t, hat_u):
        self.Nt = Nt 
        self.N = N 
        self.hat_t = hat_t 
        self.hat_u = hat_u
        self.m_mat_indep = m_matrix_same4all(Nt, N, hat_t, hat_u)
        self.mat_size = Nt * N * N 
        self.phi = self.m_mat_indep @ np.random.random(self.mat_size * 2)
        self.m_mat = self.m_mat_indep + m_matrix_xi(self.Nt, self.N, self.hat_u, np.random.random(self.mat_size * 2)) 
        self.m_mat_inv = inv(self.m_mat_indep)

    
    def _solve_with_pc(self):
        # return cg(self.m_mat, self.phi, M=self.m_mat_inv)
        return cg(self.m_mat, self.phi, M=self.m_mat_indep.T)
    
    def _solve_without_pc(self):
        return cg(self.m_mat, self.phi)
    
    def benchmark(self):
        tt = perf_counter()
        for _ in range(20):
            self._solve_without_pc()
        print('Without preconditioning: ', perf_counter() - tt)

        tt = perf_counter()
        for _ in range(20):
            self._solve_with_pc()
        print('With preconditioning:    ', perf_counter() - tt)


class PreconditionTestF():
    def __init__(self, Nt, N, hat_t, hat_u):
        self.Nt = Nt 
        self.N = N 
        self.hat_t = hat_t 
        self.hat_u = hat_u
        self.m_mat_indep = m_matrix_same4all(Nt, N, hat_t, hat_u)
        self.mat_size = Nt * N * N 
        self.phi = self.m_mat_indep @ np.random.random(self.mat_size * 2)
        self.m_mat = self.m_mat_indep + m_matrix_xi(self.Nt, self.N, self.hat_u, np.random.random(self.mat_size * 2)) 
        # self.f_mat_indep = self.m_mat_indep @ self.m_mat_indep.T
        self.f_mat_indep = self.m_mat_indep @ self.m_mat_indep.T
        
        self.m_mat_inv = inv(self.m_mat_indep)
        self.f_mat_inv = inv(self.f_mat_indep)
    
    def _solve_without_pc(self):
        f_mat = self.m_mat @ self.m_mat.T  # this is always done, actaully only once
        return cg(f_mat, self.phi,)
    
    def _solve_with_pc(self):
        f_mat = self.m_mat @ self.m_mat.T  # this is always done, actaully only once
        return cg(f_mat, self.phi, M=self.f_mat_inv)
        # return cg(self.m_mat.T, cg(self.m_mat, self.phi, M=self.m_mat_inv.T)[0], M=self.m_mat_inv)
    
    def benchmark(self):
        tt = perf_counter()
        for _ in range(20):
            self._solve_without_pc()
        print('Without preconditioning: ', perf_counter() - tt)

        tt = perf_counter()
        for _ in range(20):
            self._solve_with_pc()
        print('With preconditioning:    ', perf_counter() - tt)



sol_F = PreconditionTestF(10, 3, 1e-2, 1e-5) 
sol_F.benchmark()


sol_M = PreconditionTestM(10, 3, 1e-2, 1e-5)
sol_M.benchmark()

