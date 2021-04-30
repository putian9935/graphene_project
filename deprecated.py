__doc__ = """
Stores deprecated code. Maybe used in sanity check. 
"""

from scipy import sparse 
import numpy as np


def prop2I_old(Nt, N, hat_t, hat_U):
    ''' Return terms proportional to I (in chiral space). Uses Kronecker product for simplicity, leading to bad performance''' 
    row = []
    col = []
    val = []


    # take care of the anti-periodic boundary condition 
    # pay attention to the function call val.extend
    for n1 in range(N):
        for n2 in range(N): 
            y = tau_n2ind(0, n1, n2, N)
            row.extend([y - N*N+N*N*Nt, y, y])
            col.extend([y]*3)
            val.extend([-1, -1, hat_U,])

    # general case
    for tau in range(1, Nt):
        for n1 in range(N):
            for n2 in range(N): 
                y = tau_n2ind(tau, n1, n2, N)
                row.extend([y - N*N, y, y])
                col.extend([y]*3)
                val.extend([1, -1, hat_U, ])
    
    # for debug use only
    # print(sparse.coo_matrix((val, (row, col))).toarray())

    # not a single multiplication with matrix in QM is innocent, it's always a direct product
    return sparse.kron(  
            np.array([[1,0],[0,1]]),
            sparse.coo_matrix((val, (row, col)), shape=(Nt*N*N, Nt*N*N), dtype=float),
           )


def prop2sigma1_old(Nt, N, hat_t, hat_U):
    ''' Return terms proportional to sigma_1 (in chiral space). ''' 
    row = []
    col = []
    val = []

    mat_size = Nt * N * N

    for tau in range(Nt):
        for n1 in range(N):
            for n2 in range(N): 
                y = tau_n2ind(tau, n1, n2, N)
                
                # periodic boundary condition, much easier to implement
                row.extend([y, (y+N)%mat_size, (y-N)%mat_size, (y+1)%mat_size, (y-1)%mat_size])
                col.extend([y]*5)
                val.extend([-hat_t, -hat_t/2, -hat_t/2, -hat_t/2, -hat_t/2])
    
    # for debug use only
    print(sparse.coo_matrix((val, (row, col))).toarray())

    return sparse.kron(  
            np.array([[0,1],[1,0]]),
            sparse.coo_matrix((val, (row, col)), shape=(mat_size, mat_size), dtype=float).tocsr(),
           )


def prop2sigma2_old(Nt, N, hat_t, hat_U):
    ''' Return terms proportional to sigma_2 (in chiral space). ''' 
    row = []
    col = []
    val = []

    mat_size = Nt * N * N

    for tau in range(Nt):
        for n1 in range(N):
            for n2 in range(N): 
                y = tau_n2ind(tau, n1, n2, N)
                
                # periodic boundary condition, much easier to implement
                row.extend([(y+N)%mat_size, (y-N)%mat_size, (y+1)%mat_size, (y-1)%mat_size])
                col.extend([y]*4)
                val.extend([-hat_t/2, hat_t/2, -hat_t/2, hat_t/2])

    # for debug use only
    print(sparse.coo_matrix((val, (row, col))).toarray())

    return sparse.kron(  
            np.array([[0,-1],[1,0]]),
            sparse.csr_matrix((val, (row, col)), shape=(mat_size, mat_size), dtype=float),
           )


def prop2pL(Nt, N, xi_L, hat_U):
    ''' Return terms proportional to P_L (in chiral space). ''' 
    row = []
    col = []
    val = []

    mat_size = Nt * N * N
    sqrtU = hat_U ** .5

    for tau in range(Nt):
        for n1 in range(N):
            for n2 in range(N): 
                y = tau_n2ind(tau, n1, n2, N)
                
                # periodic boundary condition, much easier to implement
                row.extend([y])
                col.extend([y])
                val.extend([sqrtU*xi_L[y]])  # might later leave factor sqrtU into the definition of P_L
    
    # for debug use only
    # print(sparse.coo_matrix((val, (row, col))).toarray())

    return sparse.coo_matrix((val, (row, col)), shape=(mat_size*2, mat_size*2)).tocsr()


def prop2pR(Nt, N, xi_R, hat_U):
    ''' Return terms proportional to P_L (in chiral space). ''' 
    row = []
    col = []
    val = []

    mat_size = Nt * N * N
    sqrtU = hat_U ** .5

    for tau in range(Nt):
        for n1 in range(N):
            for n2 in range(N): 
                y = tau_n2ind(tau, n1, n2, N)
                
                # periodic boundary condition, much easier to implement
                row.extend([mat_size+y])
                col.extend([mat_size+y])
                val.extend([sqrtU*xi_R[y]])  # might later leave factor sqrtU into the definition of P_L
    
    # for debug use only
    # print(sparse.coo_matrix((val, (row, col))).toarray())

    return sparse.coo_matrix((val, (row, col)), shape=(mat_size*2, mat_size*2)).tocsr()


def prop2I(Nt, N, hat_t, hat_U):
    ''' Return terms proportional to I (in chiral space). ''' 
    row = []
    col = []
    val = []

    mat_size = N*N*Nt
    # take care of the anti-periodic boundary condition 
    # pay attention to the function call val.extend
    for y in range(N*N):
        row.extend([y - N*N+mat_size, y, y, y - N*N+2*mat_size, y+mat_size, y+mat_size])
        col.extend([y]*3+[y+mat_size]*3)
        val.extend([-1, -1, hat_U,-1, -1, hat_U])
        
    # general case
    for y in range(N*N, mat_size):
        row.extend([y - N*N, y, y, y - N*N+mat_size, y+mat_size, y+mat_size])
        col.extend([y]*3+[y+mat_size]*3)
        val.extend([1, -1, hat_U,1, -1, hat_U])
        
    # not a single multiplication with matrix in QM is innocent, it's always a direct product
    return sparse.csr_matrix((val, (row, col)), shape=(Nt*N*N*2, Nt*N*N*2), dtype=float)


def prop2sigma1(Nt, N, hat_t, hat_U):
    ''' Return terms proportional to sigma_1 (in chiral space). ''' 
    row = []
    col = []
    val = []

    mat_size = Nt * N * N

    for y in range(mat_size):
        # periodic boundary condition, much easier to implement
        row.extend([y+mat_size, (y+N)%mat_size+mat_size, (y-N)%mat_size+mat_size, (y+1)%mat_size+mat_size, (y-1)%mat_size+mat_size])
        col.extend([y]*5)
        val.extend([-hat_t, -hat_t/2, -hat_t/2, -hat_t/2, -hat_t/2])

        row.extend([y, (y+N)%mat_size, (y-N)%mat_size, (y+1)%mat_size, (y-1)%mat_size])
        col.extend([y+mat_size]*5)
        val.extend([-hat_t, -hat_t/2, -hat_t/2, -hat_t/2, -hat_t/2])

    return sparse.csr_matrix((val, (row, col)), shape=(mat_size*2, mat_size*2), dtype=float)


def prop2sigma2(Nt, N, hat_t, hat_U):
    ''' Return terms proportional to sigma_2 (in chiral space). ''' 
    row = []
    col = []
    val = []

    mat_size = Nt * N * N

    for y in range(mat_size):
        # periodic boundary condition, much easier to implement
        row.extend([(y+N)%mat_size, (y-N)%mat_size, (y+1)%mat_size, (y-1)%mat_size])
        col.extend([y+mat_size]*4)
        val.extend([hat_t/2, -hat_t/2, hat_t/2, -hat_t/2])

        row.extend([(y+N)%mat_size+mat_size, (y-N)%mat_size+mat_size, (y+1)%mat_size+mat_size, (y-1)%mat_size+mat_size])
        col.extend([y]*4)
        val.extend([-hat_t/2, hat_t/2, -hat_t/2, hat_t/2])
  
    return sparse.csr_matrix((val, (row, col)), shape=(mat_size*2, mat_size*2), dtype='float64')


def m_matrix_same4all_old(Nt, N, hat_t, hat_U):
    ''' calc xi-independent part of M matrix  '''
    return (prop2I(Nt, N, hat_t, hat_U) + prop2sigma1(Nt, N, hat_t, hat_U)  + prop2sigma2(Nt, N, hat_t, hat_U)).tocsr()


def m_matrix_xi_old(Nt, N, hat_U, xi):
    ''' calc xi-dependent part of M matrix  '''
    row = []
    col = []
    val = []

    mat_size = Nt * N * N
    sqrtU = hat_U ** .5

    for y in range(mat_size):   
        # periodic boundary condition, much easier to implement
        row.extend([y, mat_size+y])
        col.extend([y, mat_size+y])
        val.extend([sqrtU*xi[y],sqrtU*xi[mat_size+y]])  # treat LR component together

    # for debug use only
    # print(sparse.coo_matrix((val, (row, col))).toarray())

    return sparse.csc_matrix((val, (row, col)), shape=(mat_size*2, mat_size*2), dtype=float)



def two_point_aa(self, burnin=None):
    """ 
    Calc observables via eq. (246)

    As Feng mentioned, NEVER use matrix inverse unless you have a good reason. Following code uses cached solver to speedup solution. 
    """
    
    if not burnin: 
        burnin = self.traj.max_epochs // 2 

    print('Doing statistics...')
    ret = np.zeros((self.Nt - 1,self.N, self.N))  # a function of R and tau

    buf = np.zeros(self.N*self.N*self.Nt*2)
    for xi in tqdm(self.traj.xis[burnin:]):
        inv_solver = splu(self.traj.m_mat_indep + m_matrix_xi(self.Nt, self.N, self.hat_U, xi))
        for tau in range(self.Nt-1):      
            ind0 = tau_n2ind(tau, 0, 0, self.N)  # ind0 is irrelevant to n1, n2, thus to indr as well
            buf[ind0] = 1
            sol_buf = inv_solver.solve(buf) # thus we can solve for buf once for a fixed tau
            for n1 in range(self.N):
                for n2 in range(self.N):
                    # this line is effectively an inner product 
                    ret[tau, n1, n2,] += sol_buf[tau_n2ind(tau+1, n1, n2, self.N)] ** 2
            buf[ind0] = 0
    ret /= len(self.traj.xis) 
    return ret


def show_plot(results):
    import matplotlib.pyplot as plt
    for i, res in enumerate(results):
        plt.matshow(res) 
        plt.savefig('%d.png' % i, )


def m_matrix_same4all_wrong(Nt, N, hat_t, hat_U):
    ''' calc xi-independent part of M matrix  '''
    row = []
    col = []
    val = []

    mat_size = N*N*Nt

    hat_U *= .5  # only half of \hat U appears in eq. (121) 
     
    # take care of the anti-periodic boundary condition 
    # pay attention to the function call val.extend
    for y in range(N*N):
        row.extend([y - N*N+mat_size, y, y, y - N*N+2*mat_size, y+mat_size, y+mat_size])
        col.extend([y]*3+[y+mat_size]*3)
        val.extend([-1, -1, hat_U,-1, -1, hat_U])
        
    # general case
    for y in range(N*N, mat_size):
        row.extend([y - N*N, y, y, y - N*N+mat_size, y+mat_size, y+mat_size])
        col.extend([y]*3+[y+mat_size]*3)
        val.extend([1, -1, hat_U,1, -1, hat_U])

    for y in range(mat_size):
        # periodic boundary condition, much easier to implement
        row.extend([y+mat_size, (y+N)%mat_size+mat_size, (y-N)%mat_size+mat_size, (y+1)%mat_size+mat_size, (y-1)%mat_size+mat_size])
        col.extend([y]*5)
        val.extend([-hat_t, -hat_t/2, -hat_t/2, -hat_t/2, -hat_t/2])

        row.extend([y, (y+N)%mat_size, (y-N)%mat_size, (y+1)%mat_size, (y-1)%mat_size])
        col.extend([y+mat_size]*5)
        val.extend([-hat_t, -hat_t/2, -hat_t/2, -hat_t/2, -hat_t/2])

        row.extend([(y+N)%mat_size, (y-N)%mat_size, (y+1)%mat_size, (y-1)%mat_size])
        col.extend([y+mat_size]*4)
        val.extend([hat_t/2, -hat_t/2, hat_t/2, -hat_t/2])

        row.extend([(y+N)%mat_size+mat_size, (y-N)%mat_size+mat_size, (y+1)%mat_size+mat_size, (y-1)%mat_size+mat_size])
        col.extend([y]*4)
        val.extend([-hat_t/2, hat_t/2, -hat_t/2, hat_t/2])

        
    # not a single multiplication with matrix in QM is innocent, it's always a direct product
    return sparse.csc_matrix((val, (row, col)), shape=(Nt*N*N*2, Nt*N*N*2), dtype=float)



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
        plt.show(self.m_mat)
        self.m_mat_inv = inv(self.m_mat_indep)
        

    
    def _solve_with_pc(self):
        return cg(self.m_mat, self.phi, M=self.m_mat_inv)
        # return cg(self.m_mat, self.phi, M=self.m_mat_indep.T)
    
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



if __name__ == '__main__':
    from m_matrix import tau_n2ind 
    prop2sigma1_old(5,2,1e-2,0)
    prop2sigma2_old(5,2,1e-2,0)

