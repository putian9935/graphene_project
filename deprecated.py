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


def m_matrix_same4all_test_shift(Nt, N, hat_t, hat_U):
    
    from itertools import product 
    s1, s2 = m_matrix_tau_free(Nt,N,hat_t)
    tmp=    sparse.kron(np.array([[0,0],[1,0]]), s1, format='csc') \
        +sparse.kron(np.array([[0,1],[0,0]]), s2, format='csc')
    tmp=tmp.toarray()

    ret = np.zeros(shape=(Nt*N*N*2, Nt*N*N*2,))
    for tau, i, j in product(range(Nt), range(N), range(N)):
        for tau2, i2, j2 in product(range(Nt), range(N), range(N)):
            pos = tau*N*N+j*N+i
            pos2 = tau2*N*N + j2*N+i2
            ret[pos*2, pos2*2] = tmp[pos,pos2]
            ret[pos*2, pos2*2+1] = tmp[pos, pos2+N*N*Nt]
            ret[pos*2+1, pos2*2] = tmp[pos+N*N*Nt, pos2]
            ret[pos*2+1, pos2*2+1] = tmp[pos+N*N*Nt, pos2+N*N*Nt]
        
    return sparse.csc_matrix(ret)
    


def m_matrix_tau_free(Nt, N, hat_t):
    ''' calc xi-independent and tau_free part of M matrix  '''
    from itertools import product
    row = []
    col = []
    val = []

    mat_size = N*N*Nt
    lat_size = N*N

    # hat_U *= .5  # only half of \hat U appears in eq. (121) 
    
    for n1, n2 in product(range(N), range(N)):
        # periodic boundary condition, much easier to implement
        y = n2ind(n1,n2,N)
        row.extend([y, n2ind(n1,(n2+1)%N,N), n2ind(n1,(n2-1)%N,N), n2ind((n1+1)%N,n2, N), n2ind((n1-1)%N,n2, N)])
        col.extend([y]*5)
        val.extend([-hat_t, -hat_t/2, -hat_t/2, -hat_t/2, -hat_t/2])

    prop2s1 = sparse.kron(np.eye(Nt),
        sparse.csc_matrix((val, (row, col)), shape=(N*N, N*N), dtype=float)
        ).tocsc()
        
    row = []
    col = []
    val = []
    for n1, n2 in product(range(N), range(N)):
        # periodic boundary condition, much easier to implement
        y = n2ind(n1,n2,N)
        row.extend([n2ind(n1,(n2+1)%N,N), n2ind(n1,(n2-1)%N,N), n2ind((n1+1)%N,n2, N), n2ind((n1-1)%N,n2, N)])
        col.extend([y]*4)
        val.extend([hat_t/2, -hat_t/2, hat_t/2, -hat_t/2])

    prop2s2 = sparse.kron(np.eye(Nt),
        sparse.csc_matrix((val, (row, col)), shape=(N*N, N*N), dtype=float)
        ).tocsc()

    # lower left, upper right
    return prop2s1 + prop2s2, prop2s1 - prop2s2


def m_matrix_tau_shift(Nt, N, hat_U):
    ''' calc xi-independent part of M matrix  '''
    row = []
    col = []
    val = []

    mat_size = N*N*Nt

    # hat_U *= .5  
    # only half of \hat U appears in eq. (121), 
    # yet \hat U as whole appeared in eq. (248), 
    # however, only (248) is the correct matrix    
     
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
        
        # original version, maybe incorrect
        val.extend([1, -1, hat_U,1, -1, hat_U])  
  
    # not a single multiplication with matrix in QM is innocent, it's always a direct product
    return sparse.csc_matrix((val, (row, col)), shape=(Nt*N*N*2, Nt*N*N*2), dtype=float)


def m_matrix_tau_shift_smaller(Nt, N, hat_U):
    ''' calc xi-independent part of M matrix  '''
    row = []
    col = []
    val = []

    mat_size = N*N*Nt

    hat_U *= .5  # only half of \hat U appears in eq. (121) 
     
    # take care of the anti-periodic boundary condition 
    # pay attention to the function call val.extend
    for y in range(N*N):
        row.extend([y - N*N+mat_size, y, y])
        col.extend([y]*3)

        # original version, maybe incorrect
        val.extend([-1, -1, hat_U])
        
    # general case
    for y in range(N*N, mat_size):
        row.extend([y - N*N, y, y])
        col.extend([y]*3)
        
        # original version, maybe incorrect
        val.extend([1, -1, hat_U])  
        
        
    # not a single multiplication with matrix in QM is innocent, it's always a direct product
    return sparse.csc_matrix((val, (row, col)), shape=(Nt*N*N, Nt*N*N), dtype=float)


def ft2d_speedup(mat, Nt, N):
    r""" 
    This piece is wrong, DON'T use it unless you know what to expect!

    Fourier transform in real space of a matrix with size (Nt\times N\times N)^2
    """
    from itertools import product
    ret = np.zeros((N*N, N*N), dtype='complex128')
    lattice_size = N * N
    
    for k in  range(lattice_size):
        k1, k2 = k//N, k%N 
        k_prime = k
        kp1, kp2 = k_prime//N, k_prime%N
        
        ret[k_prime, k] = sum(
            mat[i1*N + i2, j1*N + j2] * np.exp(2.j*np.pi*(-kp1*i1-kp2*i2+k1*j1+k2*j2)/N) 
            for i1, i2, j1, j2 in product(range(N),range(N),range(N),range(N))
        ) 
    ret /= lattice_size
    # print(np.diag(ret).reshape(Nt,N, N))
    return sparse.kron(np.eye(Nt), ret).tocsc()
        

def ft2d_half(mat, Nt, N):
    r""" 
    Fourier transform in real space of a matrix with size (Nt\times N\times N)^2, thus "half"    
    """
    from itertools import product
    ret = np.zeros((Nt*N*N, Nt*N*N), dtype='complex128')
    lattice_size = N * N
    
    for tau, tau_prime, k, k_prime \
            in product(range(Nt), range(Nt), range(lattice_size), range(lattice_size)):
        k1, k2 = k//N, k%N 
        kp1, kp2 = k_prime//N, k_prime%N
        
        ret[k_prime+tau_prime*lattice_size, k+tau*lattice_size] = sum(
            mat[tau_prime*lattice_size + i1*N + i2, tau*lattice_size + j1*N + j2] 
            * np.exp(2.j*np.pi*(-kp1*i1-kp2*i2+k1*j1+k2*j2)/N) 
            for i1, i2, j1, j2 in product(range(N),range(N),range(N),range(N))
        ) 
    ret /= lattice_size
    # print(np.diag(ret).reshape(Nt,N, N))
    return ret
        

def ft2d(mat, Nt, N):
    r""" 
    Fourier transform in real space of a matrix with size Nt\times N\times N\times 2   
    """
    
    ret = np.zeros((2*N*N*Nt, 2*N*N*Nt), dtype='complex128') 
    mat_size = N*N*Nt 
    ret[:mat_size, :mat_size] = ft2d_half(mat[:mat_size, :mat_size], Nt, N)
    ret[mat_size:, :mat_size] = ft2d_half(mat[mat_size:, :mat_size], Nt, N)
    ret[:mat_size, mat_size:] = ft2d_half(mat[:mat_size, mat_size:], Nt, N)
    ret[mat_size:, mat_size:] = ft2d_half(mat[mat_size:, mat_size:], Nt, N)
    return ret


def m_matrix_same4all_old(Nt, N, hat_t, hat_U):
    ''' calc xi-independent part of M matrix, not in use  '''
    s1, s2 = m_matrix_tau_free(Nt,N,hat_t)
    return sparse.kron(np.array([[0,0],[1,0]]), s1, format='csc') \
        +sparse.kron(np.array([[0,1],[0,0]]), s2, format='csc') \
        +m_matrix_tau_shift(Nt,N,hat_U)


if __name__ == '__main__':
    from m_matrix import tau_n2ind 
    prop2sigma1_old(5,2,1e-2,0)
    prop2sigma2_old(5,2,1e-2,0)

