__doc__ = """ 
Compute the reduced M matrix. 
This whole file should be merged into Trajectory class. 
======================
Jobs done: 

1. Compressed iteration: iteration over y may be compressed as simply
  for y in range(Nt * N * N):

2. Merge different chiral index of xi into a single function;

3. Merge xi-independent part into a single function, and a one-liner;

=====================
To-do
4. matrix same4all is wrong!
"""


from scipy import sparse 
import numpy as np

def tau_n2ind(tau, n1, n2, N):
    ''' Convert a coordinate pair (R, tau) into matrix index. '''
    return tau * N * N + n1 * N + n2 



def n2ind(n1, n2, N):
    ''' Convert a coordinate pair (R, ) into matrix index. '''
    return n1 * N + n2 


def m_matrix_xi(Nt, N, hat_U, xi):
    ''' calc xi-dependent part of M matrix  '''
    return sparse.diags(xi * hat_U**.5,
        shape=(Nt*N*N*2, Nt*N*N*2,), 
        format='csc',
        dtype='float64')


def m_matrix_same4all(Nt, N, hat_t, hat_U):
    ''' calc xi-independent part of M matrix  '''
    s1, s2 = m_matrix_tau_free(Nt,N,hat_t)
    return sparse.kron(np.array([[0,0],[1,0]]), s1, format='csc') \
        +sparse.kron(np.array([[0,1],[0,0]]), s2, format='csc') \
        +m_matrix_tau_shift(Nt,N,hat_U)


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
    



def m_matrix_shifted_same4all(Nt, N, hat_t, hat_U):
    from itertools import product
    row = []
    col = []
    val = []

    mat_size = N*N*Nt
    lat_size = N*N
    for i, j in product(range(N), range(N)):
        y = n2ind(i, j, N)
        pos = y * 2
        row.extend([pos]*3)
        col.extend([(pos+1)%(2*N*N), (n2ind(i,(j+1)%N,N)*2+1)%(2*N*N), (n2ind((i-1)%N,(j+1)%N,N)*2+1)%(2*N*N),])
        val.extend([-hat_t]*3)
        row.extend([(pos+1)%(2*N*N)]*3)
        col.extend([pos, n2ind(i,(j-1)%N,N)*2, n2ind((i+1)%N,(j-1)%N,N)*2,])
        val.extend([-hat_t]*3)

    ret = sparse.kron(np.eye(Nt),
        sparse.csc_matrix((val, (row, col)), shape=(N*N*2, N*N*2), dtype=float)
        ).tocsc()

    buf = -np.eye(Nt) 
    for i in range(0, Nt-1):
        buf[i, i+1] = 1
    buf[-1,0] = -1
    return ret  +sparse.kron(buf,sparse.eye(2*N*N,format='csc')).tocsc() # +1*hat_U*sparse.eye(2*N*N*Nt, format='csc')

    
    # print(np.linalg.eig(ret.toarray())[0])
    # input()
    # import matplotlib.pyplot as plt 
    # plt.matshow(ret.toarray()) 
    # plt.show()

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
        
if __name__ == '__main__':

    from scipy.sparse.linalg import inv
    
    np.set_printoptions(linewidth=120, precision=3)
    # N = Nt = 2 is complicated enough for human eye
    N, Nt = 5, 10 
    betat = .01
    u = 1e-3
    m = m_matrix_same4all(Nt,N,1e-2,1e-3).toarray();

    from scipy.sparse.linalg import cg 

    from time import perf_counter 

    tt= perf_counter()
    f  =m.T@m
    for _ in range(10000):
        cg(f, np.random.rand(N*N*Nt*2),) 
    print(perf_counter()-tt)
    # print(s1.toarray())
    # print(s2.toarray())
    # print(m_matrix_same4all(Nt, N, 0.05, 0.).toarray())