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



def t_matrix_same4all(Nt, N, hat_t, hat_U):
    ''' xi-independent part of matrix T '''
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

    
    return ret + 1*hat_U*sparse.eye(2*N*N*Nt, format='csc')

    
    # print(np.linalg.eig(ret.toarray())[0])
    # input()
    # import matplotlib.pyplot as plt 
    # plt.matshow(ret.toarray()) 
    # plt.show()


def m_matrix_same4all(Nt, N, hat_t, hat_U):
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
    return ret  +sparse.kron(buf,sparse.eye(2*N*N,format='csc')).tocsc()+ 1*hat_U*sparse.eye(2*N*N*Nt, format='csc')

    
    # print(np.linalg.eig(ret.toarray())[0])
    # input()
    # import matplotlib.pyplot as plt 
    # plt.matshow(ret.toarray()) 
    # plt.show()

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