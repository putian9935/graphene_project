__doc__ = """ 
Compute the reduced M matrix. 
This whole file should be merged into Trajectory class. 
======================
Jobs done: 

1. Compressed iteration: iteration over y may be compressed as simply
  for y in range(Nt * N * N):

2. Merge different chiral index of xi into a single function;

3. Merge xi-independent part into a single function, and a one-liner;

"""


from scipy import sparse 
import numpy as np

def tau_n2ind(tau, n1, n2, N):
    ''' Convert a coordinate pair (R, tau) into matrix index. '''
    return tau * N * N + n1 * N + n2 


def m_matrix_xi(Nt, N, hat_U, xi):
    ''' calc xi-dependent part of M matrix  '''
    return sparse.diags(xi * hat_U**.5,
        shape=(Nt*N*N*2, Nt*N*N*2,), 
        format='csc',
        dtype='float64')


def m_matrix_same4all(Nt, N, hat_t, hat_U):
    ''' calc xi-independent part of M matrix  '''
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


if __name__ == '__main__':
    # N = Nt = 2 is complicated enough for human eye
    
    m_matrix_xi(4,2,3,np.random.randn(32))
    

