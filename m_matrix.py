__doc__ = """ 
Compute the reduced M matrix. 
This whole file should be merged into Trajectory class.
"""


from scipy import sparse 
import numpy as np

def tau_n2ind(tau, n1, n2, N):
    ''' Convert a coordinate pair (R, tau) into matrix index. '''
    return tau * N * N + n1 * N + n2 


# In the following code,
# iteration over y may be compressed as simply
# for y in range(Nt * N * N):
# I'll save this possible optimization to next round
def prop2I(Nt, N, hat_t, hat_U):
    ''' Return terms proportional to I (in chiral space). ''' 
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

def prop2I_new(Nt, N, hat_t, hat_U):
    ''' Return terms proportional to I (in chiral space). ''' 
    row = []
    col = []
    val = []

    mat_size = N*N*Nt
    # take care of the anti-periodic boundary condition 
    # pay attention to the function call val.extend
    for n1 in range(N):
        for n2 in range(N): 
            y = tau_n2ind(0, n1, n2, N)
            row.extend([y - N*N+mat_size, y, y, y - N*N+2*mat_size, y+mat_size, y+mat_size])
            col.extend([y]*3+[y+mat_size]*3)
            val.extend([-1, -1, hat_U,-1, -1, hat_U])

    # general case
    for tau in range(1, Nt):
        for n1 in range(N):
            for n2 in range(N): 
                y = tau_n2ind(tau, n1, n2, N)
                row.extend([y - N*N, y, y, y - N*N+mat_size, y+mat_size, y+mat_size])
                col.extend([y]*3+[y+mat_size]*3)
                val.extend([-1, -1, hat_U,-1, -1, hat_U])
    
    # for debug use only
    # print(sparse.coo_matrix((val, (row, col))).toarray())

    # not a single multiplication with matrix in QM is innocent, it's always a direct product
    return sparse.coo_matrix((val, (row, col)), shape=(Nt*N*N*2, Nt*N*N*2), dtype=float).tocsr()


# Function prop2sigma1 and function prop2sigma2 may be merged into a single function.
# I'll leave this work to next round. 
def prop2sigma1(Nt, N, hat_t, hat_U):
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
    # print(sparse.coo_matrix((val, (row, col))).toarray())

    return sparse.kron(  
            np.array([[0,1],[1,0]]),
            sparse.coo_matrix((val, (row, col)), shape=(mat_size, mat_size), dtype=float).tocsr(),
           )


def prop2sigma2(Nt, N, hat_t, hat_U):
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
    # print(sparse.coo_matrix((val, (row, col))).toarray())

    return sparse.kron(  
            np.array([[0,-1],[1,0]]),
            sparse.csr_matrix((val, (row, col)), shape=(mat_size, mat_size), dtype=float),
           )


"""
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
"""


def m_matrix_xi(Nt, N, hat_U, xi):
    ''' calc xi-dependent part of M matrix  '''
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
                row.extend([y, mat_size+y])
                col.extend([y, mat_size+y])
                val.extend([sqrtU*xi[y],sqrtU*xi[mat_size+y]])  # treat LR component together
    
    # for debug use only
    # print(sparse.coo_matrix((val, (row, col))).toarray())

    return sparse.csr_matrix((val, (row, col)), shape=(mat_size*2, mat_size*2), dtype=float)



def m_matrix_same4all(Nt, N, hat_t, hat_U):
    ''' calc xi-independent part of M matrix  '''
    return (prop2I(Nt, N, hat_t, hat_U) + prop2sigma1(Nt, N, hat_t, hat_U)  + prop2sigma2(Nt, N, hat_t, hat_U)).tocsr()



if __name__ == '__main__':
    # N = Nt = 2 is complicated enough for human eye
    import cProfile, pstats, io
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(4000):
        prop2I_new(2,2,2,3)
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
        
    

