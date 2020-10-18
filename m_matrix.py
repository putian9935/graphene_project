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
            sparse.coo_matrix((val, (row, col)), shape=(Nt*N*N, Nt*N*N)),
           )


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
            sparse.coo_matrix((val, (row, col)), shape=(mat_size, mat_size)),
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
            sparse.coo_matrix((val, (row, col)), shape=(mat_size, mat_size)),
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

    return sparse.kron(  
            np.array([[1,0],[0,0]]),
            sparse.coo_matrix((val, (row, col)), shape=(mat_size, mat_size)),
           )


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
                row.extend([y])
                col.extend([y])
                val.extend([sqrtU*xi_R[y]])  # might later leave factor sqrtU into the definition of P_L
    
    # for debug use only
    # print(sparse.coo_matrix((val, (row, col))).toarray())

    return sparse.kron(  
            np.array([[0,0],[0,1]]),
            sparse.coo_matrix((val, (row, col)), shape=(mat_size, mat_size)),
           )


def m_matrix(Nt, N, hat_t, hat_U, xi_L, xi_R):
    return prop2I(Nt, N, hat_t, hat_U) + prop2sigma1(Nt, N, hat_t, hat_U)  + prop2sigma2(Nt, N, hat_t, hat_U) + prop2pL(Nt, N, xi_L, hat_U) + prop2pR(Nt, N, xi_R, hat_U)
    

if __name__ == '__main__':
    # N = Nt = 2 is complicated enough for human eye
    print(prop2I(2,2,2,3).toarray())
    print(prop2sigma1(2,2,2,3).toarray())
    print(prop2sigma2(2,2,2,3).toarray())
    

