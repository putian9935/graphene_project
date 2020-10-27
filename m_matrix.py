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
    s1, s2 = m_matrix_tau_free(Nt,N,hat_t,0)
    
    return m_matrix_tau_shift(Nt,N,0) \
        +sparse.kron(np.array([[0,0],[1,0]]), s1, format='csc') \
        +sparse.kron(np.array([[0,1],[0,0]]), s2, format='csc') \
        +hat_U/2*sparse.eye(2*N*N*Nt, format='csc')


def m_matrix_tau_free(Nt, N, hat_t, hat_U):
    ''' calc xi-independent and tau_free part of M matrix  '''
    from itertools import product
    row = []
    col = []
    val = []

    mat_size = N*N*Nt
    lat_size = N*N

    hat_U *= .5  # only half of \hat U appears in eq. (121) 
    
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
    Fourier transform of a matrix with size (Nt\times N\times N)^2
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
        

def ft2d(mat, Nt, N):
    r""" 
    Fourier transform of a matrix with size (Nt\times N\times N)^2
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
        

if __name__ == '__main__':

    from scipy.sparse.linalg import inv
    
    np.set_printoptions(linewidth=120, precision=3)
    # N = Nt = 2 is complicated enough for human eye
    N, Nt = 2, 2 
    
    for Nt, hat_t in [(10, 3)]:
        mat_size = N * N * Nt
        lat_size = N * N
        s1, s2 = m_matrix_tau_free(Nt,N,hat_t,0)
        e_rl, e_lr = ft2d_speedup(s1, Nt, N), ft2d_speedup(s2, Nt, N)

        inverted = inv(
                        m_matrix_tau_shift(Nt,N,0)
                        +sparse.kron(np.array([[0,0],[1,0]]), e_rl)
                        +sparse.kron(np.array([[0,1],[0,0]]), e_lr)
                    ).real.toarray()
        
        import matplotlib.pyplot as plt 
        
        plt.figure(figsize=(2.3*N,(2.3*N)))
        for k in range(lat_size):
            ax = plt.subplot(N, N, k+1)
            plt.imshow(
                np.log(
                    np.abs(
                        inverted[k:mat_size:lat_size,k:mat_size:lat_size]
                    )
                ) 
            )
            from energy_band import energy 
            ax.title.set_text(r'$k_1=%d$, $k_2=%d$, $E=%.3f$'%(k//N, k%N, energy(k//N/N, k%N/N)))

        plt.suptitle(r'$N=%d$, $N_t=%d$, $\hat t=%.2f$'%(N, Nt, hat_t))
        plt.tight_layout()
        plt.savefig(r'N=%dN_t=%dhat_t=%.2f.png'%(N, Nt, hat_t), dpi=400)
        exit()

        plt.scatter(range(Nt),
            np.log(
                np.abs(
                    inverted[1:mat_size:lat_size,mat_size+1::lat_size]
                )
            )[0,:], 
            label=str(Nt), 
        )

    plt.legend()
    plt.grid()
    plt.show()
    #plt.savefig('lr.pdf')
    """
    print(
        np.log(
            np.abs(
                inv(
                    m_matrix_tau_shift(Nt,N,0)
                    +sparse.kron(np.array([[0,0],[1,0]]), e_rl)
                    +sparse.kron(np.array([[0,1],[0,0]]), e_lr)
                )[:mat_size,:mat_size].real.toarray()
            )
        )[:,0]
    )
    print(
        np.log(
            np.abs(
                inv(
                    m_matrix_tau_shift(Nt,N,0)
                    +sparse.kron(np.array([[0,0],[1,0]]), e_rl)
                    +sparse.kron(np.array([[0,1],[0,0]]), e_lr)
                )[:mat_size,:mat_size].real.toarray()
            )
        )[0,:]
    )
    print(
        np.log(
            np.abs(
                inv(
                    m_matrix_tau_shift(Nt,N,0)
                    +sparse.kron(np.array([[0,0],[1,0]]), e_rl)
                    +sparse.kron(np.array([[0,1],[0,0]]), e_lr)
                )[:mat_size,mat_size:].real.toarray()
            )
        )[:,0]
    )
    """

        
