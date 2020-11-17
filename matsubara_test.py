__doc__ = '''
Implement Fourier transform for fermion matsubara frequency.

Note there's a missing factor 1 in traditional FT. 

Since performance isn't concerning, an explicit construction of FT matrix is performed. 
'''

import numpy as np 

def s_mat(Nt, a=1):
    ''' Matrix S defined in Eq. (134) '''
    ret = np.eye(Nt) * (-a) 
    for i in range(Nt-1):
        ret[i, i+1] = 1 
    ret[-1, 0] = -1  # anti-periodic BC 

    return ret

def inspect_time(mat, Nt, N, at=0):
    ''' Return mat at a certain time. Beware of chiral index ''' 
    
    lat_size = N*N 
    mat_size = N*N*Nt
    ret = np.zeros((2*lat_size))  # always make a copy
    beg, end = at*lat_size, (at+1)*lat_size
    ret[:lat_size,:lat_size] = mat[beg:end, beg:end]
    ret[:lat_size,lat_size:] = mat[beg:end, mat_size+beg:mat_size+end]
    ret[lat_size:,:lat_size] = mat[mat_size+beg:mat_size+end, beg:end]
    ret[:lat_size,:lat_size] = mat[mat_size+beg:mat_size+end, mat_size+beg:mat_size+end]
    return ret 


def inspect_pos(mat, Nt, N, at=(0,0)):
    ''' Return mat at a certain pos. Beware of chiral index ''' 
        
    beg= at[0]*N+at[1]
    ret = mat[beg::N*N, beg::N*N].copy()  # always make a copy
    return ret 
    


def ft2d_matsubara_base(mat, Nt):
    ''' 
    Fourier transform with Matsubara frequency, defined in Eq.s (140), (141) 
    
    Matrix size is Nt, thus the name "base"
    '''
    u = np.zeros((Nt, Nt), dtype='complex128')  # explicitly construct matrix U, as defined in eq. (140)
    for i in range(Nt):
        for j in range(Nt):
            u[i, j] = np.exp(1j*(2*j+1)*np.pi*i/Nt)  
    u /= Nt ** .5
    
    
    return np.conjugate(u.T) @ mat @ u 


def ft2d_matsubara(mat, Nt, N):
    ''' 
    Fourier transform with Matsubara frequency, defined in Eq.s (140), (141) 
    
    Matrix size is Nt * N * N * 2, thus the name "base"
    '''
    u = np.zeros((Nt, Nt), dtype='complex128')  # explicitly construct matrix U, as defined in eq. (140)
    for i in range(Nt):
        for j in range(Nt):
            u[i, j] = np.exp(1j*(2*j+1)*np.pi*i/Nt)  
    u /= Nt ** .5

    lat_size = N*N 
    for t in range(2*Nt):
        mat[t*lat_size:(t+1)*lat_size, t*lat_size:(t+1)*lat_size] = np.conjugate(u.T) @ mat[t*lat_size:(t+1)*lat_size, t*lat_size:(t+1)*lat_size] @ u 
    return mat 

if __name__ == '__main__':
    np.set_printoptions(linewidth=120, precision=3)

    Nt, N = 3, 3
    '''
    # sanity check, compare if results match 
    print(np.diag(ft2d_matsubara(s_mat(Nt), Nt)))  # via direct transform 
    print(np.exp(1j*np.pi*(2*np.arange(Nt)+1)/Nt)-1)  # analytic solution, eq. (143) 
    '''

    from m_matrix import m_matrix_same4all, ft2d_speedup, ft2d
    
    m_mat = m_matrix_same4all(Nt, N, .1, 0.)  # construct a simple matrix 
    # print(m_mat.toarray()) 
    
    # k_space = ft2d(m_mat, Nt, N) 
    print(ft2d_matsubara(m_mat, Nt, N))
    input()
    # print(k_space.shape)
    # print(inspect_pos(ft2d_matsubara(k_space,Nt,N), Nt, N, (1,0)))




