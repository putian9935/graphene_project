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
    ret = np.zeros((2*lat_size, 2*lat_size), dtype='complex128')  # always make a copy
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
    

def inspect_pos_time(mat, Nt, N, at=(0,0,0)):
    ''' 
    Return mat at a certain pos, time. Beware of chiral index 
    pos index: at[0:1]
    time index: at[2]
    ''' 
    beg= at[0]*N+at[1]
    # .copy()  # always make a copy
    ret = np.zeros((2, 2), dtype='complex128')
    
    ret[0,0] = mat[beg::N*N, beg::N*N][at[2], at[2]]
    ret[0,1] = mat[beg::N*N, beg::N*N][at[2], Nt+at[2]]
    ret[1,0] = mat[beg::N*N, beg::N*N][Nt+at[2], at[2]]
    ret[1,1] = mat[beg::N*N, beg::N*N][Nt+at[2], Nt+at[2]]
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
    mat_size = N*N*Nt
    ret = np.zeros((2*N*N*Nt, 2*N*N*Nt), dtype='complex128')
    from itertools import product
    for p1, p2 in product(range(N*N), range(N*N)):
        ret[p1:mat_size:lat_size,p2:mat_size:lat_size] = np.conjugate(u.T) @ mat[p1:mat_size:lat_size,p2:mat_size:lat_size] @ u 
        
        ret[p1+mat_size::lat_size,p2:mat_size:lat_size] = np.conjugate(u.T) @ mat[p1+mat_size::lat_size,p2:mat_size:lat_size] @ u 
        
        ret[p1:mat_size:lat_size,p2+mat_size::lat_size] = np.conjugate(u.T) @ mat[p1:mat_size:lat_size,p2+mat_size::lat_size] @ u 
        
        ret[p1+mat_size::lat_size,p2+mat_size::lat_size] = np.conjugate(u.T) @ mat[p1+mat_size::lat_size,p2+mat_size::lat_size] @ u 

        
    return ret 


def matshow(mat):
    plt.matshow(mat.real) 
    plt.show()

    
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



    import matplotlib.pyplot as plt 
    plt.matshow(inspect_pos(m_mat, Nt, N).real.toarray())
    plt.show()
    
    # plt.matshow(inspect_pos(ft2d(m_mat, Nt, N), Nt, N).real)
    # plt.show()
    # plt.matshow(inspect_pos(ft2d_matsubara(ft2d(m_mat, Nt, N), Nt, N), Nt, N).real)
    # plt.show()
    
    print(inspect_pos_time(ft2d_matsubara(ft2d(m_mat, Nt, N), Nt, N), Nt, N))  # right 
    matshow(inspect_time(ft2d_matsubara(ft2d(m_mat, Nt, N), Nt, N), Nt, N))
    input('1')
    
    print(inspect_pos_time(ft2d(ft2d_matsubara(m_mat, Nt, N), Nt, N), Nt, N))  # also correct
    matshow(inspect_time(ft2d(ft2d_matsubara(m_mat, Nt, N), Nt, N), Nt, N))
    input('2')
    print(inspect_pos(ft2d_matsubara(ft2d(m_mat, Nt, N), Nt, N), Nt, N))
    input('3')
    print(inspect_pos_time(ft2d_matsubara(m_mat, Nt, N), Nt, N))
    input()
    final = ft2d(ft2d_matsubara(m_mat, Nt, N), Nt, N)
    print(inspect_pos_time(final, Nt, N))
    input()
    
    input()
    # print(k_space.shape)
    # print(inspect_pos(ft2d_matsubara(k_space,Nt,N), Nt, N, (1,0)))




