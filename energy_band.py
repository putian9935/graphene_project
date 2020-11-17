import numpy as np 

__doc__ = r"""
Solve the energy band as shown in fig. 3

Throughout this project, I shall assume that k is related to its two components k1, k2 via
    k = k1/N*b_1 + k2/N*b_2, 
where N is the lattice size and b_i are the basis vectors in the reciprocal space. 

Especially in this code, I implicitly assume N=1. 

Also, note that a_i\cdot b_i = 2\pi, and that the result pattern would not be exactly regular since b1 and b2 should form an angle of 60 degrees in REAL reciprocal space while in the plotting regime it is shown to be orthogonal to each other. 
"""

def f1(k1, k2):
    ''' f1 defined in eq. (51) '''
    return 1 + np.cos(2*np.pi*k1) + np.cos(2*np.pi*k2) 


def f2(k1, k2):
    ''' f2 defined in eq. (51) '''
    return np.sin(2*np.pi*k1) + np.sin(2*np.pi*k2) 


def energy(k1, k2):
    ''' only E_+ in eq. (53) will be returned; t is taken to be 1. '''
    return (f1(k1, k2)**2 + f2(k1,k2)**2)**.5 


def energy_grid(N):
    ''' Return unperturbed energy in reciprocal space. '''
    return energy(
            *np.meshgrid(
                np.linspace(0,1,N,endpoint=False),  # k1
                np.linspace(0,1,N,endpoint=False)   # k2 
            )
        )


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    K1 = np.linspace(-1,1,)
    K2 = np.linspace(-1,1,) 

    K1, K2 = np.meshgrid(K1, K2)

    E = energy(K1, K2) 
    from itertools import product
    N=7
    np.set_printoptions(precision=4)
    print(energy(*np.meshgrid(np.linspace(0,1,N,endpoint=False), np.linspace(0,1,N,endpoint=False))))
    exit()
    print(f1(0,1/3), f2(0,1/3))
    print(f1(3/5,1/5), f2(3/5,1/5))
    print(f1(2/3,0), f2(2/3,0))
    print(f1(1/3,1/3), f2(1/3,1/3))
    print(f1(0.,0.2), f2(0.,0.2))
    ax.plot_surface(K1,K2,E)
    # plt.show()
