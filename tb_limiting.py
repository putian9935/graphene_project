__doc__ = """
Limiting process in tight-binging model of graphene. 
"""

from limiting_process import s_mat 
import numpy as np 
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit 

def quadratic_func(x, a, x0, k):
    return (x-x0)**2*a + k 


def different_Nt():
    ''' Generate quadratic.pdf ''' 

    betat = .09
    
    for N in range(10,51):  
        if N % 10==0:continue   
        Delta = s_mat(N, 0,0) 
        Delta_inv =  np.linalg.inv(Delta) 
        b = betat/N
        plt.scatter(np.linspace(0,1,N,endpoint=False),
            np.log(
                np.abs(
                    np.linalg.inv(Delta-b**2*Delta_inv)
                )
            )[0,:] ,
            # label=str(N),
            s=1,
            alpha=.3,
        )

    for N in range(10,51,10):     
        Delta = s_mat(N, 0,0) 
        Delta_inv =  np.linalg.inv(Delta) 
        b = betat/N
        plt.scatter(np.linspace(0,1,N,endpoint=False),
            np.log(
                np.abs(
                    np.linalg.inv(Delta-b**2*Delta_inv)
                )
            )[0,:] ,
            label=str(N),
            s=8
        )
    

    
    plt.grid()
    plt.legend()
    # plt.show()

    plt.savefig('quadratic.pdf', )
        
        
N = 50

betae = 5e-3
Delta = s_mat(N, 0., 0.) 
Delta_inv =  np.linalg.inv(Delta) 

arr = [] 

for betae in np.linspace(-5, 2, num=40):
    """
    plt.scatter(np.linspace(0,1,N),
        np.log(
            np.abs(
                np.linalg.inv(Delta-b**2*Delta_inv)
            )
        )[0,:] ,
        label=str(N)+' '+str(betat)  
    )
    """
    
    b = np.exp(betae) / N
    popt, _ = curve_fit(
        quadratic_func, 
        np.linspace(0,1,N), 
        np.log(
            np.abs(
                np.linalg.inv(Delta-b**2*Delta_inv)
            )
        )[0,:] 
    )
    arr.append(popt[0])
    # print(popt[0])
    print(np.exp(betae), np.abs((2*popt[0])**.5 - np.exp(betae)) / np.exp(betae)*100)


# popt, _= curve_fit(lambda _, a,b: a*_**2+b, np.exp(np.linspace(-5, 2, num=40)[:10]), arr[:10])
# print(popt)

plt.scatter(np.linspace(-5, 2, num=40),np.log(arr))


plt.grid()
# plt.legend()
plt.show()