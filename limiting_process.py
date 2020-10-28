__doc__ = """
Investigate the limiting process of the inverse of eq. (134) to different N's. 
"""

import numpy as np 

def s_mat(n, e, hat_t):
    ret = np.zeros((n,n), dtype='f8')

    for i in range(n-1):
        ret[i][i] = -(1. - e*hat_t)
        ret[i, i+1] = 1.
    ret[n-1][0] = -1. 
    ret[n-1][-1] = -(1. - e*hat_t)

    return ret 

if __name__ == '__main__':
    betat = 1. 

    for Nt in range(10, 51,10):
        hat_t, e = betat/Nt, 0.3
        
        import matplotlib.pyplot as plt 
        
        plt.scatter(
            np.linspace(0,1,Nt,endpoint=False),
            np.log(np.abs(
                np.linalg.inv(s_mat(Nt, 0.3, hat_t)))
            )[0,:], 
            label='$'+str(Nt)+'$',
            s=3,
        )
        """
        print(
            np.log(
                np.abs(
                    np.linalg.det(
                        s_mat(Nt, e, hat_t)
                    )
                )
            ),
            np.log(1+(1.-hat_t*e)**Nt)
        )

        print(
            (lambda _: _[0,0]-_[0,-1])(
                np.log(
                    np.abs(
                        np.linalg.inv(s_mat(Nt, e, hat_t))
                    )
                )
            )
        )
        
        plt.colorbar(
            plt.imshow(
                np.log(np.abs(
                    np.linalg.inv(s_mat(Nt, 0.3, hat_t)))
                )
            )
        )
        """

    plt.grid()
    plt.legend()
    # plt.show()
    plt.savefig('linear.pdf')
