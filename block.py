__doc__ = """ 
Block to determine auto-correlation time. 
"""

import numpy as np 

def coarsen(arr): 
    return np.array([(arr[i<<1]+arr[i<<1|1])/2. for i in range(len(arr)//2)]) 


def exponential_blocking(arr):
    if len(arr) == 1:
        return []
        
    return [arr.std()/len(arr)] + exponential_blocking(coarsen(arr))


def linear_blocking(arr):
    return \
        [np.array(
                [arr[k*n:k*n+n].mean() for k in range(len(arr)//n-1)]
            ).var()/(len(arr)//n-2)
            for n in range(1, len(arr)//50)
        ]



if __name__ == '__main__':
    print(linear_blocking(np.array(list(range(1000)))))
