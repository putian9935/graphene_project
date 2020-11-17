__doc__ = """ 
Block to determine auto-correlation time. 
"""

import numpy as np 

def coarsen(arr): 
    return np.array([(arr[i<<1]+arr[i<<1|1])/2. for i in range(len(arr)//2)]) 


def exponential_blocking(arr):
    ''' Should not use this one. Too few information '''
    if len(arr) == 1:
        return []
        
    return [arr.std()/len(arr)] + exponential_blocking(coarsen(arr))


def linear_blocking(arr):
    return \
        [
            np.array(
                [arr[k*n:k*n+n].mean() for k in range(len(arr)//n-1)]
            ).var()/(len(arr)//n-2)
            for n in range(1, min(len(arr)//20 + 1, 400))
        ]


def linear_blocking_sanity(arr):
    for n in range(1, min(len(arr)//20 + 1, 400)):
        print(n)
        buf = []
        for k in range(len(arr)//n-1):
            buf.append(arr[k*n:k*n+n].mean())
            
        buf = np.array(buf) 
        m = buf.mean()
        print(sum((x-m)*(x-m) for x in buf)/len(buf))
        print(buf.var())
        print(len(buf))
        print(len(arr)//n-2)
        print(arr[:-1].var()/(len(arr)-1))
        input()
    


if __name__ == '__main__':
    print(linear_blocking_sanity(np.array(list(range(100)))))
