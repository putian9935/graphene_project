__doc__ = """
Show if the Fourier transform in imaginary time space is indeed unitary. 
""" 

import numpy as np 
from limiting_process import s_mat 

def q_mat(Nt):
    q = np.zeros((Nt,Nt), dtype='complex128') 

    for k in range(Nt):
        for l in range(Nt):
            q[k,l] = np.exp(1j*(2*k+1)*np.pi*l/Nt) * Nt**-.5

    return q 


np.printoptions(precision=3)

q = q_mat(10)
s = s_mat(10, 0., 0.)
e = 0.5 ** 2

diag = q@s.astype('complex128')@np.conjugate(q.T)

def inv_diag(mat):
    return np.diag(1./np.diag(mat))

Nt=10
ss = np.ones((Nt, Nt))
for i in range(Nt):
    ss[i,i] -= 2 / e
    if i < Nt-1:
        ss[i,i+1] += 2 / e 
    for j in range(i):
        ss[i,j] = -1
ss[-1][0] = -1-2 / e 

diag_inv = inv_diag(diag)


import matplotlib.pyplot as plt

plt.matshow(np.linalg.inv(ss).real)

plt.show()
print(np.linalg.inv(np.conjugate(q.T)@(diag-e*diag_inv)@q).real)
plt.matshow(np.linalg.inv(np.conjugate(q.T)@(diag-e*diag_inv)@q).real)

plt.show()

plt.matshow((np.conjugate(q.T)@(inv_diag(diag-e*diag_inv))@q).real)
plt.show()

# plt.matshow((q@np.conjugate(q.T)).real)
# plt.show()