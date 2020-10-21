__doc__ = """
A matter of interest. visualize the process of sampling. 
""" 

from hybrid_mc import Trajectory 
class Visualization(Trajectory):
    def __init__(self, Nt, N, hat_t, hat_U, max_epochs=400):
        super().__init__(Nt, N, hat_t, hat_U, max_epochs=max_epochs)
        self.evolve(time_step=.8)

import matplotlib.pyplot as plt 

Visualization(8,2, 1e-2, 1e-4, 400)


plt.figure(0, figsize=(6,6))
for state, _, prev_xi, xi in Trajectory.delta_ham[0]:
    if state == 'acc':
        plt.plot([prev_xi[0], xi[0]], [prev_xi[1], xi[1]], 'g', linewidth=.25)
        plt.scatter([xi[0],], [xi[1],], c='g',alpha=.5)
    else: 
        plt.plot([prev_xi[0], xi[0]], [prev_xi[1], xi[1]], 'r', linewidth=.25)
        plt.scatter([xi[0],], [xi[1],], c='r',alpha=.5)

plt.grid()
plt.savefig('sampling.png', dpi=450)

