__doc__ = """
A matter of interest. visualize the process of sampling. 
""" 

from hybrid_mc import Trajectory 
class Visualization(Trajectory):
    def __init__(self, Nt, N, hat_t, hat_U, max_epochs=400):
        super().__init__(Nt, N, hat_t, hat_U, max_epochs=max_epochs)
        self.evolve(time_step=.35)

import matplotlib.pyplot as plt 

Visualization(50,10, 1e-2, 1e-4, 400)

plt.figure(figsize=(6,6))
for state, _, prev_xi, xi in Trajectory.delta_ham[0]:
    if state == 'acc':
        plt.plot([prev_xi[0], xi[0]], [prev_xi[1], xi[1]], 'g', linewidth=.1, alpha=.9)
    else: 
        plt.plot([prev_xi[0], xi[0]], [prev_xi[1], xi[1]], 'r', linewidth=.1, alpha=.9)

plt.grid()
plt.show()

