import matplotlib.pyplot as plt 

def show_histogram(xs, title,std):
    plt.figure()
    plt.hist(xs, bins=int(len(xs)**.5/2), )
    plt.title(r'$\Delta t$=%s, $\sigma$=%.3e'%(title,std))
    plt.savefig('Hamiltonian_Scale_%s.png'%title)
    

