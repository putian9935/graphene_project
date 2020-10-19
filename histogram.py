import matplotlib.pyplot as plt 

def show_histogram(xs):
    plt.hist(xs, bins=int(len(xs)**.5/2), )
    plt.show()
    


if __name__ == '__main__':
    import numpy as np 

    show_histogram(np.random.randn(10000))
    