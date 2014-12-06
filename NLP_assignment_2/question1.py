"""
Question1 code
"""
import numpy as np
import math
import scipy.linalg


def plot_data(x, y, z=None):
    """
    Simple demo of a scatter plot.
    """
    import matplotlib.pyplot as plt
    plt.title("Data visualization")

    plt.scatter(x, y, c='red',  alpha=0.5)
    if z is not None:
        plt.scatter(x, z, alpha=0.5, c='blue', marker='*')
    plt.show()
    

def generateDataset(N, f, sigma):
    """
    :param N: number of examples
    :param f: method
    :param sigma: sigma of nomral distribution
    :return: dataset
    """
    x_array = np.linspace(0.0, 1.0, num=N)
    norm_array = np.random.normal(scale=sigma, size=N)
    real_y = f(x_array)
    t_array = f(x_array) + norm_array
    #plot_data(x_array, t_array, real_y)
    return x_array, t_array, real_y


def create_row(x, M):
    return [x**m for m in xrange(0, M+1)]


def create_phi(x_vector, M):
    phi = [create_row(x, M) for x in x_vector]
    return np.array(phi)


def OptimizeLS(x, t, M):
    phi = create_phi(x, M)
    # print phi.shape
    prod = np.dot(phi.T, phi)
    # print prod.shape
    i = np.linalg.inv(prod)
    # print i.shape
    m = np.dot(i, phi.T)
    # print m.shape
    # print "t ", t.shape
    w = np.dot(m, t)
    # print w.shape
    return w


def get_learned_polynomial(w_vector, x_vector):
    res = []
    w_pol = zip(w_vector, range(len(w_vector)+1))
    for x in x_vector:
        new_point = []
        new_point = [w*x**i for w, i in w_pol]
        res.append(sum(new_point))
    return res


if __name__ == "__main__":
    # Q1.1
    method = lambda x: math.sin(2*math.pi*x)
    vectorised_method = np.vectorize(method)
    x_vector, t_vector, real_vector = generateDataset(10, vectorised_method, 0.01)
    # plot_data(x_vector, t_vector)
    # End of Q1.1
    # ##########################
    #Q1.2
    # phi = create_phi(x_vector, 10)
    # print phi.shape
    # w_vector = OptimizeLS(x_vector, t_vector, 1)
    # w_vector = OptimizeLS(x_vector, t_vector, 3)
    # w_vector = OptimizeLS(x_vector, t_vector, 5)
    w_vector = OptimizeLS(x_vector, t_vector, 9)
    learned_polynomial = get_learned_polynomial(w_vector, x_vector)
    # learned_polynomial = get_learned_polynomial(z, x_vector)
    # plot_data(x_vector, learned_polynomial, real_vector)
    # END of Q1.2
    # ##########################
    # Q1.3
    


