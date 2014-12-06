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

def generateDataset3(N, f, sigma):
    x_array = np.linspace(0.0, 1.0, num=N)
    norm_array = np.random.normal(scale=sigma, size=N)
    real_y = f(x_array)
    t_array = f(x_array) + norm_array
    x_t_y = zip(x_array, t_array, real_y)
    np.random.shuffle(x_t_y)
    # train 60% test 20% c-v 20%
    train = x_t_y[0:int(N*0.6)]
    c_v = x_t_y[int(N*0.6):int(N*0.8)]
    test = x_t_y[int(N*0.8):]
    x, t, y = [list(g) for g in zip(*x_t_y)]
    train = [list(g) for g in zip(*train)]
    c_v = [list(g) for g in zip(*c_v)]
    test = [list(g) for g in zip(*test)]
    # print len(train[0])
    # print len(test[0])
    # print len(c_v[0])
    # print "t ", t_array
    # print "real ", real_y
    # print train
    # print test
    # print c_v
    return train, c_v, test


def create_row(x, M):
    return [x**m for m in xrange(0, M+1)]


def create_phi(x_vector, M):
    phi = [create_row(x, M) for x in x_vector]
    return np.array(phi)


def OptimizeLS(x, t, M):
    phi = create_phi(x, M)
    # print phi.shape
    prod = np.dot(phi.T, phi)
    print prod.shape
    i = np.linalg.inv(prod)
    # print i.shape
    m = np.dot(i, phi.T)
    # print m.shape
    # print "t ", t.shape
    w = np.dot(m, t)
    # print w.shape
    return w


def optimizePLS(x, t, M, lamb):
    phi = create_phi(x, M)
    prod = np.dot(phi.T, phi)
    prod_l = prod + lamb*np.eye(prod.shape[0])
    i = np.linalg.inv(prod_l)
    m = np.dot(i, phi.T)
    w = np.dot(m, t)
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
    # plot_data(x_vector, learned_polynomial, real_vector)
    # END of Q1.2
    # ##########################
    # Q1.3
    w_vector2 = optimizePLS(x_vector, t_vector, 5, 0.01)
    learned_polynomial = get_learned_polynomial(w_vector2, x_vector)
    # plot_data(x_vector, learned_polynomial, real_vector)
    train, c_v, test = generateDataset3(10, vectorised_method, 0.01)
    #sanity check
    #plot_data(train[0]+c_v[0]+test[0], train[1]+c_v[1]+test[1], train[2]+c_v[2]+test[2])
    
