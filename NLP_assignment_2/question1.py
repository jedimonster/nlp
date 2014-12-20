"""
Question1 code
"""
import numpy as np
import math
import scipy.linalg


def plot_data(x, y, z=None):
    """
    plot data
    """
    import matplotlib.pyplot as plt

    plt.title("Data visualization")

    plt.scatter(x, y, c='red', alpha=0.5)
    if z is not None:
        plt.scatter(x, z, alpha=0.5, c='blue', marker='*')
    plt.show()


def plot_function(x, y, method, x_points, t_points):
    import matplotlib.pyplot as plt

    real_vector = method(x)
    plt.title("Data visualization")
    a = plt.scatter(x_points, t_points, c='green', alpha=0.5, label='Ponits with noise')
    b = plt.plot(x, y, c='red', label='Learned polynom')
    c = plt.plot(x, real_vector, c='blue', label='Original polynom')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
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
    # plot_data(x_array, t_array, real_y)
    return x_array, t_array, real_y


def generateDataset3(N, f, sigma):
    x_array = np.linspace(0.0, 1.0, num=3 * N)
    norm_array = np.random.normal(scale=sigma, size=3 * N)
    real_y = f(x_array)
    t_array = f(x_array) + norm_array
    x_t_y = zip(x_array, t_array, real_y)
    np.random.shuffle(x_t_y)
    # train 60% test 20% c-v 20%
    train = x_t_y[0:N]
    c_v = x_t_y[N:2 * N]
    test = x_t_y[2 * N:]
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
    return [x ** m for m in xrange(0, M + 1)]


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
    phi_phiT = np.dot(phi.T, phi)
    prod_l = phi_phiT + lamb * np.eye(phi_phiT.shape[0])  # (phi*phiT+lambda*I)
    i = np.linalg.inv(prod_l)
    m = np.dot(i, phi.T)
    w = np.dot(m, t)
    return w


def calculate_error(x_vector, t_vector, w_vector):
    N = len(x_vector)
    res = []
    learned_polynomial = get_learned_polynomial(w_vector, x_vector)
    for l, t in zip(learned_polynomial, t_vector):
        res.append((t - l) ** 2)
    res = sum(res)
    res = math.sqrt(res)
    res = float(res) / float(N)
    return res


def optimizePLS2(xt, tt, xv, tv, M):
    best_w_vector = None
    best_lambda = None
    best_error = None
    errors = []
    train_erros = []
    log_lambda_range = np.linspace(-20, -40, num=100)
    for lambda_exponent in log_lambda_range:
        w_vector = optimizePLS(xt, tt, M, math.e ** lambda_exponent)
        terr = calculate_error(xt, tt, w_vector)
        err = calculate_error(xv, tv, w_vector)
        errors.append(err)
        train_erros.append(terr)

        if best_lambda is None or err < best_error:
            best_w_vector = w_vector
            best_lambda = lambda_exponent
            best_error = err

    print "best:"
    print best_error
    print best_lambda
    print best_w_vector
    #plot_data(log_lambda_range, errors)
    return best_w_vector, errors

# res = np.zeros((M+1, M+1))
#     print "length of x is: ", len(x)
#     for i in range(len(x)):
#         prod = np.array([x[i]**z for z in range(M+1)])
#         prod2 = np.array([x[i]**z for z in range(M+1)]).T
#         res += np.outer(prod, prod2)
#         import pdb
#         pdb.set_trace()

def bayesianEstimator(x, t, M, alpha, sigma2):

    phi = create_phi(x, M)

    res = np.zeros((M+1, M+1))
    for n in range(len(x)):
        phi_x = np.array([x[n]**i for i in range(0,M+1)])
        res += np.outer(phi_x, phi_x)


    s_inv = alpha*np.eye(M+1) + (float(1)/float(sigma2))*res

    s = np.linalg.inv(s_inv)
    print s
    print s.shape
    def variance(x):
        # import pdb
        # pdb.set_trace()
        ph_x = np.array([x**m for m in range(M+1)])
        ph_x = ph_x.T
        mul = np.dot(s, ph_x)
        # print " mul shape: ", mul.shape
        # print "s shape: ", s.shape
        # print mul.shape
        res = sigma2 + np.dot(ph_x.T, mul)
        return res

    def mean(x_inner):
        # import pdb
        # pdb.set_trace()
        inner_res = np.zeros(M+1)
        for inner_n in range(len(t)):
            phi_x = np.array([x[inner_n]**i for i in range(M+1)])
            mul = phi_x * t[inner_n]
            inner_res += mul
        ph_x = np.array([x_inner**m for m in range(M+1)])

        mid_res = np.dot(ph_x, s)
        return np.dot(mid_res, inner_res)*(float(1)/float(sigma2))
    # import pdb
    # pdb.set_trace()
    return mean, variance

def get_learned_polynomial(w_vector, x_vector):
    res = []
    w_pol = zip(w_vector, range(len(w_vector) + 1))
    for x in x_vector:
        new_point = []
        new_point = [w * x ** i for w, i in w_pol]
        res.append(sum(new_point))
    return res


if __name__ == "__main__":
    # # Q1.1
    method = lambda x: math.sin(2 * math.pi * x)
    vectorised_method = np.vectorize(method)
    # x_vector, t_vector, real_vector = generateDataset(10, vectorised_method, 0.1)
    # # plot_data(x_vector, t_vector)
    # # End of Q1.1
    # # ##########################
    # # Q1.2
    # # phi = create_phi(x_vector, 10)
    # # print phi.shape
    # w_vector = OptimizeLS(x_vector, t_vector, 1)
    # x_range = np.arange(0.0, 1.0, 0.005)
    # x_range = np.linspace(0.0, 1.0, 100000)
    # print "g ", len(x_range)
    # learned_polynomial = get_learned_polynomial(w_vector, x_range)
    # # plot_function(x_range, learned_polynomial)
    # # w_vector = OptimizeLS(x_vector, t_vector, 3)
    # # w_vector = OptimizeLS(x_vector, t_vector, 5)
    # w_vector = OptimizeLS(x_vector, t_vector, 9)
    # print w_vector
    # learned_polynomial = get_learned_polynomial(w_vector, x_range)
    # plot_function(x_range, learned_polynomial, vectorised_method, x_vector, t_vector)
    # learned_polynomial = get_learned_polynomial(w_vector, x_vector)
    # # plot_data(x_vector, learned_polynomial, real_vector)
    # # END of Q1.2
    # # ##########################
    # # Q1.3
    # w_vector2 = optimizePLS(x_vector, t_vector, 5, 0.01)
    # learned_polynomial = get_learned_polynomial(w_vector2, x_vector)
    # plot_data(x_vector, learned_polynomial, real_vector)
    train, c_v, test = generateDataset3(10, vectorised_method, 0.01)
    # sanity check
    # plot_data(train[0], train[1]+c_v[1]+test[1], train[2]+c_v[2]+test[2])
    best_w_vector, errors = optimizePLS2(train[0], train[1], c_v[0], c_v[1], 5)

    # pol_fitting = get_learned_polynomial(best_w_vector, x_vector)
    # print calculate_error(test[0], test[1], best_w_vector)
    # plot_data(x_vector, pol_fitting, real_vector)

    # plot_data(x_vector, pol_fitting)

    #QUESTION 1.4 God help us all
    x_vector, t_vector, real_vector = generateDataset(10, vectorised_method, 0.1)
    mean, variance = bayesianEstimator(x_vector, t_vector, 9, 0.005, 1/11.1)
    variance(0.5)
    mean(0.5)
    vectorised_mean = np.vectorize(mean)
    vectorized_variance = np.vectorize(variance)

    x_graph = np.linspace(0.0, 1.0, 100)
    samples = []
    for point in x_vector:
        mu = mean(point)
        var = variance(point)
        samples.append(np.random.normal(loc=mu,scale=var, size=1))
    import pylab
    pylab.scatter(x_vector,samples, c="black")
    pylab.plot(x_graph, vectorised_method(x_graph), c='red')
    pylab.scatter(x_vector, t_vector, c='green', alpha=0.5)
    pylab.fill_between(x_graph, vectorised_mean(x_graph) - np.sqrt(vectorized_variance(x_graph)),
                       vectorised_mean(x_graph) + np.sqrt(vectorized_variance(x_graph)), alpha=0.3, hatch='X')
    pylab.show()
