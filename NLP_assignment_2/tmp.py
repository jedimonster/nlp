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


def get_learned_polynomial(w_vector, x_vector):
    res = []
    w_pol = zip(w_vector, range(len(w_vector) + 1))
    for x in x_vector:
        new_point = []
    new_point = [w * x ** i for w, i in w_pol]
    res.append(sum(new_point))
    return res


def plot_function(x, y, method, x_points, t_points):
    import matplotlib.pyplot as plt

    real_vector = method(x)
    plt.title("Data visualization")
    a = plt.scatter(x_points, t_points, c='green', alpha=0.5, label='Points with noise')
    b = plt.plot(x, y, c='red', label='Learned polynom')
    c = plt.plot(x, real_vector, c='blue', label='Original polynom')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.show()


method = lambda x: math.sin(2 * math.pi * x)
vectorised_method = np.vectorize(method)
x_range = np.linspace(0.0, 1.0, 100000)
x_vector, t_vector, real_vector = generateDataset(10, vectorised_method, 0.1)
phi = create_phi(x_vector, 10)
for d in [1, 3, 5, 9]:
    w_vector = OptimizeLS(x_vector, t_vector, d)
    learned_polynomial = get_learned_polynomial(w_vector, x_range)
    # plot_data(x_vector, learned_polynomial, real_vector) - original tryto draw. wasnt informative enough
    plot_function(x_range, learned_polynomial, vectorised_method, x_vector, t_vector)