"""
Question1 code
"""
import numpy as np
import math


def plot_data(x, y, z=None):
    """
    Simple demo of a scatter plot.
    """
    import numpy as np
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
    return x_array, t_array


if __name__ == "__main__":
    # Q1.1
    method = lambda x: math.sin(2*math.pi*x)
    vectorised_method = np.vectorize(method)
    x_vector, t_vector = generateDataset(100, vectorised_method, 0.01)
    plot_data(x_vector, t_vector)
    # End of Q1.1
    # ##########################
    #Q1.2

