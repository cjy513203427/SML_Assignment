# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py.py    
@Modify Time      @Author       @Desciption
------------      -------       -----------
2021/8/3 1:28   Jonas           None
'''
 
import numpy as np
from matplotlib import pyplot as plt

def target(x):
    s = np.sin(x)
    return s + np.square(s)


def kernel(x, z):
    return np.exp(-(x - z) ** 2)


def compute_c(x, noise):
    return kernel(x, x) + noise


def predict(x, X, Y, C_inv, noise):
    k = np.array([kernel(x, val) for val in X])
    mean = k.T.dot(C_inv).dot(Y)
    std = np.sqrt(compute_c(x, noise) - k.T.dot(C_inv).dot(k))
    return mean, std


def plot_gp(X, mean, std, iteration):
    y1 = mean - 2 * std
    y2 = mean + 2 * std

    plt.plot(X, target(X), "-", color="red", label="$sin(x) + sin^2(x)$")
    plt.plot(X, mean, color="black", label="$\mu$")
    plt.fill_between(X, y1, y2, facecolor='blue', interpolate=True, alpha=.5, label="$2\sigma$")
    plt.title("$\mu$ and $\sigma$ for iteration={}".format(iteration))
    plt.xlabel("x")
    plt.ylabel("y")


def gpr():
    noise = 0.001
    step_size = 0.005
    X = np.arange(0, 2 * np.pi + step_size, step_size)
    iterations = 15

    std = np.array([np.sqrt(compute_c(x, noise)) for x in X])
    j = np.argmax(std)

    Xn = np.array([X[j]])
    Yn = np.array([target(Xn)])

    C = np.array(compute_c(X[j], noise)).reshape((1, 1))
    C_inv = np.linalg.solve(C, np.identity(C.shape[0]))

    for iteration in range(0, iterations):
        mean = np.zeros(X.shape[0])
        std = np.zeros(X.shape[0])

        for i, x in enumerate(X):
            mean[i], std[i] = predict(x, Xn, Yn, C_inv, noise)

        if iteration + 1 in [1, 2, 5, 10, 15]:
            plot_gp(X, mean, std, iteration + 1)
            plt.plot(Xn, Yn, "o", c="blue", label="sampled")
            plt.legend()
            plt.show()

        j = np.argmax(std)
        Xn = np.append(Xn, X[j])
        Yn = np.append(Yn, target(Xn[-1]))

        # update C matrix
        x_new = Xn[-1]
        k = np.array([kernel(x_new, val) for val in Xn])
        c = kernel(x_new, x_new) + noise

        dim = C.shape[0]
        C_new = np.empty((dim + 1, dim + 1))
        C_new[:-1, :-1] = C
        C_new[-1, -1] = c
        C_new[:, -1] = k
        C_new[-1:] = k.T

        C = C_new
        C_inv = np.linalg.solve(C, np.identity(C.shape[0]))

gpr()