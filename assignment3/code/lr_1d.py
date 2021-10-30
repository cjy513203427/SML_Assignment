# -*- encoding: utf-8 -*-
'''
@File    :   lr_1d.py.py    
@Modify Time      @Author       @Desciption
------------      -------       -----------
2021/7/5 22:51   Jonas           None
'''

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm



train_data = np.loadtxt("lin_reg_train.txt")
test_data = np.loadtxt("lin_reg_test.txt")


def Xy(data):
    n = len(data)
    x = np.empty((2, n))
    y = np.empty((n, 1))
    for i in range(0, n):
        x[0, i] = data[i, 0]
        x[1, i] = 1
        y[i, 0] = data[i, 1]  # read the second column from data and give to the y
    return x, y


def phiIJ(data):
    X_data = data[:, 0]
    y = data[:, 1]
    N = data.shape[0]
    k_dim = 20
    X_ = np.zeros([X_data.shape[0], 20])
    for i in range(X_data.shape[0]):
        for j in range(k_dim):
            X_[i][j] = np.power(np.e, ((-0.5 * B) * ((X_data[i] - (j + 1) * 0.1) ** 2)))

    b = np.ones(N)
    X_ = np.concatenate([X_, b.reshape(N, 1)], axis=1)
    return X_.T, y


def lambda_I(alpha, beta):
    c = alpha / beta
    I = np.zeros([21, 21])
    for j in range(0, 21):
        I[j, j] = c
    return I


# get w
def parameter_posterior(X, y, ci):
    return np.linalg.inv(X @ X.T + ci) @ X @ y


def predicted_value(x, w):
    y = np.empty((len(x.T), 1))
    for i in range(0, len(y)):
        x_i = x.T[i]
        y[i] = x_i @ w
    return y


def RMSE(y_pre, y):
    N = len(y_pre)
    sum = 0
    for i in range(0, N):
        sum = sum + pow((y_pre[i] - y[i]), 2)

    result = math.sqrt(sum / N)
    return result


def square(x_train, x_test, a, B):
    x_x = x_train @ x_train.T
    B_xx = B * x_x
    square = np.empty((len(x_test.T), 1))
    aI = np.zeros((B_xx.shape[0], B_xx.shape[0]))
    for j in range(0, B_xx.shape[0]):
        aI[j][j] = a
    inverse = np.linalg.inv((aI + B_xx))
    for i in range(0, len(square)):
        x = x_test.T[i]
        x_t = np.transpose(x)
        square[i] = (1 / B) + np.matmul((np.matmul(x, inverse)), x_t)

    return square


def Gaussian_Distribution(mean, square, y_data):
    p = np.empty((len(mean), 1))
    for i in range(0, len(square)):
        p1 = 1 / (math.sqrt(2 * math.pi * square[i]))
        p2 = ((-1) * pow((y_data[i] - mean[i]), 2)) / (2 * square[i])
        p[i] = p1 * math.exp(p2)
    return p


def average_log_likelihood(p):
    for i in range(len(p)):
        if i == 0:
            sumy = np.log(p[i])
        else:
            sumy = sumy + np.log(p[i])

    average = sumy / len(p)
    return average


if __name__ == '__main__':
    B = 1 / (0.1 ** 2)
    a = 0.01
    x_train_bayesian_ori, y_train_bayesian_ori = Xy(train_data)

    x_train, y_train = phiIJ(train_data)
    test_x, test_y = phiIJ(test_data)

    ci = lambda_I(a, B)
    w_posterior = parameter_posterior(x_train, y_train, ci)
    test_predicted_value = predicted_value(test_x, w_posterior)
    test_p = Gaussian_Distribution(test_predicted_value, square(x_train, test_x, a, B), test_y)
    log_l_test = average_log_likelihood(test_p)
    print("the log-likelihood of the test is" + str(log_l_test))
    print("RMSE test is " + str(RMSE(test_predicted_value, test_y)))

    w_posterior_train = parameter_posterior(x_train, y_train, ci)
    train_predicted_value = predicted_value(x_train, w_posterior_train)
    train_p = Gaussian_Distribution(train_predicted_value, square(x_train, x_train, a, B),
                                    y_train)

    log_l_train = average_log_likelihood(train_p)
    print("the log-likelihood of the train is" + str(log_l_train))
    print("RMSE train is " + str(RMSE(train_predicted_value, y_train)))

    x_ = np.linspace(np.min(x_train_bayesian_ori[0]), np.max(x_train_bayesian_ori[0]), num=100).reshape(100, 1)
    x_ = np.concatenate([x_, np.ones(100).reshape(100, 1)], axis=1)
    x_maped, _ = phiIJ(x_)
    y_ = predicted_value(x_maped, w_posterior)

    sig_p = square(x_maped, x_maped, a, B)
    sig_p = np.sqrt(sig_p)

    plt.plot(x_.T[0], y_, c='blue', label='prediction')
    plt.scatter(x_train_bayesian_ori[0], y_train_bayesian_ori, c='black', label='original train data points')
    for i in range(3):
        plt.fill_between(x_.T[0], y_.reshape(100) + sig_p.reshape(100) * (i + 1.),
                         y_.reshape(100) - sig_p.reshape(100) * (i + 1.),
                         color="b", alpha=0.3)
    plt.title("Bayesian Linear Regression ")
    plt.xlabel('x')
    plt.ylabel('y')

    plt.legend()
    plt.show()
