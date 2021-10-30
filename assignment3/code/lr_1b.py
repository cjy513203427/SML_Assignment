# -*- encoding: utf-8 -*-
'''
@File    :   lr_1b.py    
@Modify Time      @Author       @Desciption
------------      -------       -----------
2021/7/5 13:42   Jonas           None
'''

import matplotlib.pyplot as plt
import numpy as np

train_data = np.loadtxt("lin_reg_train.txt")
test_data = np.loadtxt("lin_reg_test.txt")


def Xy(data, degree):
    n = len(data)
    d = degree + 1
    X = np.zeros((n, d))
    y = np.zeros((n, 1))
    for j in range(0, d):
        for i in range(0, n):
            if j == 0:
                X[i, j] = 1
            elif j == 1:
                X[i, j] = data[i, 0]
            else:
                X[i, j] = pow(data[i, 0], j)

    for i in range(0, n):
        y[i][0] = data[i][1]
    return X, y


def lambdaI(c, degree):
    d = degree + 1
    I = np.zeros((d, d))
    for j in range(0, d):
        I[j][j] = c

    return I


# get w
def w(X, y, ci):
    return np.linalg.inv(X.T @ X + ci) @ X.T @ y


def predicted_poly_value(x, w):
    y = np.empty((len(x), 1))
    for i in range(0, len(x)):
        y[i] = x[i] @ w

    return y


# calculate rmse
def RMSE_poly(y_pre, y):
    n = len(y_pre)
    sum = 0
    for i in range(0, n):
        sum = sum + (y_pre[i] - y[i]) ** 2
    r = (sum / n) ** 0.5

    return r


if __name__ == '__main__':

    x_train_real = np.zeros((len(train_data), 1))
    for i in range(len(train_data)):
        x_train_real[i] = train_data[i][0]

    for degree in range(2, 5):
        x, y = Xy(train_data, degree)
        x_test, y_test = Xy(test_data, degree)
        ci = lambdaI(0.01, degree)
        w_poly = w(x, y, ci)

        y_pre = predicted_poly_value(x, w_poly)
        y_pre_test = predicted_poly_value(x_test, w_poly)
        RMSE_poly_train = RMSE_poly(y_pre, y)
        RMSE_poly_test = RMSE_poly(y_pre_test, y_test)

        print("degree = " + str(degree))
        print("rmse train is" + str(RMSE_poly_train))
        print("rmse test is" + str(RMSE_poly_test))

        x_plot = np.linspace(np.min(x_train_real), np.max(x_train_real), num=100)
        x_plot = np.concatenate([x_plot.reshape(100, 1), np.ones(100).reshape(100, 1)], axis=1)
        x_plot_m, _ = Xy(x_plot, degree)
        # draw predicted curve
        y_plot = predicted_poly_value(x_plot_m, w_poly)
        # draw data dots
        plt.plot(x_plot.T[0], y_plot, c="blue", label='predicition line')
        plt.scatter(x_train_real, y, c="black", label='training data points')
        plt.title("Linear regression with degree =" + str(degree))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()
