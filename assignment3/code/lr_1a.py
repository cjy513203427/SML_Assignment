# -*- encoding: utf-8 -*-
'''
@File    :   lr_1a.py
@Modify Time      @Author       @Desciption
------------      -------       -----------
2021/7/5 1:15   Jonas           None
'''

import matplotlib.pyplot as plt
import numpy as np

train_data = np.loadtxt("lin_reg_train.txt")
test_data = np.loadtxt("lin_reg_test.txt")

# get value of X and y
def Xy(data):
    n = len(data)
    X = np.zeros((n, 2))
    y = np.zeros((n, 1))
    for i in range(0, n):
        X[i, 0] = data[i, 0]
        X[i, 1] = 1
        y[i, 0] = data[i, 1]

    return X, y

# lamda @ I
def lambda_I(c):
    I = np.zeros((2, 2))
    for i in range(0, 2):
        I[i][i] = c

    return I

# get w
def w(X, y, ci):
    return np.linalg.inv(X.T @ X + ci) @ X.T @ y

def predicted_value(x, w):
    y = np.empty((x.shape[0], 1))
    for i in range(0, x.shape[0]):
        y[i] = x[i] @ w

    return y

def RMSE(y_pre, y):
    n = len(y_pre)
    sum = 0
    for i in range(0, n):
        sum = sum + (y_pre[i] - y[i]) ** 2
    result = (sum / n) ** 0.5

    return result

def plot(x_real, y_real, y_predict):
    plt.scatter(x_real, y_real, c = 'black')
    plt.plot(x_real, y_predict, 'b-')
    plt.show()

if __name__ == '__main__':

    x, y = Xy(train_data)
    ci = lambda_I(0.01)
    w = w(x, y, ci)
    print("w is" + str(w))

    y_pre = predicted_value(x, w)
    rmse_train = RMSE(y_pre, y)
    print("rmse train is" + str(rmse_train))

    x_test, y_test = Xy(test_data)
    y_test_pre = predicted_value(x_test, w)
    rmse_test = RMSE(y_test_pre, y_test)
    print("rmse test is" + str(rmse_test))

    x_train_real = np.empty((len(x), 1))
    for i in range(len(x)):
        x_train_real[i] = x[i][0]
    plot(x_train_real, y, y_pre)