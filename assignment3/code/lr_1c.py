# -*- encoding: utf-8 -*-
'''
@File    :   lr_1c.py    
@Modify Time      @Author       @Desciption
------------      -------       -----------
2021/7/5 18:30   Jonas           None
'''
import math
import matplotlib.pyplot as plt
import numpy as np

train_data = np.loadtxt("lin_reg_train.txt")
test_data = np.loadtxt("lin_reg_test.txt")

# get value of X and y
def Xy(data):
    n = len(data)
    X = np.zeros((2, n))
    y = np.zeros((n, 1))
    for i in range(0, n):
        X[0, i] = data[i, 0]
        X[1, i] = 1
        y[i, 0] = data[i, 1]

    return X, y

# lamda @ I
def lambda_I(alpha, beta):
    c = alpha / beta
    I = np.zeros((2, 2))
    for i in range(0, 2):
        I[i][i] = c

    return I

# get w
def parameter_posterior(X, y, ci):
    return np.linalg.inv(X @ X.T + ci) @ X @ y

def predicted_value(x, w):
    # y = np.empty((len(x), 1))
    # for i in range(0, len(y)):
    #     y[i] = x[i] @ w
    #
    # return y

    x_transpose = np.transpose(x)
    x_i=np.empty((1,2))
    y=np.empty((len(x_transpose),1))
    for i in range(0,len(y)):
        x_i=x_transpose[i]
        y[i]=np.matmul(x_i,w)
    return y

def RMSE(y_pre, y):
    n = len(y_pre)
    sum = 0
    for i in range(0, n):
        sum = sum + (y_pre[i] - y[i]) ** 2
    result = (sum / n) ** 0.5

    return result

def square(x_train, x_test, a, B):
    # lecture08 Page50
    # square = 1/B + x_test.T @ np.linalg.inv(B * x_train @ x_train.T + alphaI) @ x_test
    x_transpose = np.transpose(x_train)
    x_test_transpose = np.transpose(x_test)
    B_xx = B * (x_train @ x_transpose)
    square = np.zeros((len(x_test_transpose), 1))
    aI = np.zeros((2, 2))
    for j in range(0, 2):
        aI[j][j] = a
    inverse = np.linalg.inv((aI + B_xx))
    for i in range(0, len(square)):
        x = x_test_transpose[i]
        x_t = np.transpose(x)
        square[i] = (1/B) + x @ inverse @ x_t



    return square

def Gaussian(mean, square, y_data):
    p = np.empty((len(mean), 1))
    for i in range(0, len(square)):
        p1 = 1 / math.sqrt(2 * math.pi * square[i])
        p2 = ((-1) * pow((y_data[i] - mean[i]), 2)) / (2 * square[i])
        p[i] = p1 * math.exp(p2)

    return p

def average_log_likelihood(p):
    for i in range(len(p)):
        if i == 0:
            sum_y = np.log(p[i])
        else:
            sum_y = sum_y + np.log(p[i])

    average = sum_y / len(p)
    return average

if __name__ == '__main__':

    x_train, y_train = Xy(train_data)
    x_test, y_test = Xy(test_data)

    alpha = 0.01
    beta = 1 / (0.1 ** 2)
    ci = lambda_I(alpha, beta)
    w_posterior = parameter_posterior(x_train, y_train, ci)
    test_predicted_value = predicted_value(x_test, w_posterior)

    test_p = Gaussian(test_predicted_value, square(x_train, x_test, alpha, beta), y_test)
    log_l_test = average_log_likelihood(test_p)
    print("the log-likelihood of the test is"+str(log_l_test))
    print("rmse test is"+str(RMSE(test_predicted_value, y_test)))

    w_posterior_train = parameter_posterior(x_train, y_train, ci)
    train_predicted_value = predicted_value(x_train, w_posterior_train)
    train_p = Gaussian(train_predicted_value, square(x_train, x_train, alpha, beta), y_train)

    log_l_train = average_log_likelihood(train_p)
    print("the log-likelihood of the train is"+str(log_l_train))
    print("rmse train is" + str(RMSE(train_predicted_value, y_train)))

    x_ = np.linspace(np.min(x_train[0]), np.max(x_train[1]), num = 100).reshape(100, 1)
    x_ = np.concatenate([x_, np.ones(100).reshape(100, 1)], axis=1)
    y_ = predicted_value(x_.T, w_posterior)

    sig_ = square(x_.T, x_.T, alpha, beta)
    sig_ = np.sqrt(sig_)

    plt.scatter(x_.T[0], y_, c = 'blue', label = 'prediction')
    plt.scatter(x_train[0], y_train, c = 'black', label = 'original train data points')
    for i in range(3):
        plt.fill_between(x_.T[0], y_.reshape(100) + sig_.reshape(100) * (i+1),
                         y_.reshape(100) - sig_.reshape(100) * (i+1),
                         color = "b", alpha = 0.4)
    plt.title("Bayesian Linear Regression")
    plt.xlabel('x')
    plt.ylabel('y')

    plt.legend
    plt.show()