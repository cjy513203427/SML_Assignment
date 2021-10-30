# -*- encoding: utf-8 -*-
'''
@File    :   ridgeTest.py.py
@Modify Time      @Author       @Desciption
------------      -------       -----------
2021/6/24 15:04   Jonas           None
'''

import numpy as np
import matplotlib.pyplot as plt


def ridgeRegres(xMat, yMat, lam=0.01):
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    # (X^T X + lamda I)^-1 X^T Y
    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    # Regularize data
    yMean = np.mean(yMat)
    # print(yMean)
    yMat = yMat - yMean
    # print(xMat)
    # regularize X's
    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    # (features - mean) / variance
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    # Get different value of lambda, get coefficient
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, np.exp(i - 10))
        wMat[i, :] = ws.T
    return wMat


# import data
ex0 = np.loadtxt('lin_reg_train.txt', delimiter=' ')
ex1 = np.loadtxt('lin_reg_test.txt', delimiter=' ')
xArr = ex0[:, 0:-1]
yArr = ex0[:, -1]
# print(xArr,yArr)
ridgeWeights = ridgeTest(xArr, yArr)
# print(ridgeWeights)

plt.plot(ridgeWeights)
# plt.show()

xArr1 = ex1[:, 0:-1]
yArr1 = ex1[:, -1]
ridgeWeights1 = ridgeTest(xArr1, yArr1)
# print(ridgeWeights)
plt.plot(ridgeWeights1)

plt.xlabel("x axis label")
plt.ylabel("y axis label")
plt.title("Linear Features")
plt.legend(["Train", "Test"])
plt.show()

# Calculate MSE
mse = ((ex0 - ex1[0:50, :]) ** 2).mean(axis=None)
print(mse)
