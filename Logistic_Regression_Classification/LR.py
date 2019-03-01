#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File: LR.py
# @Time: 2019/2/28 21:37
# @Author: Hao Chen
# @Contact: haochen273@gmail.com

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

def readData(path):
    data = pd.read_csv(path, header = None)
    X= data.iloc[:,:-1]
    y = data.iloc[:,-1]
    ones = np.ones((len(y),1))
    X = np.hstack((ones,X))
    # 为了让y的形状为(m*1)
    y = y[:, np.newaxis]
    return X,y

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def CostFunction(theta_t, X_t,y_t,regularization=False,lambda_t = 1):
    m = len(y)
    if regularization:
        J = (-1/m) * (y_t.T @ np.log(sigmoid(X_t @ theta_t)) + (1 - y_t.T) @ np.log(1 - sigmoid(X_t @ theta_t)))
        reg = (lambda_t / (2 * m)) * (theta[:1]) ** 2
        J = J + reg
    else:
        J = (-1/m) * np.sum(np.multiply(y, np.log(sigmoid(np.dot(X,theta)))) +
                            np.multiply((1-y),np.log(1-sigmoid(np.dot(X,theta)))))
    return J

def gradient(theta,X,y):
     m =len(y)
     grad = ((1/m) * np.dot(X.T, sigmoid(np.dot(X,theta)) - y))
     return grad


def lrGradientDescent(theta, X, y, lambda_t):
    m = len(y)
    grad = np.zeros([m,1])
    grad = (1/m) * X.T @ (sigmoid(X @ theta) - y)
    grad[1:] = grad[1:] + (lambda_t / m) * theta[1:]
    return grad

def plotBoundary(X,y,theta_optimized):
    plot_x = [np.min(X[:, 1] - 2), np.max(X[:, 2] + 2)]
    plot_y = -1 / theta_optimized[2] * (theta_optimized[0]
                                        + np.dot(theta_optimized[1], plot_x))
    mask = y.flatten() == 1
    adm = plt.scatter(X[mask][:, 1], X[mask][:, 2])
    not_adm = plt.scatter(X[~mask][:, 1], X[~mask][:, 2])
    decision_boun = plt.plot(plot_x, plot_y)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend((adm, not_adm), ('Admitted', 'Not admitted'))
    plt.show()

def accuracy(X,y,theta,cutoff):
    pred = [sigmoid(np.dot(X,theta)) >=cutoff]
    acc = np.mean(pred==y)
    return acc*100


def mapFeature(X1, X2):
    degree = 6
    out = np.ones(X.shape[0])[:,np.newaxis]
    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.hstack((out, np.multiply(np.power(X1, i-j), np.power(X2, j))[:,np.newaxis]))
    return out




if __name__ == "__main__":
    path = r"C:\Users\HAO\Desktop\MachineLearning\data\ex2data1.txt"
    X,y = readData(path)
    theta = np.zeros((X.shape[1],1))

    # 无需设置learning rate
    temp = opt.fmin_tnc(func=CostFunction,
                        x0=theta.flatten(), # 将vector变为array(这个函数才可以运行)
                        fprime=gradient,
                        args=(X,y.flatten()))
    # the output of above function is a tuple whose first element
    # contains the optimized values of theta
    theta_optimized = temp[0]
    print("Optimal theta: %s"%(theta_optimized))

    J = CostFunction(theta_optimized[:, np.newaxis], X, y)
    print("final cost function: %s"%(J))

    plotBoundary(X,y,theta_optimized)

    final_accuracy = accuracy(X,y.flatten(), theta_optimized,0.5)

    print("We got the final accuracy is: %s"%(final_accuracy))

    print(">"*20 +"Regularization" + "<"*20)

    lambda_set = 1
    J_re = CostFunction(theta, X, y, True,lambda_t=lambda_set)

    output = opt.fmin_tnc(func=CostFunction,
                          x0=theta.flatten(), fprime=lrGradientDescent,
                          args=(X, y.flatten(), lambda_set))
    theta_final = output[0]
    print("Optimal theta is: %s"%(theta_final))  # theta contains the optimized values

    final_accuracy_re = accuracy(X, y.flatten(), theta_final, 0.5)
    print("Accuracy: %s"%(final_accuracy_re))

