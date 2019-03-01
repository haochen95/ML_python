#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File: LR.py
# @Time: 2019/2/28 21:37
# @Author: Hao Chen
# @Contact: haochen273@gmail.com

import numpy as np

def readfile(path):
    X=[]
    y=[]
    with open(path,'r') as f:
        for line in f:
            X.append([1,float(line.split(',')[0])])
            y.append(float(line.split(',')[1]))
    return X,y

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def lrCostFunction(theta_t, X_t,y_t, lambda_t):
    m = len(y_t)
    J = (-1/m) * (y_t.T * np.log(sigmoid(X_t * theta_t)) + (1-y_t)*np.log(1-sigmoid(X_t*theta_t)))
    reg = (lambda_t / (2*m)) * (theta_t[:1])**2
    J = J + reg
    return J

def lrGradientDescent(theta,X,y, alpha,lambda_t,num_iters = 1000):
    m = len(y)
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        theta = theta*(1-(alpha*lambda_t)/m) - alpha * (1/m) * X.T*(sigmoid(X*theta) - y)
        J_history[i] = lrCostFunction(theta,X,y,lambda_t)
        print("The %s iterations: %s" % (i, J_history[i]))

    return theta,J_history

def mapFeature(X1, X2):
    degree = 6
    out = np.ones(X1.shape[0])[:,np.newaxis]
    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.hstack((out, np.multiply(np.power(X1, i-j),np.power(X2, j))[:,np.newaxis]))
    return out

if __name__ == "__main__":
    theta = np.mat([[0], [0]])
    iterations = 1500
    alpha = 0.01
    iterations = 1500
    path = r"C:\Users\HAO\Desktop\MachineLearning\data\ex2data2.txt"

    x, y = readfile(path)  # 小写的X不是矩阵，是list，大写的X是矩阵。
