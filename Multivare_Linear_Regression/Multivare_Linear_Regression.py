#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File: Multivare_Linear_Regression.py
# @Time: 2019/2/28 21:03
# @Author: Hao Chen
# @Contact: haochen273@gmail.com
import numpy as np
import matplotlib.pyplot as plt

def readfile(path):
    X=[]
    y=[]
    with open(path,'r') as f:
        for line in f:
            X.append([1,float(line.split(',')[0])])
            y.append(float(line.split(',')[1]))
    return X,y

def dataplot(x,theta,y):
    plt.plot(x,y,'rx',markersize = 10)
    plt.ylabel("Profit in $10000s")
    plt.xlabel("Population of City in 10000s")
    plt.plot(X[:,1],X*theta,'-')
    plt.show()


def gradientDescent(X,y,theta,alpha,num_iters = 1500):
    m = len(y)
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        theta = theta - alpha * (X.T * (X*theta - np.mat(y).T)) / m
        J_history[i] = computeCost(X,y,theta)
        print("The %s iterations: %s"%(i,J_history[i]))
    return theta,J_history

def plotCostFunctions(J_history):
    plt.plot(range(len(J_history)),J_history)
    plt.ylabel("Cost Functions")
    plt.xlabel("Number of iterations")
    plt.show()


def computeCost(X,y,theta):
    m = len(y)
    J = 0
    for i in range(m):
        J = J +float((X[i]*theta - y[i])**2)

    return J/(2*m)


if __name__ == "__main__":
    theta = np.mat([[0], [0]])
    iterations = 1500
    alpha = 0.01
    iterations = 1500
    path = r"C:\Users\HAO\Desktop\MachineLearning\data\ex1data1.txt"

    x, y = readfile(path)  # 小写的X不是矩阵，是list，大写的X是矩阵。
    X = np.mat(x)
    J = computeCost(X, y, theta)
    theta,J_history = gradientDescent(X, y, theta, alpha, iterations)
    dataplot(X[:, 1], theta, y)
    plotCostFunctions(J_history)