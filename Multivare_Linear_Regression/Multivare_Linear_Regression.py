#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File: Multivare_Linear_Regression.py
# @Time: 2019/2/28 21:03
# @Author: Hao Chen
# @Contact: haochen273@gmail.com
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
实现了Linear Regression的基础版本和正则化版本
'''

def readData(path):
    data = pd.read_csv(path, header = None)
    X= data.iloc[:,0:-1]
    y = data.iloc[:,-1]
    ones = np.ones((len(y),1))
    X = np.hstack((ones,X))
    # 为了让y的形状为(m*1)
    y = y[:, np.newaxis]
    return X,y


def dataplot(X,theta,y):
    plt.plot(X[:,1],y,'rx',markersize = 10)
    plt.ylabel("Profit in $10000s")
    plt.xlabel("Population of City in 10000s")
    plt.plot(X[:,1],np.dot(X, theta),'-')
    plt.show()


def gradientDescent(X,y,theta,alpha,num_iters = 1500,regulaization = False, lambda_t = 1):
    '''
    对于每个GD，使用使用GD算法寻找最佳的参数值
    :param X:所有样本的值:m * (n+1)
    :param y:一个样本的y: m*1
    :param theta: 需要找的参数:(n+1)*1
    :param alpha: 学习率
    :param num_iters: 循环次数
    :return: 最优的参数，CostFucntion history data
    '''
    m = len(y)
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        # theta: (n+1) * 1
        if regulaization:
            theta = theta - alpha * (np.dot(X.T, (np.dot(X, theta) - y)) + lambda_t * theta) / m
        else:
            theta = theta - alpha * np.dot(X.T, (np.dot(X, theta) - y)) / m
        J_history[i] = computeCost(X,y,theta)

    return theta,J_history

def plotCostFunctions(J_history):
    plt.plot(range(len(J_history)),J_history)
    plt.ylabel("Cost Functions")
    plt.xlabel("Number of iterations")
    plt.show()


def computeCost(X,y,theta):
    '''
    :param X: 所有样本的值:m * (n+1)
    :param y: 所有样本的y: m*1
    :param theta: n个特征值对于的参数:(n+1)*1
    :return: 引用theta后的所有累加损失函数
    '''
    m = len(y)
    temp = np.dot(X, theta) - y
    return np.sum(np.power(temp,2))/(2*m)


if __name__ == "__main__":
    path = r"C:\Users\HAO\Desktop\MachineLearning\data\ex1data1.txt"
    X,y = readData(path)
    theta = np.zeros((X.shape[1],1))
    iterations = 1000
    alpha = 0.01

    # 基础版本
    theta,J_history = gradientDescent(X, y, theta, alpha, iterations)
    dataplot(X, theta, y)
    plotCostFunctions(J_history)
    #  正则化版本并对比
    theta_re, J_history_Re = gradientDescent(X, y, theta, alpha, iterations, True,2)

    plt.plot(range(len(J_history)),J_history,'red',label = "Linear Regression")
    plt.plot(range(len(J_history_Re)), J_history_Re, 'blue', label="Linear Regression with RE")
    plt.ylabel("Cost Functions")
    plt.xlabel("Number of iterations")
    plt.legend()
    plt.show()