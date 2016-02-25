# -*- coding: utf-8 -*-
import numpy as np


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1.0 - np.tanh(x) * np.tanh(x)


def logistic(x):
    return 1 / (1 + np.exp(-x))


def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))


class NeturaNetwork():
    def __init__(self, layers, activation='tanh'):
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        self.weight = []

        for i in range(1, len(layers) - 1):
            #5 * np.random.random_sample((3, 2)) - 5
            #生成一个矩阵（3x2) 值在0，1间，5*（0,1）-5 随机值 -5 生产一个（-5,0）的随机数组
            # array([[-3.99149989, -0.52338984],
            #        [-2.99091858, -0.79479508],
            #        [-1.23204345, -1.75224494]])
            #生成一个（65,101)的矩阵，多的行数用来存储bias（偏移量）

            self.weight.append((2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)
            self.weight.append((2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1) * 0.25)

    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        X = np.atleast_2d(X)
        #生成一个（65,101)的矩阵，多的行数用来存储bias（偏移量）
        temp = np.ones([X.shape[0], X.shape[1] + 1])
        temp[:, 0:-1] = X
        X = temp
        y = np.array(y)
        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]
            #正向更新
            for l in range(len(self.weight)):
                a.append(self.activation(np.dot(a[l], self.weight[l])))
            error = y[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]

        # start backprobagation,deltas为（10，）的矩阵，计算内积需要适用weight[l].T转化 ndarray.T == numpy.ndarray.transpose
        # >>> a
        # array([[0, 1, 2],
        #        [3, 4, 5]])
        # >>> a.T
        # array([[0, 3],
        #        [1, 4],
        #        [2, 5]])
            for l in range(len(a) - 2, 0, -1):
                print self.weight[l].shape
                deltas.append(deltas[-1].dot(self.weight[l].T) * self.activation_deriv(a[l]))

            deltas.reverse()
            for i in range(len(self.weight)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weight[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weight)):
            a = self.activation(np.dot(a, self.weight[l]))
        return a
