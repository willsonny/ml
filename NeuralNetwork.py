import numpy as np


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1.0 - np.tanh(x) * np.tanh(x)

def logistic(x):
    return 1/(1+np.exp(-x))

def logistic_derivative(x):
    return logistic(x)*(1-logistic())

class NeturaNetwork(object):
    def __init__(self,layers,activation = 'tanh'):
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tang':
            self.activation = tanh
            self.activation = tanh_deriv

        self.weight =[]
        for i in range(1,len(layers)-1):
            self.weight.append((2*np.random.random((layers[i-1])+1)))