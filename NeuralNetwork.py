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
            self.weight.append((2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)
            self.weight.append((2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1) * 0.25)

    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        X = np.atleast_2d(X)
        temp = np.ones([X.shape[0], X.shape[1] + 1])
        temp[:, 0:-1] = X
        X = temp
        y = np.array(y)

        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weight)):
                a.append(self.activation(np.dot(a[l], self.weight[l])))
            error = y[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]

            # start backprobagation
            for l in range(len(a) - 2, 0, -1):
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
