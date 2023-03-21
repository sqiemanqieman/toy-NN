# MLQP layer of NN implemented by hand

import numpy as np
from abc import abstractmethod

from activation import sigmoid


class Layer:
    @abstractmethod
    def call(self, inputs, *args, **kwargs):
        pass


class MLQPLayer(Layer):
    def __init__(self, n_outputs, activation=sigmoid):
        super(MLQPLayer, self).__init__()
        self.mu, self.nu, self.bias, self.local_grad = None, None, None, None
        self.x, self.g, self.y = None, None, None
        self.n_outputs = n_outputs
        self.first = True
        self._is_output = False
        self.activation = activation

    def build(self, input_shape):
        # self.mu = np.random.normal(0, 1, (int(input_shape[-1]), self.n_outputs))
        # self.nu = np.random.uniform(0, 1, (int(input_shape[-1]), self.n_outputs))
        # self.bias = np.random.uniform(0, 1, (1, self.n_outputs))
        self.mu = np.zeros((int(input_shape[-1]), self.n_outputs))
        self.nu = np.zeros((int(input_shape[-1]), self.n_outputs))
        self.bias = np.zeros((1, self.n_outputs))
        self.local_grad = np.zeros((1, self.n_outputs))

    def call(self, inputs, *args, **kwargs):
        inputs = np.asarray(inputs).reshape((1, -1))
        if self.first:
            self.build(inputs.shape)
            self.first = False
        self.x = inputs
        self.g = np.matmul(self.x*self.x, self.mu) + np.matmul(self.x, self.nu) + self.bias
        self.y = self.activation(self.g)  # sigmoid activation
        return self.y

    def set_output(self):
        self._is_output = True

    def is_output(self):
        return self._is_output


class NN:
    def __init__(self, layers):
        self.layers = layers
        self.layers[-1].set_output()
        self.lr = 0.01

    def forward(self, x):
        for layer in self.layers:
            x = layer.call(x)
        return x

    def backward(self, y):
        layer, next_layer, delta_mu, delta_nu, delta_bias = None, None, None, None, None
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if layer.is_output():
                layer.local_grad = (y - layer.y) * layer.activation.D(layer.g)
            else:
                next_layer = self.layers[i + 1]
                layer.local_grad = layer.activation.D(layer.g) * \
                                   (2 * layer.y * np.matmul(next_layer.local_grad, next_layer.mu.T)
                                    + np.matmul(next_layer.local_grad, next_layer.nu.T))
                next_layer.mu = next_layer.mu + delta_mu
                next_layer.nu = next_layer.nu + delta_nu
                next_layer.bias = next_layer.bias + delta_bias
            delta_mu = self.lr * np.matmul(layer.x.reshape((layer.x.shape[-1], 1)) ** 2, layer.local_grad)
            delta_nu = self.lr * np.matmul(layer.x.reshape((layer.x.shape[-1], 1)), layer.local_grad)
            delta_bias = self.lr * layer.local_grad
        layer.mu = layer.mu + delta_mu
        layer.nu = layer.nu + delta_nu
        layer.bias = layer.bias + delta_bias

    def fit(self, xs, ys, epochs, display_period=100):
        for epoch in range(epochs):
            if display_period and epoch % display_period == 0:
                print('epoch: {}/{} mse/mae: {}/{}'.format(epoch, epochs, *self.evaluate(xs, ys)))
            for x, y in zip(xs, ys):
                pred = self.forward(x)
                self.backward(y)
                with open('pred.txt', 'a+') as f:
                    f.write(str(self.layers[1].local_grad) + '\n')

    def predict(self, xs):
        return (np.asarray([self.forward(x) for x in xs])).reshape((-1,))

    def evaluate(self, xs, ys):
        predict = self.predict(xs)
        return np.mean((predict - ys) ** 2), np.mean(np.abs(predict - ys))

    def compile(self, learning_rate=0.01):
        self.lr = learning_rate
