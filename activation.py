import numpy as np


class Activation:

    def activation_function(self, a):
        raise NotImplementedError()


class Sigmoid(Activation):

    def activation_function(self, a):
        return 1 / (1 + np.exp(-a))


class SoftMax(Activation):

    def activation_function(self, a):
        return np.exp(a) / (np.sum(np.exp(a), 0))


class Relu(Activation):

    def activation_function(self, a):
        zeros = np.zeros(a.shape)
        return np.maximum(zeros, a)


class Tanh(Activation):

    def activation_function(self, a):
        ones = np.ones(a.shape)
        return (np.exp(2 * a) - ones) / (np.exp(2 * a) + ones)
