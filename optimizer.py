import numpy as np


class Optimizer(object):

    def __init_subclass__(cls, **kwargs):
        pass

    def optimize(self, func, W, y, x):
        raise NotImplementedError()


class GradientDescent(Optimizer):

    def __init__(self, learning_rate=0.001, decay=0.0):
        self.learning_rate = learning_rate
        self.decay = decay
        pass

    def optimize(self, costFunc, W, y, x):
        """
        Performs gradient descent
        :param costFunc: the cost function
        :param W: the weights
        :param y: the expected output
        :param x: the input
        :return: the new weights
        """
        cost, grad = costFunc(W, y, x, self.decay)
        # performs
        # W' = W - a(grad)
        wp = W - np.multiply(self.learning_rate, grad)
        return wp, cost
