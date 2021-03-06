__author__ = "Alexander Chang"
__jhed__ = "achang56"
__email__ = "achang56@jhu.edu"
__class__ = "cs475"


import numpy as np

# Does the optimization strategy
class Optimizer(object):

    def __init_subclass__(cls, **kwargs):
        pass

    def optimize(self, **kwargs):
        raise NotImplementedError()


class GradientDescent(Optimizer):

    def __init__(self, learning_rate=1.0, decay=0.0):
        self.learning_rate = learning_rate
        self.decay = decay
        pass

    def optimize(self, costFunc, W, x, y):
        """
        Performs gradient descent
        :param costFunc: the cost function
        :param W: the weights
        :param y: the expected output
        :param x: the input
        :return: the new weights and cost
        """

        cost, grad = costFunc(W, x, y)
        # performs
        # W' = W - a(grad)
        wp = W - np.multiply(self.learning_rate, grad)
        return wp, cost
