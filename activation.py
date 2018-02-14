import numpy as np


class Activation:

    def activation_function(self, a):
        raise NotImplementedError()


class Sigmoid(Activation):

    def activation_function(self, a):
        return 1 / (1 + np.exp(-a))
