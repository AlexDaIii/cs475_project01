__author__ = "Alexander Chang"
__jhed__ = "achang56"
__email__ = "achang56@jhu.edu"
__class__ = "cs475"

import numpy as np


class Activation:

    def activation_function(self, a):
        raise NotImplementedError()


class Sigmoid(Activation):

    def activation_function(self, a):
        return 1 / (1 + np.exp(-a))
