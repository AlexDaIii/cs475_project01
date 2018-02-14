import numpy as np
from activation import Sigmoid


class CostFunction(object):

    def cost(self, weights, x, y, decay=0.0):
        raise NotImplementedError()


class PerceptronCrossEntropy(CostFunction):

    # J = (1/m) * sum{1}{m}[((-y * log(h(x)) - ((1 - y) * log(1 - h(x)))] + (lambda/2*m)*||(W)||^2
    # h(x) = g(x, W) = sigmoid(x, W)
    # dj/dWij = (1/m)*sum{1}{m}[(h(x) - y) * x] + (lambda/m)*[(L * W)]
    # L = I (with L[0,0] = 0)
    def cost(self, W, x, y, decay=0.0):
        """
        Computes the cross entropy loss and the gradient with the sigmoid activation function for the batch
        :param W: the weights into the layer ∈ R^len(a+1)*len(a)
        :param x: the inputs into the layer ∈ R^len(a)
        :param y: the expected output of the neuron
        :param decay: regularization parameter
        :return: the cost, the gradient
        """

        # the activation function
        g = Sigmoid()

        # get the number of examples in this batch size
        m = np.size(y, 0)

        # calculate unregularized cost
        z = np.matmul(x, W)

        # calculates (-y * log(h(x))
        cost1 = np.matmul(-y.T, np.log(g.activation_function(z)))

        # calculates ((1 - y) * log(1 - h(x)))
        cost2 = np.matmul(np.transpose(np.ones((m, 1)) - y), np.log(np.ones((m, 1)) - g.activation_function(z)))

        j = (1/m) * (cost1 - cost2)

        # calculates the gradient
        # (1/m)*(h(x) - y) * x
        grad = np.multiply(1/m, np.matmul(x.T, g.activation_function(z) - y))

        return j, grad
