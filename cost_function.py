import numpy as np
from activation import Sigmoid


class CostFunction(object):

    def cost(self, W, x, y):
        raise NotImplementedError()


class CrossEntropy(CostFunction):

    # J = (1/m) * sum{1}{m}[((-y * log(h(x)) - ((1 - y) * log(1 - h(x)))] + (lambda/2*m)*||(W)||^2
    # h(x) = g(x, W) = sigmoid(x, W)
    # dJ/dWij = (1/m)*sum{1}{m}[(h(x) - y) * x] + (lambda/m)*[(L * W)]
    # L = I (with L[0,0] = 0)
    def cost(self, W, x, y):
        """
        Computes the cross entropy loss and the gradient with the sigmoid activation function for the batch
        :param W: Weights ∈ R^num_features
        :param x: the inputs ∈ R^m*num_features
        :param y: the outputs ∈ R^m
        :return: the cost, the gradient
        """

        # the activation function
        g = Sigmoid()

        # get the number of examples in this batch size
        m = np.size(y, 0)

        # calculate unregularized cost
        z = np.matmul(x, W)

        # calculates (-y * log(h(x)) y = 0
        cost1 = -y.T * np.log(g.activation_function(z))

        # calculates ((1 - y) * log(1 - h(x))) y = 1
        cost2 = np.transpose(np.ones((m, 1)) - y) * np.log(np.ones((m, 1)) - g.activation_function(z))

        j = (1 / m) * (cost1 - cost2)

        # calculates the gradient
        # (1/m)*(h(x) - y) * x
        grad = np.multiply(1 / m, np.matmul(x.T, g.activation_function(z) - y))

        return j, grad


class ZeroOneLoss(CostFunction):

    # J = sum{1}{m}(max(-y * W' * x))
    # dJ/dWij = (y - y_hat)x = (y - W' * x) * x
    def cost(self, W, x, y):
        """
        Calculates the zero one loss for the batch
        :param W: Weights ∈ R^num_features
        :param x: the inputs ∈ R^m*num_features
        :param y: the outputs ∈ R^m
        :return: the cost, the gradient
        """

        z = np.matmul(x, W)
        for idx in range(len(z)):
            if z[idx] >= 0:
                z[idx] = 1
            else:
                z[idx] = -1
            # print("z: " + str(z))

        # print("y: " + str(y))
        j = np.matmul(-y.T, z)
        # print("jb: " + str(j))
        j = max(np.array([0]), j)
        # print("ja: " + str(j))

        grad = np.zeros((np.size(x, 1), 1))
        if j != 0:
            grad = x.T * y
        # print(grad[0:3])

        return j, -grad
