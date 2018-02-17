import numpy as np
import math
from optimizer import GradientDescent
import cost_function


class Trainer(object):

    def __init__(self):
        self.step = 1

    def train(self, X, y, decay=0.0, n_epoch=5, learning_rate=1, batch_size=None, show_metric=True,
              cost_fun=cost_function.LogLoss()):
        """
        Trains the model
        :param cost_fun: The cost function we use, default is the only one we have
        :param show_metric: If we display the batch and the cost
        :param X: The inputs
        :param y: The outputs
        :param decay: lambda in regularization
        :param n_epoch: number of epochs
        :param learning_rate: the learning rate, default is 1
        :param batch_size: batch size, default is None (batch gradient descent)
        :return: the trained weights
        """
        weights = np.zeros((np.size(X, 1), 1))

        opt = GradientDescent(decay=decay, learning_rate=learning_rate)
        if batch_size is None:
            batch_size = np.size(X, 0)

        for idx in range(n_epoch):
            # split x into the mini batches
            k = int(math.ceil(np.size(X, 0) / batch_size))
            # display message
            if show_metric:
                print("Epoch: " + str(self.step))

            # split into k mini batches
            for i in range(k - 1):

                weights, cost = opt.optimize(cost_fun.cost, weights,
                                             X[i * batch_size:(i + 1) * batch_size, :],
                                             y[i * batch_size:(i + 1) * batch_size, :])
                # display message
                if show_metric:
                    print("Batch: " + str(i) + "/" + str(k) + ", Cost: " + str(cost))

            # the last rows to grad descent
            weights, cost = opt.optimize(cost_fun.cost, weights,
                                         X[(k - 1) * batch_size:, :],
                                         y[(k - 1) * batch_size:, :])
            self.step += 1
            # display message
            if show_metric:
                print("Batch: " + str(k) + "/" + str(k) + ", Cost: " + str(cost))

        return weights
