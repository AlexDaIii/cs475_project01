import numpy as np
from trainer import Trainer
from cost_function import ZeroOneLoss
import math


class Model(object):

    def __init__(self):
        self.num_input_features = None

    def fit(self, **kwargs):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()

    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()


class Useless(Model):

    def __init__(self):
        super().__init__()
        self.reference_example = None
        self.reference_label = None

    def fit(self, X, y, n_epoch=5, learning_rate=1.0, batch_size=1):
        self.num_input_features = X.shape[1]
        # Designate the first training example as the 'reference' example
        # It's shape is [1, num_features]
        self.reference_example = X[0, :]
        # Designate the first training label as the 'reference' label
        self.reference_label = y[0]
        self.opposite_label = 1 - self.reference_label

    def predict(self, X):
        if self.num_input_features is None:
            raise Exception('fit must be called before predict.')
        # Perhaps fewer features are seen at test time than train time, in
        # which case X.shape[1] < self.num_input_features. If this is the case,
        # we can simply 'grow' the rows of X with zeros. (The copy isn't
        # necessary here; it's just a simple way to avoid modifying the
        # argument X.)
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        # Or perhaps more features are seen at test time, in which case we will
        # simply ignore them.
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        # Compute the dot products between the reference example and X examples
        # The element-wise multiply relies on broadcasting; here, it's as if we first
        # replicate the reference example over rows to form a [num_examples, num_input_features]
        # array, but it's done more efficiently. This forms a [num_examples, num_input_features]
        # sparse matrix, which we then sum over axis 1.
        dot_products = X.multiply(self.reference_example).sum(axis=1)
        # dot_products is now a [num_examples, 1] dense matrix. We'll turn it into a
        # 1-D array with shape [num_examples], to be consistent with our desired predictions.
        dot_products = np.asarray(dot_products).flatten()
        # If positive, return the same label; otherwise return the opposite label.
        same_label_mask = dot_products >= 0
        opposite_label_mask = ~same_label_mask
        y_hat = np.empty([num_examples], dtype=np.int)
        y_hat[same_label_mask] = self.reference_label
        y_hat[opposite_label_mask] = self.opposite_label
        return y_hat


class SumOfFeatures(Model):

    def __init__(self):
        super().__init__()
        pass

    def fit(self, X, y, n_epoch=5, learning_rate=1.0, batch_size=1):
        # NOTE: Not needed for SumOfFeatures classifier. However, do not modify.
        pass

    def predict(self, X):
        num_examples, self.num_input_features = X.shape
        y_hat = np.zeros((num_examples, 1))
        beginning = np.sum(X[:, 0:math.floor(X.shape[1] / 2)], 1)
        end = np.sum(X[:, math.ceil(X.shape[1]/2):], 1)
        for idx in range(num_examples):
            if beginning[idx] > end[idx]:
                y_hat[idx] = 1
            else:
                y_hat[idx] = 0
        return y_hat


class Perceptron(Model):

    def __init__(self):
        super().__init__()
        # Initializations
        self.weights = None
        pass

    def fit(self, X, y, decay=0.0, n_epoch=5, learning_rate=1.0, batch_size=1):
        self.num_input_features = X.shape[1]

        # fits the model.
        to_train = Trainer()
        self.weights = to_train.train(X, y, decay=decay, n_epoch=n_epoch, learning_rate=learning_rate,
                                      batch_size=batch_size, cost_fun=ZeroOneLoss(), show_metric=False)
        pass

    def predict(self, X):
        X = X.todense()

        num_examples, num_input_features = X.shape
        # if num input features less, then append zeros
        if num_input_features < self.num_input_features:
            temp = np.zeros((num_examples, self.num_input_features - num_input_features))
            X = np.append(X, temp, 1)
        # if the num input features greater, get rid of some features
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]

        # code to make predictions.
        z = np.matmul(X, self.weights)
        y_hat = np.zeros((np.size(X, 0), 1))
        # if z >= 0, then predict y = 1, else 0
        for idx in range(0, np.size(X, 0) - 1):
            if z[idx] >= 0:
                y_hat[idx] = 1
        return y_hat

# TODO: Add other Models as necessary.
