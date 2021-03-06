__author__ = "Alexander Chang"
__jhed__ = "achang56"
__email__ = "achang56@jhu.edu"
__class__ = "cs475"


# Data normalization and cleanup as data pre-processing
import numpy as np
from numpy import matlib
import warnings


def standardize(x):
    """
    Standardizes the data to the Normal Distribution z = (X-mu)/sigma
    :param x: The data to normalize
    :return: The normalized X
    """

    # calculate mean and sigma for each of the columns
    sigma = np.std(x, 0)
    mu = np.mean(x, 0)

    # get rid of columns that don't have important features
    col = 0
    while col < np.size(x, 1):
        if sigma[0, col] == 0:
            x = np.delete(x, col, 1)
            sigma = np.delete(sigma, col, 1)
            mu = np.delete(mu, col, 1)
            col -= 1
        col += 1

    # x - mu
    x = x - np.matlib.repmat(mu, np.size(x, 0), 1)
    # divide the entire matrix by sigma
    x = np.multiply(x, np.matlib.repmat(1 / sigma, np.size(x, 0), 1))

    return x


def normalize(x):
    """
    DOES NOT WORK
    Normalizes the data. x' = (X-mu)/(X max-Xmin)
    :param x: The data to normalize
    :return: The normalized x
    """

    warnings.warn("Doesn't work - use standardize instead until I fix the x - mu part", Warning)

    # want a row vector so we can repmat it downwards
    maximum = np.amax(x, 0)
    minimum = np.amin(x, 0)

    # calculate mean and sigma for each of the columns
    mu = np.mean(x, 0)

    # get rid of columns that don't have important features
    col = 0
    while col < np.size(x, 1):
        if maximum[0, col] == minimum[0, col]:
            x = np.delete(x, col, 1)
            maximum = np.delete(maximum, col, 1)
            minimum = np.delete(minimum, col, 1)
            col -= 1
        col += 1

    # x - mu
    # NOTE: Fix this
    x = x - np.matlib.repmat(mu, np.size(x, 0), 1)
    # divide the entire matrix by sigma
    x = np.divide(x, np.matlib.repmat(maximum - minimum, np.size(x, 0), 1))

    return x


def data_preprocess_y(y):

    for idx in range(len(y)):
        if y[idx] == 0:
            y[idx] = -1

    return y
