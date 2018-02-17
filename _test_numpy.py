import numpy as np
import math

X = np.array([[1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3]])
W = np.zeros((np.size(X, 1), 1))
sigma = np.std(X, 0)
batch_size = 1
n_epoch = 2

k = int(math.ceil(np.size(X, 0)/batch_size))

print(np.sum(X[:, 0:math.floor(X.shape[1]/2)], 1))
print()
print(np.sum(X[:, math.ceil(X.shape[1]/2):], 1))

