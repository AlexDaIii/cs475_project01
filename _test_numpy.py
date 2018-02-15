import numpy as np
import math

X = np.array([[1, 2, 3], [4, 2, 5], [6, 2, 7], [8, 2, 9], [1, 2, 3], [4, 2, 5], [6, 2, 7], [8, 2, 9]])
W = np.zeros((np.size(X, 1), 1))
sigma = np.std(X, 0)
batch_size = 1
n_epoch = 2

k = int(math.ceil(np.size(X, 0)/batch_size))

for idx in range(n_epoch):
    # split x into the mini batches
    k = int(math.ceil(np.size(X, 0) / batch_size))
    # display message

    # split into k mini batches
    for i in range(k - 1):
        print(X[i * batch_size:(i + 1) * batch_size, :])

    # the last rows to grad descent
    print(X[(k - 1) * batch_size:, :])
    print()

